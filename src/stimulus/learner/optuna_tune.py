"""Optuna tuning module."""

import inspect
import json
import logging
import os
import tempfile
import uuid
from typing import Any

import optuna
import torch
from safetensors.torch import save_model as safe_save_model

from stimulus.learner.interface import model_config_parser, model_schema
from stimulus.typing.protocols import StimulusModel

logger = logging.getLogger(__name__)

# Constants for model interface
STANDARD_MODEL_RETURN_COUNT = 2  # (loss, metrics)


class _StorageCompatibleTrial:
    """Wrapper around optuna.Trial that normalizes categorical choices for persistent storage.

    This wrapper intercepts suggest_categorical calls and converts list choices to tuples
    to avoid warnings when using JournalStorage or other persistent storage backends.

    This is particularly useful for variable-length architectures where layer configurations
    are represented as lists (e.g., [128, 64] for a 2-layer network).
    """

    def __init__(self, trial: optuna.Trial):
        self._trial = trial

    def suggest_categorical(self, name: str, choices: Any) -> Any:
        """Wrap suggest_categorical to serialize complex choices for storage compatibility.

        Args:
            name: Parameter name
            choices: List of choices (may include complex types like lists)

        Returns:
            Selected choice (deserialized if it was complex)
        """
        # Check if we have complex (non-primitive) choices
        has_complex = any(isinstance(c, (list, tuple)) for c in choices)

        if has_complex:
            # Serialize to JSON strings for Optuna storage (hashable and storable)
            str_choices = tuple(json.dumps(list(c)) if isinstance(c, (list, tuple)) else c for c in choices)
            result = self._trial.suggest_categorical(name, str_choices)
            # Deserialize back to list for the model
            return json.loads(result)
        return self._trial.suggest_categorical(name, choices)

    def __getattr__(self, name: str):
        """Delegate all other methods to the original trial."""
        return getattr(self._trial, name)


def resolve_device(
    force_device: str | None = None,
    config_device: str | None = None,
) -> torch.device:
    """Resolve device based on priority: force_device > config_device > auto-detection.

    Args:
        force_device: Device specified via CLI or function parameter (highest priority)
        config_device: Device specified in model configuration (medium priority)

    Returns:
        The resolved computation device

    Raises:
        RuntimeError: If a forced or configured device is invalid or unavailable
    """
    if force_device is not None:
        device = torch.device(force_device)
        logger.info(f"Using force-specified device: {force_device}")
        return device

    if config_device is not None:
        device = torch.device(config_device)
        logger.info(f"Using config-specified device: {config_device}")
        return device

    return get_device()


def get_device() -> torch.device:
    """Get the appropriate device (CPU/GPU) for computation.

    Priority order:
    1. MPS (Apple Silicon) if available
    2. CUDA (NVIDIA GPU) if available
    3. CPU as fallback

    Returns:
        The selected computation device
    """
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        # Create a small tensor and move it to MPS as a test
        test_tensor = torch.ones((1, 1)).to(device)
        del test_tensor  # Free the memory
        logger.info("Using MPS (Metal Performance Shaders) device")
        return device

    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        logger.info(f"Using GPU: {gpu_name} with {memory:.2f} GB memory")
        return device

    logger.info("Using CPU (GPU not available)")
    return torch.device("cpu")


def _move_batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    """Move a batch to the specified device.

    Args:
        batch: Dictionary of batch data
        device: Target device

    Returns:
        Dictionary with all tensors moved to device
    """
    device_batch = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            device_batch[key] = value.to(device, non_blocking=True)
        else:
            device_batch[key] = value
    return device_batch


class Objective:
    """Objective class for Optuna tuning."""

    def __init__(
        self,
        model_class: type[StimulusModel],
        network_params: dict[str, model_schema.TunableParameter | model_schema.VariableList],
        optimizer_params: dict[str, model_schema.TunableParameter],
        data_params: dict[str, model_schema.TunableParameter],
        train_torch_dataset: torch.utils.data.Dataset,
        val_torch_dataset: torch.utils.data.Dataset,
        artifact_store: Any,
        max_samples: int = 1000,
        compute_objective_every_n_samples: int = 50,
        target_metric: str = "val_loss",
        device: torch.device | None = None,
        log_dir: str = "runs",
    ):
        """Initialize the Objective class.

        Args:
            model_class: The model class to be tuned.
            network_params: The network parameters to be tuned.
            optimizer_params: The optimizer parameters to be tuned.
            data_params: The data parameters to be tuned.
            train_torch_dataset: The training dataset.
            val_torch_dataset: The validation dataset.
            artifact_store: The artifact store to save the model and optimizer.
            max_samples: The maximum number of samples to train.
            compute_objective_every_n_samples: The number of samples to compute the objective.
            target_metric: The target metric to optimize.
            device: The device to run the training on.
            log_dir: The directory to save logs to.
        """
        self.model_class = model_class
        self.network_params = network_params
        self.optimizer_params = optimizer_params
        self.data_params = data_params

        # Store datasets directly (no wrapping needed)
        self.train_torch_dataset = train_torch_dataset
        self.val_torch_dataset = val_torch_dataset

        self.artifact_store = artifact_store
        self.target_metric = target_metric
        self.max_samples = max_samples
        self.compute_objective_every_n_samples = compute_objective_every_n_samples
        if device is None:
            self.device = torch.device("cpu")
        else:
            self.device = device
        self.log_dir = log_dir

    def _get_optimizer(self, optimizer_name: str) -> type[torch.optim.Optimizer]:
        """Get the optimizer from the name."""
        try:
            return getattr(torch.optim, optimizer_name)
        except AttributeError as e:
            raise ValueError(f"Optimizer {optimizer_name} not found in torch.optim") from e

    def _train_single_batch(
        self,
        batch: dict,
        model: StimulusModel,
        optimizer: torch.optim.Optimizer,
        logger: Any,
        step: int,
    ) -> tuple[float, dict[str, float]]:
        """Train on a single batch.

        Args:
            batch: Raw batch from DataLoader
            model: Model instance
            optimizer: Optimizer instance
            logger: ExperimentLogger for metrics
            step: Current training step

        Returns:
            Tuple of (loss value, metrics dict)
        """
        device_batch = _move_batch_to_device(batch, self.device)
        return model.train_batch(device_batch, optimizer, logger, step)

    def _periodic_evaluation(
        self,
        trial: optuna.Trial,
        model: StimulusModel,
        optimizer: torch.optim.Optimizer,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        batch_idx: int,
        complete_suggestions: dict,
    ) -> None:
        """Perform periodic evaluation and handle pruning.

        Args:
            trial: Optuna trial
            model: Model instance
            optimizer: Optimizer instance
            train_loader: Training data loader
            val_loader: Validation data loader
            batch_idx: Current batch index
            complete_suggestions: Complete hyperparameter suggestions for checkpointing

        Raises:
            optuna.TrialPruned: If trial should be pruned
        """
        metric_dict = self.objective(
            model_instance=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=self.device,
        )
        logger.info(f"Metrics at batch {batch_idx}: {metric_dict}")

        trial.report(metric_dict[self.target_metric], batch_idx)
        for metric_name, metric_value in metric_dict.items():
            trial.set_user_attr(metric_name, metric_value)

        if trial.should_prune():
            self.save_checkpoint(trial, model, optimizer, complete_suggestions)
            raise optuna.TrialPruned

    def __call__(self, trial: Any):
        """Execute a full training trial and return the objective metric value."""
        # Wrap trial for storage compatibility
        trial = _StorageCompatibleTrial(trial)

        # Setup phase - capture all parameter suggestions before conversion
        model_instance, model_suggestions = self._setup_model(trial)

        # Capture parameter suggestions before they're converted to instances
        optimizer_suggestions = model_config_parser.suggest_parameters(trial, self.optimizer_params)
        data_suggestions = model_config_parser.suggest_parameters(trial, self.data_params)

        # Combine suggestions for checkpointing
        complete_suggestions = {
            "network_params": model_suggestions,
            "optimizer_params": optimizer_suggestions,
            "data_params": data_suggestions,
        }

        optimizer = self._setup_optimizer(trial, model_instance)
        # Setup training components
        train_loader, val_loader, batch_size = self._setup_data_loaders(trial)

        # Create per-trial logger
        from stimulus.learner.logging import ExperimentLogger

        trial_logger = ExperimentLogger(
            log_dir=os.path.join(self.log_dir, f"trial-{trial.number}"),
            backend="tensorboard",
        )

        # Training loop
        batch_idx: int = 0
        metric_dict: dict = {}

        # Flatten training loop - use infinite iterator with slice
        from itertools import cycle, islice

        max_batches = (self.max_samples + batch_size - 1) // batch_size  # Ceiling division
        samples_since_eval = 0

        with trial_logger:
            for batch_idx, batch in enumerate(islice(cycle(train_loader), max_batches)):
                # Train on single batch
                model_instance.train()
                _loss, _metrics = self._train_single_batch(
                    batch,
                    model_instance,
                    optimizer,
                    trial_logger,
                    batch_idx,
                )

                # Track samples and evaluate periodically
                samples_since_eval += batch_size
                if samples_since_eval >= self.compute_objective_every_n_samples:
                    samples_since_eval = 0
                    self._periodic_evaluation(
                        trial,
                        model_instance,
                        optimizer,
                        train_loader,
                        val_loader,
                        batch_idx,
                        complete_suggestions,
                    )

            # Ensure we have computed metrics at least once before returning
            if not metric_dict:
                logger.info("Computing final objective metrics before returning")
                metric_dict = self.objective(
                    model_instance=model_instance,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    device=self.device,
                )
                for metric_name, metric_value in metric_dict.items():
                    trial.set_user_attr(metric_name, metric_value)

        # Final checkpoint and return objective value
        self.save_checkpoint(trial, model_instance, optimizer, complete_suggestions)
        return metric_dict[self.target_metric]

    def _setup_model(self, trial: optuna.Trial) -> tuple[StimulusModel, dict]:
        """Setup the model for the trial."""
        model_suggestions = model_config_parser.suggest_parameters(trial, self.network_params)

        # Inject dataset attributes if available
        # Check if the dataset has 'dataset_attributes' property
        if hasattr(self.train_torch_dataset, "dataset_attributes"):
            dataset_attrs = self.train_torch_dataset.dataset_attributes
            if dataset_attrs:
                logger.info(f"Found dataset attributes: {dataset_attrs}")
                # Update model suggestions with dataset attributes
                # Note: This overrides config-suggested parameters if they share the same name
                # This is intentional for input-dependent parameters (like input dimensions)
                original_keys = set(model_suggestions.keys())
                model_suggestions.update(dataset_attrs)

                # Log usage
                overridden = original_keys.intersection(dataset_attrs.keys())
                if overridden:
                    logger.info(f"Dataset attributes overrode config parameters: {overridden}")

        logger.info(f"Model suggestions: {model_suggestions}")
        model_instance = self.model_class(**model_suggestions)

        try:
            model_instance = model_instance.to(self.device)
            logger.info(f"Model moved to device: {self.device}")
        except RuntimeError as e:
            if self.device.type in ["cuda", "mps"]:
                logger.warning(f"Failed to move model to {self.device.type.upper()}: {e}")
                logger.warning("Falling back to CPU")
                self.device = torch.device("cpu")
                model_instance = model_instance.to(self.device)
            else:
                raise
        return model_instance, model_suggestions

    def _setup_optimizer(self, trial: optuna.Trial, model_instance: StimulusModel) -> torch.optim.Optimizer:
        """Setup the optimizer for the trial."""
        optimizer_suggestions = model_config_parser.suggest_parameters(trial, self.optimizer_params)
        logger.info(f"Optimizer suggestions: {optimizer_suggestions}")

        optimizer_class = self._get_optimizer(optimizer_suggestions["method"])
        optimizer_kwargs = {}
        optimizer_signature = inspect.signature(optimizer_class).parameters
        for k, v in optimizer_suggestions.items():
            if k in optimizer_signature and k != "method":
                optimizer_kwargs[k] = v
            elif k != "method":
                logger.warning(f"Parameter '{k}' not found in optimizer signature, skipping")

        return optimizer_class(model_instance.parameters(), **optimizer_kwargs)

    def _setup_data_loaders(
        self,
        trial: optuna.Trial,
    ) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, int]:
        """Setup the data loaders for the trial."""
        batch_size = model_config_parser.suggest_parameters(trial, self.data_params)["batch_size"]
        logger.info(f"Batch size: {batch_size}")

        train_loader = torch.utils.data.DataLoader(
            self.train_torch_dataset,
            batch_size=batch_size,
            shuffle=True,
        )
        val_loader = torch.utils.data.DataLoader(
            self.val_torch_dataset,
            batch_size=batch_size,
            shuffle=False,
        )
        return train_loader, val_loader, batch_size

    def save_checkpoint(
        self,
        trial: optuna.Trial,
        model_instance: StimulusModel,
        optimizer: torch.optim.Optimizer,
        complete_suggestions: dict,
    ) -> None:
        """Save the model and optimizer to the trial."""
        # Convert model to CPU before saving to avoid device-specific tensors
        model_instance = model_instance.cpu()
        optimizer_state = optimizer.state_dict()

        # Convert optimizer state to CPU tensors
        for param in optimizer_state["state"].values():
            for k, v in param.items():
                if isinstance(v, torch.Tensor):
                    param[k] = v.cpu()
        unique_id = str(uuid.uuid4())[:8]
        # Use log_dir for temporary files to avoid race conditions in parallel execution
        # If log_dir doesn't exist, use a temporary directory
        base_dir = self.log_dir if os.path.exists(self.log_dir) else None

        with tempfile.TemporaryDirectory(dir=base_dir) as temp_dir:
            model_path = os.path.join(temp_dir, f"{trial.number}_{unique_id}_model.safetensors")
            optimizer_path = os.path.join(temp_dir, f"{trial.number}_{unique_id}_optimizer.pt")
            model_suggestions_path = os.path.join(temp_dir, f"{trial.number}_{unique_id}_model_suggestions.json")

            safe_save_model(model_instance, model_path)
            torch.save(optimizer_state, optimizer_path)
            with open(model_suggestions_path, "w") as f:
                json.dump(complete_suggestions, f)

            artifact_id_model = optuna.artifacts.upload_artifact(
                artifact_store=self.artifact_store,
                file_path=model_path,
                study_or_trial=trial.study,
            )
            artifact_id_optimizer = optuna.artifacts.upload_artifact(
                artifact_store=self.artifact_store,
                file_path=optimizer_path,
                study_or_trial=trial.study,
            )
            artifact_id_model_suggestions = optuna.artifacts.upload_artifact(
                artifact_store=self.artifact_store,
                file_path=model_suggestions_path,
                study_or_trial=trial.study,
            )

            # Store the paths in user_attrs for reference, even though they will be deleted
            trial.set_user_attr("model_id", artifact_id_model)
            trial.set_user_attr("model_path", model_path)
            trial.set_user_attr("optimizer_id", artifact_id_optimizer)
            trial.set_user_attr("optimizer_path", optimizer_path)
            trial.set_user_attr("model_suggestions_id", artifact_id_model_suggestions)
            trial.set_user_attr("model_suggestions_path", model_suggestions_path)

    def objective(
        self,
        model_instance: StimulusModel,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        device: torch.device,
    ) -> dict[str, float]:
        """Compute the objective metric(s) for the tuning process.

        The objectives are outputed by the model's batch function in the form of loss, metric_dictionary.
        """
        train_metrics = self.get_metrics(model_instance, train_loader, device)
        val_metrics = self.get_metrics(model_instance, val_loader, device)

        # add train_ and val_ prefix to related keys.
        return {
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }

    def get_metrics(
        self,
        model_instance: StimulusModel,
        data_loader: torch.utils.data.DataLoader,
        _device: torch.device,
    ) -> dict[str, float]:
        """Compute metrics by delegating to the model's validate method.

        The model is responsible for:
        - Setting itself to eval mode
        - Iterating through the data loader
        - Computing and aggregating metrics
        - Returning a dictionary of metric values

        The framework handles device placement by moving batches as they're yielded.
        """
        # Delegate validation to model
        # Note: The model is responsible for moving batches to the appropriate device
        return model_instance.validate(data_loader)


def tune_loop(
    objective: Objective,
    pruner: optuna.pruners.BasePruner,
    sampler: optuna.samplers.BaseSampler,
    n_trials: int,
    direction: str,
    storage: optuna.storages.BaseStorage | None = None,
) -> optuna.Study:
    """Run the tuning loop.

    Args:
        objective: The objective function to optimize.
        pruner: The pruner to use.
        sampler: The sampler to use.
        n_trials: The number of trials to run.
        direction: The direction to optimize.
        storage: The storage to use.

    Returns:
        The study object.
    """
    if storage is None:
        study = optuna.create_study(direction=direction, sampler=sampler, pruner=pruner)
    else:
        study = optuna.create_study(direction=direction, sampler=sampler, pruner=pruner, storage=storage)
    study.optimize(objective, n_trials=n_trials)
    return study
