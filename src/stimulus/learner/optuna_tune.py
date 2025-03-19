"""Optuna tuning module."""

import inspect
import logging
import os
import uuid
from typing import Any

import optuna
import torch
from optuna.trial import FrozenTrial
from safetensors.torch import save_model as safe_save_model

from stimulus.learner.interface import model_config_parser, model_schema
from stimulus.learner.predict import PredictWrapper

logger = logging.getLogger(__name__)


class Objective:
    """Objective class for Optuna tuning."""

    def __init__(
        self,
        model_class: torch.nn.Module,
        network_params: dict[str, model_schema.TunableParameter | model_schema.VariableList],
        optimizer_params: dict[str, model_schema.TunableParameter],
        data_params: dict[str, model_schema.TunableParameter],
        loss_params: dict[str, model_schema.TunableParameter],
        train_torch_dataset: torch.utils.data.Dataset,
        val_torch_dataset: torch.utils.data.Dataset,
        artifact_store: Any,
        max_batches: int = 1000,
        compute_objective_every_n_batches: int = 50,
        target_metric: str = "val_loss",
        device: torch.device | None = None,
    ):
        """Initialize the Objective class.

        Args:
            model_class: The model class to be tuned.
            network_params: The network parameters to be tuned.
            optimizer_params: The optimizer parameters to be tuned.
            data_params: The data parameters to be tuned.
            loss_params: The loss parameters to be tuned.
            train_torch_dataset: The training dataset.
            val_torch_dataset: The validation dataset.
            artifact_store: The artifact store to save the model and optimizer.
            max_batches: The maximum number of batches to train.
            compute_objective_every_n_batches: The number of batches to compute the objective.
            target_metric: The target metric to optimize.
            device: The device to run the training on.
        """
        self.model_class = model_class
        self.network_params = network_params
        self.optimizer_params = optimizer_params
        self.data_params = data_params
        self.loss_params = loss_params
        self.train_torch_dataset = train_torch_dataset
        self.val_torch_dataset = val_torch_dataset
        self.artifact_store = artifact_store
        self.target_metric = target_metric
        self.max_batches = max_batches
        self.compute_objective_every_n_batches = compute_objective_every_n_batches
        if device is None:
            self.device = torch.device("cpu")
        else:
            self.device = device

    def _get_optimizer(self, optimizer_name: str) -> type[torch.optim.Optimizer]:
        """Get the optimizer from the name."""
        try:
            return getattr(torch.optim, optimizer_name)
        except AttributeError as e:
            raise ValueError(f"Optimizer {optimizer_name} not found in torch.optim") from e

    def __call__(self, trial: optuna.Trial):
        """Execute a full training trial and return the objective metric value."""
        # Setup phase
        logger.info(f"Starting trial {trial.number}")

        model_instance = self._setup_model(trial)
        optimizer = self._setup_optimizer(trial, model_instance)
        train_loader, val_loader = self._setup_data_loaders(trial)
        loss_dict = self._setup_loss_functions(trial)

        logger.info(f"Trial {trial.number} setup complete, starting training loop")

        # Training loop
        batch_idx: int = 0
        metric_dict: dict = {}
        last_report_batch = 0

        while batch_idx < self.max_batches:
            for x, y, _meta in train_loader:
                try:
                    device_x = {key: value.to(self.device, non_blocking=True) for key, value in x.items()}
                    device_y = {key: value.to(self.device, non_blocking=True) for key, value in y.items()}

                    # Perform a batch update
                    model_instance.batch(x=device_x, y=device_y, optimizer=optimizer, **loss_dict)

                except RuntimeError as e:
                    if ("CUDA out of memory" in str(e) and self.device.type == "cuda") or (
                        "MPS backend out of memory" in str(e) and self.device.type == "mps"
                    ):
                        logger.warning(f"{self.device.type.upper()} out of memory during training: {e}")
                        logger.warning("Falling back to CPU for this trial")
                        temp_device = torch.device("cpu")
                        model_instance = model_instance.to(temp_device)
                        # Consider adjusting batch size or other parameters
                        device_x = {key: value.to(temp_device) for key, value in x.items()}
                        device_y = {key: value.to(temp_device) for key, value in y.items()}
                        # Retry the batch
                        model_instance.batch(x=device_x, y=device_y, optimizer=optimizer, **loss_dict)
                    else:
                        raise

                batch_idx += 1

                # Compute objective periodically
                if batch_idx % self.compute_objective_every_n_batches == 0:
                    # Evaluate current model performance
                    metric_dict = self.objective(model_instance, train_loader, val_loader, loss_dict)
                    # Log using Optuna's trial object for better integration with Optuna's logging
                    trial.set_user_attr("latest_metrics", metric_dict)
                    logger.info(f"Trial {trial.number} objective at batch {batch_idx}: {metric_dict}")

                    # Calculate improvement since last report
                    if last_report_batch > 0:
                        batches_since_last = batch_idx - last_report_batch
                        logger.info(f"Processed {batches_since_last} batches since last report")

                    last_report_batch = batch_idx

                    # Report to Optuna
                    trial.report(metric_dict[self.target_metric], batch_idx)
                    for metric_name, metric_value in metric_dict.items():
                        trial.set_user_attr(metric_name, metric_value)

                    # Check if trial should be pruned
                    if trial.should_prune():
                        logger.info(f"Trial {trial.number} pruned at batch {batch_idx}")
                        self.save_checkpoint(trial, model_instance, optimizer)
                        raise optuna.TrialPruned()  # noqa: RSE102

                if batch_idx >= self.max_batches:
                    break

        # Final checkpoint and return objective value
        logger.info(f"Trial {trial.number} completed all {self.max_batches} batches")
        self.save_checkpoint(trial, model_instance, optimizer)
        return metric_dict[self.target_metric]

    def _setup_model(self, trial: optuna.Trial) -> torch.nn.Module:
        """Setup the model for the trial."""
        model_suggestions = model_config_parser.suggest_parameters(trial, self.network_params)
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
        return model_instance

    def _setup_optimizer(self, trial: optuna.Trial, model_instance: torch.nn.Module) -> torch.optim.Optimizer:
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
    ) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
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
        return train_loader, val_loader

    def _setup_loss_functions(self, trial: optuna.Trial) -> dict[str, torch.nn.Module]:
        """Setup the loss functions for the trial."""
        loss_dict = model_config_parser.suggest_parameters(trial, self.loss_params)
        for key, loss_fn in loss_dict.items():
            loss_dict[key] = getattr(torch.nn, loss_fn)()
        logger.info(f"Loss parameters: {loss_dict}")
        return loss_dict

    def save_checkpoint(
        self,
        trial: optuna.Trial,
        model_instance: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        """Save the model and optimizer to the trial."""
        unique_id = str(uuid.uuid4())[:8]
        model_path = f"{trial.number}_{unique_id}_model.safetensors"
        optimizer_path = f"{trial.number}_{unique_id}_optimizer.pt"
        safe_save_model(model_instance, model_path)
        torch.save(optimizer.state_dict(), optimizer_path)
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
        # delete the files from the local filesystem
        try:
            os.remove(model_path)
            os.remove(optimizer_path)
        except FileNotFoundError:
            logger.info(f"File was already deleted: {model_path} or {optimizer_path}, most likely due to pruning")
        trial.set_user_attr("model_id", artifact_id_model)
        trial.set_user_attr("model_path", model_path)
        trial.set_user_attr("optimizer_id", artifact_id_optimizer)
        trial.set_user_attr("optimizer_path", optimizer_path)

    def objective(
        self,
        model: torch.nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        loss_dict: dict[str, torch.nn.Module],
    ) -> dict[str, float]:
        """Compute the objective metric(s) for the tuning process."""
        metrics = [
            "loss",
            "rocauc",
            "prauc",
            "mcc",
            "f1score",
            "precision",
            "recall",
            "spearmanr",
        ]  # TODO maybe we report only a subset of metrics, given certain criteria (eg. if classification or regression)
        predict_val = PredictWrapper(
            model,
            val_loader,
            loss_dict=loss_dict,
            device=self.device,
        )
        predict_train = PredictWrapper(
            model,
            train_loader,
            loss_dict=loss_dict,
            device=self.device,
        )
        return {
            **{"val_" + metric: value for metric, value in predict_val.compute_metrics(metrics).items()},
            **{"train_" + metric: value for metric, value in predict_train.compute_metrics(metrics).items()},
        }


def get_device() -> torch.device:
    """Get the device to use depending on availability.

    Returns:
        torch.device: device to use.
    """
    if torch.backends.mps.is_available():
        try:
            # Try to allocate a small tensor on MPS to check if it works
            device = torch.device("mps")
            # Create a small tensor and move it to MPS as a test
            test_tensor = torch.ones((1, 1)).to(device)
            del test_tensor  # Free the memory
            logger.info("Using MPS (Metal Performance Shaders) device")
        except RuntimeError as e:
            logger.warning(f"MPS available but failed to initialize: {e}")
            logger.warning("Falling back to CPU")
            return torch.device("cpu")
        else:
            return device

    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        logger.info(f"Using GPU: {gpu_name} with {memory:.2f} GB memory")
        return device

    logger.info("Using CPU (GPU not available)")
    return torch.device("cpu")


def tune_loop(
    objective: Objective,
    pruner: optuna.pruners.BasePruner,
    sampler: optuna.samplers.BaseSampler,
    n_trials: int,
    direction: str,
    storage: optuna.storages.BaseStorage | None = None,
) -> optuna.Study:
    """Run optimization loop.

    Args:
        objective: The objective to optimize
        pruner: The pruner to use
        sampler: The sampler to use
        n_trials: Number of trials to run
        direction: Direction to optimize in
        storage: Optional storage to use

    Returns:
        optuna.Study: The study object containing results
    """
    # Constants for tuning parameters
    max_params_to_log = 10
    log_best_trial_every_n = 5

    # Create a study object, force the sampler and pruner
    study = optuna.create_study(
        direction=direction,
        pruner=pruner,
        sampler=sampler,
        storage=storage,
        load_if_exists=True,
    )

    # Callback for logging information
    def logging_callback(study: optuna.Study, trial: FrozenTrial) -> None:
        """Log information about completed trials."""
        if trial.state == optuna.trial.TrialState.COMPLETE:
            logger.info(f"Trial {trial.number} finished with value: {trial.value}")
            if len(trial.params) <= max_params_to_log:  # Only log params if not too many
                logger.info(f"Trial {trial.number} params: {trial.params}")
            else:
                logger.info(f"Trial {trial.number} has {len(trial.params)} parameters")

        if study.best_trial and trial.number % log_best_trial_every_n == 0:  # Log best trial info every 5 trials
            logger.info(f"Best trial so far: #{study.best_trial.number} with value: {study.best_trial.value}")

        if trial.state == optuna.trial.TrialState.PRUNED:
            logger.info(f"Trial {trial.number} pruned")

    try:
        study.optimize(objective, n_trials=n_trials, callbacks=[logging_callback])
    except KeyboardInterrupt:
        logger.info("Optimization interrupted by user")

    # Log final results
    logger.info("Optimization finished")
    if study.best_trial:
        logger.info(f"Best trial: #{study.best_trial.number} with value: {study.best_trial.value}")
        logger.info(f"Best parameters: {study.best_trial.params}")
    else:
        logger.warning("No trials completed successfully")

    return study
