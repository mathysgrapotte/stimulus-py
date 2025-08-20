"""Optuna tuning module."""

import inspect
import json
import logging
import os
import uuid
import numpy as np
from typing import Any, Optional

import optuna
import torch
import anndata as ad
from torch.utils.tensorboard import SummaryWriter
from safetensors.torch import save_file
from safetensors.torch import save_model as safe_save_model

from stimulus.learner.interface import model_config_parser, model_schema

from cell_eval._types._anndata import PerturbationAnndataPair
from cell_eval._types import initialize_de_comparison
from cell_eval.metrics import mae, discrimination_score, de_overlap_metric

from pdex import parallel_differential_expression

logger = logging.getLogger(__name__)

# Constants for model interface
STANDARD_MODEL_RETURN_COUNT = 2  # (loss, metrics)
EXTENDED_MODEL_RETURN_COUNT = 3  # (loss, metrics, per_sample_dict)


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
        max_samples: int = 1000,
        compute_objective_every_n_samples: int = 50,
        target_metric: str = "val_loss",
        device: torch.device | None = None,
        tb_logdir: str = "test_runs/",
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
            max_samples: The maximum number of samples to train.
            compute_objective_every_n_samples: The number of samples to compute the objective.
            target_metric: The target metric to optimize.
            device: The device to run the training on.
            tb_logdir: The TensorBoard log directory.
        """
        self.model_class = model_class
        self.network_params = network_params
        self.optimizer_params = optimizer_params
        self.data_params = data_params
        self.loss_params = loss_params
        self.tb_logdir = tb_logdir

        # Add sample_id column to datasets for per-sample loss tracking (as integers)
        self.train_torch_dataset = train_torch_dataset
        self.val_torch_dataset = val_torch_dataset

        # Initialize TensorBoard writer
        self.root_writer = SummaryWriter(log_dir=self.tb_logdir)

        self.artifact_store = artifact_store
        self.target_metric = target_metric
        self.max_samples = max_samples
        self.compute_objective_every_n_samples = compute_objective_every_n_samples
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
        

        curve_writer = SummaryWriter(os.path.join(self.tb_logdir, f"trial-{trial.number}"))

        # Setup phase - capture all parameter suggestions before conversion
        model_instance, model_suggestions = self._setup_model(trial)

        # Capture parameter suggestions before they're converted to instances
        optimizer_suggestions = model_config_parser.suggest_parameters(trial, self.optimizer_params)
        data_suggestions = model_config_parser.suggest_parameters(trial, self.data_params)

        # Create complete suggestions dictionary
        complete_suggestions = {
            "network_params": model_suggestions,
            "optimizer_params": optimizer_suggestions,
            "data_params": data_suggestions,
        }

        

        optimizer = self._setup_optimizer(trial, model_instance)
        train_loader, val_loader, batch_size = self._setup_data_loaders(trial)

        # Training loop
        batch_idx: int = 0
        metric_dict: dict = {}

        while batch_idx * batch_size < self.max_samples:
            logger.info(f"Training batch {batch_idx}")
            nb_computed_samples = 0

            for batch in train_loader:
                # set model in train mode
                model_instance.train()
                # Move all tensors to device (sample_id is now an integer tensor)
                logger.info(f"Moving batch to device: {self.device}")
                device_batch = self._move_batch_to_device(batch)
                logger.info(f"Batch {batch_idx} moved to device")


                # Perform a batch update
                _result = model_instance.train_batch(
                    batch=device_batch, 
                    optimizer=optimizer, 
                    writer=curve_writer, 
                    global_step=batch_idx)
                logger.info(f"Batch {batch_idx} training complete")

                batch_idx += 1
                nb_computed_samples += batch_size

                logger.info(f"Batch {batch_idx} processed, total samples: {nb_computed_samples}")
                # Compute objective periodically
                if nb_computed_samples >= self.compute_objective_every_n_samples:
                    logger.info(f"Computing objective at batch {batch_idx}")
                    nb_computed_samples = 0
                    # Evaluate current model performance
                    metric_dict = self.objective(
                        model_instance=model_instance,
                        train_loader=train_loader,
                        val_loader=val_loader,
                    )
                    logger.info(f"Objective: {metric_dict} at batch {batch_idx}")

                    trial.report(metric_dict[self.target_metric], batch_idx)
                    for metric_name, metric_value in metric_dict.items():
                        trial.set_user_attr(metric_name, metric_value)

                    # Check if trial should be pruned
                    if trial.should_prune():
                        self.save_checkpoint(trial, model_instance, optimizer, complete_suggestions)
                        self.root_writer.add_hparams(model_suggestions, metric_dict, run_name=f"trial-{trial.number}")
                        self.root_writer.add_hparams(optimizer_suggestions, metric_dict, run_name=f"trial-{trial.number}")
                        raise optuna.TrialPruned()  # noqa: RSE102

                if batch_idx * batch_size >= self.max_samples:
                    break

                    
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
        self.root_writer.add_hparams(model_suggestions, metric_dict, run_name=f"trial-{trial.number}")
        self.root_writer.add_hparams(optimizer_suggestions, metric_dict, run_name=f"trial-{trial.number}")
        self.save_checkpoint(trial, model_instance, optimizer, complete_suggestions)
        return metric_dict[self.target_metric]

    def _move_batch_to_device(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        # Move all tensors to device (sample_id is now an integer tensor)
        device_batch = {}
        for key, value in batch.items():
            try:
                device_batch[key] = value.to(self.device, non_blocking=True)
            except AttributeError as e:
                raise AttributeError(
                    f"Error moving '{key}' to device. Expected tensor but got {type(value)}. "
                    f"This usually happens when dataset columns contain non-tensor data. "
                    f"Original error: {e}",
                ) from e
        return device_batch

    def _setup_model(self, trial: optuna.Trial) -> tuple[torch.nn.Module, dict]:
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
        return model_instance, model_suggestions

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
        model_path = f"{trial.number}_{unique_id}_model.safetensors"
        optimizer_path = f"{trial.number}_{unique_id}_optimizer.pt"
        model_suggestions_path = f"{trial.number}_{unique_id}_model_suggestions.json"
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
        # delete the files from the local filesystem
        try:
            os.remove(model_path)
            os.remove(optimizer_path)
            os.remove(model_suggestions_path)
        except FileNotFoundError:
            logger.info(
                f"File was already deleted: {model_path} or {optimizer_path} or {model_suggestions_path}, most likely due to pruning",
            )
        trial.set_user_attr("model_id", artifact_id_model)
        trial.set_user_attr("model_path", model_path)
        trial.set_user_attr("optimizer_id", artifact_id_optimizer)
        trial.set_user_attr("optimizer_path", optimizer_path)
        trial.set_user_attr("model_suggestions_id", artifact_id_model_suggestions)
        trial.set_user_attr("model_suggestions_path", model_suggestions_path)

    def objective(
        self,
        model_instance: torch.nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
    ) -> dict[str, float]:
        """Compute the objective metric(s) for the tuning process.

        The objectives are outputed by the model's batch function in the form of loss, metric_dictionary.
        """
        val_metrics = self.get_metrics(model_instance, val_loader)

        # add train_ and val_ prefix to related keys.
        return {
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }

    def get_metrics(
        self,
        model_instance: torch.nn.Module,
        data_loader: torch.utils.data.DataLoader,
    ) -> dict[str, float]:
        """Compute the objective metric(s) for the tuning process."""

        predictions = []
        targets = []

        batch_idx = 0
        for batch in data_loader:
            logger.info(f"Processing validation batch {batch_idx} out of {len(data_loader)}")
            # Move all tensors to device (sample_id is now an integer tensor)
            logger.info(f"Moving batch to device: {self.device}")
            device_batch = self._move_batch_to_device(batch)
            # Perform a batch update
            prediction, target, _loss, _metrics = model_instance.inference(batch=device_batch)
            logger.info(f"Batch {batch_idx} inference complete")
            predictions.append(prediction.detach().cpu().numpy())
            targets.append(target.detach().cpu().numpy())
            logger.info(f"Batch {batch_idx} processed, predictions and targets collected")

        logger.info("All batches processed, aggregating predictions and targets")
        predictions = np.vstack(predictions)
        true = np.vstack(targets)

        true_with_control = np.concatenate([data_loader.dataset.control_cells, true], axis=0)  
        pred_with_control = np.concatenate([data_loader.dataset.control_cells, predictions], axis=0)

        n_controls = data_loader.dataset.control_cells.shape[0]  
        n_perts = true.shape[0]  


        pert_labels = (["control"] * n_controls +
                    [target for target in data_loader.dataset.target_names])

        logger.info(f"Creating AnnData objects for real and predicted data with {n_controls} controls and {n_perts} perturbations")

        real_adata = ad.AnnData(X=true_with_control, obs={"perturbation": pert_labels})
        real_adata.var.index = data_loader.dataset.gene_names
        pred_adata = ad.AnnData(X=pred_with_control, obs={"perturbation": pert_labels})
        pred_adata.var.index = data_loader.dataset.gene_names

        logger.info("Calculating metrics: MAE, Discrimination Score, Differential Expression Overlap")
        data_pair = PerturbationAnndataPair(real=real_adata, pred=pred_adata, pert_col="perturbation", control_pert="control")
        mae_results = mae(data_pair)
        disc_results = discrimination_score(data_pair, metric="l1", exclude_target_gene=True)

        real_de = parallel_differential_expression(
            adata=data_pair.real,
            reference="control",
            groupby_key="perturbation",
            num_workers=1,
            batch_size=100,
            metric="wilcoxon",
            as_polars=True
        )

        pred_de = parallel_differential_expression(
            adata=data_pair.pred,
            reference="control",
            groupby_key="perturbation",
            num_workers=1,
            batch_size=100,
            metric="wilcoxon",
            as_polars=True
        )

        de_comparison = initialize_de_comparison(real_de, pred_de)
        overlap_results = de_overlap_metric(de_comparison, k=None, metric="overlap", fdr_threshold=0.05)

        # Calculate averages from the result dictionaries
        average_overlap = np.mean(list(overlap_results.values()))
        average_mae = np.mean(list(mae_results.values()))
        average_disc = np.mean(list(disc_results.values()))

        return {
            "overlap": average_overlap,
            "mae": average_mae,
            "disc": average_disc
        }


def resolve_device(force_device: Optional[str] = None, config_device: Optional[str] = None) -> torch.device:
    """Resolve device based on priority: force_device > config_device > auto-detection.

    Args:
        force_device: Device specified via CLI or function parameter (highest priority).
        config_device: Device specified in model configuration (medium priority).

    Returns:
        torch.device: The resolved computation device.

    Raises:
        RuntimeError: If a forced or configured device is invalid or unavailable.
    """
    if force_device is not None:
        try:
            device = torch.device(force_device)
        except RuntimeError as e:
            raise RuntimeError(
                f"Forced device '{force_device}' is not available. Please use a valid device.",
            ) from e
        else:
            logger.info(f"Using force-specified device: {force_device}")
            return device

    if config_device is not None:
        try:
            device = torch.device(config_device)
        except RuntimeError as e:
            raise RuntimeError(
                f"Device '{config_device}' specified in model configuration is not available. Please use a valid device.",
            ) from e
        else:
            logger.info(f"Using config-specified device: {config_device}")
            return device

    return get_device()


def get_device() -> torch.device:
    """Get the appropriate device (CPU/GPU) for computation.

    Returns:
        torch.device: The selected computation device
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

