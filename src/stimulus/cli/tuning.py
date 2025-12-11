#!/usr/bin/env python3
"""CLI module for running Optuna hyperparameter tuning experiments."""

import logging
import os
from typing import Any, Optional

import optuna
import optuna.storages.journal
import yaml

from stimulus.learner import optuna_tune
from stimulus.learner.device_utils import resolve_device
from stimulus.learner.interface import model_config_parser, model_schema
from stimulus.utils import model_file_interface

logger = logging.getLogger(__name__)


def tune(
    data_path: str,
    model_path: str,
    model_config_path: str,
    optuna_results_dirpath: str = "./optuna_results",
    best_model_path: str = "best_model.safetensors",
    best_optimizer_path: str = "best_optimizer.pt",
    best_config_path: str = "best_config.json",
    force_device: Optional[str] = None,
) -> None:
    """Run model hyperparameter tuning.

    Args:
        data_path: Path to input data file.
        model_path: Path to model file.
        model_config_path: Path to model config file.
        optuna_results_dirpath: Directory for optuna results.
        best_model_path: Path to write the best model to.
        best_optimizer_path: Path to write the best optimizer to.
        force_device: Force the device to use.
    """
    # Load model config
    with open(model_config_path) as file:
        model_config_dict: dict[str, Any] = yaml.safe_load(file)
    model_config: model_schema.Model = model_schema.Model(**model_config_dict)

    # Load model class
    model_class = model_file_interface.import_class_from_file(model_path)

    # Load train and validation datasets
    from stimulus.data.interface import dataset_interface

    dataset_class = getattr(dataset_interface, model_config.dataset.type)
    dataset_dict = dataset_class.load_from_disk(data_path, **model_config.dataset.params)

    train_dataset = dataset_dict.get_torch_dataset("train")
    validation_dataset = dataset_dict.get_torch_dataset("val")

    # get the pruner
    pruner = model_config_parser.get_pruner(model_config.pruner)

    # get the sampler
    sampler = model_config_parser.get_sampler(model_config.sampler)

    # storage setups
    base_path = optuna_results_dirpath
    artifact_path = optuna_results_dirpath + "/artifacts"
    os.makedirs(base_path, exist_ok=True)
    os.makedirs(artifact_path, exist_ok=True)
    artifact_store = optuna.artifacts.FileSystemArtifactStore(base_path=artifact_path)
    storage = optuna.storages.JournalStorage(
        optuna.storages.journal.JournalFileBackend(f"{base_path}/optuna_journal_storage.log"),
    )

    device = resolve_device(force_device=force_device, config_device=model_config.device)

    objective = optuna_tune.Objective(
        model_class=model_class,
        network_params=model_config.network_params,
        optimizer_params=model_config.optimizer_params,
        data_params=model_config.data_params,
        train_torch_dataset=train_dataset,
        val_torch_dataset=validation_dataset,
        artifact_store=artifact_store,
        max_samples=model_config.max_samples,
        compute_objective_every_n_samples=model_config.compute_objective_every_n_samples,
        target_metric=model_config.objective.metric,
        device=device,
        log_dir=os.path.join(base_path, "runs"),
    )

    study = optuna_tune.tune_loop(
        objective=objective,
        storage=storage,
        sampler=sampler,
        pruner=pruner,
        n_trials=model_config.n_trials,
        direction=model_config.objective.direction,
    )

    best_trial = study.best_trial
    best_model_artifact_id = best_trial.user_attrs["model_id"]
    best_optimizer_artifact_id = best_trial.user_attrs["optimizer_id"]
    best_model_suggestions_artifact_id = best_trial.user_attrs["model_suggestions_id"]

    # Ensure output directories exist
    if os.path.dirname(best_model_path):
        os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
    if os.path.dirname(best_optimizer_path):
        os.makedirs(os.path.dirname(best_optimizer_path), exist_ok=True)
    if os.path.dirname(best_config_path):
        os.makedirs(os.path.dirname(best_config_path), exist_ok=True)

    optuna.artifacts.download_artifact(
        artifact_store=artifact_store,
        file_path=best_model_path,
        artifact_id=best_model_artifact_id,
    )
    optuna.artifacts.download_artifact(
        artifact_store=artifact_store,
        file_path=best_optimizer_path,
        artifact_id=best_optimizer_artifact_id,
    )
    optuna.artifacts.download_artifact(
        artifact_store=artifact_store,
        file_path=best_config_path,
        artifact_id=best_model_suggestions_artifact_id,
    )
