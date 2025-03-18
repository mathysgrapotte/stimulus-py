#!/usr/bin/env python3
"""CLI module for running raytune tuning experiment."""

import logging
import os
import shutil
from typing import Any

import optuna
import torch
import yaml

from stimulus.data import data_handlers
from stimulus.data.interface import data_config_parser
from stimulus.learner import optuna_tune
from stimulus.learner.interface import model_config_parser, model_schema
from stimulus.utils import model_file_interface

logger = logging.getLogger(__name__)


def load_data_config_from_path(data_path: str, data_config_path: str, split: int) -> torch.utils.data.Dataset:
    """Load the data config from a path.

    Args:
        data_path: Path to the input data file.
        data_config_path: Path to the data config file.
        split: Split index to use (0=train, 1=validation, 2=test).

    Returns:
        A TorchDataset with the configured data.
    """
    with open(data_config_path) as file:
        data_config_dict = yaml.safe_load(file)
        data_config_obj = data_config_parser.SplitTransformDict(**data_config_dict)

    encoders, input_columns, label_columns, meta_columns = data_config_parser.parse_split_transform_config(
        data_config_obj,
    )

    return data_handlers.TorchDataset(
        loader=data_handlers.DatasetLoader(
            encoders=encoders,
            input_columns=input_columns,
            label_columns=label_columns,
            meta_columns=meta_columns,
            csv_path=data_path,
            split=split,
        ),
    )


def tune(
    data_path: str,
    model_path: str,
    data_config_path: str,
    model_config_path: str,
    optuna_results_dirpath: str = "./optuna_results",
    best_model_path: str = "best_model.safetensors",
    best_optimizer_path: str = "best_optimizer.pt",
    *,
    verbose: bool = False,
) -> None:
    """Run model hyperparameter tuning.

    Args:
        data_path: Path to input data file.
        model_path: Path to model file.
        data_config_path: Path to data config file.
        model_config_path: Path to model config file.
        optuna_results_dirpath: Directory for optuna results.
        best_model_path: Path to write the best model to.
        best_optimizer_path: Path to write the best optimizer to.
        verbose: Whether to enable verbose Optuna logging.
    """
    # Configure Optuna logging if verbose mode is enabled
    if verbose:
        # Use Optuna's built-in logging configuration
        optuna.logging.set_verbosity(optuna.logging.INFO)
        # Make sure default handler is enabled
        optuna.logging.enable_default_handler()
        # Enable propagation to see logs in the console
        optuna.logging.enable_propagation()

        # Configure root logger to show INFO logs as well
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)

        # Add a stream handler to print logs directly to console
        if not any(isinstance(h, logging.StreamHandler) for h in root_logger.handlers):
            stream_handler = logging.StreamHandler()
            stream_handler.setLevel(logging.INFO)
            formatter = logging.Formatter("[%(levelname)1.1s %(asctime)s] %(message)s")
            stream_handler.setFormatter(formatter)
            root_logger.addHandler(stream_handler)

        logger.info("Verbose Optuna logging enabled")
    else:
        # If not verbose, set to warning level to suppress most logs
        optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Load train and validation datasets
    train_dataset = load_data_config_from_path(data_path, data_config_path, split=0)
    validation_dataset = load_data_config_from_path(data_path, data_config_path, split=1)

    # Load model class
    model_class = model_file_interface.import_class_from_file(model_path)

    # Load model config
    with open(model_config_path) as file:
        model_config_dict: dict[str, Any] = yaml.safe_load(file)
    model_config: model_schema.Model = model_schema.Model(**model_config_dict)

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

    device = optuna_tune.get_device()

    objective = optuna_tune.Objective(
        model_class=model_class,
        network_params=model_config.network_params,
        optimizer_params=model_config.optimizer_params,
        data_params=model_config.data_params,
        loss_params=model_config.loss_params,
        train_torch_dataset=train_dataset,
        val_torch_dataset=validation_dataset,
        artifact_store=artifact_store,
        max_batches=model_config.max_batches,
        compute_objective_every_n_batches=model_config.compute_objective_every_n_batches,
        target_metric=model_config.objective.metric,
        device=device,
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
    best_model_file_path = best_trial.user_attrs["model_path"]
    best_optimizer_file_path = best_trial.user_attrs["optimizer_path"]

    optuna.artifacts.download_artifact(
        artifact_store=artifact_store,
        file_path=best_model_file_path,
        artifact_id=best_model_artifact_id,
    )
    optuna.artifacts.download_artifact(
        artifact_store=artifact_store,
        file_path=best_optimizer_file_path,
        artifact_id=best_optimizer_artifact_id,
    )
    try:
        shutil.move(best_model_file_path, best_model_path)
        shutil.move(best_optimizer_file_path, best_optimizer_path)
    except FileNotFoundError:
        logger.info("Best model or optimizer file_path not found, creating output directories")
        os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
        os.makedirs(os.path.dirname(best_optimizer_path), exist_ok=True)
        shutil.move(best_model_file_path, best_model_path)
        shutil.move(best_optimizer_file_path, best_optimizer_path)
