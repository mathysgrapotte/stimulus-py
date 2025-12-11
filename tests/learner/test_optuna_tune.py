"""Test the optuna tune."""

import inspect
import logging
import os
import tempfile
import warnings
from typing import Any

import datasets
import optuna
import optuna.storages.journal
import pytest
import torch
import yaml

from stimulus.data.interface.dataset_interface import HuggingFaceDataset
from stimulus.learner import optuna_tune
from stimulus.learner.device_utils import get_device
from stimulus.learner.interface import model_config_parser, model_schema
from stimulus.utils import model_file_interface

logger = logging.getLogger(__name__)

warnings.filterwarnings("error")  # This will convert warnings to exceptions temporarily

TEST_CASES = [
    {
        "name": "titanic",
        "model_path": os.path.join("tests", "test_model", "titanic_perf_model.py"),
        "config_path": os.path.join("tests", "test_model", "titanic_perf_model.yaml"),
        "data_path": os.path.join("tests", "test_data", "titanic_performant", "titanic_encoded_hf"),
    },
]


@pytest.fixture(params=TEST_CASES)
def test_case(request: Any) -> dict:
    """Get a complete test case configuration."""
    case = request.param

    # Load model class
    model_class = model_file_interface.import_class_from_file(case["model_path"])

    # Load model config
    with open(case["config_path"]) as f:
        model_config = yaml.safe_load(f)
    model_config = model_schema.Model(**model_config)

    data = datasets.load_from_disk(case["data_path"])
    stimulus_data = HuggingFaceDataset(data)
    train_data = stimulus_data.get_torch_dataset("train")
    val_data = stimulus_data.get_torch_dataset("val")

    return {
        "name": case["name"],
        "model_class": model_class,
        "model_config": model_config,
        "train_data": train_data,
        "val_data": val_data,
    }


def test_parameter_suggestions(test_case: dict) -> None:
    """Test parameter suggestions for various model configurations."""
    model_config = test_case["model_config"]
    model_class = test_case["model_class"]

    # Create a study and trial
    study = optuna.create_study()
    trial = study.ask()

    # Test network params
    network_suggestions = model_config_parser.suggest_parameters(trial, model_config.network_params)

    logger.info(f"Network suggestions for {test_case['name']}: {network_suggestions}")

    model_instance = model_class(**network_suggestions)
    logger.info(f"Model instance: {model_instance}")
    assert model_instance is not None

    optimizer_suggestions = model_config_parser.suggest_parameters(trial, model_config.optimizer_params)

    logger.info(f"Optimizer suggestions: {optimizer_suggestions}")

    optimizer_class = getattr(torch.optim, optimizer_suggestions["method"])
    optimizer_signature = inspect.signature(optimizer_class)
    optimizer_kwargs = {}
    for name, value in optimizer_suggestions.items():
        if name in optimizer_signature.parameters:
            optimizer_kwargs[name] = value

    optimizer = optimizer_class(model_instance.parameters(), **optimizer_kwargs)
    assert optimizer_class is not None
    assert optimizer is not None


def test_tune_loop(test_case: dict) -> None:
    """Test the tune loop."""
    with tempfile.TemporaryDirectory() as temp_dir:
        train_data, val_data = test_case["train_data"], test_case["val_data"]
        artifact_store = optuna.artifacts.FileSystemArtifactStore(base_path=temp_dir)
        storage = optuna.storages.JournalStorage(
            optuna.storages.journal.JournalFileBackend(os.path.join(temp_dir, "optuna_journal_storage.log")),
        )
        pruner = optuna.pruners.MedianPruner(n_warmup_steps=50, n_startup_trials=2)
        device = get_device()
        objective = optuna_tune.Objective(
            model_class=test_case["model_class"],
            network_params=test_case["model_config"].network_params,
            optimizer_params=test_case["model_config"].optimizer_params,
            data_params=test_case["model_config"].data_params,
            train_torch_dataset=train_data,
            val_torch_dataset=val_data,
            artifact_store=artifact_store,
            max_samples=test_case["model_config"].max_samples,
            compute_objective_every_n_samples=test_case["model_config"].compute_objective_every_n_samples,
            target_metric=test_case["model_config"].objective.metric,
            device=device,
            log_dir=temp_dir,
        )

        logger.info(f"Objective: {objective}")
        study = optuna_tune.tune_loop(
            objective=objective,
            pruner=pruner,
            sampler=optuna.samplers.TPESampler(),
            n_trials=test_case["model_config"].n_trials,
            direction=test_case["model_config"].objective.direction,
            storage=storage,
        )
        assert study is not None
        logger.debug(f"Study: {study}")
        logger.debug(f"Study best trial: {study.best_trial}")
        logger.debug(f"Study direction: {study.direction}")
        logger.debug(f"Study best value: {study.best_value}")
        logger.debug(f"Study best params: {study.best_params}")
        logger.debug(f"Study trials count: {len(study.trials)}")
        for artifact_meta in optuna.artifacts.get_all_artifact_meta(study_or_trial=study):
            logger.debug(artifact_meta)
        # Download the best model
        trial = study.best_trial
        best_artifact_id = trial.user_attrs["model_id"]

        # Create a fresh temp dir for download verification
        with tempfile.TemporaryDirectory() as download_dir:
            download_path = os.path.join(download_dir, "downloaded_model.safetensors")
            optuna.artifacts.download_artifact(
                artifact_store=artifact_store,
                file_path=download_path,
                artifact_id=best_artifact_id,
            )
            assert os.path.exists(download_path)
