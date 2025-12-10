"""Test the model schema."""

import logging
import os
from typing import Any

import pytest
import yaml

from src.stimulus.learner.interface import model_schema
from src.stimulus.learner.interface.model_schema import Model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest.fixture
def get_config() -> dict[str, Any]:
    """Get the config."""
    config_path = os.path.join("tests", "test_model", "conv_kan.yaml")
    with open(config_path) as f:
        return yaml.safe_load(f)


def test_model_schema(get_config: dict[str, Any]) -> None:
    """Test the model schema."""
    model = Model(**get_config)
    # Test model_params
    assert isinstance(model.network_params["conv_kernel_number"], model_schema.TunableParameter)
    assert model.network_params["conv_kernel_number"].params["low"] == 3
    assert model.network_params["conv_kernel_number"].params["high"] == 10
    assert model.network_params["conv_kernel_number"].mode == "int"

    assert isinstance(model.network_params["conv_kernel_size"], model_schema.TunableParameter)
    assert model.network_params["conv_kernel_size"].params["low"] == 3
    assert model.network_params["conv_kernel_size"].params["high"] == 10
    assert model.network_params["conv_kernel_size"].mode == "int"

    assert isinstance(model.network_params["kan_grid_size"], model_schema.TunableParameter)
    assert model.network_params["kan_grid_size"].params["low"] == 3
    assert model.network_params["kan_grid_size"].params["high"] == 10
    assert model.network_params["kan_grid_size"].mode == "int"

    # Test variable list
    assert isinstance(model.network_params["kan_layers_hidden"], model_schema.VariableList)
    assert model.network_params["kan_layers_hidden"].mode == "variable_list"
    assert model.network_params["kan_layers_hidden"].length.params["low"] == 1
    assert model.network_params["kan_layers_hidden"].length.params["high"] == 5
    assert model.network_params["kan_layers_hidden"].length.mode == "int"
    assert model.network_params["kan_layers_hidden"].values.params["low"] == 3
    assert model.network_params["kan_layers_hidden"].values.params["high"] == 10
    assert model.network_params["kan_layers_hidden"].values.mode == "int"

    # Test optimizer_params
    assert isinstance(model.optimizer_params["method"], model_schema.TunableParameter)
    assert model.optimizer_params["method"].params["choices"] == ["Adam", "SGD"]
    assert model.optimizer_params["method"].mode == "categorical"

    assert isinstance(model.optimizer_params["lr"], model_schema.TunableParameter)
    assert model.optimizer_params["lr"].params["low"] == 0.0001
    assert model.optimizer_params["lr"].params["high"] == 0.1
    assert model.optimizer_params["lr"].params["log"] is True
    assert model.optimizer_params["lr"].mode == "float"

    # Test loss_params
    assert isinstance(model.loss_params["loss_fn"], model_schema.TunableParameter)
    assert model.loss_params["loss_fn"].params["choices"] == ["BCELoss"]
    assert model.loss_params["loss_fn"].mode == "categorical"

    # Test data_params
    assert isinstance(model.data_params["batch_size"], model_schema.TunableParameter)
    assert model.data_params["batch_size"].params["choices"] == [16, 32, 64, 128, 256, 512]
    assert model.data_params["batch_size"].mode == "categorical"

    # Test pruner and sampler
    assert model.pruner.name == "MedianPruner"
    assert model.pruner.params["n_warmup_steps"] == 10
    assert model.pruner.params["n_startup_trials"] == 2

    assert model.sampler.name == "TPESampler"
    assert model.sampler.params is None

    # Test objective
    assert model.objective.metric == "val_loss"
    assert model.objective.direction == "minimize"
