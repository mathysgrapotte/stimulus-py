#!/usr/bin/env python3
"""CLI module for model prediction on datasets."""

import json
import logging

import datasets
import safetensors.torch as safetensors
import torch

from stimulus.data.interface.dataset_interface import (
    StimulusDataset,
    auto_detect_dataset,
)
from stimulus.typing.protocols import StimulusModel
from stimulus.utils.model_file_interface import import_class_from_file

logger = logging.getLogger(__name__)


def load_model(model_path: str, model_config_path: str, weight_path: str) -> StimulusModel:
    """Dynamically loads the model from a .py file."""
    with open(model_config_path) as f:
        complete_config = json.load(f)

    # Extract network parameters from complete config
    # Handle both old format (direct params) and new format (nested params)
    network_params = complete_config.get("network_params", complete_config)

    # Check that the model can be loaded
    model = import_class_from_file(model_path)
    model_instance = model(**network_params)

    weights = safetensors.load_file(weight_path)
    model_instance.load_state_dict(weights)
    return model_instance


def update_statistics(statistics: dict, temp_statistics: dict) -> dict:
    """Update the statistics with the new statistics.

    Args:
        statistics: The statistics to update.
        temp_statistics: The new statistics to update with.

    Returns:
        The updated statistics.
    """
    for key, value in temp_statistics.items():
        if isinstance(value, torch.Tensor):
            if value.ndim == 0:
                try:
                    # Check if statistics[key] is zero-dimensional and reshape it if needed
                    if statistics[key].ndim == 0:
                        statistics[key] = torch.cat([statistics[key].reshape(1), value.unsqueeze(0)], dim=0)
                    else:
                        statistics[key] = torch.cat([statistics[key], value.unsqueeze(0)], dim=0)
                except RuntimeError as e:
                    raise RuntimeError(
                        f"Error updating statistics: {e}, shape of incoming tensors is {value.shape} and in-place tensor is {statistics[key].shape}, values of those tensors are {value} and {statistics[key]}",
                    ) from e
            else:
                statistics[key] = torch.cat([statistics[key], value], dim=0)
        elif isinstance(value, (int, float, list)):
            statistics[key] = statistics[key] + value
        else:
            raise TypeError(f"Invalid statistics type: {type(value)}")

    return statistics


def convert_dict_to_tensor(data: dict) -> dict:
    """Convert a dictionary to a tensor.

    Args:
        data: The dictionary to convert.

    Returns:
        The converted dictionary.
    """
    for key, value in data.items():
        if not isinstance(value, torch.Tensor):
            data[key] = torch.tensor(value)
    return data


def predict(
    data_path: str,
    model_path: str,
    model_config_path: str,
    weight_path: str,
    output: str,
    batch_size: int = 256,
    dataset_cls: type[StimulusDataset] | None = None,
) -> None:
    """Run model prediction pipeline.

    Args:
        data_path: Path to the input data file.
        model_path: Path to the model file.
        model_config_path: Path to the model config YAML file.
        weight_path: Path to the model weights file.
        output: Path to save the prediction results.
        batch_size: Batch size for prediction.
        dataset_cls: The dataset class to use for loading.
    """
    # Load model configuration to get loss function
    with open(model_config_path) as file:
        complete_config = json.load(file)

    # Get loss function from the optimized config
    loss_params = complete_config.get("loss_params", {})
    if not loss_params:
        raise ValueError(f"No loss_params found in model config: {model_config_path}")

    # Extract loss function name from optimized parameters
    # The loss function name should be directly available (e.g., {"loss_fn": "BCEWithLogitsLoss"})
    loss_fn_name = None
    for _param_name, param_value in loss_params.items():
        if isinstance(param_value, str):
            loss_fn_name = param_value
            break

    # Backward compatibility for nested params?
    # No, assuming flat loss_params for now as per schema

    if not loss_fn_name:
        raise ValueError(f"Could not extract loss function from loss_params: {loss_params}")

    # Create loss function instance
    try:
        loss_fn = getattr(torch.nn, loss_fn_name)()
    except AttributeError as e:
        raise ValueError(f"Invalid loss function '{loss_fn_name}' in config") from e

    logger.info(f"Using loss function: {loss_fn_name}")

    # Get the best model with best architecture and weights
    model = load_model(model_path, model_config_path, weight_path)

    if dataset_cls is None:
        dataset_cls = auto_detect_dataset(data_path)

    dataset = dataset_cls.load_from_disk(data_path)
    dataset.set_format(type="torch")
    splits = [dataset[split_name] for split_name in dataset]
    all_splits = datasets.concatenate_datasets(splits)
    loader = torch.utils.data.DataLoader(all_splits, batch_size=batch_size, shuffle=False)

    # create empty tensor for predictions
    is_first_batch = True
    for batch in loader:
        if is_first_batch:
            _loss, statistics = model.inference(batch, loss_fn)
            is_first_batch = False
        _loss, temp_statistics = model.inference(batch, loss_fn)
        statistics = update_statistics(statistics, temp_statistics)

    to_return: dict = convert_dict_to_tensor(statistics)
    # Predict the data
    safetensors.save_file(to_return, output)
