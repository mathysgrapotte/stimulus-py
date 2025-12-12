#!/usr/bin/env python3
"""CLI module for model evaluation on datasets."""

import csv
import json
import logging
from typing import Any, Optional

import safetensors.torch as safetensors
import torch
import yaml

from stimulus.data.interface.dataset_interface import (
    StimulusDataset,
    auto_detect_dataset,
)
from stimulus.learner.device_utils import resolve_device
from stimulus.typing.protocols import StimulusModel
from stimulus.utils.model_file_interface import import_class_from_file

logger = logging.getLogger(__name__)


def load_model(model_path: str, model_config_path: str, weight_path: str) -> StimulusModel:
    """Load model with weights from tune output files.

    Args:
        model_path: Path to the Python file containing the model class.
        model_config_path: Path to best_config.json from tune step.
        weight_path: Path to best_model.safetensors from tune step.

    Returns:
        Loaded model instance with weights.
    """
    with open(model_config_path) as f:
        complete_config = json.load(f)

    # Extract network parameters from complete config
    # best_config.json has structure: {network_params, optimizer_params, data_params}
    network_params = complete_config.get("network_params", complete_config)

    model_class = import_class_from_file(model_path)
    model_instance = model_class(**network_params)

    weights = safetensors.load_file(weight_path)
    model_instance.load_state_dict(weights)
    return model_instance


def flatten_dict(d: dict[str, Any], parent_key: str = "", sep: str = ".") -> dict[str, Any]:
    """Flatten a nested dictionary into a flat dictionary with dotted keys.

    Args:
        d: Dictionary to flatten.
        parent_key: Parent key prefix.
        sep: Separator for nested keys.

    Returns:
        Flattened dictionary.
    """
    items: list[tuple[str, Any]] = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, list):
            # For lists, include index in key
            for i, item in enumerate(v):
                if isinstance(item, dict):
                    items.extend(flatten_dict(item, f"{new_key}.{i}", sep=sep).items())
                else:
                    items.append((f"{new_key}.{i}", item))
        else:
            items.append((new_key, v))
    return dict(items)


def parse_transform_config(config_path: str) -> dict[str, Any]:
    """Parse transform config YAML and flatten all parameters.

    Args:
        config_path: Path to transform config YAML file.

    Returns:
        Flattened dictionary of all transform parameters.
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Flatten the entire config
    flat_config = flatten_dict(config)

    # Convert any remaining complex types to strings
    for key, value in flat_config.items():
        if not isinstance(value, (str, int, float, bool)):
            flat_config[key] = str(value)

    return flat_config


def write_metrics_csv(
    output_path: str,
    metrics: dict[str, float],
    transform_params: Optional[dict[str, Any]] = None,
) -> None:
    """Write metrics to a CSV file.

    Args:
        output_path: Path to output CSV file.
        metrics: Dictionary of metric names to values.
        transform_params: Optional dictionary of transform parameters.
    """
    # Build row data - transform params first, then metrics
    row: dict[str, Any] = {}

    # Add transform parameters if provided
    if transform_params is not None:
        row.update(transform_params)

    # Add all metrics (prefixed to distinguish from transform params)
    for key, value in metrics.items():
        row[f"metric_{key}"] = value

    # Write CSV with header
    with open(output_path, "w", newline="") as csvfile:
        fieldnames = list(row.keys())
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)


def evaluate(
    data_path: str,
    model_path: str,
    model_config_path: str,
    weight_path: str,
    output: str,
    batch_size: int = 256,
    transform_config: Optional[str] = None,
    dataset_cls: type[StimulusDataset] | None = None,
    force_device: Optional[str] = None,
) -> None:
    """Run model evaluation pipeline.

    Args:
        data_path: Path to the input data file (HuggingFace dataset).
        model_path: Path to the model Python file.
        model_config_path: Path to best_config.json from tune step.
        weight_path: Path to best_model.safetensors from tune step.
        output: Path to save the evaluation results CSV.
        batch_size: Batch size for evaluation.
        transform_config: Optional path to transform config YAML file.
        dataset_cls: The dataset class to use for loading.
        force_device: Force a specific device (e.g., "cuda:0", "cpu").
    """
    # Resolve device
    device = resolve_device(force_device=force_device)

    # Load model
    model = load_model(model_path, model_config_path, weight_path)
    model = model.to(device)
    model.eval()

    # Load dataset
    if dataset_cls is None:
        dataset_cls = auto_detect_dataset(data_path)

    dataset = dataset_cls.load_from_disk(data_path)

    # Get test split (or fallback to available splits)
    split_names = list(dataset.keys()) if hasattr(dataset, "keys") else dataset.split_names
    if "test" in split_names:
        split_to_use = "test"
    else:
        # Fallback: use first available split
        split_to_use = split_names[0]
        logger.warning(f"No 'test' split found. Using '{split_to_use}' split for evaluation.")

    # Get torch dataset for the split
    torch_dataset = dataset.get_torch_dataset(split_to_use)

    # Create data loader
    loader = torch.utils.data.DataLoader(
        torch_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    # Run validation - model.validate() handles everything
    logger.info(f"Running evaluation on {len(torch_dataset)} samples from '{split_to_use}' split...")
    metrics = model.validate(loader)

    logger.info(f"Evaluation metrics: {metrics}")

    # Parse transform config if provided
    transform_params = None
    if transform_config is not None:
        transform_params = parse_transform_config(transform_config)
        logger.info(f"Transform parameters: {transform_params}")

    # Write metrics to CSV
    write_metrics_csv(output, metrics, transform_params)

    logger.info(f"Results saved to {output}")
