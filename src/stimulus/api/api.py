"""Python API for Stimulus CLI functions.

This module provides Python functions that wrap the CLI functionality,
allowing users to call Stimulus functions directly from Python scripts
with in-memory objects.

All functions work with StimulusDataset wrappers, torch models, and configuration
dictionaries.
"""

import json
import logging
import os
import tempfile
from collections import defaultdict
from typing import Any, Optional

import optuna
import pandas as pd
import torch
from safetensors.torch import load_file

from stimulus.data.interface import data_config_parser
from stimulus.data.interface.dataset_interface import StimulusDataset
from stimulus.data.pipelines import encode as encode_pipeline
from stimulus.data.pipelines.transform import transform_batch
from stimulus.data.splitting import splitters
from stimulus.data.transforming import transforms
from stimulus.learner import optuna_tune
from stimulus.learner.compare import compare_tensors as compare_tensors_impl
from stimulus.learner.device_utils import resolve_device
from stimulus.learner.interface import model_config_parser, model_schema
from stimulus.typing.protocols import StimulusModel
from stimulus.utils.model_file_interface import import_class_from_file

logger = logging.getLogger(__name__)


def encode(
    dataset: StimulusDataset,
    encoders: dict[str, Any],
    num_proc: Optional[int] = None,
    *,
    remove_unencoded_columns: bool = True,
) -> StimulusDataset:
    """Encode a dataset using the provided encoders.

    Args:
        dataset: StimulusDataset to encode.
        encoders: Dictionary mapping column names to encoder instances.
        num_proc: Number of processes to use for encoding (default: None for single process).
        remove_unencoded_columns: Whether to remove columns not in encoders config (default: True).

    Returns:
        The encoded StimulusDataset.

    Example:
        >>> from stimulus.data.encoding.encoders import LabelEncoder
        >>> encoders = {"category": LabelEncoder(dtype="int64")}
        >>> encoded_dataset = encode(dataset, encoders)
    """
    logger.info(f"Loaded encoders for columns: {list(encoders.keys())}")

    # Identify columns that aren't in the encoder configuration
    columns_to_remove = set()
    if remove_unencoded_columns:
        # We check all splits to find all columns
        all_columns = set()
        for split_columns in dataset.column_names.values():
            all_columns.update(split_columns)

        encoder_columns = set(encoders.keys())
        columns_to_remove.update(all_columns - encoder_columns)
        if columns_to_remove:
            logger.info(
                f"Removing columns not in encoder configuration: {list(columns_to_remove)}",
            )

    # Apply the encoders to the data
    dataset = dataset.map(
        encode_pipeline.encode_batch,
        batched=True,
        fn_kwargs={"encoders_config": encoders},
        remove_columns=list(columns_to_remove) if columns_to_remove else None,
        num_proc=num_proc,
    )

    logger.info("Dataset encoded successfully.")
    return dataset


def predict(
    dataset: StimulusDataset,
    model: StimulusModel,
    batch_size: int = 256,
) -> dict[str, torch.Tensor]:
    """Run model prediction on the dataset.

    Args:
        dataset: StimulusDataset to predict on.
        model: PyTorch model instance (already loaded with weights).
        batch_size: Batch size for prediction (default: 256).

    Returns:
        Dictionary containing prediction tensors and statistics.

    Example:
        >>> predictions = predict(test_dataset, trained_model)
        >>> print(predictions["predictions"])
    """
    # Get torch datasets for all splits and concatenate them
    split_datasets = dataset.get_torch_dataset(dataset.split_names)
    # Concatenate all split datasets into one
    all_datasets = list(split_datasets.values())
    all_splits_dataset = torch.utils.data.ConcatDataset(all_datasets)
    loader = torch.utils.data.DataLoader(all_splits_dataset, batch_size=batch_size, shuffle=False)

    # create empty tensor for predictions
    is_first_batch = True
    model.eval()

    with torch.no_grad():
        for batch in loader:
            if is_first_batch:
                _loss, statistics = model.batch(batch)
                is_first_batch = False
            else:
                _loss, temp_statistics = model.batch(batch)
                statistics = _update_statistics(statistics, temp_statistics)

    return _convert_dict_to_tensor(statistics)


def split(
    dataset: StimulusDataset,
    splitter: splitters.AbstractSplitter,
    split_columns: list[str],
) -> StimulusDataset:
    """Split a dataset using the provided splitter.

    Args:
        dataset: StimulusDataset to split.
        splitter: Splitter instance (e.g., RandomSplitter, StratifiedSplitter).
        split_columns: List of column names to use for splitting logic.

    Returns:
        Dataset with train/test splits.

    Raises:
        ValueError: If 'test' split already exists.

    Example:
        >>> from stimulus.data.splitting.splitters import RandomSplitter
        >>> splitter = RandomSplitter(test_ratio=0.2, random_state=42)
        >>> split_dataset = split(dataset, splitter, ["target_column"])
    """
    split_names = dataset.split_names
    if "val" in split_names:
        raise ValueError("Validation split already exists. Cannot split again.")

    # We assume we are splitting the 'train' split
    if "train" not in split_names:
        # Fallback to the first split if 'train' is not present
        target_split = split_names[0]
        logger.warning(f"'train' split not found. Using '{target_split}' for splitting.")
    else:
        target_split = "train"

    column_data_dict = {}
    for col_name in split_columns:
        try:
            # Use get_column to retrieve data for splitting logic
            column_data_dict[col_name] = dataset.get_column(target_split, col_name)
        except KeyError as err:
            raise ValueError(
                f"Column '{col_name}' not found in dataset split '{target_split}'",
            ) from err

    if not column_data_dict:
        raise ValueError(
            f"No data columns were extracted for splitting. Input specified columns are {split_columns}",
        )

    train_indices, test_indices = splitter.get_split_indexes(column_data_dict)

    train_split_obj = dataset.select_split(target_split, train_indices)
    val_split_obj = dataset.select_split(target_split, test_indices)

    return dataset.create_from_splits({"train": train_split_obj, "val": val_split_obj})


def transform(
    dataset: StimulusDataset,
    transforms_config: dict[str, list[transforms.AbstractTransform]],
) -> StimulusDataset:
    """Transform a dataset using the provided transformations.

    Args:
        dataset: StimulusDataset to transform.
        transforms_config: Dictionary mapping column names to lists of transform instances.

    Returns:
        Transformed StimulusDataset.

    Example:
        >>> from stimulus.data.transforming.transforms import NoiseTransform
        >>> transforms_config = {"feature": [NoiseTransform(noise_level=0.1)]}
        >>> transformed_dataset = transform(dataset, transforms_config)
    """
    logger.info("Transforms initialized successfully.")

    # Apply the transformations to the data
    # We use map here as the existing transform_batch is designed for batch processing
    dataset = dataset.map(
        transform_batch,
        batched=True,
        fn_kwargs={"transforms_config": transforms_config},
    )

    # Filter out NaN values
    logger.debug(f"Dataset type: {type(dataset)}")
    return dataset.filter(lambda example: not any(pd.isna(value) for value in example.values()))


def tune(
    dataset: StimulusDataset,
    model_class: type[torch.nn.Module],
    model_config: model_schema.Model,
    n_trials: int = 100,
    _max_samples: int = 1000,
    _compute_objective_every_n_samples: int = 50,
    _target_metric: str = "val_loss",
    direction: str = "minimize",
    storage: Optional[optuna.storages.BaseStorage] = None,
    force_device: Optional[str] = None,
) -> tuple[dict[str, Any], torch.nn.Module, dict[str, torch.Tensor]]:
    """Run hyperparameter tuning using Optuna.

    Args:
        dataset: StimulusDataset containing train/test splits.
        model_class: PyTorch model class to tune.
        model_config: Model configuration with tunable parameters.
        n_trials: Number of trials to run (default: 100).
        _max_samples: Maximum samples per trial (default: 1000).
        _compute_objective_every_n_samples: Frequency to compute objective (default: 50).
        _target_metric: Metric to optimize (default: "val_loss").
        direction: Optimization direction ("minimize" or "maximize", default: "minimize").
        storage: Optuna storage backend (default: None for in-memory).
        force_device: Force specific device ("cpu", "cuda", "mps") (default: None for auto).

    Returns:
        Tuple of (best_config, best_model, best_metrics).

    Example:
        >>> config, model, metrics = tune(
        ...     dataset=train_dataset,
        ...     model_class=MyModel,
        ...     model_config=model_config,
        ...     n_trials=50,
        ... )
    """
    device = resolve_device(force_device=force_device, config_device=model_config.device)

    # Convert to torch datasets
    train_torch_dataset = dataset.get_torch_dataset("train")
    val_torch_dataset = dataset.get_torch_dataset("val")

    # Create temporary artifact store
    with tempfile.TemporaryDirectory() as temp_dir:
        artifact_store = optuna.artifacts.FileSystemArtifactStore(base_path=temp_dir)

        # Create objective function
        objective = optuna_tune.Objective(
            model_class=model_class,
            network_params=model_config.network_params,
            optimizer_params=model_config.optimizer_params,
            data_params=model_config.data_params,
            train_torch_dataset=train_torch_dataset,
            val_torch_dataset=val_torch_dataset,
            artifact_store=artifact_store,
            max_samples=model_config.max_samples,
            compute_objective_every_n_samples=model_config.compute_objective_every_n_samples,
            target_metric=model_config.objective.metric,
            device=device,
        )

        # Get pruner and sampler
        pruner = model_config_parser.get_pruner(model_config.pruner)
        sampler = model_config_parser.get_sampler(model_config.sampler)

        # Run tuning
        study = optuna_tune.tune_loop(
            objective=objective,
            pruner=pruner,
            sampler=sampler,
            n_trials=n_trials,
            direction=direction,
            storage=storage,
        )

        # Get best trial and create best model
        best_trial = study.best_trial
        best_config = best_trial.params

        # Recreate best model
        model_suggestions = model_config_parser.suggest_parameters(best_trial, model_config.network_params)
        best_model = model_class(**model_suggestions)

        # Load best weights if available
        if "model_id" in best_trial.user_attrs:
            model_path = artifact_store.download_artifact(
                artifact_id=best_trial.user_attrs["model_id"],
                dst_path=os.path.join(temp_dir, "best_model.safetensors"),
            )
            weights = load_file(model_path)
            best_model.load_state_dict(weights)

        # Get best metrics
        best_metrics = {k: v for k, v in best_trial.user_attrs.items() if k.startswith(("train_", "val_"))}

        return best_config, best_model, best_metrics


def compare_tensors(
    tensor_dicts: list[dict[str, torch.Tensor]],
    mode: str = "cosine_similarity",
) -> dict[str, list[float]]:
    """Compare prediction tensors using various similarity metrics.

    Args:
        tensor_dicts: List of tensor dictionaries to compare.
        mode: Comparison mode ("cosine_similarity" or "discrete_comparison", default: "cosine_similarity").

    Returns:
        Dictionary containing comparison results.

    Example:
        >>> results = compare_tensors([pred1, pred2], mode="cosine_similarity")
        >>> print(results["cosine_similarity"])
    """
    results: dict[str, list[float]] = defaultdict(list)

    for i in range(len(tensor_dicts)):
        for j in range(i + 1, len(tensor_dicts)):
            tensor1 = tensor_dicts[i]
            tensor2 = tensor_dicts[j]
            tensor_comparison = compare_tensors_impl(tensor1, tensor2, mode)

            for key, tensor in tensor_comparison.items():
                if tensor.ndim == 0:
                    results[key].append(tensor.item())
                else:
                    results[key].append(tensor.mean().item())

    return dict(results)


def check_model(
    dataset: StimulusDataset,
    model_class: type[torch.nn.Module],
    model_config: model_schema.Model,
    n_trials: int = 3,
    _max_samples: int = 100,
    force_device: Optional[str] = None,
) -> tuple[dict[str, Any], torch.nn.Module]:
    """Check model configuration and run initial tests.

    Validates that a model can be loaded and trained with the given configuration.
    Performs a small-scale hyperparameter tuning run to verify everything works.

    Args:
        dataset: StimulusDataset containing train/test splits.
        model_class: PyTorch model class to check.
        model_config: Model configuration with tunable parameters.
        n_trials: Number of trials for validation (default: 3).
        _max_samples: Maximum samples per trial (default: 100).
        force_device: Force specific device ("cpu", "cuda", "mps") (default: None for auto).

    Returns:
        Tuple of (best_config, best_model).

    Example:
        >>> config, model = check_model(dataset, MyModel, model_config)
        >>> print("Model validation successful!")
    """
    best_config, best_model, _metrics = tune(
        dataset=dataset,
        model_class=model_class,
        model_config=model_config,
        n_trials=n_trials,
        direction="minimize",
        force_device=force_device,
    )

    logger.info("Model check completed successfully!")
    return best_config, best_model


# Helper functions for creating configurations from dictionaries


def create_encoders_from_config(config_dict: dict) -> dict[str, Any]:
    """Create encoders from a configuration dictionary.

    Args:
        config_dict: Configuration dictionary matching SplitTransformDict schema.

    Returns:
        Dictionary mapping column names to encoder instances.
    """
    data_config_obj = data_config_parser.SplitTransformDict(**config_dict)
    encoders, _input_columns, _label_columns, _meta_columns = data_config_parser.parse_split_transform_config(
        data_config_obj,
    )
    return encoders


def create_splitter_from_config(config_dict: dict) -> tuple[splitters.AbstractSplitter, list[str]]:
    """Create a splitter from a configuration dictionary.

    Args:
        config_dict: Configuration dictionary matching SplitConfigDict schema.

    Returns:
        Tuple of (splitter_instance, split_columns).
    """
    data_config_obj = data_config_parser.SplitConfigDict(**config_dict)
    splitter = data_config_parser.create_splitter(data_config_obj.split)
    return splitter, data_config_obj.split.split_input_columns


def create_transforms_from_config(config_dict: dict) -> dict[str, list[Any]]:
    """Create transforms from a configuration dictionary.

    Args:
        config_dict: Configuration dictionary matching SplitTransformDict schema.

    Returns:
        Dictionary mapping column names to lists of transform instances.
    """
    data_config_obj = data_config_parser.SplitTransformDict(**config_dict)
    return data_config_parser.create_transforms([data_config_obj.transforms])


def load_model_from_files(model_path: str, config_path: str, weights_path: str) -> torch.nn.Module:
    """Load a model from files (convenience function for predict API).

    Args:
        model_path: Path to the model Python file.
        config_path: Path to the model configuration JSON file.
        weights_path: Path to the model weights file (.safetensors).

    Returns:
        Loaded PyTorch model instance.
    """
    with open(config_path) as f:
        best_config = json.load(f)

    model_class = import_class_from_file(model_path)
    model_instance = model_class(**best_config)

    weights = load_file(weights_path)
    model_instance.load_state_dict(weights)
    return model_instance


# Helper functions for internal use


def _update_statistics(statistics: dict, temp_statistics: dict) -> dict:
    """Update the statistics with the new statistics."""
    for key, value in temp_statistics.items():
        if isinstance(value, torch.Tensor):
            if value.ndim == 0:
                try:
                    if statistics[key].ndim == 0:
                        statistics[key] = torch.cat([statistics[key].reshape(1), value.unsqueeze(0)], dim=0)
                    else:
                        statistics[key] = torch.cat([statistics[key], value.unsqueeze(0)], dim=0)
                except RuntimeError as e:
                    raise RuntimeError(
                        f"Error updating statistics: {e}, shape of incoming tensors is {value.shape} and in-place tensor is {statistics[key].shape}",
                    ) from e
            else:
                statistics[key] = torch.cat([statistics[key], value], dim=0)
        elif isinstance(value, (int, float, list)):
            statistics[key] = statistics[key] + value
        else:
            raise TypeError(f"Invalid statistics type: {type(value)}")
    return statistics


def _convert_dict_to_tensor(data: dict) -> dict:
    """Convert a dictionary to contain only tensors."""
    for key, value in data.items():
        if not isinstance(value, torch.Tensor):
            data[key] = torch.tensor(value)
    return data


# Convenience aliases for backwards compatibility
encode_dataset = encode
predict_model = predict
split_dataset = split
transform_dataset = transform
hyperparameter_tune = tune
