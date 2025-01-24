#!/usr/bin/env python3
"""CLI module for model prediction on datasets."""

import argparse
import json
from collections.abc import Sequence
from typing import Any

import polars as pl
import torch
from torch.utils.data import DataLoader

from stimulus.data.handlertorch import TorchDataset
from stimulus.learner.predict import PredictWrapper
from stimulus.utils.launch_utils import get_experiment, import_class_from_file


def get_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description="Predict model outputs on a dataset.")
    parser.add_argument("-m", "--model", type=str, required=True, metavar="FILE", help="Path to model .py file.")
    parser.add_argument("-w", "--weight", type=str, required=True, metavar="FILE", help="Path to model weights file.")
    parser.add_argument(
        "-mc",
        "--model_config",
        type=str,
        required=True,
        metavar="FILE",
        help="Path to tune config file with model hyperparameters.",
    )
    parser.add_argument(
        "-ec",
        "--experiment_config",
        type=str,
        required=True,
        metavar="FILE",
        help="Path to experiment config for data modification.",
    )
    parser.add_argument("-d", "--data", type=str, required=True, metavar="FILE", help="Path to input data.")
    parser.add_argument("-o", "--output", type=str, required=True, metavar="FILE", help="Path for output predictions.")
    parser.add_argument("--split", type=int, help="Data split to use (default: None).")
    parser.add_argument("--return_labels", action="store_true", help="Include labels with predictions.")

    return parser.parse_args()


def load_model(model_class: Any, weight_path: str, mconfig: dict[str, Any]) -> Any:
    """Load model with hyperparameters and weights.

    Args:
        model_class: Model class to instantiate.
        weight_path: Path to model weights.
        mconfig: Model configuration dictionary.

    Returns:
        Loaded model instance.
    """
    hyperparameters = mconfig["model_params"]
    model = model_class(**hyperparameters)
    model.load_state_dict(torch.load(weight_path))
    return model


def get_batch_size(mconfig: dict[str, Any]) -> int:
    """Get batch size from model config.

    Args:
        mconfig: Model configuration dictionary.

    Returns:
        Batch size to use for predictions.
    """
    default_batch_size = 256
    if "data_params" in mconfig and "batch_size" in mconfig["data_params"]:
        return mconfig["data_params"]["batch_size"]
    return default_batch_size


def parse_y_keys(y: dict[str, Any], data: pl.DataFrame, y_type: str = "pred") -> dict[str, Any]:
    """Parse dictionary keys to match input data format.

    Args:
        y: Dictionary of predictions or labels.
        data: Input DataFrame.
        y_type: Type of values ('pred' or 'label').

    Returns:
        Dictionary with updated keys.
    """
    if not y:
        return y

    parsed_y = {}
    for k1, v1 in y.items():
        for k2 in data.columns:
            if k1 == k2.split(":")[0]:
                new_key = f"{k1}:{y_type}:{k2.split(':')[2]}"
                parsed_y[new_key] = v1

    return parsed_y


def add_meta_info(data: pl.DataFrame, y: dict[str, Any]) -> dict[str, Any]:
    """Add metadata columns to predictions/labels dictionary.

    Args:
        data: Input DataFrame with metadata.
        y: Dictionary of predictions/labels.

    Returns:
        Updated dictionary with metadata.
    """
    keys = get_meta_keys(data.columns)
    for key in keys:
        y[key] = data[key].to_list()
    return y


def get_meta_keys(names: Sequence[str]) -> list[str]:
    """Extract metadata column keys.

    Args:
        names: List of column names.

    Returns:
        List of metadata column keys.
    """
    return [name for name in names if name.split(":")[1] == "meta"]


def main(
    model_path: str,
    weight_path: str,
    mconfig_path: str,
    econfig_path: str,
    data_path: str,
    output: str,
    *,
    return_labels: bool = False,
    split: int | None = None,
) -> None:
    """Run model prediction pipeline.

    Args:
        model_path: Path to model file.
        weight_path: Path to model weights.
        mconfig_path: Path to model config.
        econfig_path: Path to experiment config.
        data_path: Path to input data.
        output: Path for output predictions.
        return_labels: Whether to include labels.
        split: Data split to use.
    """
    with open(mconfig_path) as in_json:
        mconfig = json.load(in_json)

    model_class = import_class_from_file(model_path)
    model = load_model(model_class, weight_path, mconfig)

    with open(econfig_path) as in_json:
        experiment_name = json.load(in_json)["experiment"]
    initialized_experiment_class = get_experiment(experiment_name)

    dataloader = DataLoader(
        TorchDataset(data_path, initialized_experiment_class, split=split),
        batch_size=get_batch_size(mconfig),
        shuffle=False,
    )

    predictor = PredictWrapper(model, dataloader)
    out = predictor.predict(return_labels=return_labels)
    y_pred, y_true = out if return_labels else (out, {})

    y_pred = {k: v.tolist() for k, v in y_pred.items()}
    y_true = {k: v.tolist() for k, v in y_true.items()}

    data = pl.read_csv(data_path)
    y_pred = parse_y_keys(y_pred, data, y_type="pred")
    y_true = parse_y_keys(y_true, data, y_type="label")

    y = {**y_pred, **y_true}
    y = add_meta_info(data, y)
    df = pl.from_dict(y)
    df.write_csv(output)


def run() -> None:
    """Execute model prediction pipeline."""
    args = get_args()
    main(
        args.model,
        args.weight,
        args.model_config,
        args.experiment_config,
        args.data,
        args.output,
        return_labels=args.return_labels,
        split=args.split,
    )


if __name__ == "__main__":
    run()
