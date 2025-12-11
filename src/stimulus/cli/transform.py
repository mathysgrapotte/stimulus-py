#!/usr/bin/env python3
"""CLI module for transforming data files."""

import logging
from typing import Any

import pandas as pd

from stimulus.data.interface.dataset_interface import (
    StimulusDataset,
    auto_detect_dataset,
)
from stimulus.data.pipelines import transform as transform_pipeline

logger = logging.getLogger(__name__)


def transform(
    data_path: str,
    config_yaml: str,
    out_path: str,
    dataset_cls: type[StimulusDataset] | None = None,
) -> None:
    """Transform the data according to the configuration.

    Args:
        data_path: Path to input data file (Parquet) or directory.
        config_yaml: Path to config YAML file.
        out_path: Path to output transformed dataset.
        dataset_cls: The dataset class to use for loading.
    """
    if dataset_cls is None:
        dataset_cls = auto_detect_dataset(data_path)

    dataset = dataset_cls.load_from_disk(data_path)

    # Create transforms from the config
    transforms = transform_pipeline.load_transforms_from_config(config_yaml)
    logger.info("Transforms initialized successfully.")

    # Separate dataset-level transforms from element/batch-level transforms
    dataset_transforms = []
    element_transforms: dict[str, list[Any]] = {}

    for col, transform_list in transforms.items():
        element_transforms[col] = []
        for t in transform_list:
            if getattr(t, "scope", "element") == "dataset":
                dataset_transforms.append(t)
            else:
                element_transforms[col].append(t)

    # Apply dataset-level transforms first
    for t in dataset_transforms:
        logger.info(f"Applying dataset transform: {t}")
        dataset = dataset.apply(t)

    # Apply element/batch-level transformations to the data
    # Only map if there are element transforms
    if any(element_transforms.values()):
        dataset = dataset.map(
            transform_pipeline.transform_batch,
            batched=True,
            fn_kwargs={"transforms_config": element_transforms},
        )

    logger.debug(f"Dataset type: {type(dataset)}")

    # Filter out NaN values
    # Some datasets (like H5adDataset) might not support row-wise filtering seamlessly yet.
    # If they are used with global transforms, they manage their own consistency.
    try:
        dataset = dataset.filter(lambda example: not any(pd.isna(value) for value in example.values()))
    except NotImplementedError:
        logger.warning(
            f"Filtering not supported for dataset type {type(dataset).__name__}. "
            "Skipping NaN filtering. Ensure your global transforms handle data consistency.",
        )

    dataset.save(out_path)
