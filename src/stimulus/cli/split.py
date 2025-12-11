#!/usr/bin/env python3
"""CLI module for splitting data files.

Module currently under modification to be integrated with huggingface datasets.
Current design choices :
- Focus on train/val splits
- Splitter class gets a dict as input
- We use save_to_disk to save the dataset to the disk with both splits at once.
"""

import logging

from stimulus.api import split as split_api
from stimulus.data.interface.dataset_interface import (
    StimulusDataset,
    auto_detect_dataset,
)
from stimulus.data.pipelines import split as split_pipeline

logger = logging.getLogger(__name__)


def split(
    data_path: str,
    config_yaml: str,
    out_path: str,
    dataset_cls: type[StimulusDataset] | None = None,
) -> None:
    """Split the data according to the configuration.

    Args:
        data_path: Path to input data file (Parquet) or directory.
        config_yaml: Path to config YAML file.
        out_path: Path to output split dataset.
        dataset_cls: The dataset class to use for loading.
    """
    # create a splitter object from the config
    splitter, split_columns = split_pipeline.load_splitters_from_config_from_path(config_yaml)

    # Load dataset using the unified loader
    if dataset_cls is None:
        dataset_cls = auto_detect_dataset(data_path)

    dataset = dataset_cls.load_from_disk(data_path)

    # Perform the split using the API
    # This will raise ValueError if 'test' split already exists
    split_dataset = split_api(dataset, splitter, split_columns)

    # Save the result
    split_dataset.save(out_path)
