"""CLI module for encoding data files."""

import logging
from typing import Optional

from stimulus.data.interface.dataset_interface import (
    StimulusDataset,
    auto_detect_dataset,
)
from stimulus.data.pipelines import encode as encode_pipeline

logger = logging.getLogger(__name__)


def encode(
    data_path: str,
    config_yaml: str,
    out_path: str,
    _num_proc: Optional[int] = None,
    dataset_cls: type[StimulusDataset] | None = None,
) -> None:
    """Encode the data according to the configuration.

    Args:
        data_path: Path to input data (Parquet or HuggingFace dataset directory).
        config_yaml: Path to config YAML file.
        out_path: Path to output encoded dataset directory.
        _num_proc: Number of processes to use for encoding.
        dataset_cls: The dataset class to use for loading.
    """
    # Load the dataset
    if dataset_cls is None:
        dataset_cls = auto_detect_dataset(data_path)

    dataset = dataset_cls.load_from_disk(data_path)

    # Load encoders from config
    encoders = encode_pipeline.load_encoders_from_config(config_yaml)
    logger.info("Encoders initialized successfully.")
    logger.info(f"Loaded encoders for columns: {list(encoders.keys())}")

    logger.info(f"Loaded encoders for columns: {list(encoders.keys())}")

    # Use EncodeTransform
    transform = encode_pipeline.EncodeTransform(encoders)
    dataset = dataset.apply(transform)

    logger.info(f"Dataset encoded successfully. Saving to: {out_path}")

    # Save the encoded dataset to disk
    dataset.save(out_path)

    logger.info(f"Encoded dataset saved to: {out_path}")
