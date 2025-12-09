"""Test the data config schema."""

import pytest
from pydantic import ValidationError

from stimulus.data.interface.data_config_schema import DatasetConfig


def test_dataset_config_defaults() -> None:
    """Test DatasetConfig defaults."""
    config = DatasetConfig()
    assert config.type == "HuggingFaceDataset"
    assert config.params == {}


def test_dataset_config_custom() -> None:
    """Test DatasetConfig with custom values."""
    config = DatasetConfig(type="CustomDataset", params={"foo": "bar"})
    assert config.type == "CustomDataset"
    assert config.params == {"foo": "bar"}


def test_dataset_config_invalid() -> None:
    """Test DatasetConfig validation."""
    with pytest.raises(ValidationError):
        DatasetConfig(params="invalid")  # params must be a dict
