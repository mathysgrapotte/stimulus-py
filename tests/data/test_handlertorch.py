"""Tests for PyTorch data handling functionality."""

import os

import pytest
import yaml

from stimulus.data import experiments, handlertorch
from stimulus.utils import yaml_data


@pytest.fixture
def titanic_config_path() -> str:
    """Get path to Titanic config file.

    Returns:
        str: Absolute path to the config file
    """
    return os.path.abspath("tests/test_data/titanic/titanic_sub_config.yaml")


@pytest.fixture
def titanic_csv_path() -> str:
    """Get path to Titanic CSV file.

    Returns:
        str: Absolute path to the CSV file
    """
    return os.path.abspath("tests/test_data/titanic/titanic_stimulus.csv")


@pytest.fixture
def titanic_yaml_config(titanic_config_path: str) -> dict:
    """Load Titanic YAML config.

    Args:
        titanic_config_path: Path to the config file

    Returns:
        dict: Loaded YAML configuration
    """
    with open(titanic_config_path) as file:
        return yaml_data.YamlSubConfigDict(**yaml.safe_load(file))


@pytest.fixture
def titanic_encoder_loader(titanic_yaml_config: yaml_data.YamlSubConfigDict) -> experiments.EncoderLoader:
    """Get Titanic encoder loader."""
    loader = experiments.EncoderLoader()
    loader.initialize_column_encoders_from_config(titanic_yaml_config.columns)
    return loader


def test_init_handlertorch(
    titanic_config_path: str,
    titanic_csv_path: str,
    titanic_encoder_loader: experiments.EncoderLoader,
) -> None:
    """Test TorchDataset initialization."""
    handlertorch.TorchDataset(
        config_path=titanic_config_path,
        csv_path=titanic_csv_path,
        encoder_loader=titanic_encoder_loader,
    )


def test_len_handlertorch(
    titanic_config_path: str,
    titanic_csv_path: str,
    titanic_encoder_loader: experiments.EncoderLoader,
) -> None:
    """Test length functionality of TorchDataset.

    Args:
        titanic_config_path: Path to config file
        titanic_csv_path: Path to CSV file
        titanic_encoder_loader: Encoder loader instance
    """
    dataset = handlertorch.TorchDataset(
        config_path=titanic_config_path,
        csv_path=titanic_csv_path,
        encoder_loader=titanic_encoder_loader,
    )
    assert len(dataset) == 712


def test_getitem_handlertorch_slice(
    titanic_config_path: str,
    titanic_csv_path: str,
    titanic_encoder_loader: experiments.EncoderLoader,
) -> None:
    """Test slice indexing functionality of TorchDataset.

    Args:
        titanic_config_path: Path to config file
        titanic_csv_path: Path to CSV file
        titanic_encoder_loader: Encoder loader instance
    """
    dataset = handlertorch.TorchDataset(
        config_path=titanic_config_path,
        csv_path=titanic_csv_path,
        encoder_loader=titanic_encoder_loader,
    )
    assert len(dataset[0:5]) == 3
    assert len(dataset[0:5][0]["pclass"]) == 5


def test_getitem_handlertorch_int(
    titanic_config_path: str,
    titanic_csv_path: str,
    titanic_encoder_loader: experiments.EncoderLoader,
) -> None:
    """Test integer indexing functionality of TorchDataset.

    Args:
        titanic_config_path: Path to config file
        titanic_csv_path: Path to CSV file
        titanic_encoder_loader: Encoder loader instance
    """
    dataset = handlertorch.TorchDataset(
        config_path=titanic_config_path,
        csv_path=titanic_csv_path,
        encoder_loader=titanic_encoder_loader,
    )
    assert len(dataset[0]) == 3
