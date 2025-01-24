"""Tests for experiment functionality and configuration."""

import pytest
import yaml

from stimulus.data import experiments
from stimulus.data.encoding.encoders import AbstractEncoder
from stimulus.data.splitters import splitters
from stimulus.data.transform import data_transformation_generators
from stimulus.utils import yaml_data


@pytest.fixture
def dna_experiment_config_path() -> str:
    """Fixture that provides the path to the DNA experiment config template YAML file.

    Returns:
        str: Path to the DNA experiment config template YAML file
    """
    return "tests/test_data/dna_experiment/dna_experiment_config_template.yaml"


@pytest.fixture
def dna_experiment_sub_yaml(dna_experiment_config_path: str) -> yaml_data.YamlConfigDict:
    """Get a sub-configuration from the DNA experiment config.

    Args:
        dna_experiment_config_path: Path to the DNA experiment config file

    Returns:
        yaml_data.YamlConfigDict: First generated sub-configuration
    """
    with open(dna_experiment_config_path) as f:
        yaml_dict = yaml.safe_load(f)
        yaml_config = yaml_data.YamlConfigDict(**yaml_dict)

    yaml_configs = yaml_data.generate_data_configs(yaml_config)
    return yaml_configs[0]


@pytest.fixture
def titanic_yaml_path() -> str:
    """Get path to Titanic YAML config file.

    Returns:
        str: Path to Titanic config file
    """
    return "tests/test_data/titanic/titanic.yaml"


@pytest.fixture
def titanic_sub_yaml_path() -> str:
    """Get path to Titanic sub-config YAML file.

    Returns:
        str: Path to Titanic sub-config file
    """
    return "tests/test_data/titanic/titanic_sub_config_0.yaml"


@pytest.fixture
def text_onehot_encoder_params() -> tuple[str, dict[str, str]]:
    """Get TextOneHotEncoder name and parameters.

    Returns:
        tuple[str, dict[str, str]]: Encoder name and parameters
    """
    return "TextOneHotEncoder", {"alphabet": "acgt"}


def test_get_encoder(text_onehot_encoder_params: tuple[str, dict[str, str]]) -> None:
    """Test the get_encoder method of the AbstractExperiment class.

    Args:
        text_onehot_encoder_params: Tuple of encoder name and parameters
    """
    experiment = experiments.EncoderLoader()
    encoder_name, encoder_params = text_onehot_encoder_params
    encoder = experiment.get_encoder(encoder_name, encoder_params)
    assert isinstance(encoder, AbstractEncoder)


def test_set_encoder_as_attribute(text_onehot_encoder_params: tuple[str, dict[str, str]]) -> None:
    """Test the set_encoder_as_attribute method of the AbstractExperiment class.

    Args:
        text_onehot_encoder_params: Tuple of encoder name and parameters
    """
    experiment = experiments.EncoderLoader()
    encoder_name, encoder_params = text_onehot_encoder_params
    encoder = experiment.get_encoder(encoder_name, encoder_params)
    experiment.set_encoder_as_attribute("ciao", encoder)
    assert hasattr(experiment, "ciao")
    assert experiment.ciao == encoder
    assert experiment.get_function_encode_all("ciao") == encoder.encode_all


def test_build_experiment_class_encoder_dict(dna_experiment_sub_yaml: yaml_data.YamlConfigDict) -> None:
    """Test the build_experiment_class_encoder_dict method.

    Args:
        dna_experiment_sub_yaml: DNA experiment sub-configuration
    """
    experiment = experiments.EncoderLoader()
    config = dna_experiment_sub_yaml.columns
    experiment.initialize_column_encoders_from_config(config)
    assert hasattr(experiment, "hello")
    assert hasattr(experiment, "bonjour")
    assert hasattr(experiment, "ciao")

    assert experiment.hello.encode_all(["a", "c", "g", "t"]) is not None


def test_get_data_transformer() -> None:
    """Test the get_data_transformer method of the TransformLoader class."""
    experiment = experiments.TransformLoader()
    transformer = experiment.get_data_transformer("ReverseComplement")
    assert isinstance(transformer, data_transformation_generators.ReverseComplement)


def test_set_data_transformer_as_attribute() -> None:
    """Test the set_data_transformer_as_attribute method."""
    experiment = experiments.TransformLoader()
    transformer = experiment.get_data_transformer("ReverseComplement")
    experiment.set_data_transformer_as_attribute("col1", transformer)
    assert hasattr(experiment, "col1")
    assert experiment.col1["ReverseComplement"] == transformer


def test_initialize_column_data_transformers_from_config(
    dna_experiment_sub_yaml: yaml_data.YamlConfigDict,
) -> None:
    """Test initializing column data transformers from config.

    Args:
        dna_experiment_sub_yaml: DNA experiment sub-configuration
    """
    experiment = experiments.TransformLoader()
    config = dna_experiment_sub_yaml.transforms
    experiment.initialize_column_data_transformers_from_config(config)

    assert hasattr(experiment, "col1")
    column_transformers = experiment.col1
    assert any(isinstance(t, data_transformation_generators.ReverseComplement) for t in column_transformers.values())


def test_initialize_splitter_from_config(
    dna_experiment_sub_yaml: yaml_data.YamlConfigDict,
) -> None:
    """Test initializing splitter from configuration.

    Args:
        dna_experiment_sub_yaml: DNA experiment sub-configuration
    """
    experiment = experiments.SplitLoader()
    config = dna_experiment_sub_yaml.split
    experiment.initialize_splitter_from_config(config)
    assert hasattr(experiment, "split")
    assert isinstance(experiment.split, splitters.RandomSplit)
