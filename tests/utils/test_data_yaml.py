"""Tests for YAML data handling functionality."""

import pytest
import yaml

from src.stimulus.utils import yaml_data
from src.stimulus.utils.yaml_data import (
    YamlConfigDict,
    YamlSplitConfigDict,
    YamlSplitTransformDict,
    generate_split_configs,
    generate_split_transform_configs,
)


@pytest.fixture
def titanic_csv_path() -> str:
    """Get path to Titanic CSV file."""
    return "tests/test_data/titanic/titanic_stimulus.csv"


@pytest.fixture
def load_titanic_yaml_from_file() -> YamlConfigDict:
    """Fixture that loads a test YAML configuration file."""
    with open("tests/test_data/titanic/titanic.yaml") as f:
        yaml_dict = yaml.safe_load(f)
        return YamlConfigDict(**yaml_dict)


@pytest.fixture
def load_split_config_yaml_from_file() -> YamlSplitConfigDict:
    """Fixture that loads a test unique split YAML configuration file."""
    with open("tests/test_data/titanic/titanic_unique_split.yaml") as f:
        yaml_dict = yaml.safe_load(f)
        return YamlSplitConfigDict(**yaml_dict)


@pytest.fixture
def load_yaml_from_file() -> YamlConfigDict:
    """Fixture that loads a test YAML configuration file."""
    with open(
        "tests/test_data/dna_experiment/dna_experiment_config_template.yaml"
    ) as f:
        yaml_dict = yaml.safe_load(f)
        return YamlConfigDict(**yaml_dict)


@pytest.fixture
def load_wrong_type_yaml() -> dict:
    """Fixture that loads a YAML configuration file with wrong typing."""
    with open("tests/test_data/yaml_files/wrong_field_type.yaml") as f:
        return yaml.safe_load(f)


def test_split_config_validation(load_titanic_yaml_from_file: YamlConfigDict) -> None:
    """Test split configuration validation."""
    split_config = generate_split_configs(load_titanic_yaml_from_file)[0]
    YamlSplitConfigDict.model_validate(split_config)


def test_sub_config_validation(
    load_split_config_yaml_from_file: YamlConfigDict,
) -> None:
    """Test sub-config validation."""
    split_config = generate_split_transform_configs(load_split_config_yaml_from_file)[0]
    print(f"{split_config=}")
    YamlSplitTransformDict.model_validate(split_config)


def test_expand_transform_parameter_combinations(
    load_yaml_from_file: YamlConfigDict,
) -> None:
    """Tests expanding transforms with parameter lists into individual transforms."""
    # Test transform with multiple parameter lists
    transform = load_yaml_from_file.transforms[0]
    results = yaml_data.expand_transform_parameter_combinations(transform)
    assert len(results) == 1  # Only one transform returned
    assert isinstance(
        results[0], yaml_data.Transform
    )  # Should return Transform objects


def test_expand_transform_list_combinations(
    load_yaml_from_file: YamlConfigDict,
) -> None:
    """Tests expanding a list of transforms into all parameter combinations."""
    results = yaml_data.expand_transform_list_combinations(
        load_yaml_from_file.transforms
    )
    # 4 combinations from first transform x 2 from second
    assert len(results) == 8
    # Each result should be a Transform
    for result in results:
        assert isinstance(result, yaml_data.Transform)
        assert isinstance(result.transformation_name, str)
        assert isinstance(result.columns, list)


def test_generate_data_configs(
    load_yaml_from_file: YamlConfigDict,
) -> None:
    """Tests generating all possible data configurations."""
    split_configs = yaml_data.generate_split_configs(load_yaml_from_file)
    configs: list[YamlSplitTransformDict] = []
    for s_conf in split_configs:
        configs.extend(generate_split_transform_configs(s_conf))

    assert len(configs) == 16  # 8 transform combinations x 2 splits

    # Check each config individually to help debug
    for i, config in enumerate(configs):
        assert isinstance(
            config,
            yaml_data.YamlSplitTransformDict,
        ), f"Config {i} is type {type(config)}, expected YamlSubConfigDict"


@pytest.mark.parametrize(
    "test_input",
    [("load_yaml_from_file", False), ("load_wrong_type_yaml", True)],
)
def test_check_yaml_schema(
    request: pytest.FixtureRequest,
    test_input: tuple[str, bool],
) -> None:
    """Tests the Pydantic schema validation."""
    data = request.getfixturevalue(test_input[0])
    if test_input[1]:
        with pytest.raises(
            ValueError, match="Wrong type on a field, see the pydantic report above"
        ):
            yaml_data.check_yaml_schema(data)
    else:
        yaml_data.check_yaml_schema(data)
