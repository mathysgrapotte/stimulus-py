"""Tests for CSV data loading and processing functionality."""

import pytest
import yaml

from stimulus.data import experiments
from stimulus.data.data_handlers import (
    DatasetLoader,
    DatasetManager,
    DatasetProcessor,
    EncodeManager,
    SplitManager,
    TransformManager,
)
from stimulus.utils.yaml_data import (
    YamlConfigDict,
    YamlTransform,
    YamlTransformColumns,
    YamlTransformColumnsTransformation,
    generate_data_configs,
)


# Fixtures
## Data fixtures
@pytest.fixture
def titanic_csv_path() -> str:
    """Get path to test Titanic CSV file.

    Returns:
        str: Path to test CSV file
    """
    return "tests/test_data/titanic/titanic_stimulus.csv"


@pytest.fixture
def config_path() -> str:
    """Get path to test config file.

    Returns:
        str: Path to test config file
    """
    return "tests/test_data/titanic/titanic.yaml"


@pytest.fixture
def base_config(config_path: str) -> YamlConfigDict:
    """Load base configuration from YAML file.

    Args:
        config_path: Path to config file

    Returns:
        YamlConfigDict: Loaded configuration
    """
    with open(config_path) as f:
        return YamlConfigDict(**yaml.safe_load(f))


@pytest.fixture
def generate_sub_configs(base_config: YamlConfigDict) -> list[YamlConfigDict]:
    """Generate all possible configurations from base config.

    Args:
        base_config: Base configuration to generate from

    Returns:
        list[YamlConfigDict]: List of generated configurations
    """
    return generate_data_configs(base_config)


@pytest.fixture
def dump_single_split_config_to_disk() -> str:
    """Get path for dumping single split config.

    Returns:
        str: Path to dump config file
    """
    return "tests/test_data/titanic/titanic_sub_config.yaml"


## Loader fixtures
@pytest.fixture
def encoder_loader(generate_sub_configs: list[YamlConfigDict]) -> experiments.EncoderLoader:
    """Create encoder loader with initialized encoders.

    Args:
        generate_sub_configs: List of configurations

    Returns:
        experiments.EncoderLoader: Initialized encoder loader
    """
    loader = experiments.EncoderLoader()
    loader.initialize_column_encoders_from_config(generate_sub_configs[0].columns)
    return loader


@pytest.fixture
def transform_loader(generate_sub_configs: list[YamlConfigDict]) -> experiments.TransformLoader:
    """Create transform loader with initialized transformers.

    Args:
        generate_sub_configs: List of configurations

    Returns:
        experiments.TransformLoader: Initialized transform loader
    """
    loader = experiments.TransformLoader()
    loader.initialize_column_data_transformers_from_config(generate_sub_configs[0].transforms)
    return loader


@pytest.fixture
def split_loader(generate_sub_configs: list[YamlConfigDict]) -> experiments.SplitLoader:
    """Create split loader with initialized splitter.

    Args:
        generate_sub_configs: List of configurations

    Returns:
        experiments.SplitLoader: Initialized split loader
    """
    loader = experiments.SplitLoader()
    loader.initialize_splitter_from_config(generate_sub_configs[0].split)
    return loader


# Test DatasetManager
def test_dataset_manager_init(dump_single_split_config_to_disk: str) -> None:
    """Test initialization of DatasetManager."""
    manager = DatasetManager(dump_single_split_config_to_disk)
    assert hasattr(manager, "config")
    assert hasattr(manager, "column_categories")


def test_dataset_manager_organize_columns(dump_single_split_config_to_disk: str) -> None:
    """Test column organization by type."""
    manager = DatasetManager(dump_single_split_config_to_disk)
    categories = manager.categorize_columns_by_type()

    assert "pclass" in categories["input"]
    assert "sex" in categories["input"]
    assert "age" in categories["input"]
    assert "survived" in categories["label"]
    assert "passenger_id" in categories["meta"]


def test_dataset_manager_organize_transforms(dump_single_split_config_to_disk: str) -> None:
    """Test transform organization."""
    manager = DatasetManager(dump_single_split_config_to_disk)
    categories = manager.categorize_columns_by_type()

    assert len(categories) == 3
    assert all(key in categories for key in ["input", "label", "meta"])


def test_dataset_manager_get_transform_logic(dump_single_split_config_to_disk: str) -> None:
    """Test getting transform logic from config."""
    manager = DatasetManager(dump_single_split_config_to_disk)
    transform_logic = manager.get_transform_logic()
    assert transform_logic["transformation_name"] == "noise"
    assert len(transform_logic["transformations"]) == 2


# Test EncodeManager
def test_encode_manager_init() -> None:
    """Test initialization of EncodeManager."""
    encoder_loader = experiments.EncoderLoader()
    manager = EncodeManager(encoder_loader)
    assert hasattr(manager, "encoder_loader")


def test_encode_manager_initialize_encoders() -> None:
    """Test encoder initialization."""
    encoder_loader = experiments.EncoderLoader()
    manager = EncodeManager(encoder_loader)
    assert hasattr(manager, "encoder_loader")


def test_encode_manager_encode_numeric() -> None:
    """Test numeric encoding."""
    encoder_loader = experiments.EncoderLoader()
    intencoder = encoder_loader.get_encoder("NumericEncoder")
    encoder_loader.set_encoder_as_attribute("test_col", intencoder)
    manager = EncodeManager(encoder_loader)
    data = [1, 2, 3]
    encoded = manager.encode_column("test_col", data)
    assert encoded is not None


# Test TransformManager
def test_transform_manager_init() -> None:
    """Test initialization of TransformManager."""
    transform_loader = experiments.TransformLoader()
    manager = TransformManager(transform_loader)
    assert hasattr(manager, "transform_loader")


def test_transform_manager_initialize_transforms() -> None:
    """Test transform initialization."""
    transform_loader = experiments.TransformLoader()
    manager = TransformManager(transform_loader)
    assert hasattr(manager, "transform_loader")


def test_transform_manager_transform_column() -> None:
    """Test column transformation."""
    transform_loader = experiments.TransformLoader()
    dummy_config = YamlTransform(
        transformation_name="GaussianNoise",
        columns=[
            YamlTransformColumns(
                column_name="test_col",
                transformations=[
                    YamlTransformColumnsTransformation(
                        name="GaussianNoise",
                        params={"std": 0.1},
                    ),
                ],
            ),
        ],
    )
    transform_loader.initialize_column_data_transformers_from_config(dummy_config)
    manager = TransformManager(transform_loader)
    data = [1, 2, 3]
    transformed, added_row = manager.transform_column("test_col", "GaussianNoise", data)
    assert len(transformed) == len(data)
    assert added_row is False


# Test SplitManager
def test_split_manager_init(split_loader: experiments.SplitLoader) -> None:
    """Test initialization of SplitManager."""
    manager = SplitManager(split_loader)
    assert hasattr(manager, "split_loader")


def test_split_manager_initialize_splits(split_loader: experiments.SplitLoader) -> None:
    """Test split initialization."""
    manager = SplitManager(split_loader)
    assert hasattr(manager, "split_loader")


def test_split_manager_apply_split(split_loader: experiments.SplitLoader) -> None:
    """Test applying splits to data."""
    manager = SplitManager(split_loader)
    data = {"col": range(100)}
    split_indices = manager.get_split_indices(data)
    assert len(split_indices) == 3
    assert len(split_indices[0]) == 70
    assert len(split_indices[1]) == 15
    assert len(split_indices[2]) == 15


# Test DatasetProcessor
def test_dataset_processor_init(
    dump_single_split_config_to_disk: str,
    titanic_csv_path: str,
) -> None:
    """Test initialization of DatasetProcessor."""
    processor = DatasetProcessor(
        config_path=dump_single_split_config_to_disk,
        csv_path=titanic_csv_path,
    )

    assert isinstance(processor.dataset_manager, DatasetManager)
    assert processor.columns is not None


def test_dataset_processor_apply_split(
    dump_single_split_config_to_disk: str,
    titanic_csv_path: str,
    split_loader: experiments.SplitLoader,
) -> None:
    """Test applying splits in DatasetProcessor."""
    processor = DatasetProcessor(
        config_path=dump_single_split_config_to_disk,
        csv_path=titanic_csv_path,
    )
    processor.data = processor.load_csv(titanic_csv_path)
    processor.add_split(split_manager=SplitManager(split_loader))
    assert "split" in processor.columns
    assert "split" in processor.data.columns
    assert len(processor.data["split"]) == 712


def test_dataset_processor_apply_transformation_group(
    dump_single_split_config_to_disk: str,
    titanic_csv_path: str,
    transform_loader: experiments.TransformLoader,
) -> None:
    """Test applying transformation groups."""
    processor = DatasetProcessor(
        config_path=dump_single_split_config_to_disk,
        csv_path=titanic_csv_path,
    )
    processor.data = processor.load_csv(titanic_csv_path)

    processor_control = DatasetProcessor(
        config_path=dump_single_split_config_to_disk,
        csv_path=titanic_csv_path,
    )
    processor_control.data = processor_control.load_csv(titanic_csv_path)

    processor.apply_transformation_group(transform_manager=TransformManager(transform_loader))

    assert processor.data["age"].to_list() != processor_control.data["age"].to_list()
    assert processor.data["fare"].to_list() != processor_control.data["fare"].to_list()
    assert processor.data["parch"].to_list() == processor_control.data["parch"].to_list()
    assert processor.data["sibsp"].to_list() == processor_control.data["sibsp"].to_list()
    assert processor.data["pclass"].to_list() == processor_control.data["pclass"].to_list()
    assert processor.data["embarked"].to_list() == processor_control.data["embarked"].to_list()
    assert processor.data["sex"].to_list() == processor_control.data["sex"].to_list()


# Test DatasetLoader
def test_dataset_loader_init(
    dump_single_split_config_to_disk: str,
    titanic_csv_path: str,
    encoder_loader: experiments.EncoderLoader,
) -> None:
    """Test initialization of DatasetLoader."""
    loader = DatasetLoader(
        config_path=dump_single_split_config_to_disk,
        csv_path=titanic_csv_path,
        encoder_loader=encoder_loader,
    )

    assert isinstance(loader.dataset_manager, DatasetManager)
    assert loader.data is not None
    assert loader.columns is not None
    assert hasattr(loader, "encoder_manager")


def test_dataset_loader_get_dataset(
    dump_single_split_config_to_disk: str,
    titanic_csv_path: str,
    encoder_loader: experiments.EncoderLoader,
) -> None:
    """Test getting dataset from loader."""
    loader = DatasetLoader(
        config_path=dump_single_split_config_to_disk,
        csv_path=titanic_csv_path,
        encoder_loader=encoder_loader,
    )

    dataset = loader.get_all_items()
    assert isinstance(dataset, tuple)
    assert len(dataset) == 3  # input_data, label_data, meta_data
