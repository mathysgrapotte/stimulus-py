"""Tests for the encode CLI command."""

import os
import pathlib
import tempfile

import datasets
import pandas as pd
import pytest
import yaml
from click.testing import CliRunner

from stimulus.cli.encode import encode
from stimulus.cli.main import cli
from stimulus.data.encoding.encoders import NumericEncoder, StrClassificationEncoder
from stimulus.data.pipelines.encode import encode_batch, load_encoders_from_config


# Fixtures
@pytest.fixture
def parquet_path() -> str:
    """Fixture that returns the path to a Parquet file."""
    return str(
        pathlib.Path(__file__).parent.parent / "test_data" / "titanic" / "titanic_stimulus.parquet",
    )


@pytest.fixture
def yaml_path() -> str:
    """Fixture that returns the path to a YAML config file."""
    return str(
        pathlib.Path(__file__).parent.parent / "test_data" / "titanic" / "titanic_unique_transform.yaml",
    )


@pytest.mark.skip(reason="EncodeTransform not yet implemented")
def test_encode_main_function(
    parquet_path: str,
    yaml_path: str,
) -> None:
    """Tests the encode main function with correct YAML files."""
    # Verify required files exist
    assert os.path.exists(parquet_path), f"Parquet file not found at {parquet_path}"
    assert os.path.exists(yaml_path), f"YAML config not found at {yaml_path}"

    # Create temporary output directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        output_path = os.path.join(tmp_dir, "encoded_dataset")

        # Run main function - should complete without errors
        encode(
            data_path=parquet_path,
            config_yaml=yaml_path,
            out_path=output_path,
        )

        # Verify output directory exists and has content
        assert os.path.exists(output_path), "Output directory was not created"

        # Verify that the dataset can be loaded back
        loaded_dataset = datasets.load_from_disk(output_path)
        assert isinstance(loaded_dataset, datasets.DatasetDict), "Failed to load encoded dataset"
        assert "train" in loaded_dataset, "Train split not found in encoded dataset"

        # Verify that the dataset has the expected columns
        expected_columns = [
            "passenger_id",
            "survived",
            "pclass",
            "sex",
            "age",
            "sibsp",
            "parch",
            "fare",
            "embarked",
        ]
        for column in expected_columns:
            assert column in loaded_dataset["train"].column_names, f"Column {column} not found"


@pytest.mark.skip(reason="Break github action runners")
def test_encode_with_parquet_input(
    yaml_path: str,
) -> None:
    """Test encode with parquet input file."""
    # Create a temporary parquet file for testing
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create a simple dataset and save as parquet
        test_data = {
            "passenger_id": [1, 2, 3],
            "survived": [1, 0, 1],
            "pclass": [1, 2, 3],
            "sex": ["male", "female", "male"],
            "age": [22, 38, 26],
            "sibsp": [1, 1, 0],
            "parch": [0, 0, 0],
            "fare": [7.25, 71.28, 7.92],
            "embarked": ["S", "C", "S"],
        }

        df = pd.DataFrame(test_data)
        parquet_path = os.path.join(tmp_dir, "test_data.parquet")
        df.to_parquet(parquet_path, index=False)

        output_path = os.path.join(tmp_dir, "encoded_dataset")

        # Run main function with parquet input
        encode(
            data_path=parquet_path,
            config_yaml=yaml_path,
            out_path=output_path,
        )

        # Verify output
        assert os.path.exists(output_path), "Output directory was not created"
        loaded_dataset = datasets.load_from_disk(output_path)
        assert isinstance(loaded_dataset, datasets.DatasetDict), "Failed to load encoded dataset"


@pytest.mark.skip(reason="Break github action runners")
def test_encode_with_dataset_directory_input(
    parquet_path: str,
    yaml_path: str,
) -> None:
    """Test encode with HuggingFace dataset directory input."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        # First create a dataset directory from Parquet
        dataset = datasets.load_dataset("parquet", data_files=parquet_path)
        dataset_dir = os.path.join(tmp_dir, "input_dataset")
        dataset.save_to_disk(dataset_dir)

        output_path = os.path.join(tmp_dir, "encoded_dataset")

        # Run main function with dataset directory input
        encode(
            data_path=dataset_dir,
            config_yaml=yaml_path,
            out_path=output_path,
        )

        # Verify output
        assert os.path.exists(output_path), "Output directory was not created"
        loaded_dataset = datasets.load_from_disk(output_path)
        assert isinstance(loaded_dataset, datasets.DatasetDict), "Failed to load encoded dataset"


@pytest.mark.skip(reason="Break github action runners")
def test_encode_with_missing_column_graceful_handling(
    parquet_path: str,
    yaml_path: str,
) -> None:
    """Test that encode handles missing columns gracefully without crashing."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create a modified config with a non-existent column
        with open(yaml_path) as f:
            config = yaml.safe_load(f)

        # Add a non-existent column to the config
        config["columns"].append(
            {
                "column_name": "non_existent_column",
                "column_type": "input",
                "encoder": [
                    {
                        "name": "NumericEncoder",
                        "params": {"dtype": "float32"},
                    },
                ],
            },
        )

        modified_yaml_path = os.path.join(tmp_dir, "modified_config.yaml")
        with open(modified_yaml_path, "w") as f:
            yaml.dump(config, f)

        output_path = os.path.join(tmp_dir, "encoded_dataset")

        # Run main function - should complete without errors even with non-existent column
        encode(
            data_path=parquet_path,
            config_yaml=modified_yaml_path,
            out_path=output_path,
        )

        # Verify output directory exists and dataset was created successfully
        assert os.path.exists(output_path), "Output directory was not created"
        loaded_dataset = datasets.load_from_disk(output_path)
        assert isinstance(loaded_dataset, datasets.DatasetDict), "Failed to load encoded dataset"

        # Verify that the original columns are still present and encoded
        expected_columns = [
            "passenger_id",
            "survived",
            "pclass",
            "sex",
            "age",
            "sibsp",
            "parch",
            "fare",
            "embarked",
        ]
        for column in expected_columns:
            assert column in loaded_dataset["train"].column_names, f"Column {column} not found"


@pytest.mark.skip(reason="Break github action runners")
def test_cli_invocation(
    parquet_path: str,
    yaml_path: str,
) -> None:
    """Test the CLI invocation of encode command."""
    runner = CliRunner()
    with runner.isolated_filesystem():
        output_path = "encoded_dataset"
        result = runner.invoke(
            cli,
            [
                "encode",
                "-d",
                parquet_path,
                "-y",
                yaml_path,
                "-o",
                output_path,
            ],
        )
        assert result.exit_code == 0
        assert os.path.exists(output_path), "Output directory was not created"


@pytest.mark.skip(reason="Break github action runners")
def test_load_encoders_from_config(yaml_path: str) -> None:
    """Test that encoders are properly loaded from config."""
    encoders = load_encoders_from_config(yaml_path)

    # Check that encoders were created for all columns
    expected_columns = [
        "passenger_id",
        "survived",
        "pclass",
        "sex",
        "age",
        "sibsp",
        "parch",
        "fare",
        "embarked",
    ]

    for column in expected_columns:
        assert column in encoders, f"Encoder for column {column} not found"

    # Check that the encoders are the correct types
    assert isinstance(encoders["passenger_id"], NumericEncoder)
    assert isinstance(encoders["survived"], NumericEncoder)
    assert isinstance(encoders["sex"], StrClassificationEncoder)
    assert isinstance(encoders["embarked"], StrClassificationEncoder)


@pytest.mark.skip(reason="Break github action runners")
def test_encode_batch_function() -> None:
    """Test the encode_batch function with sample data."""
    # Create sample batch data
    batch = {
        "numeric_col": [1.0, 2.0, 3.0],
        "string_col": ["A", "B", "A"],
    }

    # Create encoders
    encoders_config = {
        "numeric_col": NumericEncoder(),
        "string_col": StrClassificationEncoder(),
    }

    # Encode the batch
    result = encode_batch(batch, encoders_config)

    # Check results
    assert "numeric_col" in result
    assert "string_col" in result
    assert len(result["numeric_col"]) == 3
    assert len(result["string_col"]) == 3
