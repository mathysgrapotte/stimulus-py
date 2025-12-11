"""Tests for the split CLI command."""

import os
import tempfile
from pathlib import Path

import pytest
from click.testing import CliRunner

from stimulus.cli.main import cli
from stimulus.cli.split import split


@pytest.fixture
def parquet_path() -> str:
    """Get path to test Parquet file."""
    return str(
        Path(__file__).parent.parent / "test_data" / "titanic" / "titanic_stimulus.parquet",
    )


@pytest.fixture
def parquet_path_with_split() -> str:
    """Get path to test Parquet file with split column."""
    return str(
        Path(__file__).parent.parent / "test_data" / "titanic" / "titanic_stimulus_split.parquet",
    )


@pytest.fixture
def yaml_path() -> str:
    """Get path to test config YAML file."""
    return str(
        Path(__file__).parent.parent / "test_data" / "titanic" / "titanic_unique_split.yaml",
    )


# @pytest.mark.skip(reason="Break github action runners")
def test_split_main(
    parquet_path: str,
    yaml_path: str,
) -> None:
    """Test that split runs without errors.

    Args:
        parquet_path: Path to test Parquet data.
        yaml_path: Path to test config YAML.
    """
    # Verify required files exist
    assert os.path.exists(parquet_path), f"Parquet file not found at {parquet_path}"
    assert os.path.exists(yaml_path), f"YAML config not found at {yaml_path}"

    # Create temporary output directory
    with tempfile.TemporaryDirectory() as tmp_dir_name:
        output_path = tmp_dir_name

        # Run main function - should complete without errors
        split(
            data_path=parquet_path,
            config_yaml=yaml_path,
            out_path=output_path,
        )

        # Verify output directory exists and contains expected files
        assert os.path.exists(output_path), "Output directory was not created"
        assert os.path.exists(
            os.path.join(output_path, "train"),
        ), "train split directory not found"
        assert os.path.exists(
            os.path.join(output_path, "val"),
        ), "val split directory not found"


# @pytest.mark.skip(reason="Break github action runners")
def test_split_error_on_existing_split(
    parquet_path_with_split: str,
    yaml_path: str,
) -> None:
    """Test split raises error on file that already has split column.

    Args:
        parquet_path_with_split: Path to test Parquet data with split column.
        yaml_path: Path to test config YAML.
    """
    with tempfile.TemporaryDirectory() as tmp_dir_name:
        output_path = tmp_dir_name

        # 1. Split and save to output_path
        split(
            data_path=parquet_path_with_split,
            config_yaml=yaml_path,
            out_path=output_path,
        )

        # 2. Try to split again using the output_path as input
        with pytest.raises(ValueError, match="Validation split already exists"):
            split(
                data_path=output_path,  # Now pointing to the directory with splits
                config_yaml=yaml_path,
                out_path=output_path,
            )


# @pytest.mark.skip(reason="Break github action runners")
def test_cli_invocation(
    parquet_path: str,
    yaml_path: str,
) -> None:
    """Test the CLI invocation of split command.

    Args:
        parquet_path: Path to test Parquet data.
        yaml_path: Path to test config YAML.
    """
    runner = CliRunner()
    with runner.isolated_filesystem():
        output_path = "output_dataset"
        result = runner.invoke(
            cli,
            [
                "split",
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
