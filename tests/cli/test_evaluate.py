# mypy: disable-error-code="unused-ignore"
"""Test the evaluate CLI."""

import csv
import logging
import os
import shutil
import tempfile
from pathlib import Path

import pytest
import yaml

from stimulus.cli import evaluate, tuning

logger = logging.getLogger(__name__)


@pytest.fixture
def data_path() -> str:
    """Get path to test data."""
    return str(
        Path(__file__).parent.parent / "test_data" / "titanic_performant" / "titanic_encoded_hf",
    )


@pytest.fixture
def model_path() -> str:
    """Get path to test model file."""
    return str(Path(__file__).parent.parent / "test_model" / "titanic_perf_model.py")


@pytest.fixture
def model_config() -> str:
    """Get path to test model config YAML."""
    return str(Path(__file__).parent.parent / "test_model" / "titanic_perf_model.yaml")


def create_test_transform_config(temp_dir: str) -> str:
    """Create a test transform config YAML file."""
    config = {
        "global_params": {"seed": 42},
        "transforms": {
            "transformation_name": "test_transform",
            "columns": [
                {
                    "column_name": "X",
                    "transformations": [
                        {"name": "StandardScaler", "params": {"with_mean": True, "with_std": True}},
                    ],
                },
            ],
        },
    }
    config_path = os.path.join(temp_dir, "transform_config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    return config_path


def test_evaluate_with_tune_outputs_and_transform_config(
    data_path: str,
    model_path: str,
    model_config: str,
) -> None:
    """Test that evaluate runs with outputs from tune step and includes transform params."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Verify input files exist
        assert os.path.exists(data_path), f"Data file not found at {data_path}"
        assert os.path.exists(model_path), f"Model file not found at {model_path}"
        assert os.path.exists(model_config), f"Model config not found at {model_config}"

        # Setup tune output paths
        best_model_path = os.path.join(temp_dir, "best_model.safetensors")
        best_optimizer_path = os.path.join(temp_dir, "best_optimizer.pt")
        best_config_path = os.path.join(temp_dir, "best_config.json")

        # Create test transform config
        transform_config_path = create_test_transform_config(temp_dir)

        try:
            # First run tune to get best model and config
            tuning.tune(
                data_path=data_path,
                model_path=model_path,
                model_config_path=model_config,
                optuna_results_dirpath=temp_dir,
                best_model_path=best_model_path,
                best_optimizer_path=best_optimizer_path,
                best_config_path=best_config_path,
            )

            # Verify tune outputs exist
            assert os.path.exists(best_model_path), "Best model file was not created by tune"
            assert os.path.exists(best_config_path), "Best config file was not created by tune"

            # Now run evaluate with transform config
            output_csv = os.path.join(temp_dir, "metrics.csv")

            evaluate.evaluate(
                data_path=data_path,
                model_path=model_path,
                model_config_path=best_config_path,
                weight_path=best_model_path,
                output=output_csv,
                batch_size=32,
                transform_config=transform_config_path,
            )

            # Verify CSV was created
            assert os.path.exists(output_csv), "Metrics CSV was not created"

            # Verify CSV contents include transform params
            with open(output_csv, newline="") as csvfile:
                reader = csv.DictReader(csvfile)
                rows = list(reader)
                assert len(rows) == 1, "Expected exactly one row of metrics"

                row = rows[0]
                # Check transform params are present (flattened)
                assert "transforms.transformation_name" in row, "Missing transformation_name"
                assert row["transforms.transformation_name"] == "test_transform"
                assert "global_params.seed" in row, "Missing seed"
                assert row["global_params.seed"] == "42"

                # Check metrics are present (with metric_ prefix)
                assert "metric_loss" in row, "Missing loss metric"
                assert "metric_accuracy" in row, "Missing accuracy metric"

                # Verify metrics are valid numbers
                loss = float(row["metric_loss"])
                accuracy = float(row["metric_accuracy"])
                assert loss >= 0, "Loss should be non-negative"
                assert 0 <= accuracy <= 1, "Accuracy should be between 0 and 1"

        finally:
            # Clean up
            if os.path.exists("runs"):
                shutil.rmtree("runs", ignore_errors=True)


def test_evaluate_without_transform_config(
    data_path: str,
    model_path: str,
    model_config: str,
) -> None:
    """Test that evaluate works without transform_config (only metrics in CSV)."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Setup tune output paths
        best_model_path = os.path.join(temp_dir, "best_model.safetensors")
        best_optimizer_path = os.path.join(temp_dir, "best_optimizer.pt")
        best_config_path = os.path.join(temp_dir, "best_config.json")

        try:
            # First run tune
            tuning.tune(
                data_path=data_path,
                model_path=model_path,
                model_config_path=model_config,
                optuna_results_dirpath=temp_dir,
                best_model_path=best_model_path,
                best_optimizer_path=best_optimizer_path,
                best_config_path=best_config_path,
            )

            # Run evaluate without transform_config
            output_csv = os.path.join(temp_dir, "metrics_no_transform.csv")

            evaluate.evaluate(
                data_path=data_path,
                model_path=model_path,
                model_config_path=best_config_path,
                weight_path=best_model_path,
                output=output_csv,
                batch_size=32,
                transform_config=None,
            )

            # Verify CSV was created with only metrics
            with open(output_csv, newline="") as csvfile:
                reader = csv.DictReader(csvfile)
                rows = list(reader)
                assert len(rows) == 1
                row = rows[0]
                # Should only have metric columns
                assert "metric_loss" in row
                assert "metric_accuracy" in row
                # Should NOT have transform params
                assert "transforms.transformation_name" not in row

        finally:
            if os.path.exists("runs"):
                shutil.rmtree("runs", ignore_errors=True)


def test_flatten_dict() -> None:
    """Test that flatten_dict correctly flattens nested structures."""
    nested = {
        "global_params": {"seed": 42},
        "transforms": {
            "name": "test",
            "columns": [
                {"column_name": "X", "type": "numeric"},
                {"column_name": "Y", "type": "categorical"},
            ],
        },
    }

    flat = evaluate.flatten_dict(nested)

    assert flat["global_params.seed"] == 42
    assert flat["transforms.name"] == "test"
    assert flat["transforms.columns.0.column_name"] == "X"
    assert flat["transforms.columns.0.type"] == "numeric"
    assert flat["transforms.columns.1.column_name"] == "Y"
    assert flat["transforms.columns.1.type"] == "categorical"


def test_write_metrics_csv_with_transform_params() -> None:
    """Test that write_metrics_csv produces correct format with transform params."""
    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = os.path.join(temp_dir, "test.csv")

        metrics = {"loss": 0.123, "accuracy": 0.95}
        transform_params = {
            "transforms.transformation_name": "my_transform",
            "global_params.seed": 42,
            "transforms.columns.0.n_top_genes": 2000,
        }
        evaluate.write_metrics_csv(output_path, metrics, transform_params)

        with open(output_path, newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            assert len(rows) == 1
            row = rows[0]
            # Transform params come first
            assert row["transforms.transformation_name"] == "my_transform"
            assert row["global_params.seed"] == "42"
            assert row["transforms.columns.0.n_top_genes"] == "2000"
            # Metrics have prefix
            assert float(row["metric_loss"]) == pytest.approx(0.123)
            assert float(row["metric_accuracy"]) == pytest.approx(0.95)


def test_csv_concatenation_with_transform_params() -> None:
    """Test that multiple CSVs with transform params can be concatenated correctly."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create two metric files with different transforms
        csv1_path = os.path.join(temp_dir, "metrics1.csv")
        csv2_path = os.path.join(temp_dir, "metrics2.csv")

        transform1 = {
            "transforms.transformation_name": "transform_1",
            "transforms.columns.0.n_top_genes": 1000,
        }
        transform2 = {
            "transforms.transformation_name": "transform_2",
            "transforms.columns.0.n_top_genes": 2000,
        }

        evaluate.write_metrics_csv(csv1_path, {"loss": 0.1, "accuracy": 0.9}, transform1)
        evaluate.write_metrics_csv(csv2_path, {"loss": 0.2, "accuracy": 0.8}, transform2)

        # Verify both have same columns (important for concatenation)
        with open(csv1_path, newline="") as f1, open(csv2_path, newline="") as f2:
            reader1 = csv.DictReader(f1)
            reader2 = csv.DictReader(f2)
            assert reader1.fieldnames == reader2.fieldnames, "Column headers should match for concatenation"
