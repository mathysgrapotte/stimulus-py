"""Test the check_model CLI."""

import os
import warnings
from pathlib import Path

import pytest
import ray

from stimulus.cli import check_model


@pytest.fixture
def data_path() -> str:
    """Get path to test data CSV file."""
    return str(
        Path(__file__).parent.parent / "test_data" / "titanic" / "titanic_stimulus_split.csv",
    )


@pytest.fixture
def data_config() -> str:
    """Get path to test data config YAML."""
    return str(
        Path(__file__).parent.parent / "test_data" / "titanic" / "titanic_sub_config.yaml",
    )


@pytest.fixture
def model_path() -> str:
    """Get path to test model file."""
    return str(Path(__file__).parent.parent / "test_model" / "titanic_model.py")


@pytest.fixture
def model_config() -> str:
    """Get path to test model config YAML."""
    return str(Path(__file__).parent.parent / "test_model" / "titanic_model_cpu.yaml")


def test_check_model_main(
    data_path: str,
    data_config: str,
    model_path: str,
    model_config: str,
) -> None:
    """Test that check_model.main runs without errors.

    Args:
        data_path: Path to test CSV data
        data_config: Path to data config YAML
        model_path: Path to model implementation
        model_config: Path to model config YAML
    """
    # Filter ResourceWarning during Ray shutdown
    warnings.filterwarnings("ignore", category=ResourceWarning)

    # Initialize Ray with minimal resources for testing
    ray.init(ignore_reinit_error=True)
    # Verify all required files exist
    assert os.path.exists(data_path), f"Data file not found at {data_path}"
    assert os.path.exists(data_config), f"Data config not found at {data_config}"
    assert os.path.exists(model_path), f"Model file not found at {model_path}"
    assert os.path.exists(model_config), f"Model config not found at {model_config}"

    try:
        # Run main function - should complete without errors
        check_model.main(
            model_path=model_path,
            data_path=data_path,
            data_config_path=data_config,
            model_config_path=model_config,
            initial_weights=None,
            num_samples=1,
            ray_results_dirpath=None,
        )
    finally:
        # Ensure Ray is shut down properly
        if ray.is_initialized():
            ray.shutdown()

            # Clean up any ray files/directories that may have been created
            ray_results_dir = os.path.expanduser("~/ray_results")
            if os.path.exists(ray_results_dir):
                import shutil

                shutil.rmtree(ray_results_dir)
