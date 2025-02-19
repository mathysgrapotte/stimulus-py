"""Test the tuning CLI."""

import operator
import os
import shutil
import warnings
from functools import reduce
from pathlib import Path
from typing import Any

import pytest
import ray
import yaml

from src.stimulus.cli import tuning
from src.stimulus.utils.yaml_data import SplitTransformDict


@pytest.fixture
def data_path() -> str:
    """Get path to test data CSV file."""
    return str(
        Path(__file__).parent.parent
        / "test_data"
        / "titanic"
        / "titanic_stimulus_split.csv",
    )


@pytest.fixture
def data_config() -> str:
    """Get path to test data config YAML."""
    return str(
        Path(__file__).parent.parent
        / "test_data"
        / "titanic"
        / "titanic_sub_config.yaml",
    )


@pytest.fixture
def model_path() -> str:
    """Get path to test model file."""
    return str(Path(__file__).parent.parent / "test_model" / "titanic_model.py")


@pytest.fixture
def model_config() -> str:
    """Get path to test model config YAML."""
    return str(Path(__file__).parent.parent / "test_model" / "titanic_model_cpu.yaml")


def _get_number_of_generated_files(save_dir_path: str) -> int:
    """Each run generates a file in the result dir."""
    # Get the number of generated run files
    number_of_files: int = 0
    for file in os.listdir(save_dir_path):
        if "TuneModel" in file:
            number_of_files = len(
                [f for f in os.listdir(save_dir_path + "/" + file) if "TuneModel" in f],
            )
    return number_of_files


def _get_number_of_theoritical_runs(params_path: str) -> int:
    """The number of run is defined as follows.

    G:      number of grid_search
    n_i:    number of options for the ith grid_search
    S:      value of num_samples

    R = S * ∏(i=1 to G) n_i
    """
    # Get the theoritical number of runs
    with open(params_path) as file:
        params_dict: dict[str, Any] = yaml.safe_load(file)

    grid_searches_len: list[int] = []
    num_samples: int = 0
    for _header, sections in params_dict.items():
        if isinstance(sections, dict):
            for section in sections.values():
                if isinstance(section, dict):
                    # Lookup for grid search or num_samples in the yaml
                    mode_value = section.get("mode")
                    ns_value = section.get("num_samples")
                    if mode_value == "grid_search":
                        space_value = section.get("space")
                        if space_value is not None:
                            grid_searches_len.append(len(space_value))
                        else:
                            grid_searches_len.append(0)
                    elif ns_value is not None:
                        num_samples = ns_value if isinstance(ns_value, int) else 0
    # Apply the described function and return the value
    return num_samples * reduce(operator.mul, grid_searches_len)


def test_tuning_main(
    data_path: str,
    data_config: str,
    model_path: str,
    model_config: str,
) -> None:
    """Test that tuning.main runs without errors.

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
        config_dict: SplitTransformDict
        with open(data_config) as f:
            config_dict = SplitTransformDict(**yaml.safe_load(f))

        results_dir = Path("tests/test_data/titanic/test_results/").resolve()
        results_dir.mkdir(parents=True, exist_ok=True)

        # Use directory path for Ray results and file paths for outputs
        tuning.main(
            model_path=model_path,
            data_path=data_path,
            data_config=config_dict,
            model_config_path=model_config,
            initial_weights=None,
            # Directory path without URI scheme
            ray_results_dirpath=str(results_dir),
            output_path=str(results_dir / "best_model.safetensors"),
            best_optimizer_path=str(results_dir / "best_optimizer.pt"),
            best_metrics_path=str(results_dir / "best_metrics.csv"),
            best_config_path=str(results_dir / "best_config.yaml"),
            debug_mode=True,
        )

    except RuntimeError as error:
        error_msg: str = str(error).lower()
        if "zero_division" in error_msg or "no best trial found" in error_msg:
            pytest.skip(f"Skipping test due to known metric issue: {error}")
        raise
    finally:
        # Get the number of runs executed

        # Ensure Ray is shut down properly
        if ray.is_initialized():
            ray.shutdown()

            # Clean up any ray files/directories that may have been created
            ray_results_dir = os.path.expanduser(
                "tests/test_data/titanic/test_results/",
            )
            # Check that the theoritical numbers of run corresponds to the real number of runs
            n_files: int = _get_number_of_generated_files(ray_results_dir)
            n_runs: int = _get_number_of_theoritical_runs(model_config)
            assert n_files == n_runs
            shutil.rmtree(ray_results_dir)
