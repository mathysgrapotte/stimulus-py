"""Test the RayTuneLearner class."""

import os
import warnings

import pytest
import ray
import yaml

from stimulus.data.handlertorch import TorchDataset
from stimulus.data.loaders import EncoderLoader
from stimulus.learner.raytune_learner import TuneWrapper
from stimulus.utils.yaml_data import SplitTransformDict
from stimulus.utils.yaml_model_schema import Model, RayTuneModel, YamlRayConfigLoader
from tests.test_model import titanic_model


@pytest.fixture
def ray_config_loader() -> RayTuneModel:
    """Load the RayTuneModel configuration."""
    with open("tests/test_model/titanic_model_cpu.yaml") as file:
        model_config = yaml.safe_load(file)
    return YamlRayConfigLoader(Model(**model_config)).get_config()


@pytest.fixture
def encoder_loader() -> EncoderLoader:
    """Load the EncoderLoader configuration."""
    with open("tests/test_data/titanic/titanic_sub_config.yaml") as file:
        data_config = yaml.safe_load(file)
    encoder_loader = EncoderLoader()
    encoder_loader.initialize_column_encoders_from_config(
        SplitTransformDict(**data_config).columns,
    )
    return encoder_loader


@pytest.fixture
def titanic_dataset(encoder_loader: EncoderLoader) -> TorchDataset:
    """Create a TorchDataset instance for testing."""
    return TorchDataset(
        csv_path="tests/test_data/titanic/titanic_stimulus_split.csv",
        config_path="tests/test_data/titanic/titanic_sub_config.yaml",
        encoder_loader=encoder_loader,
        split=0,
    )


def test_tunewrapper_init(
    ray_config_loader: RayTuneModel,
    encoder_loader: EncoderLoader,
) -> None:
    """Test the initialization of the TuneWrapper class."""
    # Filter ResourceWarning during Ray shutdown
    warnings.filterwarnings("ignore", category=ResourceWarning)

    # Initialize Ray with minimal resources for testing
    ray.init(ignore_reinit_error=True)

    try:
        data_config: SplitTransformDict
        with open("tests/test_data/titanic/titanic_sub_config.yaml") as f:
            data_config = SplitTransformDict(**yaml.safe_load(f))

        tune_wrapper = TuneWrapper(
            model_config=ray_config_loader,
            model_class=titanic_model.ModelTitanic,
            data_path="tests/test_data/titanic/titanic_stimulus_split.csv",
            data_config=data_config,
            encoder_loader=encoder_loader,
            seed=42,
            ray_results_dir=os.path.abspath("tests/test_data/titanic/ray_results"),
            tune_run_name="test_run",
            debug=False,
            autoscaler=False,
        )

        assert isinstance(tune_wrapper, TuneWrapper)
    finally:
        # Force cleanup of Ray resources
        ray.shutdown()
        # Clear any temporary files
        if os.path.exists("tests/test_data/titanic/ray_results"):
            import shutil

            shutil.rmtree("tests/test_data/titanic/ray_results", ignore_errors=True)


def test_tune_wrapper_tune(
    ray_config_loader: RayTuneModel,
    encoder_loader: EncoderLoader,
) -> None:
    """Test the tune method of TuneWrapper class."""
    # Filter ResourceWarning during Ray shutdown
    warnings.filterwarnings("ignore", category=ResourceWarning)

    # Initialize Ray with minimal resources for testing
    ray.init(ignore_reinit_error=True)

    try:
        data_config: SplitTransformDict
        with open("tests/test_data/titanic/titanic_sub_config.yaml") as f:
            data_config = SplitTransformDict(**yaml.safe_load(f))

        tune_wrapper = TuneWrapper(
            model_config=ray_config_loader,
            model_class=titanic_model.ModelTitanic,
            data_path="tests/test_data/titanic/titanic_stimulus_split.csv",
            data_config=data_config,
            encoder_loader=encoder_loader,
            seed=42,
            ray_results_dir=os.path.abspath("tests/test_data/titanic/ray_results"),
            tune_run_name="test_run",
            debug=False,
            autoscaler=False,
        )

        tune_wrapper.tune()

    finally:
        # Force cleanup of Ray resources
        ray.shutdown()
        # Clear any temporary files
        if os.path.exists("tests/test_data/titanic/ray_results"):
            import shutil

            shutil.rmtree("tests/test_data/titanic/ray_results", ignore_errors=True)
