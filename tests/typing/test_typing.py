"""The test suite for the typing module.

As the typing module only contains types, the tests only check imports.
"""
# ruff: noqa: F401

import pytest


def test_data_handlers_types() -> None:
    """Test the data handlers types."""
    try:
        from stimulus.typing import (
            DatasetHandler,
            DatasetLoader,
            DatasetManager,
            DatasetProcessor,
            EncodeManager,
            SplitManager,
            TransformManager,
        )
    except ImportError:
        pytest.fail("Failed to import Data Handlers types")


def test_learner_types() -> None:
    """Test the learner types."""
    try:
        from stimulus.typing import (
            PredictWrapper,
            RayTuneMetrics,
            RayTuneOptimizer,
            RayTuneResult,
            TuneModel,
            TuneParser,
            TuneWrapper,
        )
    except ImportError:
        pytest.fail("Failed to import Learner types")


def test_yaml_data_types() -> None:
    """Test the YAML data types."""
    try:
        from stimulus.typing import (
            Columns,
            ColumnsEncoder,
            YamlConfigDict,
            GlobalParams,
            YamlSchema,
            Split,
            SplitConfigDict,
            SplitTransformDict,
            Transform,
            TransformColumns,
            TransformColumnsTransformation,
        )
    except ImportError:
        pytest.fail("Failed to import YAML Data types")


def test_yaml_model_schema_types() -> None:
    """Test the YAML model schema types."""
    try:
        from stimulus.typing import (
            CustomTunableParameter,
            Data,
            Loss,
            Model,
            RayTuneModel,
            RunParams,
            Scheduler,
            TunableParameter,
            Tune,
            TuneParams,
            YamlRayConfigLoader,
        )
    except ImportError:
        pytest.fail("Failed to import YAML Model Schema types")


def test_type_aliases() -> None:
    """Test the type aliases."""
    try:
        from stimulus.typing import DataManager, Loader, RayTuneData, YamlData
    except ImportError:
        pytest.fail("Failed to import Type Aliases")
