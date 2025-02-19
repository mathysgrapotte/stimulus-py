"""Typing for Stimulus Python API.

This module contains all Stimulus types which will be used for variable typing
and likely not instantiated, as well as aliases for other types to use for typing purposes.

The aliases from this module should be used for typing purposes only.
"""
# ruff: noqa: F401

from typing import TypeAlias

# these imports mostly alias everything
from stimulus.data.data_handlers import (
    DatasetHandler,
    DatasetLoader,
    DatasetManager,
    DatasetProcessor,
    EncodeManager,
    SplitManager,
    TransformManager,
)
from stimulus.data.encoding.encoders import AbstractEncoder as Encoder
from stimulus.data.handlertorch import TorchDataset
from stimulus.data.loaders import EncoderLoader, SplitLoader, TransformLoader
from stimulus.data.splitters.splitters import AbstractSplitter as Splitter
from stimulus.data.transform.data_transformation_generators import (
    AbstractDataTransformer as Transform,
)
from stimulus.learner.predict import PredictWrapper
from stimulus.learner.raytune_learner import CheckpointDict, TuneModel, TuneWrapper
from stimulus.learner.raytune_parser import (
    RayTuneMetrics,
    RayTuneOptimizer,
    RayTuneResult,
    TuneParser,
)
from stimulus.utils.performance import Performance
from stimulus.utils.yaml_data import (
    Columns,
    ColumnsEncoder,
    ConfigDict,
    GlobalParams,
    YamlSchema,
    Split,
    SplitConfigDict,
    SplitTransformDict,
    Transform,
    TransformColumns,
    TransformColumnsTransformation,
)
from stimulus.utils.yaml_model_schema import (
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

# data/data_handlers.py

DataManager: TypeAlias = (
    DatasetManager | EncodeManager | SplitManager | TransformManager
)

# data/experiments.py

Loader: TypeAlias = DatasetLoader | EncoderLoader | TransformLoader | SplitLoader

# learner/raytune_parser.py

RayTuneData: TypeAlias = RayTuneMetrics | RayTuneOptimizer | RayTuneResult

# utils/yaml_data.py

YamlData: TypeAlias = (
    Columns
    | ColumnsEncoder
    | ConfigDict
    | GlobalParams
    | YamlSchema
    | Split
    | SplitConfigDict
    | Transform
    | TransformColumns
    | TransformColumnsTransformation
)
