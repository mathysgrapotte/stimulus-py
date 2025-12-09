"""Module for validating YAML configuration files."""

import inspect
import logging
from typing import Any, Callable, Literal, Optional

import optuna
import pydantic
import torch

from stimulus.data.interface import data_config_schema

logger = logging.getLogger(__name__)


class TunableParameter(pydantic.BaseModel):
    """Tunable parameter."""

    params: dict[str, Any]
    mode: str

    @pydantic.model_validator(mode="after")
    def validate_mode(self) -> "TunableParameter":
        """Validate that mode is supported by Optuna or custom methods."""
        if self.mode not in [
            "categorical",
            "discrete_uniform",
            "float",
            "int",
            "loguniform",
            "uniform",
            "variable_list",
        ]:
            raise NotImplementedError(
                f"Mode {self.mode} not available for Optuna, please use one of the following: categorical, discrete_uniform, float, int, loguniform, uniform, variable_list",
            )

        return self

    @pydantic.model_validator(mode="after")
    def validate_params(self) -> "TunableParameter":
        """Validate that the params are supported by Optuna."""
        trial_methods: dict[str, Callable] = {
            "categorical": optuna.trial.Trial.suggest_categorical,
            "discrete_uniform": optuna.trial.Trial.suggest_discrete_uniform,
            "float": optuna.trial.Trial.suggest_float,
            "int": optuna.trial.Trial.suggest_int,
            "loguniform": optuna.trial.Trial.suggest_loguniform,
            "uniform": optuna.trial.Trial.suggest_uniform,
        }
        if self.mode in trial_methods:
            sig = inspect.signature(trial_methods[self.mode])
            required_params = {
                param.name
                for param in sig.parameters.values()
                if param.default is inspect.Parameter.empty
                and param.kind not in (inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL)
                and param.name not in ("self", "trial", "name")
            }
            missing_params = required_params - set(self.params.keys())
            if missing_params:
                raise ValueError(f"Missing required params for mode '{self.mode}': {missing_params}")
        return self


class VariableList(pydantic.BaseModel):
    """Variable list."""

    length: TunableParameter
    values: TunableParameter
    mode: Literal["variable_list"]

    def validate_length(self) -> "VariableList":
        """Validate that length is supported by Optuna."""
        if self.length.mode not in ["int"]:
            raise ValueError(
                f"length mode has to be set to int, got {self.length.mode}",
            )
        return self


class Pruner(pydantic.BaseModel):
    """Pruner parameters."""

    name: str
    params: dict[str, Any]

    @pydantic.model_validator(mode="after")
    def validate_pruner(self) -> "Pruner":
        """Validate that pruner is supported by Optuna."""
        # Get available pruners, filtering out internal ones that start with _
        available_pruners = [
            attr for attr in dir(optuna.pruners) if not attr.startswith("_") and attr != "TYPE_CHECKING"
        ]
        logger.info(f"Available pruners in Optuna: {available_pruners}")

        # Check if the pruner exists with correct case
        if not hasattr(optuna.pruners, self.name):
            # Try to find a case-insensitive match
            case_matches = [attr for attr in available_pruners if attr.lower() == self.name.lower()]
            if case_matches:
                logger.info(f"Found matching pruner with different case: {case_matches[0]}")
                self.name = case_matches[0]  # Use the correct case
            else:
                raise ValueError(
                    f"Pruner '{self.name}' not available in Optuna. Available pruners: {available_pruners}",
                )
        return self


class Objective(pydantic.BaseModel):
    """Objective parameters."""

    metric: str
    direction: str

    @pydantic.model_validator(mode="after")
    def validate_direction(self) -> "Objective":
        """Validate that direction is supported by Optuna."""
        if self.direction not in ["minimize", "maximize"]:
            raise NotImplementedError(
                f"Direction {self.direction} not available for Optuna, please use one of the following: minimize, maximize",
            )
        return self


class Sampler(pydantic.BaseModel):
    """Sampler parameters."""

    name: str
    params: Optional[dict[str, Any]] = None

    @pydantic.model_validator(mode="after")
    def validate_sampler(self) -> "Sampler":
        """Validate that sampler is supported by Optuna."""
        # Get available samplers, filtering out internal ones that start with _
        available_samplers = [
            attr for attr in dir(optuna.samplers) if not attr.startswith("_") and attr != "TYPE_CHECKING"
        ]
        logger.info(f"Available samplers in Optuna: {available_samplers}")

        if not hasattr(optuna.samplers, self.name):
            # Try to find a case-insensitive match
            case_matches = [attr for attr in available_samplers if attr.lower() == self.name.lower()]
            if case_matches:
                logger.info(f"Found matching sampler with different case: {case_matches[0]}")
                self.name = case_matches[0]  # Use the correct case
            else:
                raise ValueError(
                    f"Sampler '{self.name}' not available in Optuna. Available samplers: {available_samplers}",
                )
        return self


class Loss(pydantic.BaseModel):
    """Loss parameters."""

    loss_fn: TunableParameter


class Model(pydantic.BaseModel):
    """Model configuration."""

    network_params: dict[str, TunableParameter | VariableList]
    optimizer_params: dict[str, TunableParameter]
    loss_params: dict[str, TunableParameter]
    data_params: dict[str, TunableParameter]
    dataset: data_config_schema.DatasetConfig = pydantic.Field(default_factory=data_config_schema.DatasetConfig)
    pruner: Pruner
    sampler: Sampler
    objective: Objective
    seed: int = 42
    max_samples: int = 1000
    compute_objective_every_n_samples: int = 50
    n_trials: int = 10
    device: Optional[str] = None

    # Add a model validator to debug the input data
    @pydantic.model_validator(mode="before")
    @classmethod
    def validate_input(cls, data: dict[str, Any]) -> dict[str, Any]:
        """Print input data for debugging."""
        logger.info(f"Input data for Model: {data}")
        return data

    @pydantic.model_validator(mode="after")
    def validate_data_params(self) -> "Model":
        """Validate that data_params contains batch_size."""
        if "batch_size" not in self.data_params:
            raise ValueError("data_params must contain batch_size")
        return self

    @pydantic.model_validator(mode="after")
    def validate_device(self) -> "Model":
        """Validate that device is a valid PyTorch device if specified."""
        if self.device is not None:
            try:
                torch.device(self.device)
            except RuntimeError as e:
                raise ValueError(f"Invalid device '{self.device}': {e}") from e
        return self
