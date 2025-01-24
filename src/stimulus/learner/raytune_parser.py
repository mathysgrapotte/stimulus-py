"""Ray Tune results parser for extracting and saving best model configurations and weights."""

import json
import os
from typing import Any, TypedDict, cast

import pandas as pd
import torch
from ray.tune import ExperimentAnalysis
from safetensors.torch import load_file as safe_load_file
from safetensors.torch import save_file as safe_save_file


class RayTuneResult(TypedDict):
    """TypedDict for storing Ray Tune optimization results."""

    config: dict[str, Any]
    checkpoint: str
    metrics_dataframe: pd.DataFrame


class RayTuneMetrics(TypedDict):
    """TypedDict for storing Ray Tune metrics results."""

    checkpoint: str
    metrics_dataframe: pd.DataFrame


class RayTuneOptimizer(TypedDict):
    """TypedDict for storing Ray Tune optimizer state."""

    checkpoint: str


class TuneParser:
    """Parser class for Ray Tune results to extract best configurations and model weights."""

    def __init__(self, results: ExperimentAnalysis) -> None:
        """`results` is the output of ray.tune."""
        self.results = results

    def get_best_config(self) -> dict[str, Any]:
        """Get the best config from the results."""
        best_result = cast(RayTuneResult, self.results.best_result)
        return best_result["config"]

    def save_best_config(self, output: str) -> None:
        """Save the best config to a file.

        TODO maybe only save the relevant config values.
        """
        config = self.get_best_config()
        config = self.fix_config_values(config)
        with open(output, "w") as f:
            json.dump(config, f, indent=4)

    def fix_config_values(self, config: dict[str, Any]) -> dict[str, Any]:
        """Correct config values.

        Args:
            config: Configuration dictionary to fix

        Returns:
            Fixed configuration dictionary
        """
        # fix the model and experiment values to avoid problems with serialization
        # TODO this is a quick fix to avoid the problem with serializing class objects. maybe there is a better way.
        config["model"] = config["model"].__name__
        config["experiment"] = config["experiment"].__class__.__name__
        if "tune" in config and "tune_params" in config["tune"]:
            del config["tune"]["tune_params"]["scheduler"]
        # delete miscellaneus keys, used only during debug mode for example
        del config["_debug"], config["tune_run_path"]

        return config

    def save_best_metrics_dataframe(self, output: str) -> None:
        """Save the dataframe with the metrics at each iteration of the best sample to a file."""
        best_result = cast(RayTuneMetrics, self.results.best_result)
        metrics_df = best_result["metrics_dataframe"]
        columns = [col for col in metrics_df.columns if "config" not in col]
        metrics_df = metrics_df[columns]
        metrics_df.to_csv(output, index=False)

    def get_best_model(self) -> dict[str, torch.Tensor]:
        """Get the best model weights from the results."""
        best_result = cast(RayTuneMetrics, self.results.best_result)
        checkpoint_dir = best_result["checkpoint"]
        checkpoint = os.path.join(checkpoint_dir, "model.safetensors")
        return safe_load_file(checkpoint)

    def save_best_model(self, output: str) -> None:
        """Save the best model weights to a file."""
        safe_save_file(self.get_best_model(), output)

    def get_best_optimizer(self) -> dict[str, Any]:
        """Get the best optimizer state from the results."""
        best_result = cast(RayTuneOptimizer, self.results.best_result)
        checkpoint_dir = best_result["checkpoint"]
        checkpoint = os.path.join(checkpoint_dir, "optimizer.pt")
        return torch.load(checkpoint)

    def save_best_optimizer(self, output: str) -> None:
        """Save the best optimizer state to a file."""
        torch.save(self.get_best_optimizer(), output)
