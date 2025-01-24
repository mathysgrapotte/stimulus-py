"""Module for handling YAML configuration files and converting them to Ray Tune format."""

import random
from collections.abc import Callable
from copy import deepcopy
from typing import Any

import yaml
from ray import tune


class YamlRayConfigLoader:
    """Load and convert YAML configurations to Ray Tune format.

    This class handles loading YAML configuration files and converting them into
    formats compatible with Ray Tune's hyperparameter search spaces.
    """

    def __init__(self, config_path: str) -> None:
        """Initialize the config loader with a YAML file path.

        Args:
            config_path: Path to the YAML configuration file
        """
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
            self.config = self.convert_config_to_ray(self.config)

    def raytune_space_selector(self, mode: Callable, space: list) -> dict[str, Any]:
        """Convert space parameters to Ray Tune format based on the mode.

        Args:
            mode: Ray Tune search space function (e.g., tune.choice, tune.uniform)
            space: List of parameters defining the search space

        Returns:
            Configured Ray Tune search space

        Raises:
            NotImplementedError: If the mode is not supported
        """
        if mode.__name__ == "choice":
            return mode(space)

        if mode.__name__ in ["uniform", "loguniform", "quniform", "qloguniform", "qnormal", "randint"]:
            return mode(*tuple(space))

        raise NotImplementedError(f"Mode {mode.__name__} not implemented yet")

    def raytune_sample_from(self, mode: Callable, param: dict) -> dict[str, Any]:
        """Apply tune.sample_from to a given custom sampling function.

        Args:
            mode: Ray Tune sampling function
            param: Dictionary containing sampling parameters

        Returns:
            Configured sampling function

        Raises:
            NotImplementedError: If the sampling function is not supported
        """
        if param["function"] == "sampint":
            return mode(lambda _: self.sampint(param["sample_space"], param["n_space"]))

        raise NotImplementedError(f"Function {param['function']} not implemented yet")

    def convert_raytune(self, param: dict) -> dict[str, Any]:
        """Convert parameter configuration to Ray Tune format.

        Args:
            param: Parameter configuration dictionary

        Returns:
            Ray Tune compatible parameter configuration

        Raises:
            AttributeError: If the mode is not recognized in Ray Tune
        """
        try:
            mode = getattr(tune, param["mode"])
        except AttributeError as err:
            raise AttributeError(
                f"Mode {param['mode']} not recognized, check the ray.tune documentation at https://docs.ray.io/en/master/tune/api_docs/suggestion.html",
            ) from err

        if param["mode"] != "sample_from":
            return self.raytune_space_selector(mode, param["space"])
        return self.raytune_sample_from(mode, param)

    def convert_config_to_ray(self, config: dict) -> dict:
        """Convert YAML configuration to Ray Tune format.

        Converts parameters in model_params, loss_params, optimizer_params, and data_params
        to Ray Tune search spaces when a mode is specified.

        Args:
            config: Raw configuration dictionary from YAML

        Returns:
            Ray Tune compatible configuration dictionary
        """
        new_config = deepcopy(config)
        for key in ["model_params", "loss_params", "optimizer_params", "data_params"]:
            for sub_key in config[key]:
                if "mode" in config[key][sub_key]:
                    new_config[key][sub_key] = self.convert_raytune(config[key][sub_key])

        return new_config

    def get_config_instance(self) -> dict:
        """Generate a configuration instance with sampled values.

        Returns:
            Configuration dictionary with concrete sampled values
        """
        config_instance = deepcopy(self.config)
        for key in ["model_params", "loss_params", "optimizer_params", "data_params"]:
            config_instance[key] = {}
            for sub_key in self.config[key]:
                config_instance[key][sub_key] = self.config[key][sub_key].sample()

        return config_instance

    def get_config(self) -> dict:
        """Return the current configuration.

        Returns:
            Current configuration dictionary
        """
        return self.config

    @staticmethod
    def sampint(sample_space: list, n_space: list) -> list[int]:
        """Return a list of n random samples from the sample_space.

        This function is useful for sampling different numbers of layers,
        each with different numbers of neurons.

        Args:
            sample_space: List [min, max] defining range of values to sample from
            n_space: List [min, max] defining range for number of samples

        Returns:
            List of randomly sampled integers

        Note:
            Uses Python's random module which is not cryptographically secure.
            This is acceptable for hyperparameter sampling but should not be
            used for security-critical purposes (S311 fails when linting).
        """
        sample_space_list = list(range(sample_space[0], sample_space[1] + 1))
        n_space_list = list(range(n_space[0], n_space[1] + 1))
        n = random.choice(n_space_list)  # noqa: S311
        return random.sample(sample_space_list, n)
