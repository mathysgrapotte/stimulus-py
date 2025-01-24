"""Utility functions for general purpose operations like seed setting and tensor manipulation."""

raytune_rework
import random
from typing import Union

import numpy as np
import torch


def ensure_at_least_1d(tensor: torch.Tensor) -> torch.Tensor:
    """Function to make sure tensors given are not zero dimensional. if they are add one dimension."""
    if tensor.dim() == 0:
        tensor = tensor.unsqueeze(0)
    return tensor


def set_general_seeds(seed_value: Union[int, None]) -> None:
    """Set all relevant random seeds to a given value.

    Especially useful in case of ray.tune. Ray does not have a "generic" seed as far as ray 2.23.
    """
    # Set python seed
    random.seed(seed_value)

    # set numpy seed
    np.random.seed(seed_value)

    # set torch seed, diffrently from the two above torch can nopt take Noneas input value so it will not be called in that case.
    if seed_value is not None:
        torch.manual_seed(seed_value)


def check_path(path: str) -> str | None:
    if not os.path.exists(path):
        raise ValueError(f"Data path does not exist. Given path: {path}")
    else:
        return os.path.abspath(path)


def check_not_none(val: any, name: str) -> any:
    if val is None:
        raise ValueError(f"{name} is None. Please set a value for {name}")
    else:
        return val
