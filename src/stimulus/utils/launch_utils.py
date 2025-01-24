"""Utility functions for launching and configuring experiments and ray tuning."""

import importlib.util
import math
import os
from typing import Union

import stimulus.data.experiments as exp


def import_class_from_file(file_path: str) -> type:
    """Import and return the Model class from a specified Python file.

    Args:
        file_path (str): Path to the Python file containing the Model class.

    Returns:
        type: The Model class found in the file.

    Raises:
        ImportError: If no class starting with 'Model' is found in the file.
    """
    # Extract directory path and file name
    directory, file_name = os.path.split(file_path)
    module_name = os.path.splitext(file_name)[0]  # Remove extension to get module name

    # Create a module from the file path
    # In summary, these three lines of code are responsible for creating a module specification based on a file location, creating a module object from that specification, and then executing the module's code to populate the module object with the definitions from the Python file.
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None:
        raise ImportError(f"Could not create module spec for {file_path}")
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise ImportError(f"Module spec has no loader for {file_path}")
    spec.loader.exec_module(module)

    # Find the class dynamically
    for name in dir(module):
        model_class = getattr(module, name)
        if isinstance(model_class, type) and name.startswith("Model"):
            return model_class

    # Class not found
    raise ImportError("No class starting with 'Model' found in the file.")


def get_experiment(experiment_name: str) -> object:
    """Get an experiment instance by name.

    Args:
        experiment_name (str): Name of the experiment class to instantiate.

    Returns:
        object: An instance of the requested experiment class.
    """
    return getattr(exp, experiment_name)()


def memory_split_for_ray_init(memory_str: Union[str, None]) -> tuple[float, float]:
    """Process the input memory value into the right unit and allocates 30% for overhead and 70% for tuning.

    Useful in case ray detects them wrongly. Memory is split in two for ray: for store_object memory
    and the other actual memory for tuning. The following function takes the total possible
    usable/allocated memory as a string parameter and returns in bytes the values for store_memory
    (30% as default in ray) and memory (70%).

    Args:
        memory_str (Union[str, None]): Memory string in format like "8G", "16GB", etc.

    Returns:
        tuple[float, float]: A tuple containing (store_memory, memory) in bytes.
    """
    if memory_str is None:
        return 0.0, 0.0

    units = {"B": 1, "K": 2**10, "M": 2**20, "G": 2**30, "T": 2**40, "P": 2**50}

    # Extract the numerical part and the unit
    value_str = ""
    unit = ""

    for char in memory_str:
        if char.isdigit() or char == ".":
            value_str += char
        elif char.isalpha():
            unit += char.upper()

    value = float(value_str)

    # Normalize the unit (to handle cases like Gi, GB, Mi, etc.)
    if unit.endswith(("I", "i", "B", "b")):
        unit = unit[:-1]

    if unit not in units:
        raise ValueError(f"Unknown unit: {unit}")

    bytes_value = value * units[unit]

    # Calculate 30% and 70%
    thirty_percent = math.floor(bytes_value * 0.30)
    seventy_percent = math.floor(bytes_value * 0.70)

    return float(thirty_percent), float(seventy_percent)
