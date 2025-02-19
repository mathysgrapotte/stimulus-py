"""Utility module for handling configuration files and their validation."""

from typing import Any, Optional, Union

import yaml
from pydantic import BaseModel, ValidationError, field_validator


class GlobalParams(BaseModel):
    """Model for global parameters in configuration."""

    seed: int


class ColumnsEncoder(BaseModel):
    """Model for column encoder configuration."""

    name: str
    params: Optional[dict[str, Union[str, list[Any]]]]  # Allow both string and list values


class Columns(BaseModel):
    """Model for column configuration."""

    column_name: str
    column_type: str
    data_type: str
    encoder: list[ColumnsEncoder]


class TransformColumnsTransformation(BaseModel):
    """Model for column transformation configuration."""

    name: str
    params: Optional[dict[str, Union[list[Any], float]]]  # Allow both list and float values


class TransformColumns(BaseModel):
    """Model for transform columns configuration."""

    column_name: str
    transformations: list[TransformColumnsTransformation]


class Transform(BaseModel):
    """Model for transform configuration."""

    transformation_name: str
    columns: list[TransformColumns]

    @field_validator("columns")
    @classmethod
    def validate_param_lists_across_columns(
        cls,
        columns: list[TransformColumns],
    ) -> list[TransformColumns]:
        """Validate that parameter lists across columns have consistent lengths.

        Args:
            columns: List of transform columns to validate

        Returns:
            The validated columns list
        """
        # Get all parameter list lengths across all columns and transformations
        all_list_lengths: set[int] = set()

        for column in columns:
            for transformation in column.transformations:
                if transformation.params and any(
                    isinstance(param_value, list) and len(param_value) > 0
                    for param_value in transformation.params.values()
                ):
                    all_list_lengths.update(
                        len(param_value)
                        for param_value in transformation.params.values()
                        if isinstance(param_value, list) and len(param_value) > 0
                    )

        # Skip validation if no lists found
        if not all_list_lengths:
            return columns

        # Check if all lists either have length 1, or all have the same length
        all_list_lengths.discard(1)  # Remove length 1 as it's always valid
        if len(all_list_lengths) > 1:  # Multiple different lengths found
            raise ValueError(
                "All parameter lists across columns must either contain one element or have the same length",
            )

        return columns


class Split(BaseModel):
    """Model for split configuration."""

    split_method: str
    params: dict[str, list[float]]  # More specific type for split parameters
    split_input_columns: list[str]


class ConfigDict(BaseModel):
    """Model for main configuration."""

    global_params: GlobalParams
    columns: list[Columns]
    transforms: list[Transform]
    split: list[Split]


# TODO: Rename this class to SplitConfigDict
class SplitConfigDict(BaseModel):
    """Model for sub-configuration generated from main config."""

    global_params: GlobalParams
    columns: list[Columns]
    transforms: list[Transform]
    split: Split


class SplitTransformDict(BaseModel):
    """Model for sub-configuration generated from main config."""

    global_params: GlobalParams
    columns: list[Columns]
    transforms: Transform
    split: Split


class Schema(BaseModel):
    """Model for validating schema."""

    conf: ConfigDict


class SplitSchema(BaseModel):
    """Model for validating a Split schema."""

    conf: SplitConfigDict


def extract_transform_parameters_at_index(
    transform: Transform,
    index: int = 0,
) -> Transform:
    """Get a transform with parameters at the specified index.

    Args:
        transform: The original transform containing parameter lists
        index: Index to extract parameters from (default 0)

    Returns:
        A new transform with single parameter values at the specified index
    """
    # Create a copy of the transform
    new_transform = Transform(**transform.model_dump())

    # Process each column and transformation
    for column in new_transform.columns:
        for transformation in column.transformations:
            if transformation.params:
                # Convert each parameter list to single value at index
                new_params = {}
                for param_name, param_value in transformation.params.items():
                    if isinstance(param_value, list):
                        new_params[param_name] = param_value[index]
                    else:
                        new_params[param_name] = param_value
                transformation.params = new_params

    return new_transform


def expand_transform_parameter_combinations(
    transform: Transform,
) -> list[Transform]:
    """Get all possible transforms by extracting parameters at each valid index.

    For a transform with parameter lists, creates multiple new transforms, each containing
    single parameter values from the corresponding indices of the parameter lists.

    Args:
        transform: The original transform containing parameter lists

    Returns:
        A list of transforms, each with single parameter values from sequential indices
    """
    # Find the length of parameter lists - we only need to check the first list we find
    # since all lists must have the same length (enforced by pydantic validator)
    max_length = 1
    for column in transform.columns:
        for transformation in column.transformations:
            if transformation.params:
                list_lengths = [len(v) for v in transformation.params.values() if isinstance(v, list) and len(v) > 1]
                if list_lengths:
                    max_length = list_lengths[0]  # All lists have same length due to validator
                    break

    # Generate a transform for each index
    transforms = []
    for i in range(max_length):
        transforms.append(extract_transform_parameters_at_index(transform, i))

    return transforms


def expand_transform_list_combinations(
    transform_list: list[Transform],
) -> list[Transform]:
    """Expands a list of transforms into all possible parameter combinations.

    Takes a list of transforms where each transform may contain parameter lists,
    and expands them into separate transforms with single parameter values.
    For example, if a transform has parameters [0.1, 0.2] and [1, 2], this will
    create two transforms: one with 0.1/1 and another with 0.2/2.

    Args:
        transform_list: A list of Transform objects containing parameter lists
            that need to be expanded into individual transforms.

    Returns:
        list[Transform]: A flattened list of transforms where each transform
            has single parameter values instead of parameter lists. The length of
            the returned list will be the sum of the number of parameter combinations
            for each input transform.
    """
    sub_transforms = []
    for transform in transform_list:
        sub_transforms.extend(expand_transform_parameter_combinations(transform))
    return sub_transforms


def generate_split_configs(config: ConfigDict) -> list[SplitConfigDict]:
    """Generates all possible split configuration from a config.

    Takes a configuration that may contain parameter lists and splits,
    and generates all unique splits into separate data configurations.

    For example, if the config has:
    - Two transforms with parameters [0.1, 0.2], [0.3, 0.4]
    - Two splits [0.7/0.3] and [0.8/0.2]
    This will generate 2 configs, 2 for each split.
        config_1:
            transform: [[0.1, 0.2], [0.3, 0.4]]
            split: [0.7, 0.3]

        config_2:
            transform: [[0.1, 0.2], [0.3, 0.4]]
            split: [0.8, 0.2]

    Args:
        config: The source configuration containing transforms with
            parameter lists and multiple splits.

    Returns:
        list[SubConfigDict]: A list of data configurations, where each
            config has a list of parameters and one split configuration. The
            length will be the product of the number of parameter combinations
            and the number of splits.
    """
    if isinstance(config, dict) and not isinstance(config, ConfigDict):
        raise TypeError("Input must be a ConfigDict object")

    sub_splits = config.split
    sub_configs = []
    for split in sub_splits:
        sub_configs.append(
            SplitConfigDict(
                global_params=config.global_params,
                columns=config.columns,
                transforms=config.transforms,
                split=split,
            ),
        )
    return sub_configs


def generate_split_transform_configs(
    config: SplitConfigDict,
) -> list[SplitTransformDict]:
    """Generates all the transform configuration for a given split

    Takes a configuration that may contain a transform or a list of transform,
    and generates all unique transform for a split into separate data configurations.

    For example, if the config has:
    - Two transforms with parameters [0.1, 0.2], [0.3, 0.4]
    - A split [0.7, 0.3]
    This will generate 2 configs, 2 for each split.
        transform_config_1:
            transform: [0.1, 0.2]
            split: [0.7, 0.3]

        transform_config_2:
            transform: [0.3, 0.4]
            split: [0.7, 0.3]

    Args:
        config: The source configuration containing each
            a split with transforms with parameters lists

    Returns:
        list[SubConfigTransformDict]: A list of data configurations, where each
            config has a list of parameters and one split configuration. The
            length will be the product of the number of parameter combinations
            and the number of splits.
    """
    if isinstance(config, dict) and not isinstance(config, SplitConfigDict):
        raise TypeError("Input must be a list of SubConfigDict")

    sub_transforms = expand_transform_list_combinations(config.transforms)
    split_transform_config: list[SplitTransformDict] = []
    for transform in sub_transforms:
        split_transform_config.append(
            SplitTransformDict(
                global_params=config.global_params,
                columns=config.columns,
                transforms=transform,
                split=config.split,
            ),
        )
    return split_transform_config


def dump_yaml_list_into_files(
    config_list: list[SplitConfigDict],
    directory_path: str,
    base_name: str,
) -> None:
    """Dumps a list of configurations into separate files with custom formatting."""
    # Create a new class attribute rather than assigning to the method
    # Remove this line since we'll add ignore_aliases to CustomDumper instead

    def represent_none(dumper: yaml.Dumper, _: Any) -> yaml.Node:
        """Custom representer to format None values as empty strings in YAML output."""
        return dumper.represent_scalar("tag:yaml.org,2002:null", "")

    def custom_representer(dumper: yaml.Dumper, data: Any) -> yaml.Node:
        """Custom representer to handle different types of lists with appropriate formatting."""
        if isinstance(data, list):
            if len(data) == 0:
                return dumper.represent_scalar("tag:yaml.org,2002:null", "")
            if isinstance(data[0], dict):
                return dumper.represent_sequence(
                    "tag:yaml.org,2002:seq",
                    data,
                    flow_style=False,
                )
            if isinstance(data[0], list):
                return dumper.represent_sequence(
                    "tag:yaml.org,2002:seq",
                    data,
                    flow_style=True,
                )
        return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=True)

    class CustomDumper(yaml.Dumper):
        """Custom YAML dumper that adds extra formatting controls."""

        def ignore_aliases(self, _data: Any) -> bool:
            """Ignore aliases in the YAML output."""
            return True

        def write_line_break(self, _data: Any = None) -> None:
            """Add extra newline after root-level elements."""
            super().write_line_break(_data)
            if len(self.indents) <= 1:  # At root level
                super().write_line_break(_data)

        def increase_indent(
            self,
            *,
            flow: bool = False,
            indentless: bool = False,
        ) -> None:  # type: ignore[override]
            """Ensure consistent indentation by preventing indentless sequences."""
            return super().increase_indent(
                flow=flow,
                indentless=indentless,
            )  # Force indentless to False for better formatting

    # Register the custom representers with our dumper
    yaml.add_representer(type(None), represent_none, Dumper=CustomDumper)
    yaml.add_representer(list, custom_representer, Dumper=CustomDumper)

    for i, config_dict in enumerate(config_list):
        dict_data = config_dict.model_dump(exclude_none=True)

        def fix_params(input_dict: dict[str, Any]) -> dict[str, Any]:
            """Recursively process dictionary to properly handle params fields."""
            if isinstance(input_dict, dict):
                processed_dict: dict[str, Any] = {}
                for key, value in input_dict.items():
                    if key == "encoder" and isinstance(value, list):
                        processed_dict[key] = []
                        for encoder in value:
                            processed_encoder = dict(encoder)
                            if "params" not in processed_encoder or not processed_encoder["params"]:
                                processed_encoder["params"] = {}
                            processed_dict[key].append(processed_encoder)
                    elif key == "transformations" and isinstance(value, list):
                        processed_dict[key] = []
                        for transformation in value:
                            processed_transformation = dict(transformation)
                            if "params" not in processed_transformation or not processed_transformation["params"]:
                                processed_transformation["params"] = {}
                            processed_dict[key].append(processed_transformation)
                    elif isinstance(value, dict):
                        processed_dict[key] = fix_params(value)
                    elif isinstance(value, list):
                        processed_dict[key] = [
                            fix_params(list_item) if isinstance(list_item, dict) else list_item for list_item in value
                        ]
                    else:
                        processed_dict[key] = value
                return processed_dict
            return input_dict

        dict_data = fix_params(dict_data)

        with open(f"{directory_path}/{base_name}_{i}.yaml", "w") as f:
            yaml.dump(
                dict_data,
                f,
                Dumper=CustomDumper,
                sort_keys=False,
                default_flow_style=False,
                indent=2,
                width=float("inf"),  # Prevent line wrapping
            )


def check_schema(config: ConfigDict) -> str:
    """Validate configuration fields have correct types.

    If the children field is specific to a parent, the children fields class is hosted in the parent fields class.
    If any field in not the right type, the function prints an error message explaining the problem and exits the python code.

    Args:
        config: The ConfigDict containing the fields of the yaml configuration file

    Returns:
        str: Empty string if validation succeeds

    Raises:
        ValueError: If validation fails
    """
    try:
        Schema(conf=config)
    except ValidationError as e:
        # Use logging instead of print for error handling
        raise ValueError("Wrong type on a field, see the pydantic report above") from e
    return ""
