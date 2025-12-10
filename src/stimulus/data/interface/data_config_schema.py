"""Module for defining the data config schema."""

from typing import Any, Optional, Union

from pydantic import BaseModel, field_validator


class GlobalParams(BaseModel):
    """Model for global parameters in YAML configuration."""

    seed: int


class ColumnsEncoder(BaseModel):
    """Model for column encoder configuration."""

    name: str
    params: dict[str, Union[int, str, list[Any]]]

    @field_validator("params")
    def validate_dtype(cls, params: dict) -> dict:  # noqa: N805
        """Validate that the 'dtype' key is present in the encoder parameters."""
        if "dtype" not in params:
            raise ValueError("params must contain 'dtype' key")
        return params


class Columns(BaseModel):
    """Model for column configuration."""

    column_name: str
    column_type: str
    encoder: list[ColumnsEncoder]


class TransformColumnsTransformation(BaseModel):
    """Model for column transformation configuration."""

    name: str
    params: Optional[dict[str, Union[list[Any], float, str, int, bool]]]  # Allow list, numeric, and string values


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
    """Model for main YAML configuration."""

    global_params: GlobalParams
    columns: list[Columns]
    transforms: list[Transform]
    split: list[Split]


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
    """Model for validating YAML schema."""

    config: ConfigDict


class EncodingConfigDict(BaseModel):
    """Model for encoding-only configuration."""

    global_params: GlobalParams
    columns: list[Columns]


class IndividualSplitConfigDict(BaseModel):
    """Model for individual split configuration."""

    global_params: GlobalParams
    split: Split


class IndividualTransformConfigDict(BaseModel):
    """Model for individual transform configuration."""

    global_params: GlobalParams
    transforms: Transform


class DatasetConfig(BaseModel):
    """Dataset configuration."""

    type: str = "HuggingFaceDataset"
    params: dict[str, Any] = {}
