"""This module provides classes for handling CSV data files in the STIMULUS format.

The module contains three main classes:
- DatasetHandler: Base class for loading and managing CSV data
- DatasetProcessor: Class for preprocessing data with transformations and splits
- DatasetLoader: Class for loading processed data for model training

The data format consists of:
1. A CSV file containing the raw data
2. A YAML configuration file that defines:
   - Column names and their roles (input/label/meta)
   - Data types and encoders for each column
   - Transformations to apply (noise, augmentation, etc.)
   - Split configuration for train/val/test sets

The data handling pipeline consists of:
1. Loading raw CSV data according to the YAML config
2. Applying configured transformations
3. Splitting into train/val/test sets based on config
4. Encoding data for model training using specified encoders

See titanic.yaml in tests/test_data/titanic/ for an example configuration file format.
"""

from typing import Any, Optional, Union

import numpy as np
import polars as pl
import torch
import yaml

from stimulus.data import experiments
from stimulus.utils import yaml_data


class DatasetManager:
    """Class for managing the dataset.

    This class handles loading and organizing dataset configuration from YAML files.
    It manages column categorization into input, label and meta types based on the config.

    Attributes:
        config (dict): The loaded configuration dictionary from YAML
        column_categories (dict): Dictionary mapping column types to lists of column names

    Methods:
        _load_config(config_path: str) -> dict: Loads the config from a YAML file.
        categorize_columns_by_type() -> dict: Organizes the columns into input, label, meta based on the config.
    """

    def __init__(
        self,
        config_path: str,
    ) -> None:
        """Initialize the DatasetManager."""
        self.config = self._load_config(config_path)
        self.column_categories = self.categorize_columns_by_type()

    def categorize_columns_by_type(self) -> dict:
        """Organizes columns from config into input, label, and meta categories.

        Reads the column definitions from the config and sorts them into categories
        based on their column_type field.

        Returns:
            dict: Dictionary containing lists of column names for each category:
                {
                    "input": ["col1", "col2"],  # Input columns
                    "label": ["target"],        # Label/output columns
                    "meta": ["id"]     # Metadata columns
                }

        Example:
            >>> manager = DatasetManager("config.yaml")
            >>> categories = manager.categorize_columns_by_type()
            >>> print(categories)
            {
                'input': ['hello', 'bonjour'],
                'label': ['ciao'],
                'meta': ["id"]
            }
        """
        input_columns = []
        label_columns = []
        meta_columns = []
        for column in self.config.columns:
            if column.column_type == "input":
                input_columns.append(column.column_name)
            elif column.column_type == "label":
                label_columns.append(column.column_name)
            elif column.column_type == "meta":
                meta_columns.append(column.column_name)

        return {"input": input_columns, "label": label_columns, "meta": meta_columns}

    def _load_config(self, config_path: str) -> yaml_data.YamlConfigDict:
        """Loads and parses a YAML configuration file.

        Args:
            config_path (str): Path to the YAML config file

        Returns:
            dict: Parsed configuration dictionary

        Example:
            >>> manager = DatasetManager()
            >>> config = manager._load_config("config.yaml")
            >>> print(config["columns"][0]["column_name"])
            'hello'
        """
        with open(config_path) as file:
            return yaml_data.YamlSubConfigDict(**yaml.safe_load(file))

    def get_split_columns(self) -> list[str]:
        """Get the columns that are used for splitting."""
        return self.config.split.split_input_columns

    def get_transform_logic(self) -> dict:
        """Get the transformation logic.

        Returns a dictionary in the following structure :
        {
            "transformation_name": str,
            "transformations": list[tuple[str, str, dict]]
        }
        """
        transformation_logic = {
            "transformation_name": self.config.transforms.transformation_name,
            "transformations": [],
        }
        for column in self.config.transforms.columns:
            for transformation in column.transformations:
                transformation_logic["transformations"].append(
                    (column.column_name, transformation.name, transformation.params),
                )
        return transformation_logic


class EncodeManager:
    """Manages the encoding of data columns using configured encoders.

    This class handles encoding of data columns based on the encoders specified in the
    configuration. It uses an EncoderLoader to get the appropriate encoder for each column
    and applies the encoding.

    Attributes:
        encoder_loader (experiments.EncoderLoader): Loader that provides encoders based on config.

    Example:
        >>> encoder_loader = EncoderLoader(config)
        >>> encode_manager = EncodeManager(encoder_loader)
        >>> data = ["ACGT", "TGCA", "GCTA"]
        >>> encoded = encode_manager.encode_column("dna_seq", data)
        >>> print(encoded.shape)
        torch.Size([3, 4, 4])  # 3 sequences, length 4, one-hot encoded
    """

    def __init__(
        self,
        encoder_loader: experiments.EncoderLoader,
    ) -> None:
        """Initialize the EncodeManager.

        Args:
            encoder_loader: Loader that provides encoders based on configuration.
        """
        self.encoder_loader = encoder_loader

    def encode_column(self, column_name: str, column_data: list) -> torch.Tensor:
        """Encodes a column of data using the configured encoder.

        Gets the appropriate encoder for the column from the encoder_loader and uses it
        to encode all the data in the column.

        Args:
            column_name: Name of the column to encode.
            column_data: List of data values from the column to encode.

        Returns:
            Encoded data as a torch.Tensor. The exact shape depends on the encoder used.

        Example:
            >>> data = ["ACGT", "TGCA"]
            >>> encoded = encode_manager.encode_column("dna_seq", data)
            >>> print(encoded.shape)
            torch.Size([2, 4, 4])  # 2 sequences, length 4, one-hot encoded
        """
        encode_all_function = self.encoder_loader.get_function_encode_all(column_name)
        return encode_all_function(column_data)

    def encode_columns(self, column_data: dict) -> dict:
        """Encodes multiple columns of data using the configured encoders.

        Gets the appropriate encoder for each column from the encoder_loader and encodes
        all data values in those columns.

        Args:
            column_data: Dict mapping column names to lists of data values to encode.

        Returns:
            Dict mapping column names to their encoded tensors. The exact shape of each
            tensor depends on the encoder used for that column.

        Example:
            >>> data = {"dna_seq": ["ACGT", "TGCA"], "labels": ["1", "2"]}
            >>> encoded = encode_manager.encode_columns(data)
            >>> print(encoded["dna_seq"].shape)
            torch.Size([2, 4, 4])  # 2 sequences, length 4, one-hot encoded
        """
        return {col: self.encode_column(col, values) for col, values in column_data.items()}

    def encode_dataframe(self, dataframe: pl.DataFrame) -> dict[str, torch.Tensor]:
        """Encode the dataframe using the encoders."""
        return {col: self.encode_column(col, dataframe[col].to_list()) for col in dataframe.columns}


class TransformManager:
    """Class for managing the transformations."""

    def __init__(
        self,
        transform_loader: experiments.TransformLoader,
    ) -> None:
        """Initialize the TransformManager."""
        self.transform_loader = transform_loader

    def transform_column(self, column_name: str, transform_name: str, column_data: list) -> tuple[list, bool]:
        """Transform a column of data using the specified transformation.

        Args:
            column_name (str): The name of the column to transform.
            transform_name (str): The name of the transformation to use.
            column_data (list): The data to transform.

        Returns:
            list: The transformed data.
            bool: Whether the transformation added new rows to the data.
        """
        transformer = self.transform_loader.__getattribute__(column_name)[transform_name]
        return transformer.transform_all(column_data), transformer.add_row


class SplitManager:
    """Class for managing the splitting."""

    def __init__(
        self,
        split_loader: experiments.SplitLoader,
    ) -> None:
        """Initialize the SplitManager."""
        self.split_loader = split_loader

    def get_split_indices(self, data: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get the indices for train, validation, and test splits."""
        return self.split_loader.get_function_split()(data)


class DatasetHandler:
    """Main class for handling dataset loading, encoding, transformation and splitting.

    This class coordinates the interaction between different managers to process
    CSV datasets according to the provided configuration.

    Attributes:
        encoder_manager (EncodeManager): Manager for handling data encoding operations.
        transform_manager (TransformManager): Manager for handling data transformations.
        split_manager (SplitManager): Manager for handling dataset splitting.
        dataset_manager (DatasetManager): Manager for organizing dataset columns and config.
    """

    def __init__(
        self,
        config_path: str,
        csv_path: str,
    ) -> None:
        """Initialize the DatasetHandler with required config.

        Args:
            config_path (str): Path to the dataset configuration file.
            csv_path (str): Path to the CSV data file.
        """
        self.dataset_manager = DatasetManager(config_path)
        self.columns = self.read_csv_header(csv_path)
        self.data = self.load_csv(csv_path)

    def read_csv_header(self, csv_path: str) -> list:
        """Get the column names from the header of the CSV file.

        Args:
            csv_path (str): Path to the CSV file to read headers from.

        Returns:
            list: List of column names from the CSV header.
        """
        with open(csv_path) as f:
            return f.readline().strip().split(",")

    def select_columns(self, columns: list) -> dict:
        """Select specific columns from the DataFrame and return as a dictionary.

        Args:
            columns (list): List of column names to select.

        Returns:
            dict: A dictionary where keys are column names and values are lists containing the column data.

        Example:
            >>> handler = DatasetHandler(...)
            >>> data_dict = handler.select_columns(["col1", "col2"])
            >>> # Returns {'col1': [1, 2, 3], 'col2': [4, 5, 6]}
        """
        df = self.data.select(columns)
        return {col: df[col].to_list() for col in columns}

    def load_csv(self, csv_path: str) -> pl.DataFrame:
        """Load the CSV file into a polars DataFrame.

        Args:
            csv_path (str): Path to the CSV file to load.

        Returns:
            pl.DataFrame: Polars DataFrame containing the loaded CSV data.
        """
        return pl.read_csv(csv_path)

    def save(self, path: str) -> None:
        """Saves the data to a csv file."""
        self.data.write_csv(path)


class DatasetProcessor(DatasetHandler):
    """Class for loading dataset, applying transformations and splitting."""

    def __init__(self, config_path: str, csv_path: str) -> None:
        """Initialize the DatasetProcessor."""
        super().__init__(config_path, csv_path)

    def add_split(self, split_manager: SplitManager, *, force: bool = False) -> None:
        """Add a column specifying the train, validation, test splits of the data.

        An error exception is raised if the split column is already present in the csv file. This behaviour can be overriden by setting force=True.

        Args:
            split_manager (SplitManager): Manager for handling dataset splitting
            force (bool): If True, the split column present in the csv file will be overwritten.
        """
        if ("split" in self.columns) and (not force):
            raise ValueError(
                "The category split is already present in the csv file. If you want to still use this function, set force=True",
            )
        # get relevant split columns from the dataset_manager
        split_columns = self.dataset_manager.get_split_columns()
        split_input_data = self.select_columns(split_columns)

        # get the split indices
        train, validation, test = split_manager.get_split_indices(split_input_data)

        # add the split column to the data
        split_column = np.full(len(self.data), -1).astype(int)
        split_column[train] = 0
        split_column[validation] = 1
        split_column[test] = 2
        self.data = self.data.with_columns(pl.Series("split", split_column))

        if "split" not in self.columns:
            self.columns.append("split")

    def apply_transformation_group(self, transform_manager: TransformManager) -> None:
        """Apply the transformation group to the data."""
        for column_name, transform_name, _params in self.dataset_manager.get_transform_logic()["transformations"]:
            transformed_data, add_row = transform_manager.transform_column(
                column_name,
                transform_name,
                self.data[column_name],
            )
            if add_row:
                new_rows = self.data.with_columns(pl.Series(column_name, transformed_data))
                self.data = pl.vstack(self.data, new_rows)
            else:
                self.data = self.data.with_columns(pl.Series(column_name, transformed_data))

    def shuffle_labels(self, seed: Optional[float] = None) -> None:
        """Shuffles the labels in the data."""
        # set the np seed
        np.random.seed(seed)

        label_keys = self.dataset_manager.column_categories["label"]
        for key in label_keys:
            self.data = self.data.with_columns(pl.Series(key, np.random.permutation(list(self.data[key]))))


class DatasetLoader(DatasetHandler):
    """Class for loading dataset and passing it to the deep learning model."""

    def __init__(
        self,
        config_path: str,
        csv_path: str,
        encoder_loader: experiments.EncoderLoader,
        split: Union[int, None] = None,
    ) -> None:
        """Initialize the DatasetLoader."""
        super().__init__(config_path, csv_path)
        self.encoder_manager = EncodeManager(encoder_loader)
        self.data = self.load_csv_per_split(csv_path, split) if split is not None else self.load_csv(csv_path)

    def get_all_items(self) -> tuple[dict, dict, dict]:
        """Get the full dataset as three separate dictionaries for inputs, labels and metadata.

        Returns:
            tuple[dict, dict, dict]: Three dictionaries containing:
                - Input dictionary mapping input column names to encoded input data
                - Label dictionary mapping label column names to encoded label data
                - Meta dictionary mapping meta column names to meta data

        Example:
            >>> handler = DatasetHandler(...)
            >>> input_dict, label_dict, meta_dict = handler.get_dataset()
            >>> print(input_dict.keys())
            dict_keys(['age', 'fare'])
            >>> print(label_dict.keys())
            dict_keys(['survived'])
            >>> print(meta_dict.keys())
            dict_keys(['passenger_id'])
        """
        input_columns, label_columns, meta_columns = (
            self.dataset_manager.column_categories["input"],
            self.dataset_manager.column_categories["label"],
            self.dataset_manager.column_categories["meta"],
        )
        input_data = self.encoder_manager.encode_dataframe(self.data[input_columns])
        label_data = self.encoder_manager.encode_dataframe(self.data[label_columns])
        meta_data = {key: self.data[key].to_list() for key in meta_columns}
        return input_data, label_data, meta_data

    def get_all_items_and_length(self) -> tuple[tuple[dict, dict, dict], int]:
        """Get the full dataset as three separate dictionaries for inputs, labels and metadata, and the length of the data."""
        return self.get_all_items(), len(self.data)

    def load_csv_per_split(self, csv_path: str, split: int) -> pl.DataFrame:
        """Load the part of csv file that has the specified split value.

        Split is a number that for 0 is train, 1 is validation, 2 is test.
        This is accessed through the column with category `split`. Example column name could be `split:split:int`.

        NOTE that the aim of having this function is that depending on the training, validation and test scenarios,
        we are gonna load only the relevant data for it.
        """
        if "split" not in self.columns:
            raise ValueError("The category split is not present in the csv file")
        if split not in [0, 1, 2]:
            raise ValueError(f"The split value should be 0, 1 or 2. The specified split value is {split}")
        return pl.scan_csv(csv_path).filter(pl.col("split") == split).collect()

    def __len__(self) -> int:
        """Return the length of the first list in input, assumes that all are the same length."""
        return len(self.data)

    def __getitem__(self, idx: Any) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, list]]:
        """Get the data at a given index, and encodes the input and label, leaving meta as it is.

        Args:
            idx: The index of the data to be returned, it can be a single index, a list of indexes or a slice
        """
        # Handle different index types
        if isinstance(idx, slice):
            data_at_index = self.data.slice(idx.start or 0, idx.stop or len(self.data))
        elif isinstance(idx, int):
            data_at_index = self.data.slice(idx, idx + 1)
        else:
            data_at_index = self.data[idx]

        input_columns, label_columns, meta_columns = (
            self.dataset_manager.column_categories["input"],
            self.dataset_manager.column_categories["label"],
            self.dataset_manager.column_categories["meta"],
        )
        input_data = self.encoder_manager.encode_dataframe(data_at_index[input_columns])
        label_data = self.encoder_manager.encode_dataframe(data_at_index[label_columns])
        meta_data = {key: data_at_index[key].to_list() for key in meta_columns}
        return input_data, label_data, meta_data
