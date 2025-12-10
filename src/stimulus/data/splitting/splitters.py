"""This file contains the splitter classes for splitting data accordingly."""

from abc import ABC, abstractmethod
from typing import Any, Optional, Union

import numpy as np

# Constants
SPLIT_SIZE = 2  # Number of splits (train/test)


class AbstractSplitter(ABC):
    """Abstract class for splitters.

    A splitter splits the data into train and test sets.

    Methods:
        get_split_indexes: calculates split indices for the data
        distance: calculates the distance between two elements of the data
    """

    def __init__(self, seed: float = 42) -> None:
        """Initialize the splitter.

        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed

    @abstractmethod
    def get_split_indexes(self, data: dict) -> tuple[list, list]:
        """Splits the data. Always return indices mapping to the original list.

        This is an abstract method that should be implemented by the child class.

        Args:
            data (dict): the data to be split

        Returns:
            train_indices (list): the indices for train set
            test_indices (list): the indices for test set
        """
        raise NotImplementedError

    @abstractmethod
    def distance(self, data_one: Any, data_two: Any) -> float:
        """Calculates the distance between two elements of the data.

        This is an abstract method that should be implemented by the child class.

        Args:
            data_one (Any): the first data point
            data_two (Any): the second data point

        Returns:
            distance (float): the distance between the two data points
        """
        raise NotImplementedError


class RandomSplit(AbstractSplitter):
    """This splitter randomly splits the data."""

    def __init__(self, split: Optional[list] = None, seed: int = 42) -> None:
        """Initialize the random splitter.

        Args:
            split: List of proportions for train/val/test splits
            seed: Random seed for reproducibility
        """
        super().__init__()
        self.split = [0.7, 0.3] if split is None else split
        self.seed = seed
        if len(self.split) != SPLIT_SIZE:
            raise ValueError(
                "The split argument should be a list with length 2 that contains the proportions for [train, validation, test] splits.",
            )

    def get_split_indexes(
        self,
        data: dict,
    ) -> tuple[list, list]:
        """Splits the data indices into train and test sets.

        One can use these lists of indices to parse the data afterwards.

        Args:
            data (dict): Dictionary mapping column names to lists of data values.

        Returns:
            train (list): The indices for the training set.
            test (list): The indices for the test set.

        Raises:
            ValueError: If the split argument is not a list with length 3.
            ValueError: If the sum of the split proportions is not 1.
        """
        # Use round to avoid errors due to floating point imprecisions
        if round(sum(self.split), 3) < 1.0:
            raise ValueError(f"The sum of the split proportions should be 1. Instead, it is {sum(self.split)}.")

        if not data:
            raise ValueError("No data provided for splitting")
        # Get length from first column's data list
        length_of_data = len(next(iter(data.values())))

        # Generate a list of indices and shuffle it
        indices = np.arange(length_of_data)
        np.random.seed(self.seed)
        np.random.shuffle(indices)

        # Calculate the sizes of the train and test sets
        train_size = int(self.split[0] * length_of_data)
        test_size = int(self.split[1] * length_of_data)

        # Split the shuffled indices according to the calculated sizes
        train = indices[:train_size].tolist()
        test = indices[train_size : train_size + test_size].tolist()

        return train, test

    def distance(self, data_one: Any, data_two: Any) -> float:
        """Calculate distance between two data points.

        Args:
            data_one: First data point
            data_two: Second data point

        Returns:
            Distance between the points
        """
        raise NotImplementedError

class TargetGeneSplitter(AbstractSplitter):
    """Split the data into train and test sets based on target genes."""

    def __init__(
        self,
        target_gene_col: str = "target_gene",
        target_genes: Optional[list[str]] = None,
        split_ratio: Optional[Union[float, list[float]]] = None,
        seed: int = 42,
    ) -> None:
        """Initialize the target gene splitter.

        Args:
            target_gene_col: Name of the column containing target genes
            target_genes: List of specific target genes to include in the test set.
            split_ratio: Ratio of genes to include in the test set. Can be a float (test ratio)
                         or a list [train_ratio, test_ratio].
            seed: Random seed for reproducibility
        """
        super().__init__(seed)
        self.target_gene_col = target_gene_col
        self.target_genes = target_genes
        self.split_ratio = split_ratio

        if self.target_genes is None and self.split_ratio is None:
            raise ValueError("Either target_genes or split_ratio must be provided.")

    def get_split_indexes(self, data: dict) -> tuple[list, list]:
        """Splits the data indices based on target genes.

        Args:
            data (dict): Dictionary where keys are column names and values are lists of data.
                         Must contain the target_gene_col.

        Returns:
            val_indices (list): The indices for the training set.
            test_indices (list): The indices for the test set.
        """
        if self.target_gene_col not in data:
            raise ValueError(f"Column {self.target_gene_col} not found in data.")

        genes = np.array(data[self.target_gene_col])
        unique_genes = np.unique(genes)

        np.random.seed(self.seed)

        if self.target_genes is not None:
            # Case 1: Specific list of target genes for the test set
            test_genes_set = set(self.target_genes)
        else:
            # Case 2: Random ratio of genes
            # Determine test ratio
            if isinstance(self.split_ratio, list):
                if len(self.split_ratio) != 2:
                    raise ValueError("split_ratio list must have length 2 [train, test].")
                if abs(sum(self.split_ratio) - 1.0) > 1e-6:
                    raise ValueError(f"split_ratio must sum to 1. Got {sum(self.split_ratio)}")
                test_ratio = self.split_ratio[1]
            else:
                test_ratio = self.split_ratio

            # Randomly select genes
            n_test_genes = int(len(unique_genes) * test_ratio)
            # Ensure at least one gene if ratio > 0 and genes exist
            if n_test_genes == 0 and test_ratio > 0 and len(unique_genes) > 0:
                n_test_genes = 1
            
            shuffled_genes = unique_genes.copy()
            np.random.shuffle(shuffled_genes)
            test_genes_set = set(shuffled_genes[:n_test_genes])

        # Create masks
        # Using numpy for efficiency if genes is numpy array, else list comp
        # genes is already converted to numpy array above
        
        # We need to return indices
        # np.isin returns boolean mask
        is_test = np.isin(genes, list(test_genes_set))
        
        # Get indices
        all_indices = np.arange(len(genes))
        test_indices = all_indices[is_test].tolist()
        train_indices = all_indices[~is_test].tolist()

        return train_indices, test_indices

    def distance(self, data_one: Any, data_two: Any) -> float:
        """Calculate distance between two data points."""
        raise NotImplementedError
