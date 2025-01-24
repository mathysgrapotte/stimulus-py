"""This file contains the splitter classes for splitting data accordingly."""

from abc import ABC, abstractmethod
from typing import Any, Optional

import numpy as np

# Constants
SPLIT_SIZE = 3  # Number of splits (train/val/test)


class AbstractSplitter(ABC):
    """Abstract class for splitters.

    A splitter splits the data into train, validation, and test sets.

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
    def get_split_indexes(self, data: dict) -> tuple[list, list, list]:
        """Splits the data. Always return indices mapping to the original list.

        This is an abstract method that should be implemented by the child class.

        Args:
            data (pl.DataFrame): the data to be split

        Returns:
            split_indices (list): the indices for train, validation, and test sets
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
        self.split = [0.7, 0.2, 0.1] if split is None else split
        self.seed = seed
        if len(self.split) != SPLIT_SIZE:
            raise ValueError(
                "The split argument should be a list with length 3 that contains the proportions for [train, validation, test] splits.",
            )

    def get_split_indexes(
        self,
        data: dict,
    ) -> tuple[list, list, list]:
        """Splits the data indices into train, validation, and test sets.

        One can use these lists of indices to parse the data afterwards.

        Args:
            data (dict): Dictionary mapping column names to lists of data values.

        Returns:
            train (list): The indices for the training set.
            validation (list): The indices for the validation set.
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

        # Calculate the sizes of the train, validation, and test sets
        train_size = int(self.split[0] * length_of_data)
        validation_size = int(self.split[1] * length_of_data)

        # Split the shuffled indices according to the calculated sizes
        train = indices[:train_size].tolist()
        validation = indices[train_size : train_size + validation_size].tolist()
        test = indices[train_size + validation_size :].tolist()

        return train, validation, test

    def distance(self, data_one: Any, data_two: Any) -> float:
        """Calculate distance between two data points.

        Args:
            data_one: First data point
            data_two: Second data point

        Returns:
            Distance between the points
        """
        raise NotImplementedError
