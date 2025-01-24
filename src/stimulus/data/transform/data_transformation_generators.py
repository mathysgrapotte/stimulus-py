"""This file contains noise generators classes for generating various types of noise."""

import multiprocessing as mp
from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class AbstractDataTransformer(ABC):
    """Abstract class for data transformers.

    Data transformers implement in_place or augmentation transformations.
    Whether it is in_place or augmentation is specified in the "add_row" attribute (should be True or False and set in children classes constructor)

    Child classes should override the `transform` and `transform_all` methods.

    `transform_all` should always return a list

    Both methods should take an optional `seed` argument set to `None` by default to be compliant with stimulus' core principle of reproducibility.
    Seed should be initialized through `np.random.seed(seed)` in the method implementation.

    Attributes:
        add_row (bool): whether the transformer adds rows to the data

    Methods:
        transform: transforms a data point
        transform_all: transforms a list of data points
    """

    def __init__(self) -> None:
        """Initialize the data transformer."""
        self.add_row: bool = False
        self.seed: int = 42

    @abstractmethod
    def transform(self, data: Any) -> Any:
        """Transforms a single data point.

        This is an abstract method that should be implemented by the child class.

        Args:
            data (Any): the data to be transformed

        Returns:
            transformed_data (Any): the transformed data
        """
        #  np.random.seed(self.seed)
        raise NotImplementedError

    @abstractmethod
    def transform_all(self, data: list) -> list:
        """Transforms a list of data points.

        This is an abstract method that should be implemented by the child class.

        Args:
            data (list): the data to be transformed

        Returns:
            transformed_data (list): the transformed data
        """
        #  np.random.seed(self.seed)
        raise NotImplementedError


class AbstractNoiseGenerator(AbstractDataTransformer):
    """Abstract class for noise generators.

    All noise function should have the seed in it. This is because the multiprocessing of them could unset the seed.
    """

    def __init__(self) -> None:
        """Initialize the noise generator."""
        super().__init__()
        self.add_row = False


class AbstractAugmentationGenerator(AbstractDataTransformer):
    """Abstract class for augmentation generators.

    All augmentation function should have the seed in it. This is because the multiprocessing of them could unset the seed.
    """

    def __init__(self) -> None:
        """Initialize the augmentation generator."""
        super().__init__()
        self.add_row = True


class UniformTextMasker(AbstractNoiseGenerator):
    """Mask characters in text.

    This noise generators replace characters with a masking character with a given probability.

    Methods:
        transform: adds character masking to a single data point
        transform_all: adds character masking to a list of data points
    """

    def __init__(self, probability: float = 0.1, mask: str = "*", seed: int = 42) -> None:
        """Initialize the text masker.

        Args:
            probability: Probability of masking each character
            mask: Character to use for masking
            seed: Random seed for reproducibility
        """
        super().__init__()
        self.probability = probability
        self.mask = mask
        self.seed = seed

    def transform(self, data: str) -> str:
        """Adds character masking to the data.

        Args:
            data (str): the data to be transformed

        Returns:
            transformed_data (str): the transformed data point
        """
        np.random.seed(self.seed)
        return "".join([c if np.random.rand() > self.probability else self.mask for c in data])

    def transform_all(self, data: list) -> list:
        """Adds character masking to multiple data points using multiprocessing.

        Args:
            data (list): the data to be transformed


        Returns:
            transformed_data (list): the transformed data points
        """
        with mp.Pool(mp.cpu_count()) as pool:
            function_specific_input = list(data)
            return pool.starmap(self.transform, function_specific_input)


class GaussianNoise(AbstractNoiseGenerator):
    """Add Gaussian noise to data.

    This noise generator adds Gaussian noise to float values.

    Methods:
        transform: adds noise to a single data point
        transform_all: adds noise to a list of data points
    """

    def __init__(self, mean: float = 0, std: float = 1, seed: int = 42) -> None:
        """Initialize the Gaussian noise generator.

        Args:
            mean: Mean of the Gaussian noise
            std: Standard deviation of the Gaussian noise
            seed: Random seed for reproducibility
        """
        super().__init__()
        self.mean = mean
        self.std = std
        self.seed = seed

    def transform(self, data: float) -> float:
        """Adds Gaussian noise to a single point of data.

        Args:
            data (float): the data to be transformed

        Returns:
            transformed_data (float): the transformed data point
        """
        np.random.seed(self.seed)
        return data + np.random.normal(self.mean, self.std)

    def transform_all(self, data: list) -> list:
        """Adds Gaussian noise to a list of data points.

        Args:
            data (list): the data to be transformed

        Returns:
            transformed_data (list): the transformed data points
        """
        np.random.seed(self.seed)
        return list(np.array(data) + np.random.normal(self.mean, self.std, len(data)))


class ReverseComplement(AbstractAugmentationGenerator):
    """Reverse complement biological sequences.

    This augmentation strategy reverse complements the input nucleotide sequences.

    Methods:
        transform: reverse complements a single data point
        transform_all: reverse complements a list of data points

    Raises:
        ValueError: if the type of the sequence is not DNA or RNA
    """

    def __init__(self, sequence_type: str = "DNA") -> None:
        """Initialize the reverse complement generator.

        Args:
            sequence_type: Type of sequence ('DNA' or 'RNA')
        """
        super().__init__()
        if sequence_type not in ("DNA", "RNA"):
            raise ValueError(
                "Currently only DNA and RNA sequences are supported. Update the class ReverseComplement to support other types.",
            )
        if sequence_type == "DNA":
            self.complement_mapping = str.maketrans("ATCG", "TAGC")
        elif sequence_type == "RNA":
            self.complement_mapping = str.maketrans("AUCG", "UAGC")

    def transform(self, data: str) -> str:
        """Returns the reverse complement of a list of string data using the complement_mapping.

        Args:
            data (str): the sequence to be transformed

        Returns:
            transformed_data (str): the reverse complement of the sequence
        """
        return data.translate(self.complement_mapping)[::-1]

    def transform_all(self, data: list) -> list:
        """Reverse complement multiple data points using multiprocessing.

        Args:
            data (list): the sequences to be transformed

        Returns:
            transformed_data (list): the reverse complement of the sequences
        """
        with mp.Pool(mp.cpu_count()) as pool:
            function_specific_input = list(data)
            return pool.map(self.transform, function_specific_input)


class GaussianChunk(AbstractAugmentationGenerator):
    """Subset data around a random midpoint.

    This augmentation strategy chunks the input sequences, for which the middle positions are obtained through a gaussian distribution.

    In concrete, it changes the middle position (ie. peak summit) to another position. This position is chosen based on a gaussian distribution, so the region close to the middle point are more likely to be chosen than the rest.
    Then a chunk with size `chunk_size` around the new middle point is returned.
    This process will be repeated for each sequence with `transform_all`.

    Methods:
        transform: chunk a single list
        transform_all: chunks multiple lists
    """

    def __init__(self, chunk_size: int, seed: int = 42, std: float = 1) -> None:
        """Initialize the Gaussian chunk generator.

        Args:
            chunk_size: Size of chunks to extract
            seed: Random seed for reproducibility
            std: Standard deviation for the Gaussian distribution
        """
        super().__init__()
        self.chunk_size = chunk_size
        self.seed = seed
        self.std = std

    def transform(self, data: str) -> str:
        """Chunks a sequence of size chunk_size from the middle position +/- a value obtained through a gaussian distribution.

        Args:
            data (str): the sequence to be transformed

        Returns:
            transformed_data (str): the chunk of the sequence

        Raises:
            AssertionError: if the input data is shorter than the chunk size
        """
        np.random.seed(self.seed)

        # make sure that the data is longer than chunk_size otherwise raise an error
        if len(data) <= self.chunk_size:
            raise ValueError("The input data is shorter than the chunk size")

        # Get the middle position of the input sequence
        middle_position = len(data) // 2

        # Change the middle position by a value obtained through a gaussian distribution
        new_middle_position = int(middle_position + np.random.normal(0, self.std))

        # Get the start and end position of the chunk
        start_position = new_middle_position - self.chunk_size // 2
        end_position = new_middle_position + self.chunk_size // 2

        # if the start position is negative, set it to 0
        start_position = max(start_position, 0)

        # Get the chunk of size chunk_size from the start position if the end position is smaller than the length of the data
        if end_position < len(data):
            return data[start_position : start_position + self.chunk_size]
        # Otherwise return the chunk of the sequence from the end of the sequence of size chunk_size
        return data[-self.chunk_size :]

    def transform_all(self, data: list) -> list:
        """Adds chunks to multiple lists using multiprocessing.

        Args:
            data (list): the sequences to be transformed

        Returns:
            transformed_data (list): the transformed sequences
        """
        with mp.Pool(mp.cpu_count()) as pool:
            function_specific_input = list(data)
            return pool.starmap(self.transform, function_specific_input)
