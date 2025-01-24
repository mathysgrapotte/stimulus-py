"""This file contains encoders classes for encoding various types of data."""

import logging
import multiprocessing as mp
from abc import ABC, abstractmethod
from typing import Any, Union

import numpy as np
import torch
from sklearn import preprocessing

logger = logging.getLogger(__name__)


class AbstractEncoder(ABC):
    """Abstract class for encoders.

    Encoders are classes that encode the raw data into torch.tensors.
    Different encoders provide different encoding methods.
    Different encoders may take different types of data as input.

    Methods:
        encode: encodes a single data point
        encode_all: encodes a list of data points into a torch.tensor
        encode_multiprocess: encodes a list of data points using multiprocessing
        decode: decodes a single data point
    """

    @abstractmethod
    def encode(self, data: Any) -> Any:
        """Encode a single data point.

        This is an abstract method, child classes should overwrite it.

        Args:
            data (Any): a single data point

        Returns:
            encoded_data_point (Any): the encoded data point
        """
        raise NotImplementedError

    @abstractmethod
    def encode_all(self, data: list[Any]) -> torch.Tensor:
        """Encode a list of data points.

        This is an abstract method, child classes should overwrite it.

        Args:
            data (list[Any]): a list of data points

        Returns:
            encoded_data (torch.Tensor): encoded data points
        """
        raise NotImplementedError

    @abstractmethod
    def decode(self, data: Any) -> Any:
        """Decode a single data point.

        This is an abstract method, child classes should overwrite it.

        Args:
            data (Any): a single encoded data point

        Returns:
            decoded_data_point (Any): the decoded data point
        """
        raise NotImplementedError

    def encode_multiprocess(self, data: list[Any]) -> list[Any]:
        """Helper function for encoding the data using multiprocessing.

        Args:
            data (list[Any]): a list of data points

        Returns:
            encoded_data (list[Any]): encoded data points
        """
        with mp.Pool(mp.cpu_count()) as pool:
            return pool.map(self.encode, data)


class TextOneHotEncoder(AbstractEncoder):
    """One hot encoder for text data.

    NOTE encodes based on the given alphabet
    If a character c is not in the alphabet, c will be represented by a vector of zeros.

    Attributes:
        alphabet (str): the alphabet to one hot encode the data with.
        convert_lowercase (bool): whether to convert the sequence and alphabet to lowercase. Default is False.
        padding (bool): whether to pad the sequences with zeros. Default is False.
        encoder (OneHotEncoder): preprocessing.OneHotEncoder object initialized with self.alphabet

    Methods:
        encode: encodes a single data point
        encode_all: encodes a list of data points into a numpy array
        encode_multiprocess: encodes a list of data points using multiprocessing
        decode: decodes a single data point
        _sequence_to_array: transforms a sequence into a numpy array
    """

    def __init__(self, alphabet: str = "acgt", *, convert_lowercase: bool = False, padding: bool = False) -> None:
        """Initialize the TextOneHotEncoder class.

        Args:
            alphabet (str): the alphabet to one hot encode the data with.

        Raises:
            TypeError: If the input alphabet is not a string.
        """
        if not isinstance(alphabet, str):
            error_msg = f"Expected a string input for alphabet, got {type(alphabet).__name__}"
            logger.error(error_msg)
            raise TypeError(error_msg)

        if convert_lowercase:
            alphabet = alphabet.lower()

        self.alphabet = alphabet
        self.convert_lowercase = convert_lowercase
        self.padding = padding

        self.encoder = preprocessing.OneHotEncoder(
            categories=[list(alphabet)],
            handle_unknown="ignore",
        )  # handle_unknown='ignore' unsures that a vector of zeros is returned for unknown characters, such as 'Ns' in DNA sequences
        self.encoder.fit(np.array(list(alphabet)).reshape(-1, 1))

    def _sequence_to_array(self, sequence: str) -> np.ndarray:
        """This function transforms the given sequence to an array.

        Args:
            sequence (str): a sequence of characters.

        Returns:
            sequence_array (np.ndarray): the sequence as a numpy array

        Raises:
            TypeError: If the input data is not a string.

        Examples:
            >>> encoder = TextOneHotEncoder(alphabet="acgt")
            >>> encoder._sequence_to_array("acctg")
            array(['a'],['c'],['c'],['t'],['g'])
        """
        if not isinstance(sequence, str):
            error_msg = f"Expected string input for sequence, got {type(sequence).__name__}"
            logger.error(error_msg)
            raise TypeError(error_msg)

        if self.convert_lowercase:
            sequence = sequence.lower()

        sequence_array = np.array(list(sequence))
        return sequence_array.reshape(-1, 1)

    def encode(self, data: str) -> torch.Tensor:
        """One hot encodes a single sequence.

        Takes a single string sequence and returns a torch tensor of shape (sequence_length, alphabet_length).
        The returned tensor corresponds to the one hot encoding of the sequence.
        Unknown characters are represented by a vector of zeros.

        Args:
            data (str): single sequence

        Returns:
            encoded_data_point (torch.Tensor): one hot encoded sequence

        Raises:
            TypeError: If the input data is not a string.

        Examples:
            >>> encoder = TextOneHotEncoder(alphabet="acgt")
            >>> encoder.encode("acgt")
            tensor([[1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]])
            >>> encoder.encode("acgtn")
            tensor([[1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                    [0, 0, 0, 0]])

            >>> encoder = TextOneHotEncoder(alphabet="ACgt")
            >>> encoder.encode("acgt")
            tensor([[0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]])
            >>> encoder.encode("ACgt")
            tensor([[1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]])
        """
        sequence_array = self._sequence_to_array(data)
        transformed = self.encoder.transform(sequence_array)
        numpy_array = np.squeeze(np.stack(transformed.toarray()))
        return torch.from_numpy(numpy_array)

    def encode_all(self, data: Union[str, list[str]]) -> torch.Tensor:
        """Encodes a list of sequences.

        Takes a list of string sequences and returns a torch tensor of shape (number_of_sequences, sequence_length, alphabet_length).
        The returned tensor corresponds to the one hot encoding of the sequences.
        Unknown characters are represented by a vector of zeros.

        Args:
            data (Union[str, list[str]]): list of sequences or a single sequence

        Returns:
            encoded_data (torch.Tensor): one hot encoded sequences

        Raises:
            TypeError: If the input data is not a list or a string.
            ValueError: If all sequences do not have the same length when padding is False.

        Examples:
            >>> encoder = TextOneHotEncoder(alphabet="acgt")
            >>> encoder.encode_all(["acgt", "acgtn"])
            tensor([[[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1],
                     [0, 0, 0, 0]], // this is padded with zeros

                    [[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1],
                     [0, 0, 0, 0]]])
        """
        # encode data
        if isinstance(data, str):
            encoded_data = self.encode(data)
            return torch.stack([encoded_data])
        if isinstance(data, list):
            # TODO instead maybe we can run encode_multiprocess when data size is larger than a certain threshold.
            encoded_data = self.encode_multiprocess(data)  # type: ignore[assignment]
        else:
            error_msg = f"Expected list or string input for data, got {type(data).__name__}"
            logger.error(error_msg)
            raise TypeError(error_msg)

        # handle padding
        if self.padding:
            max_length = max([len(d) for d in encoded_data])
            encoded_data = [np.pad(d, ((0, max_length - len(d)), (0, 0))) for d in encoded_data]  # type: ignore[assignment]
        else:
            lengths = {len(d) for d in encoded_data}
            if len(lengths) > 1:
                error_msg = "All sequences must have the same length when padding is False."
                logger.error(error_msg)
                raise ValueError(error_msg)

        return torch.from_numpy(np.array(encoded_data))

    def decode(self, data: torch.Tensor) -> Union[str, list[str]]:
        """Decodes one-hot encoded tensor back to sequences.

        Args:
            data (torch.Tensor): 2D or 3D tensor of one-hot encoded sequences
                - 2D shape: (sequence_length, alphabet_size)
                - 3D shape: (batch_size, sequence_length, alphabet_size)

        NOTE that when decoding 3D shape tensor, it assumes all sequences have the same length.

        Returns:
            Union[str, list[str]]: Single sequence string or list of sequence strings

        Raises:
            TypeError: If the input data is not a 2D or 3D tensor
        """
        expected_2d_tensor = 2
        expected_3d_tensor = 3

        if data.dim() == expected_2d_tensor:
            # Single sequence
            data_np = data.numpy().reshape(-1, len(self.alphabet))
            decoded = self.encoder.inverse_transform(data_np).flatten()
            return "".join([i for i in decoded if i is not None])

        if data.dim() == expected_3d_tensor:
            # Multiple sequences
            batch_size, seq_len, _ = data.shape
            data_np = data.reshape(-1, len(self.alphabet)).numpy()
            decoded = self.encoder.inverse_transform(data_np)
            sequences = decoded.reshape(batch_size, seq_len)
            # Convert to masked array where None values are masked
            masked_sequences = np.ma.masked_equal(sequences, None)
            # Fill masked values with "-"
            filled_sequences = masked_sequences.filled("-")
            return ["".join(seq) for seq in filled_sequences]

        raise ValueError(f"Expected 2D or 3D tensor, got {data.dim()}D")


class NumericEncoder(AbstractEncoder):
    """Encoder for float/int data.

    Attributes:
        dtype (torch.dtype): The data type of the encoded data. Default = torch.float32 (32-bit floating point)
    """

    def __init__(self, dtype: torch.dtype = torch.float32) -> None:
        """Initialize the NumericEncoder class.

        Args:
            dtype (torch.dtype): the data type of the encoded data. Default = torch.float (32-bit floating point)
        """
        self.dtype = dtype

    def encode(self, data: float) -> torch.Tensor:
        """Encodes the data.

        This method takes as input a single data point, should be mappable to a single output.

        Args:
            data (float): a single data point

        Returns:
            encoded_data_point (torch.Tensor): the encoded data point
        """
        return self.encode_all([data])

    def encode_all(self, data: list[float]) -> torch.Tensor:
        """Encodes the data.

        This method takes as input a list of data points, or a single float, and returns a torch.tensor.

        Args:
            data (list[float]): a list of data points or a single data point

        Returns:
            encoded_data (torch.Tensor): the encoded data
        """
        if not isinstance(data, list):
            data = [data]

        self._check_input_dtype(data)
        self._warn_float_is_converted_to_int(data)

        return torch.tensor(data, dtype=self.dtype)

    def decode(self, data: torch.Tensor) -> list[float]:
        """Decodes the data.

        Args:
            data (torch.Tensor): the encoded data

        Returns:
            decoded_data (list[float]): the decoded data
        """
        return data.cpu().numpy().tolist()

    def _check_input_dtype(self, data: list[float]) -> None:
        """Check if the input data is int or float data.

        Args:
            data (list[float]): a list of float data points

        Raises:
            ValueError: If the input data contains a non-integer or non-float data point
        """
        if not all(isinstance(d, (int, float)) for d in data):
            err_msg = "Expected input data to be a float or int"
            logger.error(err_msg)
            raise ValueError(err_msg)

    def _warn_float_is_converted_to_int(self, data: list[float]) -> None:
        """Warn if float data is encoded into int data.

        Args:
            data (list[float]): a list of float data points
        """
        if any(isinstance(d, float) for d in data) and (
            self.dtype in [torch.int, torch.int8, torch.int16, torch.int32, torch.int64]
        ):
            logger.warning("Encoding float data to torch.int data type.")


class StrClassificationEncoder(AbstractEncoder):
    """A string classification encoder that converts lists of strings into numeric labels using scikit-learn.

    When scale is set to True, the labels are scaled to be between 0 and 1.

    Attributes:
        scale (bool): Whether to scale the labels to be between 0 and 1. Default = False

    Methods:
        encode(data: str) -> int:
            Raises a NotImplementedError, as encoding a single string is not meaningful in this context.
        encode_all(data: list[str]) -> torch.tensor:
            Encodes an entire list of string data into a numeric representation using LabelEncoder and
            returns a torch tensor. Ensures that the provided data items are valid strings prior to encoding.
        decode(data: Any) -> Any:
            Raises a NotImplementedError, as decoding is not supported with the current design.
        _check_dtype(data: list[str]) -> None:
            Validates that all items in the data list are strings, raising a ValueError otherwise.
    """

    def __init__(self, *, scale: bool = False) -> None:
        """Initialize the StrClassificationEncoder class.

        Args:
            scale (bool): whether to scale the labels to be between 0 and 1. Default = False
        """
        self.scale = scale

    def encode(self, data: str) -> int:
        """Returns an error since encoding a single string does not make sense.

        Args:
            data (str): a single string
        """
        raise NotImplementedError("Encoding a single string does not make sense. Use encode_all instead.")

    def encode_all(self, data: Union[str, list[str]]) -> torch.Tensor:
        """Encodes the data.

        This method takes as input a list of data points, should be mappable to a single output, using LabelEncoder from scikit learn and returning a numpy array.
        For more info visit : https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html

        Args:
            data (Union[str, list[str]]): a list of strings or single string

        Returns:
            encoded_data (torch.tensor): the encoded data
        """
        if not isinstance(data, list):
            data = [data]

        self._check_dtype(data)

        encoder = preprocessing.LabelEncoder()
        encoded_data = torch.tensor(encoder.fit_transform(data))
        if self.scale:
            encoded_data = encoded_data / max(len(encoded_data) - 1, 1)

        return encoded_data

    def decode(self, data: Any) -> Any:
        """Returns an error since decoding does not make sense without encoder information, which is not yet supported."""
        raise NotImplementedError("Decoding is not yet supported for StrClassification.")

    def _check_dtype(self, data: list[str]) -> None:
        """Check if the input data is string data.

        Args:
            data (list[str]): a list of strings

        Raises:
            ValueError: If the input data is not a string
        """
        if not all(isinstance(d, str) for d in data):
            err_msg = "Expected input data to be a list of strings"
            logger.error(err_msg)
            raise ValueError(err_msg)


class NumericRankEncoder(AbstractEncoder):
    """Encoder for float/int data that encodes the data based on their rank.

    Attributes:
        scale (bool): whether to scale the ranks to be between 0 and 1. Default = False

    Methods:
        encode: encodes a single data point
        encode_all: encodes a list of data points into a torch.tensor
        decode: decodes a single data point
        _check_input_dtype: checks if the input data is int or float data
    """

    def __init__(self, *, scale: bool = False) -> None:
        """Initialize the NumericRankEncoder class.

        Args:
            scale (bool): whether to scale the ranks to be between 0 and 1. Default = False
        """
        self.scale = scale

    def encode(self, data: Any) -> torch.Tensor:
        """Returns an error since encoding a single float does not make sense."""
        raise NotImplementedError("Encoding a single float does not make sense. Use encode_all instead.")

    def encode_all(self, data: list[Union[int, float]]) -> torch.Tensor:
        """Encodes the data.

        This method takes as input a list of data points, and returns the ranks of the data points.
        The ranks are normalized to be between 0 and 1, when scale is set to True.

        Args:
            data (list[Union[int, float]]): a list of numeric values

        Returns:
            encoded_data (torch.Tensor): the encoded data
        """
        if not isinstance(data, list):
            data = [data]
        self._check_input_dtype(data)

        # Get ranks (0 is lowest, n-1 is highest)
        # and normalize to be between 0 and 1
        array_data: np.ndarray = np.array(data)
        ranks: np.ndarray = np.argsort(np.argsort(array_data))
        if self.scale:
            ranks = ranks / max(len(ranks) - 1, 1)
        return torch.tensor(ranks)

    def decode(self, data: Any) -> Any:
        """Returns an error since decoding does not make sense without encoder information, which is not yet supported."""
        raise NotImplementedError("Decoding is not yet supported for NumericRank.")

    def _check_input_dtype(self, data: list[Union[int, float]]) -> None:
        """Check if the input data is int or float data.

        Args:
            data (list[Union[int, float]]): a list of numeric values

        Raises:
            ValueError: If the input data is not a float
        """
        if not all(isinstance(d, (int, float)) for d in data):
            err_msg = f"Expected input data to be a float or int, got {type(data).__name__}"
            logger.error(err_msg)
            raise ValueError(err_msg)
