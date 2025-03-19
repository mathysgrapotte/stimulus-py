"""This file contains encoders classes for encoding various types of data."""

import logging
from abc import ABC, abstractmethod
from typing import Any, Union

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from sklearn import preprocessing

from stimulus.learner.optuna_tune import get_device

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


class TextOneHotEncoder(AbstractEncoder):
    """One hot encoder for text data with highly optimized implementation.

    If a character c is not in the alphabet, c will be represented by a vector of zeros.
    This encoder is optimized for processing large batches of sequences efficiently on GPU.
    """

    # Constants to replace magic numbers
    TENSOR_3D_SHAPE = 3
    ASCII_MAX_VALUE = 128

    def __init__(
        self,
        alphabet: str = "acgt",
        dtype: torch.dtype = torch.float32,
        *,
        convert_lowercase: bool = False,
        force_cpu: bool = True,
        padding: bool = False,
    ) -> None:
        """Initialize the TextOneHotEncoder class.

        Args:
            alphabet (str): the alphabet to one hot encode the data with.
            dtype (torch.dtype): the data type of the encoded data. Default = torch.float32 (32-bit floating point)
            convert_lowercase (bool): whether to convert sequences to lowercase.
            force_cpu (bool): whether to force the encoder to run on CPU.
            padding (bool): whether to pad sequences of different lengths.
        """
        if convert_lowercase:
            alphabet = alphabet.lower()
        self.convert_lowercase = convert_lowercase
        self.alphabet = alphabet
        self.padding = padding
        self.dtype = dtype
        self.device = get_device() if not force_cpu else torch.device("cpu")

        # Pre-compute and cache character mappings for the entire ASCII range
        self.UNKNOWN_IDX = -1
        self.alphabet_size = len(alphabet)

        # Build a fast ASCII-to-index mapping table directly on the GPU
        # This is more efficient than dictionary lookups
        self.lookup_table = torch.full((self.ASCII_MAX_VALUE,), self.UNKNOWN_IDX, dtype=torch.int64, device=self.device)

        # Fill the lookup table with character indices
        for idx, char in enumerate(alphabet):
            self.lookup_table[ord(char)] = idx
            # Handle case conversion if needed
            if convert_lowercase:
                if "A" <= char <= "Z":
                    self.lookup_table[ord(char.lower())] = idx
                elif "a" <= char <= "z":
                    self.lookup_table[ord(char.upper())] = idx

        # Pre-allocate a mask for invalid characters to avoid nonzero operations
        # Initialize with ones to mark valid positions by default
        self.alphabet_mask = torch.ones(self.alphabet_size + 1, dtype=self.dtype, device=self.device)
        # Set the last position (for invalid characters) to zero
        self.alphabet_mask[-1] = 0.0

    def encode_all(self, data: Union[str, list[str]]) -> torch.Tensor:
        """Encode all sequences in a batch using fully vectorized operations.

        Args:
            data (Union[str, list[str]]): A single sequence or list of sequences

        Returns:
            torch.Tensor: Tensor of shape (batch_size, max_seq_length, alphabet_size)
        """
        # Handle single string case
        if isinstance(data, str):
            data = [data]
        elif not isinstance(data, list):
            error_msg = f"Expected list or string input for data, got {type(data).__name__}"
            logger.error(error_msg)
            raise TypeError(error_msg)

        # Early check for sequence length consistency when padding=False
        if not self.padding:
            lengths = {len(seq) for seq in data}
            if len(lengths) > 1:
                error_msg = "All sequences must have the same length when padding is False."
                logger.error(error_msg)
                raise ValueError(error_msg)

        # Find max length for processing all sequences at once
        max_length = max(len(seq) for seq in data)
        batch_size = len(data)

        # OPTIMIZATION: Process all sequences as a single byte array
        # This eliminates Python loops and character-by-character processing
        ascii_array = np.zeros((batch_size, max_length), dtype=np.uint8)

        # Convert sequences to bytes more efficiently
        for i, seq_input in enumerate(data):
            seq = seq_input.lower() if self.convert_lowercase else seq_input

            # OPTIMIZATION: Use numpy byte array conversion to avoid Python loop
            seq_bytes = np.frombuffer(seq.encode("ascii", errors="ignore"), dtype=np.uint8)
            ascii_array[i, : len(seq_bytes)] = seq_bytes

        # Transfer to GPU in one operation
        # OPTIMIZATION: Use torch.tensor directly on device rather than to() to avoid copy
        ascii_tensor = torch.tensor(ascii_array, dtype=torch.int64, device=self.device)

        # OPTIMIZATION: Create valid ASCII mask directly
        # This combines multiple operations into one
        valid_mask = (ascii_tensor > 0) & (ascii_tensor < self.ASCII_MAX_VALUE)

        # Create indices tensor - use -1 for padding/invalid chars
        indices = torch.full_like(ascii_tensor, self.UNKNOWN_IDX)

        # Only lookup valid ASCII values (avoiding unnecessary computation)
        valid_indices = valid_mask.nonzero(as_tuple=True)
        indices[valid_indices] = self.lookup_table[ascii_tensor[valid_indices]]

        # For one-hot encoding, we need non-negative indices
        # OPTIMIZATION: Use a single mask for padding and unknown chars
        valid_indices_mask = indices >= 0
        safe_indices = indices.clone()
        safe_indices[~valid_indices_mask] = 0  # Temporary index for one_hot

        # Apply one-hot encoding - FIX: removed the 'out' parameter
        one_hot = F.one_hot(safe_indices, num_classes=self.alphabet_size + 1).to(self.dtype)

        # Apply alphabet mask to zero out invalid indices
        # This creates zeros for unknown characters
        result = one_hot.clone()
        result[~valid_indices_mask] = 0.0

        # Remove the last dimension (sentinel value) to get the final shape
        return result[:, :, : self.alphabet_size]

    def encode(self, data: str) -> torch.Tensor:
        """Encode a single sequence by delegating to encode_all.

        Args:
            data (str): The sequence to encode

        Returns:
            torch.Tensor: One-hot encoded tensor of shape (sequence_length, alphabet_size)
        """
        result = self.encode_all([data])
        return result.squeeze(0)

    def decode(self, data: torch.Tensor) -> Union[str, list[str]]:
        """Decode a one-hot encoded tensor back to sequences.

        Args:
            data (torch.Tensor): 2D or 3D tensor of one-hot encoded sequences
                - 2D shape: (sequence_length, alphabet_size)
                - 3D shape: (batch_size, sequence_length, alphabet_size)

        Returns:
            decoded_data (Union[str, list[str]]): decoded data points
        """
        # Check if we have a batch or single sequence
        is_batch = len(data.shape) == self.TENSOR_3D_SHAPE

        if not is_batch:
            # Add batch dimension if single sequence
            data = data.unsqueeze(0)

        # Get indices of maximum values along the alphabet dimension
        indices = torch.argmax(data, dim=2)

        # Check if any row is all zeros (unknown character)
        all_zeros = data.sum(dim=2) == 0

        # Convert to CPU for processing
        indices = indices.cpu().numpy()
        all_zeros = all_zeros.cpu().numpy()

        # Decode each sequence
        result = []
        for i, seq_indices in enumerate(indices):
            chars = []
            for j, idx in enumerate(seq_indices):
                # If the row is all zeros (unknown char) or no valid one-hot encoding
                if all_zeros[i, j]:
                    chars.append("-")
                # Only add valid characters from the alphabet
                elif 0 <= idx < len(self.alphabet):
                    chars.append(self.alphabet[idx])
                else:
                    chars.append("-")
            result.append("".join(chars))

        # Return a single string if input was a single sequence
        if not is_batch:
            return result[0]

        return result


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
