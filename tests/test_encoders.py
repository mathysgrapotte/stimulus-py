import unittest
import numpy as np
import numpy.testing as npt
from abc import ABC, abstractmethod
from src.stimulus.data.encoding.encoders import TextOneHotEncoder, IntRankEncoder, StrClassificationIntEncoder

class TestEncoder(ABC):
    """Base class for testing encoders."""

    def setUp(self):
        if type(self) is TestEncoder:
            raise NotImplementedError("TestEncoder is a base class and should not be instantiated directly.")
        self.encoder = None
        self.input_data = None
        self.expected_encoded = None
        self.expected_decoded = None

    def test_encode(self):
        """Test encoding of data."""
        encoded_data = self.encoder.encode(self.input_data)
        self._assert_encoded(encoded_data, self.expected_encoded)

    def test_decode(self):
        """Test decoding of data."""
        decoded_data = self.encoder.decode(self.expected_encoded)
        self._assert_decoded(decoded_data, self.expected_decoded)

    def test_encode_all(self):
        """Test encoding of multiple data points."""
        encoded_data = self.encoder.encode_all(self.input_data_list)
        self._assert_encoded_all(encoded_data, self.expected_encoded_list)

    @abstractmethod
    def _assert_encoded(self, encoded, expected):
        pass

    @abstractmethod
    def _assert_decoded(self, decoded, expected):
        pass

    @abstractmethod
    def _assert_encoded_all(self, encoded, expected):
        pass

class TestTextOneHotEncoder(TestEncoder, unittest.TestCase):
    """Test TextOneHotEncoder."""

    def setUp(self):
        super().setUp()
        self.encoder = TextOneHotEncoder("acgt")
        self.input_data = "ACGT"
        self.expected_encoded = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        self.expected_decoded = np.array(['a', 'c', 'g', 't']).reshape(-1, 1)
        self.input_data_list = ["ACGT", "ACG", "AC"]
        self.expected_encoded_list = [
            np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]),
            np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]),
            np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        ]

    def _assert_encoded(self, encoded, expected):
        self.assertIsInstance(encoded, np.ndarray)
        self.assertEqual(encoded.shape, expected.shape)
        npt.assert_array_equal(encoded, expected)

    def _assert_decoded(self, decoded, expected):
        self.assertIsInstance(decoded, np.ndarray)
        self.assertEqual(decoded.shape, expected.shape)
        npt.assert_array_equal(decoded, expected)

    def _assert_encoded_all(self, encoded, expected):
        self.assertIsInstance(encoded, list)
        self.assertEqual(len(encoded), len(expected))
        for enc, exp in zip(encoded, expected):
            self._assert_encoded(enc, exp)

    def test_encode_out_of_alphabet(self):
        """Test encoding of characters outside the alphabet."""
        encoded_data = self.encoder.encode("Bubba")
        expected_output = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0]])
        self._assert_encoded(encoded_data, expected_output)

class TestIntRankEncoder(TestEncoder, unittest.TestCase):
    """Test IntRankEncoder."""

    def setUp(self):
        super().setUp()
        self.encoder = IntRankEncoder()
        self.input_data = [1, 2, 3]
        self.expected_encoded = np.array([0, 0.5, 1])
        self.expected_decoded = np.array([1, 2, 3])
        self.input_data_list = [[1, 2, 3], [4, 5, 6]]
        self.expected_encoded_list = [np.array([0, 0.5, 1]), np.array([0, 0.5, 1])]

    def _assert_encoded(self, encoded, expected):
        self.assertIsInstance(encoded, np.ndarray)
        self.assertEqual(encoded.shape, expected.shape)
        npt.assert_array_almost_equal(encoded, expected)

    def _assert_decoded(self, decoded, expected):
        self.assertIsInstance(decoded, np.ndarray)
        self.assertEqual(decoded.shape, expected.shape)
        npt.assert_array_equal(decoded, expected)

    def _assert_encoded_all(self, encoded, expected):
        self.assertIsInstance(encoded, np.ndarray)
        self.assertEqual(encoded.shape, (len(self.input_data_list), len(self.input_data_list[0])))
        for enc, exp in zip(encoded, expected):
            npt.assert_array_almost_equal(enc, exp)

class TestStrClassificationIntEncoder(TestEncoder, unittest.TestCase):
    """Test StrClassificationIntEncoder."""

    def setUp(self):
        super().setUp()
        self.encoder = StrClassificationIntEncoder()
        self.input_data = ["A", "B", "C", "A"]
        self.expected_encoded = np.array([0, 1, 2, 0])
        self.expected_decoded = np.array(["A", "B", "C", "A"])
        self.input_data_list = [["A", "B", "C", "A"], ["D", "E", "F"]]
        self.expected_encoded_list = [np.array([0, 1, 2, 0]), np.array([0, 1, 2])]

    def _assert_encoded(self, encoded, expected):
        self.assertIsInstance(encoded, np.ndarray)
        self.assertEqual(encoded.shape, expected.shape)
        npt.assert_array_equal(encoded, expected)

    def _assert_decoded(self, decoded, expected):
        self.assertIsInstance(decoded, np.ndarray)
        self.assertEqual(decoded.shape, expected.shape)
        npt.assert_array_equal(decoded, expected)

    def _assert_encoded_all(self, encoded, expected):
        self.assertIsInstance(encoded, np.ndarray)
        self.assertEqual(encoded.shape, (len(self.input_data_list), len(max(self.input_data_list, key=len))))
        for enc, exp in zip(encoded, expected):
            npt.assert_array_equal(enc[:len(exp)], exp)

if __name__ == "__main__":
    unittest.main()
