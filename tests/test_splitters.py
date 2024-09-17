"""
unit test cases for the noise_generators file shold be written like the following

Test case for the Splitter class.

To write test cases for a new noise generator class:
1. Create a new test case class by subclassing unittest.TestCase.
2. Write test methods to test the behavior of the noise generator class methods.
3. Use assertions (e.g., self.assertIsInstance, self.assertEqual) to verify the behavior of the noise generator class methods.

"""


import unittest
from abc import ABC, abstractmethod
import numpy as np
import polars as pl
from src.stimulus.data.splitters.splitters import RandomSplitter

def sample_data():
    """Create a sample dataframe for testing."""
    return pl.DataFrame({
        'A': [1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
        'B': [6, 7, 8, 9, 10, 6, 7, 8, 9, 10]
    })

class TestSplitterBase(ABC):
    """Base class for testing splitter classes."""

    @abstractmethod
    def setUp(self):
        self.splitter = None
        self.sample_data = None

    def test_get_split_indexes(self):
        """Test splitting with custom split proportions."""
        custom_split = [0.6, 0.3, 0.1]
        train, validation, test = self.splitter.get_split_indexes(
            data=self.sample_data, 
            split=custom_split, 
            seed=123
        )
        self._assert_split_indexes(train, validation, test)

    @abstractmethod
    def _assert_split_indexes(self, train, validation, test):
        pass

class TestRandomSplitter(TestSplitterBase, unittest.TestCase):
    """Test cases for RandomSplitter."""

    def setUp(self):
        np.random.seed(123)
        self.splitter = RandomSplitter()
        self.sample_data = sample_data()

    def _assert_split_indexes(self, train, validation, test):
        self.assertEqual(train, [4, 0, 7, 5, 8, 3])
        self.assertEqual(validation, [1, 6, 9])
        self.assertEqual(test, [2])

if __name__ == "__main__":
    unittest.main()