import json
import os
import unittest
import numpy as np
import numpy.testing as npt
import polars as pl

from src.stimulus.data.csv import CsvProcessing, CsvLoader
from src.stimulus.data.experiments import DnaToFloatExperiment, ProtDnaToFloatExperiment

class TestCsvProcessing(unittest.TestCase):
    """Base class for testing CsvProcessing."""

    def setUp(self):
        self.csv_processing = None
        self.configs = None
        self.data_length = None

    def test_len(self):
        """Test if data is loaded with correct shape."""
        self.assertEqual(len(self.csv_processing.data), self.data_length)

    def test_add_split(self):
        """Test adding split to the data."""
        self.csv_processing.add_split(self.configs['split'])
        self._test_random_splitter(self.expected_splits)

    def test_transform(self):
        """Test data transformation."""
        self.csv_processing.transform(self.configs['transform'])
        self._test_transformed_data()

    def _test_random_splitter(self, expected_splits):
        for i in range(self.data_length):
            self.assertEqual(self.csv_processing.data['split:split:int'][i], expected_splits[i])

    def _test_transformed_data(self):
        raise NotImplementedError("Subclasses should implement this method.")

    def _test_column_values(self, column_name, expected_values):
        observed_values = list(self.csv_processing.data[column_name])
        observed_values = [round(v, 6) if isinstance(v, float) else v for v in observed_values]
        self.assertEqual(observed_values, expected_values)

class TestDnaToFloatCsvProcessing(TestCsvProcessing):
    """Test CsvProcessing for DnaToFloatExperiment."""

    def setUp(self):
        np.random.seed(123)
        pl.set_random_seed(123)
        self.experiment = DnaToFloatExperiment()
        self.csv_path = os.path.abspath("tests/test_data/dna_experiment/test.csv")
        self.csv_processing = CsvProcessing(self.experiment, self.csv_path)
        self.csv_shuffle_long_path = os.path.abspath("tests/test_data/dna_experiment/test_shuffling_long.csv")
        self.csv_shuffle_long = CsvProcessing(self.experiment, self.csv_shuffle_long_path)
        self.csv_shuffle_long_shuffled_path = os.path.abspath("tests/test_data/dna_experiment/test_shuffling_long_shuffled.csv")
        self.csv_shuffle_long_shuffled = CsvProcessing(self.experiment, self.csv_shuffle_long_shuffled_path)
        with open('tests/test_data/dna_experiment/test_config.json', 'r') as f:
            self.configs = json.load(f)
        self.data_length = 2
        self.expected_splits = [1, 0]

    def _test_transformed_data(self):
        self.data_length *= 2
        self._test_column_values('pet:meta:str', ['cat', 'dog', 'cat', 'dog'])
        self._test_column_values('hola:label:float', [12.676405, 12.540016, 12.676405, 12.540016])
        self._test_column_values('hello:input:dna', ['ACTGACTGATCGATNN', 'ACTGACTGATCGATNN', 'NNATCGATCAGTCAGT', 'NNATCGATCAGTCAGT'])
        self._test_column_values('split:split:int', [1, 0, 1, 0])

    def test_shuffle_labels(self):
        """Test shuffling of labels."""
        self.csv_shuffle_long.shuffle_labels(seed=42)
        npt.assert_array_equal(self.csv_shuffle_long.data['hola:label:float'], self.csv_shuffle_long_shuffled.data['hola:label:float'])

class TestProtDnaToFloatCsvProcessing(TestCsvProcessing):
    """Test CsvProcessing for ProtDnaToFloatExperiment."""

    def setUp(self):
        self.experiment = ProtDnaToFloatExperiment()
        self.csv_path = os.path.abspath("tests/test_data/prot_dna_experiment/test.csv")
        self.csv_processing = CsvProcessing(self.experiment, self.csv_path)
        with open('tests/test_data/prot_dna_experiment/test_config.json', 'r') as f:
            self.configs = json.load(f)
        self.data_length = 2
        self.expected_splits = [1, 0]

    def _test_transformed_data(self):
        self.data_length *= 2
        self._test_column_values('pet:meta:str', ['cat', 'dog', 'cat', 'dog'])
        self._test_column_values('hola:label:float', [12.676405, 12.540016, 12.676405, 12.540016])
        self._test_column_values('hello:input:dna', ['ACTGACTGATCGATNN', 'ACTGACTGATCGATNN', 'NNATCGATCAGTCAGT', 'NNATCGATCAGTCAGT'])
        self._test_column_values('split:split:int', [1, 0, 1, 0])
        self._test_column_values('bonjour:input:prot', ['GPRTTIKAKQLETLX', 'GPRTTIKAKQLETLX', 'GPRTTIKAKQLETLX', 'GPRTTIKAKQLETLX'])

class TestCsvLoader(unittest.TestCase):
    """Base class for testing CsvLoader."""

    def setUp(self):
        self.csv_loader = None
        self.data_shape = None
        self.data_shape_split = None
        self.shape_splits = None

    def test_len(self):
        """Test the length of the dataset."""
        self.assertEqual(len(self.csv_loader), self.data_shape[0])

    def test_parse_csv_to_input_label_meta(self):
        """Test parsing of CSV to input, label, and meta."""
        self.assertIsInstance(self.csv_loader.input, dict)
        self.assertIsInstance(self.csv_loader.label, dict)
        self.assertIsInstance(self.csv_loader.meta, dict)

    def test_get_encoded_item_unique(self):
        """Test getting a single encoded item."""
        encoded_item = self.csv_loader[0]
        self._assert_encoded_item(encoded_item, expected_length=1)

    def test_get_encoded_item_multiple(self):
        """Test getting multiple encoded items."""
        encoded_item = self.csv_loader[slice(0, 2)]
        self._assert_encoded_item(encoded_item, expected_length=2)

    def test_load_with_split(self):
        """Test loading with split."""
        self.csv_loader_split = CsvLoader(self.experiment, self.csv_path_split)
        self.assertEqual(len(self.csv_loader_split), self.data_shape_split[0])

        for i in [0, 1, 2]:
            self.csv_loader_split = CsvLoader(self.experiment, self.csv_path_split, split=i)
            self.assertEqual(len(self.csv_loader_split.input['hello:dna']), self.shape_splits[i])

        with self.assertRaises(ValueError):
            CsvLoader(self.experiment, self.csv_path_split, split=3)

    def test_get_all_items(self):
        """Test getting all items."""
        input_data, label_data, meta_data = self.csv_loader.get_all_items()
        self.assertIsInstance(input_data, dict)
        self.assertIsInstance(label_data, dict)
        self.assertIsInstance(meta_data, dict)

    def _assert_encoded_item(self, encoded_item, expected_length):
        self.assertEqual(len(encoded_item), 3)
        for i in range(3):
            self.assertIsInstance(encoded_item[i], dict)
            for key in encoded_item[i].keys():
                self.assertIsInstance(encoded_item[i][key], np.ndarray)
                if expected_length > 1: # If the expected length is 0, this will fail as we are trying to find the length of an object size 0.
                    self.assertEqual(len(encoded_item[i][key]), expected_length)

class TestDnaToFloatCsvLoader(TestCsvLoader):
    """Test CsvLoader for DnaToFloatExperiment."""

    def setUp(self):
        self.csv_path = os.path.abspath("tests/test_data/dna_experiment/test.csv")
        self.csv_path_split = os.path.abspath("tests/test_data/dna_experiment/test_with_split.csv")
        self.experiment = DnaToFloatExperiment()
        self.csv_loader = CsvLoader(self.experiment, self.csv_path)
        self.data_shape = [2, 3]
        self.data_shape_split = [48, 4]
        self.shape_splits = {0: 16, 1: 16, 2: 16}

class TestProtDnaToFloatCsvLoader(TestCsvLoader):
    """Test CsvLoader for ProtDnaToFloatExperiment."""

    def setUp(self):
        self.csv_path = os.path.abspath("tests/test_data/prot_dna_experiment/test.csv")
        self.csv_path_split = os.path.abspath("tests/test_data/prot_dna_experiment/test_with_split.csv")
        self.experiment = ProtDnaToFloatExperiment()
        self.csv_loader = CsvLoader(self.experiment, self.csv_path)
        self.data_shape = [2, 4]
        self.data_shape_split = [3, 5]
        self.shape_splits = {0: 1, 1: 1, 2: 1}

if __name__ == "__main__":
    unittest.main()