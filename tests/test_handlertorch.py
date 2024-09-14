import unittest
import os
import torch
import numpy as np
from abc import ABC, abstractmethod
from typing import Any, Dict
from src.stimulus.data.handlertorch import TorchDataset
from src.stimulus.data.experiments import DnaToFloatExperiment, ProtDnaToFloatExperiment, TitanicExperiment

class TestTorchDataset(ABC):
    """Base class for testing TorchDataset."""

    def setUp(self):
        if type(self) is TestTorchDataset:
            raise NotImplementedError("TestTorchDataset is a base class and should not be instantiated directly.")
        self.torchdataset = None
        self.expected_len = None
        self.expected_input_shape = None
        self.expected_label_shape = None
        self.expected_item_shape = None

    def test_len(self):
        """Test the length of the dataset."""
        self.assertEqual(len(self.torchdataset), self.expected_len)

    def test_convert_dict_to_dict_of_tensor(self):
        """Test conversion of dict to dict of tensors."""
        self._test_convert_dict_to_dict_of_tensor(self.torchdataset.input, self.expected_input_shape)
        self._test_convert_dict_to_dict_of_tensor(self.torchdataset.label, self.expected_label_shape)

    def test_get_item(self):
        """Test getting items from the dataset."""
        self._test_get_item_shape(0, self.expected_item_shape)
        self._test_get_item_shape(slice(0, 2), {k: [2] + v for k, v in self.expected_item_shape.items()})

    def _test_convert_dict_to_dict_of_tensor(self, data: Dict[str, torch.Tensor], expected_shape: Dict[str, list]):
        for key in data:
            self.assertIsInstance(data[key], torch.Tensor)
            self.assertEqual(data[key].shape, torch.Size(expected_shape[key]))

    def _test_get_item_shape(self, idx: Any, expected_size: Dict[str, list]):
        x, y, meta = self.torchdataset[idx]
        self.assertIsInstance(x, dict)
        self.assertIsInstance(y, dict)
        self.assertIsInstance(meta, dict)
        for key, value in {**x, **y, **meta}.items():
            if key in expected_size:
                self.assertEqual(value.shape, torch.Size(expected_size[key]))

class TestDnaToFloatTorchDatasetSameLength(TestTorchDataset, unittest.TestCase):
    """Test TorchDataset for DnaToFloatExperiment with same length sequences."""

    def setUp(self):
        super().setUp()
        self.torchdataset = TorchDataset(csvpath=os.path.abspath("tests/test_data/dna_experiment/test.csv"), 
                                         experiment=DnaToFloatExperiment())
        self.expected_len = 2
        self.expected_input_shape = {"hello": [2, 16, 4]}
        self.expected_label_shape = {"hola": [2]}
        self.expected_item_shape = {'hello': [16, 4]}

class TestDnaToFloatTorchDatasetDifferentLength(TestTorchDataset, unittest.TestCase):
    """Test TorchDataset for DnaToFloatExperiment with different length sequences."""

    def setUp(self):
        super().setUp()
        self.torchdataset = TorchDataset(csvpath=os.path.abspath("tests/test_data/dna_experiment/test_unequal_dna_float.csv"), 
                                         experiment=DnaToFloatExperiment())
        self.expected_len = 4
        self.expected_input_shape = {"hello": [4, 31, 4]}
        self.expected_label_shape = {"hola": [4]}
        self.expected_item_shape = {'hello': [31, 4]}

class TestProtDnaToFloatTorchDatasetSameLength(TestTorchDataset, unittest.TestCase):
    """Test TorchDataset for ProtDnaToFloatExperiment with same length sequences."""

    def setUp(self):
        super().setUp()
        self.torchdataset = TorchDataset(csvpath=os.path.abspath("tests/test_data/prot_dna_experiment/test.csv"), 
                                         experiment=ProtDnaToFloatExperiment())
        self.expected_len = 2
        self.expected_input_shape = {"hello": [2, 16, 4], "bonjour": [2, 15, 20]}
        self.expected_label_shape = {"hola": [2]}
        self.expected_item_shape = {'hello': [16, 4], 'bonjour': [15, 20]}

class TestTitanicTorchDataset(TestTorchDataset, unittest.TestCase):
    """Test TorchDataset for TitanicExperiment."""

    def setUp(self):
        super().setUp()
        self.torchdataset = TorchDataset(csvpath=os.path.abspath("tests/test_data/titanic/titanic_stimulus.csv"), 
                                         experiment=TitanicExperiment())
        self.expected_len = 712
        # Add expected shapes for Titanic dataset if known
        self.expected_input_shape = {}  # Fill this with the expected input shape
        self.expected_label_shape = {}  # Fill this with the expected label shape
        self.expected_item_shape = {}   # Fill this with the expected item shape

    def test_convert_dict_to_dict_of_tensor(self):
        """Override this method if Titanic dataset has different requirements."""
        pass

    def test_get_item(self):
        """Override this method if Titanic dataset has different requirements."""
        pass

if __name__ == "__main__":
    unittest.main()