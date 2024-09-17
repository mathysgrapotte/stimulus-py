import unittest
import numpy as np
from abc import ABC, abstractmethod
from src.stimulus.data.transform.data_transformation_generators import UniformTextMasker, GaussianNoise, ReverseComplement, GaussianChunk

class TestDataTransformer(ABC):
    """Base class for testing data transformers."""

    @abstractmethod
    def setUp(self):
        self.transformer = None

    def test_transform_single(self):
        """Test transforming a single item."""
        transformed_data = self.transformer.transform(self.single_input, **self.single_params)
        self.assertIsInstance(transformed_data, self.expected_type)
        self.assertEqual(transformed_data, self.expected_single_output)

    def test_transform_all_single_item(self):
        """Test transforming a list with a single item."""
        transformed_data = self.transformer.transform_all([self.single_input], **self.single_params)
        self.assertIsInstance(transformed_data, list)
        self.assertIsInstance(transformed_data[0], self.expected_type)
        self.assertEqual(transformed_data, [self.expected_single_output])

    def test_transform_all_multiple_items(self):
        """Test transforming a list with multiple items."""
        transformed_data = self.transformer.transform_all(self.multiple_inputs, **self.multiple_params)
        self.assertIsInstance(transformed_data, list)
        for item in transformed_data:
            self.assertIsInstance(item, self.expected_type)
        self.assertEqual(transformed_data, self.expected_multiple_outputs)

class TestUniformTextMasker(TestDataTransformer, unittest.TestCase):
    def setUp(self):
        self.transformer = UniformTextMasker(mask='N')
        self.single_input = "ACGTACGT"
        self.single_params = {"seed": 42, "probability": 0.1}
        self.expected_type = str
        self.expected_single_output = "ACGTACNT"
        self.multiple_inputs = ["ATCGATCGATCG", "ATCG"]
        self.multiple_params = {"seed": 42, "probability": 0.1}
        self.expected_multiple_outputs = ['ATCGATNGATNG', 'ATCG']

class TestGaussianNoise(TestDataTransformer, unittest.TestCase):
    def setUp(self):
        self.transformer = GaussianNoise()
        self.single_input = 5.0
        self.single_params = {"seed": 42, "mean": 0, "std": 1}
        self.expected_type = float
        self.expected_single_output = 5.4967141530112327
        self.multiple_inputs = [1.0, 2.0, 3.0]
        self.multiple_params = {"seed": 42, "mean": 0, "std": 1}
        self.expected_multiple_outputs = [1.4967141530112327, 2.0211241446210543, 3.7835298641951802]

    def test_transform_single(self):
        transformed_data = self.transformer.transform(self.single_input, **self.single_params)
        self.assertIsInstance(transformed_data, self.expected_type)
        self.assertAlmostEqual(transformed_data, self.expected_single_output, places=7)

    def test_transform_all_multiple_items(self):
        transformed_data = self.transformer.transform_all(self.multiple_inputs, **self.multiple_params)
        self.assertIsInstance(transformed_data, list)
        for item, expected in zip(transformed_data, self.expected_multiple_outputs):
            self.assertIsInstance(item, self.expected_type)
            self.assertAlmostEqual(item, expected, places=7)

class TestReverseComplement(TestDataTransformer, unittest.TestCase):
    def setUp(self):
        self.transformer = ReverseComplement()
        self.single_input = "ACCCCTACGTNN"
        self.single_params = {}
        self.expected_type = str
        self.expected_single_output = "NNACGTAGGGGT"
        self.multiple_inputs = ["ACCCCTACGTNN", "ACTGA"]
        self.multiple_params = {}
        self.expected_multiple_outputs = ['NNACGTAGGGGT', 'TCAGT']

class TestGaussianChunk(TestDataTransformer, unittest.TestCase):
    def setUp(self):
        self.transformer = GaussianChunk()
        self.single_input = "AGCATGCTAGCTAGATCAAAATCGATGCATGCTAGCGGCGCGCATGCATGAGGAGACTGAC"
        self.single_params = {"seed": 42, "chunk_size": 10, "std": 1}
        self.expected_type = str
        self.expected_single_output = "TGCATGCTAG"
        self.multiple_inputs = [
            "AGCATGCTAGCTAGATCAAAATCGATGCATGCTAGCGGCGCGCATGCATGAGGAGACTGAC",
            "AGCATGCTAGCTAGATCAAAATCGATGCATGCTAGCGGCGCGCATGCATGAGGAGACTGAC"
        ]
        self.multiple_params = {"seed": 42, "chunk_size": 10, "std": 1}
        self.expected_multiple_outputs = ["TGCATGCTAG", "GCATGCTAGC"]

    def test_transform_single(self):
        transformed_data = self.transformer.transform(self.single_input, **self.single_params)
        self.assertIsInstance(transformed_data, self.expected_type)
        self.assertEqual(len(transformed_data), 10)
        self.assertEqual(transformed_data, self.expected_single_output)

    def test_transform_all_multiple_items(self):
        transformed_data = self.transformer.transform_all(self.multiple_inputs, **self.multiple_params)
        self.assertIsInstance(transformed_data, list)
        for item in transformed_data:
            self.assertIsInstance(item, self.expected_type)
            self.assertEqual(len(item), 10)
        self.assertEqual(transformed_data, self.expected_multiple_outputs)

if __name__ == "__main__":
    unittest.main()