import unittest
from abc import ABC, abstractmethod
from src.stimulus.utils.performance import Performance

class TestPerformanceBase(ABC):
    """Base class for testing Performance metrics."""

    @abstractmethod
    def setUp(self):
        self.labels = None
        self.predictions = None
        self.metrics = None

    def test_metrics(self):
        """Test all metrics for the given labels and predictions."""
        for metric, expected_val in self.metrics.items():
            with self.subTest(metric=metric):
                performance = Performance(self.labels, self.predictions, metric=metric)
                calculated_value = round(performance.val, 2)
                self.assertEqual(calculated_value, expected_val)

class TestBinaryClassificationPerformance(TestPerformanceBase, unittest.TestCase):
    """Test Performance metrics for binary classification."""

    def setUp(self):
        self.labels = [0, 1, 0, 1]
        self.predictions = [0.1, 0.9, 0.7, 0.6]
        self.metrics = {
            "rocauc": 0.75,
            "prauc": 0.83,
            "mcc": 0.58,
            "f1score": 0.8,
            "precision": 0.67,
            "recall": 1.0,
            "spearmanr": 0.45
        }

if __name__ == "__main__":
    unittest.main()
