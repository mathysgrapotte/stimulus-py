"""Utility module for computing various performance metrics for machine learning models."""

from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from scipy.stats import spearmanr
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

# Constants for threshold and number of classes
BINARY_THRESHOLD = 0.5
BINARY_CLASS_COUNT = 2


class Performance:
    """Returns the value of a given metric.

    Parameters
    ----------
    labels (np.array) : labels
    predictions (np.array) : predictions
    metric (str) : the metric to compute

    Returns:
    -------
    value (float) : the value of the metric

    TODO we can add more metrics here

    TODO currently for classification  metrics like precision, recall, f1score and mcc,
    we are using a threshold of 0.5 to convert the probabilities to binary predictions.
    However for models with imbalanced predictions, where the meaningful threshold is not
    located at 0.5, one can end up with full of 0s or 1s, and thus meaningless performance
    metrics.
    """

    def __init__(self, labels: Any, predictions: Any, metric: str = "rocauc") -> None:
        """Initialize Performance class with labels, predictions and metric type.

        Args:
            labels: Ground truth labels
            predictions: Model predictions
            metric: Type of metric to compute (default: "rocauc")
        """
        labels_arr = self.data2array(labels)
        predictions_arr = self.data2array(predictions)
        labels_arr, predictions_arr = self.handle_multiclass(labels_arr, predictions_arr)
        if labels_arr.shape != predictions_arr.shape:
            raise ValueError(
                f"The labels have shape {labels_arr.shape} whereas predictions have shape {predictions_arr.shape}.",
            )
        function = getattr(self, metric)
        self.val = function(labels_arr, predictions_arr)

    def data2array(self, data: Any) -> NDArray[np.float64]:
        """Convert input data to numpy array.

        Args:
            data: Input data in various formats

        Returns:
            NDArray[np.float64]: Converted numpy array

        Raises:
            ValueError: If input data type is not supported
        """
        if isinstance(data, list):
            return np.array(data, dtype=np.float64)
        if isinstance(data, np.ndarray):
            return data.astype(np.float64)
        if isinstance(data, torch.Tensor):
            return data.detach().cpu().numpy().astype(np.float64)
        if isinstance(data, (int, float)):
            return np.array([data], dtype=np.float64)
        raise ValueError(f"The data must be a list, np.array, torch.Tensor, int or float. Instead it is {type(data)}")

    def handle_multiclass(
        self,
        labels: NDArray[np.float64],
        predictions: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Handle the case of multiclass classification.

        TODO currently only two class predictions are handled. Needs to handle the other scenarios.
        """
        # if only one columns for labels and predictions
        if (len(labels.shape) == 1) and (len(predictions.shape) == 1):
            return labels, predictions

        # if one columns for labels, but two columns for predictions
        if (len(labels.shape) == 1) and (predictions.shape[1] == BINARY_CLASS_COUNT):
            predictions = predictions[:, 1]  # assumes the second column is the positive class
            return labels, predictions

        # other scenarios not implemented yet
        raise ValueError(f"Labels have shape {labels.shape} and predictions have shape {predictions.shape}.")

    def rocauc(self, labels: NDArray[np.float64], predictions: NDArray[np.float64]) -> float:
        """Compute ROC AUC score."""
        return float(roc_auc_score(labels, predictions))

    def prauc(self, labels: NDArray[np.float64], predictions: NDArray[np.float64]) -> float:
        """Compute PR AUC score."""
        return float(average_precision_score(labels, predictions))

    def mcc(self, labels: NDArray[np.float64], predictions: NDArray[np.float64]) -> float:
        """Compute Matthews Correlation Coefficient."""
        predictions_binary = np.array([1 if p > BINARY_THRESHOLD else 0 for p in predictions])
        return float(matthews_corrcoef(labels, predictions_binary))

    def f1score(self, labels: NDArray[np.float64], predictions: NDArray[np.float64]) -> float:
        """Compute F1 score."""
        predictions_binary = np.array([1 if p > BINARY_THRESHOLD else 0 for p in predictions])
        return float(f1_score(labels, predictions_binary))

    def precision(self, labels: NDArray[np.float64], predictions: NDArray[np.float64]) -> float:
        """Compute precision score."""
        predictions_binary = np.array([1 if p > BINARY_THRESHOLD else 0 for p in predictions])
        return float(precision_score(labels, predictions_binary))

    def recall(self, labels: NDArray[np.float64], predictions: NDArray[np.float64]) -> float:
        """Compute recall score."""
        predictions_binary = np.array([1 if p > BINARY_THRESHOLD else 0 for p in predictions])
        return float(recall_score(labels, predictions_binary))

    def spearmanr(self, labels: NDArray[np.float64], predictions: NDArray[np.float64]) -> float:
        """Compute Spearman correlation coefficient."""
        return float(spearmanr(labels, predictions)[0])
