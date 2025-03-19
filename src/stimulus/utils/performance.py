"""Utility module for computing various performance metrics for machine learning models."""

import logging
from typing import Any, ClassVar

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

logger = logging.getLogger(__name__)

# Constants for binary classification
BINARY_THRESHOLD = 0.5

# Constants for prediction thresholds
NEAR_ZERO_THRESHOLD = 0.1
NEAR_HALF_LOWER_THRESHOLD = 0.4
NEAR_HALF_UPPER_THRESHOLD = 0.6
NEAR_ONE_THRESHOLD = 0.9

# Constant for minimum class diversity
MIN_CLASS_COUNT = 2

# Constants for array dimensions
DIMENSION_2D = 2
BINARY_CLASS_COUNT = 2


class Performance:
    """Class to compute performance metrics on predictions."""

    # Map metric names to methods
    METRIC_MAP: ClassVar[dict[str, str]] = {
        "rocauc": "rocauc",
        "prauc": "prauc",
        "f1score": "f1score",
        "mcc": "mcc",
        "precision": "precision",
        "recall": "recall",
        "spearmanr": "spearmanr",
    }

    def __init__(self, labels: Any, predictions: Any, metric: str = "rocauc") -> None:
        """Initialize the Performance class.

        Args:
            labels: True labels
            predictions: Model predictions
            metric: Metric to compute (default: "rocauc")

        Raises:
            ValueError: If the metric is not supported
        """
        self.metric_name = metric
        if metric not in self.METRIC_MAP:
            raise ValueError(f"Metric {metric} not supported")

        # Basic input diagnostics
        logger.debug("Initializing Performance class")
        if isinstance(labels, (list, np.ndarray, torch.Tensor)):
            logger.info(f"Raw labels shape: {np.array(labels).shape}")
            logger.info(f"Raw labels unique values: {np.unique(np.array(labels), return_counts=True)}")
        if isinstance(predictions, (list, np.ndarray, torch.Tensor)):
            logger.info(f"Raw predictions shape: {np.array(predictions).shape}")
            pred_array = np.array(predictions)
            logger.info(
                f"Raw predictions min: {np.min(pred_array):.4f}, max: {np.max(pred_array):.4f}, mean: {np.mean(pred_array):.4f}",
            )
            # Log count of predictions near boundaries
            logger.info(f"Predictions near 0 (< {NEAR_ZERO_THRESHOLD}): {np.sum(pred_array < NEAR_ZERO_THRESHOLD)}")
            logger.info(
                f"Predictions near 0.5 ({NEAR_HALF_LOWER_THRESHOLD}-{NEAR_HALF_UPPER_THRESHOLD}): {np.sum((pred_array >= NEAR_HALF_LOWER_THRESHOLD) & (pred_array <= NEAR_HALF_UPPER_THRESHOLD))}",
            )
            logger.info(f"Predictions near 1 (> {NEAR_ONE_THRESHOLD}): {np.sum(pred_array > NEAR_ONE_THRESHOLD)}")

        labels_arr = self.data2array(labels)
        predictions_arr = self.data2array(predictions)
        labels_arr, predictions_arr = self.handle_multiclass(labels_arr, predictions_arr)
        if labels_arr.shape != predictions_arr.shape:
            raise ValueError(
                f"Labels and predictions must have the same shape, got {labels_arr.shape} and {predictions_arr.shape}",
            )

        # Compute the metric
        metric_fn = getattr(self, self.METRIC_MAP[metric])
        self.value = metric_fn(labels_arr, predictions_arr)
        # For backward compatibility
        self.val = self.value
        logger.info(f"Metric {metric}: {self.value}")

    def data2array(self, data: Any) -> NDArray[np.float64]:
        """Convert data to numpy array.

        Args:
            data: Data to convert

        Returns:
            NDArray: Converted data as numpy array
        """
        if data is None:
            raise ValueError("Data cannot be None")

        # Convert torch tensor to numpy
        if isinstance(data, torch.Tensor):
            return data.detach().cpu().numpy()

        # Convert list to numpy array
        if isinstance(data, list):
            data = np.array(data)

        # Make sure data is a numpy array
        if not isinstance(data, np.ndarray):
            raise TypeError(f"Expected data to be a numpy array, got {type(data)}")

        # Force float64 for consistency
        return data.astype(np.float64)

    def handle_multiclass(
        self,
        labels: NDArray[np.float64],
        predictions: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Handle multi-class predictions.

        If predictions are 2D, assumed to be multi-class probabilities,
        convert to binary prediction based on argmax.

        Args:
            labels: True labels
            predictions: Predicted probabilities

        Returns:
            Tuple of processed labels and predictions
        """
        # If labels have shape (N, 1), squeeze to (N,)
        if len(labels.shape) == DIMENSION_2D and labels.shape[1] == 1:
            labels = labels.squeeze(axis=1)

        # Handle multi-class case
        if len(predictions.shape) == DIMENSION_2D and predictions.shape[1] > 1:
            logger.info(f"Multi-class predictions detected with {predictions.shape[1]} classes")

            # If labels are one-hot encoded, convert to class indices
            if len(labels.shape) == DIMENSION_2D and labels.shape[1] > 1:
                logger.info("Converting one-hot encoded labels to class indices")
                labels = np.argmax(labels, axis=1)

            # Convert probabilities to binary prediction based on argmax
            predictions = (
                predictions[:, 1] if predictions.shape[1] == BINARY_CLASS_COUNT else np.argmax(predictions, axis=1)
            )

        return labels, predictions

    def rocauc(self, labels: NDArray[np.float64], predictions: NDArray[np.float64]) -> float:
        """Compute ROC AUC score.

        If computation fails (e.g., only one class present in labels),
        return 0.5 (random classifier performance).
        """
        try:
            if len(np.unique(labels)) < MIN_CLASS_COUNT:
                logger.info("Only one class present in labels for ROC AUC. Returning 0.5.")
                return 0.5
            return float(roc_auc_score(labels, predictions))
        except (ValueError, TypeError) as e:
            logger.info(f"ROC AUC calculation failed: {e!s}. Returning 0.5.")
            return 0.5

    def prauc(self, labels: NDArray[np.float64], predictions: NDArray[np.float64]) -> float:
        """Compute PR AUC score.

        If computation fails, return 0.0 (worst case).
        """
        try:
            if len(np.unique(labels)) < MIN_CLASS_COUNT:
                logger.info("Only one class present in labels for PR AUC. Returning 0.0.")
                return 0.0
            return float(average_precision_score(labels, predictions))
        except (ValueError, TypeError) as e:
            logger.info(f"PR AUC calculation failed: {e!s}. Returning 0.0.")
            return 0.0

    def mcc(self, labels: NDArray[np.float64], predictions: NDArray[np.float64]) -> float:
        """Compute Matthews Correlation Coefficient.

        If computation fails, return 0.0 (worst case).
        """
        try:
            predictions_binary = np.array([1 if p > BINARY_THRESHOLD else 0 for p in predictions])
            if len(np.unique(labels)) < MIN_CLASS_COUNT or len(np.unique(predictions_binary)) < MIN_CLASS_COUNT:
                logger.info("Not enough class diversity for MCC. Returning 0.0.")
                return 0.0
            return float(matthews_corrcoef(labels, predictions_binary))
        except (ValueError, TypeError) as e:
            logger.info(f"MCC calculation failed: {e!s}. Returning 0.0.")
            return 0.0

    def f1score(self, labels: NDArray[np.float64], predictions: NDArray[np.float64]) -> float:
        """Compute F1 score.

        If computation fails, return 0.0 (worst case).
        """
        try:
            predictions_binary = np.array([1 if p > BINARY_THRESHOLD else 0 for p in predictions])
            if len(np.unique(labels)) < MIN_CLASS_COUNT:
                logger.info("Only one class present in labels for F1. Returning 0.0.")
                return 0.0
            return float(f1_score(labels, predictions_binary))
        except (ValueError, TypeError) as e:
            logger.info(f"F1 score calculation failed: {e!s}. Returning 0.0.")
            return 0.0

    def precision(self, labels: NDArray[np.float64], predictions: NDArray[np.float64]) -> float:
        """Compute precision score.

        If computation fails, return 0.0 (worst case).
        """
        try:
            predictions_binary = np.array([1 if p > BINARY_THRESHOLD else 0 for p in predictions])
            if len(np.unique(labels)) < MIN_CLASS_COUNT:
                logger.info("Only one class present in labels for precision. Returning 0.0.")
                return 0.0
            return float(precision_score(labels, predictions_binary, zero_division=0))
        except (ValueError, TypeError) as e:
            logger.info(f"Precision calculation failed: {e!s}. Returning 0.0.")
            return 0.0

    def recall(self, labels: NDArray[np.float64], predictions: NDArray[np.float64]) -> float:
        """Compute recall score.

        If computation fails, return 0.0 (worst case).
        """
        try:
            predictions_binary = np.array([1 if p > BINARY_THRESHOLD else 0 for p in predictions])
            if len(np.unique(labels)) < MIN_CLASS_COUNT:
                logger.info("Only one class present in labels for recall. Returning 0.0.")
                return 0.0
            return float(recall_score(labels, predictions_binary, zero_division=0))
        except (ValueError, TypeError) as e:
            logger.info(f"Recall calculation failed: {e!s}. Returning 0.0.")
            return 0.0

    def spearmanr(self, labels: NDArray[np.float64], predictions: NDArray[np.float64]) -> float:
        """Compute Spearman correlation coefficient.

        If computation fails or values are constant, return 0.0.
        """
        try:
            if np.allclose(labels, labels[0]) or np.allclose(predictions, predictions[0]):
                logger.info("Labels or predictions are constant. Returning 0.0.")
                return 0.0
            result = spearmanr(labels, predictions)[0]
            if np.isnan(result):
                logger.info("Spearman correlation returned NaN. Returning 0.0.")
                return 0.0
            return float(result)
        except (ValueError, TypeError) as e:
            logger.info(f"Spearman correlation calculation failed: {e!s}. Returning 0.0.")
            return 0.0
