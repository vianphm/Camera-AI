"""Anomaly detection base interfaces and result structures."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class AnomalyResult:
    """Represents the anomaly scoring outcome for a sequence."""
    anomaly_score: float         # Normalized [0.0..1.0] where 1.0 is maximally anomalous
    is_anomaly: bool             # Thresholded binary flag
    reconstruction_error: float  # Raw reconstruction MSE/L1 error
    baseline_threshold: float    # Decision threshold applied


class AnomalyDetector(ABC):
    """Abstract interface for behavior anomaly detectors."""

    @abstractmethod
    def score(self, sequence: np.ndarray) -> AnomalyResult:
        """Compute anomaly score for a normalized skeletal sequence (T, 17, 3).

        Args:
            sequence: Array of shape (T, 17, 3).

        Returns:
            AnomalyResult instance.
        """
        pass
