"""Detection data structures and base interface."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np


@dataclass
class Detection:
    """Represents a detected person bounding box."""
    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2)
    confidence: float
    class_id: int = 0  # 0 corresponds to 'person' in standard COCO models

    @property
    def width(self) -> float:
        return max(0.0, self.bbox[2] - self.bbox[0])

    @property
    def height(self) -> float:
        return max(0.0, self.bbox[3] - self.bbox[1])

    @property
    def aspect_ratio(self) -> float:
        """Width / Height ratio. High values (>1.2) suggest horizontal/lying orientation."""
        h = self.height
        return (self.width / h) if h > 1e-4 else 1.0

    @property
    def center(self) -> Tuple[float, float]:
        """Center coordinate (cx, cy)."""
        return ((self.bbox[0] + self.bbox[2]) / 2.0, (self.bbox[1] + self.bbox[3]) / 2.0)

    @property
    def bottom_center(self) -> Tuple[float, float]:
        """Foot / contact point with ground plane."""
        return ((self.bbox[0] + self.bbox[2]) / 2.0, self.bbox[3])


class PersonDetector(ABC):
    """Abstract interface for Person Detectors."""

    @abstractmethod
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Detect persons in the given BGR image frame.

        Args:
            frame: Input numpy array (H, W, 3) in BGR order.

        Returns:
            List of Detection objects filtered for class 'person'.
        """
        pass
