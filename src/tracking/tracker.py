"""Tracking data structures and base interfaces."""

from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from typing import List, Tuple, Deque, Optional
from src.detection.detector import Detection


@dataclass
class Track:
    """Represents a tracked person instance with trajectory and bounding box history."""
    track_id: int
    bbox: Tuple[float, float, float, float]
    confidence: float
    last_seen: float
    bbox_history: Deque[Tuple[float, float, float, float]] = field(default_factory=lambda: deque(maxlen=120))
    trajectory: Deque[Tuple[float, float]] = field(default_factory=lambda: deque(maxlen=120))
    state: str = "tracked"  # "tracked", "lost", "removed"
    standing_height_baseline: Optional[float] = None
    age: int = 0
    time_since_update: int = 0

    @property
    def width(self) -> float:
        return max(0.0, self.bbox[2] - self.bbox[0])

    @property
    def height(self) -> float:
        return max(0.0, self.bbox[3] - self.bbox[1])

    @property
    def aspect_ratio(self) -> float:
        h = self.height
        return (self.width / h) if h > 1e-4 else 1.0

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.bbox[0] + self.bbox[2]) / 2.0, (self.bbox[1] + self.bbox[3]) / 2.0)

    @property
    def bottom_center(self) -> Tuple[float, float]:
        """Ground contact point (cx, y2)."""
        return ((self.bbox[0] + self.bbox[2]) / 2.0, self.bbox[3])


class Tracker(ABC):
    """Abstract interface for Multi-Object Trackers."""

    @abstractmethod
    def update(self, detections: List[Detection], timestamp: float) -> List[Track]:
        """Update tracker state with new frame detections.

        Args:
            detections: Detected person instances in current frame.
            timestamp: Acquisition timestamp in seconds.

        Returns:
            List of confirmed active Tracks.
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset internal tracker state and IDs."""
        pass
