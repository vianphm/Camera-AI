"""Person detection module."""

from src.detection.detector import Detection, PersonDetector
from src.detection.person_detector import YOLOv8PersonDetector

__all__ = ["Detection", "PersonDetector", "YOLOv8PersonDetector"]
