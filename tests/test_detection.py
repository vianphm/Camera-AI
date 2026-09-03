"""Unit tests for detection data structures."""

import pytest
from src.detection.detector import Detection


def test_detection_properties():
    # Standing person bbox: x1=100, y1=100, x2=200, y2=400 (W=100, H=300)
    det = Detection(bbox=(100.0, 100.0, 200.0, 400.0), confidence=0.88, class_id=0)
    assert det.width == 100.0
    assert det.height == 300.0
    assert pytest.approx(det.aspect_ratio, 0.01) == 0.333
    assert det.center == (150.0, 250.0)
    assert det.bottom_center == (150.0, 400.0)


def test_lying_person_aspect_ratio():
    # Horizontal/lying person bbox: W=400, H=100
    det = Detection(bbox=(50.0, 200.0, 450.0, 300.0), confidence=0.92, class_id=0)
    assert det.width == 400.0
    assert det.height == 100.0
    assert pytest.approx(det.aspect_ratio, 0.01) == 4.0
