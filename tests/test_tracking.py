"""Unit tests for Multi-Object Tracking with ByteTrackManager."""

import pytest
from src.detection.detector import Detection
from src.tracking.track_manager import ByteTrackManager, bbox_iou


def test_bbox_iou():
    box1 = (0.0, 0.0, 10.0, 10.0)
    box2 = (0.0, 0.0, 10.0, 10.0)
    assert bbox_iou(box1, box2) == 1.0

    box3 = (10.0, 10.0, 20.0, 20.0)
    assert bbox_iou(box1, box3) == 0.0

    box4 = (5.0, 0.0, 15.0, 10.0)
    # intersection: (5..10, 0..10) -> area 50, union 150 -> iou 1/3
    assert pytest.approx(bbox_iou(box1, box4), 0.01) == 0.333


def test_bytetrack_manager_lifecycle():
    tracker = ByteTrackManager(track_high_thresh=0.5, new_track_thresh=0.6)
    tracker.reset()

    # Frame 1: Detection appears
    dets_f1 = [Detection(bbox=(100.0, 100.0, 150.0, 300.0), confidence=0.90)]
    tracks_f1 = tracker.update(dets_f1, timestamp=1.0)
    assert len(tracks_f1) == 1
    tid = tracks_f1[0].track_id

    # Frame 2: Person moves slightly, score drops to 0.45 (low conf tier)
    dets_f2 = [Detection(bbox=(102.0, 103.0, 152.0, 303.0), confidence=0.45)]
    tracks_f2 = tracker.update(dets_f2, timestamp=1.1)
    assert len(tracks_f2) == 1
    # ID must be preserved via low-score association
    assert tracks_f2[0].track_id == tid
    assert len(tracks_f2[0].trajectory) == 2
