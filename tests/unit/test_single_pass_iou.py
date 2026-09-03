"""Unit tests for Single-Pass IoU Matching and Translation Offset Compensation."""

import numpy as np
import pytest

from src.pose.pose_estimator import (
    SinglePassDetectionItem,
    compute_bbox_iou,
    translate_keypoints,
    YOLOv8PoseEstimator,
)
from src.tracking.tracker import Track


def test_compute_bbox_iou():
    box1 = (100.0, 100.0, 200.0, 300.0)  # Area = 100 * 200 = 20000
    box2 = (100.0, 100.0, 200.0, 300.0)  # Exact match
    assert pytest.approx(compute_bbox_iou(box1, box2), 0.001) == 1.0

    box3 = (300.0, 300.0, 400.0, 400.0)  # Disjoint
    assert compute_bbox_iou(box1, box3) == 0.0

    # Partial overlap
    box4 = (150.0, 100.0, 250.0, 300.0)  # Overlap = 50 * 200 = 10000; Union = 30000 -> IoU = 0.333
    assert pytest.approx(compute_bbox_iou(box1, box4), 0.01) == 0.333


def test_iou_matching_two_people_close_no_cross_assignment():
    """Vulnerability 1 check: When 2 people are close or occluded, keypoints must NOT be swapped."""
    # Person A: at (100, 100, 200, 400)
    # Person B: standing right next to Person A at (190, 100, 290, 400)
    track_A = Track(track_id=1, bbox=(100.0, 100.0, 200.0, 400.0), confidence=0.9, last_seen=1.0)
    track_B = Track(track_id=2, bbox=(190.0, 100.0, 290.0, 400.0), confidence=0.9, last_seen=1.0)

    kp_A = np.full((17, 3), 10.0, dtype=np.float32)  # Unique signature for Person A
    kp_B = np.full((17, 3), 90.0, dtype=np.float32)  # Unique signature for Person B

    # YOLO-Pose outputs detections with IoU >= 0.70 to corresponding tracks
    det_A = SinglePassDetectionItem(bbox=(102.0, 100.0, 202.0, 400.0), confidence=0.88, keypoints=kp_A)
    det_B = SinglePassDetectionItem(bbox=(192.0, 100.0, 292.0, 400.0), confidence=0.85, keypoints=kp_B)

    matched = YOLOv8PoseEstimator.match_tracks_to_keypoints(
        active_tracks=[track_A, track_B],
        items=[det_A, det_B],
        iou_threshold=0.70,
    )

    assert 1 in matched
    assert 2 in matched
    # Verify Person A got kp_A, NOT kp_B
    assert np.all(matched[1] == 10.0)
    assert np.all(matched[2] == 90.0)


def test_iou_matching_rejects_distant_person_on_occlusion():
    """When Person B is temporarily occluded, do NOT assign Person A's keypoints to Track B."""
    track_A = Track(track_id=1, bbox=(100.0, 100.0, 200.0, 400.0), confidence=0.9, last_seen=1.0)
    # Track B was extrapolated by Kalman filter to (180, 100, 280, 400), but detection only found Person A
    track_B = Track(track_id=2, bbox=(180.0, 100.0, 280.0, 400.0), confidence=0.4, last_seen=1.0)

    kp_A = np.full((17, 3), 15.0, dtype=np.float32)
    det_A = SinglePassDetectionItem(bbox=(100.0, 100.0, 200.0, 400.0), confidence=0.92, keypoints=kp_A)

    matched = YOLOv8PoseEstimator.match_tracks_to_keypoints(
        active_tracks=[track_A, track_B],
        items=[det_A],
        iou_threshold=0.70,
    )

    assert 1 in matched
    assert np.all(matched[1] == 15.0)
    # Track B must NOT have been assigned Person A's keypoints!
    assert 2 not in matched


def test_translation_offset_compensation():
    """Vulnerability 3 check: Translation offset smoothly moves skeleton with Kalman-moved BBox."""
    prev_bbox = (100.0, 100.0, 200.0, 400.0)  # Center: (150, 250)
    curr_bbox = (120.0, 110.0, 220.0, 410.0)  # Center: (170, 260) -> dx = +20, dy = +10

    prev_kp = np.zeros((17, 3), dtype=np.float32)
    prev_kp[0] = [150.0, 120.0, 0.90]  # Nose
    prev_kp[11] = [140.0, 260.0, 0.90] # Left Hip

    compensated_kp = translate_keypoints(prev_kp, prev_bbox, curr_bbox, confidence_decay=0.98)

    # Nose should move by dx=+20, dy=+10
    assert pytest.approx(compensated_kp[0, 0], 0.01) == 170.0
    assert pytest.approx(compensated_kp[0, 1], 0.01) == 130.0
    assert pytest.approx(compensated_kp[0, 2], 0.01) == 0.90 * 0.98

    # Hip should move by dx=+20, dy=+10
    assert pytest.approx(compensated_kp[11, 0], 0.01) == 160.0
    assert pytest.approx(compensated_kp[11, 1], 0.01) == 270.0
