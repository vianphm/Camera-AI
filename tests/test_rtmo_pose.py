"""Unit tests for RTMOPoseEstimator class."""

import numpy as np
import pytest
from pathlib import Path
import cv2

from src.pose.rtmo_estimator import RTMOPoseEstimator
from src.tracking.tracker import Track


def test_rtmo_initialization():
    """Test session initialization, model existence, and active provider."""
    estimator = RTMOPoseEstimator()
    assert estimator.session is not None, "RTMO session must initialize successfully"
    assert estimator.active_provider in ("DmlExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider")
    assert estimator.input_name == "input"
    assert "dets" in estimator.output_names
    assert "keypoints" in estimator.output_names


def test_rtmo_preprocess():
    """Test letterboxing, aspect ratio retention, and tensor properties."""
    estimator = RTMOPoseEstimator()
    dummy = np.zeros((480, 800, 3), dtype=np.uint8)

    tensor, scale_ratio, pad_w, pad_h = estimator.preprocess(dummy)

    assert tensor.shape == (1, 3, 640, 640)
    assert tensor.dtype == np.float32
    assert tensor.flags.c_contiguous
    assert scale_ratio == 640.0 / 800.0  # 0.8
    assert pad_w == 0.0
    assert pad_h == (640.0 - 480.0 * 0.8) / 2.0  # 128.0


def test_rtmo_vectorized_postprocess():
    """Test vectorized unscaling logic on synthetic outputs."""
    estimator = RTMOPoseEstimator(conf_threshold=0.5)

    # 2 synthetic detections: 1 high conf, 1 low conf
    raw_dets = np.array([[[100.0, 100.0, 200.0, 300.0, 0.9], [50.0, 50.0, 150.0, 150.0, 0.2]]], dtype=np.float32)
    raw_kps = np.zeros((1, 2, 17, 3), dtype=np.float32)
    raw_kps[0, 0, 0] = [150.0, 120.0, 0.95]  # Nose of first detection

    dets, items = estimator.postprocess(
        dets_raw=raw_dets,
        kps_raw=raw_kps,
        scale_ratio=0.5,
        pad_w=20.0,
        pad_h=10.0,
        orig_w=1000,
        orig_h=800,
    )

    # Low conf detection (0.2) filtered out
    assert len(dets) == 1
    assert len(items) == 1

    # Unscaled coordinates: (x - pad_w) / scale_ratio = (100 - 20) / 0.5 = 160.0
    # (y - pad_h) / scale_ratio = (100 - 10) / 0.5 = 180.0
    assert pytest.approx(dets[0].bbox[0], 0.1) == 160.0
    assert pytest.approx(dets[0].bbox[1], 0.1) == 180.0
    assert pytest.approx(items[0].keypoints[0, 0], 0.1) == (150.0 - 20.0) / 0.5
    assert pytest.approx(items[0].keypoints[0, 1], 0.1) == (120.0 - 10.0) / 0.5


def test_rtmo_inference_on_real_image():
    """Test single-pass inference on bus.jpg."""
    bus_path = Path(".venv/Lib/site-packages/ultralytics/assets/bus.jpg")
    if not bus_path.exists():
        pytest.skip("bus.jpg not available in environment")

    img = cv2.imread(str(bus_path))
    estimator = RTMOPoseEstimator(conf_threshold=0.35)

    detections, items = estimator.detect_and_estimate_single_pass(img)

    assert len(detections) >= 3, f"Expected at least 3 persons in bus.jpg, found {len(detections)}"
    assert len(detections) == len(items)

    for item in items:
        assert item.keypoints.shape == (17, 3)
        assert item.confidence >= 0.35
        # All keypoint coordinates should be within image boundaries (1080, 810)
        assert np.all(item.keypoints[:, 0] >= 0.0)
        assert np.all(item.keypoints[:, 0] <= 810.0)
        assert np.all(item.keypoints[:, 1] >= 0.0)
        assert np.all(item.keypoints[:, 1] <= 1080.0)


def test_rtmo_track_matching():
    """Test strict IoU track matching."""
    estimator = RTMOPoseEstimator()

    track1 = Track(track_id=1, bbox=(100.0, 100.0, 200.0, 300.0), confidence=0.9, last_seen=0.0)
    track2 = Track(track_id=2, bbox=(400.0, 100.0, 500.0, 300.0), confidence=0.9, last_seen=0.0)

    from src.pose.pose_estimator import SinglePassDetectionItem
    item1 = SinglePassDetectionItem(bbox=(102.0, 101.0, 198.0, 299.0), confidence=0.9, keypoints=np.ones((17, 3)))
    item2 = SinglePassDetectionItem(bbox=(405.0, 98.0, 498.0, 302.0), confidence=0.88, keypoints=np.ones((17, 3)) * 2)

    matched = estimator.match_tracks_to_keypoints([track1, track2], [item1, item2], iou_threshold=0.70)
    assert 1 in matched
    assert 2 in matched
    assert np.all(matched[1] == 1.0)
    assert np.all(matched[2] == 2.0)
