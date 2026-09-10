"""Unit and performance verification tests for RTMO-s TensorRT INT8 Engine."""

from pathlib import Path
import cv2
import numpy as np
import pytest

from src.pose.rtmo_estimator import RTMOPoseEstimator
from src.tracking.tracker import Track

ENGINE_PATH = Path("models/pose/rtmo-s_int8.engine")
BUS_IMG_PATH = Path(".venv/Lib/site-packages/ultralytics/assets/bus.jpg")


@pytest.mark.skipif(not ENGINE_PATH.exists(), reason="rtmo-s_int8.engine not found")
def test_int8_engine_file_properties():
    """Verify compiled TensorRT INT8 engine exists and has optimal file size."""
    size_mb = ENGINE_PATH.stat().st_size / (1024 * 1024)
    # INT8 engine should be ~33 MB (much smaller than 121 MB FP32 engine)
    assert 20.0 < size_mb < 50.0, f"Expected INT8 engine size ~33MB, got {size_mb:.2f}MB"


@pytest.mark.skipif(not ENGINE_PATH.exists(), reason="rtmo-s_int8.engine not found")
def test_rtmo_int8_initialization():
    """Verify RTMOPoseEstimator initializes correctly with TensorRT INT8 engine."""
    estimator = RTMOPoseEstimator(model_path=ENGINE_PATH)
    assert estimator.is_tensorrt is True, "Estimator should be running in TensorRT mode"
    assert estimator.active_provider == "TensorRT"
    assert estimator.trt_engine is not None
    assert estimator.trt_context is not None


@pytest.mark.skipif(not ENGINE_PATH.exists() or not BUS_IMG_PATH.exists(), reason="Engine or test image not found")
def test_rtmo_int8_inference_fidelity():
    """Verify single-pass detection accuracy and keypoint coordinates using INT8 engine."""
    img = cv2.imread(str(BUS_IMG_PATH))
    assert img is not None

    estimator = RTMOPoseEstimator(model_path=ENGINE_PATH, conf_threshold=0.35)
    detections, items = estimator.detect_and_estimate_single_pass(img)

    # Should detect at least 3-4 persons on bus.jpg
    assert len(detections) >= 3, f"Expected >= 3 detections, got {len(detections)}"
    assert len(detections) == len(items)

    orig_h, orig_w = img.shape[:2]
    for item in items:
        assert item.confidence >= 0.35
        assert item.keypoints.shape == (17, 3)
        # All coordinates within image frame
        assert np.all(item.keypoints[:, 0] >= 0.0)
        assert np.all(item.keypoints[:, 0] <= orig_w)
        assert np.all(item.keypoints[:, 1] >= 0.0)
        assert np.all(item.keypoints[:, 1] <= orig_h)


@pytest.mark.skipif(not ENGINE_PATH.exists() or not BUS_IMG_PATH.exists(), reason="Engine or test image not found")
def test_rtmo_int8_tracking_estimate():
    """Verify full estimate() interface with ByteTrack tracks using INT8 engine."""
    img = cv2.imread(str(BUS_IMG_PATH))
    estimator = RTMOPoseEstimator(model_path=ENGINE_PATH, conf_threshold=0.35)

    # First get ground truth detection bboxes
    detections, _ = estimator.detect_and_estimate_single_pass(img)
    assert len(detections) >= 1

    # Create mock tracks matching the first detection
    d0 = detections[0]
    mock_track = Track(
        track_id=42,
        bbox=d0.bbox,
        confidence=d0.confidence,
        last_seen=0.0,
    )

    pose_results = estimator.estimate(img, [mock_track])
    assert len(pose_results) == 1
    assert pose_results[0].track_id == 42
    assert pose_results[0].keypoints.shape == (17, 3)
    assert pose_results[0].normalized_keypoints.shape == (17, 3)
