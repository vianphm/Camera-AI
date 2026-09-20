"""Unit tests for temporal sequence buffer, kinematics, and PyTorch temporal models."""

import numpy as np
import pytest

try:
    import torch
    from src.temporal.temporal_model import SpatialTemporalTransformer, TCNSequenceClassifier
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    SpatialTemporalTransformer = None
    TCNSequenceClassifier = None
    TORCH_AVAILABLE = False

from src.temporal.sequence_buffer import SequenceBuffer, TrackSnapshot
from src.temporal.temporal_features import TemporalFeatureExtractor


def test_sequence_buffer_sliding():
    buf = SequenceBuffer(window_size=10)
    assert not buf.is_ready(track_id=1)

    # Push 12 snapshots
    for i in range(12):
        snap = TrackSnapshot(
            timestamp=float(i),
            bbox=(10.0, 10.0, 20.0, 40.0),
            keypoints=np.zeros((17, 3), dtype=np.float32),
            normalized_keypoints=np.ones((17, 3), dtype=np.float32),
            torso_angle=10.0,
        )
        buf.add_snapshot(1, snap)

    assert buf.is_ready(track_id=1)
    seq = buf.get_normalized_sequence(track_id=1)
    assert seq.shape == (10, 17, 3)


def test_temporal_feature_extractor():
    extractor = TemporalFeatureExtractor(fps=15.0)
    snapshots = []
    # Simulate a sudden downward fall: y position increases rapidly
    for i in range(5):
        snap = TrackSnapshot(
            timestamp=i * (1.0 / 15.0),
            bbox=(10.0, 10.0 + i * 40.0, 50.0, 100.0 + i * 40.0),
            keypoints=np.zeros((17, 3), dtype=np.float32),
            normalized_keypoints=np.zeros((17, 3), dtype=np.float32),
            torso_angle=float(i * 20.0),  # Rotates towards horizontal
        )
        snapshots.append(snap)

    kin = extractor.extract(snapshots)
    assert kin.vertical_velocity > 0.0
    assert kin.drop_severity_score > 0.2


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed in lightweight ONNX runtime")
def test_spatial_temporal_transformer_forward():
    model = SpatialTemporalTransformer(input_dim=51, num_classes=10, d_model=64, nhead=2, num_layers=2)
    # Batch=2, Time=30, Keypoints=17, Channels=3
    dummy_x = torch.randn(2, 30, 17, 3)
    out = model(dummy_x)
    assert out.shape == (2, 10)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed in lightweight ONNX runtime")
def test_tcn_classifier_forward():
    model = TCNSequenceClassifier(input_dim=51, num_classes=10, num_channels=[32, 64])
    dummy_x = torch.randn(2, 30, 17, 3)
    out = model(dummy_x)
    assert out.shape == (2, 10)
