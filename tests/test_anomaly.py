"""Unit tests for unsupervised pose sequence autoencoder and anomaly scoring."""

import numpy as np
import pytest

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False

from src.anomaly.anomaly_score import PoseSequenceAutoencoder, ReconstructionAnomalyScorer


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed in lightweight ONNX runtime")
def test_pose_autoencoder_reconstruction_shape():
    model = PoseSequenceAutoencoder(input_dim=51, latent_dim=32)
    dummy_x = torch.randn(2, 30, 17, 3)
    out = model(dummy_x)
    assert out.shape == (2, 30, 17, 3)


def test_reconstruction_anomaly_scorer():
    scorer = ReconstructionAnomalyScorer()
    dummy_seq = np.zeros((30, 17, 3), dtype=np.float32)
    res = scorer.score(dummy_seq)
    assert 0.0 <= res.anomaly_score <= 1.0
    assert isinstance(res.is_anomaly, bool)
