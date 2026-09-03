"""Unit tests for complete RealtimePipeline end-to-end orchestration."""

import numpy as np
import pytest
from src.pipeline.realtime_pipeline import RealtimePipeline


def test_realtime_pipeline_frame_processing():
    pipeline = RealtimePipeline()

    # Create dummy 720p frame
    dummy_frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    # Process 3 frames
    for i in range(3):
        res = pipeline.process_frame(dummy_frame, timestamp=float(i))
        assert res.frame_idx == i + 1
        assert res.annotated_frame.shape == (720, 1280, 3)
        assert isinstance(res.tracks, list)
        assert isinstance(res.poses, list)
        assert isinstance(res.assessments, list)
        assert isinstance(res.alerts, list)
        assert "fps" in res.telemetry
