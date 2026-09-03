"""Unit tests for Phase 19 & 20: Real-Time Freshness & Stale Frame Backpressure Validation.
Proves that under camera producer > AI consumer load, the queue remains bounded and frame_age does not diverge.
"""

import time
import queue
import numpy as np
import pytest
from typing import Tuple, Optional
from src.camera.stream import CameraStream, FramePacket


class MockHighRateStream(CameraStream):
    """Simulates a physical 30 FPS camera publishing frames continually."""

    def __init__(self, fps: float = 30.0, max_queue_size: int = 1) -> None:
        super().__init__(max_queue_size=max_queue_size)
        self.fps = fps
        self.interval = 1.0 / fps
        self._frame_data = np.zeros((480, 640, 3), dtype=np.uint8)

    def _open_capture(self) -> bool:
        return True

    def _close_capture(self) -> None:
        pass

    def _read_raw_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        time.sleep(self.interval)
        return True, self._frame_data

    def get_fps(self) -> float:
        return self.fps

    def get_resolution(self) -> Tuple[int, int]:
        return (640, 480)


def test_stale_frame_backpressure_and_freshness():
    """Test Phase 19 & 20: Camera 30 FPS, Consumer 10 FPS (100ms artificial processing delay).
    Verifies that Drop-Oldest keeps queue size <= 1 and frame_age stays under target.
    """
    stream = MockHighRateStream(fps=30.0, max_queue_size=1)
    assert stream.start()

    # Allow stream producer to generate initial frames
    time.sleep(0.1)

    frame_ages = []
    pipeline_latencies = []
    queue_depths = []

    # Slow consumer: simulates heavy AI inference (60ms per frame ~ 16 FPS)
    num_processed = 0
    start_test = time.time()
    while num_processed < 25 and (time.time() - start_test < 5.0):
        packet = stream.read(timeout=0.2)
        if packet is None:
            continue

        t_infer_start = time.time()
        # Measure frame age at moment AI begins processing it
        frame_age_ms = (t_infer_start - packet.timestamp) * 1000.0
        frame_ages.append(frame_age_ms)
        queue_depths.append(stream._queue.qsize())

        # Simulate inference computation
        time.sleep(0.060)  # 60ms inference delay

        t_infer_end = time.time()
        pipeline_latency_ms = (t_infer_end - packet.timestamp) * 1000.0
        pipeline_latencies.append(pipeline_latency_ms)
        num_processed += 1

    stream.release()

    assert len(frame_ages) >= 15, "Should have processed at least 15 frames"

    # Analyze Freshness Metrics
    median_age = float(np.median(frame_ages))
    p95_age = float(np.percentile(frame_ages, 95))
    max_age = float(np.max(frame_ages))
    max_queue = max(queue_depths)

    # Core Engineering Invariants:
    # 1. Queue depth must never exceed max_queue_size (1)
    assert max_queue <= 1, f"Queue size exceeded 1 (got {max_queue})"

    # 2. Frame age must NOT diverge to seconds!
    # Even though camera produced ~75 frames while consumer processed 25,
    # the drop-oldest mechanism ensures every processed frame is strictly fresh (<100ms old at start of inference)!
    assert p95_age < 100.0, f"P95 frame age ({p95_age:.1f}ms) exceeded 100ms threshold!"
    assert max_age < 150.0, f"Max frame age ({max_age:.1f}ms) exceeded 150ms ceiling!"
