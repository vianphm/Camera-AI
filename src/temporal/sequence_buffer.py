"""Sliding window sequence buffer for tracking temporal skeletal history."""

from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple
import numpy as np


@dataclass
class TrackSnapshot:
    """A single frame temporal snapshot for a tracked person."""
    timestamp: float
    bbox: Tuple[float, float, float, float]
    keypoints: np.ndarray             # (17, 3) raw image coordinates
    normalized_keypoints: np.ndarray  # (17, 3) root-centered & normalized
    torso_angle: float               # [0..90] degrees


class SequenceBuffer:
    """Manages sliding window history of skeletal poses per tracked identity."""

    def __init__(self, window_size: int = 30, max_inactive_frames: int = 45) -> None:
        """Initialize SequenceBuffer.

        Args:
            window_size: Number of frames per temporal window (e.g. 30 frames ~ 2s at 15 FPS).
            max_inactive_frames: Maximum frames to retain buffer for an unobserved track.
        """
        self.window_size = window_size
        self.max_inactive_frames = max_inactive_frames
        self._buffers: Dict[int, deque[TrackSnapshot]] = {}
        self._inactive_counts: Dict[int, int] = {}

    def add_snapshot(self, track_id: int, snapshot: TrackSnapshot) -> None:
        """Add a new snapshot for the given track_id."""
        if track_id not in self._buffers:
            self._buffers[track_id] = deque(maxlen=self.window_size)
        self._buffers[track_id].append(snapshot)
        self._inactive_counts[track_id] = 0

    def is_ready(self, track_id: int) -> bool:
        """Check if buffer has accumulated enough frames to perform temporal inference."""
        if track_id not in self._buffers:
            return False
        # Ready if we have at least half of the window size
        return len(self._buffers[track_id]) >= max(10, self.window_size // 2)

    def get_normalized_sequence(self, track_id: int) -> Optional[np.ndarray]:
        """Get padded/interpolated normalized keypoint sequence array of shape (T, 17, 3)."""
        if track_id not in self._buffers or len(self._buffers[track_id]) == 0:
            return None

        snapshots = list(self._buffers[track_id])
        num_present = len(snapshots)

        # Build raw array (N, 17, 3)
        arr = np.array([s.normalized_keypoints for s in snapshots], dtype=np.float32)

        # If shorter than window_size, pad beginning with first frame
        if num_present < self.window_size:
            pad_len = self.window_size - num_present
            first_frame = np.repeat(arr[0:1], pad_len, axis=0)
            arr = np.concatenate([first_frame, arr], axis=0)
        elif num_present > self.window_size:
            arr = arr[-self.window_size:]

        return arr

    def get_snapshots(self, track_id: int) -> List[TrackSnapshot]:
        """Retrieve recent snapshot history for kinematic feature extraction."""
        if track_id not in self._buffers:
            return []
        return list(self._buffers[track_id])

    def update_activity(self, active_track_ids: Set[int]) -> None:
        """Mark inactive tracks and remove stale buffers."""
        stale_ids = []
        for tid in list(self._buffers.keys()):
            if tid not in active_track_ids:
                self._inactive_counts[tid] = self._inactive_counts.get(tid, 0) + 1
                if self._inactive_counts[tid] > self.max_inactive_frames:
                    stale_ids.append(tid)
            else:
                self._inactive_counts[tid] = 0

        for tid in stale_ids:
            self._buffers.pop(tid, None)
            self._inactive_counts.pop(tid, None)

    def reset(self) -> None:
        """Clear all buffers."""
        self._buffers.clear()
        self._inactive_counts.clear()
