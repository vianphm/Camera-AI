"""Kinematic feature extraction from temporal skeletal and bounding box trajectories."""

from dataclasses import dataclass
from typing import List, Optional
import numpy as np
from src.temporal.sequence_buffer import TrackSnapshot


@dataclass
class KinematicFeatures:
    """Quantitative physical kinematics extracted over a temporal sequence."""
    vertical_velocity: float       # Downward velocity of hip center (norm units / sec)
    vertical_acceleration: float   # Downward acceleration (Ay)
    aspect_ratio_curr: float       # Current Width / Height ratio (>1.2 = lying horizontal)
    aspect_ratio_change: float     # Rate of change from vertical to horizontal
    torso_angle_curr: float        # Current angle relative to vertical [0..90]
    torso_angle_velocity: float    # Angular rate of change (deg / sec)
    immobility_index: float        # [0.0 = high motion, 1.0 = completely stationary]
    drop_severity_score: float     # [0.0 = none, 1.0 = severe sudden drop]


class TemporalFeatureExtractor:
    """Extracts physics-based kinematic indicators from temporal snapshots."""

    def __init__(self, fps: float = 15.0) -> None:
        self.fps = fps

    def extract(self, snapshots: List[TrackSnapshot]) -> KinematicFeatures:
        """Extract kinematic features from snapshot history.

        Args:
            snapshots: Sequential history of snapshots for a single track.

        Returns:
            KinematicFeatures dataclass instance.
        """
        if len(snapshots) < 2:
            return KinematicFeatures(
                vertical_velocity=0.0,
                vertical_acceleration=0.0,
                aspect_ratio_curr=0.5,
                aspect_ratio_change=0.0,
                torso_angle_curr=0.0,
                torso_angle_velocity=0.0,
                immobility_index=0.0,
                drop_severity_score=0.0,
            )

        n = len(snapshots)
        curr = snapshots[-1]
        prev = snapshots[-2]
        dt = max(1e-3, curr.timestamp - prev.timestamp)

        # 1. Bounding box aspect ratio
        curr_bw = max(1.0, curr.bbox[2] - curr.bbox[0])
        curr_bh = max(1.0, curr.bbox[3] - curr.bbox[1])
        aspect_ratio_curr = curr_bw / curr_bh

        # Aspect ratio change over last ~1 second (min(n, int(self.fps)))
        hist_idx = max(0, n - int(self.fps))
        past = snapshots[hist_idx]
        past_bw = max(1.0, past.bbox[2] - past.bbox[0])
        past_bh = max(1.0, past.bbox[3] - past.bbox[1])
        past_ar = past_bw / past_bh
        aspect_ratio_change = aspect_ratio_curr - past_ar

        # 2. Vertical velocity and acceleration of Hip Center (in image coordinates: +y is downward)
        # Using raw keypoints 11 & 12
        hip_y_series = []
        for s in snapshots:
            kp = s.keypoints
            if kp[11, 2] > 0.2 and kp[12, 2] > 0.2:
                y = (kp[11, 1] + kp[12, 1]) / 2.0
            elif kp[11, 2] > 0.2:
                y = kp[11, 1]
            elif kp[12, 2] > 0.2:
                y = kp[12, 1]
            else:
                y = (s.bbox[1] + s.bbox[3]) / 2.0
            hip_y_series.append(y)

        # Velocity: pixels / sec normalized by person height
        height_ref = max(30.0, curr_bh)
        v_y = (hip_y_series[-1] - hip_y_series[-2]) / (height_ref * dt)

        # Acceleration: rate of velocity change
        if len(hip_y_series) >= 3:
            v_prev = (hip_y_series[-2] - hip_y_series[-3]) / (height_ref * dt)
            a_y = (v_y - v_prev) / dt
        else:
            a_y = 0.0

        # 3. Torso angle and angular velocity
        torso_angle_curr = curr.torso_angle
        torso_angle_velocity = (curr.torso_angle - past.torso_angle) / max(1e-3, curr.timestamp - past.timestamp)

        # 4. Immobility Index: measure lack of movement across recent frames (last 2-5s)
        # Low motion variance => high immobility index
        window_pts = [s.normalized_keypoints[:, :2] for s in snapshots[-min(n, int(self.fps * 3)):]]
        if len(window_pts) >= 3:
            pts_arr = np.array(window_pts)  # (N, 17, 2)
            # Std deviation across time axis
            motion_std = float(np.mean(np.std(pts_arr, axis=0)))
            # When motion_std is very small (<0.04), person is virtually immobile
            immobility_index = float(np.clip(1.0 - (motion_std / 0.08), 0.0, 1.0))
        else:
            immobility_index = 0.0

        # 5. Drop Severity Score: heuristic combination of rapid drop & horizontal posture
        # High score if:
        # a) Downward velocity/acceleration was high
        # b) Posture is now horizontal (aspect ratio > 1.0 or torso angle > 60 deg)
        posture_horiz_factor = np.clip((torso_angle_curr - 30.0) / 45.0, 0.0, 1.0)
        aspect_ratio_factor = np.clip((aspect_ratio_curr - 0.7) / 0.8, 0.0, 1.0)
        drop_velocity_factor = np.clip(v_y / 2.5, 0.0, 1.0)

        drop_severity = (
            0.40 * max(posture_horiz_factor, aspect_ratio_factor) +
            0.35 * drop_velocity_factor +
            0.25 * (1.0 if a_y > 1.5 else 0.0)
        )
        drop_severity = float(np.clip(drop_severity, 0.0, 1.0))

        return KinematicFeatures(
            vertical_velocity=float(v_y),
            vertical_acceleration=float(a_y),
            aspect_ratio_curr=float(aspect_ratio_curr),
            aspect_ratio_change=float(aspect_ratio_change),
            torso_angle_curr=float(torso_angle_curr),
            torso_angle_velocity=float(torso_angle_velocity),
            immobility_index=float(immobility_index),
            drop_severity_score=drop_severity,
        )
