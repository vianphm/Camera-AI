"""Biomechanical Kinematic Engine with Torso-Normalized Tri-Condition Fall Confirmation.

Extracts normalized drop velocities, angular accelerations, post-impact geometry,
and scale-invariant immobility indices for accurate elderly fall detection.
"""

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np


@dataclass
class KeypointSnapshot:
    """Represents a timestamped skeletal pose and bounding box snapshot."""
    timestamp: float
    keypoints: np.ndarray  # shape: (17, 3) -> [x, y, conf]
    bbox: Tuple[float, float, float, float]  # [x1, y1, x2, y2]
    mid_shoulder: np.ndarray = field(init=False)
    mid_hip: np.ndarray = field(init=False)
    torso_length: float = field(init=False)
    torso_angle: float = field(init=False)
    aspect_ratio: float = field(init=False)

    def __post_init__(self) -> None:
        # K5: Left Shoulder, K6: Right Shoulder
        self.mid_shoulder = (self.keypoints[5, :2] + self.keypoints[6, :2]) / 2.0
        # K11: Left Hip, K12: Right Hip
        self.mid_hip = (self.keypoints[11, :2] + self.keypoints[12, :2]) / 2.0
        
        # Torso Length: L_torso = ||MidShoulder - MidHip||_2
        raw_torso = float(np.linalg.norm(self.mid_shoulder - self.mid_hip))
        self.torso_length = max(raw_torso, 10.0)

        # Torso Tilt Angle relative to vertical (0 deg = standing upright, 90 deg = horizontal)
        dx = abs(self.mid_shoulder[0] - self.mid_hip[0])
        dy = abs(self.mid_shoulder[1] - self.mid_hip[1])
        self.torso_angle = float(np.degrees(np.arctan2(dx, dy + 1e-5)))

        # Bounding box aspect ratio W / H
        w = max(0.0, self.bbox[2] - self.bbox[0])
        h = max(0.0, self.bbox[3] - self.bbox[1])
        self.aspect_ratio = float(w / (h + 1e-5))


@dataclass
class BiomechanicalFeatures:
    """Extracted kinematic and geometric indicators for a tracked individual."""
    v_y_norm: float             # Downward vertical velocity normalized by torso length (torso/s)
    a_y_norm: float             # Downward vertical acceleration normalized by torso length (torso/s^2)
    torso_angle: float          # Torso tilt relative to vertical [0..90] degrees
    torso_angle_velocity: float # Angular velocity of torso in degrees/s
    aspect_ratio: float         # Bbox width-to-height ratio
    immobility_score: float     # Scale-invariant movement energy of 6 key joints
    is_immediate_impact: bool   # Current frame shows a sudden impact spike
    is_impact: bool             # High downward velocity or sudden tilt detected in recent window
    is_horizontal: bool         # Body lying parallel to floor (high angle or W/H > 1.15)
    is_immobile: bool           # Immobility index indicates lack of movement
    is_recovering: bool         # Body returning to upright orientation with hip elevation
    risk_score: float           # Calibrated risk metric [0.0..1.0]


class KinematicEngine:
    """Sliding-window Biomechanical Kinematic Engine supporting multi-person tracking."""

    # Key joints for immobility: shoulders (5, 6), hips (11, 12), knees (13, 14)
    KEY_JOINTS = [5, 6, 11, 12, 13, 14]

    def __init__(
        self,
        buffer_size: int = 30,
        vy_impact_threshold: float = 1.15,
        tilt_vel_impact_threshold: float = 30.0,
        horizontal_angle_threshold: float = 55.0,
        aspect_ratio_threshold: float = 1.15,
        immobility_threshold: float = 0.08,
        recovery_angle_threshold: float = 30.0,
    ) -> None:
        """Initialize KinematicEngine with physical thresholds.

        Args:
            buffer_size: Number of frames to store per track_id (default T=30, ~2s at 15 FPS).
            vy_impact_threshold: Downward velocity threshold in torso-lengths/sec.
            tilt_vel_impact_threshold: Torso angular velocity in deg/s.
            horizontal_angle_threshold: Tilt angle threshold for horizontal posture.
            aspect_ratio_threshold: Bbox W/H threshold for floor posture.
            immobility_threshold: Maximum normalized movement to classify as immobile.
            recovery_angle_threshold: Maximum torso tilt angle to qualify as recovered/upright.
        """
        self.buffer_size = buffer_size
        self.vy_impact_thresh = vy_impact_threshold
        self.tilt_vel_thresh = tilt_vel_impact_threshold
        self.horizontal_angle_thresh = horizontal_angle_threshold
        self.aspect_ratio_thresh = aspect_ratio_threshold
        self.immobility_thresh = immobility_threshold
        self.recovery_angle_thresh = recovery_angle_threshold

        # Track ID -> deque of KeypointSnapshot
        self.buffers: Dict[int, deque[KeypointSnapshot]] = {}
        # Track ID -> baseline upright hip height
        self.baseline_hip_y: Dict[int, float] = {}
        # Track ID -> timestamp of last confirmed impact
        self.last_impact_time: Dict[int, float] = {}

    def update(
        self,
        track_id: int,
        keypoints: np.ndarray,
        bbox: Tuple[float, float, float, float],
        timestamp: float,
    ) -> BiomechanicalFeatures:
        """Feed a new frame observation for track_id and compute biomechanical features."""
        if track_id not in self.buffers:
            self.buffers[track_id] = deque(maxlen=self.buffer_size)

        snapshot = KeypointSnapshot(
            timestamp=timestamp,
            keypoints=keypoints.copy(),
            bbox=bbox,
        )
        buf = self.buffers[track_id]
        buf.append(snapshot)

        # Establish baseline standing hip height if upright
        if snapshot.torso_angle < 25.0 and snapshot.aspect_ratio < 0.6:
            if track_id not in self.baseline_hip_y:
                self.baseline_hip_y[track_id] = snapshot.mid_hip[1]
            else:
                # Running smooth average
                self.baseline_hip_y[track_id] = 0.95 * self.baseline_hip_y[track_id] + 0.05 * snapshot.mid_hip[1]

        return self.evaluate(track_id)

    def evaluate(self, track_id: int) -> BiomechanicalFeatures:
        """Extract Tri-Condition fall indicators and calculate risk score."""
        buf = self.buffers.get(track_id)
        if not buf or len(buf) < 3:
            return BiomechanicalFeatures(
                v_y_norm=0.0,
                a_y_norm=0.0,
                torso_angle=10.0,
                torso_angle_velocity=0.0,
                aspect_ratio=0.5,
                immobility_score=0.2,
                is_immediate_impact=False,
                is_impact=False,
                is_horizontal=False,
                is_immobile=False,
                is_recovering=False,
                risk_score=0.0,
            )

        curr = buf[-1]
        
        # 1. Downward Vertical Velocity and Acceleration over short window (k = 3-5 frames)
        k = min(5, len(buf) - 1)
        prev = buf[-1 - k]
        dt = max(curr.timestamp - prev.timestamp, 1e-4)

        # y increases downwards in pixel space; positive dy means falling
        dy = curr.mid_hip[1] - prev.mid_hip[1]
        l_torso = curr.torso_length
        v_y_norm = float(dy / (l_torso * dt))

        # Previous velocity for acceleration calculation
        if len(buf) >= 2 * k + 1:
            prev_prev = buf[-1 - 2 * k]
            dt_prev = max(prev.timestamp - prev_prev.timestamp, 1e-4)
            dy_prev = prev.mid_hip[1] - prev_prev.mid_hip[1]
            v_prev_norm = float(dy_prev / (prev.torso_length * dt_prev))
            a_y_norm = float((v_y_norm - v_prev_norm) / dt)
        else:
            a_y_norm = 0.0

        # 2. Torso Angular Velocity
        d_angle = abs(curr.torso_angle - prev.torso_angle)
        torso_angle_vel = float(d_angle / dt)

        # 3. Peak Impact Check within past buffer window
        # An impact might have occurred 5-15 frames ago before the person came to rest
        max_vy = v_y_norm
        max_tilt_vel = torso_angle_vel
        for i in range(1, min(len(buf), 15)):
            s_cur = buf[-i]
            s_prv = buf[-i - 1]
            t_diff = max(s_cur.timestamp - s_prv.timestamp, 1e-4)
            step_vy = (s_cur.mid_hip[1] - s_prv.mid_hip[1]) / (s_cur.torso_length * t_diff)
            step_tilt_vel = abs(s_cur.torso_angle - s_prv.torso_angle) / t_diff
            if step_vy > max_vy:
                max_vy = step_vy
            if step_tilt_vel > max_tilt_vel:
                max_tilt_vel = step_tilt_vel

        is_immediate_impact = (max_vy >= self.vy_impact_thresh) or (max_tilt_vel >= self.tilt_vel_thresh)
        if is_immediate_impact:
            self.last_impact_time[track_id] = curr.timestamp

        # Impact is retained if observed within the last 3.5 seconds
        time_since_impact = curr.timestamp - self.last_impact_time.get(track_id, -100.0)
        is_impact_confirmed = is_immediate_impact or (0.0 <= time_since_impact <= 3.5)

        # 4. Post-Impact Geometry (Horizontal Posture)
        is_horizontal = (curr.torso_angle >= self.horizontal_angle_thresh) or (curr.aspect_ratio >= self.aspect_ratio_thresh)

        # 5. Immobility Index over recent window
        immobility_score = self._compute_immobility(buf)
        # Immobility is relevant when the body is in horizontal / fallen posture
        is_immobile = is_horizontal and (immobility_score <= self.immobility_thresh)

        # 6. Self-Recovery Check
        # If torso returned upright (< 30 deg) and hip elevated back towards standing baseline
        baseline_hip = self.baseline_hip_y.get(track_id, curr.mid_hip[1])
        hip_elevated = (curr.mid_hip[1] <= baseline_hip + 0.6 * l_torso)
        is_recovering = (curr.torso_angle < self.recovery_angle_thresh) and hip_elevated and (curr.aspect_ratio < 0.8)

        # 7. Multi-Signal Risk Score Formulation
        risk_score = 0.0
        if is_recovering:
            risk_score = 0.10
        elif is_impact_confirmed and is_horizontal and is_immobile:
            # Tri-condition satisfied: Impact + Floor posture + Immobility
            risk_score = min(1.0, 0.70 + 0.30 * min(1.0, time_since_impact / 2.5))
        elif is_impact_confirmed and is_horizontal:
            # Impact + Horizontal, still moving or settling
            risk_score = 0.65
        elif is_horizontal and is_immobile:
            # Lying immobile without sudden drop (e.g., intentional resting, sleeping)
            risk_score = 0.30
        elif is_immediate_impact:
            # Sudden stumble / impact spike without horizontal posture yet
            risk_score = 0.50
        else:
            # Normal ADL activities (walking, sitting, bending)
            risk_score = 0.05

        return BiomechanicalFeatures(
            v_y_norm=float(v_y_norm),
            a_y_norm=float(a_y_norm),
            torso_angle=float(curr.torso_angle),
            torso_angle_velocity=float(torso_angle_vel),
            aspect_ratio=float(curr.aspect_ratio),
            immobility_score=float(immobility_score),
            is_immediate_impact=bool(is_immediate_impact),
            is_impact=bool(is_impact_confirmed),
            is_horizontal=bool(is_horizontal),
            is_immobile=bool(is_immobile),
            is_recovering=bool(is_recovering),
            risk_score=float(round(risk_score, 4)),
        )

    def _compute_immobility(self, buf: deque[KeypointSnapshot], window_frames: int = 15) -> float:
        """Calculate mean displacement of 6 key joints normalized by torso length."""
        n = min(len(buf), window_frames)
        if n < 2:
            return 1.0

        total_disp = 0.0
        count = 0

        # Loop through consecutive snapshots in the window
        snapshots = list(buf)[-n:]
        for i in range(1, len(snapshots)):
            s_curr = snapshots[i]
            s_prev = snapshots[i - 1]
            l_ref = s_curr.torso_length

            for j in self.KEY_JOINTS:
                p_c = s_curr.keypoints[j, :2]
                p_p = s_prev.keypoints[j, :2]
                dist = np.linalg.norm(p_c - p_p)
                total_disp += (dist / l_ref)
                count += 1

        return float(total_disp / max(count, 1))

    def cleanup_inactive(self, active_track_ids: set[int]) -> None:
        """Purge state memory for terminated or inactive tracks."""
        stale_ids = [tid for tid in self.buffers if tid not in active_track_ids]
        for tid in stale_ids:
            self.buffers.pop(tid, None)
            self.baseline_hip_y.pop(tid, None)
            self.last_impact_time.pop(tid, None)
