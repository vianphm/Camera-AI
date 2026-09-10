"""Debounced Event State Machine with Hysteresis Timer and Self-Recovery.

Guarantees low false alarm rate, rapid response time (1.5 - 3.5s), and non-diagnostic
medical safety alert generation.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from src.analytics.kinematic_engine import BiomechanicalFeatures


class MonitorState(str, Enum):
    """Core operational states of the monitoring finite state machine."""
    NORMAL = "NORMAL"
    SUSPICIOUS = "SUSPICIOUS"
    FALL_DETECTED = "FALL_DETECTED"
    ALERT_SENT = "ALERT_SENT"
    RECOVERED = "RECOVERED"


@dataclass
class AlertEvent:
    """Standard safety alert structure complying with non-diagnostic clinical principles."""
    event_id: str
    track_id: int
    timestamp: float
    state: MonitorState
    risk_score: float
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    should_blur_face: bool = False
    face_bbox: Optional[Tuple[float, float, float, float]] = None


class TrackFSM:
    """Individual finite state machine for a single tracked person."""

    def __init__(
        self,
        track_id: int,
        immobility_confirm_duration: float = 3.0,
        suspicious_timeout: float = 1.0,
        recovery_duration: float = 1.5,
        cooldown_duration: float = 60.0,
    ) -> None:
        self.track_id = track_id
        self.immobility_confirm_dur = immobility_confirm_duration
        self.suspicious_timeout = suspicious_timeout
        self.recovery_dur = recovery_duration
        self.cooldown_dur = cooldown_duration

        self.state: MonitorState = MonitorState.NORMAL
        self.state_enter_time: float = 0.0
        self.last_impact_time: float = -100.0
        self.fall_detected_time: float = -100.0
        self.alert_sent_time: float = -100.0
        self.recovery_start_time: float = -100.0
        self.last_normal_reset_time: float = -100.0
        self.alert_dispatched: bool = False

    def update(
        self,
        features: BiomechanicalFeatures,
        keypoints: Optional[np.ndarray],
        timestamp: float,
    ) -> Tuple[MonitorState, Optional[AlertEvent]]:
        """Advance the state machine given the latest biomechanical observations."""
        prev_state = self.state
        raised_alert: Optional[AlertEvent] = None

        # -----------------------------------------------------------
        # 1. State: NORMAL
        # -----------------------------------------------------------
        if self.state == MonitorState.NORMAL:
            if features.is_immediate_impact or (features.is_impact and (timestamp - self.last_normal_reset_time > 2.0)):
                self.state = MonitorState.SUSPICIOUS
                self.state_enter_time = timestamp
                self.last_impact_time = timestamp

        # -----------------------------------------------------------
        # 2. State: SUSPICIOUS
        # -----------------------------------------------------------
        elif self.state == MonitorState.SUSPICIOUS:
            # Check for immediate self-recovery / false impact (e.g. slight stumble recovered)
            if features.is_recovering or (features.torso_angle < 30.0 and not features.is_horizontal):
                if timestamp - self.state_enter_time >= 0.2:
                    self.state = MonitorState.NORMAL
                    self.state_enter_time = timestamp
                    self.last_normal_reset_time = timestamp
            elif features.is_horizontal:
                # Transition to FALL_DETECTED within 0.5s - 1.0s after impact
                self.state = MonitorState.FALL_DETECTED
                self.state_enter_time = timestamp
                self.fall_detected_time = timestamp
            elif timestamp - self.last_impact_time > self.suspicious_timeout:
                # Timed out without horizontal posture (e.g. quick bend or heavy step)
                self.state = MonitorState.NORMAL
                self.state_enter_time = timestamp
                self.last_normal_reset_time = timestamp

        # -----------------------------------------------------------
        # 3. State: FALL_DETECTED
        # -----------------------------------------------------------
        elif self.state == MonitorState.FALL_DETECTED:
            time_since_fall = timestamp - self.fall_detected_time

            # Check for Self-Recovery (person standing/sitting up in < 3.0 seconds)
            if features.is_recovering or (features.torso_angle < 30.0 and features.aspect_ratio < 0.8):
                self.state = MonitorState.RECOVERED
                self.state_enter_time = timestamp
                self.recovery_start_time = timestamp
            # Check for Immobility Confirmation to raise Alert
            elif features.is_immobile and time_since_fall >= self.immobility_confirm_dur:
                # Trigger Alert
                self.state = MonitorState.ALERT_SENT
                self.state_enter_time = timestamp
                self.alert_sent_time = timestamp
                self.alert_dispatched = True

                # Generate safety message conforming to non-diagnostic standard
                face_bbox = self._compute_face_bbox(keypoints) if keypoints is not None else None
                msg = (
                    f"Possible medical emergency / abnormal behavior detected. "
                    f"Please check the person (Track ID: {self.track_id})."
                )
                raised_alert = AlertEvent(
                    event_id=f"alert_{int(timestamp * 1000)}_{self.track_id}",
                    track_id=self.track_id,
                    timestamp=timestamp,
                    state=MonitorState.ALERT_SENT,
                    risk_score=features.risk_score,
                    message=msg,
                    details={
                        "fall_duration_s": round(time_since_fall, 2),
                        "torso_angle": round(features.torso_angle, 1),
                        "aspect_ratio": round(features.aspect_ratio, 2),
                        "immobility_score": round(features.immobility_score, 4),
                        "v_y_norm": round(features.v_y_norm, 2),
                    },
                    should_blur_face=True,
                    face_bbox=face_bbox,
                )
            elif not features.is_horizontal and features.torso_angle < 35.0:
                # Not horizontal anymore, standing up
                self.state = MonitorState.RECOVERED
                self.state_enter_time = timestamp

        # -----------------------------------------------------------
        # 4. State: ALERT_SENT
        # -----------------------------------------------------------
        elif self.state == MonitorState.ALERT_SENT:
            # Person recovered after alert was already sent
            if features.is_recovering or (features.torso_angle < 30.0 and features.aspect_ratio < 0.8):
                self.state = MonitorState.RECOVERED
                self.state_enter_time = timestamp

        # -----------------------------------------------------------
        # 5. State: RECOVERED
        # -----------------------------------------------------------
        elif self.state == MonitorState.RECOVERED:
            # Maintain normal behavior for recovery_dur to reset back to NORMAL
            if features.torso_angle < 35.0:
                if timestamp - self.state_enter_time >= self.recovery_dur:
                    self.state = MonitorState.NORMAL
                    self.state_enter_time = timestamp
                    self.alert_dispatched = False
            else:
                # Re-entered abnormal posture
                if features.is_horizontal:
                    self.state = MonitorState.FALL_DETECTED
                    self.state_enter_time = timestamp
                    self.fall_detected_time = timestamp

        return self.state, raised_alert

    @staticmethod
    def _compute_face_bbox(keypoints: np.ndarray) -> Optional[Tuple[float, float, float, float]]:
        """Calculate bounding box surrounding head/face keypoints (K0..K4)."""
        # K0: Nose, K1: L-Eye, K2: R-Eye, K3: L-Ear, K4: R-Ear
        head_kps = keypoints[:5]
        valid = head_kps[head_kps[:, 2] > 0.15]
        if len(valid) == 0:
            return None

        min_x, min_y = np.min(valid[:, :2], axis=0)
        max_x, max_y = np.max(valid[:, :2], axis=0)

        # Add margin
        head_h = max(max_y - min_y, 25.0)
        head_w = max(max_x - min_x, 25.0)
        margin_x = head_w * 0.4
        margin_y = head_h * 0.4

        return (
            float(max(0.0, min_x - margin_x)),
            float(max(0.0, min_y - margin_y)),
            float(max_x + margin_x),
            float(max_y + margin_y),
        )


class EventStateMachine:
    """Multi-Track Event State Machine coordinator."""

    def __init__(
        self,
        immobility_confirm_duration: float = 3.0,
        suspicious_timeout: float = 1.0,
        recovery_duration: float = 1.5,
        cooldown_duration: float = 60.0,
    ) -> None:
        self.immobility_confirm_dur = immobility_confirm_duration
        self.suspicious_timeout = suspicious_timeout
        self.recovery_dur = recovery_duration
        self.cooldown_dur = cooldown_duration
        self.track_fsms: Dict[int, TrackFSM] = {}

    def update(
        self,
        track_id: int,
        features: BiomechanicalFeatures,
        keypoints: Optional[np.ndarray],
        timestamp: float,
    ) -> Tuple[MonitorState, Optional[AlertEvent]]:
        """Update FSM for the given track_id."""
        if track_id not in self.track_fsms:
            self.track_fsms[track_id] = TrackFSM(
                track_id=track_id,
                immobility_confirm_duration=self.immobility_confirm_dur,
                suspicious_timeout=self.suspicious_timeout,
                recovery_duration=self.recovery_dur,
                cooldown_duration=self.cooldown_dur,
            )

        fsm = self.track_fsms[track_id]
        return fsm.update(features=features, keypoints=keypoints, timestamp=timestamp)

    def get_state(self, track_id: int) -> MonitorState:
        """Get the current state for a track_id."""
        fsm = self.track_fsms.get(track_id)
        return fsm.state if fsm else MonitorState.NORMAL

    def cleanup_inactive(self, active_track_ids: set[int]) -> None:
        """Remove state tracking for dead tracks."""
        stale = [tid for tid in self.track_fsms if tid not in active_track_ids]
        for tid in stale:
            self.track_fsms.pop(tid, None)
