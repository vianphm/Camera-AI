"""Contextual Risk Engine and Multi-Signal Fusion Reasoner."""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Set
import numpy as np

from src.action.action_classifier import ActionPrediction
from src.anomaly.anomaly_detector import AnomalyResult
from src.temporal.temporal_features import KinematicFeatures
from src.risk.threshold_manager import ThresholdManager
from src.risk.state_machine import MonitorState, EventStateMachine


@dataclass
class RiskAssessment:
    """Consolidated outcome of the multi-signal risk evaluation."""
    track_id: int
    risk_score: float
    state: str
    state_changed: bool
    should_alert: bool
    evidence: Dict[str, Any]


class RiskEngine:
    """Multi-signal decision engine evaluating risk severity and dispatching alerts."""

    def __init__(self, thresholds: Optional[ThresholdManager] = None) -> None:
        self.thresholds = thresholds or ThresholdManager()
        self._state_machines: Dict[int, EventStateMachine] = {}
        self.assessments: Dict[int, RiskAssessment] = {}
        self._inactive_counts: Dict[int, int] = {}

    def assess(
        self,
        track_id: int,
        action_pred: ActionPrediction,
        anomaly_res: AnomalyResult,
        kinematics: KinematicFeatures,
        timestamp: float,
    ) -> RiskAssessment:
        """Fuse signals and update state machine for a tracked person.

        Args:
            track_id: Identity of the person.
            action_pred: Supervised action classification result.
            anomaly_res: Unsupervised anomaly score result.
            kinematics: Kinematic feature indicators.
            timestamp: Current timestamp.

        Returns:
            RiskAssessment object with state and alert trigger.
        """
        # Multi-signal weighted linear combination
        w_act = self.thresholds.w_action
        w_anom = self.thresholds.w_anomaly
        w_kin = self.thresholds.w_kinematics
        w_immob = self.thresholds.w_immobility

        # 1. Action Signal: weighted by severity and emergency probability
        s_action = max(action_pred.emergency_prob, action_pred.severity_score)

        # 2. Anomaly Signal
        s_anomaly = anomaly_res.anomaly_score

        # 3. Kinematics Signal: drop severity
        s_kin = kinematics.drop_severity_score

        # 4. Immobility Signal: relevant especially if down/lying
        is_down = (kinematics.aspect_ratio_curr > 0.90 or kinematics.torso_angle_curr > 50.0)
        s_immob = kinematics.immobility_index if is_down else 0.0

        # Base fused score
        fused_score = (
            w_act * s_action +
            w_anom * s_anomaly +
            w_kin * s_kin +
            w_immob * s_immob
        )

        # 5. Synergistic Multi-Signal Escalation:
        # A) Acute Fall: When physical drop AND emergency action mutually confirm each other
        if s_kin >= 0.45 and s_action >= 0.45:
            synergy = 0.50 * s_action + 0.50 * s_kin
            fused_score = max(fused_score, synergy)

        # B) Prolonged Immobility on Floor: When person remains down/lying with high immobility index
        if is_down and s_immob >= 0.40:
            immob_score = 0.45 * s_action + 0.40 * s_immob + 0.15 * s_anomaly
            fused_score = max(fused_score, immob_score)

        risk_score = float(np.clip(fused_score, 0.0, 1.0))

        # Get or create track state machine
        if track_id not in self._state_machines:
            self._state_machines[track_id] = EventStateMachine(self.thresholds, initial_time=timestamp)

        self._inactive_counts[track_id] = 0
        sm = self._state_machines[track_id]
        new_state, state_changed = sm.update(
            risk_score=risk_score,
            action=action_pred.primary_action,
            timestamp=timestamp,
        )

        should_alert = (new_state == MonitorState.ALERT_SENT and state_changed)

        evidence = {
            "primary_action": action_pred.primary_action,
            "action_confidence": round(action_pred.confidence, 3),
            "emergency_probability": round(action_pred.emergency_prob, 3),
            "anomaly_score": round(anomaly_res.anomaly_score, 3),
            "reconstruction_error": round(anomaly_res.reconstruction_error, 4),
            "drop_severity": round(kinematics.drop_severity_score, 3),
            "immobility_index": round(kinematics.immobility_index, 3),
            "torso_angle_deg": round(kinematics.torso_angle_curr, 1),
            "aspect_ratio": round(kinematics.aspect_ratio_curr, 2),
            "vertical_velocity": round(kinematics.vertical_velocity, 2),
        }

        assessment = RiskAssessment(
            track_id=track_id,
            risk_score=risk_score,
            state=new_state.value,
            state_changed=state_changed,
            should_alert=should_alert,
            evidence=evidence,
        )
        self.assessments[track_id] = assessment
        return assessment

    def cleanup_inactive(self, active_track_ids: Set[int]) -> None:
        """Evict state machines for tracks that have left the scene, with a grace period."""
        for tid in list(self._state_machines.keys()):
            if tid not in active_track_ids:
                self._inactive_counts[tid] = self._inactive_counts.get(tid, 0) + 1
                if self._inactive_counts[tid] > 45:  # ~1.5s - 3s grace buffer
                    self._state_machines.pop(tid, None)
                    self.assessments.pop(tid, None)
                    self._inactive_counts.pop(tid, None)
            else:
                self._inactive_counts[tid] = 0
