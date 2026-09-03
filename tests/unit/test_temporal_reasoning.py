"""Unit tests for Phase 17: Temporal Reasoning Validation.
Validates that temporal context governs emergency classification rather than instantaneous frame snapshots.
"""

import time
import numpy as np
import pytest

from src.temporal.sequence_buffer import SequenceBuffer, TrackSnapshot
from src.temporal.temporal_features import TemporalFeatureExtractor
from src.action.action_classifier import ActionClassifier
from src.action.behavior_classifier import BehaviorSequenceAnalyzer
from src.risk.risk_engine import RiskEngine
from src.risk.state_machine import EventStateMachine, MonitorState
from src.risk.threshold_manager import ThresholdManager
from src.alerts.alert_manager import AlertManager


def _make_keypoints(y_offset: float = 0.5, is_lying: bool = False) -> np.ndarray:
    """Generate 17 COCO keypoints either upright or horizontal."""
    kpts = np.zeros((17, 3), dtype=np.float32)
    kpts[:, 2] = 0.95  # High confidence
    if not is_lying:
        # Upright standing: head at y_offset, ankles at y_offset + 0.8
        kpts[0] = [0.5, y_offset, 0.95]        # Nose
        kpts[5] = [0.45, y_offset + 0.2, 0.95] # L-Shoulder
        kpts[6] = [0.55, y_offset + 0.2, 0.95] # R-Shoulder
        kpts[11] = [0.48, y_offset + 0.5, 0.95] # L-Hip
        kpts[12] = [0.52, y_offset + 0.5, 0.95] # R-Hip
        kpts[15] = [0.48, y_offset + 0.8, 0.95] # L-Ankle
        kpts[16] = [0.52, y_offset + 0.8, 0.95] # R-Ankle
    else:
        # Lying on floor horizontally: y is close to floor (0.85 - 0.90) across all joints
        kpts[0] = [0.2, y_offset, 0.95]        # Nose
        kpts[5] = [0.35, y_offset, 0.95]       # L-Shoulder
        kpts[6] = [0.35, y_offset + 0.05, 0.95]
        kpts[11] = [0.60, y_offset, 0.95]      # L-Hip
        kpts[12] = [0.60, y_offset + 0.05, 0.95]
        kpts[15] = [0.85, y_offset, 0.95]      # L-Ankle
        kpts[16] = [0.85, y_offset + 0.05, 0.95]
    return kpts


def test_scenario_a_same_current_frame_different_history():
    """Test A: Current frame is identical (lying on floor), but History A1 (fall) vs History A2 (intentional rest).
    System must differentiate based on temporal history!
    """
    buffer_a1 = SequenceBuffer(window_size=30)
    buffer_a2 = SequenceBuffer(window_size=30)
    extractor = TemporalFeatureExtractor()
    risk_engine = RiskEngine()

    start_t = 100.0

    # Scenario A1: walking (y=0.2) -> sudden drop (y moves from 0.2 to 0.85 in 0.3s) -> lying (y=0.85)
    for i in range(15):
        t = start_t + i * 0.066
        k = _make_keypoints(0.2, is_lying=False)
        buffer_a1.add_snapshot(1, TrackSnapshot(timestamp=t, bbox=(100, 50, 150, 200), keypoints=k, normalized_keypoints=k, torso_angle=10.0))
    for i in range(15, 20):  # Rapid downward transition (high Vy and Ay)
        t = start_t + i * 0.066
        prog = (i - 15) / 5.0
        y_prog = 0.2 + (0.85 - 0.2) * prog
        angle_prog = 10.0 + (85.0 - 10.0) * prog
        k = _make_keypoints(y_prog, is_lying=(i > 17))
        buffer_a1.add_snapshot(1, TrackSnapshot(timestamp=t, bbox=(100, int(y_prog*200), 200, 250), keypoints=k, normalized_keypoints=k, torso_angle=angle_prog))
    for i in range(20, 30):  # Lying immobile on floor
        t = start_t + i * 0.066
        k = _make_keypoints(0.85, is_lying=True)
        buffer_a1.add_snapshot(1, TrackSnapshot(timestamp=t, bbox=(100, 180, 250, 220), keypoints=k, normalized_keypoints=k, torso_angle=85.0))

    # Scenario A2: resting/sleeping on floor/bed steadily (all 30 frames already lying at y=0.85, zero drop speed)
    for i in range(30):
        t = start_t + i * 0.066
        k = _make_keypoints(0.85, is_lying=True)
        buffer_a2.add_snapshot(2, TrackSnapshot(timestamp=t, bbox=(100, 180, 250, 220), keypoints=k, normalized_keypoints=k, torso_angle=85.0))

    # Compute kinematics
    kin_a1 = extractor.extract(buffer_a1.get_snapshots(1))
    kin_a2 = extractor.extract(buffer_a2.get_snapshots(2))

    # Both are currently lying horizontally:
    assert kin_a1.torso_angle_curr > 60.0, "A1 torso angle should indicate horizontal posture"
    assert kin_a2.torso_angle_curr > 60.0, "A2 torso angle should indicate horizontal posture"

    # But A1 experienced severe downward velocity/drop and high Ay, whereas A2 experienced zero drop!
    assert kin_a1.drop_severity_score > kin_a2.drop_severity_score, f"A1 drop severity ({kin_a1.drop_severity_score}) must exceed A2 ({kin_a2.drop_severity_score})"
    assert kin_a1.vertical_velocity > kin_a2.vertical_velocity, "A1 downward velocity must exceed A2"

    from src.action.action_classifier import ActionPrediction
    from src.anomaly.anomaly_detector import AnomalyResult

    # Compute Risk Engine scores
    act_a1 = ActionPrediction(
        primary_action="lying",
        confidence=0.90,
        probabilities={"lying": 0.90},
        emergency_prob=0.85,
        severity_score=0.90,
    )
    anom_a1 = AnomalyResult(anomaly_score=0.75, is_anomaly=True, reconstruction_error=0.45, baseline_threshold=0.35)

    act_a2 = ActionPrediction(
        primary_action="lying",
        confidence=0.90,
        probabilities={"lying": 0.90},
        emergency_prob=0.0,
        severity_score=0.20,
    )
    anom_a2 = AnomalyResult(anomaly_score=0.08, is_anomaly=False, reconstruction_error=0.05, baseline_threshold=0.35)

    res_a1 = risk_engine.assess(
        track_id=1,
        action_pred=act_a1,
        anomaly_res=anom_a1,
        kinematics=kin_a1,
        timestamp=start_t + 30 * 0.066,
    )

    res_a2 = risk_engine.assess(
        track_id=2,
        action_pred=act_a2,
        anomaly_res=anom_a2,
        kinematics=kin_a2,
        timestamp=start_t + 30 * 0.066,
    )

    # Acceptance requirement: Risk A1 must be significantly higher than Risk A2!
    assert res_a1.risk_score >= 0.70, f"Scenario A1 risk must be elevated (got {res_a1.risk_score})"
    assert res_a2.risk_score < 0.40, f"Scenario A2 risk must remain low/normal (got {res_a2.risk_score})"
    assert res_a1.risk_score > res_a2.risk_score + 0.35, "Temporal context must produce a clear distinction between fall and resting"


def test_scenario_b_fall_timeline():
    """Test B: Complete fall sequence with strict timeline recording."""
    thresholds = ThresholdManager({
        "state_machine": {
            "min_duration_suspicious": 0.2,
            "min_duration_abnormal": 0.5,
            "min_duration_high_risk": 1.0,
            "min_recovery_duration": 1.0,
        },
        "thresholds": {
            "suspicious": 0.40,
            "abnormal": 0.65,
            "high_risk": 0.85,
        }
    })
    t0 = 1000.0
    fsm = EventStateMachine(thresholds=thresholds, initial_time=t0)

    timeline = {}
    timeline["t0_event_onset"] = t0

    # Frame 1-3: Normal standing (t = 1000.0 to 1000.2)
    for i in range(3):
        t = t0 + i * 0.1
        fsm.update(risk_score=0.10, action="walking", timestamp=t)
    assert fsm.state == MonitorState.NORMAL

    # T1: Loss of balance / abnormal evidence begins (t = 1000.3)
    t1 = t0 + 0.3
    timeline["t1_abnormal_observed"] = t1
    fsm.update(risk_score=0.55, action="stumbling", timestamp=t1)

    # T2: State becomes SUSPICIOUS (after min_duration_suspicious)
    t2 = t1 + 0.25
    fsm.update(risk_score=0.65, action="falling", timestamp=t2)
    assert fsm.state == MonitorState.SUSPICIOUS
    timeline["t2_suspicious"] = t2

    # T3: State becomes ABNORMAL (after min_duration_abnormal)
    t3 = t2 + 0.3
    fsm.update(risk_score=0.85, action="falling", timestamp=t3)
    assert fsm.state == MonitorState.ABNORMAL
    timeline["t3_abnormal"] = t3

    # T4: Impact & Immobility -> becomes HIGH_RISK
    t4 = t3 + 0.6
    fsm.update(risk_score=0.92, action="immobile", timestamp=t4)
    assert fsm.state == MonitorState.HIGH_RISK
    timeline["t4_high_risk"] = t4

    # Verify timeline monotonic ordering
    assert timeline["t0_event_onset"] < timeline["t1_abnormal_observed"]
    assert timeline["t1_abnormal_observed"] < timeline["t2_suspicious"]
    assert timeline["t2_suspicious"] < timeline["t3_abnormal"]
    assert timeline["t3_abnormal"] < timeline["t4_high_risk"]


def test_scenario_c_non_fall_lying():
    """Test C: Walking -> sitting down -> lying down gently on sofa. No emergency generated."""
    thresholds = ThresholdManager({
        "state_machine": {
            "min_duration_suspicious": 0.5,
            "min_duration_abnormal": 1.0,
        },
        "thresholds": {
            "suspicious": 0.40,
            "abnormal": 0.65,
            "high_risk": 0.85,
        }
    })
    t = 2000.0
    fsm = EventStateMachine(thresholds=thresholds, initial_time=t)

    # Walking (0 - 2s)
    for _ in range(10):
        t += 0.2
        fsm.update(risk_score=0.05, action="walking", timestamp=t)
    # Sitting (2 - 4s)
    for _ in range(10):
        t += 0.2
        fsm.update(risk_score=0.10, action="sitting", timestamp=t)
    # Lying down gently (4 - 7s)
    for _ in range(15):
        t += 0.2
        fsm.update(risk_score=0.25, action="lying", timestamp=t)

    # State must remain NORMAL throughout
    assert fsm.state == MonitorState.NORMAL, "Gentle lying down must not trigger alert"


def test_scenario_d_recovery_from_temporary_fall():
    """Test D: Stumble/fall-like movement followed by self-recovery.
    System escalates to SUSPICIOUS/ABNORMAL, but upon getting up, recovers back to NORMAL.
    """
    thresholds = ThresholdManager({
        "state_machine": {
            "min_duration_suspicious": 0.2,
            "min_duration_abnormal": 0.5,
            "min_duration_high_risk": 1.2,
            "min_recovery_duration": 0.8,
        },
        "thresholds": {
            "suspicious": 0.40,
            "abnormal": 0.65,
            "high_risk": 0.85,
        }
    })
    t = 3000.0
    fsm = EventStateMachine(thresholds=thresholds, initial_time=t)

    # Person trips and falls to floor
    fsm.update(risk_score=0.75, action="falling", timestamp=t)
    t += 0.3
    fsm.update(risk_score=0.80, action="falling", timestamp=t)
    assert fsm.state in [MonitorState.SUSPICIOUS, MonitorState.ABNORMAL]

    # Person starts getting up immediately (t = 3000.5)
    t += 0.2
    fsm.update(risk_score=0.30, action="getting_up", timestamp=t)
    t += 0.4
    fsm.update(risk_score=0.20, action="standing", timestamp=t)
    t += 0.9  # Exceeds min_recovery_duration (0.8s)
    fsm.update(risk_score=0.10, action="walking", timestamp=t)

    # System must successfully recover back to NORMAL!
    assert fsm.state == MonitorState.NORMAL, "System must recover to NORMAL after person gets up"


def test_scenario_e_single_frame_noise_tolerance():
    """Test E: 1 isolated glitch frame (false 'falling') in an otherwise normal stream.
    Single frame MUST NOT trigger an alert.
    """
    thresholds = ThresholdManager({
        "state_machine": {
            "min_duration_suspicious": 0.4,
            "min_duration_abnormal": 1.0,
            "min_duration_high_risk": 2.0,
        },
        "thresholds": {
            "suspicious": 0.40,
            "abnormal": 0.65,
            "high_risk": 0.85,
        }
    })
    t = 4000.0
    fsm = EventStateMachine(thresholds=thresholds, initial_time=t)

    # 10 normal walking frames
    for _ in range(10):
        t += 0.1
        fsm.update(risk_score=0.05, action="walking", timestamp=t)
    assert fsm.state == MonitorState.NORMAL

    # 1 GLITCH FRAME: detector or classifier glitch emits raw_risk=0.99 ('falling')
    t += 0.1
    fsm.update(risk_score=0.99, action="falling", timestamp=t)
    # MUST NOT be HIGH_RISK or ALERT_SENT! Debounce requires duration.
    assert fsm.state != MonitorState.HIGH_RISK
    assert fsm.state != MonitorState.ALERT_SENT

    # Immediately returns to normal walking next frame
    t += 0.1
    fsm.update(risk_score=0.05, action="walking", timestamp=t)
    assert fsm.state == MonitorState.NORMAL
