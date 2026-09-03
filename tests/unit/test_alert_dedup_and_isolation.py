"""Unit tests for Phase 23 (Alert Deduplication & Correctness) and Phase 24 (Multi-Person Temporal Isolation)."""

import time
import pytest
from src.risk.threshold_manager import ThresholdManager
from src.risk.state_machine import EventStateMachine, MonitorState
from src.risk.risk_engine import RiskEngine, RiskAssessment
from src.alerts.alert_manager import AlertManager, AlertEvent
from src.action.action_classifier import ActionPrediction
from src.anomaly.anomaly_detector import AnomalyResult
from src.temporal.temporal_features import KinematicFeatures


def _dummy_kin(drop_sev: float = 0.0, immob: float = 0.0, angle: float = 10.0, ar: float = 0.5) -> KinematicFeatures:
    return KinematicFeatures(
        vertical_velocity=3.0 if drop_sev > 0.5 else 0.0,
        vertical_acceleration=2.0 if drop_sev > 0.5 else 0.0,
        aspect_ratio_curr=ar,
        aspect_ratio_change=2.0 if drop_sev > 0.5 else 0.0,
        torso_angle_curr=angle,
        torso_angle_velocity=0.0,
        immobility_index=immob,
        drop_severity_score=drop_sev,
    )


def test_alert_deduplication_under_sustained_emergency():
    """Phase 23: Fall occurs and persists for 30 seconds (300 frames at 10 FPS).
    Verifies that system issues exactly 1 primary alert and does NOT spam 300 alerts!
    """
    thresholds = ThresholdManager({
        "alerts": {
            "cooldown_seconds": 60.0,  # 60s cooldown
            "deduplication_window_seconds": 30.0,
        },
    })

    alert_mgr = AlertManager(thresholds=thresholds)

    dispatched_alerts = []
    alert_mgr.dispatcher.subscribe_websocket(lambda ev: dispatched_alerts.append(ev))

    t = 100.0

    # Simulate 300 frames where should_alert is repeatedly asserted by downstream or sustained event
    for i in range(300):
        t += 0.1
        assessment = RiskAssessment(
            track_id=1,
            risk_score=0.95,
            state=MonitorState.ALERT_SENT.value,
            state_changed=(i == 0),
            should_alert=True,
            evidence={"action": "immobile", "drop_severity": 0.9},
        )
        alert_mgr.process_assessment(assessment=assessment, frame=None, timestamp=t)

    # Core Verification:
    # Under 30 seconds of sustained emergency, exactly 1 alert is emitted, NOT 300!
    assert len(dispatched_alerts) == 1, f"Expected exactly 1 alert, but got {len(dispatched_alerts)}!"
    assert dispatched_alerts[0].person_id == 1
    assert "Possible medical emergency" in dispatched_alerts[0].message


def test_alert_recovery_and_re_arm():
    """Phase 23: Person falls -> 1 alert -> recovers -> new fall 70s later -> 2nd alert permitted."""
    thresholds = ThresholdManager({
        "alerts": {
            "cooldown_seconds": 60.0,
        },
    })
    alert_mgr = AlertManager(thresholds=thresholds)
    dispatched = []
    alert_mgr.dispatcher.subscribe_websocket(lambda ev: dispatched.append(ev))

    t = 1000.0

    # Event 1: Fall (triggers alert)
    assessment1 = RiskAssessment(
        track_id=1,
        risk_score=0.95,
        state=MonitorState.ALERT_SENT.value,
        state_changed=True,
        should_alert=True,
        evidence={"action": "falling"},
    )
    alert_mgr.process_assessment(assessment1, None, timestamp=t)
    assert len(dispatched) == 1

    # Person recovers and walks normally for 65 seconds
    for _ in range(65):
        t += 1.0
        normal_ass = RiskAssessment(
            track_id=1,
            risk_score=0.05,
            state=MonitorState.NORMAL.value,
            state_changed=False,
            should_alert=False,
            evidence={"action": "walking"},
        )
        alert_mgr.process_assessment(normal_ass, None, timestamp=t)

    # Event 2: New genuine fall after 65 seconds (exceeds 60s cooldown)
    t += 1.0
    assessment2 = RiskAssessment(
        track_id=1,
        risk_score=0.95,
        state=MonitorState.ALERT_SENT.value,
        state_changed=True,
        should_alert=True,
        evidence={"action": "falling"},
    )
    alert_mgr.process_assessment(assessment2, None, timestamp=t)

    # Total alerts should now be exactly 2
    assert len(dispatched) == 2, f"Expected 2 alerts after recovery and re-arm, got {len(dispatched)}"


def test_multi_person_temporal_isolation():
    """Phase 24: 3 individuals tracked concurrently in the same scene:
    Person 101: Normal walking (ADL)
    Person 102: Sudden fall and collapse (Emergency)
    Person 103: Intentional lying on sofa (Normal recumbent)

    Verifies strict isolation: Person 102 receives HIGH_RISK/ALERT, while Person 101 and 103
    remain in NORMAL state with zero history or risk contamination.
    """
    # Use fast debounce durations for test
    thresholds = ThresholdManager({
        "state_machine": {
            "min_duration_suspicious": 0.2,
            "min_duration_abnormal": 0.4,
            "min_duration_high_risk": 0.6,
        },
        "alerts": {
            "cooldown_seconds": 60.0,
        }
    })
    risk_engine = RiskEngine(thresholds=thresholds)
    alert_mgr = AlertManager(thresholds=thresholds)
    alerts_emitted = []
    alert_mgr.dispatcher.subscribe_websocket(lambda ev: alerts_emitted.append(ev))

    t = 500.0
    kin_101 = _dummy_kin(drop_sev=0.0, immob=0.0, angle=10.0, ar=0.45)
    kin_102 = _dummy_kin(drop_sev=0.92, immob=0.85, angle=82.0, ar=3.2)
    kin_103 = _dummy_kin(drop_sev=0.0, immob=0.75, angle=85.0, ar=3.0)

    # Predictions
    pred_101 = ActionPrediction("walking", 0.95, {"walking": 0.95}, 0.0, 0.0)
    pred_102 = ActionPrediction("falling", 0.92, {"falling": 0.92}, 0.92, 0.95)
    pred_103 = ActionPrediction("lying", 0.90, {"lying": 0.90}, 0.0, 0.20)

    anom_101 = AnomalyResult(0.05, False, 0.05, 0.35)
    anom_102 = AnomalyResult(0.85, True, 0.65, 0.35)
    anom_103 = AnomalyResult(0.08, False, 0.08, 0.35)

    # Run for 20 steps (2.0s), which exceeds 0.2 + 0.4 + 0.6 = 1.2s to trigger ALERT_SENT
    for step in range(20):
        t += 0.1
        # Assess Person 101
        res_101 = risk_engine.assess(101, pred_101, anom_101, kin_101, t)
        alert_mgr.process_assessment(res_101, None, t)

        # Assess Person 102
        res_102 = risk_engine.assess(102, pred_102, anom_102, kin_102, t)
        alert_mgr.process_assessment(res_102, None, t)

        # Assess Person 103
        res_103 = risk_engine.assess(103, pred_103, anom_103, kin_103, t)
        alert_mgr.process_assessment(res_103, None, t)

    # 1. State Isolation:
    assert res_101.state == MonitorState.NORMAL.value, f"Person 101 must be NORMAL (got {res_101.state})"
    assert res_101.risk_score < 0.15, f"Person 101 risk must be low (got {res_101.risk_score})"

    assert res_103.state == MonitorState.NORMAL.value, f"Person 103 must be NORMAL (got {res_103.state})"
    assert res_103.risk_score < 0.35, f"Person 103 risk must be low (got {res_103.risk_score})"

    assert res_102.state in [MonitorState.HIGH_RISK.value, MonitorState.ALERT_SENT.value, MonitorState.WAITING_FOR_CONFIRMATION.value]
    assert res_102.risk_score >= 0.80, f"Person 102 risk must be elevated (got {res_102.risk_score})"

    # 2. Alert Isolation:
    # All emitted alerts must strictly attribute person_id == 102!
    assert len(alerts_emitted) >= 1, "Expected at least 1 alert for Person 102"
    for a in alerts_emitted:
        assert a.person_id == 102, f"Alert wrongly attributed to Person {a.person_id} instead of 102!"
