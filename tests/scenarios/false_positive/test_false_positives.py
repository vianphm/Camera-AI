"""False Positive Hard-Negative Scenarios Test Suite.
Verifies that benign activities of daily living (ADL) and sensor noise do not trigger emergency alerts.
"""

import pytest
import numpy as np
from typing import Dict, Any, List

from src.risk.threshold_manager import ThresholdManager
from src.risk.state_machine import EventStateMachine, MonitorState
from src.risk.risk_engine import RiskEngine
from src.action.action_classifier import ActionPrediction
from src.anomaly.anomaly_detector import AnomalyResult
from src.temporal.temporal_features import KinematicFeatures


def run_scenario(
    scenario_id: str,
    action_stream: List[str],
    risk_scores: List[float],
    kinematics_list: List[KinematicFeatures],
    expected_final_state: MonitorState,
    expected_alerts: int = 0,
    fps: float = 15.0,
) -> Dict[str, Any]:
    """Execute an event sequence through the Risk and State Machine engines."""
    thresholds = ThresholdManager({
        "state_machine": {
            "min_duration_suspicious": 0.4,
            "min_duration_abnormal": 1.0,
            "min_duration_high_risk": 2.0,
            "min_recovery_duration": 1.5,
        },
        "thresholds": {
            "suspicious": 0.40,
            "abnormal": 0.65,
            "high_risk": 0.85,
        }
    })
    fsm = EventStateMachine(thresholds=thresholds, initial_time=0.0)
    alert_count = 0
    max_risk = 0.0
    states_visited = []

    t = 0.0
    for i, (action, risk, kin) in enumerate(zip(action_stream, risk_scores, kinematics_list)):
        t += 1.0 / fps
        max_risk = max(max_risk, risk)
        new_state, changed = fsm.update(risk_score=risk, action=action, timestamp=t)
        states_visited.append(new_state)
        if new_state == MonitorState.ALERT_SENT and changed:
            alert_count += 1

    passed = (fsm.state == expected_final_state) and (alert_count == expected_alerts)
    return {
        "scenario_id": scenario_id,
        "expected_state": expected_final_state.value,
        "actual_state": fsm.state.value,
        "alert_generated": alert_count > 0,
        "alert_count": alert_count,
        "duplicate_alert_count": max(0, alert_count - 1),
        "max_risk": round(max_risk, 3),
        "duration_s": round(t, 2),
        "status": "PASS" if passed else "FAIL",
    }


def _dummy_kin(torso_angle: float = 10.0, aspect_ratio: float = 0.5, immobility: float = 0.0, drop_sev: float = 0.0) -> KinematicFeatures:
    return KinematicFeatures(
        vertical_velocity=0.0,
        vertical_acceleration=0.0,
        aspect_ratio_curr=aspect_ratio,
        aspect_ratio_change=0.0,
        torso_angle_curr=torso_angle,
        torso_angle_velocity=0.0,
        immobility_index=immobility,
        drop_severity_score=drop_sev,
    )


def test_hard_negative_scenarios():
    """Run all 14 specified false-positive scenarios and assert zero false alarms."""
    results = []

    # 1. Intentional Lying (resting on floor mat or sofa for 10 seconds)
    n = 150
    res = run_scenario(
        scenario_id="HN-01-intentional_lying",
        action_stream=["lying"] * n,
        risk_scores=[0.25] * n,
        kinematics_list=[_dummy_kin(85.0, 3.0, immobility=0.8, drop_sev=0.0)] * n,
        expected_final_state=MonitorState.NORMAL,
        expected_alerts=0,
    )
    results.append(res)
    assert res["status"] == "PASS"

    # 2. Sleeping (stationary in bed for 15 seconds)
    n = 200
    res = run_scenario(
        scenario_id="HN-02-sleeping",
        action_stream=["lying"] * n,
        risk_scores=[0.15] * n,
        kinematics_list=[_dummy_kin(88.0, 3.5, immobility=1.0, drop_sev=0.0)] * n,
        expected_final_state=MonitorState.NORMAL,
        expected_alerts=0,
    )
    results.append(res)
    assert res["status"] == "PASS"

    # 3. Sitting Down Fast (plumping onto chair)
    # 5 frames high downward speed, but stops at chair height and remains upright (torso angle 15 deg)
    actions = ["standing"] * 10 + ["sitting"] * 20
    risks = [0.05] * 10 + [0.45, 0.48, 0.40, 0.20, 0.15] + [0.10] * 15
    kins = [_dummy_kin(10.0, 0.5)] * 10 + [_dummy_kin(15.0, 0.8, drop_sev=0.2)] * 20
    res = run_scenario("HN-03-sitting_down_fast", actions, risks, kins, MonitorState.NORMAL, 0)
    results.append(res)
    assert res["status"] == "PASS"

    # 4. Standing Up Fast
    actions = ["sitting"] * 10 + ["getting_up"] * 10 + ["standing"] * 15
    risks = [0.10] * 10 + [0.25] * 10 + [0.05] * 15
    kins = [_dummy_kin(15.0, 0.8)] * 10 + [_dummy_kin(10.0, 0.5)] * 25
    res = run_scenario("HN-04-standing_up_fast", actions, risks, kins, MonitorState.NORMAL, 0)
    results.append(res)
    assert res["status"] == "PASS"

    # 5. Bending to Pick Object (flexing torso forward for 1.8s, then returning upright)
    # Torso tilts to 60 deg, then recovers upright within 2 seconds
    actions = ["standing"] * 5 + ["bending"] * 25 + ["standing"] * 15
    risks = [0.05] * 5 + [0.35] * 25 + [0.05] * 15
    kins = [_dummy_kin(10.0, 0.5)] * 5 + [_dummy_kin(55.0, 0.75, drop_sev=0.1)] * 25 + [_dummy_kin(10.0, 0.5)] * 15
    res = run_scenario("HN-05-bending_to_pick_object", actions, risks, kins, MonitorState.NORMAL, 0)
    results.append(res)
    assert res["status"] == "PASS"

    # 6. Exercise / Calisthenics
    actions = (["standing"] * 10 + ["bending"] * 10) * 3
    risks = ([0.05] * 10 + [0.30] * 10) * 3
    kins = ([_dummy_kin(10.0, 0.5)] * 10 + [_dummy_kin(45.0, 0.7)] * 10) * 3
    res = run_scenario("HN-06-exercise", actions, risks, kins, MonitorState.NORMAL, 0)
    results.append(res)
    assert res["status"] == "PASS"

    # 7. Stretching (reaching upward/lateral)
    actions = ["standing"] * 30
    risks = [0.10] * 30
    kins = [_dummy_kin(12.0, 0.45)] * 30
    res = run_scenario("HN-07-stretching", actions, risks, kins, MonitorState.NORMAL, 0)
    results.append(res)
    assert res["status"] == "PASS"

    # 8. Crawling (moving on hands and knees horizontally)
    actions = ["lying"] * 30
    risks = [0.30] * 30  # Moving, not immobile
    kins = [_dummy_kin(70.0, 1.8, immobility=0.10, drop_sev=0.0)] * 30
    res = run_scenario("HN-08-crawling", actions, risks, kins, MonitorState.NORMAL, 0)
    results.append(res)
    assert res["status"] == "PASS"

    # 9. Kneeling (gardening or praying)
    actions = ["sitting"] * 30
    risks = [0.20] * 30
    kins = [_dummy_kin(30.0, 0.7, immobility=0.6, drop_sev=0.0)] * 30
    res = run_scenario("HN-09-kneeling", actions, risks, kins, MonitorState.NORMAL, 0)
    results.append(res)
    assert res["status"] == "PASS"

    # 10. Camera Shake (jitter in bounding box and pose)
    actions = ["walking"] * 30
    risks = [0.15] * 30
    kins = [_dummy_kin(10.0, 0.5, immobility=0.0)] * 30
    res = run_scenario("HN-10-camera_shake", actions, risks, kins, MonitorState.NORMAL, 0)
    results.append(res)
    assert res["status"] == "PASS"

    # 11. Lighting Change (sudden brightness shift causing brief confidence drop)
    actions = ["standing"] * 10 + ["unknown"] * 3 + ["standing"] * 15
    risks = [0.05] * 10 + [0.25] * 3 + [0.05] * 15
    kins = [_dummy_kin(10.0, 0.5)] * 28
    res = run_scenario("HN-11-lighting_change", actions, risks, kins, MonitorState.NORMAL, 0)
    results.append(res)
    assert res["status"] == "PASS"

    # 12. Partial Occlusion (walking behind dining table)
    actions = ["walking"] * 15 + ["standing"] * 15
    risks = [0.10] * 30
    kins = [_dummy_kin(10.0, 0.6)] * 30
    res = run_scenario("HN-12-partial_occlusion", actions, risks, kins, MonitorState.NORMAL, 0)
    results.append(res)
    assert res["status"] == "PASS"

    # 13. Person Temporarily Hidden (behind wall for 2 seconds)
    actions = ["walking"] * 15
    risks = [0.05] * 15
    kins = [_dummy_kin(10.0, 0.5)] * 15
    res = run_scenario("HN-13-person_temporarily_hidden", actions, risks, kins, MonitorState.NORMAL, 0)
    results.append(res)
    assert res["status"] == "PASS"

    # 14. Person Leaves Frame
    actions = ["walking"] * 20
    risks = [0.05] * 20
    kins = [_dummy_kin(10.0, 0.5)] * 20
    res = run_scenario("HN-14-person_leaves_frame", actions, risks, kins, MonitorState.NORMAL, 0)
    results.append(res)
    assert res["status"] == "PASS"

    # Verify overall False Positive Metrics
    total_scenarios = len(results)
    passed_scenarios = sum(1 for r in results if r["status"] == "PASS")
    total_alerts = sum(r["alert_count"] for r in results)

    assert passed_scenarios == total_scenarios, f"All {total_scenarios} hard negative scenarios must pass"
    assert total_alerts == 0, f"Expected 0 false alarms across all hard negative scenarios, got {total_alerts}"
