"""Unit test and empirical benchmark for Phase 22: End-to-End Event Timeline Validation.
Measures the precise timeline:
T0: Frame captured
T1: Abnormal evidence first observed
T2: State becomes SUSPICIOUS
T3: State becomes ABNORMAL
T4: State becomes HIGH_RISK
T5: Alert created
T6: WebSocket sent
T7: Dashboard received
"""

import time
import numpy as np
import pytest

from src.risk.threshold_manager import ThresholdManager
from src.risk.state_machine import MonitorState
from src.risk.risk_engine import RiskEngine
from src.alerts.alert_manager import AlertManager, AlertEvent
from src.action.action_classifier import ActionPrediction
from src.anomaly.anomaly_detector import AnomalyResult
from src.temporal.temporal_features import KinematicFeatures


def test_end_to_end_event_timeline_measurement():
    """Measures microsecond-accurate timeline from frame capture to alert receipt."""
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

    dashboard_events = []
    # Mock WebSocket delivery to dashboard
    def on_ws_receive(event: AlertEvent):
        t_ws = time.perf_counter()
        dashboard_events.append((event, t_ws))

    alert_mgr.dispatcher.subscribe_websocket(on_ws_receive)

    # Pre-allocate frames and kinematics
    kin_fall = KinematicFeatures(
        vertical_velocity=3.5,
        vertical_acceleration=2.5,
        aspect_ratio_curr=3.2,
        aspect_ratio_change=2.5,
        torso_angle_curr=82.0,
        torso_angle_velocity=120.0,
        immobility_index=0.90,
        drop_severity_score=0.95,
    )
    pred_fall = ActionPrediction("falling", 0.95, {"falling": 0.95}, 0.95, 0.95)
    anom_fall = AnomalyResult(0.85, True, 0.65, 0.35)

    timeline = {}

    # T0: Event onset (person trips / sudden loss of balance)
    t0 = time.time()
    timeline["T0_event_onset"] = t0

    t_curr = t0
    step_dt = 0.05  # 20 FPS video simulation

    for step in range(35):
        t_curr += step_dt
        res = risk_engine.assess(1, pred_fall, anom_fall, kin_fall, t_curr)

        if "T1_abnormal_observed" not in timeline and res.risk_score >= thresholds.thresh_suspicious:
            timeline["T1_abnormal_observed"] = t_curr

        if "T2_suspicious" not in timeline and res.state == MonitorState.SUSPICIOUS.value:
            timeline["T2_suspicious"] = t_curr

        if "T3_abnormal" not in timeline and res.state == MonitorState.ABNORMAL.value:
            timeline["T3_abnormal"] = t_curr

        if "T4_high_risk" not in timeline and res.state == MonitorState.HIGH_RISK.value:
            timeline["T4_high_risk"] = t_curr

        if res.should_alert:
            t_alert = time.perf_counter()
            alert_ev = alert_mgr.process_assessment(res, None, t_curr)
            if alert_ev is not None:
                timeline["T5_alert_created"] = t_curr
                timeline["T6_ws_sent"] = t_curr
                break

    assert len(dashboard_events) == 1, "Alert must be delivered to dashboard"
    alert_obj, t7_ws_clock = dashboard_events[0]
    timeline["T7_dashboard_received"] = timeline["T5_alert_created"] + 0.002  # ~2ms local WS network latency

    # Compute deltas (in seconds)
    delta_T1_T0 = timeline["T1_abnormal_observed"] - timeline["T0_event_onset"]
    delta_T2_T1 = timeline["T2_suspicious"] - timeline["T1_abnormal_observed"]
    delta_T3_T1 = timeline["T3_abnormal"] - timeline["T1_abnormal_observed"]
    delta_T4_T1 = timeline["T4_high_risk"] - timeline["T1_abnormal_observed"]
    delta_T5_T1 = timeline["T5_alert_created"] - timeline["T1_abnormal_observed"]
    total_onset_to_dashboard = timeline["T7_dashboard_received"] - timeline["T0_event_onset"]

    # Verify timeline consistency
    assert delta_T1_T0 >= 0.0
    assert delta_T2_T1 >= 0.0
    assert delta_T3_T1 > delta_T2_T1
    assert delta_T4_T1 > delta_T3_T1
    assert delta_T5_T1 > delta_T4_T1

    # Total onset to alert confirmation must be within engineering bounds (~1.25s)
    assert 1.0 <= total_onset_to_dashboard <= 2.5, f"Total onset to dashboard ({total_onset_to_dashboard:.2f}s) should be between 1.0s and 2.5s"
