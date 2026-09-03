"""Unit tests for Multi-Signal Fusion Risk Engine."""

import pytest
from src.action.action_classifier import ActionPrediction
from src.anomaly.anomaly_detector import AnomalyResult
from src.temporal.temporal_features import KinematicFeatures
from src.risk.risk_engine import RiskEngine
from src.risk.threshold_manager import ThresholdManager


def test_risk_engine_normal_behavior():
    engine = RiskEngine()

    action_pred = ActionPrediction(
        primary_action="walking",
        confidence=0.95,
        probabilities={"walking": 0.95},
        emergency_prob=0.0,
        severity_score=0.0,
    )
    anomaly_res = AnomalyResult(
        anomaly_score=0.05,
        is_anomaly=False,
        reconstruction_error=0.02,
        baseline_threshold=0.22,
    )
    kinematics = KinematicFeatures(
        vertical_velocity=0.0,
        vertical_acceleration=0.0,
        aspect_ratio_curr=0.35,
        aspect_ratio_change=0.0,
        torso_angle_curr=5.0,
        torso_angle_velocity=0.0,
        immobility_index=0.0,
        drop_severity_score=0.0,
    )

    assessment = engine.assess(1, action_pred, anomaly_res, kinematics, timestamp=1.0)
    assert assessment.risk_score < 0.20
    assert assessment.state == "NORMAL"
    assert not assessment.should_alert


def test_risk_engine_emergency_behavior():
    engine = RiskEngine()

    action_pred = ActionPrediction(
        primary_action="falling",
        confidence=0.98,
        probabilities={"falling": 0.98},
        emergency_prob=0.98,
        severity_score=0.95,
    )
    anomaly_res = AnomalyResult(
        anomaly_score=0.90,
        is_anomaly=True,
        reconstruction_error=0.45,
        baseline_threshold=0.22,
    )
    kinematics = KinematicFeatures(
        vertical_velocity=3.2,
        vertical_acceleration=2.5,
        aspect_ratio_curr=1.8,
        aspect_ratio_change=1.2,
        torso_angle_curr=85.0,
        torso_angle_velocity=45.0,
        immobility_index=0.95,
        drop_severity_score=0.95,
    )

    assessment = engine.assess(1, action_pred, anomaly_res, kinematics, timestamp=1.0)
    assert assessment.risk_score > 0.85
