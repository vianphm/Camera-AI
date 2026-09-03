"""Unit tests for Tier 1 Kinematic Gated Trigger with anti-dead-gate safeguards."""

import numpy as np
import pytest

from src.temporal.gated_trigger import KinematicGatedTrigger
from src.temporal.sequence_buffer import TrackSnapshot
from src.temporal.temporal_features import KinematicFeatures


def make_dummy_snapshot(
    timestamp: float,
    bbox=(100.0, 100.0, 180.0, 300.0),
    torso_angle=5.0,
    hip_y=200.0,
) -> TrackSnapshot:
    kp = np.zeros((17, 3), dtype=np.float32)
    kp[:, 2] = 0.9
    kp[11, 1] = hip_y
    kp[12, 1] = hip_y
    norm_kp = np.zeros((17, 3), dtype=np.float32)
    return TrackSnapshot(
        timestamp=timestamp,
        bbox=bbox,
        keypoints=kp,
        normalized_keypoints=norm_kp,
        torso_angle=torso_angle,
    )


def test_normal_upright_walking_gate_closed():
    """Stable upright walking should keep Gate CLOSED with 0 ms AI overhead."""
    trigger = KinematicGatedTrigger(hold_on_frames=45)

    snapshots = [
        make_dummy_snapshot(timestamp=1.0, bbox=(100.0, 100.0, 160.0, 300.0), torso_angle=8.0, hip_y=200.0),
        make_dummy_snapshot(timestamp=1.1, bbox=(105.0, 100.0, 165.0, 300.0), torso_angle=10.0, hip_y=200.0),
    ]
    kinematics = KinematicFeatures(
        vertical_velocity=0.10,
        vertical_acceleration=0.05,
        aspect_ratio_curr=0.30,
        aspect_ratio_change=0.01,
        torso_angle_curr=10.0,
        torso_angle_velocity=2.0,
        immobility_index=0.10,
        drop_severity_score=0.0,
    )

    res = trigger.evaluate(track_id=1, snapshots=snapshots, kinematics=kinematics, current_state="NORMAL")
    assert res.should_run_ai is False
    assert res.heuristic_prediction is not None
    assert res.heuristic_prediction.primary_action in ["standing", "walking"]
    assert res.hold_on_frames_remaining == 0


def test_hard_fall_triggers_gate_open():
    """High vertical velocity/acceleration immediately triggers Gate OPEN."""
    trigger = KinematicGatedTrigger(hold_on_frames=45)

    snapshots = [
        make_dummy_snapshot(timestamp=1.0, bbox=(100.0, 100.0, 160.0, 300.0), torso_angle=10.0, hip_y=200.0),
        make_dummy_snapshot(timestamp=1.2, bbox=(100.0, 180.0, 250.0, 260.0), torso_angle=45.0, hip_y=240.0),
    ]
    kinematics = KinematicFeatures(
        vertical_velocity=1.80,  # > 1.10
        vertical_acceleration=1.50,  # > 0.75
        aspect_ratio_curr=0.90,
        aspect_ratio_change=0.60,
        torso_angle_curr=45.0,
        torso_angle_velocity=35.0,  # > 28 deg/s
        immobility_index=0.0,
        drop_severity_score=0.85,
    )

    res = trigger.evaluate(track_id=1, snapshots=snapshots, kinematics=kinematics, current_state="NORMAL")
    assert res.should_run_ai is True
    assert "high_vy" in res.trigger_reason or "high_ay" in res.trigger_reason
    assert res.hold_on_frames_remaining == 45


def test_slow_slide_ground_proximity_triggers_gate_open():
    """Vulnerability 2 check: Slow wall slide where Vy/Ay are low, but hip drops to ground plane."""
    trigger = KinematicGatedTrigger(hold_on_frames=45)

    # Establish upright baseline (bottom y = 400.0, standing height = 250.0)
    for t in range(5):
        s = make_dummy_snapshot(timestamp=float(t), bbox=(100.0, 150.0, 180.0, 400.0), torso_angle=5.0, hip_y=280.0)
        trigger.evaluate(track_id=2, snapshots=[s], kinematics=KinematicFeatures(0,0,0.3,0,5,0,0,0))

    # Person slowly slides down wall: bottom y is 400.0, but hip drops to 380.0 (distance to floor = 20px < 0.22*250 = 55px)
    # Vy and Ay remain low, but Ground Proximity triggers gate open!
    slow_slide_snap = make_dummy_snapshot(
        timestamp=6.0,
        bbox=(100.0, 310.0, 220.0, 400.0),  # Height collapsed to 90px
        torso_angle=20.0,
        hip_y=380.0,
    )
    low_kinematics = KinematicFeatures(
        vertical_velocity=0.30,  # Low!
        vertical_acceleration=0.15,  # Low!
        aspect_ratio_curr=0.80,  # < 1.15!
        aspect_ratio_change=0.20,
        torso_angle_curr=20.0,
        torso_angle_velocity=3.0,
        immobility_index=0.40,
        drop_severity_score=0.10,
    )

    res = trigger.evaluate(track_id=2, snapshots=[slow_slide_snap], kinematics=low_kinematics, current_state="NORMAL")
    assert res.should_run_ai is True
    assert "ground_proximity" in res.trigger_reason or "vertical_collapse" in res.trigger_reason
    assert res.hold_on_frames_remaining == 45


def test_gate_hold_on_timer_persists_during_immobility():
    """Vulnerability 2 check: Gate Hold-on Timer keeps AI active during post-fall immobility."""
    trigger = KinematicGatedTrigger(hold_on_frames=45)

    # 1. Trigger fall
    fall_snap = make_dummy_snapshot(timestamp=1.0, bbox=(100.0, 200.0, 350.0, 300.0), torso_angle=75.0, hip_y=280.0)
    kinematics = KinematicFeatures(1.5, 1.2, 2.5, 1.5, 75.0, 40.0, 0.0, 0.9)
    res1 = trigger.evaluate(track_id=3, snapshots=[fall_snap], kinematics=kinematics)
    assert res1.should_run_ai is True
    assert res1.hold_on_frames_remaining == 45

    # 2. Subsequent frame: Person lies completely still (Vy=0, Ay=0, tilt_vel=0)
    # The gate MUST stay open because of hold_on timer
    still_snap = make_dummy_snapshot(timestamp=1.1, bbox=(100.0, 200.0, 200.0, 300.0), torso_angle=10.0, hip_y=220.0)
    calm_kinematics = KinematicFeatures(0.0, 0.0, 0.5, 0.0, 10.0, 0.0, 0.95, 0.0)

    res2 = trigger.evaluate(track_id=3, snapshots=[still_snap], kinematics=calm_kinematics)
    assert res2.should_run_ai is True
    assert res2.hold_on_frames_remaining == 44
    assert "gate_hold_on" in res2.trigger_reason
