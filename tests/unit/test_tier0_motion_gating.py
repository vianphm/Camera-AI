"""Unit and integration tests for Tier-0 Motion Gating (Coarse Background Subtraction)."""

import numpy as np
import pytest
import cv2

from src.detection.motion_gater import MotionGater, MotionGateResult
from src.pipeline.realtime_pipeline import RealtimePipeline


def test_motion_gater_disabled():
    """Verify that disabling gating always allows AI and full FPS."""
    gater = MotionGater(enabled=False, active_fps=30.0)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    res = gater.evaluate(frame)
    assert res.should_run_ai is True
    assert res.motion_detected is True
    assert res.target_fps == 30.0
    assert res.is_throttled_frame is False


def test_motion_gater_static_room_shuts_off_ai_and_throttles_fps():
    """Verify that a static room shuts off AI (0% GPU) and throttles FPS to 10-15."""
    gater = MotionGater(
        enabled=True,
        method="mog2",
        min_motion_ratio=0.005,
        history=10,
        cooldown_frames=5,
        idle_fps=12.0,
        active_fps=30.0,
        idle_stride=2,
        periodic_check_interval=20,
    )

    static_frame = np.full((240, 320, 3), 128, dtype=np.uint8)

    # Feed 15 static frames to establish background and exhaust initial cooldown
    for _ in range(15):
        res = gater.evaluate(static_frame)

    # After initial stabilization and cooldown expiration, AI should be shut off
    assert res.motion_detected is False
    assert res.should_run_ai is False
    assert res.target_fps == 12.0
    assert "static" in res.rationale.lower() or res.status == "IDLE"

    # Verify throttling alternates every idle_stride frames
    res1 = gater.evaluate(static_frame)
    res2 = gater.evaluate(static_frame)
    assert res1.is_throttled_frame != res2.is_throttled_frame


def test_motion_gater_instant_wakeup_on_movement():
    """Verify that movement in the frame immediately wakes up AI to 30 FPS."""
    gater = MotionGater(
        enabled=True,
        method="mog2",
        min_motion_ratio=0.005,
        history=10,
        cooldown_frames=10,
        idle_fps=12.0,
        active_fps=30.0,
        idle_stride=2,
    )

    bg_frame = np.full((240, 320, 3), 100, dtype=np.uint8)
    for _ in range(15):
        gater.evaluate(bg_frame)

    # Introduce a moving bright block (simulating person walking into frame)
    moving_frame = bg_frame.copy()
    cv2.rectangle(moving_frame, (50, 50), (180, 180), (255, 255, 255), -1)

    res = gater.evaluate(moving_frame)
    assert res.motion_detected is True
    assert res.should_run_ai is True
    assert res.target_fps == 30.0
    assert res.is_throttled_frame is False
    assert res.cooldown_remaining == 10
    assert res.status == "ACTIVE"


def test_motion_gater_cooldown_hold_on():
    """Verify cooldown timer holds AI active when human pauses briefly."""
    cooldown = 6
    gater = MotionGater(
        enabled=True,
        method="mog2",
        min_motion_ratio=0.005,
        history=10,
        cooldown_frames=cooldown,
        idle_fps=12.0,
        active_fps=30.0,
    )

    bg = np.full((240, 320, 3), 100, dtype=np.uint8)
    for _ in range(12):
        gater.evaluate(bg)

    # 1. Trigger motion
    moving = bg.copy()
    cv2.rectangle(moving, (60, 60), (160, 160), (255, 255, 255), -1)
    res_active = gater.evaluate(moving)
    assert res_active.status == "ACTIVE"
    assert res_active.should_run_ai is True

    # 2. Movement stops (person pauses)
    # The gate must remain OPEN for all `cooldown` frames
    for rem in range(cooldown - 1, -1, -1):
        res_hold = gater.evaluate(bg)
        assert res_hold.should_run_ai is True
        assert res_hold.status == "COOLDOWN"
        assert res_hold.cooldown_remaining == rem

    # 3. Next frame after cooldown: Gate closes
    res_closed = gater.evaluate(bg)
    assert res_closed.should_run_ai is False
    assert res_closed.status == "IDLE"


def test_motion_gater_active_track_override():
    """Verify that if a person is actively tracked, AI remains ON even if motionless."""
    gater = MotionGater(
        enabled=True,
        method="mog2",
        cooldown_frames=2,
    )

    bg = np.full((240, 320, 3), 50, dtype=np.uint8)
    for _ in range(10):
        gater.evaluate(bg)

    # With has_active_tracks=True, AI must NOT be disabled
    res = gater.evaluate(bg, has_active_tracks=True)
    assert res.should_run_ai is True
    assert res.status == "OVERRIDE"
    assert "active tracks" in res.rationale.lower()


def test_motion_gater_periodic_keep_alive():
    """Verify anti-blindness periodic keep-alive check triggers every N idle frames."""
    interval = 8
    gater = MotionGater(
        enabled=True,
        method="mog2",
        cooldown_frames=0,
        periodic_check_interval=interval,
    )

    bg = np.full((240, 320, 3), 80, dtype=np.uint8)
    for _ in range(5):
        gater.evaluate(bg)
    gater._idle_frame_counter = 0

    # Run for interval - 1 frames -> IDLE
    for _ in range(interval - 1):
        res = gater.evaluate(bg)
        assert res.status == "IDLE"
        assert res.should_run_ai is False

    # Frame at interval threshold triggers periodic check
    res_keep_alive = gater.evaluate(bg)
    assert res_keep_alive.status == "KEEP_ALIVE"
    assert res_keep_alive.should_run_ai is True



def test_motion_gater_frame_differencing_method():
    """Verify alternative Frame Differencing algorithm."""
    gater = MotionGater(
        enabled=True,
        method="frame_diff",
        min_motion_ratio=0.005,
        cooldown_frames=3,
    )

    frame1 = np.full((240, 320, 3), 120, dtype=np.uint8)
    gater.evaluate(frame1)

    # Identical frame -> No motion
    res_static = gater.evaluate(frame1)
    assert res_static.motion_detected is False

    # Frame with large change -> Motion detected
    frame2 = frame1.copy()
    cv2.circle(frame2, (160, 120), 40, (0, 0, 0), -1)
    res_motion = gater.evaluate(frame2)
    assert res_motion.motion_detected is True
    assert res_motion.should_run_ai is True


def test_realtime_pipeline_tier0_integration():
    """Verify RealtimePipeline seamlessly integrates Tier-0 and reports telemetry."""
    custom_gater = MotionGater(
        enabled=True,
        method="mog2",
        cooldown_frames=2,
        idle_stride=2,
    )
    pipeline = RealtimePipeline(motion_gater=custom_gater)

    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Process several static frames
    res_list = [pipeline.process_frame(dummy_frame, timestamp=float(i)) for i in range(10)]

    # Later frames should be flagged as idle and AI bypassed
    last_res = res_list[-1]
    assert "tier0_status" in last_res.telemetry
    assert last_res.telemetry["ai_bypassed"] is True
    assert len(last_res.tracks) == 0
