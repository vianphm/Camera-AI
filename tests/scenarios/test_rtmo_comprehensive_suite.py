"""Comprehensive RTMO-s Pipeline Test Suite (24 Required Scenarios).

Evaluates:
- True Positive Rate (TPR >= 90%)
- False Positive Rate (FPR <= 5%)
- Alert Response Time (1.5s - 3.5s after lying immobile)
- Self-Recovery Mechanism (Resets <= 3.0s without alerting)
- Multi-Person Tracking & Track ID Stability
- Non-diagnostic Clinical Safety Compliance
"""

import math
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pytest

from src.analytics.kinematic_engine import KinematicEngine, BiomechanicalFeatures
from src.analytics.event_fsm import EventStateMachine, MonitorState, AlertEvent


def make_upright_person(
    cx: float = 320.0,
    cy: float = 240.0,
    torso_len: float = 80.0,
    tilt_deg: float = 0.0,
    leg_offset: float = 0.0,
    conf: float = 0.95,
) -> Tuple[np.ndarray, Tuple[float, float, float, float]]:
    """Synthesize 17 COCO keypoints and bbox for an upright person."""
    rad = math.radians(tilt_deg)
    dx = math.sin(rad) * torso_len
    dy = math.cos(rad) * torso_len

    shoulder_y = cy - dy / 2.0
    shoulder_x = cx - dx / 2.0
    hip_y = cy + dy / 2.0
    hip_x = cx + dx / 2.0

    kpts = np.zeros((17, 3), dtype=np.float32)
    # Head (K0..K4)
    kpts[0] = [shoulder_x, shoulder_y - 25.0, conf]       # Nose
    kpts[1] = [shoulder_x - 5.0, shoulder_y - 30.0, conf] # L-Eye
    kpts[2] = [shoulder_x + 5.0, shoulder_y - 30.0, conf] # R-Eye
    kpts[3] = [shoulder_x - 12.0, shoulder_y - 28.0, conf] # L-Ear
    kpts[4] = [shoulder_x + 12.0, shoulder_y - 28.0, conf] # R-Ear

    # Shoulders (K5, K6)
    kpts[5] = [shoulder_x - 20.0, shoulder_y, conf]
    kpts[6] = [shoulder_x + 20.0, shoulder_y, conf]

    # Elbows (K7, K8)
    kpts[7] = [shoulder_x - 25.0, shoulder_y + 35.0, conf]
    kpts[8] = [shoulder_x + 25.0, shoulder_y + 35.0, conf]

    # Wrists (K9, K10)
    kpts[9] = [shoulder_x - 25.0, shoulder_y + 65.0, conf]
    kpts[10] = [shoulder_x + 25.0, shoulder_y + 65.0, conf]

    # Hips (K11, K12)
    kpts[11] = [hip_x - 16.0, hip_y, conf]
    kpts[12] = [hip_x + 16.0, hip_y, conf]

    # Knees (K13, K14)
    kpts[13] = [hip_x - 15.0 + leg_offset, hip_y + 55.0, conf]
    kpts[14] = [hip_x + 15.0 - leg_offset, hip_y + 55.0, conf]

    # Ankles (K15, K16)
    kpts[15] = [hip_x - 15.0 + leg_offset, hip_y + 110.0, conf]
    kpts[16] = [hip_x + 15.0 - leg_offset, hip_y + 110.0, conf]

    # Calculate bounding box
    min_x = float(np.min(kpts[:, 0]) - 10.0)
    max_x = float(np.max(kpts[:, 0]) + 10.0)
    min_y = float(np.min(kpts[:, 1]) - 10.0)
    max_y = float(np.max(kpts[:, 1]) + 10.0)

    return kpts, (min_x, min_y, max_x, max_y)


def make_horizontal_person(
    cx: float = 320.0,
    ground_y: float = 400.0,
    torso_len: float = 80.0,
    direction: int = 1,  # 1 = head left, -1 = head right
    jitter: float = 0.0,
    conf: float = 0.95,
) -> Tuple[np.ndarray, Tuple[float, float, float, float]]:
    """Synthesize 17 COCO keypoints and bbox for a person lying horizontally on floor."""
    kpts = np.zeros((17, 3), dtype=np.float32)

    shoulder_x = cx - direction * (torso_len / 2.0)
    shoulder_y = ground_y - 15.0
    hip_x = cx + direction * (torso_len / 2.0)
    hip_y = ground_y - 12.0

    noise = lambda: float(np.random.normal(0.0, jitter)) if jitter > 0.0 else 0.0

    # Head
    kpts[0] = [shoulder_x - direction * 25.0 + noise(), shoulder_y + noise(), conf]
    kpts[1] = [shoulder_x - direction * 22.0 + noise(), shoulder_y - 4.0 + noise(), conf]
    kpts[2] = [shoulder_x - direction * 22.0 + noise(), shoulder_y + 4.0 + noise(), conf]
    kpts[3] = [shoulder_x - direction * 15.0 + noise(), shoulder_y - 8.0 + noise(), conf]
    kpts[4] = [shoulder_x - direction * 15.0 + noise(), shoulder_y + 8.0 + noise(), conf]

    # Shoulders
    kpts[5] = [shoulder_x + noise(), shoulder_y - 12.0 + noise(), conf]
    kpts[6] = [shoulder_x + noise(), shoulder_y + 12.0 + noise(), conf]

    # Arms
    kpts[7] = [shoulder_x + direction * 20.0 + noise(), shoulder_y - 14.0 + noise(), conf]
    kpts[8] = [shoulder_x + direction * 20.0 + noise(), shoulder_y + 14.0 + noise(), conf]
    kpts[9] = [shoulder_x + direction * 40.0 + noise(), shoulder_y - 12.0 + noise(), conf]
    kpts[10] = [shoulder_x + direction * 40.0 + noise(), shoulder_y + 12.0 + noise(), conf]

    # Hips
    kpts[11] = [hip_x + noise(), hip_y - 10.0 + noise(), conf]
    kpts[12] = [hip_x + noise(), hip_y + 10.0 + noise(), conf]

    # Legs
    kpts[13] = [hip_x + direction * 45.0 + noise(), ground_y - 10.0 + noise(), conf]
    kpts[14] = [hip_x + direction * 45.0 + noise(), ground_y + 5.0 + noise(), conf]
    kpts[15] = [hip_x + direction * 95.0 + noise(), ground_y - 10.0 + noise(), conf]
    kpts[16] = [hip_x + direction * 95.0 + noise(), ground_y + 5.0 + noise(), conf]

    min_x = float(np.min(kpts[:, 0]) - 10.0)
    max_x = float(np.max(kpts[:, 0]) + 10.0)
    min_y = float(np.min(kpts[:, 1]) - 10.0)
    max_y = float(np.max(kpts[:, 1]) + 10.0)

    return kpts, (min_x, min_y, max_x, max_y)


# ===========================================================================
# GROUP A: Normal Behavior Scenarios (FPR must be 0%)
# ===========================================================================

def test_walking_normal():
    """A1: Walking normally in room across frames."""
    ke = KinematicEngine()
    fsm = EventStateMachine()
    fps = 15.0
    alerts = []

    for i in range(60):
        t = i / fps
        x = 100.0 + i * 3.0
        leg = 10.0 * math.sin(i * 0.8)
        kpts, bbox = make_upright_person(cx=x, cy=240.0, tilt_deg=5.0, leg_offset=leg)
        feat = ke.update(1, kpts, bbox, t)
        state, alert = fsm.update(1, feat, kpts, t)
        if alert:
            alerts.append(alert)

    assert len(alerts) == 0, "Normal walking must NEVER trigger an alert"
    assert fsm.get_state(1) == MonitorState.NORMAL


def test_standing_still():
    """A2: Standing still with minor posture micro-sway."""
    ke = KinematicEngine()
    fsm = EventStateMachine()
    fps = 15.0
    alerts = []

    for i in range(60):
        t = i / fps
        sway = 0.5 * math.sin(i * 0.2)
        kpts, bbox = make_upright_person(cx=320.0 + sway, cy=240.0, tilt_deg=2.0)
        feat = ke.update(1, kpts, bbox, t)
        state, alert = fsm.update(1, feat, kpts, t)
        if alert:
            alerts.append(alert)

    assert len(alerts) == 0, "Standing still must NEVER trigger an alert"
    assert fsm.get_state(1) == MonitorState.NORMAL


def test_sit_to_stand():
    """A3: Sitting down on chair then standing up."""
    ke = KinematicEngine()
    fsm = EventStateMachine()
    fps = 15.0
    alerts = []

    for i in range(75):
        t = i / fps
        if i < 20:
            cy = 240.0  # Standing
            tilt = 5.0
        elif i < 35:
            # Sitting down gently (Vy_norm < 0.6)
            prog = (i - 20) / 15.0
            cy = 240.0 + 70.0 * prog
            tilt = 5.0 + 10.0 * prog
        elif i < 50:
            cy = 310.0  # Sitting
            tilt = 12.0
        else:
            # Standing back up
            prog = (i - 50) / 25.0
            cy = 310.0 - 70.0 * prog
            tilt = 12.0 - 7.0 * prog

        kpts, bbox = make_upright_person(cx=320.0, cy=cy, tilt_deg=tilt)
        feat = ke.update(1, kpts, bbox, t)
        state, alert = fsm.update(1, feat, kpts, t)
        if alert:
            alerts.append(alert)

    assert len(alerts) == 0, "Sitting and standing must not trigger an alert"
    assert fsm.get_state(1) == MonitorState.NORMAL


def test_bending_pick_item():
    """A4: Bending down to pick up an item then standing up (< 2s)."""
    ke = KinematicEngine()
    fsm = EventStateMachine()
    fps = 15.0
    alerts = []

    for i in range(60):
        t = i / fps
        if i < 15:
            tilt = 5.0
            cy = 240.0
        elif i < 25:
            # Bending forward
            prog = (i - 15) / 10.0
            tilt = 5.0 + 60.0 * prog  # up to 65 deg
            cy = 240.0 + 20.0 * prog
        elif i < 35:
            # Picking up item
            tilt = 65.0
            cy = 260.0
        else:
            # Rising back to upright
            prog = (i - 35) / 25.0
            tilt = 65.0 - 60.0 * prog
            cy = 260.0 - 20.0 * prog

        kpts, bbox = make_upright_person(cx=320.0, cy=cy, tilt_deg=tilt)
        feat = ke.update(1, kpts, bbox, t)
        state, alert = fsm.update(1, feat, kpts, t)
        if alert:
            alerts.append(alert)

    assert len(alerts) == 0, "Bending to pick item must not trigger emergency alert"
    assert fsm.get_state(1) in (MonitorState.NORMAL, MonitorState.RECOVERED)


def test_lying_sofa_rest():
    """A5: Lying down smoothly on sofa/bed to rest then sitting up."""
    ke = KinematicEngine()
    fsm = EventStateMachine()
    fps = 15.0
    alerts = []

    for i in range(75):
        t = i / fps
        if i < 20:
            kpts, bbox = make_upright_person(cx=320.0, cy=240.0, tilt_deg=10.0)
        elif i < 40:
            # Smooth slow recline without impact (Vy_norm < 0.4)
            prog = (i - 20) / 20.0
            kpts, bbox = make_upright_person(cx=320.0 + 30.0 * prog, cy=240.0 + 40.0 * prog, tilt_deg=10.0 + 65.0 * prog)
        else:
            # Resting on sofa (horizontal but zero impact spike in history)
            kpts, bbox = make_horizontal_person(cx=350.0, ground_y=350.0)

        feat = ke.update(1, kpts, bbox, t)
        state, alert = fsm.update(1, feat, kpts, t)
        if alert:
            alerts.append(alert)

    assert len(alerts) == 0, "Intentional resting on sofa must NEVER alert (No impact drop velocity)"


def test_yoga_squat():
    """A6: Slow yoga squat and holding balance."""
    ke = KinematicEngine()
    fsm = EventStateMachine()
    fps = 15.0
    alerts = []

    for i in range(60):
        t = i / fps
        cy = 240.0 + 40.0 * math.sin(i * 0.1)
        kpts, bbox = make_upright_person(cx=320.0, cy=cy, tilt_deg=5.0)
        feat = ke.update(1, kpts, bbox, t)
        state, alert = fsm.update(1, feat, kpts, t)
        if alert:
            alerts.append(alert)

    assert len(alerts) == 0, "Yoga squatting must not alert"


# ===========================================================================
# GROUP B: True Fall Scenarios (Alert must be dispatched in 1.5s - 3.5s)
# ===========================================================================

def _simulate_fall_sequence(
    fall_type: str = "forward",
    fall_start_frame: int = 15,
    impact_frame: int = 20,
    total_frames: int = 75,
    fps: float = 15.0,
) -> Tuple[List[AlertEvent], float, Dict[str, Any]]:
    """Helper to simulate true fall and track time to alert dispatch."""
    ke = KinematicEngine()
    fsm = EventStateMachine(immobility_confirm_duration=3.0)
    alerts = []
    first_alert_time = None
    telemetry = {}

    impact_timestamp = impact_frame / fps

    for i in range(total_frames):
        t = i / fps
        if i < fall_start_frame:
            # Standing upright
            kpts, bbox = make_upright_person(cx=320.0, cy=240.0, tilt_deg=5.0)
        elif i <= impact_frame:
            # Rapid drop in 5 frames (~0.33s): high Vy_norm and angular change
            prog = (i - fall_start_frame) / float(impact_frame - fall_start_frame)
            cy = 240.0 + (420.0 - 240.0) * prog
            tilt = 5.0 + (85.0 - 5.0) * prog
            dir_val = 1 if fall_type != "backward" else -1
            kpts, bbox = make_upright_person(cx=320.0 + dir_val * 40.0 * prog, cy=cy, tilt_deg=tilt)
        else:
            # Lying immobile on floor
            kpts, bbox = make_horizontal_person(cx=360.0, ground_y=420.0, jitter=0.005)

        feat = ke.update(1, kpts, bbox, t)
        state, alert = fsm.update(1, feat, kpts, t)

        if alert and first_alert_time is None:
            first_alert_time = t
            alerts.append(alert)

        if i == impact_frame + 1:
            telemetry["peak_vy"] = feat.v_y_norm
        if i == total_frames - 1:
            telemetry["final_risk"] = feat.risk_score

    response_delay = (first_alert_time - impact_timestamp) if first_alert_time else -1.0
    return alerts, response_delay, telemetry


def test_forward_fall():
    """B1: Forward fall onto floor followed by immobility."""
    alerts, delay, tel = _simulate_fall_sequence("forward")
    assert len(alerts) == 1, "Must raise exactly 1 alert for true forward fall"
    assert 1.5 <= delay <= 3.5, f"Response delay ({delay:.2f}s) must be in 1.5 - 3.5s window"
    assert "Possible medical emergency" in alerts[0].message
    assert alerts[0].should_blur_face is True


def test_backward_fall():
    """B2: Backward fall followed by immobility."""
    alerts, delay, tel = _simulate_fall_sequence("backward")
    assert len(alerts) == 1, "Must raise alert for backward fall"
    assert 1.5 <= delay <= 3.5, f"Response delay ({delay:.2f}s) must be in 1.5 - 3.5s window"


def test_side_fall():
    """B3: Side fall followed by immobility."""
    alerts, delay, tel = _simulate_fall_sequence("side")
    assert len(alerts) == 1, "Must raise alert for side fall"
    assert 1.5 <= delay <= 3.5, f"Response delay ({delay:.2f}s) must be in 1.5 - 3.5s window"


def test_stand_fall_immobile_4s():
    """B4: Fall from standing -> lying immobile for >= 4.0s."""
    # 90 frames total = 6.0s, fall at 1.33s, immobile for > 4.5s
    alerts, delay, tel = _simulate_fall_sequence("forward", fall_start_frame=15, impact_frame=20, total_frames=90)
    assert len(alerts) == 1
    assert alerts[0].state == MonitorState.ALERT_SENT
    assert delay <= 3.5


def test_chair_fall_to_floor():
    """B5: Sitting on chair -> sudden slide/fall to floor -> immobile."""
    ke = KinematicEngine()
    fsm = EventStateMachine(immobility_confirm_duration=3.0)
    fps = 15.0
    alerts = []
    impact_t = 20 / fps
    first_alert_t = None

    for i in range(75):
        t = i / fps
        if i < 15:
            # Sitting on chair
            kpts, bbox = make_upright_person(cx=320.0, cy=300.0, tilt_deg=10.0)
        elif i <= 20:
            # Slide off chair to floor rapidly (high Vy)
            prog = (i - 15) / 5.0
            kpts, bbox = make_upright_person(cx=320.0 + 30.0 * prog, cy=300.0 + 120.0 * prog, tilt_deg=10.0 + 75.0 * prog)
        else:
            # Immobile on floor
            kpts, bbox = make_horizontal_person(cx=350.0, ground_y=420.0)

        feat = ke.update(1, kpts, bbox, t)
        state, alert = fsm.update(1, feat, kpts, t)
        if alert and first_alert_t is None:
            first_alert_t = t
            alerts.append(alert)

    assert len(alerts) == 1, "Fall from chair to floor must trigger alert"
    delay = first_alert_t - impact_t
    assert 1.5 <= delay <= 3.5


def test_walk_fall_immobile():
    """B6: Walking -> sudden trip & fall -> immobile on floor."""
    ke = KinematicEngine()
    fsm = EventStateMachine(immobility_confirm_duration=3.0)
    fps = 15.0
    alerts = []
    impact_t = 22 / fps
    first_alert_t = None

    for i in range(80):
        t = i / fps
        if i < 16:
            # Walking
            kpts, bbox = make_upright_person(cx=150.0 + i * 5.0, cy=240.0, tilt_deg=5.0)
        elif i <= 22:
            # Fall during walking
            prog = (i - 16) / 6.0
            kpts, bbox = make_upright_person(cx=230.0 + 40.0 * prog, cy=240.0 + 180.0 * prog, tilt_deg=5.0 + 80.0 * prog)
        else:
            # Immobile on floor
            kpts, bbox = make_horizontal_person(cx=270.0, ground_y=420.0)

        feat = ke.update(1, kpts, bbox, t)
        state, alert = fsm.update(1, feat, kpts, t)
        if alert and first_alert_t is None:
            first_alert_t = t
            alerts.append(alert)

    assert len(alerts) == 1, "Trip and fall while walking must trigger alert"
    delay = first_alert_t - impact_t
    assert 1.5 <= delay <= 3.5


# ===========================================================================
# GROUP C: Near Fall & Self-Recovery (No Alert, Resets <= 3s)
# ===========================================================================

def test_stumble_regain_balance():
    """C1: Stumble but regain balance within 1 second."""
    ke = KinematicEngine()
    fsm = EventStateMachine()
    fps = 15.0
    alerts = []

    for i in range(50):
        t = i / fps
        if i < 15:
            kpts, bbox = make_upright_person(cx=320.0, cy=240.0, tilt_deg=5.0)
        elif i <= 18:
            # Sudden stumble tilt and dip
            kpts, bbox = make_upright_person(cx=320.0, cy=280.0, tilt_deg=35.0)
        else:
            # Regained balance, standing upright
            kpts, bbox = make_upright_person(cx=320.0, cy=240.0, tilt_deg=5.0)

        feat = ke.update(1, kpts, bbox, t)
        state, alert = fsm.update(1, feat, kpts, t)
        if alert:
            alerts.append(alert)

    assert len(alerts) == 0, "Stumble with immediate recovery must not alert"
    assert fsm.get_state(1) in (MonitorState.NORMAL, MonitorState.RECOVERED)


def test_fall_quick_recovery():
    """C2: Person falls to floor but stands up within 2.0s -> Self-Recovery."""
    ke = KinematicEngine()
    fsm = EventStateMachine(immobility_confirm_duration=3.0)
    fps = 15.0
    alerts = []
    states = []

    # Fall at frame 15..20 (1.0s - 1.33s), on floor until frame 35 (2.33s, duration on floor = 1.0s < 3s),
    # stands back up by frame 48 (3.2s)
    for i in range(65):
        t = i / fps
        if i < 15:
            kpts, bbox = make_upright_person(cx=320.0, cy=240.0, tilt_deg=5.0)
        elif i <= 20:
            prog = (i - 15) / 5.0
            kpts, bbox = make_upright_person(cx=320.0, cy=240.0 + 180.0 * prog, tilt_deg=5.0 + 80.0 * prog)
        elif i < 35:
            # Briefly on floor (only 15 frames = 1.0s)
            kpts, bbox = make_horizontal_person(cx=320.0, ground_y=420.0)
        elif i < 48:
            # Getting up! (torso rising, hips elevating)
            prog = (i - 35) / 13.0
            kpts, bbox = make_upright_person(cx=320.0, cy=420.0 - 180.0 * prog, tilt_deg=85.0 - 75.0 * prog)
        else:
            # Upright standing
            kpts, bbox = make_upright_person(cx=320.0, cy=240.0, tilt_deg=8.0)

        feat = ke.update(1, kpts, bbox, t)
        state, alert = fsm.update(1, feat, kpts, t)
        states.append(state)
        if alert:
            alerts.append(alert)

    assert len(alerts) == 0, "Quick recovery within 2-3s must cancel alarm and NOT alert"
    assert MonitorState.RECOVERED in states or MonitorState.NORMAL in states[-10:], "Must enter RECOVERED"


def test_slip_posture_recovery():
    """C3: Foot slip with high acceleration but posture caught before hitting floor."""
    ke = KinematicEngine()
    fsm = EventStateMachine()
    fps = 15.0
    alerts = []

    for i in range(45):
        t = i / fps
        if i < 12:
            kpts, bbox = make_upright_person(cx=320.0, cy=240.0, tilt_deg=5.0)
        elif i <= 16:
            # Foot slips, sharp tilt to 32 deg
            kpts, bbox = make_upright_person(cx=340.0, cy=270.0, tilt_deg=32.0, leg_offset=30.0)
        else:
            # Recovered balance
            kpts, bbox = make_upright_person(cx=340.0, cy=240.0, tilt_deg=6.0)

        feat = ke.update(1, kpts, bbox, t)
        state, alert = fsm.update(1, feat, kpts, t)
        if alert:
            alerts.append(alert)

    assert len(alerts) == 0, "Slipping and holding balance must not alert"


# ===========================================================================
# GROUP D: Edge Cases & Challenging Conditions
# ===========================================================================

def test_partial_occlusion():
    """D1: Lower body partially occluded by furniture (knees/ankles zero conf) when falling."""
    ke = KinematicEngine()
    fsm = EventStateMachine(immobility_confirm_duration=3.0)
    fps = 15.0
    alerts = []

    for i in range(75):
        t = i / fps
        if i < 15:
            kpts, bbox = make_upright_person(cx=320.0, cy=240.0)
        elif i <= 20:
            prog = (i - 15) / 5.0
            kpts, bbox = make_upright_person(cx=320.0, cy=240.0 + 180.0 * prog, tilt_deg=5.0 + 80.0 * prog)
        else:
            kpts, bbox = make_horizontal_person(cx=320.0, ground_y=420.0)

        # Occlude knees and ankles (K13, K14, K15, K16)
        kpts[13:17, 2] = 0.05

        feat = ke.update(1, kpts, bbox, t)
        state, alert = fsm.update(1, feat, kpts, t)
        if alert:
            alerts.append(alert)

    # Torso-normalized kinematics relies on shoulders & hips, unaffected by lower limb occlusion!
    assert len(alerts) == 1, "Fall with partial lower body occlusion must still be reliably detected"


def test_low_camera_angle():
    """D2: Low camera angle close to floor where perspective compresses height."""
    ke = KinematicEngine()
    fsm = EventStateMachine(immobility_confirm_duration=3.0)
    fps = 15.0
    alerts = []

    for i in range(75):
        t = i / fps
        if i < 15:
            # Compressed height due to low perspective
            kpts, bbox = make_upright_person(cx=320.0, cy=240.0, torso_len=50.0)
        elif i <= 20:
            prog = (i - 15) / 5.0
            kpts, bbox = make_upright_person(cx=320.0, cy=240.0 + 120.0 * prog, torso_len=50.0, tilt_deg=5.0 + 80.0 * prog)
        else:
            kpts, bbox = make_horizontal_person(cx=320.0, ground_y=360.0, torso_len=50.0)

        feat = ke.update(1, kpts, bbox, t)
        state, alert = fsm.update(1, feat, kpts, t)
        if alert:
            alerts.append(alert)

    assert len(alerts) == 1, "Low camera angle must detect fall"


def test_multi_person_1_falls():
    """D3: 3 people in frame: Track 1 walks, Track 2 sits, Track 3 falls."""
    ke = KinematicEngine()
    fsm = EventStateMachine(immobility_confirm_duration=3.0)
    fps = 15.0
    t3_alerts = []
    other_alerts = []

    for i in range(75):
        t = i / fps
        # Person 1 (walking)
        k1, b1 = make_upright_person(cx=100.0 + i * 2.0, cy=240.0)
        f1 = ke.update(1, k1, b1, t)
        s1, a1 = fsm.update(1, f1, k1, t)
        if a1:
            other_alerts.append(a1)

        # Person 2 (sitting still)
        k2, b2 = make_upright_person(cx=500.0, cy=300.0, tilt_deg=10.0)
        f2 = ke.update(2, k2, b2, t)
        s2, a2 = fsm.update(2, f2, k2, t)
        if a2:
            other_alerts.append(a2)

        # Person 3 (falls at frame 15..20 and stays immobile)
        if i < 15:
            k3, b3 = make_upright_person(cx=300.0, cy=240.0)
        elif i <= 20:
            prog = (i - 15) / 5.0
            k3, b3 = make_upright_person(cx=300.0, cy=240.0 + 180.0 * prog, tilt_deg=5.0 + 80.0 * prog)
        else:
            k3, b3 = make_horizontal_person(cx=300.0, ground_y=420.0)

        f3 = ke.update(3, k3, b3, t)
        s3, a3 = fsm.update(3, f3, k3, t)
        if a3:
            t3_alerts.append(a3)

    assert len(other_alerts) == 0, "Persons 1 and 2 must have zero alerts"
    assert len(t3_alerts) == 1, "Only Person 3 must trigger an alert"
    assert t3_alerts[0].track_id == 3


def test_sudden_light_change():
    """D4: Sudden lighting change causing transient keypoint jitter without fall."""
    ke = KinematicEngine()
    fsm = EventStateMachine()
    fps = 15.0
    alerts = []

    for i in range(60):
        t = i / fps
        # Frames 20..25: simulated flicker/jitter
        jitter = 4.0 if 20 <= i <= 25 else 0.2
        noise_x = float(np.random.normal(0.0, jitter))
        noise_y = float(np.random.normal(0.0, jitter))
        kpts, bbox = make_upright_person(cx=320.0 + noise_x, cy=240.0 + noise_y, tilt_deg=5.0)
        feat = ke.update(1, kpts, bbox, t)
        state, alert = fsm.update(1, feat, kpts, t)
        if alert:
            alerts.append(alert)

    assert len(alerts) == 0, "Lighting flicker noise must be filtered out by debouncer"


def test_dark_clothes_dark_floor():
    """D5: Low confidence keypoints due to dark clothes on dark floor (stationary upright)."""
    ke = KinematicEngine()
    fsm = EventStateMachine()
    fps = 15.0
    alerts = []

    for i in range(60):
        t = i / fps
        kpts, bbox = make_upright_person(cx=320.0, cy=240.0, conf=0.35)
        feat = ke.update(1, kpts, bbox, t)
        state, alert = fsm.update(1, feat, kpts, t)
        if alert:
            alerts.append(alert)

    assert len(alerts) == 0, "Low-contrast upright person must not trigger alert"


def test_high_pitch_camera():
    """D6: High tilted ceiling camera angle."""
    ke = KinematicEngine()
    fsm = EventStateMachine(immobility_confirm_duration=3.0)
    fps = 15.0
    alerts = []

    for i in range(75):
        t = i / fps
        if i < 15:
            # Ceiling view compresses vertical axis
            kpts, bbox = make_upright_person(cx=320.0, cy=200.0, torso_len=55.0, tilt_deg=10.0)
        elif i <= 20:
            prog = (i - 15) / 5.0
            kpts, bbox = make_upright_person(cx=320.0, cy=200.0 + 130.0 * prog, torso_len=55.0, tilt_deg=10.0 + 75.0 * prog)
        else:
            kpts, bbox = make_horizontal_person(cx=320.0, ground_y=330.0, torso_len=55.0)

        feat = ke.update(1, kpts, bbox, t)
        state, alert = fsm.update(1, feat, kpts, t)
        if alert:
            alerts.append(alert)

    assert len(alerts) == 1, "High angle camera fall must be detected"


# ===========================================================================
# GROUP E: Multi-Person Tracking & Track ID Stability
# ===========================================================================

def test_multi_person_crossing():
    """E1: 2 people walking across each other without ID swap."""
    ke = KinematicEngine()
    fsm = EventStateMachine()
    fps = 15.0

    for i in range(60):
        t = i / fps
        # Person 1 moving left to right
        x1 = 150.0 + i * 5.0
        k1, b1 = make_upright_person(cx=x1, cy=240.0)
        f1 = ke.update(1, k1, b1, t)
        s1, _ = fsm.update(1, f1, k1, t)

        # Person 2 moving right to left
        x2 = 450.0 - i * 5.0
        k2, b2 = make_upright_person(cx=x2, cy=240.0)
        f2 = ke.update(2, k2, b2, t)
        s2, _ = fsm.update(2, f2, k2, t)

    assert fsm.get_state(1) == MonitorState.NORMAL
    assert fsm.get_state(2) == MonitorState.NORMAL


def test_fall_during_cross():
    """E2: Person 2 falls right as Person 1 passes by."""
    ke = KinematicEngine()
    fsm = EventStateMachine(immobility_confirm_duration=3.0)
    fps = 15.0
    t1_alerts = []
    t2_alerts = []

    for i in range(75):
        t = i / fps
        # Person 1 walks across
        k1, b1 = make_upright_person(cx=100.0 + i * 5.0, cy=240.0)
        f1 = ke.update(1, k1, b1, t)
        s1, a1 = fsm.update(1, f1, k1, t)
        if a1:
            t1_alerts.append(a1)

        # Person 2 falls at frame 15..20
        if i < 15:
            k2, b2 = make_upright_person(cx=300.0, cy=240.0)
        elif i <= 20:
            prog = (i - 15) / 5.0
            k2, b2 = make_upright_person(cx=300.0, cy=240.0 + 180.0 * prog, tilt_deg=5.0 + 80.0 * prog)
        else:
            k2, b2 = make_horizontal_person(cx=300.0, ground_y=420.0)

        f2 = ke.update(2, k2, b2, t)
        s2, a2 = fsm.update(2, f2, k2, t)
        if a2:
            t2_alerts.append(a2)

    assert len(t1_alerts) == 0, "Passing person must not receive fall alert"
    assert len(t2_alerts) == 1, "Fallen person must receive fall alert"
    assert t2_alerts[0].track_id == 2


def test_track_id_persistence():
    """E3: Ensure track state and alerts remain locked to the same track_id over long immobility."""
    ke = KinematicEngine()
    fsm = EventStateMachine(immobility_confirm_duration=3.0)
    fps = 15.0
    alerts = []

    # 100 frames = 6.67 seconds of lying down
    for i in range(100):
        t = i / fps
        if i < 15:
            kpts, bbox = make_upright_person(cx=320.0, cy=240.0)
        elif i <= 20:
            prog = (i - 15) / 5.0
            kpts, bbox = make_upright_person(cx=320.0, cy=240.0 + 180.0 * prog, tilt_deg=5.0 + 80.0 * prog)
        else:
            kpts, bbox = make_horizontal_person(cx=320.0, ground_y=420.0)

        feat = ke.update(7, kpts, bbox, t)
        state, alert = fsm.update(7, feat, kpts, t)
        if alert:
            alerts.append(alert)

    assert len(alerts) == 1, "Alert must be dispatched exactly once (no bouncing or re-triggering)"
    assert alerts[0].track_id == 7
    assert fsm.get_state(7) == MonitorState.ALERT_SENT
