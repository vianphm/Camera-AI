"""Unit tests for Event State Machine debounce and hysteresis."""

import pytest
from src.risk.threshold_manager import ThresholdManager
from src.risk.state_machine import MonitorState, EventStateMachine


def test_state_machine_escalation_requires_duration():
    tm = ThresholdManager()
    sm = EventStateMachine(tm, initial_time=0.0)
    assert sm.state == MonitorState.NORMAL

    # High risk score on a SINGLE frame at t=0.1s -> should promote only to SUSPICIOUS
    state, changed = sm.update(risk_score=0.92, action="falling", timestamp=0.1)
    assert state == MonitorState.SUSPICIOUS
    assert changed

    # Immediately on next frame at t=0.2s (only 0.1s in state, min is 0.5s) -> should remain in SUSPICIOUS
    state, changed = sm.update(risk_score=0.92, action="falling", timestamp=0.2)
    assert state == MonitorState.SUSPICIOUS
    assert not changed

    # Once time reaches 0.7s (>0.5s debounce) -> promotes to ABNORMAL
    state, changed = sm.update(risk_score=0.92, action="falling", timestamp=0.7)
    assert state == MonitorState.ABNORMAL
    assert changed


def test_state_machine_recovery_on_getting_up():
    tm = ThresholdManager()
    sm = EventStateMachine(tm, initial_time=0.0)
    sm.state = MonitorState.SUSPICIOUS
    sm.state_entry_time = 0.0

    # Person stands up and risk drops to 0.1 at t=1.0s
    state, changed = sm.update(risk_score=0.10, action="getting_up", timestamp=1.0)
    assert state == MonitorState.SUSPICIOUS

    # Continues standing normally for > min_recovery_duration (3.0s) -> t=4.5s
    state, changed = sm.update(risk_score=0.05, action="standing", timestamp=4.5)
    assert state == MonitorState.NORMAL
    assert changed
