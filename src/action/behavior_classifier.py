"""Behavior sequence analysis for tracking multi-stage transitions."""

from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class BehaviorStage:
    action: str
    duration_seconds: float
    confidence: float


class BehaviorSequenceAnalyzer:
    """Tracks sequential transitions between actions over extended time (e.g. 5-15 seconds)."""

    def __init__(self, history_len: int = 100) -> None:
        self.history_len = history_len
        self._action_histories: Dict[int, deque[str]] = {}
        self._timestamp_histories: Dict[int, deque[float]] = {}

    def update(self, track_id: int, action: str, timestamp: float) -> None:
        """Record latest action for track_id."""
        if track_id not in self._action_histories:
            self._action_histories[track_id] = deque(maxlen=self.history_len)
            self._timestamp_histories[track_id] = deque(maxlen=self.history_len)

        self._action_histories[track_id].append(action)
        self._timestamp_histories[track_id].append(timestamp)

    def detect_fall_and_collapse_pattern(self, track_id: int) -> bool:
        """Detect the signature pattern: (walking/standing) -> (stumbling/falling) -> (lying/immobile)."""
        if track_id not in self._action_histories:
            return False

        actions = list(self._action_histories[track_id])
        if len(actions) < 8:
            return False

        # Check recent actions (last 4 frames): must be lying or immobile
        recent = actions[-4:]
        is_currently_down = any(a in ["immobile", "falling", "lying"] for a in recent)
        if not is_currently_down:
            return False

        # Check middle actions: must have had falling or stumbling
        had_fall_transition = any(a in ["falling", "stumbling"] for a in actions)
        # Check earlier actions: was upright (standing or walking)
        earlier = actions[: len(actions) // 2]
        was_upright = any(a in ["standing", "walking"] for a in earlier)

        return had_fall_transition and is_currently_down and was_upright

    def get_duration_in_action(self, track_id: int, target_action: str) -> float:
        """Calculate continuous duration (seconds) track has remained in target_action."""
        if track_id not in self._action_histories:
            return 0.0

        actions = list(self._action_histories[track_id])
        timestamps = list(self._timestamp_histories[track_id])

        if not actions or actions[-1] != target_action:
            return 0.0

        start_time = timestamps[-1]
        for a, t in zip(reversed(actions), reversed(timestamps)):
            if a == target_action:
                start_time = t
            else:
                break

        return max(0.0, timestamps[-1] - start_time)

    def reset_track(self, track_id: int) -> None:
        self._action_histories.pop(track_id, None)
        self._timestamp_histories.pop(track_id, None)
