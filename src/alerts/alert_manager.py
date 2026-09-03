"""Alert Manager orchestrating event cooldowns, deduplication, and dispatch."""

import time
from typing import Dict, Optional
import numpy as np
from src.alerts.event_logger import EventLogger, AlertEvent
from src.alerts.notification import NotificationDispatcher
from src.risk.risk_engine import RiskAssessment
from src.risk.threshold_manager import ThresholdManager


class AlertManager:
    """Manages alert rate-limiting, cooldown timers, and notifications."""

    def __init__(
        self,
        thresholds: Optional[ThresholdManager] = None,
        logger: Optional[EventLogger] = None,
        dispatcher: Optional[NotificationDispatcher] = None,
    ) -> None:
        self.thresholds = thresholds or ThresholdManager()
        self.logger = logger or EventLogger()
        self.dispatcher = dispatcher or NotificationDispatcher()
        self._last_alert_timestamps: Dict[int, float] = {}

    def process_assessment(
        self,
        assessment: RiskAssessment,
        frame: Optional[np.ndarray] = None,
        timestamp: Optional[float] = None,
    ) -> Optional[AlertEvent]:
        """Check if assessment requires alert dispatch, respecting cooldown.

        Args:
            assessment: RiskAssessment produced by RiskEngine.
            frame: Optional frame for snapshot evidence.
            timestamp: Current timestamp in seconds.

        Returns:
            AlertEvent if dispatched, else None.
        """
        if not assessment.should_alert:
            return None

        curr_time = timestamp if timestamp is not None else time.time()
        tid = assessment.track_id

        # Cooldown check
        last_alert = self._last_alert_timestamps.get(tid, 0.0)
        if (curr_time - last_alert) < self.thresholds.cooldown_seconds:
            # Within cooldown, suppress duplicate notification
            return None

        self._last_alert_timestamps[tid] = curr_time

        # 1. Log event and save blurred snapshot
        event = self.logger.log_event(
            person_id=tid,
            risk_score=assessment.risk_score,
            evidence=assessment.evidence,
            frame=frame,
        )

        # 2. Dispatch notifications across external channels
        self.dispatcher.dispatch(event)

        return event

    def reset_cooldown(self, track_id: int) -> None:
        """Clear cooldown for a specific track ID."""
        self._last_alert_timestamps.pop(track_id, None)
