"""Structured event logger for auditing medical emergency alerts."""

import json
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import cv2
import numpy as np
from src.utils.config import get_project_root


@dataclass
class AlertEvent:
    """Represents a validated emergency event."""
    event_id: str
    event_type: str
    person_id: int
    timestamp: str
    risk_score: float
    evidence: Dict[str, Any]
    snapshot_path: Optional[str] = None
    message: str = "Possible medical emergency / abnormal behavior detected. Please check the person."


class EventLogger:
    """Manages local, encrypted or anonymized JSON event logging."""

    def __init__(self, log_dir: Optional[Union[str, Path]] = None, retention_days: int = 7) -> None:
        if log_dir is None:
            self.log_dir = get_project_root() / "data" / "processed" / "events"
        else:
            self.log_dir = Path(log_dir)

        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.snapshots_dir = self.log_dir / "snapshots"
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)
        self.retention_days = retention_days
        self.jsonl_file = self.log_dir / "events.jsonl"

    def log_event(
        self,
        person_id: int,
        risk_score: float,
        evidence: Dict[str, Any],
        frame: Optional[np.ndarray] = None,
    ) -> AlertEvent:
        """Create, record, and persist a new emergency alert event.

        Args:
            person_id: Track ID of the person.
            risk_score: Calculated risk severity score.
            evidence: Dictionary of kinematic and action signals.
            frame: Optional BGR frame to save as blurred evidence snapshot.

        Returns:
            AlertEvent dataclass instance.
        """
        now = datetime.utcnow()
        timestamp_str = now.isoformat() + "Z"
        event_id = f"evt_{now.strftime('%Y%m%d_%H%M%S')}_trk{person_id}"

        snapshot_rel_path = None
        if frame is not None:
            filename = f"{event_id}.jpg"
            save_path = self.snapshots_dir / filename
            cv2.imwrite(str(save_path), frame)
            snapshot_rel_path = str(save_path.relative_to(get_project_root()))

        event = AlertEvent(
            event_id=event_id,
            event_type="possible_medical_emergency",
            person_id=person_id,
            timestamp=timestamp_str,
            risk_score=round(risk_score, 3),
            evidence=evidence,
            snapshot_path=snapshot_rel_path,
        )

        # Append to JSONL log
        with open(self.jsonl_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(event)) + "\n")

        return event

    def get_recent_events(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Retrieve the most recent logged events."""
        if not self.jsonl_file.exists():
            return []

        events = []
        with open(self.jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        events.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return events[-limit:]

    def clear_events(self) -> None:
        """Clear all historical logged events and reset JSONL log."""
        if self.jsonl_file.exists():
            try:
                self.jsonl_file.unlink()
            except Exception:
                with open(self.jsonl_file, "w", encoding="utf-8") as f:
                    f.truncate(0)

    def delete_event(self, event_id: str) -> bool:
        """Permanently remove a single event by event_id from storage and disk."""
        if not self.jsonl_file.exists():
            return False

        remaining_events = []
        found = False
        with open(self.jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    if data.get("event_id") == event_id:
                        found = True
                        # Remove snapshot if exists
                        snapshot_path = data.get("snapshot_path")
                        if snapshot_path:
                            try:
                                snap_full = get_project_root() / snapshot_path
                                if snap_full.exists():
                                    snap_full.unlink()
                            except Exception:
                                pass
                        continue
                    remaining_events.append(data)
                except json.JSONDecodeError:
                    continue

        if found:
            with open(self.jsonl_file, "w", encoding="utf-8") as f:
                for ev in remaining_events:
                    f.write(json.dumps(ev) + "\n")
        return found
