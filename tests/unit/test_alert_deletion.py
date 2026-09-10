"""Unit tests for single alert deletion and bulk clearance."""

import shutil
import tempfile
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from src.alerts.event_logger import EventLogger
from src.api.server import app


def test_event_logger_single_and_bulk_delete():
    """Test EventLogger.delete_event and clear_events."""
    temp_dir = Path(tempfile.mkdtemp())
    try:
        logger = EventLogger(log_dir=temp_dir)
        ev1 = logger.log_event(person_id=1, risk_score=0.92, evidence={"action": "fall"})
        ev2 = logger.log_event(person_id=2, risk_score=0.88, evidence={"action": "lying"})
        
        events = logger.get_recent_events(limit=10)
        assert len(events) == 2

        # Delete single event ev1
        deleted = logger.delete_event(ev1.event_id)
        assert deleted is True

        events_after = logger.get_recent_events(limit=10)
        assert len(events_after) == 1
        assert events_after[0]["event_id"] == ev2.event_id

        # Deleting non-existent event returns False
        assert logger.delete_event("non_existent_id") is False

        # Bulk clear
        logger.clear_events()
        assert len(logger.get_recent_events(limit=10)) == 0
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_api_single_delete_endpoint():
    """Test DELETE /events/{event_id} and POST /events/{event_id}/delete."""
    client = TestClient(app)
    # Clear first
    res_clear = client.post("/events/clear")
    assert res_clear.status_code == 200

    # Test delete endpoint on an ID
    res_del = client.delete("/events/evt_sample_123")
    assert res_del.status_code == 200
    data = res_del.json()
    assert data["status"] == "success"
    assert data["event_id"] == "evt_sample_123"
