"""
Unit tests for event deduplication.

Tests:
- test_duplicate_event_id_rejected: Verify unique constraint on event_id prevents duplicate revenue
- test_event_dedup_metric_incremented: Verify duplicate_events_rejected counter increments
- test_integration_zero_duplicates_7d: Integration test for zero duplicates over 7-day window
"""

import pytest
from datetime import datetime, timedelta
from core.identity import IdentityResolutionAgent
from schemas.event import EventSchema
from sqlalchemy.exc import IntegrityError


class TestEventDeduplication:
    """Test suite for event_id deduplication (Q_001)"""

    @pytest.fixture
    def identity_agent(self, db_session):
        """Fixture to provide IdentityResolutionAgent with test DB"""
        return IdentityResolutionAgent(session=db_session)

    @pytest.fixture
    def sample_event(self):
        """Sample event payload"""
        return {
            "event_id": "evt_test_001",
            "event_type": "order_completed",
            "user_key": "uhash_abc123",
            "timestamp": datetime.utcnow().isoformat(),
            "revenue": 19800,
            "currency": "JPY",
            "order_id": "o123",
            "items": [{"sku": "SKU-1", "qty": 1, "price": 19800}]
        }

    def test_duplicate_event_id_rejected(self, identity_agent, sample_event, db_session):
        """
        ACCEPTANCE: unit: test_event_dedup.py::test_duplicate_event_id_rejected passes
        RISK_GATE: idempotency_key=event_id, DB unique constraint events(event_id)
        """
        # First submission should succeed
        result1 = identity_agent.ingest_event(sample_event)
        assert result1["ok"] is True
        assert result1["event_id"] == "evt_test_001"
        db_session.commit()

        # Second submission with same event_id should fail
        with pytest.raises(IntegrityError) as exc_info:
            identity_agent.ingest_event(sample_event)
            db_session.commit()

        # Verify it's the unique constraint violation
        assert "events_event_id_unique" in str(exc_info.value)
        db_session.rollback()

    def test_event_dedup_metric_incremented(self, identity_agent, sample_event, metrics_client):
        """
        ACCEPTANCE: metric: duplicate_events_rejected counter >0 on retry
        """
        # First submission
        identity_agent.ingest_event(sample_event)

        # Reset metrics counter
        initial_count = metrics_client.get_counter("duplicate_events_rejected")

        # Second submission (should be rejected and metric incremented)
        try:
            identity_agent.ingest_event(sample_event)
        except IntegrityError:
            pass  # Expected

        final_count = metrics_client.get_counter("duplicate_events_rejected")
        assert final_count > initial_count, "Metric duplicate_events_rejected should increment on duplicate"

    def test_api_endpoint_returns_409_on_duplicate(self, client, sample_event):
        """
        DRY_RUN_PROBE: Submit same event_id twice to /ingest/orders endpoint; 
        assert 2nd returns 409 Conflict
        """
        # First POST should succeed (201 Created)
        response1 = client.post("/api/v2/ingest/orders", json=sample_event)
        assert response1.status_code == 201
        assert response1.json()["ok"] is True

        # Second POST with same event_id should return 409 Conflict
        response2 = client.post("/api/v2/ingest/orders", json=sample_event)
        assert response2.status_code == 409
        assert "duplicate event_id" in response2.json()["error"].lower()

    @pytest.mark.integration
    def test_integration_zero_duplicates_7d(self, db_session):
        """
        ACCEPTANCE: integration: zero duplicate events in fct_order over 7d window
        """
        cutoff = datetime.utcnow() - timedelta(days=7)

        # Query for duplicate event_ids in fct_order table
        duplicate_query = """
        SELECT event_id, COUNT(*) as cnt
        FROM fct_order
        WHERE order_date >= :cutoff
        GROUP BY event_id
        HAVING COUNT(*) > 1
        """
        result = db_session.execute(duplicate_query, {"cutoff": cutoff})
        duplicates = result.fetchall()

        assert len(duplicates) == 0, f"Found {len(duplicates)} duplicate event_ids in fct_order (7d window)"

    def test_event_schema_has_unique_constraint(self, db_session):
        """
        ACCEPTANCE: contract: EventSchema has unique constraint on event_id
        """
        from sqlalchemy import inspect

        inspector = inspect(db_session.bind)
        constraints = inspector.get_unique_constraints("events")

        # Find constraint on event_id column
        event_id_constraint = None
        for constraint in constraints:
            if "event_id" in constraint["column_names"]:
                event_id_constraint = constraint
                break

        assert event_id_constraint is not None, "Missing unique constraint on events.event_id"
        assert event_id_constraint["name"] == "events_event_id_unique"

    def test_concurrent_duplicate_submissions_rejected(self, identity_agent, sample_event, db_session):
        """
        Concurrency test: Multiple threads submitting same event_id
        Only one should succeed, others should fail
        """
        import threading
        from queue import Queue

        results = Queue()

        def submit_event():
            try:
                result = identity_agent.ingest_event(sample_event)
                db_session.commit()
                results.put(("success", result))
            except IntegrityError as e:
                db_session.rollback()
                results.put(("error", str(e)))

        # Spawn 5 threads submitting same event_id
        threads = []
        for _ in range(5):
            t = threading.Thread(target=submit_event)
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # Collect results
        successes = []
        errors = []
        while not results.empty():
            status, data = results.get()
            if status == "success":
                successes.append(data)
            else:
                errors.append(data)

        # Only 1 success, 4 errors expected
        assert len(successes) == 1, f"Expected 1 success, got {len(successes)}"
        assert len(errors) == 4, f"Expected 4 errors, got {len(errors)}"
