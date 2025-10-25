"""
Identity Resolution Agent with event deduplication.

Components:
- IdentityResolutionAgent: Main agent for identity resolution
- ingest_event: Event ingestion with deduplication enforcement

Reference: Q_001 (C01 Red/CRITICAL)
"""

import hashlib
from datetime import datetime
from typing import Dict, Optional
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from schemas.event import EventSchema, EventModel
from core.metrics import MetricsClient


class IdentityResolutionAgent:
    """
    Identity Resolution Agent with event deduplication.
    
    Features:
    - Event ingestion with event_id deduplication
    - Privacy-safe user_key hashing
    - Idempotent retry handling
    
    Kill Switch: DEDUP_ENFORCEMENT_ENABLED (default: true)
    Metric: duplicate_events_rejected
    """

    def __init__(
        self,
        session: Session,
        metrics_client: Optional[MetricsClient] = None,
        dedup_enabled: bool = True
    ):
        self.session = session
        self.metrics = metrics_client or MetricsClient()
        self.dedup_enabled = dedup_enabled

    def ingest_event(self, event_data: Dict) -> Dict:
        """
        Ingest event with deduplication enforcement.
        
        Args:
            event_data: Event payload (conforms to EventSchema)
        
        Returns:
            {"ok": true, "event_id": "...", "status": "inserted|duplicate"}
        
        Raises:
            IntegrityError: If event_id already exists (dedup enforcement)
            ValidationError: If event_data fails schema validation
        
        Contract:
        - event_id uniqueness enforced at DB level
        - Duplicate submissions return 409 Conflict
        - Metric duplicate_events_rejected incremented on retry
        """
        # Validate schema
        event_schema = EventSchema(**event_data)

        # Check kill switch
        if not self.dedup_enabled:
            # Bypass deduplication (for emergency rollback only)
            return self._insert_event_unchecked(event_schema)

        # Standard path: enforce deduplication
        try:
            event_model = EventModel(
                event_id=event_schema.event_id,
                event_type=event_schema.event_type,
                user_key=event_schema.user_key,
                timestamp=event_schema.timestamp,
                revenue=event_schema.revenue,
                currency=event_schema.currency,
                order_id=event_schema.order_id,
                items=event_schema.items,
                utm_source=event_schema.utm_source,
                utm_medium=event_schema.utm_medium,
                utm_campaign=event_schema.utm_campaign
            )

            self.session.add(event_model)
            self.session.flush()  # Force constraint check before commit

            return {
                "ok": True,
                "event_id": event_schema.event_id,
                "status": "inserted"
            }

        except IntegrityError as e:
            self.session.rollback()

            # Increment deduplication metric
            self.metrics.increment("duplicate_events_rejected", tags={
                "event_type": event_schema.event_type,
                "event_id": event_schema.event_id[:50]  # Truncate for cardinality
            })

            # Check if it's the event_id constraint
            if "events_event_id_unique" in str(e):
                # Re-raise for API layer to return 409
                raise IntegrityError(
                    statement="Duplicate event_id",
                    params={"event_id": event_schema.event_id},
                    orig=e.orig
                )
            else:
                # Other integrity error (unexpected)
                raise

    def _insert_event_unchecked(self, event_schema: EventSchema) -> Dict:
        """
        Insert event without deduplication check (kill switch bypass).
        
        WARNING: Only use for emergency rollback
        """
        event_model = EventModel(
            event_id=event_schema.event_id,
            event_type=event_schema.event_type,
            user_key=event_schema.user_key,
            timestamp=event_schema.timestamp,
            revenue=event_schema.revenue,
            currency=event_schema.currency,
            order_id=event_schema.order_id,
            items=event_schema.items,
            utm_source=event_schema.utm_source,
            utm_medium=event_schema.utm_medium,
            utm_campaign=event_schema.utm_campaign
        )

        self.session.add(event_model)
        self.session.flush()

        return {
            "ok": True,
            "event_id": event_schema.event_id,
            "status": "inserted_unchecked"
        }

    def get_dedup_stats(self, hours: int = 24) -> Dict:
        """
        Get deduplication statistics for monitoring.
        
        Returns:
            {
                "duplicate_rejections": count,
                "total_events": count,
                "dedup_rate": percentage
            }
        """
        duplicate_count = self.metrics.get_counter(
            "duplicate_events_rejected",
            lookback_hours=hours
        )

        total_events = self.session.query(EventModel).filter(
            EventModel.created_at >= datetime.utcnow() - timedelta(hours=hours)
        ).count()

        return {
            "duplicate_rejections": duplicate_count,
            "total_events": total_events,
            "dedup_rate": (duplicate_count / total_events * 100) if total_events > 0 else 0
        }


# Metrics
METRICS = {
    "duplicate_events_rejected": {
        "type": "counter",
        "description": "Count of duplicate event_id rejections",
        "tags": ["event_type", "event_id"]
    }
}
