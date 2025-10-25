"""
Event schema with deduplication contract.

Contract:
- event_id: unique constraint prevents duplicate revenue tracking
- idempotency_key: event_id ensures retry-safe ingestion
"""

from pydantic import BaseModel, Field, validator
from datetime import datetime
from typing import List, Dict, Optional
from sqlalchemy import Column, String, DateTime, Float, JSON, Integer, UniqueConstraint
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class EventSchema(BaseModel):
    """
    Event schema for order/conversion tracking.
    
    Contract: event_id unique constraint prevents duplicate revenue
    Reference: Q_001 (C01 Red/CRITICAL)
    """
    event_id: str = Field(
        ..., 
        description="Unique event identifier (idempotency key)",
        min_length=1,
        max_length=255,
        regex=r"^evt_[a-zA-Z0-9_-]+$"
    )
    event_type: str = Field(
        ...,
        description="Type of event",
        regex=r"^(order_completed|order_cancelled|refund|subscription_start)$"
    )
    user_key: str = Field(
        ...,
        description="Hashed user identifier (privacy-safe)",
        min_length=1,
        max_length=255
    )
    timestamp: datetime = Field(
        ...,
        description="Event timestamp (UTC)"
    )
    revenue: float = Field(
        default=0.0,
        description="Revenue amount (positive for orders, negative for refunds)",
        ge=-1000000,
        le=1000000
    )
    currency: str = Field(
        default="JPY",
        description="Currency code (ISO 4217)",
        min_length=3,
        max_length=3
    )
    order_id: Optional[str] = Field(
        None,
        description="Order identifier (if applicable)",
        max_length=255
    )
    items: List[Dict] = Field(
        default_factory=list,
        description="Order items [{sku, qty, price}]"
    )
    utm_source: Optional[str] = Field(None, max_length=255)
    utm_medium: Optional[str] = Field(None, max_length=255)
    utm_campaign: Optional[str] = Field(None, max_length=255)

    class Config:
        schema_extra = {
            "example": {
                "event_id": "evt_20251019_o123",
                "event_type": "order_completed",
                "user_key": "uhash_abc123",
                "timestamp": "2025-10-19T08:00:00Z",
                "revenue": 19800,
                "currency": "JPY",
                "order_id": "o123",
                "items": [{"sku": "SKU-1", "qty": 1, "price": 19800}],
                "utm_source": "google",
                "utm_medium": "cpc"
            }
        }

    @validator("event_id")
    def validate_event_id_format(cls, v):
        """Ensure event_id follows evt_* format"""
        if not v.startswith("evt_"):
            raise ValueError("event_id must start with 'evt_'")
        return v

    @validator("timestamp")
    def validate_timestamp_not_future(cls, v):
        """Prevent future timestamps"""
        if v > datetime.utcnow():
            raise ValueError("timestamp cannot be in the future")
        return v


class EventModel(Base):
    """
    SQLAlchemy model for events table.
    
    DB Contract: unique constraint on event_id column
    """
    __tablename__ = "events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    event_id = Column(String(255), nullable=False, index=True)
    event_type = Column(String(50), nullable=False)
    user_key = Column(String(255), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    revenue = Column(Float, default=0.0)
    currency = Column(String(3), default="JPY")
    order_id = Column(String(255), nullable=True, index=True)
    items = Column(JSON, default=list)
    utm_source = Column(String(255), nullable=True)
    utm_medium = Column(String(255), nullable=True)
    utm_campaign = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    __table_args__ = (
        UniqueConstraint("event_id", name="events_event_id_unique"),
        {"comment": "Events table with event_id deduplication (Q_001)"}
    )

    def __repr__(self):
        return f"<Event(event_id={self.event_id}, type={self.event_type}, revenue={self.revenue})>"


# Migration SQL (for reference)
MIGRATION_SQL = """
-- Add unique constraint to events.event_id
-- Reference: Q_001 (C01 Red/CRITICAL - Event deduplication)
-- Rollback: DROP CONSTRAINT events_event_id_unique;

ALTER TABLE events
ADD CONSTRAINT events_event_id_unique UNIQUE (event_id);

-- Create index for performance
CREATE INDEX IF NOT EXISTS idx_events_event_id ON events(event_id);
CREATE INDEX IF NOT EXISTS idx_events_user_key ON events(user_key);
CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp);
CREATE INDEX IF NOT EXISTS idx_events_order_id ON events(order_id) WHERE order_id IS NOT NULL;
"""
