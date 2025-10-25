"""Event schema for the MBI event-driven system.

Defines the Event contract for all events flowing through the system.
Every event includes a correlation_id for distributed tracing.
"""

from datetime import datetime
from typing import Dict, Any, Optional
from uuid import uuid4
from pydantic import BaseModel, Field


class Event(BaseModel):
    """Base event schema for MBI event system.
    
    All events include correlation_id for E2E tracing across agents.
    
    Attributes:
        event_id: Unique identifier for this event
        event_type: Type of event (e.g., 'order_completed', 'spend_ingested')
        correlation_id: Correlation ID for distributed tracing
        timestamp: Event timestamp (UTC)
        data: Event payload
        source: Event source identifier
        metadata: Additional metadata
    """
    event_id: str = Field(default_factory=lambda: str(uuid4()))
    event_type: str
    correlation_id: str = Field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    data: Dict[str, Any]
    source: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
