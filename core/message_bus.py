"""
Message Bus - Event-driven architecture with idempotency enforcement
Implements T001: Event deduplication via Redis SETNX to prevent double-counting
"""
import asyncio
import hashlib
import logging
from datetime import datetime
from typing import Dict, List, Callable, Any, Optional
from dataclasses import dataclass

from prometheus_client import Counter

from core.exceptions import DuplicateEventError
from core.contracts import EventSchema


logger = logging.getLogger(__name__)

# Prometheus metrics
dedup_hits_counter = Counter(
    'message_bus_dedup_hits_total',
    'Total number of duplicate events detected and rejected'
)

events_published_counter = Counter(
    'message_bus_events_published_total',
    'Total number of unique events published',
    ['event_type']
)


@dataclass
class Event:
    """Event with idempotency key for deduplication"""
    event_id: str
    event_type: str
    timestamp: datetime
    data: Dict[str, Any]
    idempotency_key: Optional[str] = None
    
    def __post_init__(self):
        """Generate idempotency_key if not provided"""
        if self.idempotency_key is None:
            self.idempotency_key = self._generate_key()
    
    def _generate_key(self) -> str:
        """Generate idempotency_key = SHA256(event_id + timestamp)"""
        raw = f"{self.event_id}_{self.timestamp.isoformat()}"
        return hashlib.sha256(raw.encode()).hexdigest()


class MessageBus:
    """
    Event bus with Redis-backed idempotency enforcement
    
    T001 Risk Gates:
    - idempotency_key = event_id + timestamp hash
    - Redis SETNX with TTL=24h for dedup keys
    - Fallback to pass-through mode if Redis unavailable
    """
    
    def __init__(self, redis_client):
        """
        Initialize MessageBus with Redis client
        
        Args:
            redis_client: Async Redis client for deduplication
        """
        self.redis = redis_client
        self._handlers: Dict[str, List[Callable]] = {}
        self._dedup_ttl_seconds = 86400  # 24 hours
    
    def subscribe(self, event_type: str, handler: Callable):
        """Register event handler for event_type"""
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(handler)
        logger.info(f"Subscribed handler to {event_type}")
    
    async def publish(self, event: Event):
        """
        Publish event with idempotency enforcement
        
        T001 Implementation:
        1. Generate idempotency_key from event_id + timestamp
        2. Attempt Redis SETNX (set if not exists)
        3. If SETNX returns False → duplicate, raise DuplicateEventError
        4. If SETNX returns True → set TTL=24h, process event
        5. If Redis unavailable → log warning, pass through (rollback strategy)
        
        Args:
            event: Event to publish
        
        Raises:
            DuplicateEventError: If event_id already processed within 24h
        """
        idempotency_key = self._generate_idempotency_key(event)
        redis_key = f"dedup:event:{idempotency_key}"
        
        try:
            # Atomic check-and-set with Redis SETNX
            is_new = await self.redis.setnx(redis_key, "1")
            
            if not is_new:
                # Duplicate detected
                dedup_hits_counter.inc()
                logger.warning(
                    f"Duplicate event detected: {event.event_id} "
                    f"(idempotency_key={idempotency_key})"
                )
                raise DuplicateEventError(
                    f"Event {event.event_id} already processed within 24h"
                )
            
            # Set TTL for dedup key (24 hours)
            await self.redis.expire(redis_key, self._dedup_ttl_seconds)
            
            # Event is new, process it
            await self._dispatch_event(event)
            
            events_published_counter.labels(event_type=event.event_type).inc()
            logger.info(f"Published event: {event.event_id} ({event.event_type})")
            
        except (ConnectionError, TimeoutError) as e:
            # Rollback strategy: Redis unavailable → pass through with warning
            logger.warning(
                f"Redis unavailable for deduplication, passing through event {event.event_id}: {e}"
            )
            await self._dispatch_event(event)
    
    async def _dispatch_event(self, event: Event):
        """Dispatch event to registered handlers"""
        handlers = self._handlers.get(event.event_type, [])
        
        if not handlers:
            logger.debug(f"No handlers registered for {event.event_type}")
            return
        
        # Execute handlers concurrently
        tasks = [handler(event) for handler in handlers]
        await asyncio.gather(*tasks, return_exceptions=True)
    
    def _generate_idempotency_key(self, event: Event) -> str:
        """
        Generate idempotency key for event
        
        Formula: SHA256(event_id + timestamp_iso)
        
        Args:
            event: Event to generate key for
        
        Returns:
            Hex digest of SHA256 hash
        """
        if event.idempotency_key:
            return event.idempotency_key
        
        raw = f"{event.event_id}_{event.timestamp.isoformat()}"
        return hashlib.sha256(raw.encode()).hexdigest()


class EventPublisher:
    """Convenience wrapper for publishing events"""
    
    def __init__(self, message_bus: MessageBus):
        self.bus = message_bus
    
    async def publish_event(
        self,
        event_id: str,
        event_type: str,
        data: Dict[str, Any],
        timestamp: Optional[datetime] = None
    ):
        """
        Publish event to message bus
        
        Args:
            event_id: Unique event identifier
            event_type: Event type (e.g., 'order_completed')
            data: Event payload
            timestamp: Event timestamp (default: now)
        """
        if timestamp is None:
            timestamp = datetime.utcnow()
        
        event = Event(
            event_id=event_id,
            event_type=event_type,
            timestamp=timestamp,
            data=data
        )
        
        await self.bus.publish(event)
