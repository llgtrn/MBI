"""Tests for core.message_bus - Event deduplication and idempotency"""
import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta
import hashlib

from core.message_bus import MessageBus, Event, EventPublisher
from core.exceptions import DuplicateEventError


class TestEventDeduplication:
    """Test event_id deduplication with Redis race condition handling"""
    
    @pytest.fixture
    def mock_redis(self):
        """Mock Redis client for testing"""
        redis = AsyncMock()
        redis.setnx = AsyncMock()
        redis.expire = AsyncMock()
        redis.get = AsyncMock()
        return redis
    
    @pytest.fixture
    def message_bus(self, mock_redis):
        """MessageBus instance with mocked Redis"""
        bus = MessageBus(redis_client=mock_redis)
        return bus
    
    @pytest.mark.asyncio
    async def test_duplicate_event_id_single_process(self, message_bus, mock_redis):
        """
        T001 Acceptance: 1000 duplicate events should result in exactly 1 processed event
        Tests idempotency_key enforcement with Redis SETNX
        """
        # Setup: Mock Redis to accept first SETNX, reject subsequent
        call_count = 0
        
        async def mock_setnx(key, value):
            nonlocal call_count
            call_count += 1
            return call_count == 1  # Only first call succeeds
        
        mock_redis.setnx = mock_setnx
        mock_redis.expire = AsyncMock()
        
        # Create event with fixed event_id
        event = Event(
            event_id="test_event_001",
            event_type="order_completed",
            timestamp=datetime.utcnow(),
            data={"order_id": "o123", "revenue": 100}
        )
        
        # Track processing
        processed_count = 0
        
        async def mock_handler(evt):
            nonlocal processed_count
            processed_count += 1
        
        message_bus.subscribe("order_completed", mock_handler)
        
        # Publish same event 1000 times
        for i in range(1000):
            try:
                await message_bus.publish(event)
            except DuplicateEventError:
                pass  # Expected for duplicates
        
        # Wait for async processing
        await asyncio.sleep(0.1)
        
        # Assertion: Only 1 event should be processed
        assert processed_count == 1, f"Expected 1 processed event, got {processed_count}"
        assert call_count == 1000, f"Expected 1000 SETNX calls, got {call_count}"
    
    @pytest.mark.asyncio
    async def test_idempotency_key_generation(self, message_bus):
        """Test idempotency_key = hash(event_id + timestamp)"""
        event = Event(
            event_id="evt_001",
            event_type="spend_ingested",
            timestamp=datetime(2025, 10, 18, 12, 0, 0),
            data={}
        )
        
        expected_key = hashlib.sha256(
            f"evt_001_{event.timestamp.isoformat()}".encode()
        ).hexdigest()
        
        actual_key = message_bus._generate_idempotency_key(event)
        
        assert actual_key == expected_key
    
    @pytest.mark.asyncio
    async def test_redis_ttl_24h(self, message_bus, mock_redis):
        """Test Redis dedup key has 24h TTL"""
        mock_redis.setnx.return_value = True
        
        event = Event(
            event_id="evt_002",
            event_type="test",
            timestamp=datetime.utcnow(),
            data={}
        )
        
        await message_bus.publish(event)
        
        # Verify expire called with 24h = 86400 seconds
        mock_redis.expire.assert_called_once()
        call_args = mock_redis.expire.call_args
        assert call_args[0][1] == 86400  # 24 hours in seconds
    
    @pytest.mark.asyncio
    async def test_duplicate_event_raises_error(self, message_bus, mock_redis):
        """Test duplicate event_id raises DuplicateEventError"""
        # First event succeeds
        mock_redis.setnx.return_value = True
        
        event = Event(
            event_id="evt_003",
            event_type="test",
            timestamp=datetime.utcnow(),
            data={}
        )
        
        await message_bus.publish(event)
        
        # Second event fails (SETNX returns False)
        mock_redis.setnx.return_value = False
        
        with pytest.raises(DuplicateEventError) as exc_info:
            await message_bus.publish(event)
        
        assert "evt_003" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_dedup_metric_increment(self, message_bus, mock_redis):
        """Test message_bus_dedup_hits_total counter increments on duplicate"""
        mock_redis.setnx.return_value = False  # Simulate duplicate
        
        event = Event(
            event_id="evt_004",
            event_type="test",
            timestamp=datetime.utcnow(),
            data={}
        )
        
        # Mock Prometheus counter
        with patch('core.message_bus.dedup_hits_counter') as mock_counter:
            try:
                await message_bus.publish(event)
            except DuplicateEventError:
                pass
            
            mock_counter.inc.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_redis_unavailable_fallback(self, message_bus, mock_redis):
        """Test fallback to pass-through mode if Redis unavailable"""
        # Simulate Redis connection error
        mock_redis.setnx.side_effect = ConnectionError("Redis unavailable")
        
        event = Event(
            event_id="evt_005",
            event_type="test",
            timestamp=datetime.utcnow(),
            data={}
        )
        
        processed = []
        
        async def handler(evt):
            processed.append(evt)
        
        message_bus.subscribe("test", handler)
        
        # Should not raise, should log warning and pass through
        with patch('core.message_bus.logger') as mock_logger:
            await message_bus.publish(event)
            
            mock_logger.warning.assert_called_once()
            assert "Redis unavailable" in str(mock_logger.warning.call_args)
        
        await asyncio.sleep(0.1)
        assert len(processed) == 1  # Event processed despite Redis failure


class TestEventSchema:
    """Test Event contract includes required idempotency_key field"""
    
    def test_event_schema_has_idempotency_key(self):
        """Contract: Event schema must include idempotency_key (str, required)"""
        from core.contracts import EventSchema
        
        schema_fields = EventSchema.model_json_schema()['properties']
        
        assert 'idempotency_key' in schema_fields
        assert schema_fields['idempotency_key']['type'] == 'string'
        assert 'idempotency_key' in EventSchema.model_json_schema()['required']
    
    def test_event_validates_idempotency_key(self):
        """Test Event validation rejects missing idempotency_key"""
        from core.contracts import EventSchema
        from pydantic import ValidationError
        
        with pytest.raises(ValidationError) as exc_info:
            EventSchema(
                event_id="evt_001",
                event_type="test",
                timestamp=datetime.utcnow(),
                data={}
                # Missing idempotency_key
            )
        
        assert 'idempotency_key' in str(exc_info.value)
