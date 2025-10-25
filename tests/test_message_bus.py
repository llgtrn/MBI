# tests/test_message_bus.py
"""
Test suite for MessageBus event deduplication
Covers: Q_058 (event_id dedup), Q_108 (Redis TTL=7d), A_005 (idempotency)
"""
import pytest
from unittest.mock import Mock, patch
from datetime import datetime
from core.message_bus import MessageBus, Event, PublishResult


class TestEventDeduplication:
    """Q_058, Q_108, A_005: Event deduplication with Redis"""
    
    @pytest.fixture
    def mock_redis(self):
        """Mock Redis client"""
        redis = Mock()
        redis.exists.return_value = False
        redis.get.return_value = None
        redis.setex.return_value = True
        redis.ttl.return_value = 604800  # 7 days
        return redis
    
    @pytest.fixture
    def message_bus(self, mock_redis):
        """MessageBus with mocked Redis"""
        return MessageBus(redis_client=mock_redis)
    
    def test_event_dedup_idempotency(self, message_bus, mock_redis):
        """Q_058: Duplicate events with same event_id processed only once"""
        event = Event(
            event_id="evt_001",
            event_type="order_completed",
            data={"order_id": "o123", "revenue": 19800}
        )
        
        # First call - should process
        result1 = message_bus.publish(event)
        assert result1.processed == True
        assert result1.reason is None
        
        # Simulate Redis having the key now
        mock_redis.exists.return_value = True
        mock_redis.get.return_value = '{"status": "processed", "at": "2025-10-18T20:00:00Z"}'
        
        # Second call (duplicate) - should skip
        result2 = message_bus.publish(event)
        assert result2.processed == False
        assert result2.reason == "duplicate_skipped"
        assert result2.cached == '{"status": "processed", "at": "2025-10-18T20:00:00Z"}'
        
        # Verify Redis was called correctly
        mock_redis.exists.assert_called_with("event:dedup:evt_001")
        mock_redis.get.assert_called_with("event:dedup:evt_001")
    
    def test_event_dedup_ttl_seven_days(self, message_bus, mock_redis):
        """Q_108: Redis TTL set to 7 days (604800 seconds)"""
        event = Event(
            event_id="evt_002",
            event_type="spend_ingested",
            data={"channel": "meta", "spend": 120000}
        )
        
        # Process event
        result = message_bus.publish(event)
        assert result.processed == True
        
        # Verify Redis setex was called with 7-day TTL
        mock_redis.setex.assert_called_once()
        call_args = mock_redis.setex.call_args
        assert call_args[0][0] == "event:dedup:evt_002"  # key
        assert call_args[0][1] == 604800  # TTL in seconds (7 days)
        assert call_args[0][2] is not None  # result data
    
    def test_event_dedup_different_events_processed(self, message_bus, mock_redis):
        """Different event_ids should both be processed"""
        event1 = Event(event_id="evt_003", event_type="test", data={})
        event2 = Event(event_id="evt_004", event_type="test", data={})
        
        result1 = message_bus.publish(event1)
        result2 = message_bus.publish(event2)
        
        assert result1.processed == True
        assert result2.processed == True
    
    def test_event_dedup_redis_failure_fail_open(self, message_bus, mock_redis):
        """A_005: Redis failure should fail open (process anyway) for availability"""
        # Simulate Redis connection failure
        mock_redis.exists.side_effect = Exception("Redis connection failed")
        
        event = Event(event_id="evt_005", event_type="critical", data={})
        
        # Should still process event (fail open)
        result = message_bus.publish(event)
        assert result.processed == True
        assert result.reason == "redis_unavailable_failopen"
    
    def test_event_dedup_key_format(self, message_bus, mock_redis):
        """Verify Redis key format: event:dedup:{event_id}"""
        event = Event(event_id="abc-123-xyz", event_type="test", data={})
        
        message_bus.publish(event)
        
        # Check that exists was called with correct key format
        mock_redis.exists.assert_called_with("event:dedup:abc-123-xyz")
        
        # Check that setex was called with correct key format
        call_args = mock_redis.setex.call_args
        assert call_args[0][0] == "event:dedup:abc-123-xyz"
    
    def test_event_dedup_logs_duplicate_skip(self, message_bus, mock_redis, caplog):
        """Duplicate events should be logged"""
        mock_redis.exists.return_value = True
        mock_redis.get.return_value = '{"status": "processed"}'
        
        event = Event(event_id="evt_006", event_type="test", data={})
        
        with caplog.at_level("INFO"):
            result = message_bus.publish(event)
        
        assert result.processed == False
        assert "Duplicate event skipped: evt_006" in caplog.text


class TestEventDeduplicationIntegration:
    """Integration tests with real Redis (requires Redis running)"""
    
    @pytest.mark.integration
    @pytest.mark.skipif(not pytest.config.getoption("--integration"), reason="Integration tests disabled")
    def test_event_dedup_real_redis(self):
        """Integration test with real Redis instance"""
        import redis
        
        redis_client = redis.Redis(host='localhost', port=6379, db=15, decode_responses=True)
        redis_client.flushdb()  # Clean test database
        
        bus = MessageBus(redis_client=redis_client)
        event = Event(event_id="evt_real_001", event_type="test", data={})
        
        # First publish
        result1 = bus.publish(event)
        assert result1.processed == True
        
        # Verify key exists in Redis
        assert redis_client.exists("event:dedup:evt_real_001") == 1
        
        # Verify TTL
        ttl = redis_client.ttl("event:dedup:evt_real_001")
        assert 604700 < ttl <= 604800  # Allow small timing variance
        
        # Second publish (duplicate)
        result2 = bus.publish(event)
        assert result2.processed == False
        assert result2.reason == "duplicate_skipped"
        
        # Cleanup
        redis_client.flushdb()
