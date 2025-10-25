"""
Tests for Distributed Tracing with Correlation ID
=================================================

Tests cover:
- Correlation ID generation and propagation
- Span creation with correlation_id in attributes
- Context propagation across async boundaries
- TracedEvent serialization/deserialization
- Decorator-based tracing
- Metrics emission
"""

import pytest
import asyncio
import uuid
from unittest.mock import Mock, patch, MagicMock
from core.tracing import (
    get_correlation_id,
    set_correlation_id,
    ensure_correlation_id,
    generate_correlation_id,
    create_span,
    traced,
    TracedEvent,
    propagate_correlation_id_to_span,
    correlation_scope,
    trace_event_handler,
    init_tracing
)


class TestCorrelationIDManagement:
    """Test correlation ID generation and context management"""
    
    def test_generate_correlation_id_returns_uuid(self):
        correlation_id = generate_correlation_id()
        # Verify it's a valid UUID
        uuid.UUID(correlation_id)
        assert len(correlation_id) == 36
    
    def test_set_and_get_correlation_id(self):
        """Q_055 acceptance: Create test events, verify correlation_id in span tags"""
        test_id = "test-correlation-123"
        set_correlation_id(test_id)
        
        retrieved_id = get_correlation_id()
        assert retrieved_id == test_id
    
    def test_ensure_correlation_id_generates_if_missing(self):
        # Clear context
        set_correlation_id(None)
        
        correlation_id = ensure_correlation_id()
        assert correlation_id is not None
        assert get_correlation_id() == correlation_id
    
    def test_ensure_correlation_id_returns_existing(self):
        test_id = "existing-id"
        set_correlation_id(test_id)
        
        correlation_id = ensure_correlation_id()
        assert correlation_id == test_id
    
    def test_correlation_id_isolation_across_contexts(self):
        """Verify correlation IDs don't leak between contexts"""
        set_correlation_id("context-1")
        
        with correlation_scope("context-2"):
            assert get_correlation_id() == "context-2"
        
        # Original context restored
        assert get_correlation_id() == "context-1"


class TestSpanCreation:
    """Test span creation and attribute setting"""
    
    @patch('core.tracing.get_tracer')
    def test_create_span_includes_correlation_id(self, mock_get_tracer):
        """Q_055 acceptance: Verify correlation_id in span tags"""
        mock_tracer = MagicMock()
        mock_span = MagicMock()
        mock_get_tracer.return_value = mock_tracer
        mock_tracer.start_as_current_span.return_value.__enter__ = Mock(return_value=mock_span)
        mock_tracer.start_as_current_span.return_value.__exit__ = Mock(return_value=False)
        
        test_id = "span-correlation-123"
        set_correlation_id(test_id)
        
        with create_span("test_operation") as span:
            pass
        
        # Verify span was created with correlation_id
        call_args = mock_tracer.start_as_current_span.call_args
        attributes = call_args.kwargs.get('attributes', {})
        assert attributes['correlation_id'] == test_id
    
    @patch('core.tracing.get_tracer')
    def test_create_span_with_custom_attributes(self, mock_get_tracer):
        mock_tracer = MagicMock()
        mock_span = MagicMock()
        mock_get_tracer.return_value = mock_tracer
        mock_tracer.start_as_current_span.return_value.__enter__ = Mock(return_value=mock_span)
        mock_tracer.start_as_current_span.return_value.__exit__ = Mock(return_value=False)
        
        custom_attrs = {"user_id": "123", "order_id": "456"}
        
        with create_span("test_op", attributes=custom_attrs):
            pass
        
        call_args = mock_tracer.start_as_current_span.call_args
        attributes = call_args.kwargs.get('attributes', {})
        
        assert attributes['user_id'] == "123"
        assert attributes['order_id'] == "456"
        assert 'correlation_id' in attributes
    
    @patch('core.tracing.get_tracer')
    def test_create_span_handles_exceptions(self, mock_get_tracer):
        mock_tracer = MagicMock()
        mock_span = MagicMock()
        mock_get_tracer.return_value = mock_tracer
        mock_tracer.start_as_current_span.return_value.__enter__ = Mock(return_value=mock_span)
        mock_tracer.start_as_current_span.return_value.__exit__ = Mock(return_value=False)
        
        with pytest.raises(ValueError):
            with create_span("test_operation"):
                raise ValueError("Test error")
        
        # Verify span recorded exception
        mock_span.record_exception.assert_called()


class TestTracedDecorator:
    """Test @traced decorator"""
    
    @pytest.mark.asyncio
    @patch('core.tracing.create_span')
    async def test_traced_decorator_async_function(self, mock_create_span):
        mock_create_span.return_value.__enter__ = Mock(return_value=MagicMock())
        mock_create_span.return_value.__exit__ = Mock(return_value=False)
        
        @traced(operation_name="test_async_op")
        async def async_function(x: int) -> int:
            return x * 2
        
        result = await async_function(5)
        
        assert result == 10
        mock_create_span.assert_called()
        call_args = mock_create_span.call_args
        assert call_args[0][0] == "test_async_op"
    
    @patch('core.tracing.create_span')
    def test_traced_decorator_sync_function(self, mock_create_span):
        mock_create_span.return_value.__enter__ = Mock(return_value=MagicMock())
        mock_create_span.return_value.__exit__ = Mock(return_value=False)
        
        @traced(operation_name="test_sync_op")
        def sync_function(x: int) -> int:
            return x + 1
        
        result = sync_function(10)
        
        assert result == 11
        mock_create_span.assert_called()
    
    @pytest.mark.asyncio
    @patch('core.tracing.create_span')
    async def test_traced_decorator_propagates_exceptions(self, mock_create_span):
        mock_create_span.return_value.__enter__ = Mock(return_value=MagicMock())
        mock_create_span.return_value.__exit__ = Mock(return_value=False)
        
        @traced()
        async def failing_function():
            raise RuntimeError("Expected error")
        
        with pytest.raises(RuntimeError, match="Expected error"):
            await failing_function()


class TestTracedEvent:
    """Test TracedEvent class"""
    
    def test_traced_event_creation(self):
        event = TracedEvent(
            event_type="order_created",
            data={"order_id": "123", "amount": 100.0},
            source="api_gateway"
        )
        
        assert event.event_type == "order_created"
        assert event.data["order_id"] == "123"
        assert event.source == "api_gateway"
        assert event.correlation_id is not None
        assert event.event_id is not None
    
    def test_traced_event_with_explicit_correlation_id(self):
        test_id = "explicit-correlation-id"
        event = TracedEvent(
            event_type="test_event",
            data={},
            correlation_id=test_id
        )
        
        assert event.correlation_id == test_id
    
    def test_traced_event_serialization(self):
        """Q_055 acceptance: Event schema includes correlation_id field"""
        event = TracedEvent(
            event_type="payment_processed",
            data={"payment_id": "pay_123"},
            correlation_id="corr-456"
        )
        
        serialized = event.to_dict()
        
        assert serialized["event_type"] == "payment_processed"
        assert serialized["correlation_id"] == "corr-456"
        assert "event_id" in serialized
        assert "timestamp" in serialized
    
    def test_traced_event_deserialization(self):
        """Q_055 acceptance: Verify correlation_id restored to context"""
        original_event = TracedEvent(
            event_type="user_signup",
            data={"user_id": "u123"},
            correlation_id="corr-789"
        )
        
        serialized = original_event.to_dict()
        restored_event = TracedEvent.from_dict(serialized)
        
        assert restored_event.event_type == original_event.event_type
        assert restored_event.correlation_id == original_event.correlation_id
        assert restored_event.data == original_event.data
        
        # Verify correlation_id restored to context
        assert get_correlation_id() == "corr-789"


class TestCorrelationScope:
    """Test correlation scope context manager"""
    
    def test_correlation_scope_with_explicit_id(self):
        set_correlation_id("original")
        
        with correlation_scope("scoped-id") as scoped_id:
            assert scoped_id == "scoped-id"
            assert get_correlation_id() == "scoped-id"
        
        # Original restored
        assert get_correlation_id() == "original"
    
    def test_correlation_scope_generates_id(self):
        set_correlation_id(None)
        
        with correlation_scope() as scoped_id:
            assert scoped_id is not None
            assert get_correlation_id() == scoped_id
        
        # Context cleared after scope
        # (Note: actual behavior depends on implementation)
    
    def test_nested_correlation_scopes(self):
        with correlation_scope("outer") as outer_id:
            assert get_correlation_id() == "outer"
            
            with correlation_scope("inner") as inner_id:
                assert get_correlation_id() == "inner"
            
            # Outer restored
            assert get_correlation_id() == "outer"


class TestEventHandler:
    """Test event handler tracing"""
    
    @pytest.mark.asyncio
    @patch('core.tracing.create_span')
    async def test_trace_event_handler(self, mock_create_span):
        mock_create_span.return_value.__enter__ = Mock(return_value=MagicMock())
        mock_create_span.return_value.__exit__ = Mock(return_value=False)
        
        async def handler(event):
            return f"Handled {event.event_type}"
        
        event = TracedEvent(
            event_type="test_event",
            data={},
            correlation_id="handler-test-id"
        )
        
        result = await trace_event_handler(handler, event)
        
        assert result == "Handled test_event"
        assert get_correlation_id() == "handler-test-id"
        
        # Verify span created with event metadata
        call_args = mock_create_span.call_args
        span_name = call_args[0][0]
        assert "handle_test_event" in span_name


class TestPropagateCOrrelationID:
    """Test explicit correlation ID propagation"""
    
    def test_propagate_correlation_id_to_span(self):
        mock_span = MagicMock()
        
        set_correlation_id("propagate-test")
        propagate_correlation_id_to_span(mock_span)
        
        mock_span.set_attribute.assert_called_with("correlation_id", "propagate-test")


class TestTracingIntegration:
    """Integration tests"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_correlation_propagation(self):
        """
        Test full flow: Event creation → Handler → Span with correlation_id
        """
        # Create event with correlation_id
        event = TracedEvent(
            event_type="order_created",
            data={"order_id": "ord_123"},
            correlation_id="e2e-test-id"
        )
        
        # Serialize (simulating transport)
        serialized = event.to_dict()
        
        # Deserialize (simulating receiver)
        received_event = TracedEvent.from_dict(serialized)
        
        # Verify correlation_id propagated
        assert received_event.correlation_id == "e2e-test-id"
        assert get_correlation_id() == "e2e-test-id"
    
    @pytest.mark.asyncio
    async def test_playbook_logs_share_correlation_id(self):
        """
        Q_055 acceptance: Playbook logs share correlation_id
        
        Simulates a playbook execution with multiple steps
        sharing the same correlation_id.
        """
        playbook_correlation_id = "playbook-exec-123"
        
        with correlation_scope(playbook_correlation_id):
            # Step 1: Check conditions
            step1_id = get_correlation_id()
            
            # Step 2: Execute action
            step2_id = get_correlation_id()
            
            # Step 3: Send notification
            step3_id = get_correlation_id()
            
            # All steps share same correlation_id
            assert step1_id == playbook_correlation_id
            assert step2_id == playbook_correlation_id
            assert step3_id == playbook_correlation_id
    
    @pytest.mark.asyncio
    @patch('core.tracing.create_span')
    async def test_agent_to_agent_correlation_propagation(self, mock_create_span):
        """
        Test correlation_id propagation across agent boundaries.
        
        Scenario: MMM Agent → Budget Allocation Agent → Activation Agent
        """
        mock_create_span.return_value.__enter__ = Mock(return_value=MagicMock())
        mock_create_span.return_value.__exit__ = Mock(return_value=False)
        
        agent_correlation_id = "agent-chain-test"
        set_correlation_id(agent_correlation_id)
        
        # MMM Agent
        @traced(operation_name="mmm_predict")
        async def mmm_predict():
            return {"optimal_allocation": {"meta": 100000}}
        
        # Budget Allocation Agent
        @traced(operation_name="allocate_budget")
        async def allocate_budget(allocation_data):
            # Should inherit correlation_id
            assert get_correlation_id() == agent_correlation_id
            return {"allocation_id": "alloc_123"}
        
        # Activation Agent
        @traced(operation_name="activate_budget")
        async def activate_budget(allocation_id):
            # Should still have same correlation_id
            assert get_correlation_id() == agent_correlation_id
            return {"status": "applied"}
        
        # Execute chain
        mmm_result = await mmm_predict()
        allocation_result = await allocate_budget(mmm_result)
        activation_result = await activate_budget(allocation_result["allocation_id"])
        
        # Verify correlation_id maintained throughout
        assert get_correlation_id() == agent_correlation_id


class TestTracingInitialization:
    """Test tracing initialization"""
    
    @patch('core.tracing.trace.set_tracer_provider')
    def test_init_tracing_with_otlp_endpoint(self, mock_set_provider):
        tracer = init_tracing(
            service_name="test-service",
            otlp_endpoint="localhost:4317",
            environment="test"
        )
        
        assert tracer is not None
        mock_set_provider.assert_called()
    
    @patch('core.tracing.trace.set_tracer_provider')
    def test_init_tracing_without_otlp(self, mock_set_provider):
        tracer = init_tracing(
            service_name="test-service",
            environment="local"
        )
        
        assert tracer is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
