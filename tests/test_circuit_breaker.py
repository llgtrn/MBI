"""
Tests for Circuit Breaker Implementation
========================================

Tests cover:
- State machine transitions (CLOSED → OPEN → HALF_OPEN → CLOSED)
- Threshold enforcement
- Timeout behavior
- Success/failure counting
- Metrics emission
- Kill switch behavior
- Edge cases
"""

import pytest
import time
from unittest.mock import Mock, patch
from core.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerError,
    CircuitState,
    get_circuit_breaker,
    reset_all_circuits
)


class TestCircuitBreakerConfig:
    """Test configuration validation"""
    
    def test_valid_config(self):
        config = CircuitBreakerConfig(
            failure_threshold=5,
            timeout_seconds=60,
            success_threshold=2
        )
        assert config.failure_threshold == 5
        assert config.timeout_seconds == 60
        assert config.success_threshold == 2
        assert config.enabled is True
    
    def test_invalid_failure_threshold(self):
        with pytest.raises(ValueError, match="failure_threshold must be > 0"):
            CircuitBreakerConfig(failure_threshold=0)
    
    def test_invalid_timeout(self):
        with pytest.raises(ValueError, match="timeout_seconds must be > 0"):
            CircuitBreakerConfig(timeout_seconds=-1)
    
    def test_invalid_success_threshold(self):
        with pytest.raises(ValueError, match="success_threshold must be > 0"):
            CircuitBreakerConfig(success_threshold=0)


class TestCircuitBreakerStateTransitions:
    """Test state machine transitions"""
    
    def test_initial_state_is_closed(self):
        circuit = CircuitBreaker(
            "test_service",
            CircuitBreakerConfig(failure_threshold=3)
        )
        assert circuit.state == CircuitState.CLOSED
    
    def test_circuit_opens_on_threshold(self):
        """Q_054 acceptance: Mock ad API failures without actual API calls; verify circuit state transitions"""
        circuit = CircuitBreaker(
            "test_ad_api",
            CircuitBreakerConfig(failure_threshold=3, timeout_seconds=60)
        )
        
        # Mock failing function
        def failing_call():
            raise Exception("API failure")
        
        # Execute failures up to threshold
        for i in range(3):
            with pytest.raises(Exception):
                circuit.call(failing_call)
            
            # Check state after each failure
            if i < 2:
                assert circuit.state == CircuitState.CLOSED
            else:
                assert circuit.state == CircuitState.OPEN
        
        # Verify circuit is now open
        assert circuit.state == CircuitState.OPEN
        state = circuit.get_state_snapshot()
        assert state.failure_count == 3
        assert state.opened_at is not None
    
    def test_circuit_rejects_calls_when_open(self):
        """Q_054 acceptance: Verify circuit opens and rejects calls"""
        circuit = CircuitBreaker(
            "test_service",
            CircuitBreakerConfig(failure_threshold=2)
        )
        
        # Trigger failures to open circuit
        def failing_call():
            raise Exception("Failure")
        
        for _ in range(2):
            with pytest.raises(Exception):
                circuit.call(failing_call)
        
        assert circuit.state == CircuitState.OPEN
        
        # Now any call should be rejected immediately
        with pytest.raises(CircuitBreakerError, match="Circuit breaker is OPEN"):
            circuit.call(lambda: "should not execute")
    
    def test_circuit_transitions_to_half_open_after_timeout(self):
        """Q_054 acceptance: Circuit closes after timeout"""
        circuit = CircuitBreaker(
            "test_service",
            CircuitBreakerConfig(
                failure_threshold=2,
                timeout_seconds=1,  # Short timeout for testing
                success_threshold=1
            )
        )
        
        # Open the circuit
        def failing_call():
            raise Exception("Failure")
        
        for _ in range(2):
            with pytest.raises(Exception):
                circuit.call(failing_call)
        
        assert circuit.state == CircuitState.OPEN
        
        # Wait for timeout
        time.sleep(1.1)
        
        # Next call attempt should transition to HALF_OPEN
        # Use a successful call
        result = circuit.call(lambda: "success")
        
        # Should be CLOSED now (1 success with threshold=1)
        assert circuit.state == CircuitState.CLOSED
        assert result == "success"
    
    def test_half_open_closes_on_success_threshold(self):
        circuit = CircuitBreaker(
            "test_service",
            CircuitBreakerConfig(
                failure_threshold=2,
                timeout_seconds=1,
                success_threshold=2
            )
        )
        
        # Open circuit
        for _ in range(2):
            with pytest.raises(Exception):
                circuit.call(lambda: (_ for _ in ()).throw(Exception("fail")))
        
        assert circuit.state == CircuitState.OPEN
        
        # Wait for timeout
        time.sleep(1.1)
        
        # First success → should be HALF_OPEN
        circuit.call(lambda: "success1")
        # State transitions happen in _check_state, called during call
        # After 1 success with threshold=2, still in HALF_OPEN
        
        # Second success → should close
        circuit.call(lambda: "success2")
        assert circuit.state == CircuitState.CLOSED
    
    def test_half_open_reopens_on_failure(self):
        circuit = CircuitBreaker(
            "test_service",
            CircuitBreakerConfig(
                failure_threshold=2,
                timeout_seconds=1,
                success_threshold=2
            )
        )
        
        # Open circuit
        for _ in range(2):
            with pytest.raises(Exception):
                circuit.call(lambda: (_ for _ in ()).throw(Exception("fail")))
        
        time.sleep(1.1)
        
        # First call transitions to HALF_OPEN
        # But if it fails, should go back to OPEN
        with pytest.raises(Exception):
            circuit.call(lambda: (_ for _ in ()).throw(Exception("fail again")))
        
        assert circuit.state == CircuitState.OPEN


class TestCircuitBreakerBehavior:
    """Test circuit breaker behavior"""
    
    def test_successful_calls_reset_failure_count(self):
        circuit = CircuitBreaker(
            "test_service",
            CircuitBreakerConfig(failure_threshold=3)
        )
        
        # One failure
        with pytest.raises(Exception):
            circuit.call(lambda: (_ for _ in ()).throw(Exception("fail")))
        
        state = circuit.get_state_snapshot()
        assert state.failure_count == 1
        
        # Success resets counter
        circuit.call(lambda: "success")
        
        state = circuit.get_state_snapshot()
        assert state.failure_count == 0
        assert circuit.state == CircuitState.CLOSED
    
    def test_kill_switch_bypasses_circuit(self):
        """Test ENABLE_CIRCUIT_BREAKERS kill switch"""
        circuit = CircuitBreaker(
            "test_service",
            CircuitBreakerConfig(
                failure_threshold=1,
                enabled=False  # Kill switch OFF
            )
        )
        
        # Even with failures, should not open circuit
        for _ in range(10):
            with pytest.raises(Exception):
                circuit.call(lambda: (_ for _ in ()).throw(Exception("fail")))
        
        # Circuit should still be closed because kill switch is off
        assert circuit.state == CircuitState.CLOSED
    
    def test_state_snapshot_immutability(self):
        circuit = CircuitBreaker(
            "test_service",
            CircuitBreakerConfig()
        )
        
        snapshot1 = circuit.get_state_snapshot()
        
        # Modify circuit state
        with pytest.raises(Exception):
            circuit.call(lambda: (_ for _ in ()).throw(Exception("fail")))
        
        snapshot2 = circuit.get_state_snapshot()
        
        # Original snapshot should be unchanged
        assert snapshot1.failure_count == 0
        assert snapshot2.failure_count == 1


class TestCircuitBreakerMetrics:
    """Test metrics emission"""
    
    @patch('core.circuit_breaker.circuit_state_changes')
    def test_metrics_on_state_change(self, mock_counter):
        circuit = CircuitBreaker(
            "test_service",
            CircuitBreakerConfig(failure_threshold=2)
        )
        
        # Trigger state change
        for _ in range(2):
            with pytest.raises(Exception):
                circuit.call(lambda: (_ for _ in ()).throw(Exception("fail")))
        
        # Should have recorded CLOSED → OPEN transition
        mock_counter.labels.assert_called_with(
            service="test_service",
            from_state="CLOSED",
            to_state="OPEN"
        )
    
    @patch('core.circuit_breaker.circuit_failures')
    def test_failure_counter_increments(self, mock_counter):
        circuit = CircuitBreaker(
            "test_service",
            CircuitBreakerConfig()
        )
        
        with pytest.raises(Exception):
            circuit.call(lambda: (_ for _ in ()).throw(Exception("fail")))
        
        mock_counter.labels.assert_called_with(service="test_service")
    
    @patch('core.circuit_breaker.circuit_rejected_calls')
    def test_rejection_counter_increments(self, mock_counter):
        circuit = CircuitBreaker(
            "test_service",
            CircuitBreakerConfig(failure_threshold=1)
        )
        
        # Open circuit
        with pytest.raises(Exception):
            circuit.call(lambda: (_ for _ in ()).throw(Exception("fail")))
        
        # Rejected call
        with pytest.raises(CircuitBreakerError):
            circuit.call(lambda: "should not execute")
        
        mock_counter.labels.assert_called_with(service="test_service")


class TestCircuitBreakerRegistry:
    """Test global circuit breaker registry"""
    
    def test_get_or_create_circuit_breaker(self):
        reset_all_circuits()  # Clean state
        
        config = CircuitBreakerConfig(failure_threshold=10)
        circuit1 = get_circuit_breaker("service_a", config)
        circuit2 = get_circuit_breaker("service_a")
        
        # Should return same instance
        assert circuit1 is circuit2
    
    def test_different_services_get_different_circuits(self):
        reset_all_circuits()
        
        circuit_a = get_circuit_breaker("service_a")
        circuit_b = get_circuit_breaker("service_b")
        
        assert circuit_a is not circuit_b
    
    def test_reset_all_circuits(self):
        reset_all_circuits()
        
        circuit = get_circuit_breaker("test_service", CircuitBreakerConfig(failure_threshold=1))
        
        # Open circuit
        with pytest.raises(Exception):
            circuit.call(lambda: (_ for _ in ()).throw(Exception("fail")))
        
        assert circuit.state == CircuitState.OPEN
        
        # Reset all
        reset_all_circuits()
        
        assert circuit.state == CircuitState.CLOSED


class TestCircuitBreakerAsync:
    """Test async support"""
    
    @pytest.mark.asyncio
    async def test_async_call_success(self):
        circuit = CircuitBreaker(
            "test_service",
            CircuitBreakerConfig()
        )
        
        async def async_func():
            return "async success"
        
        result = await circuit.call_async(async_func)
        assert result == "async success"
    
    @pytest.mark.asyncio
    async def test_async_call_failure(self):
        circuit = CircuitBreaker(
            "test_service",
            CircuitBreakerConfig(failure_threshold=2)
        )
        
        async def async_failing():
            raise Exception("async fail")
        
        for _ in range(2):
            with pytest.raises(Exception):
                await circuit.call_async(async_failing)
        
        assert circuit.state == CircuitState.OPEN
    
    @pytest.mark.asyncio
    async def test_async_circuit_open_rejection(self):
        circuit = CircuitBreaker(
            "test_service",
            CircuitBreakerConfig(failure_threshold=1)
        )
        
        async def async_failing():
            raise Exception("fail")
        
        with pytest.raises(Exception):
            await circuit.call_async(async_failing)
        
        with pytest.raises(CircuitBreakerError):
            await circuit.call_async(lambda: "should not execute")


class TestCircuitBreakerDecorator:
    """Test decorator pattern"""
    
    def test_protected_decorator(self):
        circuit = CircuitBreaker(
            "test_service",
            CircuitBreakerConfig(failure_threshold=2)
        )
        
        call_count = 0
        
        @circuit.protected
        def protected_func():
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise Exception("fail")
            return "success"
        
        # First 2 calls fail and open circuit
        for _ in range(2):
            with pytest.raises(Exception):
                protected_func()
        
        # Circuit is open, next call rejected
        with pytest.raises(CircuitBreakerError):
            protected_func()
        
        assert call_count == 2  # Third call was rejected before execution


# Integration test
class TestCircuitBreakerIntegration:
    """Integration tests simulating real scenarios"""
    
    def test_ad_platform_api_circuit_breaker_scenario(self):
        """
        Scenario: Meta Ads API starts failing
        1. First 5 calls fail → circuit opens
        2. Circuit rejects calls for 60 seconds
        3. After timeout, 2 successful calls → circuit closes
        """
        circuit = CircuitBreaker(
            "meta_ads_api",
            CircuitBreakerConfig(
                failure_threshold=5,
                timeout_seconds=2,  # Short for testing
                success_threshold=2
            )
        )
        
        # Simulate Meta API failures
        failure_count = 0
        def mock_meta_api_call():
            nonlocal failure_count
            if failure_count < 5:
                failure_count += 1
                raise Exception("Meta API 500 Internal Server Error")
            return {"campaigns": []}
        
        # Trigger failures
        for i in range(5):
            with pytest.raises(Exception):
                circuit.call(mock_meta_api_call)
            if i < 4:
                assert circuit.state == CircuitState.CLOSED
        
        # Circuit should be open now
        assert circuit.state == CircuitState.OPEN
        
        # Fast-fail rejections
        with pytest.raises(CircuitBreakerError):
            circuit.call(mock_meta_api_call)
        
        # Wait for timeout
        time.sleep(2.1)
        
        # Circuit tries HALF_OPEN, successful calls close it
        result1 = circuit.call(mock_meta_api_call)
        # After 1 success, might still be in HALF_OPEN
        result2 = circuit.call(mock_meta_api_call)
        
        # After 2 successes, should be CLOSED
        assert circuit.state == CircuitState.CLOSED
        assert result2 == {"campaigns": []}


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
