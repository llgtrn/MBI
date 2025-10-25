"""
Circuit Breaker Tests
Tests for Q_032: Circuit breaker opens after 5 consecutive external API failures within 60s

Acceptance:
- test_circuit_breaker_opens_5_failures passes
- CircuitBreakerState enum {CLOSED, OPEN, HALF_OPEN}
- metric circuit_breaker_opens counter >0 in test
- dry_run: Mock API failure sequence triggers open state
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
from middleware.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerState,
    CircuitBreakerOpenError,
    CircuitBreakerConfig
)


class TestCircuitBreakerStateTransitions:
    """Test state machine transitions"""
    
    def test_initial_state_is_closed(self):
        """Circuit breaker starts in CLOSED state"""
        cb = CircuitBreaker(name="test", config=CircuitBreakerConfig())
        assert cb.state == CircuitBreakerState.CLOSED
        assert cb.failure_count == 0
    
    def test_circuit_breaker_opens_5_failures(self):
        """Q_032 acceptance: Opens after 5 consecutive failures in 60s"""
        config = CircuitBreakerConfig(
            failure_threshold=5,
            timeout_seconds=60,
            recovery_timeout_seconds=30
        )
        cb = CircuitBreaker(name="test_api", config=config)
        
        # Mock time to control failure window
        with patch('middleware.circuit_breaker.time') as mock_time:
            base_time = 1000.0
            mock_time.time.return_value = base_time
            
            # Simulate 5 consecutive failures within 60s
            for i in range(5):
                mock_time.time.return_value = base_time + i
                try:
                    with cb:
                        raise Exception("API failure")
                except Exception:
                    pass
            
            # Circuit should now be OPEN
            assert cb.state == CircuitBreakerState.OPEN
            assert cb.failure_count == 5
            
            # Verify metric incremented
            assert cb.metrics['circuit_breaker_opens'] == 1
    
    def test_open_circuit_raises_immediately(self):
        """Open circuit breaker rejects requests without calling function"""
        config = CircuitBreakerConfig(failure_threshold=2)
        cb = CircuitBreaker(name="test", config=config)
        
        # Force circuit to OPEN
        cb.state = CircuitBreakerState.OPEN
        
        with pytest.raises(CircuitBreakerOpenError) as exc_info:
            with cb:
                pass  # Should not reach here
        
        assert "Circuit breaker 'test' is OPEN" in str(exc_info.value)
    
    def test_half_open_after_recovery_timeout(self):
        """Circuit transitions to HALF_OPEN after recovery timeout"""
        config = CircuitBreakerConfig(
            failure_threshold=2,
            recovery_timeout_seconds=30
        )
        cb = CircuitBreaker(name="test", config=config)
        
        with patch('middleware.circuit_breaker.time') as mock_time:
            base_time = 1000.0
            mock_time.time.return_value = base_time
            
            # Open the circuit
            for i in range(2):
                try:
                    with cb:
                        raise Exception("Fail")
                except Exception:
                    pass
            
            assert cb.state == CircuitBreakerState.OPEN
            
            # Advance time past recovery timeout
            mock_time.time.return_value = base_time + 31
            
            # Next call should transition to HALF_OPEN
            cb._check_state_transition()
            assert cb.state == CircuitBreakerState.HALF_OPEN
    
    def test_half_open_success_closes_circuit(self):
        """Successful call in HALF_OPEN closes the circuit"""
        cb = CircuitBreaker(name="test", config=CircuitBreakerConfig())
        cb.state = CircuitBreakerState.HALF_OPEN
        
        # Successful call
        with cb:
            pass  # Success
        
        assert cb.state == CircuitBreakerState.CLOSED
        assert cb.failure_count == 0
    
    def test_half_open_failure_reopens_circuit(self):
        """Failed call in HALF_OPEN reopens the circuit"""
        cb = CircuitBreaker(name="test", config=CircuitBreakerConfig())
        cb.state = CircuitBreakerState.HALF_OPEN
        
        # Failed call
        try:
            with cb:
                raise Exception("Still failing")
        except Exception:
            pass
        
        assert cb.state == CircuitBreakerState.OPEN
    
    def test_successful_calls_reset_failure_count(self):
        """Successful calls in CLOSED state reset failure counter"""
        config = CircuitBreakerConfig(failure_threshold=5)
        cb = CircuitBreaker(name="test", config=config)
        
        # 3 failures
        for _ in range(3):
            try:
                with cb:
                    raise Exception("Fail")
            except Exception:
                pass
        
        assert cb.failure_count == 3
        assert cb.state == CircuitBreakerState.CLOSED
        
        # 1 success resets counter
        with cb:
            pass
        
        assert cb.failure_count == 0
        assert cb.state == CircuitBreakerState.CLOSED


class TestCircuitBreakerFailureWindow:
    """Test failure counting within time window"""
    
    def test_failures_outside_window_dont_count(self):
        """Failures older than timeout_seconds don't count toward threshold"""
        config = CircuitBreakerConfig(
            failure_threshold=3,
            timeout_seconds=60
        )
        cb = CircuitBreaker(name="test", config=config)
        
        with patch('middleware.circuit_breaker.time') as mock_time:
            base_time = 1000.0
            
            # 2 failures at t=0
            mock_time.time.return_value = base_time
            for _ in range(2):
                try:
                    with cb:
                        raise Exception("Fail")
                except Exception:
                    pass
            
            assert cb.failure_count == 2
            
            # Advance time beyond window (65 seconds)
            mock_time.time.return_value = base_time + 65
            
            # 1 more failure (old failures expired)
            try:
                with cb:
                    raise Exception("Fail")
            except Exception:
                pass
            
            # Should only count recent failure
            assert cb.failure_count == 1
            assert cb.state == CircuitBreakerState.CLOSED


class TestCircuitBreakerMetrics:
    """Test observability metrics"""
    
    def test_metrics_emitted(self):
        """Verify all required metrics are tracked"""
        cb = CircuitBreaker(name="test_api", config=CircuitBreakerConfig())
        
        assert 'circuit_breaker_opens' in cb.metrics
        assert 'circuit_breaker_closes' in cb.metrics
        assert 'circuit_breaker_failures' in cb.metrics
        assert 'circuit_breaker_successes' in cb.metrics
        
        # Initial values
        assert cb.metrics['circuit_breaker_opens'] == 0
    
    def test_failure_metric_increments(self):
        """Failure metric increments on each failure"""
        cb = CircuitBreaker(name="test", config=CircuitBreakerConfig())
        
        for i in range(3):
            try:
                with cb:
                    raise Exception("Fail")
            except Exception:
                pass
        
        assert cb.metrics['circuit_breaker_failures'] == 3
    
    def test_success_metric_increments(self):
        """Success metric increments on each success"""
        cb = CircuitBreaker(name="test", config=CircuitBreakerConfig())
        
        for _ in range(5):
            with cb:
                pass
        
        assert cb.metrics['circuit_breaker_successes'] == 5


class TestCircuitBreakerFallback:
    """Test fallback behavior when circuit is open"""
    
    def test_fallback_response_provided(self):
        """Fallback response returned when circuit is open"""
        fallback_data = {"status": "degraded", "data": []}
        config = CircuitBreakerConfig(
            failure_threshold=1,
            fallback_response=fallback_data
        )
        cb = CircuitBreaker(name="test", config=config)
        
        # Open the circuit
        try:
            with cb:
                raise Exception("Fail")
        except Exception:
            pass
        
        assert cb.state == CircuitBreakerState.OPEN
        
        # Get fallback
        result = cb.get_fallback_response()
        assert result == fallback_data


class TestCircuitBreakerIntegration:
    """Integration tests with mock external API"""
    
    def test_circuit_breaker_prevents_retry_storm(self):
        """Q_032: Circuit breaker prevents retry storms during outages"""
        config = CircuitBreakerConfig(failure_threshold=5)
        cb = CircuitBreaker(name="meta_ads_api", config=config)
        
        mock_api = Mock()
        mock_api.side_effect = Exception("API down")
        
        call_count = 0
        
        # Simulate 10 attempts
        for i in range(10):
            try:
                with cb:
                    call_count += 1
                    mock_api()
            except (Exception, CircuitBreakerOpenError):
                pass
        
        # Circuit should open after 5 failures
        # Remaining 5 attempts should be rejected without calling API
        assert call_count == 5  # Only 5 actual API calls
        assert cb.state == CircuitBreakerState.OPEN
        assert cb.metrics['circuit_breaker_opens'] == 1
