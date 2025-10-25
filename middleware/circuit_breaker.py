"""
Circuit Breaker Middleware
Implements Q_032: Circuit breaker opens after 5 consecutive external API failures

State Machine:
- CLOSED: Normal operation, requests pass through
- OPEN: Circuit tripped, requests fail fast with fallback
- HALF_OPEN: Testing if service recovered, single probe request

Acceptance:
- Opens after failure_threshold consecutive failures within timeout_seconds
- Transitions to HALF_OPEN after recovery_timeout_seconds
- Closes on successful probe in HALF_OPEN
- Emits metrics: circuit_breaker_opens, failures, successes
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Callable
from collections import deque
import time
import logging

logger = logging.getLogger(__name__)


class CircuitBreakerState(Enum):
    """Circuit breaker states per Q_032 acceptance"""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open and request is rejected"""
    pass


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior"""
    failure_threshold: int = 5  # Q_032: Open after 5 failures
    timeout_seconds: int = 60  # Window for counting failures
    recovery_timeout_seconds: int = 30  # Time before attempting recovery
    fallback_response: Optional[Dict[str, Any]] = None
    

class CircuitBreaker:
    """
    Circuit breaker for external API calls
    
    Prevents retry storms by failing fast when service is unavailable.
    Implements state machine: CLOSED → OPEN → HALF_OPEN → CLOSED
    
    Usage:
        cb = CircuitBreaker(name="meta_ads_api", config=config)
        
        try:
            with cb:
                response = external_api_call()
        except CircuitBreakerOpenError:
            response = cb.get_fallback_response()
    """
    
    def __init__(
        self,
        name: str,
        config: CircuitBreakerConfig
    ):
        self.name = name
        self.config = config
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.last_failure_time: Optional[float] = None
        self.opened_at: Optional[float] = None
        
        # Track failure timestamps for windowing
        self.failure_timestamps: deque = deque(maxlen=config.failure_threshold)
        
        # Metrics (would integrate with Prometheus in production)
        self.metrics: Dict[str, int] = {
            'circuit_breaker_opens': 0,
            'circuit_breaker_closes': 0,
            'circuit_breaker_failures': 0,
            'circuit_breaker_successes': 0
        }
        
        logger.info(
            f"Circuit breaker '{name}' initialized: "
            f"threshold={config.failure_threshold}, "
            f"window={config.timeout_seconds}s"
        )
    
    def __enter__(self):
        """Context manager entry: check if circuit allows request"""
        self._check_state_transition()
        
        if self.state == CircuitBreakerState.OPEN:
            logger.warning(
                f"Circuit breaker '{self.name}' is OPEN, "
                f"rejecting request (opened {time.time() - self.opened_at:.1f}s ago)"
            )
            raise CircuitBreakerOpenError(
                f"Circuit breaker '{self.name}' is OPEN. "
                f"Service unavailable."
            )
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit: record success or failure"""
        if exc_type is None:
            # Success
            self._record_success()
        else:
            # Failure
            self._record_failure()
        
        # Don't suppress the exception
        return False
    
    def _check_state_transition(self):
        """Check if state should transition based on time"""
        current_time = time.time()
        
        if self.state == CircuitBreakerState.OPEN:
            # Check if recovery timeout elapsed
            if (self.opened_at and 
                current_time - self.opened_at >= self.config.recovery_timeout_seconds):
                self._transition_to_half_open()
    
    def _record_success(self):
        """Record successful request"""
        self.metrics['circuit_breaker_successes'] += 1
        
        if self.state == CircuitBreakerState.HALF_OPEN:
            # Success in HALF_OPEN closes the circuit
            self._transition_to_closed()
        elif self.state == CircuitBreakerState.CLOSED:
            # Success resets failure count
            self.failure_count = 0
            self.failure_timestamps.clear()
            logger.debug(f"Circuit breaker '{self.name}': success, counter reset")
    
    def _record_failure(self):
        """Record failed request and check threshold"""
        self.metrics['circuit_breaker_failures'] += 1
        current_time = time.time()
        self.last_failure_time = current_time
        
        if self.state == CircuitBreakerState.HALF_OPEN:
            # Failure in HALF_OPEN reopens the circuit
            self._transition_to_open()
            return
        
        # Add failure timestamp
        self.failure_timestamps.append(current_time)
        
        # Count failures within time window
        window_start = current_time - self.config.timeout_seconds
        recent_failures = sum(
            1 for ts in self.failure_timestamps 
            if ts >= window_start
        )
        
        self.failure_count = recent_failures
        
        logger.debug(
            f"Circuit breaker '{self.name}': failure recorded "
            f"({self.failure_count}/{self.config.failure_threshold})"
        )
        
        # Check if threshold reached
        if self.failure_count >= self.config.failure_threshold:
            self._transition_to_open()
    
    def _transition_to_open(self):
        """Transition to OPEN state"""
        if self.state != CircuitBreakerState.OPEN:
            self.state = CircuitBreakerState.OPEN
            self.opened_at = time.time()
            self.metrics['circuit_breaker_opens'] += 1
            
            logger.error(
                f"Circuit breaker '{self.name}' OPENED after "
                f"{self.failure_count} failures. "
                f"Will attempt recovery in {self.config.recovery_timeout_seconds}s"
            )
    
    def _transition_to_half_open(self):
        """Transition to HALF_OPEN state for recovery probe"""
        if self.state == CircuitBreakerState.OPEN:
            self.state = CircuitBreakerState.HALF_OPEN
            logger.info(
                f"Circuit breaker '{self.name}' transitioning to HALF_OPEN "
                f"for recovery probe"
            )
    
    def _transition_to_closed(self):
        """Transition to CLOSED state after successful recovery"""
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.failure_timestamps.clear()
        self.opened_at = None
        self.metrics['circuit_breaker_closes'] += 1
        
        logger.info(
            f"Circuit breaker '{self.name}' CLOSED after successful recovery"
        )
    
    def get_fallback_response(self) -> Optional[Dict[str, Any]]:
        """Get fallback response when circuit is open"""
        return self.config.fallback_response
    
    def get_state(self) -> CircuitBreakerState:
        """Get current circuit breaker state"""
        return self.state
    
    def reset(self):
        """Manually reset circuit breaker (admin operation)"""
        logger.warning(f"Circuit breaker '{self.name}' manually reset")
        self._transition_to_closed()


# Global registry for circuit breakers by API endpoint
_circuit_breakers: Dict[str, CircuitBreaker] = {}


def get_circuit_breaker(
    name: str,
    config: Optional[CircuitBreakerConfig] = None
) -> CircuitBreaker:
    """
    Get or create circuit breaker for named endpoint
    
    Args:
        name: Unique identifier (e.g., 'meta_ads_api', 'google_ads_api')
        config: Configuration (uses default if not provided)
    
    Returns:
        CircuitBreaker instance
    """
    if name not in _circuit_breakers:
        if config is None:
            config = CircuitBreakerConfig()
        _circuit_breakers[name] = CircuitBreaker(name, config)
    
    return _circuit_breakers[name]


def reset_all_circuit_breakers():
    """Reset all circuit breakers (testing/admin operation)"""
    for cb in _circuit_breakers.values():
        cb.reset()
