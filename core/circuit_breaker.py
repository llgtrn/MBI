"""
Circuit Breaker and Kill-Switch Implementation

Provides emergency halt capabilities for the MBI system with <5s response time.
Implements circuit breaker patterns for fault isolation and graceful degradation.

Requirements:
- Q_016: Kill-switch must halt all operations within 5 seconds
- Emit Prometheus metrics for all state transitions
- Support graceful shutdown with configurable grace periods
- Enable global coordination across all services
- Ensure no data corruption during emergency halts

Related: Q_013/Q_140 emergency stop scenarios
Component: Infra_ExternalAPIs (CRITICAL+P0)
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import List, Optional, Callable, Any, Dict
from enum import Enum
import logging
from dataclasses import dataclass, field
from contextlib import asynccontextmanager

from prometheus_client import Counter, Histogram, Gauge

logger = logging.getLogger(__name__)


# Prometheus Metrics
kill_switch_activations = Counter(
    'kill_switch_activations_total',
    'Total kill-switch activations',
    ['service_name', 'reason']
)

kill_switch_halt_duration = Histogram(
    'kill_switch_halt_duration_seconds',
    'Time taken for emergency halt to complete',
    ['service_name'],
    buckets=[0.1, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0]
)

kill_switches_active = Gauge(
    'kill_switches_active',
    'Number of currently active kill-switches'
)

circuit_breaker_state_changes = Counter(
    'circuit_breaker_state_changes_total',
    'Circuit breaker state transitions',
    ['service_name', 'from_state', 'to_state']
)


class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"      # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing recovery


class KillSwitchError(Exception):
    """Raised when operation attempted while kill-switch is active"""
    pass


class EmergencyHaltError(Exception):
    """Raised when emergency halt fails or times out"""
    pass


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior"""
    failure_threshold: int = 5  # Failures before opening
    success_threshold: int = 2  # Successes to close from half-open
    timeout_seconds: int = 60   # Time before attempting half-open
    kill_switch_name: Optional[str] = None


@dataclass
class KillSwitchState:
    """Internal state for kill-switch"""
    active: bool = False
    enabled: bool = True
    activation_reason: Optional[str] = None
    activated_at: Optional[datetime] = None
    deactivated_at: Optional[datetime] = None
    grace_period_seconds: int = 0
    grace_period_end: Optional[datetime] = None


class KillSwitch:
    """
    Emergency kill-switch for halting operations
    
    Features:
    - <5s emergency halt guarantee
    - Graceful shutdown with configurable grace periods
    - Prometheus metrics emission
    - Global coordination support
    - State persistence before shutdown
    
    Example:
        ks = KillSwitch(name="payment_service")
        
        # In normal operation
        ks.check_if_active()  # Raises if active
        
        # Emergency activation
        ks.activate(reason="Payment provider outage")
        
        # Emergency halt
        await ks.emergency_halt(operations=running_tasks)
    """
    
    _registry: Dict[str, 'KillSwitch'] = {}
    
    def __init__(self, name: str, enabled: bool = True):
        self.name = name
        self._state = KillSwitchState(enabled=enabled)
        self._lock = asyncio.Lock()
        
        # Auto-register for global halt coordination
        KillSwitch._registry[name] = self
        
        logger.info(f"KillSwitch '{name}' initialized (enabled={enabled})")
    
    def is_enabled(self) -> bool:
        """Check if kill-switch is enabled"""
        return self._state.enabled
    
    def is_active(self) -> bool:
        """Check if kill-switch is currently active"""
        return self._state.active
    
    @property
    def activation_reason(self) -> Optional[str]:
        """Get reason for current/last activation"""
        return self._state.activation_reason
    
    @property
    def activated_at(self) -> Optional[datetime]:
        """Get timestamp of current/last activation"""
        return self._state.activated_at
    
    @property
    def deactivated_at(self) -> Optional[datetime]:
        """Get timestamp of last deactivation"""
        return self._state.deactivated_at
    
    def activate(self, reason: str, grace_period_seconds: int = 0):
        """
        Activate kill-switch with optional grace period
        
        Args:
            reason: Reason for activation (for logging/metrics)
            grace_period_seconds: Time to allow in-flight operations to complete
        """
        if not self._state.enabled:
            logger.warning(f"KillSwitch '{self.name}' is disabled, ignoring activation")
            return
        
        self._state.active = True
        self._state.activation_reason = reason
        self._state.activated_at = datetime.utcnow()
        self._state.grace_period_seconds = grace_period_seconds
        
        if grace_period_seconds > 0:
            self._state.grace_period_end = (
                datetime.utcnow() + timedelta(seconds=grace_period_seconds)
            )
        
        # Update metrics
        kill_switch_activations.labels(
            service_name=self.name,
            reason=reason
        ).inc()
        
        kill_switches_active.inc()
        
        logger.critical(
            f"KillSwitch '{self.name}' ACTIVATED: {reason} "
            f"(grace_period={grace_period_seconds}s)"
        )
    
    def deactivate(self):
        """Deactivate kill-switch and clear state"""
        if not self._state.active:
            logger.warning(f"KillSwitch '{self.name}' already inactive")
            return
        
        self._state.active = False
        self._state.deactivated_at = datetime.utcnow()
        self._state.activation_reason = None
        self._state.grace_period_end = None
        
        kill_switches_active.dec()
        
        logger.info(f"KillSwitch '{self.name}' deactivated")
    
    def check_if_active(self):
        """
        Check if kill-switch is active, raise if so
        
        Raises:
            KillSwitchError: If kill-switch is currently active
        """
        if not self._state.active:
            return
        
        # Check if still in grace period
        if self._state.grace_period_end:
            if datetime.utcnow() < self._state.grace_period_end:
                # Still in grace period, allow operation
                return
        
        raise KillSwitchError(
            f"KillSwitch '{self.name}' is active: {self._state.activation_reason}"
        )
    
    async def emergency_halt(
        self,
        operations: List[asyncio.Task] = None,
        timeout_seconds: int = 5,
        force_after_timeout: bool = True,
        state_manager: Optional[Any] = None
    ):
        """
        Execute emergency halt of all operations
        
        CRITICAL: Must complete within 5 seconds (Q_016 requirement)
        
        Args:
            operations: List of async tasks to cancel
            timeout_seconds: Max time to wait for graceful shutdown
            force_after_timeout: Force-kill operations after timeout
            state_manager: Optional state manager for persistence
        
        Raises:
            EmergencyHaltError: If halt fails or exceeds timeout
        """
        start_time = time.time()
        operations = operations or []
        
        logger.critical(
            f"KillSwitch '{self.name}' initiating EMERGENCY HALT "
            f"({len(operations)} operations, timeout={timeout_seconds}s)"
        )
        
        try:
            # Step 1: Persist critical state (if state manager provided)
            if state_manager:
                await asyncio.wait_for(
                    state_manager.persist_state(),
                    timeout=1.0  # 1s max for state persistence
                )
            
            # Step 2: Cancel all operations
            for op in operations:
                if not op.done():
                    op.cancel()
            
            # Step 3: Wait for operations to complete (with timeout)
            if operations:
                try:
                    await asyncio.wait_for(
                        asyncio.gather(*operations, return_exceptions=True),
                        timeout=timeout_seconds - 1  # Reserve 1s buffer
                    )
                except asyncio.TimeoutError:
                    if force_after_timeout:
                        logger.error(
                            f"Emergency halt timeout exceeded, "
                            f"force-terminating {len(operations)} operations"
                        )
                        # Force cancel any remaining
                        for op in operations:
                            if not op.done():
                                op.cancel()
                    else:
                        raise EmergencyHaltError(
                            f"Emergency halt exceeded {timeout_seconds}s timeout"
                        )
            
            elapsed = time.time() - start_time
            
            # Record metrics
            self._record_halt_duration(elapsed)
            
            # Verify <5s requirement
            if elapsed >= 5.0:
                logger.error(
                    f"REQUIREMENT VIOLATION: Emergency halt took {elapsed:.2f}s, "
                    f"exceeds 5s requirement (Q_016)"
                )
                raise EmergencyHaltError(
                    f"Emergency halt took {elapsed:.2f}s, exceeds 5s requirement"
                )
            
            logger.info(
                f"Emergency halt completed successfully in {elapsed:.2f}s "
                f"({len(operations)} operations)"
            )
            
        except Exception as e:
            elapsed = time.time() - start_time
            logger.exception(f"Emergency halt failed after {elapsed:.2f}s")
            raise EmergencyHaltError(f"Emergency halt failed: {e}") from e
    
    def _record_halt_duration(self, duration_seconds: float):
        """Record halt duration in metrics"""
        kill_switch_halt_duration.labels(
            service_name=self.name
        ).observe(duration_seconds)


class CircuitBreaker:
    """
    Circuit breaker for fault isolation
    
    Integrates with kill-switch for emergency scenarios.
    Implements standard circuit breaker pattern with CLOSED/OPEN/HALF_OPEN states.
    """
    
    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[datetime] = None
        self._kill_switch: Optional[KillSwitch] = None
        
        if config.kill_switch_name:
            self._kill_switch = get_kill_switch(config.kill_switch_name)
        
        logger.info(f"CircuitBreaker initialized: {config}")
    
    def is_open(self) -> bool:
        """Check if circuit is open (failing)"""
        # Check kill-switch first
        if self._kill_switch and self._kill_switch.is_active():
            return True
        
        return self._state == CircuitState.OPEN
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function through circuit breaker
        
        Raises:
            KillSwitchError: If kill-switch is active
            Exception: If circuit is open or call fails
        """
        # Check kill-switch
        if self._kill_switch:
            self._kill_switch.check_if_active()
        
        # Check circuit state
        if self._state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self._transition_to(CircuitState.HALF_OPEN)
            else:
                raise Exception("Circuit breaker is OPEN")
        
        try:
            # Execute function
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            # Record success
            self._on_success()
            return result
            
        except Exception as e:
            # Record failure
            self._on_failure()
            raise
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset"""
        if not self._last_failure_time:
            return False
        
        elapsed = (datetime.utcnow() - self._last_failure_time).total_seconds()
        return elapsed >= self.config.timeout_seconds
    
    def _on_success(self):
        """Handle successful call"""
        if self._state == CircuitState.HALF_OPEN:
            self._success_count += 1
            if self._success_count >= self.config.success_threshold:
                self._transition_to(CircuitState.CLOSED)
                self._success_count = 0
        
        self._failure_count = 0
    
    def _on_failure(self):
        """Handle failed call"""
        self._failure_count += 1
        self._last_failure_time = datetime.utcnow()
        
        if self._state == CircuitState.HALF_OPEN:
            self._transition_to(CircuitState.OPEN)
            self._success_count = 0
        
        elif self._failure_count >= self.config.failure_threshold:
            self._transition_to(CircuitState.OPEN)
    
    def _transition_to(self, new_state: CircuitState):
        """Transition to new circuit state"""
        old_state = self._state
        self._state = new_state
        
        circuit_breaker_state_changes.labels(
            service_name=self.config.kill_switch_name or "unknown",
            from_state=old_state.value,
            to_state=new_state.value
        ).inc()
        
        logger.warning(f"Circuit breaker state: {old_state.value} -> {new_state.value}")


# Global functions

def get_kill_switch(name: str) -> KillSwitch:
    """Get or create kill-switch by name"""
    if name not in KillSwitch._registry:
        return KillSwitch(name=name)
    return KillSwitch._registry[name]


async def emergency_halt_all(reason: str, timeout_seconds: int = 5):
    """
    Execute emergency halt across ALL registered kill-switches
    
    CRITICAL: Must complete within 5 seconds total
    
    Args:
        reason: Reason for global emergency halt
        timeout_seconds: Max time for global halt
    """
    start_time = time.time()
    
    logger.critical(
        f"GLOBAL EMERGENCY HALT initiated: {reason} "
        f"({len(KillSwitch._registry)} services)"
    )
    
    # Activate all kill-switches
    for ks in KillSwitch._registry.values():
        ks.activate(reason=f"Global halt: {reason}")
    
    # Halt all services concurrently
    halt_tasks = [
        ks.emergency_halt(timeout_seconds=timeout_seconds - 1)
        for ks in KillSwitch._registry.values()
    ]
    
    try:
        await asyncio.wait_for(
            asyncio.gather(*halt_tasks, return_exceptions=True),
            timeout=timeout_seconds
        )
    except asyncio.TimeoutError:
        elapsed = time.time() - start_time
        logger.error(
            f"GLOBAL EMERGENCY HALT timeout after {elapsed:.2f}s "
            f"(requirement: {timeout_seconds}s)"
        )
        raise EmergencyHaltError(
            f"Global emergency halt exceeded {timeout_seconds}s timeout"
        )
    
    elapsed = time.time() - start_time
    
    if elapsed >= 5.0:
        logger.error(
            f"REQUIREMENT VIOLATION: Global halt took {elapsed:.2f}s, "
            f"exceeds 5s requirement"
        )
    
    logger.info(
        f"Global emergency halt completed in {elapsed:.2f}s "
        f"({len(KillSwitch._registry)} services halted)"
    )


@asynccontextmanager
async def kill_switch_protection(switch: KillSwitch):
    """
    Context manager for kill-switch protected operations
    
    Example:
        async with kill_switch_protection(my_switch):
            # Operation will be checked before execution
            await do_critical_operation()
    """
    switch.check_if_active()
    try:
        yield
    finally:
        pass
