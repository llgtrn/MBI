"""
Kill-Switch Circuit Breaker Implementation

Requirement: Q_016 - Emergency kill-switch must halt all operations <5s
Acceptance:
- All operations stop within 5 seconds when kill-switch activated
- Graceful shutdown with proper state persistence
- Prometheus metrics emitted for kill-switch events
- No data corruption during emergency halt

Related: Q_013/Q_140 emergency stop scenarios
Component: Infra_ExternalAPIs (CRITICAL+P0)
Owner: Infra team
"""

import pytest
import asyncio
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
from prometheus_client import REGISTRY

from core.circuit_breaker import (
    KillSwitch,
    KillSwitchError,
    EmergencyHaltError,
    CircuitBreakerConfig,
    get_kill_switch,
    emergency_halt_all
)


class TestKillSwitchBasics:
    """Test basic kill-switch functionality and configuration"""
    
    def test_kill_switch_default_enabled(self):
        """Kill-switch should be enabled by default"""
        ks = KillSwitch(name="test_service")
        assert ks.is_enabled() is True
        assert ks.is_active() is False
    
    def test_kill_switch_activation(self):
        """Activating kill-switch should set active flag"""
        ks = KillSwitch(name="test_service")
        ks.activate(reason="test activation")
        
        assert ks.is_active() is True
        assert ks.activation_reason == "test activation"
        assert ks.activated_at is not None
    
    def test_kill_switch_deactivation(self):
        """Deactivating kill-switch should clear active flag"""
        ks = KillSwitch(name="test_service")
        ks.activate(reason="test")
        ks.deactivate()
        
        assert ks.is_active() is False
        assert ks.deactivated_at is not None


class TestEmergencyHalt:
    """Test emergency halt functionality and timing requirements"""
    
    @pytest.mark.asyncio
    async def test_halt_completes_within_5_seconds(self):
        """CRITICAL: Emergency halt must complete within 5 seconds"""
        ks = KillSwitch(name="critical_service")
        
        # Create mock long-running operations
        mock_ops = [AsyncMock() for _ in range(10)]
        for op in mock_ops:
            op.cancel = Mock()
        
        start_time = time.time()
        
        await ks.emergency_halt(
            operations=mock_ops,
            timeout_seconds=5
        )
        
        elapsed = time.time() - start_time
        
        # Must complete within 5 seconds
        assert elapsed < 5.0, f"Halt took {elapsed}s, exceeds 5s requirement"
        
        # All operations should be cancelled
        for op in mock_ops:
            op.cancel.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_halt_forces_termination_after_timeout(self):
        """Operations not completing within timeout should be force-terminated"""
        ks = KillSwitch(name="test_service")
        
        # Create operation that won't complete
        async def stuck_operation():
            await asyncio.sleep(100)  # Intentionally stuck
        
        task = asyncio.create_task(stuck_operation())
        
        with pytest.raises(EmergencyHaltError):
            await ks.emergency_halt(
                operations=[task],
                timeout_seconds=1,
                force_after_timeout=True
            )
        
        # Task should be cancelled
        assert task.cancelled()
    
    @pytest.mark.asyncio
    async def test_halt_persists_state_before_shutdown(self):
        """Emergency halt must persist critical state before shutdown"""
        ks = KillSwitch(name="stateful_service")
        
        mock_state_manager = Mock()
        mock_state_manager.persist_state = AsyncMock()
        
        await ks.emergency_halt(
            operations=[],
            state_manager=mock_state_manager
        )
        
        # State persistence must be called
        mock_state_manager.persist_state.assert_called_once()


class TestKillSwitchMetrics:
    """Test Prometheus metrics emission for kill-switch events"""
    
    def test_activation_increments_counter(self):
        """Kill-switch activation should increment Prometheus counter"""
        # Get baseline metric value
        before = self._get_metric_value('kill_switch_activations_total')
        
        ks = KillSwitch(name="metrics_test")
        ks.activate(reason="test metric")
        
        after = self._get_metric_value('kill_switch_activations_total')
        
        assert after > before, "Activation counter should increment"
    
    def test_halt_time_recorded_as_histogram(self):
        """Emergency halt duration should be recorded in histogram"""
        ks = KillSwitch(name="timing_test")
        
        # Record halt event
        ks._record_halt_duration(duration_seconds=3.5)
        
        # Verify histogram has sample
        metric = self._get_histogram('kill_switch_halt_duration_seconds')
        assert metric is not None
        assert metric._sum.get() > 0
    
    def test_active_kill_switches_gauge(self):
        """Active kill-switches should update gauge metric"""
        before = self._get_metric_value('kill_switches_active')
        
        ks1 = KillSwitch(name="gauge_test_1")
        ks1.activate(reason="test")
        
        middle = self._get_metric_value('kill_switches_active')
        assert middle > before
        
        ks1.deactivate()
        
        after = self._get_metric_value('kill_switches_active')
        assert after == before
    
    def _get_metric_value(self, metric_name: str) -> float:
        """Helper to get current metric value"""
        for metric in REGISTRY.collect():
            if metric.name == metric_name:
                for sample in metric.samples:
                    if sample.name == metric_name:
                        return sample.value
        return 0.0
    
    def _get_histogram(self, metric_name: str):
        """Helper to get histogram metric"""
        for metric in REGISTRY.collect():
            if metric.name == metric_name:
                return metric
        return None


class TestGracefulShutdown:
    """Test graceful shutdown behavior during kill-switch activation"""
    
    @pytest.mark.asyncio
    async def test_inflight_requests_complete_before_halt(self):
        """In-flight requests should complete before full halt"""
        ks = KillSwitch(name="graceful_test")
        
        # Simulate in-flight requests
        inflight = []
        for i in range(5):
            async def request():
                await asyncio.sleep(0.5)
                return f"completed_{i}"
            
            inflight.append(asyncio.create_task(request()))
        
        # Activate kill-switch with grace period
        ks.activate(reason="graceful test", grace_period_seconds=2)
        
        # Wait for grace period
        await asyncio.sleep(2.1)
        
        # All inflight should complete
        results = await asyncio.gather(*inflight)
        assert len(results) == 5
        assert all("completed" in r for r in results)
    
    @pytest.mark.asyncio
    async def test_new_requests_rejected_during_grace(self):
        """New requests during grace period should be rejected"""
        ks = KillSwitch(name="reject_test")
        ks.activate(reason="test", grace_period_seconds=5)
        
        # Attempt new request during grace period
        with pytest.raises(KillSwitchError) as exc_info:
            ks.check_if_active()
        
        assert "kill-switch active" in str(exc_info.value).lower()


class TestCircuitBreakerIntegration:
    """Test kill-switch integration with circuit breaker patterns"""
    
    @pytest.mark.asyncio
    async def test_circuit_opens_on_kill_switch(self):
        """Circuit breaker should open when kill-switch activates"""
        config = CircuitBreakerConfig(
            failure_threshold=3,
            timeout_seconds=60,
            kill_switch_name="integration_test"
        )
        
        ks = get_kill_switch("integration_test")
        ks.activate(reason="circuit test")
        
        # Circuit should immediately open
        from core.circuit_breaker import CircuitBreaker
        cb = CircuitBreaker(config)
        
        assert cb.is_open() is True
    
    @pytest.mark.asyncio
    async def test_operations_fail_fast_when_killed(self):
        """Operations should fail immediately when kill-switch active"""
        ks = KillSwitch(name="fail_fast_test")
        ks.activate(reason="test")
        
        start = time.time()
        
        with pytest.raises(KillSwitchError):
            ks.check_if_active()
        
        elapsed = time.time() - start
        
        # Should fail instantly, not wait
        assert elapsed < 0.1, "Should fail fast, not wait"


class TestKillSwitchCoordination:
    """Test global kill-switch coordination across services"""
    
    @pytest.mark.asyncio
    async def test_emergency_halt_all_services(self):
        """emergency_halt_all() should halt all registered services <5s"""
        # Register multiple services
        services = [
            KillSwitch(name=f"service_{i}") 
            for i in range(5)
        ]
        
        start = time.time()
        
        await emergency_halt_all(reason="global emergency test")
        
        elapsed = time.time() - start
        
        # Must complete within 5 seconds
        assert elapsed < 5.0
        
        # All services should be active
        for svc in services:
            assert svc.is_active()
    
    def test_kill_switch_registry(self):
        """All kill-switches should auto-register for global halt"""
        initial_count = len(get_kill_switch._registry)
        
        ks = KillSwitch(name="registry_test")
        
        final_count = len(get_kill_switch._registry)
        
        assert final_count == initial_count + 1


class TestRollbackScenarios:
    """Test kill-switch deactivation and service recovery"""
    
    def test_deactivate_clears_state(self):
        """Deactivating kill-switch should clear all active state"""
        ks = KillSwitch(name="rollback_test")
        ks.activate(reason="test")
        
        assert ks.is_active()
        
        ks.deactivate()
        
        assert not ks.is_active()
        assert ks.activation_reason is None
        assert ks.deactivated_at is not None
    
    def test_multiple_activation_cycles(self):
        """Kill-switch should support multiple activate/deactivate cycles"""
        ks = KillSwitch(name="cycle_test")
        
        for i in range(3):
            ks.activate(reason=f"cycle {i}")
            assert ks.is_active()
            
            ks.deactivate()
            assert not ks.is_active()


# Contract validation
def test_kill_switch_contract():
    """Verify KillSwitch implements required contract"""
    ks = KillSwitch(name="contract_test")
    
    # Required methods
    assert hasattr(ks, 'activate')
    assert hasattr(ks, 'deactivate')
    assert hasattr(ks, 'is_active')
    assert hasattr(ks, 'is_enabled')
    assert hasattr(ks, 'check_if_active')
    assert hasattr(ks, 'emergency_halt')
    
    # Required attributes
    assert hasattr(ks, 'name')
    assert hasattr(ks, 'activation_reason')
    assert hasattr(ks, 'activated_at')
    assert hasattr(ks, 'deactivated_at')
