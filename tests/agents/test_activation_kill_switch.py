"""
Test suite for ActivationAgent kill-switch and emergency halt mechanisms.
Validates Q_030: Kill-switch halts operations in <5 seconds.
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta
from agents.activation_agent import ActivationAgent, EmergencyStopError
from core.circuit_breaker import CircuitBreakerState


class TestActivationKillSwitch:
    """Q_030: Kill-switch <5s halt tested"""

    @pytest.fixture
    def activation_agent(self):
        """Create ActivationAgent instance with mocked dependencies"""
        agent = ActivationAgent(
            meta_api=AsyncMock(),
            google_api=AsyncMock(),
            circuit_breaker=Mock(),
            metrics_client=Mock()
        )
        return agent

    @pytest.mark.asyncio
    async def test_kill_switch_halt_under_5s(self, activation_agent):
        """
        ACCEPTANCE: Kill-switch halts all operations in <5 seconds
        METRIC: kill_switch_halt_latency_seconds < 5.0
        """
        # Simulate ongoing operations
        async def long_running_operation():
            for i in range(10):
                await asyncio.sleep(1)
                activation_agent._check_emergency_stop()  # Check kill-switch each iteration

        # Trigger kill-switch after 0.5s
        async def trigger_kill_switch():
            await asyncio.sleep(0.5)
            activation_agent.trigger_emergency_stop(reason="Test kill-switch")

        # Measure halt latency
        start_time = time.perf_counter()
        
        with pytest.raises(EmergencyStopError):
            await asyncio.gather(
                long_running_operation(),
                trigger_kill_switch()
            )
        
        halt_latency = time.perf_counter() - start_time
        
        # ACCEPTANCE: <5s halt
        assert halt_latency < 5.0, f"Kill-switch halt took {halt_latency:.2f}s, exceeds 5s limit"
        
        # Metric validation
        activation_agent.metrics_client.gauge.assert_called_with(
            'kill_switch_halt_latency_seconds',
            halt_latency
        )

    @pytest.mark.asyncio
    async def test_kill_switch_blocks_new_operations(self, activation_agent):
        """Verify kill-switch blocks new operations immediately after trigger"""
        activation_agent.trigger_emergency_stop(reason="Block new ops")
        
        with pytest.raises(EmergencyStopError) as exc_info:
            await activation_agent.push_budget_allocation({
                "campaign_id": "c123",
                "new_budget": 100000
            })
        
        assert "EMERGENCY STOP" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_kill_switch_propagates_to_all_apis(self, activation_agent):
        """Verify kill-switch stops all external API calls"""
        # Setup multiple pending API calls
        meta_call = activation_agent.meta_api.update_budget(campaign_id="c1", budget=50000)
        google_call = activation_agent.google_api.update_budget(campaign_id="c2", budget=30000)
        
        # Trigger kill-switch
        activation_agent.trigger_emergency_stop(reason="API halt test")
        
        # Verify all calls raise EmergencyStopError
        with pytest.raises(EmergencyStopError):
            await meta_call
        
        with pytest.raises(EmergencyStopError):
            await google_call

    @pytest.mark.asyncio
    async def test_circuit_breaker_integration_with_kill_switch(self, activation_agent):
        """Verify circuit breaker opens on kill-switch trigger"""
        activation_agent.trigger_emergency_stop(reason="Circuit breaker integration")
        
        # Circuit breaker should transition to OPEN
        assert activation_agent.circuit_breaker.state == CircuitBreakerState.OPEN
        
        # Metric validation
        activation_agent.metrics_client.increment.assert_called_with(
            'circuit_breaker_opened',
            tags=['reason:emergency_stop']
        )

    @pytest.mark.asyncio
    async def test_kill_switch_reset_mechanism(self, activation_agent):
        """Verify kill-switch can be reset after resolving emergency"""
        # Trigger kill-switch
        activation_agent.trigger_emergency_stop(reason="Test reset")
        assert activation_agent.is_emergency_stopped is True
        
        # Reset kill-switch
        activation_agent.reset_emergency_stop(authorized_by="admin@company.com")
        assert activation_agent.is_emergency_stopped is False
        
        # Verify operations can resume
        result = await activation_agent.push_budget_allocation({
            "campaign_id": "c123",
            "new_budget": 100000
        })
        assert result is not None

    @pytest.mark.asyncio
    async def test_kill_switch_audit_log_entry(self, activation_agent):
        """Verify kill-switch triggers are logged to audit trail"""
        activation_agent.trigger_emergency_stop(
            reason="Runaway campaign detected",
            triggered_by="playbook:creative_fatigue_v1"
        )
        
        # Verify audit log
        activation_agent.metrics_client.event.assert_called_with(
            'kill_switch_triggered',
            {
                'reason': 'Runaway campaign detected',
                'triggered_by': 'playbook:creative_fatigue_v1',
                'timestamp': pytest.approx(datetime.utcnow(), abs=timedelta(seconds=1))
            }
        )

    @pytest.mark.asyncio
    async def test_kill_switch_dry_run_probe(self, activation_agent):
        """
        DRY-RUN PROBE: Simulate kill-switch without actual API calls
        Validates latency measurement accuracy
        """
        # Mock internal operations
        with patch.object(activation_agent, '_execute_api_mutation', new=AsyncMock()):
            # Create simulated workload
            async def simulated_workload():
                for i in range(5):
                    await asyncio.sleep(0.1)
                    activation_agent._check_emergency_stop()
            
            # Measure dry-run latency
            start = time.perf_counter()
            
            async def dry_run_trigger():
                await asyncio.sleep(0.2)
                activation_agent.trigger_emergency_stop(reason="Dry-run test")
            
            with pytest.raises(EmergencyStopError):
                await asyncio.gather(simulated_workload(), dry_run_trigger())
            
            dry_run_latency = time.perf_counter() - start
            
            # Verify dry-run latency is reasonable (<1s)
            assert dry_run_latency < 1.0, f"Dry-run latency {dry_run_latency:.2f}s too high"
            
            # Verify no actual API calls were made
            activation_agent._execute_api_mutation.assert_not_called()
