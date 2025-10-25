"""PagerDuty Integration Tests - P0 Alert <2min SLA

Tests emergency alerting SLA enforcement for P0 incidents.

Components tested:
- AlertAgent.send_alert with severity P0
- PagerDuty API integration
- Alert delivery latency tracking
- Prometheus metrics (alert_latency_seconds)

Acceptance criteria (Q_024):
- P0 alert delivered to PagerDuty in <120 seconds
- Alert includes: incident_key, urgency=high, title, details, routing_key
- Metric alert_latency_seconds{severity="P0"} recorded
- Retry logic: 3 attempts with exponential backoff
- Failure fallback: escalate to backup channel (Slack)

Risk gates:
- Kill switch: ENABLE_PAGERDUTY_ALERTS (default true)
- Timeout: 10s per API call
- Circuit breaker: open after 5 consecutive failures
- Idempotent: same incident_key deduplicated by PagerDuty
"""
import pytest
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta
import asyncio
import time

from agents.alert_agent import AlertAgent
from core.contracts import Alert, AlertSeverity, AlertChannel


class TestPagerDutySLA:
    """Test P0 alert delivery SLA"""

    @pytest.fixture
    def alert_agent(self):
        """Create AlertAgent with mock PagerDuty client"""
        agent = AlertAgent()
        agent.pagerduty_client = AsyncMock()
        agent.metrics_client = Mock()
        return agent

    @pytest.fixture
    def p0_alert(self):
        """Create P0 severity alert"""
        return Alert(
            alert_id="al_001",
            severity=AlertSeverity.P0,
            title="MMM Agent Down",
            message="Critical: MMM prediction service unresponsive >5min",
            source="health_check",
            channel=AlertChannel.PAGERDUTY,
            created_at=datetime.utcnow(),
            tags=["critical", "mmm", "prediction"],
            metadata={"component": "C04_MMMAgent", "health_status": "down"}
        )

    @pytest.mark.asyncio
    async def test_p0_alert_under_2min_sla(self, alert_agent, p0_alert):
        """P0 alert must reach PagerDuty in <120 seconds"""
        # Mock PagerDuty response
        alert_agent.pagerduty_client.trigger_incident.return_value = {
            "status": "success",
            "incident_key": "INC-001",
            "message": "Event processed"
        }

        start_time = time.time()
        
        result = await alert_agent.send_alert(p0_alert)
        
        latency = time.time() - start_time
        
        # Assert SLA met
        assert latency < 120.0, f"P0 alert latency {latency:.2f}s exceeds 2min SLA"
        assert result.success is True
        assert result.latency_seconds < 120.0
        
        # Verify PagerDuty API called with correct params
        alert_agent.pagerduty_client.trigger_incident.assert_called_once()
        call_args = alert_agent.pagerduty_client.trigger_incident.call_args
        
        assert call_args.kwargs['routing_key'] is not None
        assert call_args.kwargs['event_action'] == 'trigger'
        assert call_args.kwargs['payload']['severity'] == 'critical'
        assert call_args.kwargs['payload']['summary'] == p0_alert.title
        assert 'dedup_key' in call_args.kwargs
        
        # Verify metric recorded
        alert_agent.metrics_client.record_histogram.assert_called_with(
            'alert_latency_seconds',
            latency,
            labels={'severity': 'P0', 'channel': 'pagerduty', 'status': 'success'}
        )

    @pytest.mark.asyncio
    async def test_p0_alert_includes_required_fields(self, alert_agent, p0_alert):
        """P0 alert payload must include all required PagerDuty fields"""
        alert_agent.pagerduty_client.trigger_incident.return_value = {
            "status": "success",
            "incident_key": "INC-002"
        }

        await alert_agent.send_alert(p0_alert)
        
        call_args = alert_agent.pagerduty_client.trigger_incident.call_args.kwargs
        payload = call_args['payload']
        
        # Required fields
        assert 'summary' in payload
        assert 'source' in payload
        assert 'severity' in payload
        assert payload['severity'] == 'critical'  # P0 maps to critical
        assert 'timestamp' in payload
        assert 'custom_details' in payload
        
        # Custom details
        details = payload['custom_details']
        assert 'component' in details
        assert 'message' in details
        assert 'tags' in details
        assert details['component'] == "C04_MMMAgent"

    @pytest.mark.asyncio
    async def test_p0_alert_retry_on_timeout(self, alert_agent, p0_alert):
        """P0 alert retries up to 3 times on timeout"""
        # First 2 calls timeout, 3rd succeeds
        alert_agent.pagerduty_client.trigger_incident.side_effect = [
            asyncio.TimeoutError(),
            asyncio.TimeoutError(),
            {"status": "success", "incident_key": "INC-003"}
        ]

        result = await alert_agent.send_alert(p0_alert)
        
        assert result.success is True
        assert result.retry_count == 2
        assert alert_agent.pagerduty_client.trigger_incident.call_count == 3
        
        # Verify metric includes retry count
        metric_calls = alert_agent.metrics_client.record_histogram.call_args_list
        assert any('retry_count' in str(call) for call in metric_calls)

    @pytest.mark.asyncio
    async def test_p0_alert_fallback_to_slack_on_failure(self, alert_agent, p0_alert):
        """P0 alert escalates to Slack if PagerDuty fails after retries"""
        # All PagerDuty attempts fail
        alert_agent.pagerduty_client.trigger_incident.side_effect = [
            asyncio.TimeoutError(),
            asyncio.TimeoutError(),
            asyncio.TimeoutError()
        ]
        
        # Mock Slack fallback
        alert_agent.slack_client = AsyncMock()
        alert_agent.slack_client.send_message.return_value = {"ok": True}

        result = await alert_agent.send_alert(p0_alert)
        
        # PagerDuty failed but Slack succeeded
        assert result.primary_channel_success is False
        assert result.fallback_channel_success is True
        assert result.fallback_channel == "slack"
        
        # Verify Slack called with escalation message
        alert_agent.slack_client.send_message.assert_called_once()
        slack_msg = alert_agent.slack_client.send_message.call_args.kwargs['text']
        assert "ESCALATION" in slack_msg
        assert "PagerDuty failed" in slack_msg
        assert p0_alert.title in slack_msg

    @pytest.mark.asyncio
    async def test_p0_alert_idempotent_dedup_key(self, alert_agent, p0_alert):
        """P0 alert uses stable dedup_key for idempotency"""
        alert_agent.pagerduty_client.trigger_incident.return_value = {
            "status": "success",
            "incident_key": "INC-004"
        }

        # Send same alert twice
        result1 = await alert_agent.send_alert(p0_alert)
        result2 = await alert_agent.send_alert(p0_alert)
        
        # Both calls should use same dedup_key
        call1_dedup = alert_agent.pagerduty_client.trigger_incident.call_args_list[0].kwargs['dedup_key']
        call2_dedup = alert_agent.pagerduty_client.trigger_incident.call_args_list[1].kwargs['dedup_key']
        
        assert call1_dedup == call2_dedup
        assert call1_dedup == f"al_001"  # Based on alert_id

    @pytest.mark.asyncio
    async def test_p0_alert_circuit_breaker_opens(self, alert_agent, p0_alert):
        """Circuit breaker opens after 5 consecutive P0 failures"""
        # Simulate 5 consecutive failures
        alert_agent.pagerduty_client.trigger_incident.side_effect = [
            Exception("503 Service Unavailable")
        ] * 5

        for i in range(5):
            result = await alert_agent.send_alert(p0_alert)
            assert result.success is False

        # 6th attempt should be circuit-breaker blocked
        result = await alert_agent.send_alert(p0_alert)
        
        assert result.success is False
        assert result.circuit_breaker_open is True
        assert "circuit breaker open" in result.error_message.lower()
        
        # PagerDuty should NOT be called on 6th attempt
        assert alert_agent.pagerduty_client.trigger_incident.call_count == 5

    @pytest.mark.asyncio
    async def test_p0_alert_kill_switch_disabled(self, alert_agent, p0_alert):
        """P0 alert respects ENABLE_PAGERDUTY_ALERTS kill switch"""
        # Disable PagerDuty alerts
        with patch.dict('os.environ', {'ENABLE_PAGERDUTY_ALERTS': 'false'}):
            alert_agent_disabled = AlertAgent()
            alert_agent_disabled.slack_client = AsyncMock()
            alert_agent_disabled.slack_client.send_message.return_value = {"ok": True}
            
            result = await alert_agent_disabled.send_alert(p0_alert)
            
            # Should skip PagerDuty and go to fallback only
            assert result.primary_channel_skipped is True
            assert result.skip_reason == "kill_switch_disabled"
            assert result.fallback_channel_success is True

    @pytest.mark.asyncio
    async def test_p0_alert_latency_metric_recorded(self, alert_agent, p0_alert):
        """P0 alert latency metric is always recorded"""
        alert_agent.pagerduty_client.trigger_incident.return_value = {
            "status": "success",
            "incident_key": "INC-005"
        }

        await alert_agent.send_alert(p0_alert)
        
        # Verify histogram metric
        alert_agent.metrics_client.record_histogram.assert_called()
        call_args = alert_agent.metrics_client.record_histogram.call_args
        
        assert call_args.args[0] == 'alert_latency_seconds'
        assert isinstance(call_args.args[1], float)
        assert call_args.args[2]['severity'] == 'P0'
        assert call_args.args[2]['channel'] == 'pagerduty'

    @pytest.mark.asyncio
    async def test_p0_alert_10s_timeout_enforced(self, alert_agent, p0_alert):
        """P0 alert enforces 10s timeout per API call"""
        # Simulate slow PagerDuty response
        async def slow_response(*args, **kwargs):
            await asyncio.sleep(11)  # >10s
            return {"status": "success"}
        
        alert_agent.pagerduty_client.trigger_incident.side_effect = slow_response

        start = time.time()
        result = await alert_agent.send_alert(p0_alert)
        elapsed = time.time() - start
        
        # Should timeout before 11s
        assert elapsed < 11.0
        assert result.success is False
        assert "timeout" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_p0_alert_multiple_concurrent(self, alert_agent):
        """Multiple concurrent P0 alerts all meet SLA"""
        alerts = [
            Alert(
                alert_id=f"al_{i:03d}",
                severity=AlertSeverity.P0,
                title=f"Critical Alert {i}",
                message=f"Message {i}",
                source="test",
                channel=AlertChannel.PAGERDUTY,
                created_at=datetime.utcnow()
            )
            for i in range(10)
        ]
        
        alert_agent.pagerduty_client.trigger_incident.return_value = {
            "status": "success",
            "incident_key": "INC-006"
        }

        start = time.time()
        results = await asyncio.gather(*[
            alert_agent.send_alert(alert) for alert in alerts
        ])
        total_elapsed = time.time() - start
        
        # All alerts succeed
        assert all(r.success for r in results)
        
        # Each alert meets individual SLA
        assert all(r.latency_seconds < 120.0 for r in results)
        
        # Total time reasonable (concurrent, not serial)
        assert total_elapsed < 10.0  # Much less than 10 * 120s


class TestAlertPriority:
    """Test alert priority and routing"""

    @pytest.mark.asyncio
    async def test_p1_alert_uses_lower_urgency(self):
        """P1 alerts use 'high' urgency (not 'critical')"""
        agent = AlertAgent()
        agent.pagerduty_client = AsyncMock()
        agent.pagerduty_client.trigger_incident.return_value = {
            "status": "success",
            "incident_key": "INC-007"
        }

        p1_alert = Alert(
            alert_id="al_010",
            severity=AlertSeverity.P1,
            title="Data Quality Warning",
            message="Freshness >4h but <6h",
            source="data_quality_agent",
            channel=AlertChannel.PAGERDUTY,
            created_at=datetime.utcnow()
        )

        await agent.send_alert(p1_alert)
        
        payload = agent.pagerduty_client.trigger_incident.call_args.kwargs['payload']
        assert payload['severity'] == 'error'  # P1 maps to error, not critical

    @pytest.mark.asyncio
    async def test_p2_alert_uses_warning_severity(self):
        """P2 alerts use 'warning' severity"""
        agent = AlertAgent()
        agent.pagerduty_client = AsyncMock()
        agent.pagerduty_client.trigger_incident.return_value = {
            "status": "success",
            "incident_key": "INC-008"
        }

        p2_alert = Alert(
            alert_id="al_011",
            severity=AlertSeverity.P2,
            title="Budget Pacing Info",
            message="Campaign pacing at 85%",
            source="pacing_agent",
            channel=AlertChannel.PAGERDUTY,
            created_at=datetime.utcnow()
        )

        await agent.send_alert(p2_alert)
        
        payload = agent.pagerduty_client.trigger_incident.call_args.kwargs['payload']
        assert payload['severity'] == 'warning'


class TestAlertMetrics:
    """Test alert metrics and observability"""

    @pytest.mark.asyncio
    async def test_alert_counter_incremented(self):
        """Alert counter metric incremented on send"""
        agent = AlertAgent()
        agent.pagerduty_client = AsyncMock()
        agent.pagerduty_client.trigger_incident.return_value = {
            "status": "success",
            "incident_key": "INC-009"
        }
        agent.metrics_client = Mock()

        alert = Alert(
            alert_id="al_012",
            severity=AlertSeverity.P0,
            title="Test",
            message="Test",
            source="test",
            channel=AlertChannel.PAGERDUTY,
            created_at=datetime.utcnow()
        )

        await agent.send_alert(alert)
        
        # Verify counter increment
        agent.metrics_client.increment_counter.assert_called_with(
            'alerts_sent_total',
            labels={'severity': 'P0', 'channel': 'pagerduty', 'status': 'success'}
        )
