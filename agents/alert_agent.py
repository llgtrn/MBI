"""AlertAgent - PagerDuty Integration with <2min P0 SLA

Handles emergency alerting to PagerDuty with strict SLA enforcement.

Features:
- P0 alerts delivered in <120 seconds (acceptance: Q_024)
- Automatic retries with exponential backoff (3 attempts)
- Circuit breaker protection (opens after 5 consecutive failures)
- Fallback to Slack on PagerDuty failure
- Idempotent deduplication via stable incident keys
- Prometheus metrics: alert_latency_seconds, alerts_sent_total
- Kill switch: ENABLE_PAGERDUTY_ALERTS (default true)
- Per-call timeout: 10s

Risk gates:
- Timeout enforcement prevents infinite hangs
- Circuit breaker prevents cascade failures
- Fallback channel ensures P0 alerts always delivered
- Idempotent dedup prevents alert storms
- Metrics visibility into SLA compliance

Rollback:
- Set ENABLE_PAGERDUTY_ALERTS=false to disable
- Alerts route to Slack fallback only
- No PagerDuty API calls made
"""
import os
import asyncio
import time
import hashlib
from typing import Optional, Dict, Any
from datetime import datetime
from enum import Enum

from core.contracts import Alert, AlertSeverity, AlertChannel
from core.metrics import MetricsClient
from infrastructure.pagerduty_client import PagerDutyClient
from infrastructure.slack_client import SlackClient


class CircuitBreakerState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"      # Blocking calls after threshold failures
    HALF_OPEN = "half_open"  # Testing recovery


class AlertResult:
    """Result of alert send attempt"""
    def __init__(
        self,
        success: bool,
        latency_seconds: float,
        retry_count: int = 0,
        primary_channel_success: Optional[bool] = None,
        fallback_channel_success: Optional[bool] = None,
        fallback_channel: Optional[str] = None,
        primary_channel_skipped: bool = False,
        skip_reason: Optional[str] = None,
        circuit_breaker_open: bool = False,
        error_message: Optional[str] = None
    ):
        self.success = success
        self.latency_seconds = latency_seconds
        self.retry_count = retry_count
        self.primary_channel_success = primary_channel_success
        self.fallback_channel_success = fallback_channel_success
        self.fallback_channel = fallback_channel
        self.primary_channel_skipped = primary_channel_skipped
        self.skip_reason = skip_reason
        self.circuit_breaker_open = circuit_breaker_open
        self.error_message = error_message


class CircuitBreaker:
    """Circuit breaker for PagerDuty API calls"""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.state = CircuitBreakerState.CLOSED
        self.last_failure_time: Optional[float] = None
    
    def record_success(self):
        """Record successful call"""
        self.failure_count = 0
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.state = CircuitBreakerState.CLOSED
    
    def record_failure(self):
        """Record failed call"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitBreakerState.OPEN
    
    def is_open(self) -> bool:
        """Check if circuit breaker is open"""
        if self.state == CircuitBreakerState.CLOSED:
            return False
        
        if self.state == CircuitBreakerState.OPEN:
            # Check if recovery timeout elapsed
            if self.last_failure_time and \
               (time.time() - self.last_failure_time) > self.recovery_timeout:
                self.state = CircuitBreakerState.HALF_OPEN
                return False
            return True
        
        # HALF_OPEN: allow one test call
        return False


class AlertAgent:
    """Alert agent with PagerDuty integration"""
    
    def __init__(self):
        self.pagerduty_client = PagerDutyClient()
        self.slack_client = SlackClient()
        self.metrics_client = MetricsClient()
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=60
        )
        
        # Configuration
        self.enable_pagerduty = os.getenv(
            'ENABLE_PAGERDUTY_ALERTS',
            'true'
        ).lower() == 'true'
        self.max_retries = 3
        self.timeout_seconds = 10
        self.retry_delays = [1, 2, 4]  # Exponential backoff
    
    def _get_pagerduty_severity(self, severity: AlertSeverity) -> str:
        """Map internal severity to PagerDuty severity"""
        severity_map = {
            AlertSeverity.P0: 'critical',
            AlertSeverity.P1: 'error',
            AlertSeverity.P2: 'warning',
            AlertSeverity.P3: 'info'
        }
        return severity_map.get(severity, 'info')
    
    def _generate_dedup_key(self, alert: Alert) -> str:
        """Generate stable deduplication key for idempotency"""
        # Use alert_id as dedup key (stable across retries)
        return alert.alert_id
    
    async def _send_to_pagerduty(
        self,
        alert: Alert,
        attempt: int = 0
    ) -> Dict[str, Any]:
        """Send alert to PagerDuty with timeout"""
        routing_key = os.getenv('PAGERDUTY_ROUTING_KEY')
        
        payload = {
            'summary': alert.title,
            'source': alert.source,
            'severity': self._get_pagerduty_severity(alert.severity),
            'timestamp': alert.created_at.isoformat(),
            'custom_details': {
                'message': alert.message,
                'component': alert.metadata.get('component', 'unknown'),
                'tags': alert.tags,
                'alert_id': alert.alert_id
            }
        }
        
        # Apply timeout
        try:
            result = await asyncio.wait_for(
                self.pagerduty_client.trigger_incident(
                    routing_key=routing_key,
                    event_action='trigger',
                    dedup_key=self._generate_dedup_key(alert),
                    payload=payload
                ),
                timeout=self.timeout_seconds
            )
            return result
        except asyncio.TimeoutError:
            raise asyncio.TimeoutError(
                f"PagerDuty API call exceeded {self.timeout_seconds}s timeout"
            )
    
    async def _send_to_slack_fallback(self, alert: Alert, reason: str):
        """Send alert to Slack as fallback"""
        message = f"""🚨 **ESCALATION: PagerDuty Failed**

**Alert:** {alert.title}
**Severity:** {alert.severity.value}
**Reason:** {reason}
**Message:** {alert.message}
**Component:** {alert.metadata.get('component', 'unknown')}
**Time:** {alert.created_at.isoformat()}

PagerDuty delivery failed. Manual intervention required.
"""
        
        await self.slack_client.send_message(
            channel='#incidents',
            text=message
        )
    
    async def send_alert(self, alert: Alert) -> AlertResult:
        """
        Send alert with SLA enforcement
        
        P0 alerts must be delivered to PagerDuty in <120 seconds.
        Retries up to 3 times on failure with exponential backoff.
        Falls back to Slack if PagerDuty unavailable.
        
        Args:
            alert: Alert object to send
        
        Returns:
            AlertResult with success status and metrics
        """
        start_time = time.time()
        
        # Check kill switch
        if not self.enable_pagerduty:
            # Skip PagerDuty, go directly to fallback
            await self._send_to_slack_fallback(
                alert,
                "PagerDuty disabled via kill switch"
            )
            latency = time.time() - start_time
            
            self._record_metrics(
                alert,
                latency,
                success=True,
                channel='slack',
                retry_count=0
            )
            
            return AlertResult(
                success=True,
                latency_seconds=latency,
                primary_channel_skipped=True,
                skip_reason="kill_switch_disabled",
                fallback_channel_success=True,
                fallback_channel="slack"
            )
        
        # Check circuit breaker
        if self.circuit_breaker.is_open():
            await self._send_to_slack_fallback(
                alert,
                "Circuit breaker open (too many PagerDuty failures)"
            )
            latency = time.time() - start_time
            
            return AlertResult(
                success=False,
                latency_seconds=latency,
                circuit_breaker_open=True,
                error_message="Circuit breaker open - PagerDuty unavailable",
                fallback_channel_success=True,
                fallback_channel="slack"
            )
        
        # Retry loop
        last_error = None
        for attempt in range(self.max_retries):
            try:
                result = await self._send_to_pagerduty(alert, attempt)
                
                # Success
                latency = time.time() - start_time
                self.circuit_breaker.record_success()
                
                self._record_metrics(
                    alert,
                    latency,
                    success=True,
                    channel='pagerduty',
                    retry_count=attempt
                )
                
                return AlertResult(
                    success=True,
                    latency_seconds=latency,
                    retry_count=attempt,
                    primary_channel_success=True
                )
            
            except (asyncio.TimeoutError, Exception) as e:
                last_error = str(e)
                
                # Retry with backoff (except on last attempt)
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delays[attempt])
                    continue
        
        # All retries failed
        self.circuit_breaker.record_failure()
        
        # Fallback to Slack
        try:
            await self._send_to_slack_fallback(alert, last_error)
            fallback_success = True
        except Exception:
            fallback_success = False
        
        latency = time.time() - start_time
        
        self._record_metrics(
            alert,
            latency,
            success=False,
            channel='pagerduty',
            retry_count=self.max_retries
        )
        
        return AlertResult(
            success=fallback_success,  # Overall success if fallback worked
            latency_seconds=latency,
            retry_count=self.max_retries,
            primary_channel_success=False,
            fallback_channel_success=fallback_success,
            fallback_channel="slack",
            error_message=last_error
        )
    
    def _record_metrics(
        self,
        alert: Alert,
        latency: float,
        success: bool,
        channel: str,
        retry_count: int
    ):
        """Record alert metrics"""
        # Histogram: alert_latency_seconds
        self.metrics_client.record_histogram(
            'alert_latency_seconds',
            latency,
            labels={
                'severity': alert.severity.value,
                'channel': channel,
                'status': 'success' if success else 'failure'
            }
        )
        
        # Counter: alerts_sent_total
        self.metrics_client.increment_counter(
            'alerts_sent_total',
            labels={
                'severity': alert.severity.value,
                'channel': channel,
                'status': 'success' if success else 'failure'
            }
        )
        
        # Counter: alert_retries_total (if retried)
        if retry_count > 0:
            self.metrics_client.increment_counter(
                'alert_retries_total',
                value=retry_count,
                labels={
                    'severity': alert.severity.value,
                    'channel': channel
                }
            )
