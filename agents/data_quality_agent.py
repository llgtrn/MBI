"""
Data Quality Agent - Monitors data freshness and triggers auto-pause for MMM/MTA
when data staleness exceeds threshold.

Implements:
- Q_004: Data >6h auto-pause MMM/MTA
- A_016: C02_DataQuality risk gate with circuit breaker
"""

from datetime import datetime, timedelta
from typing import Dict, Optional
import logging
from collections import deque

logger = logging.getLogger(__name__)

class DataQualityAgent:
    """
    Monitors data quality and triggers auto-pause for downstream agents
    when data staleness exceeds threshold.
    """
    
    def __init__(
        self,
        feature_flag_data_age_pause: bool = True,
        staleness_threshold_hours: float = 6.0,
        circuit_breaker_max_triggers: int = 3,
        circuit_breaker_window_hours: int = 1
    ):
        self.feature_flag = feature_flag_data_age_pause
        self.threshold_hours = staleness_threshold_hours
        self.circuit_breaker_max = circuit_breaker_max_triggers
        self.circuit_breaker_window = timedelta(hours=circuit_breaker_window_hours)
        
        # Track pause triggers for circuit breaker
        self.pause_history = deque(maxlen=100)
        
        # Idempotency tracking
        self.processed_checks = set()
    
    def check_data_age_and_pause(self, check_id: Optional[str] = None) -> Dict:
        """
        Check data age and determine if MMM/MTA should be paused.
        
        Args:
            check_id: Optional idempotency key to prevent duplicate actions
            
        Returns:
            Dict with pause decisions, metrics, and escalation flags
        """
        # Generate check_id if not provided
        if check_id is None:
            check_id = f"data_quality_check_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        # Idempotency check
        if check_id in self.processed_checks:
            logger.info(f"Check {check_id} already processed, skipping")
            return {
                'pause_mmm': False,
                'pause_mta': False,
                'reason': 'duplicate_check_id_skipped',
                'check_id': check_id,
                'metrics_emitted': []
            }
        
        # Kill switch check
        if not self.feature_flag:
            logger.info("Data age pause feature flag disabled")
            return {
                'pause_mmm': False,
                'pause_mta': False,
                'reason': 'feature_flag_disabled',
                'check_id': check_id,
                'metrics_emitted': []
            }
        
        # Get latest data timestamp
        latest_data = self.get_latest_data()
        now = self.get_current_time()
        
        data_age = now - latest_data.last_updated
        data_age_hours = data_age.total_seconds() / 3600
        
        # Determine if pause is needed
        should_pause = data_age_hours > self.threshold_hours
        
        result = {
            'pause_mmm': should_pause,
            'pause_mta': should_pause,
            'data_age_hours': data_age_hours,
            'threshold_hours': self.threshold_hours,
            'check_id': check_id,
            'timestamp': now.isoformat(),
            'metrics_emitted': []
        }
        
        if should_pause:
            result['reason'] = 'data_staleness_threshold_exceeded'
            
            # Record pause trigger
            self.pause_history.append(now)
            
            # Emit metrics
            self._emit_metric('data_quality.stale_data_paused', 1)
            self._emit_metric('data_quality.data_age_hours', data_age_hours)
            result['metrics_emitted'].extend([
                'data_quality.stale_data_paused',
                'data_quality.data_age_hours'
            ])
            
            # Circuit breaker check
            recent_triggers = [
                t for t in self.pause_history
                if now - t < self.circuit_breaker_window
            ]
            
            if len(recent_triggers) > self.circuit_breaker_max:
                result['circuit_breaker_triggered'] = True
                result['escalate_to_human'] = True
                self._emit_metric('data_quality.circuit_breaker_tripped', 1)
                result['metrics_emitted'].append('data_quality.circuit_breaker_tripped')
                
                logger.error(
                    f"Circuit breaker triggered: {len(recent_triggers)} pauses "
                    f"in {self.circuit_breaker_window.total_seconds()/3600}h window"
                )
            else:
                result['circuit_breaker_triggered'] = False
                result['escalate_to_human'] = False
            
            logger.warning(
                f"Data staleness detected: {data_age_hours:.1f}h > {self.threshold_hours}h. "
                f"Pausing MMM/MTA. Check: {check_id}"
            )
        else:
            result['reason'] = 'data_fresh'
            logger.info(f"Data is fresh: {data_age_hours:.1f}h ≤ {self.threshold_hours}h")
        
        # Mark check as processed
        self.processed_checks.add(check_id)
        
        return result
    
    def get_latest_data(self):
        """
        Get latest data timestamp from feature store.
        Override in production with actual implementation.
        """
        # Placeholder - will be implemented with real feature store integration
        class MockData:
            def __init__(self):
                self.last_updated = datetime.utcnow() - timedelta(hours=5)
                self.source = "ad_spend_daily"
        
        return MockData()
    
    def get_current_time(self) -> datetime:
        """Get current time. Mockable for testing."""
        return datetime.utcnow()
    
    def _emit_metric(self, metric_name: str, value: float):
        """Emit metric to monitoring system."""
        # Placeholder - integrate with Prometheus/CloudWatch
        logger.info(f"METRIC: {metric_name} = {value}")
