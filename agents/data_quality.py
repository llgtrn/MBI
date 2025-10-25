"""
Data Quality Agent with freshness SLA enforcement
Implements Q_004 + A_012 + Q_027: >6h rejection with metrics for MTA integrity
"""
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class DataFreshnessError(Exception):
    """Raised when data exceeds freshness SLA"""
    pass


class DataQualityAgent:
    """
    Data quality validation with freshness SLA enforcement
    
    Features:
    - Rejects data with export_time >6h (configurable)
    - Special handling for GA4/MTA data integrity
    - Kill switch for freshness checks (ENABLE_FRESHNESS_CHECK)
    - Backpressure monitoring for rejection rates
    - Prometheus metrics for violations
    """
    
    def __init__(
        self,
        freshness_sla_hours: int = 6,
        enable_freshness_check: bool = True,
        rejection_backpressure_threshold: int = 100
    ):
        self.freshness_sla_hours = freshness_sla_hours
        self.enable_freshness_check = enable_freshness_check
        self.rejection_backpressure_threshold = rejection_backpressure_threshold
        
        # Metrics
        self._metrics = {
            'data_freshness_violations_total': 0,
            'rejected_records_total': 0,
            'accepted_records_total': 0,
            'freshness_checks_disabled_total': 0,
            'backpressure_alerts_total': 0,
            'max_staleness_hours': 0.0
        }
    
    async def validate_freshness(self, data: Dict[str, Any]) -> bool:
        """
        Validate data freshness against SLA
        
        Args:
            data: Data payload with export_time field
        
        Returns:
            True if data is fresh (within SLA)
        
        Raises:
            DataFreshnessError: If data exceeds freshness SLA
        """
        # Kill switch check
        if not self.enable_freshness_check:
            logger.warning("Freshness checks disabled by kill switch")
            self._metrics['freshness_checks_disabled_total'] += 1
            return True
        
        try:
            # Extract export timestamp
            export_time_str = data.get('export_time')
            if not export_time_str:
                logger.warning("No export_time field in data, skipping freshness check")
                return True
            
            export_time = datetime.fromisoformat(export_time_str)
            current_time = datetime.utcnow()
            
            # Compute staleness
            staleness = current_time - export_time
            staleness_hours = staleness.total_seconds() / 3600
            
            # Update max staleness metric
            if staleness_hours > self._metrics['max_staleness_hours']:
                self._metrics['max_staleness_hours'] = staleness_hours
            
            # Check SLA
            if staleness_hours > self.freshness_sla_hours:
                # Violation detected
                source = data.get('source', 'unknown')
                record_count = len(data.get('records', []))
                
                error_msg = (
                    f"Data freshness SLA violation: export_time {staleness_hours:.2f}h old "
                    f"(SLA: {self.freshness_sla_hours}h) from source '{source}'"
                )
                
                # Special handling for GA4 (MTA integrity)
                if 'ga4' in source.lower():
                    error_msg += (
                        " - GA4 stale data can corrupt MTA attribution paths. "
                        "Ensure BigQuery export schedule is functioning."
                    )
                
                # Update metrics
                self._metrics['data_freshness_violations_total'] += 1
                self._metrics['rejected_records_total'] += record_count
                
                # Backpressure monitoring
                if self._metrics['rejected_records_total'] > self.rejection_backpressure_threshold:
                    self._metrics['backpressure_alerts_total'] += 1
                    logger.critical(
                        f"Rejection backpressure threshold exceeded: "
                        f"{self._metrics['rejected_records_total']} records rejected"
                    )
                
                logger.error(error_msg)
                raise DataFreshnessError(error_msg)
            
            # Data is fresh
            record_count = len(data.get('records', []))
            self._metrics['accepted_records_total'] += record_count
            
            logger.debug(
                f"Data freshness OK: {staleness_hours:.2f}h old "
                f"(SLA: {self.freshness_sla_hours}h)"
            )
            
            return True
        
        except DataFreshnessError:
            raise  # Re-raise freshness errors
        except Exception as e:
            logger.error(f"Freshness validation failed: {e}")
            # Fail open: accept data if validation fails
            return True
    
    async def validate_batch(self, batch: list[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validate multiple data payloads in batch
        
        Args:
            batch: List of data payloads
        
        Returns:
            Dict with validation results
        """
        results = {
            'accepted': [],
            'rejected': [],
            'total': len(batch)
        }
        
        for data in batch:
            try:
                await self.validate_freshness(data)
                results['accepted'].append(data)
            except DataFreshnessError as e:
                results['rejected'].append({
                    'data': data,
                    'error': str(e)
                })
        
        return results
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get current metrics for monitoring
        
        Returns:
            Dict with metric counters
        """
        return self._metrics.copy()
    
    def reset_metrics(self):
        """Reset metrics counters (for testing)"""
        for key in self._metrics:
            if isinstance(self._metrics[key], (int, float)):
                self._metrics[key] = 0 if isinstance(self._metrics[key], int) else 0.0
    
    def get_freshness_stats(self) -> Dict[str, Any]:
        """
        Get freshness statistics summary
        
        Returns:
            Dict with stats
        """
        total_records = (
            self._metrics['accepted_records_total'] +
            self._metrics['rejected_records_total']
        )
        
        rejection_rate = (
            self._metrics['rejected_records_total'] / total_records
            if total_records > 0 else 0.0
        )
        
        return {
            'freshness_sla_hours': self.freshness_sla_hours,
            'total_records_processed': total_records,
            'accepted_count': self._metrics['accepted_records_total'],
            'rejected_count': self._metrics['rejected_records_total'],
            'rejection_rate': rejection_rate,
            'violations': self._metrics['data_freshness_violations_total'],
            'max_staleness_hours': self._metrics['max_staleness_hours'],
            'backpressure_alerts': self._metrics['backpressure_alerts_total']
        }
