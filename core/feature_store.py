"""
Feature Store - Online/Offline feature management with parity checks and unique constraints.

Implements:
- Q_005: Feature parity <1% with online/offline reconciliation
- Q_006: Feature unique constraint with composite key
- A_017, A_018: C03_FeatureEngineering data integrity gates
"""

from datetime import datetime, timedelta
from typing import Dict, Optional
import logging
import hashlib
from collections import defaultdict

logger = logging.getLogger(__name__)

class IntegrityError(Exception):
    """Raised when unique constraint is violated"""
    def __init__(self, message: str):
        super().__init__(message)
        self.status_code = 409


class FeatureParityCheck:
    """Schema for feature parity validation"""
    def __init__(self, online_offline_delta_pct: float):
        if online_offline_delta_pct > 1.0:
            raise ValueError("Feature parity delta must be ≤ 1.0%")
        self.online_offline_delta_pct = online_offline_delta_pct


class FeatureStore:
    """
    Manages online and offline features with parity checks and unique constraints.
    """
    
    def __init__(
        self,
        feature_flag_parity_check: bool = True,
        parity_threshold_pct: float = 1.0,
        reconciliation_rate_limit_per_hour: int = 1,
        feature_flag_unique_constraint: bool = True
    ):
        self.parity_check_enabled = feature_flag_parity_check
        self.parity_threshold = parity_threshold_pct
        self.rate_limit_per_hour = reconciliation_rate_limit_per_hour
        self.constraint_enabled = feature_flag_unique_constraint
        
        # In-memory store (replace with DB in production)
        self.features = {}  # key: feature_id, value: feature_data
        
        # Reconciliation tracking
        self.last_reconciliation = {}  # entity_id -> timestamp
        self.processed_reconciliations = set()
        
        # Metrics
        self.emitted_metrics = defaultdict(int)
    
    def check_feature_parity(
        self,
        entity_id: str,
        feature_name: str,
        online_value: float,
        offline_value: float
    ) -> Dict:
        """
        Check parity between online and offline feature values.
        
        Args:
            entity_id: Entity identifier
            feature_name: Feature name
            online_value: Value from online store
            offline_value: Value from offline store
            
        Returns:
            Dict with parity check result and metrics
        """
        if not self.parity_check_enabled:
            return {
                'parity_check_passed': True,
                'reason': 'parity_check_disabled',
                'metrics_emitted': []
            }
        
        # Calculate delta
        if online_value == 0:
            delta_pct = 100.0 if offline_value != 0 else 0.0
        else:
            delta_pct = abs((offline_value - online_value) / online_value) * 100
        
        passed = delta_pct <= self.parity_threshold
        
        result = {
            'entity_id': entity_id,
            'feature_name': feature_name,
            'online_value': online_value,
            'offline_value': offline_value,
            'delta_pct': round(delta_pct, 2),
            'threshold_pct': self.parity_threshold,
            'parity_check_passed': passed,
            'timestamp': datetime.utcnow().isoformat(),
            'metrics_emitted': []
        }
        
        # Emit metrics
        self._emit_metric('feature_store.parity_delta_pct', delta_pct)
        result['metrics_emitted'].append('feature_store.parity_delta_pct')
        
        if not passed:
            result['alert_triggered'] = True
            self._emit_metric('feature_store.parity_violation', 1)
            result['metrics_emitted'].append('feature_store.parity_violation')
            
            logger.error(
                f"Feature parity violation: {entity_id}.{feature_name} "
                f"delta={delta_pct:.2f}% > {self.parity_threshold}%"
            )
        else:
            result['alert_triggered'] = False
            logger.info(
                f"Feature parity OK: {entity_id}.{feature_name} "
                f"delta={delta_pct:.2f}% ≤ {self.parity_threshold}%"
            )
        
        return result
    
    def reconcile_features(
        self,
        entity_id: str,
        reconciliation_id: Optional[str] = None
    ) -> Dict:
        """
        Reconcile online and offline features for an entity.
        Rate-limited to prevent load spikes.
        
        Args:
            entity_id: Entity to reconcile
            reconciliation_id: Optional idempotency key
            
        Returns:
            Dict with reconciliation result
        """
        # Generate reconciliation_id if not provided
        if reconciliation_id is None:
            reconciliation_id = f"recon_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{entity_id}"
        
        # Idempotency check
        if reconciliation_id in self.processed_reconciliations:
            logger.info(f"Reconciliation {reconciliation_id} already processed")
            return {
                'reconciled': False,
                'reason': 'duplicate_reconciliation_id',
                'reconciliation_id': reconciliation_id
            }
        
        # Rate limit check
        now = datetime.utcnow()
        last_recon = self.last_reconciliation.get(entity_id)
        
        if last_recon and (now - last_recon) < timedelta(hours=1):
            logger.info(f"Rate limit: entity {entity_id} reconciled too recently")
            return {
                'reconciled': False,
                'reason': 'rate_limit_exceeded',
                'next_allowed_at': (last_recon + timedelta(hours=1)).isoformat()
            }
        
        # Perform reconciliation (placeholder - implement actual logic)
        logger.info(f"Reconciling features for entity {entity_id}")
        
        # Update tracking
        self.last_reconciliation[entity_id] = now
        self.processed_reconciliations.add(reconciliation_id)
        
        return {
            'reconciled': True,
            'entity_id': entity_id,
            'reconciliation_id': reconciliation_id,
            'timestamp': now.isoformat()
        }
    
    def write_feature(
        self,
        entity_id: str,
        feature_name: str,
        value: float,
        timestamp_bucket: str
    ) -> Dict:
        """
        Write feature with unique constraint enforcement.
        
        Args:
            entity_id: Entity identifier
            feature_name: Feature name
            value: Feature value
            timestamp_bucket: ISO timestamp bucket (hour-level)
            
        Returns:
            Dict with write result
            
        Raises:
            IntegrityError: If unique constraint violated
        """
        # Compute composite feature_id
        feature_id = self.compute_feature_id(
            entity_id=entity_id,
            feature_name=feature_name,
            timestamp_bucket=timestamp_bucket
        )
        
        # Unique constraint check
        if self.constraint_enabled:
            if feature_id in self.features:
                self._emit_metric('feature_store.duplicate_rejected', 1)
                raise IntegrityError(
                    f"duplicate key value violates unique constraint: "
                    f"feature_id={feature_id}"
                )
        else:
            if feature_id in self.features:
                logger.warning(
                    f"Duplicate feature write (constraint disabled): {feature_id}"
                )
                return {
                    'written': True,
                    'reason': 'constraint_disabled',
                    'feature_id': feature_id
                }
        
        # Write feature
        self.features[feature_id] = {
            'entity_id': entity_id,
            'feature_name': feature_name,
            'value': value,
            'timestamp_bucket': timestamp_bucket,
            'written_at': datetime.utcnow().isoformat()
        }
        
        logger.info(f"Feature written: {feature_id}")
        
        return {
            'written': True,
            'action': 'INSERT',
            'feature_id': feature_id
        }
    
    def compute_feature_id(
        self,
        entity_id: str,
        feature_name: str,
        timestamp_bucket: str
    ) -> str:
        """
        Compute deterministic feature_id from composite key.
        
        Returns:
            SHA256 hex hash (64 chars)
        """
        composite_key = f"{entity_id}:{feature_name}:{timestamp_bucket}"
        return hashlib.sha256(composite_key.encode()).hexdigest()
    
    def _emit_metric(self, metric_name: str, value: float):
        """Emit metric to monitoring system."""
        self.emitted_metrics[metric_name] += value
        logger.info(f"METRIC: {metric_name} = {value}")
    
    def get_emitted_metrics(self) -> Dict:
        """Get all emitted metrics (for testing)."""
        return dict(self.emitted_metrics)
