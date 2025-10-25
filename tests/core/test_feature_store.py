# Test: Feature Store - Online/Offline Parity and Unique Constraints

import pytest
from datetime import datetime
from unittest.mock import Mock, patch
from core.feature_store import FeatureStore, FeatureParityCheck, IntegrityError

class TestFeatureStoreParity:
    """Test online/offline feature parity detection"""
    
    @pytest.fixture
    def store(self):
        return FeatureStore(
            feature_flag_parity_check=True,
            parity_threshold_pct=1.0,
            reconciliation_rate_limit_per_hour=1
        )
    
    def test_feature_parity_check_pass_when_delta_low(self, store):
        """Test: delta=0.5% passes parity check"""
        # Arrange
        online_value = 100.0
        offline_value = 100.5  # 0.5% delta
        
        # Act
        result = store.check_feature_parity(
            entity_id="user_123",
            feature_name="avg_order_value",
            online_value=online_value,
            offline_value=offline_value
        )
        
        # Assert
        assert result['parity_check_passed'] is True
        assert result['delta_pct'] == 0.5
        assert result['threshold_pct'] == 1.0
        assert 'feature_store.parity_delta_pct' in result['metrics_emitted']
    
    def test_feature_parity_check_fail_when_delta_high(self, store):
        """Test: delta=1.5% fails parity check and alerts"""
        # Arrange
        online_value = 100.0
        offline_value = 101.5  # 1.5% delta
        
        # Act
        result = store.check_feature_parity(
            entity_id="user_123",
            feature_name="avg_order_value",
            online_value=online_value,
            offline_value=offline_value
        )
        
        # Assert
        assert result['parity_check_passed'] is False
        assert result['delta_pct'] == 1.5
        assert result['alert_triggered'] is True
        assert 'feature_store.parity_violation' in result['metrics_emitted']
    
    def test_reconciliation_rate_limited(self, store):
        """Test: reconciliation max 1/hour to avoid load"""
        # Arrange - First reconciliation
        result1 = store.reconcile_features(entity_id="user_123")
        
        # Act - Attempt second reconciliation immediately
        result2 = store.reconcile_features(entity_id="user_123")
        
        # Assert
        assert result1['reconciled'] is True
        assert result2['reconciled'] is False
        assert result2['reason'] == "rate_limit_exceeded"
    
    def test_idempotency_key_prevents_duplicate_reconciliation(self, store):
        """Test: Same reconciliation_id skips duplicate work"""
        # Arrange
        reconciliation_id = "recon_20251019_090000"
        
        # Act
        result1 = store.reconcile_features(
            entity_id="user_123",
            reconciliation_id=reconciliation_id
        )
        result2 = store.reconcile_features(
            entity_id="user_123",
            reconciliation_id=reconciliation_id
        )
        
        # Assert
        assert result1['reconciled'] is True
        assert result2['reconciled'] is False
        assert result2['reason'] == "duplicate_reconciliation_id"


class TestFeatureStoreUniqueConstraint:
    """Test unique constraint on (entity_id, feature_name, timestamp)"""
    
    @pytest.fixture
    def store(self):
        return FeatureStore(
            feature_flag_unique_constraint=True
        )
    
    def test_feature_unique_constraint_rejects_duplicate(self, store):
        """Test: duplicate (entity_id, feature_name, timestamp) raises IntegrityError"""
        # Arrange
        feature_data = {
            'entity_id': 'user_123',
            'feature_name': 'avg_order_value',
            'value': 100.0,
            'timestamp_bucket': '2025-10-19T09:00:00Z'
        }
        
        # Act - First insert succeeds
        result1 = store.write_feature(**feature_data)
        
        # Act - Second insert with same key fails
        with pytest.raises(IntegrityError) as exc_info:
            store.write_feature(**feature_data)
        
        # Assert
        assert result1['written'] is True
        assert "duplicate key value violates unique constraint" in str(exc_info.value)
        assert exc_info.value.status_code == 409
    
    def test_feature_update_with_newer_timestamp_succeeds(self, store):
        """Test: same (entity_id, feature_name) with newer timestamp updates"""
        # Arrange
        feature_v1 = {
            'entity_id': 'user_123',
            'feature_name': 'avg_order_value',
            'value': 100.0,
            'timestamp_bucket': '2025-10-19T09:00:00Z'
        }
        feature_v2 = {
            'entity_id': 'user_123',
            'feature_name': 'avg_order_value',
            'value': 105.0,
            'timestamp_bucket': '2025-10-19T10:00:00Z'  # Newer
        }
        
        # Act
        result1 = store.write_feature(**feature_v1)
        result2 = store.write_feature(**feature_v2)
        
        # Assert
        assert result1['written'] is True
        assert result2['written'] is True
        assert result2['action'] == 'INSERT'  # New timestamp = new row
    
    def test_feature_id_composite_hash(self, store):
        """Test: feature_id = hash(entity_id, feature_name, timestamp_bucket)"""
        # Arrange
        feature_data = {
            'entity_id': 'user_123',
            'feature_name': 'avg_order_value',
            'timestamp_bucket': '2025-10-19T09:00:00Z'
        }
        
        # Act
        feature_id = store.compute_feature_id(**feature_data)
        
        # Assert
        assert len(feature_id) == 64  # SHA256 hex
        assert feature_id == store.compute_feature_id(**feature_data)  # Deterministic
    
    def test_duplicate_rejected_metric_emitted(self, store):
        """Test: 409 emits 'feature_store.duplicate_rejected' counter"""
        # Arrange
        feature_data = {
            'entity_id': 'user_123',
            'feature_name': 'avg_order_value',
            'value': 100.0,
            'timestamp_bucket': '2025-10-19T09:00:00Z'
        }
        
        store.write_feature(**feature_data)
        
        # Act
        with pytest.raises(IntegrityError):
            result = store.write_feature(**feature_data)
        
        # Assert
        metrics = store.get_emitted_metrics()
        assert 'feature_store.duplicate_rejected' in metrics
    
    def test_kill_switch_disables_constraint(self, store):
        """Test: Feature flag off disables unique constraint check"""
        # Arrange
        store_disabled = FeatureStore(feature_flag_unique_constraint=False)
        feature_data = {
            'entity_id': 'user_123',
            'feature_name': 'avg_order_value',
            'value': 100.0,
            'timestamp_bucket': '2025-10-19T09:00:00Z'
        }
        
        # Act - Both inserts succeed when constraint disabled
        result1 = store_disabled.write_feature(**feature_data)
        result2 = store_disabled.write_feature(**feature_data)
        
        # Assert
        assert result1['written'] is True
        assert result2['written'] is True
        assert result2['reason'] == 'constraint_disabled'
