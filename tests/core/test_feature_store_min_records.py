# Test: PII MIN_RECORDS Enforcement

import pytest
from core.feature_store import FeatureStore
from core.exceptions import InsufficientDataError


def test_min_records_enforcement_99_rejects():
    """Q_005: Verify MIN_RECORDS=100 enforcement rejects 99 records"""
    store = FeatureStore()
    
    # Attempt to create feature set with 99 records
    with pytest.raises(InsufficientDataError) as exc_info:
        store.create_feature_set(
            name="test_features",
            records=[{"user_key": f"u{i}", "value": i} for i in range(99)]
        )
    
    assert "Minimum 100 records required" in str(exc_info.value)
    assert exc_info.value.record_count == 99
    assert exc_info.value.min_required == 100


def test_min_records_enforcement_100_accepts():
    """Q_005: Verify MIN_RECORDS=100 enforcement accepts exactly 100 records"""
    store = FeatureStore()
    
    # Should succeed with exactly 100 records
    result = store.create_feature_set(
        name="test_features",
        records=[{"user_key": f"u{i}", "value": i} for i in range(100)]
    )
    
    assert result.status == "created"
    assert result.record_count == 100


def test_min_records_metric_emitted():
    """Q_005: Verify insufficient_records_total metric increments on rejection"""
    from unittest.mock import Mock
    from core.metrics import prometheus_counter
    
    store = FeatureStore()
    counter_mock = Mock()
    
    with pytest.MonkeyPatch.context() as m:
        m.setattr("core.feature_store.insufficient_records_total", counter_mock)
        
        try:
            store.create_feature_set(
                name="test_features",
                records=[{"user_key": f"u{i}", "value": i} for i in range(50)]
            )
        except InsufficientDataError:
            pass
        
        counter_mock.inc.assert_called_once_with(
            labels={"feature_set": "test_features", "record_count": 50}
        )
