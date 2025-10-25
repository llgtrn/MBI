"""
Unit tests for schema drift detection with <5min alert
Validates Q_003 + A_010: Observability gap with drift detector
"""
import pytest
from datetime import datetime
from unittest.mock import Mock, patch, AsyncMock
import asyncio


class TestSchemaValidatorDrift:
    """Test suite for schema drift detection (Q_003 + A_010)"""
    
    @pytest.mark.asyncio
    async def test_v2_schema_alert_within_5min(self):
        """
        Acceptance: v2 schema detected and alert fires within 300s
        Contract: SchemaChangeEvent emitted within 5min
        Metric: schema_drift_detected_total increments
        """
        from agents.schema_validator import SchemaValidator
        
        # Mock alert system
        mock_alerter = AsyncMock()
        
        # Mock schema registry with v2 schema
        mock_registry = Mock()
        mock_registry.get_latest_version.return_value = "v2"
        mock_registry.get_schema.return_value = {
            "version": "v2",
            "fields": {
                "user_key": "string",
                "new_field": "integer"  # Breaking change
            }
        }
        
        validator = SchemaValidator(
            schema_registry=mock_registry,
            alerter=mock_alerter,
            enable_drift_alerts=True
        )
        
        # Current schema (v1)
        current_schema = {
            "version": "v1",
            "fields": {
                "user_key": "string"
            }
        }
        
        # Detect drift
        start_time = datetime.utcnow()
        drift_detected = await validator.check_drift(current_schema)
        end_time = datetime.utcnow()
        
        # Assertions
        assert drift_detected is True, "Must detect v1→v2 drift"
        
        # Verify alert was sent
        mock_alerter.send_alert.assert_called_once()
        alert_call = mock_alerter.send_alert.call_args
        assert alert_call[1]['event_type'] == 'SchemaChangeEvent'
        assert alert_call[1]['severity'] == 'WARNING'
        assert 'v1' in alert_call[1]['message']
        assert 'v2' in alert_call[1]['message']
        
        # Verify alert timing <5min
        alert_time = alert_call[1]['timestamp']
        delta = (datetime.fromisoformat(alert_time) - start_time).total_seconds()
        assert delta < 300, f"Alert must fire within 300s, got {delta}s"
        
        # Check metrics
        metrics = validator.get_metrics()
        assert metrics['schema_drift_detected_total'] == 1
    
    @pytest.mark.asyncio
    async def test_drift_severity_threshold(self):
        """
        Risk gate: Only alert on MAJOR/BREAKING changes, not MINOR
        """
        from agents.schema_validator import SchemaValidator, DriftSeverity
        
        mock_alerter = AsyncMock()
        mock_registry = Mock()
        
        validator = SchemaValidator(
            schema_registry=mock_registry,
            alerter=mock_alerter,
            drift_severity_threshold=DriftSeverity.MAJOR
        )
        
        current = {"version": "v1", "fields": {"user_key": "string"}}
        
        # MINOR change: new optional field
        new_minor = {"version": "v1.1", "fields": {"user_key": "string", "optional_field": "string?"}}
        mock_registry.get_schema.return_value = new_minor
        
        drift = await validator.check_drift(current)
        
        # Should detect drift but NOT alert (below threshold)
        assert drift is True
        mock_alerter.send_alert.assert_not_called()
        
        # MAJOR change: type change
        new_major = {"version": "v2", "fields": {"user_key": "integer"}}  # Breaking
        mock_registry.get_schema.return_value = new_major
        
        drift = await validator.check_drift(current)
        
        # Should detect AND alert (above threshold)
        assert drift is True
        mock_alerter.send_alert.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_kill_switch_disables_alerts(self):
        """
        Risk gate: ENABLE_DRIFT_ALERTS kill switch
        """
        from agents.schema_validator import SchemaValidator
        
        mock_alerter = AsyncMock()
        mock_registry = Mock()
        mock_registry.get_schema.return_value = {
            "version": "v2",
            "fields": {"user_key": "integer"}
        }
        
        validator = SchemaValidator(
            schema_registry=mock_registry,
            alerter=mock_alerter,
            enable_drift_alerts=False  # Kill switch OFF
        )
        
        current = {"version": "v1", "fields": {"user_key": "string"}}
        
        drift = await validator.check_drift(current)
        
        # Drift detected but no alert sent
        assert drift is True
        mock_alerter.send_alert.assert_not_called()
        
        # Metrics still tracked
        metrics = validator.get_metrics()
        assert metrics['schema_drift_detected_total'] == 1
        assert metrics['alerts_suppressed_total'] == 1
    
    @pytest.mark.asyncio
    async def test_schema_registry_version_check(self):
        """
        Risk gate: Verify schema registry version before alerting
        """
        from agents.schema_validator import SchemaValidator
        
        mock_alerter = AsyncMock()
        mock_registry = Mock()
        
        # Simulate stale registry (no new version)
        mock_registry.get_latest_version.return_value = "v1"
        mock_registry.get_schema.return_value = {
            "version": "v1",
            "fields": {"user_key": "string"}
        }
        
        validator = SchemaValidator(
            schema_registry=mock_registry,
            alerter=mock_alerter
        )
        
        current = {"version": "v1", "fields": {"user_key": "string"}}
        
        drift = await validator.check_drift(current)
        
        # No drift (same version)
        assert drift is False
        mock_alerter.send_alert.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_prometheus_counter_increments(self):
        """
        Verify prometheus metric schema_drift_detected_total increments correctly
        """
        from agents.schema_validator import SchemaValidator
        
        mock_alerter = AsyncMock()
        mock_registry = Mock()
        mock_registry.get_schema.return_value = {
            "version": "v2",
            "fields": {"new_field": "string"}
        }
        
        validator = SchemaValidator(
            schema_registry=mock_registry,
            alerter=mock_alerter
        )
        
        current = {"version": "v1", "fields": {"user_key": "string"}}
        
        # Initial state
        metrics = validator.get_metrics()
        initial_count = metrics['schema_drift_detected_total']
        
        # Trigger drift
        await validator.check_drift(current)
        
        # Check increment
        metrics = validator.get_metrics()
        assert metrics['schema_drift_detected_total'] == initial_count + 1
        
        # Trigger again
        await validator.check_drift(current)
        metrics = validator.get_metrics()
        assert metrics['schema_drift_detected_total'] == initial_count + 2
