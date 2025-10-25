"""
Unit tests for GA4 schema drift detection and alerting.

Tests:
- test_schema_drift_alert_1h: Verify alert sent within 1h of drift detection
- test_schema_drift_metric: Verify schema_drift_detected_minutes <60
- test_ci_schema_validation: Verify dbt schema test fails on drift before deploy
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
from agents.analytics_agent import AnalyticsAgent, GA4SchemaMonitor


class TestSchemaDrift:
    """Test suite for GA4 schema drift detection (Q_003)"""

    @pytest.fixture
    def analytics_agent(self, bigquery_client, alert_client, metrics_client):
        """Fixture to provide AnalyticsAgent with test dependencies"""
        return AnalyticsAgent(
            bigquery=bigquery_client,
            alert_client=alert_client,
            metrics=metrics_client,
            monitoring_enabled=True
        )

    @pytest.fixture
    def schema_monitor(self, bigquery_client, alert_client):
        """Fixture for GA4SchemaMonitor"""
        return GA4SchemaMonitor(
            bigquery=bigquery_client,
            alert_client=alert_client,
            alert_channel="#data-ops-alerts",
            sla_minutes=60
        )

    @pytest.fixture
    def mock_ga4_export_schema(self):
        """Mock GA4 BigQuery export schema"""
        return {
            "event_name": "STRING",
            "event_timestamp": "TIMESTAMP",
            "user_pseudo_id": "STRING",
            "event_params": "RECORD",
            "user_properties": "RECORD"
        }

    def test_schema_drift_alert_1h(self, schema_monitor, mock_ga4_export_schema, alert_client):
        """
        ACCEPTANCE: unit: test_schema_drift.py::test_schema_drift_alert_1h passes
        RISK_GATE: alert_channel=#data-ops-alerts, sla_minutes=60
        """
        # Store baseline schema
        schema_monitor.store_baseline_schema("events_*", mock_ga4_export_schema)

        # Simulate drift: new column added
        drifted_schema = mock_ga4_export_schema.copy()
        drifted_schema["new_column"] = "STRING"

        # Mock BigQuery to return drifted schema
        with patch.object(schema_monitor.bigquery, "get_table_schema", return_value=drifted_schema):
            # Run drift detection
            drift_start = datetime.utcnow()
            drift_result = schema_monitor.check_drift("events_*")

            assert drift_result["drift_detected"] is True, "Drift should be detected"
            assert "new_column" in drift_result["added_columns"]

            # Verify alert was sent
            alert_client.send_alert.assert_called_once()
            alert_call_args = alert_client.send_alert.call_args

            # Check alert was sent within 1h
            alert_time = datetime.utcnow()
            alert_latency = (alert_time - drift_start).total_seconds() / 60  # minutes

            assert alert_latency < 60, f"Alert latency {alert_latency:.1f}min exceeds 60min SLA"

            # Verify alert content
            assert "#data-ops-alerts" in str(alert_call_args)
            assert "schema drift detected" in str(alert_call_args).lower()

    def test_schema_drift_metric(self, schema_monitor, mock_ga4_export_schema, metrics_client):
        """
        ACCEPTANCE: metric: schema_drift_detected_minutes <60
        """
        # Store baseline
        schema_monitor.store_baseline_schema("events_*", mock_ga4_export_schema)

        # Simulate drift
        drifted_schema = mock_ga4_export_schema.copy()
        drifted_schema["unexpected_field"] = "INTEGER"

        with patch.object(schema_monitor.bigquery, "get_table_schema", return_value=drifted_schema):
            drift_start = datetime.utcnow()

            # Detect drift
            schema_monitor.check_drift("events_*")

            # Check metric
            metric_value = metrics_client.get_gauge("schema_drift_detected_minutes")

            drift_latency = (datetime.utcnow() - drift_start).total_seconds() / 60
            assert metric_value < 60, f"schema_drift_detected_minutes={metric_value:.1f} exceeds 60min SLA"
            assert metric_value == pytest.approx(drift_latency, abs=0.1)

    def test_ci_schema_validation(self, analytics_agent):
        """
        ACCEPTANCE: CI: dbt schema test fails on drift before production deploy
        DRY_RUN_PROBE: Inject GA4 BigQuery export with new column; 
        monitor alert latency in test env
        """
        # Simulate dbt schema test during CI
        baseline_schema = {
            "event_name": {"type": "STRING", "mode": "NULLABLE"},
            "event_timestamp": {"type": "TIMESTAMP", "mode": "NULLABLE"}
        }

        # Current schema has extra column (drift)
        current_schema = baseline_schema.copy()
        current_schema["extra_column"] = {"type": "STRING", "mode": "NULLABLE"}

        # Run schema validation (should fail)
        validation_result = analytics_agent.validate_schema_against_baseline(
            table_name="events_*",
            baseline=baseline_schema,
            current=current_schema
        )

        assert validation_result["valid"] is False, "Schema validation should fail on drift"
        assert "extra_column" in validation_result["diff"]["added"]

        # Verify CI would block deployment
        ci_gate_result = analytics_agent.ci_schema_gate(validation_result)
        assert ci_gate_result["deploy_blocked"] is True
        assert "schema drift detected" in ci_gate_result["reason"].lower()

    def test_slack_notification_format(self, schema_monitor, mock_ga4_export_schema, alert_client):
        """
        Verify Slack notification contains required information
        """
        schema_monitor.store_baseline_schema("events_*", mock_ga4_export_schema)

        # Drift: removed column
        drifted_schema = mock_ga4_export_schema.copy()
        del drifted_schema["event_params"]

        with patch.object(schema_monitor.bigquery, "get_table_schema", return_value=drifted_schema):
            schema_monitor.check_drift("events_*")

            # Verify Slack message structure
            alert_call = alert_client.send_alert.call_args[1]
            message = alert_call["message"]

            assert "schema drift" in message.lower()
            assert "events_*" in message
            assert "event_params" in message  # Removed column
            assert "runbook" in message.lower()

    def test_multiple_table_monitoring(self, schema_monitor):
        """
        Test monitoring multiple GA4 tables simultaneously
        """
        tables = [
            "events_20251019",
            "events_20251018",
            "events_intraday_*"
        ]

        baseline = {"col1": "STRING", "col2": "INT64"}
        for table in tables:
            schema_monitor.store_baseline_schema(table, baseline)

        # Drift in one table only
        drifted_schema = baseline.copy()
        drifted_schema["col3"] = "BOOL"

        with patch.object(schema_monitor.bigquery, "get_table_schema") as mock_get:
            # Return drifted schema for events_20251019, baseline for others
            def side_effect(table_name):
                if table_name == "events_20251019":
                    return drifted_schema
                return baseline

            mock_get.side_effect = side_effect

            # Check all tables
            results = schema_monitor.check_all_tables(tables)

            assert results["events_20251019"]["drift_detected"] is True
            assert results["events_20251018"]["drift_detected"] is False
            assert results["events_intraday_*"]["drift_detected"] is False

    def test_kill_switch_disables_monitoring(self, analytics_agent):
        """
        Test SCHEMA_MONITORING_ENABLED kill switch
        """
        # Disable monitoring
        analytics_agent.monitoring_enabled = False

        # Attempt drift check (should skip)
        result = analytics_agent.check_schema_drift("events_*")

        assert result["checked"] is False
        assert result["reason"] == "monitoring_disabled"

    def test_dry_run_mode_no_alerts(self, schema_monitor, mock_ga4_export_schema, alert_client):
        """
        Test dry-run mode: detect drift but don't send alerts
        """
        schema_monitor.dry_run = True
        schema_monitor.store_baseline_schema("events_*", mock_ga4_export_schema)

        # Drift
        drifted_schema = mock_ga4_export_schema.copy()
        drifted_schema["new_field"] = "STRING"

        with patch.object(schema_monitor.bigquery, "get_table_schema", return_value=drifted_schema):
            result = schema_monitor.check_drift("events_*")

            assert result["drift_detected"] is True
            # No alert should be sent in dry-run
            alert_client.send_alert.assert_not_called()

    def test_schema_drift_recovery(self, schema_monitor, mock_ga4_export_schema):
        """
        Test recovery when schema returns to baseline
        """
        schema_monitor.store_baseline_schema("events_*", mock_ga4_export_schema)

        # First check: drift detected
        drifted_schema = mock_ga4_export_schema.copy()
        drifted_schema["temp_column"] = "STRING"

        with patch.object(schema_monitor.bigquery, "get_table_schema", return_value=drifted_schema):
            result1 = schema_monitor.check_drift("events_*")
            assert result1["drift_detected"] is True

        # Second check: schema restored to baseline
        with patch.object(schema_monitor.bigquery, "get_table_schema", return_value=mock_ga4_export_schema):
            result2 = schema_monitor.check_drift("events_*")
            assert result2["drift_detected"] is False
            assert result2["status"] == "recovered"
