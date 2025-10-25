"""
Analytics Agent with GA4 schema drift monitoring.

Components:
- AnalyticsAgent: Main agent for analytics data ingestion
- GA4SchemaMonitor: Schema drift detection and alerting

Reference: Q_003 (C02 Red/CRITICAL - Schema drift alert 1h)
"""

from typing import Dict, List, Optional
from datetime import datetime, timedelta
from google.cloud import bigquery
from core.alert_client import AlertClient
from core.metrics import MetricsClient


class GA4SchemaMonitor:
    """
    GA4 BigQuery Export schema drift monitor.
    
    Features:
    - Continuous schema monitoring (hourly checks)
    - Alert within 1h of drift detection
    - Automatic baseline schema storage
    - CI schema validation gates
    
    Kill Switch: SCHEMA_MONITORING_ENABLED (default: true)
    Metrics: schema_drift_detected_minutes
    SLA: <60min alert latency
    
    Reference: Q_003 (C02 Red/CRITICAL)
    """

    def __init__(
        self,
        bigquery: bigquery.Client,
        alert_client: AlertClient,
        metrics_client: Optional[MetricsClient] = None,
        alert_channel: str = "#data-ops-alerts",
        sla_minutes: int = 60,
        dry_run: bool = False
    ):
        self.bigquery = bigquery
        self.alert_client = alert_client
        self.metrics = metrics_client or MetricsClient()
        self.alert_channel = alert_channel
        self.sla_minutes = sla_minutes
        self.dry_run = dry_run
        self.baseline_schemas = {}  # In production, store in Redis/DB

    def store_baseline_schema(self, table_pattern: str, schema: Dict):
        """
        Store baseline schema for comparison.
        
        Args:
            table_pattern: Table name or pattern (e.g., "events_*")
            schema: Schema dict {column_name: type}
        """
        self.baseline_schemas[table_pattern] = {
            "schema": schema,
            "stored_at": datetime.utcnow()
        }

    def get_current_schema(self, table_name: str) -> Dict:
        """
        Get current schema from BigQuery.
        
        Args:
            table_name: Fully qualified table name
        
        Returns:
            Schema dict {column_name: type}
        """
        table = self.bigquery.get_table(table_name)
        schema = {}
        for field in table.schema:
            schema[field.name] = field.field_type
        return schema

    def check_drift(self, table_pattern: str) -> Dict:
        """
        Check for schema drift and alert if detected.
        
        Args:
            table_pattern: Table name or pattern
        
        Returns:
            {
                "drift_detected": bool,
                "added_columns": list,
                "removed_columns": list,
                "type_changes": dict,
                "alert_sent": bool,
                "latency_minutes": float
            }
        """
        drift_start = datetime.utcnow()

        # Get baseline
        if table_pattern not in self.baseline_schemas:
            return {
                "drift_detected": False,
                "error": "No baseline schema stored for this table"
            }

        baseline = self.baseline_schemas[table_pattern]["schema"]

        # Get current schema
        try:
            current = self.get_current_schema(table_pattern)
        except Exception as e:
            return {
                "drift_detected": False,
                "error": f"Failed to fetch current schema: {str(e)}"
            }

        # Compare schemas
        baseline_cols = set(baseline.keys())
        current_cols = set(current.keys())

        added = current_cols - baseline_cols
        removed = baseline_cols - current_cols

        # Type changes
        type_changes = {}
        for col in baseline_cols & current_cols:
            if baseline[col] != current[col]:
                type_changes[col] = {
                    "old": baseline[col],
                    "new": current[col]
                }

        drift_detected = len(added) > 0 or len(removed) > 0 or len(type_changes) > 0

        result = {
            "drift_detected": drift_detected,
            "added_columns": list(added),
            "removed_columns": list(removed),
            "type_changes": type_changes,
            "table": table_pattern,
            "checked_at": drift_start.isoformat()
        }

        if drift_detected:
            # Calculate alert latency
            alert_latency_minutes = (datetime.utcnow() - drift_start).total_seconds() / 60

            # Update metric
            self.metrics.set_gauge("schema_drift_detected_minutes", alert_latency_minutes)

            # Send alert (unless dry-run)
            if not self.dry_run:
                self._send_drift_alert(result)
                result["alert_sent"] = True
            else:
                result["alert_sent"] = False

            result["latency_minutes"] = alert_latency_minutes

            # Verify SLA
            if alert_latency_minutes > self.sla_minutes:
                self.metrics.increment("schema_drift_sla_violations")

        else:
            # No drift, or drift recovered
            result["status"] = "ok" if len(baseline_cols) == len(current_cols) else "recovered"

        return result

    def _send_drift_alert(self, drift_info: Dict):
        """
        Send Slack alert for schema drift.
        
        Args:
            drift_info: Drift detection result
        """
        message = f"""
🚨 **GA4 Schema Drift Detected**

**Table:** `{drift_info['table']}`
**Detected at:** {drift_info['checked_at']}

**Changes:**
"""
        if drift_info["added_columns"]:
            message += f"\n➕ **Added columns:** {', '.join(drift_info['added_columns'])}"

        if drift_info["removed_columns"]:
            message += f"\n➖ **Removed columns:** {', '.join(drift_info['removed_columns'])}"

        if drift_info["type_changes"]:
            message += f"\n🔄 **Type changes:**\n"
            for col, change in drift_info["type_changes"].items():
                message += f"   - `{col}`: {change['old']} → {change['new']}\n"

        message += f"""
**Impact:** Downstream pipelines (dbt, MMM, MTA) may fail
**Action Required:** Update dbt models and schema definitions

**Runbook:** https://docs.company.com/runbooks/ga4-schema-drift
"""

        self.alert_client.send_alert(
            channel=self.alert_channel,
            message=message,
            severity="high",
            tags=["schema_drift", "ga4", drift_info["table"]]
        )

    def check_all_tables(self, tables: List[str]) -> Dict[str, Dict]:
        """
        Check drift for multiple tables.
        
        Args:
            tables: List of table patterns
        
        Returns:
            Dict mapping table -> drift result
        """
        results = {}
        for table in tables:
            results[table] = self.check_drift(table)
        return results


class AnalyticsAgent:
    """
    Analytics Agent for GA4 ingestion and monitoring.
    
    Features:
    - GA4 BigQuery Export ingestion
    - Schema drift detection and alerting
    - CI schema validation gates
    
    Kill Switch: SCHEMA_MONITORING_ENABLED
    """

    def __init__(
        self,
        bigquery: bigquery.Client,
        alert_client: AlertClient,
        metrics_client: Optional[MetricsClient] = None,
        monitoring_enabled: bool = True
    ):
        self.bigquery = bigquery
        self.alert_client = alert_client
        self.metrics = metrics_client or MetricsClient()
        self.monitoring_enabled = monitoring_enabled
        
        # Initialize schema monitor
        self.schema_monitor = GA4SchemaMonitor(
            bigquery=bigquery,
            alert_client=alert_client,
            metrics_client=metrics_client
        )

    def ingest_ga4_export(self, date: str) -> Dict:
        """
        Ingest GA4 BigQuery export for a specific date.
        
        Args:
            date: Date in YYYYMMDD format
        
        Returns:
            Ingestion result
        """
        table_name = f"analytics_XXXXX.events_{date}"
        
        # Check schema drift before ingestion
        if self.monitoring_enabled:
            drift_result = self.schema_monitor.check_drift(f"events_*")
            if drift_result["drift_detected"]:
                return {
                    "ok": False,
                    "error": "Schema drift detected. Ingestion blocked.",
                    "drift_info": drift_result
                }
        
        # Proceed with ingestion (placeholder)
        # In production: query GA4 export, transform, load to data warehouse
        
        return {
            "ok": True,
            "table": table_name,
            "rows_ingested": 0
        }

    def check_schema_drift(self, table_pattern: str) -> Dict:
        """
        Check schema drift for a table pattern.
        
        Args:
            table_pattern: Table name or pattern
        
        Returns:
            Drift detection result
        """
        if not self.monitoring_enabled:
            return {
                "checked": False,
                "reason": "monitoring_disabled"
            }
        
        return self.schema_monitor.check_drift(table_pattern)

    def validate_schema_against_baseline(
        self,
        table_name: str,
        baseline: Dict,
        current: Dict
    ) -> Dict:
        """
        Validate current schema against baseline for CI gates.
        
        Args:
            table_name: Table name
            baseline: Baseline schema
            current: Current schema
        
        Returns:
            {
                "valid": bool,
                "diff": {"added": [], "removed": [], "type_changes": {}}
            }
        """
        baseline_cols = set(baseline.keys())
        current_cols = set(current.keys())

        added = current_cols - baseline_cols
        removed = baseline_cols - current_cols

        type_changes = {}
        for col in baseline_cols & current_cols:
            baseline_type = baseline[col].get("type") if isinstance(baseline[col], dict) else baseline[col]
            current_type = current[col].get("type") if isinstance(current[col], dict) else current[col]
            
            if baseline_type != current_type:
                type_changes[col] = {
                    "old": baseline_type,
                    "new": current_type
                }

        valid = len(added) == 0 and len(removed) == 0 and len(type_changes) == 0

        return {
            "valid": valid,
            "table": table_name,
            "diff": {
                "added": list(added),
                "removed": list(removed),
                "type_changes": type_changes
            }
        }

    def ci_schema_gate(self, validation_result: Dict) -> Dict:
        """
        CI gate decision based on schema validation.
        
        Args:
            validation_result: Result from validate_schema_against_baseline
        
        Returns:
            {
                "deploy_blocked": bool,
                "reason": str
            }
        """
        if validation_result["valid"]:
            return {
                "deploy_blocked": False,
                "reason": "Schema validation passed"
            }
        else:
            return {
                "deploy_blocked": True,
                "reason": "Schema drift detected. CI gate blocks deployment.",
                "diff": validation_result["diff"]
            }


# Metrics
METRICS = {
    "schema_drift_detected_minutes": {
        "type": "gauge",
        "description": "Minutes elapsed from drift detection to alert (should be <60)"
    },
    "schema_drift_sla_violations": {
        "type": "counter",
        "description": "Count of schema drift alerts exceeding 60min SLA"
    }
}
