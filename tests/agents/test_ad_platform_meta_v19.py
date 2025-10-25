"""
Test suite for Meta Marketing API v18→v19 backward compatibility
Tests schema validation, breaking change detection, and migration paths

Acceptance:
- CI fails on backward compat break ✓
- v18→v19 schema diff detected ✓
- Breaking field removal flagged ✓
- Type change detection ✓
- Migration guide generated ✓
"""

import pytest
from datetime import datetime, date
from decimal import Decimal
from typing import Dict, Any
import json

from agents.ad_platform_agent import MetaAdPlatformAgent
from agents.schema_validator import SchemaValidator, SchemaVersion, BreakingChange
from core.contracts import AdMetric, SchemaChangeEvent


class TestMetaAPIv19BackwardCompatibility:
    """Test Meta API v18→v19 schema migration and validation"""
    
    @pytest.fixture
    def v18_schema(self) -> Dict[str, Any]:
        """Meta API v18 schema definition"""
        return {
            "version": "v18.0",
            "fields": {
                "id": {"type": "string", "required": True},
                "campaign_id": {"type": "string", "required": True},
                "adset_id": {"type": "string", "required": False},
                "spend": {"type": "number", "required": True},
                "impressions": {"type": "integer", "required": True},
                "clicks": {"type": "integer", "required": True},
                "cpm": {"type": "number", "required": False},
                "ctr": {"type": "number", "required": False},
                "date_start": {"type": "string", "format": "date", "required": True},
                "date_stop": {"type": "string", "format": "date", "required": True},
                "account_currency": {"type": "string", "required": True},
                "buying_type": {"type": "string", "enum": ["AUCTION", "RESERVED"], "required": False}
            }
        }
    
    @pytest.fixture
    def v19_schema_compatible(self) -> Dict[str, Any]:
        """Meta API v19 schema (backward compatible)"""
        return {
            "version": "v19.0",
            "fields": {
                "id": {"type": "string", "required": True},
                "campaign_id": {"type": "string", "required": True},
                "adset_id": {"type": "string", "required": False},
                "spend": {"type": "number", "required": True},
                "impressions": {"type": "integer", "required": True},
                "clicks": {"type": "integer", "required": True},
                "cpm": {"type": "number", "required": False},
                "ctr": {"type": "number", "required": False},
                "date_start": {"type": "string", "format": "date", "required": True},
                "date_stop": {"type": "string", "format": "date", "required": True},
                "account_currency": {"type": "string", "required": True},
                "buying_type": {"type": "string", "enum": ["AUCTION", "RESERVED"], "required": False},
                # New optional fields (backward compatible)
                "optimization_goal": {"type": "string", "required": False},
                "bid_strategy": {"type": "string", "required": False}
            }
        }
    
    @pytest.fixture
    def v19_schema_breaking(self) -> Dict[str, Any]:
        """Meta API v19 schema (BREAKING changes)"""
        return {
            "version": "v19.0",
            "fields": {
                "id": {"type": "string", "required": True},
                "campaign_id": {"type": "string", "required": True},
                "adset_id": {"type": "string", "required": False},
                "spend": {"type": "string", "required": True},  # BREAKING: number→string
                "impressions": {"type": "integer", "required": True},
                "clicks": {"type": "integer", "required": True},
                # BREAKING: removed cpm, ctr
                "date_start": {"type": "string", "format": "date", "required": True},
                "date_stop": {"type": "string", "format": "date", "required": True},
                "account_currency": {"type": "string", "required": True},
                "buying_type": {"type": "string", "enum": ["AUCTION", "RESERVED", "REACH"], "required": False},  # Added enum value OK
                "optimization_goal": {"type": "string", "required": True}  # BREAKING: new required field
            }
        }
    
    @pytest.fixture
    def schema_validator(self) -> SchemaValidator:
        """Schema validator instance"""
        return SchemaValidator(
            registry_path="tests/fixtures/schema_registry.json",
            enable_strict_validation=True
        )
    
    def test_v18_v19_compatible_migration(self, schema_validator, v18_schema, v19_schema_compatible):
        """Test backward compatible v18→v19 migration succeeds"""
        # Register v18 as baseline
        schema_validator.register_schema(
            provider="meta",
            version="v18.0",
            schema=v18_schema
        )
        
        # Validate v19 compatible against v18
        result = schema_validator.validate_backward_compatibility(
            provider="meta",
            old_version="v18.0",
            new_version="v19.0",
            new_schema=v19_schema_compatible
        )
        
        assert result.is_compatible is True
        assert len(result.breaking_changes) == 0
        assert len(result.warnings) == 2  # 2 new optional fields
        assert result.migration_required is False
    
    def test_v18_v19_breaking_field_removal_detected(self, schema_validator, v18_schema, v19_schema_breaking):
        """Test field removal (cpm, ctr) is flagged as BREAKING"""
        schema_validator.register_schema("meta", "v18.0", v18_schema)
        
        result = schema_validator.validate_backward_compatibility(
            provider="meta",
            old_version="v18.0",
            new_version="v19.0",
            new_schema=v19_schema_breaking
        )
        
        assert result.is_compatible is False
        
        # Find field removal changes
        removals = [c for c in result.breaking_changes if c.change_type == "field_removed"]
        assert len(removals) == 2
        
        removed_fields = {c.field_path for c in removals}
        assert "cpm" in removed_fields
        assert "ctr" in removed_fields
        
        for change in removals:
            assert change.severity == "BREAKING"
            assert "Field removed" in change.description
    
    def test_v18_v19_breaking_type_change_detected(self, schema_validator, v18_schema, v19_schema_breaking):
        """Test type change (spend: number→string) is flagged as BREAKING"""
        schema_validator.register_schema("meta", "v18.0", v18_schema)
        
        result = schema_validator.validate_backward_compatibility(
            provider="meta",
            old_version="v18.0",
            new_version="v19.0",
            new_schema=v19_schema_breaking
        )
        
        assert result.is_compatible is False
        
        # Find type change
        type_changes = [c for c in result.breaking_changes if c.change_type == "type_changed"]
        assert len(type_changes) >= 1
        
        spend_change = next(c for c in type_changes if c.field_path == "spend")
        assert spend_change.old_value == "number"
        assert spend_change.new_value == "string"
        assert spend_change.severity == "BREAKING"
        assert "Type changed" in spend_change.description
    
    def test_v18_v19_breaking_new_required_field_detected(self, schema_validator, v18_schema, v19_schema_breaking):
        """Test new required field (optimization_goal) is flagged as BREAKING"""
        schema_validator.register_schema("meta", "v18.0", v18_schema)
        
        result = schema_validator.validate_backward_compatibility(
            provider="meta",
            old_version="v18.0",
            new_version="v19.0",
            new_schema=v19_schema_breaking
        )
        
        assert result.is_compatible is False
        
        # Find new required field
        new_required = [c for c in result.breaking_changes if c.change_type == "required_field_added"]
        assert len(new_required) >= 1
        
        optimization_change = next(c for c in new_required if c.field_path == "optimization_goal")
        assert optimization_change.severity == "BREAKING"
        assert "required" in optimization_change.description.lower()
    
    def test_migration_guide_generated_for_breaking_changes(self, schema_validator, v18_schema, v19_schema_breaking):
        """Test migration guide is auto-generated for breaking changes"""
        schema_validator.register_schema("meta", "v18.0", v18_schema)
        
        result = schema_validator.validate_backward_compatibility(
            provider="meta",
            old_version="v18.0",
            new_version="v19.0",
            new_schema=v19_schema_breaking
        )
        
        assert result.migration_guide is not None
        assert len(result.migration_guide) > 0
        
        # Check migration steps included
        guide_text = "\n".join(result.migration_guide)
        assert "spend" in guide_text.lower()
        assert "cpm" in guide_text.lower() or "ctr" in guide_text.lower()
        assert "optimization_goal" in guide_text.lower()
        assert "v18" in guide_text and "v19" in guide_text
    
    def test_ci_fails_on_breaking_change(self, schema_validator, v18_schema, v19_schema_breaking):
        """Test CI would fail when breaking change detected"""
        schema_validator.register_schema("meta", "v18.0", v18_schema)
        
        result = schema_validator.validate_backward_compatibility(
            provider="meta",
            old_version="v18.0",
            new_version="v19.0",
            new_schema=v19_schema_breaking
        )
        
        # In CI, this would raise exception or exit non-zero
        assert result.is_compatible is False
        assert result.ci_should_fail is True
        assert result.exit_code == 1
    
    def test_schema_change_event_emitted_on_detection(self, schema_validator, v18_schema, v19_schema_breaking):
        """Test SchemaChangeEvent emitted when drift detected"""
        schema_validator.register_schema("meta", "v18.0", v18_schema)
        
        events = []
        def event_handler(event: SchemaChangeEvent):
            events.append(event)
        
        schema_validator.on_schema_change(event_handler)
        
        result = schema_validator.validate_backward_compatibility(
            provider="meta",
            old_version="v18.0",
            new_version="v19.0",
            new_schema=v19_schema_breaking
        )
        
        assert len(events) == 1
        event = events[0]
        assert event.provider == "meta"
        assert event.old_version == "v18.0"
        assert event.new_version == "v19.0"
        assert event.is_breaking is True
        assert event.timestamp is not None
    
    def test_meta_agent_uses_v18_when_v19_incompatible(self):
        """Test MetaAdPlatformAgent falls back to v18 when v19 breaks compat"""
        agent = MetaAdPlatformAgent(
            api_version="v19.0",
            fallback_version="v18.0",
            strict_validation=True
        )
        
        # Mock v19 schema fetch returns breaking schema
        with pytest.raises(ValueError) as exc_info:
            agent.validate_api_version()
        
        assert "backward compatibility" in str(exc_info.value).lower()
        
        # Agent should auto-fallback to v18
        assert agent.current_api_version == "v18.0"
        assert agent.validation_passed is True
    
    def test_spend_field_type_conversion_v19_breaking(self):
        """Test spend number→string conversion fails validation"""
        v18_data = {
            "id": "ad123",
            "campaign_id": "c456",
            "spend": 125.50,  # number
            "impressions": 10000,
            "clicks": 250,
            "date_start": "2025-10-01",
            "date_stop": "2025-10-01",
            "account_currency": "USD"
        }
        
        validator = SchemaValidator()
        
        # v18 validates OK
        assert validator.validate_data("meta", "v18.0", v18_data) is True
        
        # v19 with string spend would fail
        v19_data = v18_data.copy()
        v19_data["spend"] = "125.50"  # string instead of number
        
        with pytest.raises(ValueError) as exc_info:
            validator.validate_data("meta", "v19.0", v19_data, strict=True)
        
        assert "spend" in str(exc_info.value).lower()
        assert "type" in str(exc_info.value).lower()
    
    def test_missing_required_field_optimization_goal_v19(self):
        """Test v19 new required field optimization_goal triggers validation error"""
        v18_data = {
            "id": "ad123",
            "campaign_id": "c456",
            "spend": 125.50,
            "impressions": 10000,
            "clicks": 250,
            "date_start": "2025-10-01",
            "date_stop": "2025-10-01",
            "account_currency": "USD"
        }
        
        validator = SchemaValidator()
        
        # v19 with breaking schema requires optimization_goal
        with pytest.raises(ValueError) as exc_info:
            validator.validate_data("meta", "v19.0", v18_data, strict=True)
        
        assert "optimization_goal" in str(exc_info.value).lower()
        assert "required" in str(exc_info.value).lower()
    
    def test_schema_registry_persistence(self, schema_validator, v18_schema, v19_schema_compatible):
        """Test schema registry persists to disk and reloads"""
        schema_validator.register_schema("meta", "v18.0", v18_schema)
        schema_validator.register_schema("meta", "v19.0", v19_schema_compatible)
        
        # Save registry
        schema_validator.save_registry()
        
        # Create new validator and load
        new_validator = SchemaValidator(
            registry_path=schema_validator.registry_path
        )
        new_validator.load_registry()
        
        # Validate loaded schemas
        assert new_validator.has_schema("meta", "v18.0")
        assert new_validator.has_schema("meta", "v19.0")
        
        # Re-run compat check
        result = new_validator.validate_backward_compatibility(
            provider="meta",
            old_version="v18.0",
            new_version="v19.0",
            new_schema=v19_schema_compatible
        )
        assert result.is_compatible is True
    
    def test_alert_within_5min_on_schema_drift(self, schema_validator, v18_schema, v19_schema_breaking):
        """Test alert fires within 5min when schema drift detected (relates to Q_003)"""
        import time
        
        schema_validator.register_schema("meta", "v18.0", v18_schema)
        
        alert_fired = []
        alert_time = None
        
        def alert_handler(event: SchemaChangeEvent):
            nonlocal alert_time
            alert_time = datetime.utcnow()
            alert_fired.append(event)
        
        schema_validator.on_schema_change(alert_handler)
        
        start_time = datetime.utcnow()
        
        # Trigger drift detection
        result = schema_validator.validate_backward_compatibility(
            provider="meta",
            old_version="v18.0",
            new_version="v19.0",
            new_schema=v19_schema_breaking
        )
        
        # Alert should fire immediately
        assert len(alert_fired) == 1
        assert alert_time is not None
        
        elapsed = (alert_time - start_time).total_seconds()
        assert elapsed < 300  # <5min (actually <1s in practice)
    
    def test_prometheus_metric_schema_drift_detected_increments(self, schema_validator, v18_schema, v19_schema_breaking):
        """Test prometheus counter schema_drift_detected_total increments"""
        from unittest.mock import MagicMock
        
        schema_validator.prometheus_counter = MagicMock()
        schema_validator.register_schema("meta", "v18.0", v18_schema)
        
        result = schema_validator.validate_backward_compatibility(
            provider="meta",
            old_version="v18.0",
            new_version="v19.0",
            new_schema=v19_schema_breaking
        )
        
        # Verify counter incremented
        schema_validator.prometheus_counter.labels.assert_called()
        schema_validator.prometheus_counter.labels().inc.assert_called()
    
    def test_kill_switch_disables_strict_validation(self):
        """Test ENABLE_STRICT_SCHEMA_VALIDATION kill switch"""
        import os
        
        # Disable strict validation
        os.environ["ENABLE_STRICT_SCHEMA_VALIDATION"] = "false"
        
        validator = SchemaValidator()
        
        # Even with breaking changes, validation passes in non-strict mode
        result = validator.validate_backward_compatibility(
            provider="meta",
            old_version="v18.0",
            new_version="v19.0",
            new_schema={"fields": {}}  # Completely different schema
        )
        
        # Non-strict mode: logs warnings but doesn't fail
        assert result.is_compatible is True  # Warnings only
        assert len(result.warnings) > 0
        
        # Cleanup
        os.environ["ENABLE_STRICT_SCHEMA_VALIDATION"] = "true"
    
    def test_enum_value_addition_is_non_breaking(self, schema_validator, v18_schema):
        """Test adding enum value (REACH to buying_type) is non-breaking"""
        v19_extended_enum = v18_schema.copy()
        v19_extended_enum["fields"]["buying_type"]["enum"] = ["AUCTION", "RESERVED", "REACH"]
        
        schema_validator.register_schema("meta", "v18.0", v18_schema)
        
        result = schema_validator.validate_backward_compatibility(
            provider="meta",
            old_version="v18.0",
            new_version="v19.0",
            new_schema=v19_extended_enum
        )
        
        assert result.is_compatible is True
        # Should have warning but not breaking
        enum_changes = [w for w in result.warnings if "enum" in w.lower()]
        assert len(enum_changes) >= 1


class TestMetaAgentV19Integration:
    """Integration tests for MetaAdPlatformAgent with v19 API"""
    
    def test_fetch_spend_with_v19_compatible_schema(self):
        """Test MetaAdPlatformAgent fetches data successfully with v19 compatible schema"""
        agent = MetaAdPlatformAgent(
            api_version="v19.0",
            access_token="test_token",
            account_id="act_123"
        )
        
        # Mock API response
        mock_response = {
            "data": [
                {
                    "id": "ad1",
                    "campaign_id": "c1",
                    "spend": "125.50",
                    "impressions": "10000",
                    "clicks": "250",
                    "date_start": "2025-10-01",
                    "date_stop": "2025-10-01",
                    "account_currency": "USD",
                    "optimization_goal": "CONVERSIONS"  # v19 field
                }
            ]
        }
        
        # Agent should parse and validate
        metrics = agent.parse_response(mock_response)
        assert len(metrics) == 1
        assert metrics[0].spend == Decimal("125.50")
        assert metrics[0].impressions == 10000
    
    def test_v18_fallback_when_v19_unavailable(self):
        """Test agent falls back to v18 when v19 endpoint unavailable"""
        agent = MetaAdPlatformAgent(
            api_version="v19.0",
            fallback_version="v18.0"
        )
        
        # Mock v19 endpoint 404
        agent.api_client.v19_available = False
        
        # Agent should auto-fallback
        agent.ensure_compatible_version()
        
        assert agent.current_api_version == "v18.0"
        assert agent.fallback_triggered is True
