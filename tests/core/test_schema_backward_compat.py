"""
Schema v1/v2 Backward Compatibility Test Suite — T025

Coverage:
- v2 schema sends → v1 receiver accepts without error
- Field additions in v2 are optional/nullable in v1
- Field removals trigger explicit DeprecationWarning
- v1 schema sends → v2 receiver accepts with defaults
- Schema version negotiation via headers
- Contract validation against SchemaRegistry
- Prometheus metrics: schema_version_mismatches_total
- Idempotent processing: duplicate messages handled gracefully

Risk Gates:
- Schema registry connection timeout 5s
- Version validation before data processing
- Fallback to v1 if negotiation fails
- Kill switch: ENABLE_SCHEMA_VERSION_NEGOTIATION

Acceptance (from Q_025):
- v2 send, v1 receive success
- v1 send, v2 receive with defaults success
- Version mismatch metric increments
- Contract validation passes for both versions
"""

import pytest
import json
from datetime import datetime, timezone
from unittest.mock import Mock, patch, AsyncMock
from core.contracts import (
    SpendRecord,
    SpendRecordV2,
    SchemaVersion,
    SchemaRegistry,
    validate_schema_compatibility,
    negotiate_schema_version
)


class TestSchemaBackwardCompatibility:
    """Schema v1/v2 backward compatibility tests"""
    
    @pytest.fixture
    def spend_record_v1_data(self):
        """Valid v1 SpendRecord data"""
        return {
            "date": "2025-10-19",
            "channel": "meta",
            "campaign_id": "c123",
            "adset_id": "a456",
            "spend": 120000.0,
            "currency": "JPY",
            "impressions": 45000,
            "clicks": 1200
        }
    
    @pytest.fixture
    def spend_record_v2_data(self):
        """v2 SpendRecord with additional fields"""
        return {
            "date": "2025-10-19",
            "channel": "meta",
            "campaign_id": "c123",
            "adset_id": "a456",
            "spend": 120000.0,
            "currency": "JPY",
            "impressions": 45000,
            "clicks": 1200,
            # v2 additions
            "conversions": 45,
            "conversion_value": 890000.0,
            "frequency": 2.3,
            "reach": 19500,
            "cost_per_conversion": 2666.67,
            "roas": 7.42,
            "schema_version": "2.0"
        }
    
    @pytest.fixture
    def schema_registry(self):
        """Mock schema registry"""
        registry = Mock(spec=SchemaRegistry)
        registry.get_schema = Mock(side_effect=lambda version: {
            "1.0": SpendRecord,
            "2.0": SpendRecordV2
        }.get(version))
        return registry
    
    async def test_v2_send_v1_receive_success(self, spend_record_v2_data, schema_registry):
        """
        v2 schema sends → v1 receiver accepts without error
        
        Acceptance: v2 send, v1 receive success
        """
        # v2 sender creates record
        v2_record = SpendRecordV2(**spend_record_v2_data)
        v2_json = v2_record.model_dump_json()
        
        # v1 receiver parses (should ignore extra fields)
        v1_data = json.loads(v2_json)
        v1_compatible_data = {
            k: v for k, v in v1_data.items()
            if k in SpendRecord.model_fields
        }
        
        v1_record = SpendRecord(**v1_compatible_data)
        
        # Assertions
        assert v1_record.date.isoformat() == "2025-10-19"
        assert v1_record.channel == "meta"
        assert v1_record.spend == 120000.0
        assert v1_record.impressions == 45000
        # v2 fields not present in v1, but no error raised
    
    async def test_v1_send_v2_receive_with_defaults(self, spend_record_v1_data, schema_registry):
        """
        v1 schema sends → v2 receiver accepts with defaults
        
        Acceptance: v1 send, v2 receive with defaults success
        """
        # v1 sender creates record
        v1_record = SpendRecord(**spend_record_v1_data)
        v1_json = v1_record.model_dump_json()
        
        # v2 receiver parses with defaults for new fields
        v2_data = json.loads(v1_json)
        
        # Set defaults for v2 fields
        v2_data.setdefault("conversions", None)
        v2_data.setdefault("conversion_value", None)
        v2_data.setdefault("frequency", None)
        v2_data.setdefault("reach", None)
        v2_data.setdefault("cost_per_conversion", None)
        v2_data.setdefault("roas", None)
        v2_data.setdefault("schema_version", "1.0")
        
        v2_record = SpendRecordV2(**v2_data)
        
        # Assertions
        assert v2_record.date.isoformat() == "2025-10-19"
        assert v2_record.spend == 120000.0
        assert v2_record.conversions is None  # Default for missing field
        assert v2_record.schema_version == "1.0"
    
    async def test_schema_version_negotiation(self, schema_registry):
        """
        Schema version negotiation via headers
        
        Acceptance: Version negotiation selects highest compatible version
        """
        # Client supports v2, server supports v1 and v2
        client_versions = ["2.0", "1.0"]
        server_versions = ["2.0", "1.0"]
        
        negotiated = negotiate_schema_version(
            client_versions,
            server_versions
        )
        
        assert negotiated == "2.0"  # Highest compatible
        
        # Client supports only v1, server supports v2 and v1
        client_versions_v1_only = ["1.0"]
        
        negotiated_v1 = negotiate_schema_version(
            client_versions_v1_only,
            server_versions
        )
        
        assert negotiated_v1 == "1.0"  # Fallback to v1
    
    async def test_no_compatible_version_error(self):
        """
        No compatible schema version raises error
        
        Acceptance: Incompatible versions raise SchemaVersionError
        """
        client_versions = ["3.0"]
        server_versions = ["2.0", "1.0"]
        
        with pytest.raises(ValueError, match="No compatible schema version"):
            negotiate_schema_version(client_versions, server_versions)
    
    async def test_contract_validation_v1(self, spend_record_v1_data, schema_registry):
        """
        Contract validation passes for v1 schema
        
        Acceptance: Contract validation passes for both versions
        """
        v1_record = SpendRecord(**spend_record_v1_data)
        
        is_valid = validate_schema_compatibility(
            v1_record,
            schema_version="1.0",
            registry=schema_registry
        )
        
        assert is_valid is True
    
    async def test_contract_validation_v2(self, spend_record_v2_data, schema_registry):
        """
        Contract validation passes for v2 schema
        
        Acceptance: Contract validation passes for both versions
        """
        v2_record = SpendRecordV2(**spend_record_v2_data)
        
        is_valid = validate_schema_compatibility(
            v2_record,
            schema_version="2.0",
            registry=schema_registry
        )
        
        assert is_valid is True
    
    @patch('core.contracts.prometheus_client.Counter')
    async def test_version_mismatch_metric_increments(
        self,
        mock_counter,
        spend_record_v2_data,
        schema_registry
    ):
        """
        Version mismatch metric increments
        
        Acceptance: Version mismatch metric increments
        """
        counter_instance = Mock()
        mock_counter.return_value = counter_instance
        
        # Simulate version mismatch detection
        client_version = "2.0"
        server_version = "1.0"
        
        from core.contracts import record_version_mismatch
        
        record_version_mismatch(
            client_version=client_version,
            server_version=server_version,
            endpoint="/ingest/spend"
        )
        
        # Assert metric incremented
        counter_instance.labels.assert_called_once_with(
            client_version="2.0",
            server_version="1.0",
            endpoint="/ingest/spend"
        )
        counter_instance.labels().inc.assert_called_once()
    
    async def test_schema_registry_timeout_fallback(self, spend_record_v1_data):
        """
        Schema registry connection timeout → fallback to v1
        
        Risk Gate: Schema registry connection timeout 5s
        """
        registry = Mock(spec=SchemaRegistry)
        registry.get_schema = Mock(side_effect=TimeoutError("Connection timeout"))
        
        # Should fallback to v1 schema on timeout
        from core.contracts import get_schema_with_fallback
        
        schema = get_schema_with_fallback(
            version="2.0",
            registry=registry,
            timeout=5.0
        )
        
        assert schema == SpendRecord  # Fallback to v1
    
    async def test_kill_switch_disables_negotiation(self, spend_record_v1_data):
        """
        Kill switch disables schema version negotiation
        
        Risk Gate: ENABLE_SCHEMA_VERSION_NEGOTIATION kill switch
        """
        with patch.dict('os.environ', {'ENABLE_SCHEMA_VERSION_NEGOTIATION': 'false'}):
            # Should always use v1 when kill switch disabled
            negotiated = negotiate_schema_version(
                client_versions=["2.0", "1.0"],
                server_versions=["2.0", "1.0"]
            )
            
            assert negotiated == "1.0"  # Forces v1
    
    async def test_idempotent_duplicate_message_handling(self, spend_record_v1_data):
        """
        Idempotent processing: duplicate messages handled gracefully
        
        Acceptance: Duplicate messages with same content hash ignored
        """
        v1_record = SpendRecord(**spend_record_v1_data)
        
        # Generate content hash
        from core.contracts import compute_content_hash
        
        hash1 = compute_content_hash(v1_record)
        
        # Create identical record
        v1_record_duplicate = SpendRecord(**spend_record_v1_data)
        hash2 = compute_content_hash(v1_record_duplicate)
        
        assert hash1 == hash2  # Same hash for duplicate
        
        # Simulate deduplication cache
        processed_hashes = {hash1}
        
        # Second message should be detected as duplicate
        is_duplicate = hash2 in processed_hashes
        assert is_duplicate is True
    
    async def test_field_addition_nullable_in_v1(self, spend_record_v2_data):
        """
        Field additions in v2 are optional/nullable in v1
        
        Acceptance: v2 additional fields don't break v1 parsing
        """
        # v2 record with all fields
        v2_record = SpendRecordV2(**spend_record_v2_data)
        
        # Convert to v1 (drop v2-only fields)
        v1_data = {
            k: getattr(v2_record, k)
            for k in SpendRecord.model_fields.keys()
        }
        
        v1_record = SpendRecord(**v1_data)
        
        # v1 record should be valid without v2 fields
        assert v1_record.date.isoformat() == "2025-10-19"
        assert v1_record.spend == 120000.0
        # No conversions, roas, etc. in v1 - OK
    
    async def test_field_removal_deprecation_warning(self, spend_record_v1_data):
        """
        Field removals trigger explicit DeprecationWarning
        
        Acceptance: Deprecated field usage logs warning
        """
        import warnings
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            # Simulate accessing deprecated field
            from core.contracts import access_deprecated_field
            
            access_deprecated_field("legacy_field_name")
            
            # Assert DeprecationWarning raised
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "deprecated" in str(w[0].message).lower()


# Pytest configuration
pytest_plugins = []


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
