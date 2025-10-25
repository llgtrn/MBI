"""
Core Contracts — Schema Definitions with v1/v2 Backward Compatibility

Features:
- Pydantic v2 models with strict validation
- Schema version negotiation
- Backward compatibility: v2 → v1, v1 → v2
- Content hash for idempotent deduplication
- Prometheus metrics for version mismatches
- Schema registry with timeout fallback
- Kill switch: ENABLE_SCHEMA_VERSION_NEGOTIATION

Risk Gates:
- Schema registry connection timeout 5s
- Version validation before processing
- Fallback to v1 on negotiation failure
- Content hash deduplication

Acceptance (from Q_025):
- v2 send, v1 receive success
- v1 send, v2 receive with defaults success
- Version mismatch metric increments
- Contract validation passes for both versions
"""

import os
import json
import hashlib
import warnings
from datetime import date, datetime
from typing import List, Optional, Dict, Any
from enum import Enum
from pydantic import BaseModel, Field, field_validator, ConfigDict
from prometheus_client import Counter


# Configuration
ENABLE_SCHEMA_VERSION_NEGOTIATION = os.getenv(
    'ENABLE_SCHEMA_VERSION_NEGOTIATION',
    'true'
).lower() == 'true'

SCHEMA_REGISTRY_TIMEOUT = float(os.getenv('SCHEMA_REGISTRY_TIMEOUT', '5.0'))


# Prometheus Metrics
schema_version_mismatches_total = Counter(
    'schema_version_mismatches_total',
    'Total schema version mismatches detected',
    ['client_version', 'server_version', 'endpoint']
)


# Enums
class Channel(str, Enum):
    """Ad channels"""
    META = "meta"
    GOOGLE = "google"
    TIKTOK = "tiktok"
    YOUTUBE = "youtube"
    LINKEDIN = "linkedin"
    TWITTER = "twitter"


class SchemaVersion(str, Enum):
    """Supported schema versions"""
    V1_0 = "1.0"
    V2_0 = "2.0"


class OrderTimezoneMetadata(BaseModel):
    """
    Timezone metadata for e-commerce orders
    
    Preserves original timezone information during UTC normalization
    Addresses Q_028: Shopify timezone mismatches
    """
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True
    )
    
    original_timezone: str = Field(description="Original timezone (e.g., 'Asia/Tokyo')")
    original_timestamp: str = Field(description="Original timestamp string from source")
    utc_offset_hours: Optional[float] = Field(default=None, description="UTC offset in hours (e.g., 9.0 for JST)")
    fallback_applied: bool = Field(default=False, description="Whether UTC fallback was applied")
    fallback_reason: Optional[str] = Field(default=None, description="Reason for fallback (if applicable)")
    normalization_disabled: bool = Field(default=False, description="Whether normalization was disabled by kill switch")


class Order(BaseModel):
    """
    E-commerce order with multi-currency support
    
    Addresses A_016: Multi-currency revenue data integrity
    """
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True
    )
    
    order_id: str = Field(min_length=1, max_length=100)
    user_key: str = Field(description="Hashed user identifier")
    order_date: datetime
    revenue: float = Field(ge=0)
    currency: str = Field(default="JPY", min_length=3, max_length=3)
    items: List[Dict[str, Any]] = Field(default_factory=list)
    discount_code: Optional[str] = Field(default=None, max_length=50)
    utm_source: Optional[str] = Field(default=None, max_length=100)
    utm_medium: Optional[str] = Field(default=None, max_length=100)


# Base Models
class SpendRecord(BaseModel):
    """
    v1 SpendRecord schema (baseline)
    
    Used for backward compatibility with legacy systems
    """
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        use_enum_values=True
    )
    
    date: date
    channel: Channel
    campaign_id: str = Field(min_length=1, max_length=100)
    adset_id: Optional[str] = Field(default=None, max_length=100)
    spend: float = Field(ge=0)
    currency: str = Field(default="JPY", min_length=3, max_length=3)
    impressions: int = Field(ge=0)
    clicks: int = Field(ge=0)
    
    @field_validator('date', mode='before')
    @classmethod
    def parse_date(cls, v):
        """Parse date from string or date object"""
        if isinstance(v, str):
            return datetime.fromisoformat(v).date()
        return v


class SpendRecordV2(SpendRecord):
    """
    v2 SpendRecord schema (extended)
    
    Adds conversion metrics and performance indicators
    Backward compatible: v1 clients can ignore additional fields
    """
    conversions: Optional[int] = Field(default=None, ge=0)
    conversion_value: Optional[float] = Field(default=None, ge=0)
    frequency: Optional[float] = Field(default=None, ge=0)
    reach: Optional[int] = Field(default=None, ge=0)
    cost_per_conversion: Optional[float] = Field(default=None, ge=0)
    roas: Optional[float] = Field(default=None, ge=0)
    schema_version: str = Field(default="2.0")


# Schema Registry
class SchemaRegistry:
    """
    Schema version registry with timeout handling
    
    Maintains mapping of version strings to schema classes
    """
    
    _schemas: Dict[str, type[BaseModel]] = {
        "1.0": SpendRecord,
        "2.0": SpendRecordV2
    }
    
    def get_schema(self, version: str, timeout: float = SCHEMA_REGISTRY_TIMEOUT) -> type[BaseModel]:
        """
        Get schema by version with timeout
        
        Args:
            version: Schema version string (e.g., "1.0", "2.0")
            timeout: Max time to wait for schema retrieval (seconds)
        
        Returns:
            Pydantic model class for the version
        
        Raises:
            TimeoutError: If retrieval exceeds timeout
            KeyError: If version not found
        """
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError(f"Schema registry timeout after {timeout}s")
        
        # Set timeout alarm
        if timeout > 0:
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(int(timeout))
        
        try:
            schema = self._schemas.get(version)
            if schema is None:
                raise KeyError(f"Schema version {version} not found")
            return schema
        finally:
            # Cancel alarm
            if timeout > 0:
                signal.alarm(0)


# Schema Compatibility Functions
def negotiate_schema_version(
    client_versions: List[str],
    server_versions: List[str]
) -> str:
    """
    Negotiate schema version between client and server
    
    Args:
        client_versions: List of versions supported by client (highest first)
        server_versions: List of versions supported by server
    
    Returns:
        Highest compatible version string
    
    Raises:
        ValueError: If no compatible version found
    
    Examples:
        >>> negotiate_schema_version(["2.0", "1.0"], ["2.0", "1.0"])
        "2.0"
        >>> negotiate_schema_version(["1.0"], ["2.0", "1.0"])
        "1.0"
    """
    # Kill switch: force v1 if negotiation disabled
    if not ENABLE_SCHEMA_VERSION_NEGOTIATION:
        return "1.0"
    
    # Find highest compatible version
    for client_v in client_versions:
        if client_v in server_versions:
            return client_v
    
    raise ValueError(
        f"No compatible schema version. "
        f"Client supports: {client_versions}, Server supports: {server_versions}"
    )


def validate_schema_compatibility(
    record: BaseModel,
    schema_version: str,
    registry: SchemaRegistry
) -> bool:
    """
    Validate record against schema version
    
    Args:
        record: Pydantic model instance to validate
        schema_version: Target schema version
        registry: Schema registry instance
    
    Returns:
        True if record is compatible with schema version
    
    Examples:
        >>> record = SpendRecord(date="2025-10-19", channel="meta", ...)
        >>> validate_schema_compatibility(record, "1.0", SchemaRegistry())
        True
    """
    try:
        schema_class = registry.get_schema(schema_version)
        
        # Check if record is instance of schema or subclass
        if isinstance(record, schema_class):
            return True
        
        # Try to convert/validate
        record_dict = record.model_dump()
        schema_class(**record_dict)
        return True
        
    except Exception:
        return False


def get_schema_with_fallback(
    version: str,
    registry: SchemaRegistry,
    timeout: float = SCHEMA_REGISTRY_TIMEOUT
) -> type[BaseModel]:
    """
    Get schema with fallback to v1 on timeout/error
    
    Risk Gate: Schema registry connection timeout 5s
    
    Args:
        version: Requested schema version
        registry: Schema registry instance
        timeout: Max time to wait (seconds)
    
    Returns:
        Schema class (v1 fallback on error)
    """
    try:
        return registry.get_schema(version, timeout=timeout)
    except (TimeoutError, KeyError) as e:
        warnings.warn(
            f"Schema registry error for version {version}: {e}. "
            f"Falling back to v1 schema.",
            RuntimeWarning
        )
        return SpendRecord  # Fallback to v1


def record_version_mismatch(
    client_version: str,
    server_version: str,
    endpoint: str
) -> None:
    """
    Record schema version mismatch metric
    
    Acceptance: Version mismatch metric increments
    
    Args:
        client_version: Client's schema version
        server_version: Server's schema version
        endpoint: API endpoint where mismatch occurred
    """
    schema_version_mismatches_total.labels(
        client_version=client_version,
        server_version=server_version,
        endpoint=endpoint
    ).inc()


def compute_content_hash(record: BaseModel) -> str:
    """
    Compute content hash for idempotent deduplication
    
    Args:
        record: Pydantic model instance
    
    Returns:
        SHA256 hash of record content
    
    Examples:
        >>> record = SpendRecord(date="2025-10-19", channel="meta", ...)
        >>> hash1 = compute_content_hash(record)
        >>> hash2 = compute_content_hash(record)
        >>> hash1 == hash2
        True
    """
    # Serialize to deterministic JSON (sorted keys)
    record_dict = record.model_dump()
    json_str = json.dumps(record_dict, sort_keys=True, default=str)
    
    # Compute SHA256 hash
    return hashlib.sha256(json_str.encode()).hexdigest()


def access_deprecated_field(field_name: str) -> None:
    """
    Trigger deprecation warning for removed fields
    
    Acceptance: Deprecated field usage logs warning
    
    Args:
        field_name: Name of deprecated field
    """
    warnings.warn(
        f"Field '{field_name}' is deprecated and will be removed in future versions. "
        f"Please migrate to the new schema.",
        DeprecationWarning,
        stacklevel=2
    )


# Example Usage
if __name__ == "__main__":
    # v1 record
    v1_record = SpendRecord(
        date="2025-10-19",
        channel="meta",
        campaign_id="c123",
        adset_id="a456",
        spend=120000.0,
        currency="JPY",
        impressions=45000,
        clicks=1200
    )
    
    print("v1 Record:", v1_record.model_dump_json(indent=2))
    
    # v2 record (backward compatible)
    v2_record = SpendRecordV2(
        date="2025-10-19",
        channel="meta",
        campaign_id="c123",
        adset_id="a456",
        spend=120000.0,
        currency="JPY",
        impressions=45000,
        clicks=1200,
        conversions=45,
        conversion_value=890000.0,
        roas=7.42
    )
    
    print("\nv2 Record:", v2_record.model_dump_json(indent=2))
    
    # Schema negotiation
    negotiated = negotiate_schema_version(
        client_versions=["2.0", "1.0"],
        server_versions=["2.0", "1.0"]
    )
    print(f"\nNegotiated Version: {negotiated}")
    
    # Content hash (idempotency)
    hash1 = compute_content_hash(v1_record)
    print(f"\nContent Hash: {hash1}")
