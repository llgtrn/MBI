"""
Identity Resolution Data Contracts — MBI System

Schemas for privacy-safe identity matching with PII hashing and GDPR compliance.

Security & Privacy:
- PII never stored in plaintext (SHA-256 hashing only)
- TTL-based deletion (90 days default for GDPR Article 17)
- Deterministic hashing for stable matching
- Separate PII mapping table (not in these schemas)

Schema Version: 1.0.0
Last Updated: 2025-10-19
"""

from pydantic import BaseModel, Field, validator
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from enum import Enum
import hashlib


class LifecycleStageEnum(str, Enum):
    """Customer lifecycle stages"""
    PROSPECT = "prospect"
    LEAD = "lead"
    CUSTOMER = "customer"
    CHURNED = "churned"
    REACTIVATED = "reactivated"


class MatchTypeEnum(str, Enum):
    """Identity matching method"""
    DETERMINISTIC = "deterministic"  # Email/phone/customer_id exact match
    PROBABILISTIC = "probabilistic"  # Behavioral/device fingerprint
    MERGED = "merged"               # Combined from multiple signals


class IdentitySignals(BaseModel):
    """
    Input signals for identity resolution.
    
    Privacy (Q_011):
    - email_hash: SHA-256 hash of normalized email (never plaintext)
    - phone_hash: SHA-256 hash of normalized phone (never plaintext)
    - customer_id: First-party identifier (if authenticated)
    - Behavioral signals: Privacy-safe (no PII)
    """
    
    # Hashed PII (Q_011)
    email_hash: Optional[str] = Field(
        None,
        min_length=64,
        max_length=64,
        description="SHA-256 hash of normalized email (lowercase, trimmed)"
    )
    phone_hash: Optional[str] = Field(
        None,
        min_length=64,
        max_length=64,
        description="SHA-256 hash of normalized phone (E.164 format)"
    )
    
    # First-party identifiers
    customer_id: Optional[str] = Field(None, max_length=255)
    session_id: Optional[str] = Field(None, max_length=255)
    
    # Behavioral signals (privacy-safe, no PII)
    device_fingerprint: Optional[str] = Field(None, max_length=255)
    ip_hash: Optional[str] = Field(
        None,
        min_length=64,
        max_length=64,
        description="SHA-256 hash of IP address (privacy-safe)"
    )
    user_agent: Optional[str] = Field(None, max_length=2048)
    timezone: Optional[str] = Field(None, max_length=100)
    
    # Timestamps
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        schema_extra = {
            "example": {
                "email_hash": "a" * 64,
                "phone_hash": "b" * 64,
                "customer_id": "cust_12345",
                "device_fingerprint": "fp_abc123",
                "ip_hash": "c" * 64,
                "timezone": "Asia/Tokyo",
                "timestamp": "2025-10-19T10:30:00Z"
            }
        }


class UnifiedProfile(BaseModel):
    """
    Unified customer profile (SSOT).
    
    Privacy & GDPR Compliance (Q_011, Q_012):
    - user_key: SHA-256 hash (irreversible, stable identifier)
    - No plaintext PII stored
    - ttl_expires_at: GDPR Article 17 deletion date (90 days default)
    - created_at: Profile creation timestamp
    """
    
    # Primary identifier (Q_011)
    user_key: str = Field(
        ...,
        min_length=64,
        max_length=64,
        description="SHA-256 hash of deterministic match key (irreversible)"
    )
    
    # Lifecycle
    lifecycle_stage: LifecycleStageEnum = Field(default=LifecycleStageEnum.PROSPECT)
    
    # Segmentation (privacy-safe)
    segments: List[str] = Field(
        default_factory=list,
        description="Behavioral segments (e.g., high_value, engaged)"
    )
    
    # Attributes (no PII)
    country: Optional[str] = Field(None, max_length=2, description="ISO 3166-1 alpha-2")
    device: Optional[str] = Field(None, max_length=50)
    preferred_language: Optional[str] = Field(None, max_length=10)
    
    # Metrics
    ltv: Optional[float] = Field(None, ge=0, description="Lifetime value")
    order_count: int = Field(default=0, ge=0)
    last_order_date: Optional[datetime] = None
    
    # GDPR Compliance (Q_012)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    ttl_expires_at: datetime = Field(
        ...,
        description="GDPR deletion date (90 days from created_at by default)"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "user_key": "a" * 64,
                "lifecycle_stage": "customer",
                "segments": ["high_value", "engaged"],
                "country": "JP",
                "device": "mobile",
                "preferred_language": "ja",
                "ltv": 50000.0,
                "order_count": 3,
                "last_order_date": "2025-10-15T12:00:00Z",
                "created_at": "2025-08-01T10:00:00Z",
                "updated_at": "2025-10-19T10:30:00Z",
                "ttl_expires_at": "2025-10-30T10:00:00Z"
            }
        }
    
    @validator('ttl_expires_at', pre=True, always=True)
    def set_default_ttl(cls, v, values):
        """Set default TTL to 90 days from created_at if not provided"""
        if v is None and 'created_at' in values:
            return values['created_at'] + timedelta(days=90)
        return v
    
    @validator('ttl_expires_at')
    def validate_ttl_minimum(cls, v, values):
        """Ensure TTL is at least 7 days (prevent immediate deletion)"""
        if 'created_at' in values:
            min_ttl = values['created_at'] + timedelta(days=7)
            if v < min_ttl:
                raise ValueError(f"TTL must be at least 7 days from created_at: {v} < {min_ttl}")
        return v


class ResolutionResult(BaseModel):
    """
    Result of identity resolution process.
    
    Contains matched profile and metadata about the match.
    """
    
    # Matched profile
    user_key: str = Field(..., min_length=64, max_length=64)
    profile: UnifiedProfile
    
    # Match metadata
    match_type: MatchTypeEnum
    confidence: float = Field(..., ge=0.0, le=1.0, description="Match confidence score")
    
    # Matched signals (which signals contributed to the match)
    matched_on: List[str] = Field(
        ...,
        description="Signals used for matching (e.g., ['email_hash', 'customer_id'])"
    )
    
    # New profile?
    is_new_profile: bool = Field(
        default=False,
        description="True if this is a newly created profile"
    )
    
    # Timestamp
    resolved_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        schema_extra = {
            "example": {
                "user_key": "a" * 64,
                "profile": {
                    "user_key": "a" * 64,
                    "lifecycle_stage": "customer",
                    "segments": ["high_value"],
                    "country": "JP",
                    "created_at": "2025-08-01T10:00:00Z",
                    "updated_at": "2025-10-19T10:30:00Z",
                    "ttl_expires_at": "2025-10-30T10:00:00Z"
                },
                "match_type": "deterministic",
                "confidence": 1.0,
                "matched_on": ["email_hash", "customer_id"],
                "is_new_profile": False,
                "resolved_at": "2025-10-19T10:30:00Z"
            }
        }


class IdentityGraph(BaseModel):
    """
    Identity graph edge (links between identifiers).
    
    Privacy-safe representation of identifier relationships.
    """
    
    user_key: str = Field(..., min_length=64, max_length=64)
    
    # Linked identifiers (hashed)
    linked_email_hashes: List[str] = Field(default_factory=list)
    linked_phone_hashes: List[str] = Field(default_factory=list)
    linked_customer_ids: List[str] = Field(default_factory=list)
    linked_device_fingerprints: List[str] = Field(default_factory=list)
    
    # Edge metadata
    link_count: int = Field(default=0, ge=0)
    first_linked_at: datetime = Field(default_factory=datetime.utcnow)
    last_linked_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        schema_extra = {
            "example": {
                "user_key": "a" * 64,
                "linked_email_hashes": ["b" * 64, "c" * 64],
                "linked_phone_hashes": ["d" * 64],
                "linked_customer_ids": ["cust_12345"],
                "linked_device_fingerprints": ["fp_abc", "fp_xyz"],
                "link_count": 5,
                "first_linked_at": "2025-08-01T10:00:00Z",
                "last_linked_at": "2025-10-19T10:30:00Z"
            }
        }


# ============================================================================
# Helper Functions (for hash generation)
# ============================================================================

def hash_email(email: str, salt: bytes) -> str:
    """
    Hash email address for privacy-safe storage.
    
    Args:
        email: Email address (will be normalized)
        salt: Cryptographic salt for deterministic hashing
        
    Returns:
        64-character hex SHA-256 hash
    """
    # Normalize: lowercase, trim whitespace
    normalized = email.lower().strip()
    
    # Hash with salt
    salted = (normalized + salt.decode('utf-8')).encode('utf-8')
    return hashlib.sha256(salted).hexdigest()


def hash_phone(phone: str, salt: bytes) -> str:
    """
    Hash phone number for privacy-safe storage.
    
    Args:
        phone: Phone number (should be E.164 format, e.g., +819012345678)
        salt: Cryptographic salt for deterministic hashing
        
    Returns:
        64-character hex SHA-256 hash
    """
    # Normalize: remove spaces, ensure E.164 format
    normalized = phone.strip().replace(' ', '').replace('-', '')
    if not normalized.startswith('+'):
        raise ValueError(f"Phone must be in E.164 format (+country...): {phone}")
    
    # Hash with salt
    salted = (normalized + salt.decode('utf-8')).encode('utf-8')
    return hashlib.sha256(salted).hexdigest()


def hash_user_key(deterministic_match: str, salt: bytes) -> str:
    """
    Generate user_key from deterministic match signal.
    
    Args:
        deterministic_match: Unique identifier (e.g., email_hash or customer_id)
        salt: Cryptographic salt for deterministic hashing
        
    Returns:
        64-character hex SHA-256 hash (user_key)
    """
    salted = (deterministic_match + salt.decode('utf-8')).encode('utf-8')
    return hashlib.sha256(salted).hexdigest()
