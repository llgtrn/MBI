"""
Privacy Configuration Schema for GDPR Compliance

Defines privacy settings with validation for GDPR Article 32 requirements.
Related: Q_002, A_002
"""

from pydantic import BaseModel, Field, validator
from typing import Literal


class PrivacyConfig(BaseModel):
    """
    Privacy configuration for identity resolution.
    
    Enforces GDPR Article 32 requirements:
    - Salt rotation every ≥30 days
    - TTL enforcement for hashed identifiers
    - Secure hash algorithms
    """
    
    salt_rotation_days: int = Field(
        default=90,
        ge=30,
        description="Days between salt rotations (minimum 30 for GDPR compliance)"
    )
    
    ttl_days: int = Field(
        default=90,
        ge=30,
        description="TTL for hashed user identifiers in days"
    )
    
    hash_algorithm: Literal['sha256', 'sha512'] = Field(
        default='sha256',
        description="Cryptographic hash algorithm for PII"
    )
    
    min_entropy_bits: int = Field(
        default=256,
        ge=256,
        description="Minimum entropy in bits for salt generation"
    )
    
    allowed_pii_fields: set[str] = Field(
        default={'email', 'phone', 'customer_id'},
        description="Permitted PII fields for identity resolution"
    )
    
    audit_retention_days: int = Field(
        default=365,
        ge=365,
        description="Audit log retention period (minimum 1 year for compliance)"
    )
    
    @validator('salt_rotation_days')
    def validate_rotation_days(cls, v):
        """Enforce minimum 30-day rotation for GDPR compliance"""
        if v < 30:
            raise ValueError("Salt rotation must be ≥ 30 days for GDPR Article 32 compliance")
        return v
    
    @validator('ttl_days')
    def validate_ttl(cls, v):
        """Enforce minimum TTL"""
        if v < 30:
            raise ValueError("TTL must be ≥ 30 days for data minimization")
        return v
    
    class Config:
        frozen = True  # Immutable after creation
        schema_extra = {
            "example": {
                "salt_rotation_days": 90,
                "ttl_days": 90,
                "hash_algorithm": "sha256",
                "min_entropy_bits": 256,
                "allowed_pii_fields": ["email", "phone", "customer_id"],
                "audit_retention_days": 365
            }
        }
