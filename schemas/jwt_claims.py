"""
JWT Claims Schema
Defines JWT token structure and role enum
"""
from enum import Enum
from datetime import datetime
from pydantic import BaseModel, Field


class RoleEnum(str, Enum):
    """
    Role enumeration for RBAC
    
    Q_107: Prevents typo escalation (e.g., 'admn' bypassing checks)
    
    Hierarchy (highest to lowest):
    - admin: Full system access
    - pii_admin: PII mapping access + campaign management
    - ad-ops: Campaign management only
    - viewer: Read-only access
    """
    ADMIN = "admin"
    PII_ADMIN = "pii_admin"
    AD_OPS = "ad-ops"
    VIEWER = "viewer"


class JWTClaims(BaseModel):
    """
    JWT token claims structure
    
    CONTRACT: All JWTs must contain these claims
    """
    sub: str = Field(..., description="Subject (user ID)")
    role: RoleEnum = Field(..., description="User role for RBAC")
    exp: datetime = Field(..., description="Expiration timestamp")
    iat: datetime = Field(..., description="Issued at timestamp")
    
    class Config:
        use_enum_values = False  # Keep enum type for validation
