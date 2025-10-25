"""
JWT RBAC Authentication Middleware
Implements role-based access control with privilege escalation prevention
"""
import os
from typing import Optional
from datetime import datetime
from fastapi import HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt
from schemas.jwt_claims import RoleEnum, JWTClaims
from prometheus_client import Counter
import logging

logger = logging.getLogger(__name__)

# Security configuration
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "")  # Must be from Secret Manager in prod
JWT_ALGORITHM = "HS256"
RBAC_ENFORCEMENT = os.getenv("RBAC_ENFORCEMENT", "true").lower() == "true"

# Metrics
auth_metrics = type('AuthMetrics', (), {
    'unauthorized_attempts': Counter('mbi_unauthorized_access_attempts_total', 'Unauthorized access attempts', ['role', 'resource']),
    'invalid_role_attempts': Counter('mbi_invalid_role_attempts_total', 'Invalid role attempts', ['attempted_role'])
})()

security = HTTPBearer()


def is_rbac_enabled() -> bool:
    """
    Check if RBAC enforcement is enabled
    
    RISK GATE: Kill switch for emergency bypass
    Default: enabled (secure by default)
    """
    return RBAC_ENFORCEMENT


def validate_role(role_str: str) -> RoleEnum:
    """
    Validate role string against RoleEnum
    
    Q_107: Prevents typo escalation (e.g., 'admn' bypassing checks)
    
    Args:
        role_str: Role string from JWT
        
    Returns:
        RoleEnum value
        
    Raises:
        ValueError: If role is invalid
    """
    try:
        return RoleEnum(role_str)
    except ValueError:
        # Log invalid role attempt
        auth_metrics.invalid_role_attempts.labels(attempted_role=role_str).inc()
        logger.warning(f"Invalid role attempted: {role_str}")
        raise ValueError(f"Invalid role: {role_str}")


def validate_token(token: str) -> JWTClaims:
    """
    Validate JWT token and extract claims
    
    Args:
        token: JWT token string
        
    Returns:
        Validated JWT claims
        
    Raises:
        HTTPException: 401 if token invalid/expired
    """
    if not JWT_SECRET_KEY:
        logger.error("JWT_SECRET_KEY not configured")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication not configured"
        )
    
    try:
        # Decode and verify token
        payload = jwt.decode(
            token,
            JWT_SECRET_KEY,
            algorithms=[JWT_ALGORITHM]
        )
        
        # Validate required claims
        if 'role' not in payload:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing role claim in token"
            )
        
        # Validate role enum
        role = validate_role(payload['role'])
        
        # Create claims object
        claims = JWTClaims(
            sub=payload['sub'],
            role=role,
            exp=datetime.fromtimestamp(payload['exp']),
            iat=datetime.fromtimestamp(payload['iat'])
        )
        
        return claims
    
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token expired"
        )
    except jwt.InvalidTokenError as e:
        logger.warning(f"Invalid token: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )


def check_role_permission(token: str, required_role: str) -> JWTClaims:
    """
    Check if token has required role permission
    
    Q_036: Prevents ad-ops from accessing admin endpoints
    
    Args:
        token: JWT token string
        required_role: Required role for access
        
    Returns:
        Validated claims if authorized
        
    Raises:
        HTTPException: 403 if insufficient privileges
    """
    claims = validate_token(token)
    
    # If RBAC disabled (kill switch), allow all
    if not is_rbac_enabled():
        logger.warning("RBAC enforcement disabled - allowing access")
        return claims
    
    # Validate required role enum
    required_role_enum = validate_role(required_role)
    
    # Check permission
    if not has_permission(claims.role.value, required_role_enum.value):
        # Log unauthorized attempt
        auth_metrics.unauthorized_attempts.labels(
            role=claims.role.value,
            resource=required_role
        ).inc()
        
        logger.warning(
            f"Unauthorized access attempt: user={claims.sub} "
            f"role={claims.role.value} required={required_role}"
        )
        
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Insufficient privileges: {claims.role.value} cannot access {required_role} resources"
        )
    
    return claims


def has_permission(user_role: str, required_role: str) -> bool:
    """
    Check if user role has permission for required role
    
    CONTRACT: Role hierarchy
    admin > pii_admin > ad-ops > viewer
    
    Args:
        user_role: User's role
        required_role: Required role for resource
        
    Returns:
        True if user has permission
    """
    # Define role hierarchy (higher number = more privileges)
    hierarchy = {
        "admin": 4,
        "pii_admin": 3,
        "ad-ops": 2,
        "viewer": 1
    }
    
    # Special resource permissions
    resource_permissions = {
        "admin_panel": {"admin"},
        "pii_mapping": {"admin", "pii_admin"},
        "campaigns": {"admin", "pii_admin", "ad-ops"},
        "reports": {"admin", "pii_admin", "ad-ops", "viewer"}
    }
    
    # Check resource-based permission first
    if required_role in resource_permissions:
        return user_role in resource_permissions[required_role]
    
    # Fall back to hierarchy check
    user_level = hierarchy.get(user_role, 0)
    required_level = hierarchy.get(required_role, 999)
    
    return user_level >= required_level


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = security
) -> JWTClaims:
    """
    FastAPI dependency to get current authenticated user
    
    Usage:
        @app.get("/admin/endpoint")
        async def admin_endpoint(user: JWTClaims = Depends(get_current_user)):
            if user.role != RoleEnum.ADMIN:
                raise HTTPException(403)
    """
    token = credentials.credentials
    return validate_token(token)


async def require_role(required_role: str):
    """
    FastAPI dependency factory for role-based endpoints
    
    Usage:
        @app.get("/admin/endpoint")
        async def admin_endpoint(user: JWTClaims = Depends(require_role("admin"))):
            # Only admin can access
    """
    async def role_checker(
        credentials: HTTPAuthorizationCredentials = security
    ) -> JWTClaims:
        token = credentials.credentials
        return check_role_permission(token, required_role)
    
    return role_checker
