"""
API Gateway with JWT Authentication and RBAC Authorization
Component: Infra_Auth
Related: Q_019 (JWT RBAC 403 insufficient perm enforcement)

Implements:
- JWT token generation and validation
- Role-based access control (RBAC)
- Permission checking with wildcard support
- Prometheus metrics for authorization events
- 403 Forbidden for insufficient permissions
"""

import jwt
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from prometheus_client import Counter

logger = logging.getLogger(__name__)

# Prometheus metrics
rbac_authorization_total = Counter(
    'mbi_rbac_authorization_total',
    'Total RBAC authorization attempts',
    ['role', 'permission', 'result']
)


class InsufficientPermissionsError(Exception):
    """Raised when user lacks required permissions for an action"""
    
    def __init__(self, user_role: str, required_permission: str, message: str = None):
        self.status_code = 403
        self.user_role = user_role
        self.required_permission = required_permission
        self.message = message or f"User with role '{user_role}' lacks permission '{required_permission}'"
        super().__init__(self.message)


class AuthenticationError(Exception):
    """Raised when authentication fails"""
    
    def __init__(self, message: str = "Authentication failed"):
        self.status_code = 401
        self.message = message
        super().__init__(self.message)


@dataclass
class TokenPayload:
    """JWT token payload structure"""
    user_id: str
    role: str
    exp: datetime
    iat: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JWT encoding"""
        return {
            "user_id": self.user_id,
            "role": self.role,
            "exp": self.exp,
            "iat": self.iat or datetime.utcnow()
        }


class JWTAuth:
    """
    JWT Authentication handler
    
    Generates and validates JWT tokens with HS256 signature
    """
    
    def __init__(self, secret_key: str, algorithm: str = "HS256"):
        """
        Initialize JWT authentication
        
        Args:
            secret_key: Secret key for signing tokens
            algorithm: JWT algorithm (default: HS256)
        """
        self.secret_key = secret_key
        self.algorithm = algorithm
    
    def generate_token(self, payload: Dict[str, Any]) -> str:
        """
        Generate JWT token from payload
        
        Args:
            payload: Token payload containing user_id, role, exp
            
        Returns:
            JWT token string
        """
        try:
            token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
            return token
        except Exception as e:
            logger.error(f"Failed to generate JWT token: {e}")
            raise
    
    def validate_token(self, token: str) -> Dict[str, Any]:
        """
        Validate and decode JWT token
        
        Args:
            token: JWT token string
            
        Returns:
            Decoded payload
            
        Raises:
            jwt.ExpiredSignatureError: If token is expired
            jwt.InvalidSignatureError: If signature is invalid
            jwt.DecodeError: If token is malformed
        """
        try:
            decoded = jwt.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm]
            )
            return decoded
        except jwt.ExpiredSignatureError:
            logger.warning("JWT token expired")
            raise
        except jwt.InvalidSignatureError:
            logger.error("JWT token has invalid signature")
            raise
        except jwt.DecodeError as e:
            logger.error(f"JWT token decode error: {e}")
            raise
    
    def extract_token_from_header(self, headers: Dict[str, str]) -> Optional[str]:
        """
        Extract bearer token from Authorization header
        
        Args:
            headers: Request headers dictionary
            
        Returns:
            Token string or None
        """
        auth_header = headers.get("Authorization", "")
        
        if not auth_header.startswith("Bearer "):
            return None
        
        return auth_header.replace("Bearer ", "", 1)


class RBACAuthorizer:
    """
    Role-Based Access Control (RBAC) authorizer
    
    Manages role definitions and permission checking with wildcard support
    """
    
    def __init__(self, roles: Dict[str, List[str]]):
        """
        Initialize RBAC authorizer
        
        Args:
            roles: Dictionary mapping role names to list of permissions
                   e.g., {"viewer": ["read:metrics"], "admin": ["read:*", "write:*"]}
        """
        self.roles = roles
    
    def has_permission(self, role: str, required_permission: str) -> bool:
        """
        Check if role has required permission
        
        Supports wildcard permissions (e.g., "read:*" matches "read:anything")
        
        Args:
            role: User role
            required_permission: Permission to check (e.g., "read:metrics")
            
        Returns:
            True if role has permission, False otherwise
        """
        if role not in self.roles:
            logger.debug(f"Unknown role: {role}")
            return False
        
        role_permissions = self.roles[role]
        
        # Direct match
        if required_permission in role_permissions:
            return True
        
        # Wildcard match
        # e.g., "read:*" matches "read:metrics"
        permission_prefix = required_permission.split(":")[0] if ":" in required_permission else ""
        wildcard_permission = f"{permission_prefix}:*"
        
        if wildcard_permission in role_permissions:
            return True
        
        return False
    
    def get_role_permissions(self, role: str) -> List[str]:
        """
        Get all permissions for a role
        
        Args:
            role: Role name
            
        Returns:
            List of permissions
        """
        return self.roles.get(role, [])


class APIGateway:
    """
    API Gateway with JWT authentication and RBAC authorization
    
    Validates JWT tokens and enforces role-based access control
    """
    
    def __init__(self, auth: JWTAuth, rbac: RBACAuthorizer):
        """
        Initialize API Gateway
        
        Args:
            auth: JWT authentication handler
            rbac: RBAC authorizer
        """
        self.auth = auth
        self.rbac = rbac
    
    def authorize_request(
        self,
        request: Any,
        endpoint_permissions: Dict[str, Dict[str, str]]
    ) -> Dict[str, Any]:
        """
        Authorize incoming request
        
        Q_019 ACCEPTANCE: Returns 403 when viewer→admin endpoint
        
        Args:
            request: Request object with headers, path, method
            endpoint_permissions: Map of {path: {method: permission}}
                e.g., {"/api/admin/users": {"POST": "admin:users"}}
                
        Returns:
            Authorization result with status and user info
            
        Raises:
            AuthenticationError: If token is missing or invalid (401)
            InsufficientPermissionsError: If user lacks permission (403)
        """
        # Step 1: Extract and validate JWT token
        token = self.auth.extract_token_from_header(request.headers)
        
        if not token:
            raise AuthenticationError("Missing or invalid Authorization header")
        
        try:
            payload = self.auth.validate_token(token)
        except (jwt.ExpiredSignatureError, jwt.InvalidSignatureError, jwt.DecodeError) as e:
            raise AuthenticationError(f"Token validation failed: {str(e)}")
        
        user_id = payload.get("user_id")
        role = payload.get("role")
        
        if not user_id or not role:
            raise AuthenticationError("Token missing required fields (user_id, role)")
        
        # Step 2: Determine required permission for endpoint
        path = request.path
        method = request.method
        
        if path not in endpoint_permissions:
            logger.warning(f"No permissions defined for path: {path}")
            raise InsufficientPermissionsError(
                role,
                "unknown",
                f"No permissions defined for path {path}"
            )
        
        if method not in endpoint_permissions[path]:
            logger.warning(f"No permissions defined for {method} {path}")
            raise InsufficientPermissionsError(
                role,
                "unknown",
                f"No permissions defined for {method} {path}"
            )
        
        required_permission = endpoint_permissions[path][method]
        
        # Step 3: Check RBAC permission
        has_permission = self.rbac.has_permission(role, required_permission)
        
        if not has_permission:
            # Q_019 ACCEPTANCE: Return 403 for insufficient permissions
            logger.warning(
                f"Access denied: user={user_id} role={role} "
                f"required={required_permission} path={path} method={method}"
            )
            
            # Emit metric for denied access
            rbac_authorization_total.labels(
                role=role,
                permission=required_permission,
                result="denied"
            ).inc()
            
            raise InsufficientPermissionsError(
                user_role=role,
                required_permission=required_permission,
                message=f"Insufficient permissions: role '{role}' cannot access {method} {path}"
            )
        
        # Step 4: Authorization successful
        logger.info(
            f"Access granted: user={user_id} role={role} "
            f"permission={required_permission} path={path} method={method}"
        )
        
        # Emit metric for successful authorization
        rbac_authorization_total.labels(
            role=role,
            permission=required_permission,
            result="success"
        ).inc()
        
        return {
            "status": 200,
            "user_id": user_id,
            "role": role,
            "permission": required_permission,
            "authorized": True
        }


# Default role definitions for MBI system
DEFAULT_ROLES = {
    "viewer": [
        "read:metrics",
        "read:campaigns",
        "read:brand_metrics",
        "read:reports"
    ],
    "analyst": [
        "read:metrics",
        "read:campaigns",
        "read:brand_metrics",
        "read:reports",
        "write:reports",
        "read:audit_logs"
    ],
    "performance_lead": [
        "read:*",
        "write:campaigns",
        "write:budgets",
        "write:bids",
        "approve:budget_changes"
    ],
    "admin": [
        "read:*",
        "write:*",
        "delete:*",
        "admin:users",
        "admin:roles",
        "admin:system"
    ],
    "pii_admin": [
        "read:pii_mapping",
        "write:pii_mapping",
        "admin:privacy"
    ]
}


def create_default_gateway(secret_key: str) -> APIGateway:
    """
    Create API Gateway with default role definitions
    
    Args:
        secret_key: JWT secret key
        
    Returns:
        Configured APIGateway instance
    """
    jwt_auth = JWTAuth(secret_key=secret_key)
    rbac_authorizer = RBACAuthorizer(roles=DEFAULT_ROLES)
    
    return APIGateway(auth=jwt_auth, rbac=rbac_authorizer)
