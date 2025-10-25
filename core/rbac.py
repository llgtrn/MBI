"""
RBAC (Role-Based Access Control) Manager for MBI System
Implements least-privilege access control with PII mapping restrictions (Q_020)

Key Features:
- Role-based permission model with explicit PII mapping permissions
- Audit logging for all access decisions (especially denials)
- Prometheus metrics for security monitoring
- Emergency readonly mode support
- No permission escalation
"""
from __future__ import annotations
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Any
from datetime import datetime, UTC
import logging
from prometheus_client import Counter

logger = logging.getLogger(__name__)


class Permission(str, Enum):
    """System permissions with explicit PII separation"""
    # Data access (hashed user_key only)
    DATA_READ = "data:read"
    DATA_WRITE = "data:write"
    
    # PII access (plaintext email/phone via reverse hash)
    PII_READ = "pii:read"
    PII_WRITE = "pii:write"
    
    # PII MAPPING access (user_key <-> email/phone reverse lookup)
    # CRITICAL: Only pii_admin role should have these
    PII_MAPPING_READ = "pii_mapping:read"
    PII_MAPPING_WRITE = "pii_mapping:write"
    
    # Analytics & reporting
    METRICS_READ = "metrics:read"
    DASHBOARD_READ = "dashboard:read"
    REPORT_READ = "report:read"
    REPORT_WRITE = "report:write"
    
    # Feature store
    FEATURE_STORE_READ = "feature_store:read"
    FEATURE_STORE_WRITE = "feature_store:write"
    
    # Campaign management
    CAMPAIGN_READ = "campaign:read"
    CAMPAIGN_WRITE = "campaign:write"
    CAMPAIGN_APPROVE = "campaign:approve"
    
    # Admin
    ADMIN_USER_MANAGE = "admin:user:manage"
    ADMIN_ROLE_MANAGE = "admin:role:manage"
    ADMIN_EMERGENCY_ACCESS = "admin:emergency:access"


@dataclass
class Role:
    """Role definition with associated permissions"""
    name: str
    permissions: List[Permission]
    description: str = ""
    max_query_rows: Optional[int] = None
    allowed_ip_ranges: List[str] = field(default_factory=list)
    

@dataclass
class AccessDecision:
    """Result of access authorization check"""
    allowed: bool
    user_id: str
    user_role: str
    resource_type: str
    action: str
    reason: str
    timestamp: datetime
    http_status: int = 200  # 200 if allowed, 403 if denied
    

class AccessDenied(Exception):
    """Raised when RBAC denies access to a resource"""
    def __init__(self, message: str, user_id: str, resource: str, 
                 action: str, required_permission: str, http_status: int = 403):
        super().__init__(message)
        self.user_id = user_id
        self.resource = resource
        self.action = action
        self.required_permission = required_permission
        self.http_status = http_status


class RBACManager:
    """
    RBAC Manager with PII mapping access control
    
    Security Principles:
    - Least privilege: minimal permissions per role
    - Defense in depth: app-level + DB-level (RLS/ACLs)
    - Audit everything: log all access decisions
    - Fail closed: deny by default
    """
    
    def __init__(self):
        self._roles: Dict[str, Role] = {}
        self._emergency_readonly = False
        self._audit_log: List[Dict[str, Any]] = []
        
        # Prometheus metrics
        self._access_denied_counter = Counter(
            'rbac_access_denied_total',
            'Total number of RBAC access denials',
            ['resource', 'action', 'role']
        )
        
        self._access_granted_counter = Counter(
            'rbac_access_granted_total',
            'Total number of RBAC access grants',
            ['resource', 'action', 'role']
        )
        
        # Initialize default roles
        self._init_default_roles()
    
    def _init_default_roles(self):
        """Initialize standard system roles"""
        # PII Admin: ONLY role with PII mapping access
        self.define_role(Role(
            name="pii_admin",
            permissions=[
                Permission.PII_READ,
                Permission.PII_WRITE,
                Permission.PII_MAPPING_READ,
                Permission.PII_MAPPING_WRITE,
                Permission.ADMIN_EMERGENCY_ACCESS
            ],
            description="Restricted role for PII access and reverse mapping lookups",
            max_query_rows=1000,
            allowed_ip_ranges=["10.0.0.0/8"]  # Internal only
        ))
        
        # Analyst: metrics/dashboards but NO PII
        self.define_role(Role(
            name="analyst",
            permissions=[
                Permission.METRICS_READ,
                Permission.DASHBOARD_READ,
                Permission.REPORT_READ,
                Permission.DATA_READ  # Can read hashed user_key
            ],
            description="Analytics team with read-only access to aggregated metrics",
            max_query_rows=1_000_000
        ))
        
        # Data Ops: feature store + hashed data but NO PII mapping
        self.define_role(Role(
            name="data_ops",
            permissions=[
                Permission.DATA_READ,
                Permission.DATA_WRITE,
                Permission.FEATURE_STORE_READ,
                Permission.FEATURE_STORE_WRITE,
                Permission.METRICS_READ
            ],
            description="Data engineering team with feature store access",
            max_query_rows=10_000_000
        ))
        
        # Campaign Manager: campaign CRUD but NO PII
        self.define_role(Role(
            name="campaign_manager",
            permissions=[
                Permission.CAMPAIGN_READ,
                Permission.CAMPAIGN_WRITE,
                Permission.METRICS_READ,
                Permission.REPORT_READ
            ],
            description="Marketing team managing campaigns",
            max_query_rows=100_000
        ))
        
        # Admin: all permissions including role management
        self.define_role(Role(
            name="admin",
            permissions=list(Permission),
            description="System administrators with full access",
            allowed_ip_ranges=["10.0.0.0/8"]
        ))
    
    def define_role(self, role: Role):
        """Register a role with its permissions"""
        self._roles[role.name] = role
        logger.info(f"RBAC role defined: {role.name} with {len(role.permissions)} permissions")
    
    def get_role_permissions(self, role_name: str) -> Set[Permission]:
        """Get all permissions for a role"""
        if role_name not in self._roles:
            return set()
        return set(self._roles[role_name].permissions)
    
    def check_permission(
        self,
        user_context: Dict[str, Any],
        permission: Permission
    ) -> bool:
        """
        Check if user has a specific permission
        
        Args:
            user_context: {"user_id": "...", "role": "..."}
            permission: Permission to check
        
        Returns:
            True if user's role has the permission, False otherwise
        """
        role_name = user_context.get("role")
        if not role_name or role_name not in self._roles:
            return False
        
        return permission in self._roles[role_name].permissions
    
    def authorize_resource_access(
        self,
        user_context: Dict[str, Any],
        resource_type: str,
        action: str
    ) -> AccessDecision:
        """
        Authorize access to a resource
        
        Args:
            user_context: {"user_id": "...", "role": "...", "ip": "..."}
            resource_type: e.g., "pii_mapping_table", "campaign", "metrics"
            action: "read" or "write"
        
        Returns:
            AccessDecision object
        
        Raises:
            AccessDenied if user does not have required permission
        """
        user_id = user_context.get("user_id", "unknown")
        role_name = user_context.get("role", "unknown")
        user_ip = user_context.get("ip")
        
        # Map resource + action to required permission
        required_permission = self._get_required_permission(resource_type, action)
        
        # Check emergency readonly mode
        if self._emergency_readonly and action == "write":
            self._log_access_denied(user_id, role_name, resource_type, action, 
                                    "Emergency readonly mode active")
            raise AccessDenied(
                f"Access denied: Emergency readonly mode active. No writes allowed.",
                user_id=user_id,
                resource=resource_type,
                action=action,
                required_permission=str(required_permission)
            )
        
        # Check role exists
        if role_name not in self._roles:
            self._log_access_denied(user_id, role_name, resource_type, action,
                                    f"Unknown role: {role_name}")
            raise AccessDenied(
                f"Access denied: Unknown role '{role_name}'",
                user_id=user_id,
                resource=resource_type,
                action=action,
                required_permission=str(required_permission)
            )
        
        role = self._roles[role_name]
        
        # Check IP allowlist if defined
        if role.allowed_ip_ranges and user_ip:
            if not self._ip_in_ranges(user_ip, role.allowed_ip_ranges):
                self._log_access_denied(user_id, role_name, resource_type, action,
                                        f"IP {user_ip} not in allowed ranges")
                raise AccessDenied(
                    f"Access denied: IP {user_ip} not in allowed ranges for role {role_name}",
                    user_id=user_id,
                    resource=resource_type,
                    action=action,
                    required_permission=str(required_permission)
                )
        
        # Check permission
        if required_permission not in role.permissions:
            self._log_access_denied(user_id, role_name, resource_type, action,
                                    f"Missing permission: {required_permission}")
            raise AccessDenied(
                f"Access denied: Role '{role_name}' lacks permission '{required_permission}' "
                f"for {action} on {resource_type}",
                user_id=user_id,
                resource=resource_type,
                action=action,
                required_permission=str(required_permission)
            )
        
        # Access granted
        self._log_access_granted(user_id, role_name, resource_type, action)
        
        return AccessDecision(
            allowed=True,
            user_id=user_id,
            user_role=role_name,
            resource_type=resource_type,
            action=action,
            reason=f"Permission {required_permission} granted to role {role_name}",
            timestamp=datetime.now(UTC),
            http_status=200
        )
    
    def grant_permission(
        self,
        user_context: Dict[str, Any],
        target_user: str,
        permission: Permission
    ):
        """
        Grant permission to a user (admin only)
        
        Prevents permission escalation: users cannot grant permissions to themselves
        """
        user_id = user_context.get("user_id")
        role_name = user_context.get("role")
        
        # Check requester has admin permission
        if not self.check_permission(user_context, Permission.ADMIN_ROLE_MANAGE):
            raise AccessDenied(
                "Access denied: Only admins can grant permissions",
                user_id=user_id,
                resource="permissions",
                action="grant",
                required_permission=str(Permission.ADMIN_ROLE_MANAGE)
            )
        
        # Prevent self-escalation
        if user_id == target_user:
            raise AccessDenied(
                "Permission escalation not allowed: Cannot grant permissions to yourself",
                user_id=user_id,
                resource="permissions",
                action="grant",
                required_permission=str(Permission.ADMIN_ROLE_MANAGE)
            )
        
        # In production, this would modify user-role mappings in database
        logger.warning(f"RBAC: {user_id} granted {permission} to {target_user}")
    
    def enable_emergency_readonly(self):
        """Enable emergency readonly mode - blocks ALL writes"""
        self._emergency_readonly = True
        logger.critical("EMERGENCY READONLY MODE ENABLED - All writes blocked")
    
    def disable_emergency_readonly(self):
        """Disable emergency readonly mode"""
        self._emergency_readonly = False
        logger.warning("Emergency readonly mode disabled - Writes allowed")
    
    def get_recent_audit_log(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent audit log entries"""
        return self._audit_log[-limit:]
    
    def get_metric_value(self, metric_name: str, labels: Dict[str, str]) -> float:
        """Get current value of a Prometheus metric"""
        # In production, query Prometheus API
        # For testing, track internally
        if metric_name == "rbac_access_denied_total":
            return self._access_denied_counter.labels(**labels)._value.get()
        elif metric_name == "rbac_access_granted_total":
            return self._access_granted_counter.labels(**labels)._value.get()
        return 0.0
    
    def _get_required_permission(self, resource_type: str, action: str) -> Permission:
        """Map resource type + action to required permission"""
        # PII mapping requires special permissions
        if resource_type == "pii_mapping_table":
            return Permission.PII_MAPPING_READ if action == "read" else Permission.PII_MAPPING_WRITE
        
        # Other PII resources
        if "pii" in resource_type.lower():
            return Permission.PII_READ if action == "read" else Permission.PII_WRITE
        
        # Campaign resources
        if "campaign" in resource_type.lower():
            return Permission.CAMPAIGN_READ if action == "read" else Permission.CAMPAIGN_WRITE
        
        # Feature store
        if "feature" in resource_type.lower():
            return Permission.FEATURE_STORE_READ if action == "read" else Permission.FEATURE_STORE_WRITE
        
        # Metrics/dashboards
        if resource_type in ["metrics", "dashboard", "report"]:
            return Permission.METRICS_READ if action == "read" else Permission.REPORT_WRITE
        
        # Default to data permissions
        return Permission.DATA_READ if action == "read" else Permission.DATA_WRITE
    
    def _ip_in_ranges(self, ip: str, ranges: List[str]) -> bool:
        """Check if IP is in allowed CIDR ranges"""
        # Simplified implementation - in production use ipaddress module
        return True  # Placeholder
    
    def _log_access_denied(self, user_id: str, role: str, resource: str, 
                          action: str, reason: str):
        """Log access denial for audit and emit metric"""
        log_entry = {
            "event_type": "access_denied",
            "timestamp": datetime.now(UTC).isoformat(),
            "user_id": user_id,
            "role": role,
            "resource_type": resource,
            "action": action,
            "reason": reason
        }
        self._audit_log.append(log_entry)
        
        # Emit Prometheus metric
        self._access_denied_counter.labels(
            resource=resource,
            action=action,
            role=role
        ).inc()
        
        logger.warning(f"RBAC DENIED: {user_id} ({role}) attempted {action} on {resource}: {reason}")
    
    def _log_access_granted(self, user_id: str, role: str, resource: str, action: str):
        """Log access grant for audit and emit metric"""
        log_entry = {
            "event_type": "access_granted",
            "timestamp": datetime.now(UTC).isoformat(),
            "user_id": user_id,
            "role": role,
            "resource_type": resource,
            "action": action
        }
        self._audit_log.append(log_entry)
        
        # Emit Prometheus metric
        self._access_granted_counter.labels(
            resource=resource,
            action=action,
            role=role
        ).inc()
        
        logger.info(f"RBAC GRANTED: {user_id} ({role}) {action} on {resource}")
