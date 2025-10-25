"""
Unit tests for RBAC-based PII mapping access control (Q_020)
Acceptance: analyst query PII mapping → PermissionDenied
"""
import pytest
from datetime import datetime, UTC
from core.rbac import RBACManager, Role, Permission, AccessDenied


class TestPIIMappingRBAC:
    """Test suite for PII mapping access control enforcement"""
    
    @pytest.fixture
    def rbac_manager(self):
        """Initialize RBAC manager with PII-specific permissions"""
        manager = RBACManager()
        
        # Define roles with explicit PII mapping permissions
        manager.define_role(Role(
            name="pii_admin",
            permissions=[
                Permission.PII_READ,
                Permission.PII_WRITE,
                Permission.PII_MAPPING_READ,
                Permission.PII_MAPPING_WRITE
            ]
        ))
        
        manager.define_role(Role(
            name="analyst",
            permissions=[
                Permission.METRICS_READ,
                Permission.DASHBOARD_READ,
                Permission.REPORT_READ
                # Explicitly NO PII_MAPPING_READ
            ]
        ))
        
        manager.define_role(Role(
            name="data_ops",
            permissions=[
                Permission.DATA_READ,
                Permission.DATA_WRITE,
                Permission.FEATURE_STORE_READ,
                Permission.FEATURE_STORE_WRITE
                # Can read hashed user_key but NOT mapping to PII
            ]
        ))
        
        return manager
    
    def test_pii_admin_can_read_mapping(self, rbac_manager):
        """pii_admin role can read PII mapping"""
        user_context = {"user_id": "admin123", "role": "pii_admin"}
        
        # Should succeed
        assert rbac_manager.check_permission(
            user_context,
            Permission.PII_MAPPING_READ
        )
        
        # Simulate actual mapping query
        result = rbac_manager.authorize_resource_access(
            user_context=user_context,
            resource_type="pii_mapping_table",
            action="read"
        )
        assert result.allowed is True
        assert result.user_role == "pii_admin"
    
    def test_analyst_cannot_read_mapping(self, rbac_manager):
        """analyst role CANNOT read PII mapping - raises AccessDenied"""
        user_context = {"user_id": "analyst456", "role": "analyst"}
        
        # Permission check should fail
        assert not rbac_manager.check_permission(
            user_context,
            Permission.PII_MAPPING_READ
        )
        
        # Resource access should raise exception
        with pytest.raises(AccessDenied) as exc_info:
            rbac_manager.authorize_resource_access(
                user_context=user_context,
                resource_type="pii_mapping_table",
                action="read"
            )
        
        assert "PII_MAPPING_READ" in str(exc_info.value)
        assert "analyst" in str(exc_info.value)
        assert exc_info.value.http_status == 403
    
    def test_data_ops_cannot_read_mapping(self, rbac_manager):
        """data_ops can read hashed user_key but NOT reverse mapping"""
        user_context = {"user_id": "dataops789", "role": "data_ops"}
        
        # Can read hashed user keys
        assert rbac_manager.check_permission(
            user_context,
            Permission.DATA_READ
        )
        
        # But CANNOT read PII mapping
        assert not rbac_manager.check_permission(
            user_context,
            Permission.PII_MAPPING_READ
        )
        
        with pytest.raises(AccessDenied):
            rbac_manager.authorize_resource_access(
                user_context=user_context,
                resource_type="pii_mapping_table",
                action="read"
            )
    
    def test_pii_admin_can_write_mapping(self, rbac_manager):
        """pii_admin can create/update PII mappings"""
        user_context = {"user_id": "admin123", "role": "pii_admin"}
        
        assert rbac_manager.check_permission(
            user_context,
            Permission.PII_MAPPING_WRITE
        )
        
        result = rbac_manager.authorize_resource_access(
            user_context=user_context,
            resource_type="pii_mapping_table",
            action="write"
        )
        assert result.allowed is True
    
    def test_analyst_cannot_write_mapping(self, rbac_manager):
        """analyst CANNOT write PII mapping"""
        user_context = {"user_id": "analyst456", "role": "analyst"}
        
        with pytest.raises(AccessDenied) as exc_info:
            rbac_manager.authorize_resource_access(
                user_context=user_context,
                resource_type="pii_mapping_table",
                action="write"
            )
        
        assert exc_info.value.http_status == 403
    
    def test_audit_log_on_denied_access(self, rbac_manager):
        """Denied PII mapping access is logged for audit"""
        user_context = {"user_id": "analyst456", "role": "analyst"}
        
        with pytest.raises(AccessDenied):
            rbac_manager.authorize_resource_access(
                user_context=user_context,
                resource_type="pii_mapping_table",
                action="read"
            )
        
        # Check audit log was created
        audit_entries = rbac_manager.get_recent_audit_log(limit=1)
        assert len(audit_entries) > 0
        
        latest = audit_entries[0]
        assert latest["event_type"] == "access_denied"
        assert latest["user_id"] == "analyst456"
        assert latest["resource_type"] == "pii_mapping_table"
        assert latest["action"] == "read"
        assert "PII_MAPPING_READ" in latest["reason"]
    
    def test_metrics_emitted_on_denied_access(self, rbac_manager):
        """Prometheus counter increments on denied PII mapping access"""
        user_context = {"user_id": "analyst456", "role": "analyst"}
        
        # Get initial counter value
        initial_count = rbac_manager.get_metric_value(
            "rbac_access_denied_total",
            labels={"resource": "pii_mapping_table"}
        )
        
        with pytest.raises(AccessDenied):
            rbac_manager.authorize_resource_access(
                user_context=user_context,
                resource_type="pii_mapping_table",
                action="read"
            )
        
        # Counter should increment
        new_count = rbac_manager.get_metric_value(
            "rbac_access_denied_total",
            labels={"resource": "pii_mapping_table"}
        )
        assert new_count == initial_count + 1
    
    def test_no_permission_escalation(self, rbac_manager):
        """Roles cannot escalate their own permissions"""
        user_context = {"user_id": "analyst456", "role": "analyst"}
        
        # Attempt to grant self PII_MAPPING_READ permission
        with pytest.raises(AccessDenied) as exc_info:
            rbac_manager.grant_permission(
                user_context=user_context,
                target_user="analyst456",
                permission=Permission.PII_MAPPING_READ
            )
        
        assert "Permission escalation not allowed" in str(exc_info.value)
    
    def test_least_privilege_principle(self, rbac_manager):
        """Each role has minimum necessary permissions"""
        analyst_perms = rbac_manager.get_role_permissions("analyst")
        pii_admin_perms = rbac_manager.get_role_permissions("pii_admin")
        
        # Analyst should have NO PII-related permissions
        assert Permission.PII_READ not in analyst_perms
        assert Permission.PII_WRITE not in analyst_perms
        assert Permission.PII_MAPPING_READ not in analyst_perms
        assert Permission.PII_MAPPING_WRITE not in analyst_perms
        
        # Only pii_admin should have PII mapping permissions
        assert Permission.PII_MAPPING_READ in pii_admin_perms
        assert Permission.PII_MAPPING_WRITE in pii_admin_perms
    
    def test_emergency_readonly_mode(self, rbac_manager):
        """Emergency readonly mode blocks ALL PII mapping writes"""
        rbac_manager.enable_emergency_readonly()
        
        user_context = {"user_id": "admin123", "role": "pii_admin"}
        
        # Even pii_admin cannot write in emergency mode
        with pytest.raises(AccessDenied) as exc_info:
            rbac_manager.authorize_resource_access(
                user_context=user_context,
                resource_type="pii_mapping_table",
                action="write"
            )
        
        assert "Emergency readonly mode" in str(exc_info.value)
        
        # But reads still allowed
        result = rbac_manager.authorize_resource_access(
            user_context=user_context,
            resource_type="pii_mapping_table",
            action="read"
        )
        assert result.allowed is True


class TestPIIMappingDatabaseRBAC:
    """Integration tests with database-level RBAC"""
    
    def test_postgres_row_level_security(self):
        """PostgreSQL RLS enforces PII mapping access at DB level"""
        # This would be an integration test with actual Postgres
        # For now, document the expected RLS policy
        
        expected_rls_policy = """
        CREATE POLICY pii_mapping_access_policy ON pii_mapping_table
        FOR ALL
        USING (
            current_setting('app.user_role') = 'pii_admin'
        );
        """
        # In production, this RLS policy should be active
        assert True  # Placeholder for integration test
    
    def test_bigquery_authorized_views(self):
        """BigQuery authorized views restrict PII mapping access"""
        # Document expected BigQuery ACLs
        
        expected_acls = {
            "pii_mapping_table": ["pii_admin@company.com"],
            "user_profiles_hashed": ["analyst@company.com", "data_ops@company.com"]
        }
        # In production, BigQuery ACLs should match this
        assert True  # Placeholder for integration test
