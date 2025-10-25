"""
JWT RBAC Tests
Tests role-based access control and privilege escalation prevention
"""
import pytest
from unittest.mock import Mock, patch
from datetime import datetime, timedelta
import jwt
from fastapi import HTTPException, status


# Mock JWT payload structure
def create_test_jwt(role: str, user_id: str = "test_user") -> str:
    """Create test JWT with specified role"""
    payload = {
        "sub": user_id,
        "role": role,
        "exp": datetime.utcnow() + timedelta(hours=1),
        "iat": datetime.utcnow()
    }
    # Use test secret (in real code, from Secret Manager)
    secret = "test_secret_key_do_not_use_in_prod"
    return jwt.encode(payload, secret, algorithm="HS256")


class TestJWTRBAC:
    """Test JWT role-based access control"""
    
    def test_ad_ops_gets_403_on_admin_endpoint(self):
        """
        ACCEPTANCE: ad-ops role must receive 403 on admin endpoints
        Q_036: Prevents privilege escalation
        DRY RUN: Mock auth middleware without real API
        """
        from middleware.auth import check_role_permission
        
        # Mock request with ad-ops JWT
        token = create_test_jwt(role="ad-ops")
        
        # Test admin endpoint access
        with pytest.raises(HTTPException) as exc_info:
            check_role_permission(token, required_role="admin")
        
        # Verify 403 Forbidden
        assert exc_info.value.status_code == status.HTTP_403_FORBIDDEN
        assert "insufficient privileges" in str(exc_info.value.detail).lower()
    
    def test_invalid_role_rejected(self):
        """
        ACCEPTANCE: Invalid role (typo like 'admn') must be rejected
        Q_107: RoleEnum validation prevents typo escalation
        """
        from middleware.auth import validate_role
        from schemas.jwt_claims import RoleEnum
        
        # Test valid roles
        assert validate_role("admin") == RoleEnum.ADMIN
        assert validate_role("ad-ops") == RoleEnum.AD_OPS
        assert validate_role("pii_admin") == RoleEnum.PII_ADMIN
        assert validate_role("viewer") == RoleEnum.VIEWER
        
        # Test invalid role (typo)
        with pytest.raises(ValueError) as exc_info:
            validate_role("admn")  # Typo
        
        assert "invalid role" in str(exc_info.value).lower()
        
        # Test completely invalid role
        with pytest.raises(ValueError):
            validate_role("super_admin")  # Non-existent role
    
    def test_role_hierarchy_enforcement(self):
        """
        CONTRACT: Role hierarchy must be enforced
        admin > pii_admin > ad-ops > viewer
        """
        from middleware.auth import has_permission
        
        # Admin can access everything
        assert has_permission(role="admin", resource="admin_panel") == True
        assert has_permission(role="admin", resource="pii_mapping") == True
        assert has_permission(role="admin", resource="campaigns") == True
        
        # pii_admin can access PII but not admin panel
        assert has_permission(role="pii_admin", resource="pii_mapping") == True
        assert has_permission(role="pii_admin", resource="campaigns") == True
        assert has_permission(role="pii_admin", resource="admin_panel") == False
        
        # ad-ops can access campaigns but not PII or admin
        assert has_permission(role="ad-ops", resource="campaigns") == True
        assert has_permission(role="ad-ops", resource="pii_mapping") == False
        assert has_permission(role="ad-ops", resource="admin_panel") == False
        
        # viewer can only read
        assert has_permission(role="viewer", resource="campaigns") == False
        assert has_permission(role="viewer", resource="reports") == True
    
    def test_unauthorized_access_metric_logged(self):
        """
        METRIC: unauthorized_access_attempts must be logged
        """
        from middleware.auth import check_role_permission, get_auth_metrics
        
        # Reset metrics
        with patch('middleware.auth.auth_metrics') as mock_metrics:
            token = create_test_jwt(role="viewer")
            
            # Attempt unauthorized access
            try:
                check_role_permission(token, required_role="admin")
            except HTTPException:
                pass  # Expected
            
            # Verify metric increment called
            mock_metrics.unauthorized_attempts.inc.assert_called_once()
    
    def test_role_enum_prevents_sql_injection(self):
        """
        SECURITY: RoleEnum prevents SQL injection via role parameter
        """
        from schemas.jwt_claims import RoleEnum
        
        # Malicious input should fail enum validation
        malicious_inputs = [
            "admin'; DROP TABLE users; --",
            "admin OR 1=1",
            "<script>alert('xss')</script>",
            "../../etc/passwd"
        ]
        
        for malicious in malicious_inputs:
            with pytest.raises(ValueError):
                RoleEnum(malicious)


class TestJWTValidation:
    """Test JWT token validation"""
    
    def test_expired_token_rejected(self):
        """
        SECURITY: Expired tokens must be rejected
        """
        from middleware.auth import validate_token
        
        # Create expired token
        payload = {
            "sub": "test_user",
            "role": "admin",
            "exp": datetime.utcnow() - timedelta(hours=1),  # Expired
            "iat": datetime.utcnow() - timedelta(hours=2)
        }
        secret = "test_secret_key"
        expired_token = jwt.encode(payload, secret, algorithm="HS256")
        
        with pytest.raises(HTTPException) as exc_info:
            validate_token(expired_token)
        
        assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED
        assert "expired" in str(exc_info.value.detail).lower()
    
    def test_tampered_token_rejected(self):
        """
        SECURITY: Tampered tokens must be rejected
        """
        from middleware.auth import validate_token
        
        # Create valid token
        token = create_test_jwt(role="viewer")
        
        # Tamper with token (change role to admin)
        parts = token.split('.')
        # This will fail signature verification
        tampered_token = parts[0] + ".eyJyb2xlIjoiYWRtaW4ifQ." + parts[2]
        
        with pytest.raises(HTTPException) as exc_info:
            validate_token(tampered_token)
        
        assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED
    
    def test_missing_role_claim_rejected(self):
        """
        CONTRACT: JWT must contain 'role' claim
        """
        from middleware.auth import validate_token
        
        # Create token without role
        payload = {
            "sub": "test_user",
            "exp": datetime.utcnow() + timedelta(hours=1)
            # Missing 'role'
        }
        secret = "test_secret_key"
        invalid_token = jwt.encode(payload, secret, algorithm="HS256")
        
        with pytest.raises(HTTPException) as exc_info:
            validate_token(invalid_token)
        
        assert "missing role" in str(exc_info.value.detail).lower()


class TestRBACKillSwitch:
    """Test RBAC enforcement kill switch"""
    
    def test_rbac_enforcement_env_flag(self):
        """
        RISK GATE: RBAC_ENFORCEMENT env flag controls access control
        """
        import os
        
        # When RBAC disabled, all access allowed (emergency mode)
        with patch.dict(os.environ, {'RBAC_ENFORCEMENT': 'false'}):
            from middleware.auth import is_rbac_enabled
            assert is_rbac_enabled() == False
        
        # Default: RBAC enabled
        with patch.dict(os.environ, {'RBAC_ENFORCEMENT': 'true'}):
            assert is_rbac_enabled() == True
        
        # Default when not set
        with patch.dict(os.environ, {}, clear=True):
            assert is_rbac_enabled() == True  # Secure by default
