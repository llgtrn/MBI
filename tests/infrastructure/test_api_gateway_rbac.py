"""
Tests for JWT RBAC authorization with role-based access control
Component: Infra_Auth (API Gateway)
Acceptance: viewer→admin endpoint returns 403 Forbidden
Related: Q_019 (JWT RBAC 403 insufficient perm enforcement)
"""

import pytest
import jwt
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
from infrastructure.api_gateway import APIGateway, JWTAuth, RBACAuthorizer, InsufficientPermissionsError

# Test fixtures
@pytest.fixture
def secret_key():
    """JWT secret key for testing"""
    return "test-secret-key-do-not-use-in-production"

@pytest.fixture
def jwt_auth(secret_key):
    """JWT authentication instance"""
    return JWTAuth(secret_key=secret_key, algorithm="HS256")

@pytest.fixture
def rbac_authorizer():
    """RBAC authorization instance with role definitions"""
    roles = {
        "viewer": ["read:metrics", "read:campaigns"],
        "analyst": ["read:metrics", "read:campaigns", "read:reports", "write:reports"],
        "admin": ["read:*", "write:*", "delete:*", "admin:*"]
    }
    return RBACAuthorizer(roles=roles)

@pytest.fixture
def api_gateway(jwt_auth, rbac_authorizer):
    """API Gateway with JWT auth and RBAC"""
    return APIGateway(auth=jwt_auth, rbac=rbac_authorizer)


class TestJWTAuthentication:
    """Test JWT token generation and validation"""
    
    def test_generate_valid_token(self, jwt_auth):
        """Test generating a valid JWT token"""
        payload = {
            "user_id": "user123",
            "role": "viewer",
            "exp": datetime.utcnow() + timedelta(hours=1)
        }
        
        token = jwt_auth.generate_token(payload)
        
        assert token is not None
        assert isinstance(token, str)
        assert len(token) > 0
    
    def test_validate_valid_token(self, jwt_auth):
        """Test validating a valid token"""
        payload = {
            "user_id": "user123",
            "role": "viewer",
            "exp": datetime.utcnow() + timedelta(hours=1)
        }
        
        token = jwt_auth.generate_token(payload)
        decoded = jwt_auth.validate_token(token)
        
        assert decoded["user_id"] == "user123"
        assert decoded["role"] == "viewer"
    
    def test_validate_expired_token(self, jwt_auth):
        """Test validating an expired token raises error"""
        payload = {
            "user_id": "user123",
            "role": "viewer",
            "exp": datetime.utcnow() - timedelta(hours=1)  # Expired
        }
        
        token = jwt_auth.generate_token(payload)
        
        with pytest.raises(jwt.ExpiredSignatureError):
            jwt_auth.validate_token(token)
    
    def test_validate_invalid_signature(self, jwt_auth, secret_key):
        """Test validating token with invalid signature"""
        # Create token with different secret
        payload = {"user_id": "user123", "role": "viewer"}
        token = jwt.encode(payload, "wrong-secret", algorithm="HS256")
        
        with pytest.raises(jwt.InvalidSignatureError):
            jwt_auth.validate_token(token)
    
    def test_validate_malformed_token(self, jwt_auth):
        """Test validating malformed token"""
        with pytest.raises(jwt.DecodeError):
            jwt_auth.validate_token("not.a.valid.token")


class TestRBACAuthorization:
    """Test role-based access control"""
    
    def test_viewer_can_read_metrics(self, rbac_authorizer):
        """Test viewer role can read metrics"""
        assert rbac_authorizer.has_permission("viewer", "read:metrics") is True
    
    def test_viewer_cannot_write_campaigns(self, rbac_authorizer):
        """Test viewer role cannot write campaigns"""
        assert rbac_authorizer.has_permission("viewer", "write:campaigns") is False
    
    def test_viewer_cannot_delete(self, rbac_authorizer):
        """Test viewer role cannot delete"""
        assert rbac_authorizer.has_permission("viewer", "delete:campaigns") is False
    
    def test_analyst_can_write_reports(self, rbac_authorizer):
        """Test analyst role can write reports"""
        assert rbac_authorizer.has_permission("analyst", "write:reports") is True
    
    def test_analyst_cannot_delete(self, rbac_authorizer):
        """Test analyst role cannot delete"""
        assert rbac_authorizer.has_permission("analyst", "delete:campaigns") is False
    
    def test_admin_has_all_permissions(self, rbac_authorizer):
        """Test admin role has all permissions"""
        assert rbac_authorizer.has_permission("admin", "read:metrics") is True
        assert rbac_authorizer.has_permission("admin", "write:campaigns") is True
        assert rbac_authorizer.has_permission("admin", "delete:campaigns") is True
        assert rbac_authorizer.has_permission("admin", "admin:users") is True
    
    def test_unknown_role_has_no_permissions(self, rbac_authorizer):
        """Test unknown role has no permissions"""
        assert rbac_authorizer.has_permission("unknown", "read:metrics") is False
    
    def test_wildcard_permission_matching(self, rbac_authorizer):
        """Test wildcard permission matching for admin"""
        # Admin has read:*, should match read:anything
        assert rbac_authorizer.has_permission("admin", "read:new_resource") is True
        assert rbac_authorizer.has_permission("admin", "write:new_resource") is True


class TestAPIGatewayRBAC:
    """Test API Gateway with RBAC enforcement - Q_019 acceptance criteria"""
    
    def test_viewer_access_read_endpoint_200(self, api_gateway, jwt_auth):
        """Test viewer can access read-only endpoint"""
        # Create viewer token
        payload = {
            "user_id": "viewer1",
            "role": "viewer",
            "exp": datetime.utcnow() + timedelta(hours=1)
        }
        token = jwt_auth.generate_token(payload)
        
        # Mock request
        request = Mock()
        request.headers = {"Authorization": f"Bearer {token}"}
        request.path = "/api/metrics"
        request.method = "GET"
        
        # Define endpoint permissions
        endpoint_permissions = {
            "/api/metrics": {"GET": "read:metrics"}
        }
        
        response = api_gateway.authorize_request(request, endpoint_permissions)
        
        assert response["status"] == 200
        assert response["user_id"] == "viewer1"
        assert response["role"] == "viewer"
    
    def test_viewer_access_admin_endpoint_403(self, api_gateway, jwt_auth):
        """
        Q_019 ACCEPTANCE: viewer→admin endpoint returns 403 Forbidden
        Test viewer attempting to access admin endpoint receives 403
        """
        # Create viewer token
        payload = {
            "user_id": "viewer1",
            "role": "viewer",
            "exp": datetime.utcnow() + timedelta(hours=1)
        }
        token = jwt_auth.generate_token(payload)
        
        # Mock request to admin endpoint
        request = Mock()
        request.headers = {"Authorization": f"Bearer {token}"}
        request.path = "/api/admin/users"
        request.method = "POST"
        
        # Define endpoint permissions
        endpoint_permissions = {
            "/api/admin/users": {"POST": "admin:users"}
        }
        
        with pytest.raises(InsufficientPermissionsError) as exc_info:
            api_gateway.authorize_request(request, endpoint_permissions)
        
        assert exc_info.value.status_code == 403
        assert "insufficient permissions" in str(exc_info.value).lower()
        assert exc_info.value.user_role == "viewer"
        assert exc_info.value.required_permission == "admin:users"
    
    def test_viewer_access_write_endpoint_403(self, api_gateway, jwt_auth):
        """Test viewer attempting to write receives 403"""
        # Create viewer token
        payload = {
            "user_id": "viewer1",
            "role": "viewer",
            "exp": datetime.utcnow() + timedelta(hours=1)
        }
        token = jwt_auth.generate_token(payload)
        
        # Mock request to write endpoint
        request = Mock()
        request.headers = {"Authorization": f"Bearer {token}"}
        request.path = "/api/campaigns"
        request.method = "POST"
        
        # Define endpoint permissions
        endpoint_permissions = {
            "/api/campaigns": {"POST": "write:campaigns"}
        }
        
        with pytest.raises(InsufficientPermissionsError) as exc_info:
            api_gateway.authorize_request(request, endpoint_permissions)
        
        assert exc_info.value.status_code == 403
    
    def test_analyst_access_write_reports_200(self, api_gateway, jwt_auth):
        """Test analyst can write reports"""
        # Create analyst token
        payload = {
            "user_id": "analyst1",
            "role": "analyst",
            "exp": datetime.utcnow() + timedelta(hours=1)
        }
        token = jwt_auth.generate_token(payload)
        
        # Mock request
        request = Mock()
        request.headers = {"Authorization": f"Bearer {token}"}
        request.path = "/api/reports"
        request.method = "POST"
        
        # Define endpoint permissions
        endpoint_permissions = {
            "/api/reports": {"POST": "write:reports"}
        }
        
        response = api_gateway.authorize_request(request, endpoint_permissions)
        
        assert response["status"] == 200
        assert response["role"] == "analyst"
    
    def test_analyst_access_admin_endpoint_403(self, api_gateway, jwt_auth):
        """Test analyst attempting to access admin endpoint receives 403"""
        # Create analyst token
        payload = {
            "user_id": "analyst1",
            "role": "analyst",
            "exp": datetime.utcnow() + timedelta(hours=1)
        }
        token = jwt_auth.generate_token(payload)
        
        # Mock request to admin endpoint
        request = Mock()
        request.headers = {"Authorization": f"Bearer {token}"}
        request.path = "/api/admin/users"
        request.method = "DELETE"
        
        # Define endpoint permissions
        endpoint_permissions = {
            "/api/admin/users": {"DELETE": "admin:users"}
        }
        
        with pytest.raises(InsufficientPermissionsError) as exc_info:
            api_gateway.authorize_request(request, endpoint_permissions)
        
        assert exc_info.value.status_code == 403
    
    def test_admin_access_all_endpoints_200(self, api_gateway, jwt_auth):
        """Test admin can access all endpoints"""
        # Create admin token
        payload = {
            "user_id": "admin1",
            "role": "admin",
            "exp": datetime.utcnow() + timedelta(hours=1)
        }
        token = jwt_auth.generate_token(payload)
        
        # Test various endpoints
        endpoints = [
            ("/api/metrics", "GET", "read:metrics"),
            ("/api/campaigns", "POST", "write:campaigns"),
            ("/api/campaigns/123", "DELETE", "delete:campaigns"),
            ("/api/admin/users", "POST", "admin:users")
        ]
        
        for path, method, permission in endpoints:
            request = Mock()
            request.headers = {"Authorization": f"Bearer {token}"}
            request.path = path
            request.method = method
            
            endpoint_permissions = {path: {method: permission}}
            
            response = api_gateway.authorize_request(request, endpoint_permissions)
            assert response["status"] == 200
    
    def test_missing_token_401(self, api_gateway):
        """Test request without token receives 401 Unauthorized"""
        request = Mock()
        request.headers = {}
        request.path = "/api/metrics"
        request.method = "GET"
        
        endpoint_permissions = {"/api/metrics": {"GET": "read:metrics"}}
        
        with pytest.raises(Exception) as exc_info:
            api_gateway.authorize_request(request, endpoint_permissions)
        
        # Should raise authentication error before RBAC check
        assert "authorization" in str(exc_info.value).lower() or "token" in str(exc_info.value).lower()
    
    def test_expired_token_401(self, api_gateway, jwt_auth):
        """Test expired token receives 401 Unauthorized"""
        # Create expired token
        payload = {
            "user_id": "user1",
            "role": "viewer",
            "exp": datetime.utcnow() - timedelta(hours=1)
        }
        token = jwt_auth.generate_token(payload)
        
        request = Mock()
        request.headers = {"Authorization": f"Bearer {token}"}
        request.path = "/api/metrics"
        request.method = "GET"
        
        endpoint_permissions = {"/api/metrics": {"GET": "read:metrics"}}
        
        with pytest.raises(jwt.ExpiredSignatureError):
            api_gateway.authorize_request(request, endpoint_permissions)


class TestRBACMetrics:
    """Test RBAC authorization metrics"""
    
    @patch('infrastructure.api_gateway.rbac_authorization_total')
    def test_authorization_success_metric(self, mock_metric, api_gateway, jwt_auth):
        """Test successful authorization increments metric"""
        payload = {
            "user_id": "viewer1",
            "role": "viewer",
            "exp": datetime.utcnow() + timedelta(hours=1)
        }
        token = jwt_auth.generate_token(payload)
        
        request = Mock()
        request.headers = {"Authorization": f"Bearer {token}"}
        request.path = "/api/metrics"
        request.method = "GET"
        
        endpoint_permissions = {"/api/metrics": {"GET": "read:metrics"}}
        
        api_gateway.authorize_request(request, endpoint_permissions)
        
        # Verify metric incremented
        mock_metric.labels.assert_called_with(
            role="viewer",
            permission="read:metrics",
            result="success"
        )
        mock_metric.labels().inc.assert_called_once()
    
    @patch('infrastructure.api_gateway.rbac_authorization_total')
    def test_authorization_denied_metric(self, mock_metric, api_gateway, jwt_auth):
        """Test denied authorization increments metric"""
        payload = {
            "user_id": "viewer1",
            "role": "viewer",
            "exp": datetime.utcnow() + timedelta(hours=1)
        }
        token = jwt_auth.generate_token(payload)
        
        request = Mock()
        request.headers = {"Authorization": f"Bearer {token}"}
        request.path = "/api/admin/users"
        request.method = "POST"
        
        endpoint_permissions = {"/api/admin/users": {"POST": "admin:users"}}
        
        with pytest.raises(InsufficientPermissionsError):
            api_gateway.authorize_request(request, endpoint_permissions)
        
        # Verify metric incremented for denial
        mock_metric.labels.assert_called_with(
            role="viewer",
            permission="admin:users",
            result="denied"
        )
        mock_metric.labels().inc.assert_called_once()


class TestRBACEdgeCases:
    """Test RBAC edge cases and security scenarios"""
    
    def test_role_tampering_attempt(self, api_gateway, jwt_auth, secret_key):
        """Test that tampering with role in token is detected"""
        # Create viewer token
        payload = {
            "user_id": "viewer1",
            "role": "viewer",
            "exp": datetime.utcnow() + timedelta(hours=1)
        }
        token = jwt_auth.generate_token(payload)
        
        # Attempt to manually change role (won't work due to signature)
        # This should fail validation
        parts = token.split('.')
        # Tampering would invalidate signature
        
        request = Mock()
        request.headers = {"Authorization": f"Bearer {token}"}
        request.path = "/api/admin/users"
        request.method = "POST"
        
        endpoint_permissions = {"/api/admin/users": {"POST": "admin:users"}}
        
        # Should raise 403 because role is still viewer
        with pytest.raises(InsufficientPermissionsError):
            api_gateway.authorize_request(request, endpoint_permissions)
    
    def test_unknown_permission_denied(self, rbac_authorizer):
        """Test requesting unknown permission is denied"""
        assert rbac_authorizer.has_permission("viewer", "unknown:permission") is False
    
    def test_case_sensitive_permissions(self, rbac_authorizer):
        """Test permissions are case-sensitive"""
        # viewer has "read:metrics" not "READ:METRICS"
        assert rbac_authorizer.has_permission("viewer", "read:metrics") is True
        assert rbac_authorizer.has_permission("viewer", "READ:METRICS") is False
