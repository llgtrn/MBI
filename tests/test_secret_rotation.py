"""
Tests for secrets management with rotation enforcement

Tests cover:
- Secret retrieval from GCP Secret Manager (mocked)
- Rotation policy enforcement (≥30 days)
- Caching behavior (5-minute TTL)
- Dev mode fallback to env vars
- Thread safety
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import os
from src.config.secrets import (
    SecretsManager,
    SecretConfig,
    SecretMetadata,
    SecretsProvider,
    get_secret
)


@pytest.fixture
def mock_gcp_client():
    """Mock GCP Secret Manager client"""
    mock_client = Mock()
    mock_response = Mock()
    mock_response.payload.data = b"secret_value_123"
    mock_response.create_time = datetime.utcnow() - timedelta(days=45)
    mock_client.access_secret_version.return_value = mock_response
    return mock_client


@pytest.fixture
def secrets_manager_gcp(mock_gcp_client):
    """Secrets manager with mocked GCP client"""
    with patch.dict(os.environ, {
        "SECRETS_PROVIDER": "gcp",
        "GCP_PROJECT_ID": "test-project",
        "ENVIRONMENT": "test"
    }):
        manager = SecretsManager()
        manager.gcp_client = mock_gcp_client
        manager.config.provider = SecretsProvider.GCP_SECRET_MANAGER
        manager._cache.clear()
        manager._metadata.clear()
        yield manager


@pytest.fixture
def secrets_manager_env():
    """Secrets manager with ENV provider (dev mode)"""
    with patch.dict(os.environ, {
        "SECRETS_PROVIDER": "env",
        "ENVIRONMENT": "dev",
        "TEST_SECRET": "env_secret_value"
    }):
        manager = SecretsManager()
        manager.config.provider = SecretsProvider.ENV_VARS
        manager._cache.clear()
        manager._metadata.clear()
        yield manager


class TestSecretRetrieval:
    """Test secret retrieval from different providers"""
    
    def test_retrieval_from_secret_manager(self, secrets_manager_gcp, mock_gcp_client):
        """Test successful retrieval from GCP Secret Manager"""
        secret_value = secrets_manager_gcp.get_secret("META_API_KEY")
        
        assert secret_value == "secret_value_123"
        mock_gcp_client.access_secret_version.assert_called_once()
        
        # Verify metadata created
        assert "META_API_KEY" in secrets_manager_gcp._metadata
        metadata = secrets_manager_gcp._metadata["META_API_KEY"]
        assert metadata.provider == SecretsProvider.GCP_SECRET_MANAGER
    
    def test_retrieval_from_env_vars(self, secrets_manager_env):
        """Test fallback to environment variables in dev mode"""
        secret_value = secrets_manager_env.get_secret("TEST_SECRET")
        
        assert secret_value == "env_secret_value"
    
    def test_env_vars_blocked_in_prod(self):
        """Test that ENV provider is blocked in production"""
        with patch.dict(os.environ, {
            "SECRETS_PROVIDER": "env",
            "ENVIRONMENT": "prod"
        }):
            with pytest.raises(ValueError, match="not allowed in production"):
                SecretConfig()


class TestRotationPolicy:
    """Test rotation policy enforcement"""
    
    def test_rotation_policy_enforced(self):
        """Test that rotation_interval_days must be ≥30"""
        # Valid config
        config = SecretConfig(rotation_interval_days=90)
        assert config.rotation_interval_days == 90
        
        # Invalid config (too short)
        with pytest.raises(ValueError):
            SecretConfig(rotation_interval_days=15)
    
    def test_rotation_warning_when_overdue(
        self, 
        secrets_manager_gcp, 
        mock_gcp_client,
        caplog
    ):
        """Test warning logged when secret rotation is overdue"""
        # Create metadata with overdue rotation
        old_date = datetime.utcnow() - timedelta(days=120)
        mock_response = Mock()
        mock_response.payload.data = b"old_secret"
        mock_response.create_time = old_date
        mock_gcp_client.access_secret_version.return_value = mock_response
        
        # Manually set metadata to trigger rotation check
        secrets_manager_gcp._metadata["OLD_SECRET"] = SecretMetadata(
            name="OLD_SECRET",
            version="1",
            created_at=old_date,
            rotation_due_at=old_date + timedelta(days=90),
            provider=SecretsProvider.GCP_SECRET_MANAGER
        )
        
        # Retrieve secret (should log warning)
        with caplog.at_level("WARNING"):
            secrets_manager_gcp.get_secret("OLD_SECRET")
        
        assert "overdue for rotation" in caplog.text


class TestCaching:
    """Test caching behavior"""
    
    def test_cache_hit_within_ttl(self, secrets_manager_gcp, mock_gcp_client):
        """Test that cache is used within TTL period"""
        # First call
        secrets_manager_gcp.get_secret("CACHED_SECRET")
        assert mock_gcp_client.access_secret_version.call_count == 1
        
        # Second call within TTL (should use cache)
        secrets_manager_gcp.get_secret("CACHED_SECRET")
        assert mock_gcp_client.access_secret_version.call_count == 1
    
    def test_cache_miss_after_ttl(self, secrets_manager_gcp, mock_gcp_client):
        """Test that cache expires after TTL"""
        # First call
        secrets_manager_gcp.get_secret("TTL_SECRET")
        assert mock_gcp_client.access_secret_version.call_count == 1
        
        # Manually expire cache
        cache_key = "TTL_SECRET:latest"
        if cache_key in secrets_manager_gcp._cache:
            value, _ = secrets_manager_gcp._cache[cache_key]
            expired_time = datetime.utcnow() - timedelta(seconds=400)
            secrets_manager_gcp._cache[cache_key] = (value, expired_time)
        
        # Second call after TTL (should fetch again)
        secrets_manager_gcp.get_secret("TTL_SECRET")
        assert mock_gcp_client.access_secret_version.call_count == 2
    
    def test_clear_cache(self, secrets_manager_gcp):
        """Test cache clearing"""
        secrets_manager_gcp.get_secret("CLEAR_TEST")
        assert len(secrets_manager_gcp._cache) > 0
        
        secrets_manager_gcp.clear_cache()
        assert len(secrets_manager_gcp._cache) == 0


class TestErrorHandling:
    """Test error handling"""
    
    def test_secret_not_found(self, secrets_manager_gcp, mock_gcp_client):
        """Test error when secret doesn't exist"""
        mock_gcp_client.access_secret_version.side_effect = Exception(
            "Secret not found"
        )
        
        with pytest.raises(ValueError, match="not found in GCP"):
            secrets_manager_gcp.get_secret("NONEXISTENT_SECRET")
    
    def test_env_var_not_found(self, secrets_manager_env):
        """Test error when env var doesn't exist"""
        with pytest.raises(ValueError, match="not found in environment"):
            secrets_manager_env.get_secret("NONEXISTENT_ENV_VAR")


class TestThreadSafety:
    """Test thread safety of singleton pattern"""
    
    def test_singleton_pattern(self):
        """Test that only one instance is created"""
        with patch.dict(os.environ, {
            "SECRETS_PROVIDER": "env",
            "ENVIRONMENT": "dev"
        }):
            manager1 = SecretsManager()
            manager2 = SecretsManager()
            
            assert manager1 is manager2


class TestConvenienceFunction:
    """Test convenience function"""
    
    def test_get_secret_function(self, secrets_manager_env):
        """Test global get_secret function"""
        # Replace global instance
        import src.config.secrets as secrets_module
        original_manager = secrets_module.secrets_manager
        secrets_module.secrets_manager = secrets_manager_env
        
        try:
            value = get_secret("TEST_SECRET")
            assert value == "env_secret_value"
        finally:
            # Restore
            secrets_module.secrets_manager = original_manager


# Integration test scenario
class TestIntegrationScenario:
    """End-to-end test scenario"""
    
    def test_full_workflow(self, secrets_manager_gcp, mock_gcp_client):
        """Test complete workflow: retrieve, cache, rotation check"""
        # 1. Retrieve secret
        secret = secrets_manager_gcp.get_secret("WORKFLOW_SECRET")
        assert secret == "secret_value_123"
        
        # 2. Verify cached
        assert "WORKFLOW_SECRET:latest" in secrets_manager_gcp._cache
        
        # 3. Check rotation status
        status = secrets_manager_gcp.get_rotation_status()
        assert "WORKFLOW_SECRET" in status
        
        # 4. Clear cache
        secrets_manager_gcp.clear_cache()
        assert len(secrets_manager_gcp._cache) == 0
