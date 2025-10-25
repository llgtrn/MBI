"""Tests for Secret Manager Integration

Validates:
- Secret retrieval from GCP Secret Manager (mocked)
- Env var fallback for local dev
- Rotation policy enforcement (≥30 days)
- Caching behavior (5-min TTL)
- Kill-switch provider selection
"""

import os
import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

from core.secrets import (
    SecretsConfig,
    get_secret,
    get_default_config,
    CACHE_TTL_SECONDS
)


class TestSecretsConfig:
    """Test SecretsConfig initialization and validation"""
    
    def test_rotation_interval_minimum_30_days(self):
        """Rotation interval must be ≥ 30 days for compliance"""
        # Should raise for < 30 days
        with pytest.raises(ValueError, match="rotation_interval_days must be ≥ 30"):
            SecretsConfig(rotation_interval_days=29)
        
        # Should succeed for ≥ 30 days
        config = SecretsConfig(rotation_interval_days=30)
        assert config.rotation_interval_days == 30
        
        config = SecretsConfig(rotation_interval_days=90)
        assert config.rotation_interval_days == 90
    
    def test_gcp_provider_requires_project_id(self):
        """GCP Secret Manager provider requires project_id"""
        # Clear env var if set
        original = os.environ.pop('GCP_PROJECT_ID', None)
        
        try:
            with pytest.raises(ValueError, match="GCP_PROJECT_ID required"):
                SecretsConfig(provider='gcp_secret_manager', project_id=None)
            
            # Should succeed with project_id
            config = SecretsConfig(
                provider='gcp_secret_manager',
                project_id='test-project-123'
            )
            assert config.project_id == 'test-project-123'
        
        finally:
            if original:
                os.environ['GCP_PROJECT_ID'] = original
    
    def test_env_provider_initialization(self):
        """Env provider should initialize without project_id"""
        config = SecretsConfig(provider='env')
        assert config.provider == 'env'
        # project_id can be None for env provider
        assert config.rotation_interval_days == 90  # default


class TestSecretRetrieval:
    """Test secret retrieval from different providers"""
    
    @patch('core.secrets.secretmanager.SecretManagerServiceClient')
    def test_retrieval_from_gcp_secret_manager(self, mock_sm_client):
        """Should retrieve secret from GCP Secret Manager and cache it"""
        # Mock GCP Secret Manager response
        mock_client = Mock()
        mock_response = Mock()
        mock_response.payload.data = b'secret-value-123'
        mock_client.access_secret_version.return_value = mock_response
        mock_sm_client.return_value = mock_client
        
        config = SecretsConfig(
            provider='gcp_secret_manager',
            project_id='test-project'
        )
        
        # First call: should hit GCP API
        value = config.get_secret('meta-api-key')
        assert value == 'secret-value-123'
        
        # Verify API was called with correct path
        mock_client.access_secret_version.assert_called_once()
        call_args = mock_client.access_secret_version.call_args
        expected_name = 'projects/test-project/secrets/meta-api-key/versions/latest'
        assert call_args[1]['request']['name'] == expected_name
        
        # Second call: should use cache (no new API call)
        mock_client.reset_mock()
        value2 = config.get_secret('meta-api-key')
        assert value2 == 'secret-value-123'
        mock_client.access_secret_version.assert_not_called()
    
    def test_retrieval_from_env_fallback(self, monkeypatch):
        """Should retrieve from env vars when provider='env'"""
        monkeypatch.setenv('META_API_KEY', 'env-secret-456')
        
        config = SecretsConfig(provider='env')
        value = config.get_secret('meta-api-key')
        
        assert value == 'env-secret-456'
    
    def test_env_fallback_raises_on_missing_var(self):
        """Should raise ValueError if env var not found"""
        # Ensure env var doesn't exist
        if 'NONEXISTENT_SECRET' in os.environ:
            del os.environ['NONEXISTENT_SECRET']
        
        config = SecretsConfig(provider='env')
        
        with pytest.raises(ValueError, match="not found in environment"):
            config.get_secret('nonexistent-secret')
    
    def test_unknown_provider_raises(self):
        """Should raise for unknown provider"""
        config = SecretsConfig.__new__(SecretsConfig)
        config.provider = 'invalid-provider'
        
        with pytest.raises(ValueError, match="Unknown secrets provider"):
            config.get_secret('test-secret')


class TestRotationPolicy:
    """Test rotation policy enforcement"""
    
    @patch('core.secrets.secretmanager.SecretManagerServiceClient')
    def test_rotation_policy_enforced(self, mock_sm_client):
        """Should verify secret age is within rotation policy"""
        mock_client = Mock()
        
        # Mock secret created 60 days ago
        mock_version = Mock()
        mock_version.create_time = datetime.utcnow() - timedelta(days=60)
        mock_client.access_secret_version.return_value = mock_version
        mock_sm_client.return_value = mock_client
        
        config = SecretsConfig(
            provider='gcp_secret_manager',
            project_id='test-project',
            rotation_interval_days=90
        )
        
        # Should be compliant (60 < 90)
        is_compliant = config.verify_rotation_policy('meta-api-key')
        assert is_compliant is True
        
        # Mock secret created 100 days ago
        mock_version.create_time = datetime.utcnow() - timedelta(days=100)
        
        # Should be non-compliant (100 > 90)
        is_compliant = config.verify_rotation_policy('meta-api-key')
        assert is_compliant is False
    
    def test_rotation_policy_skipped_for_env_provider(self):
        """Rotation policy check should skip for env provider"""
        config = SecretsConfig(provider='env')
        
        # Should return True (skip check) and log warning
        is_compliant = config.verify_rotation_policy('meta-api-key')
        assert is_compliant is True


class TestCaching:
    """Test secret caching behavior"""
    
    @patch('core.secrets.secretmanager.SecretManagerServiceClient')
    def test_cache_ttl_5_minutes(self, mock_sm_client):
        """Cached secrets should expire after 5 minutes"""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.payload.data = b'cached-secret'
        mock_client.access_secret_version.return_value = mock_response
        mock_sm_client.return_value = mock_client
        
        config = SecretsConfig(
            provider='gcp_secret_manager',
            project_id='test-project'
        )
        
        # First call: cache miss
        value1 = config.get_secret('test-secret')
        assert mock_client.access_secret_version.call_count == 1
        
        # Second call within TTL: cache hit
        value2 = config.get_secret('test-secret')
        assert mock_client.access_secret_version.call_count == 1  # No new call
        assert value1 == value2
        
        # Simulate TTL expiry by clearing cache
        from core.secrets import _secret_cache
        _secret_cache._cache.clear()
        
        # Third call after expiry: cache miss
        value3 = config.get_secret('test-secret')
        assert mock_client.access_secret_version.call_count == 2  # New call
        assert value3 == 'cached-secret'


class TestKillSwitch:
    """Test kill-switch provider selection via env var"""
    
    def test_kill_switch_env_var(self, monkeypatch):
        """SECRETS_PROVIDER env var should control backend"""
        # Test GCP provider
        monkeypatch.setenv('SECRETS_PROVIDER', 'gcp_secret_manager')
        monkeypatch.setenv('GCP_PROJECT_ID', 'test-project')
        
        config = SecretsConfig()
        assert config.provider == 'gcp_secret_manager'
        
        # Test env fallback
        monkeypatch.setenv('SECRETS_PROVIDER', 'env')
        config = SecretsConfig()
        assert config.provider == 'env'
    
    def test_default_provider_gcp(self, monkeypatch):
        """Default provider should be gcp_secret_manager if not set"""
        monkeypatch.delenv('SECRETS_PROVIDER', raising=False)
        monkeypatch.setenv('GCP_PROJECT_ID', 'test-project')
        
        config = SecretsConfig()
        assert config.provider == 'gcp_secret_manager'


class TestConvenienceFunctions:
    """Test module-level convenience functions"""
    
    def test_get_default_config_singleton(self):
        """get_default_config should return singleton"""
        from core.secrets import _default_config
        
        # Reset singleton
        import core.secrets as secrets_module
        secrets_module._default_config = None
        
        config1 = get_default_config()
        config2 = get_default_config()
        
        assert config1 is config2  # Same instance
    
    @patch('core.secrets.get_default_config')
    def test_get_secret_convenience(self, mock_get_config):
        """get_secret() should use default config"""
        mock_config = Mock()
        mock_config.get_secret.return_value = 'convenience-secret'
        mock_get_config.return_value = mock_config
        
        value = get_secret('test-secret')
        
        assert value == 'convenience-secret'
        mock_config.get_secret.assert_called_once_with('test-secret', 'latest')


# Integration test (requires real GCP credentials - skip in CI)
@pytest.mark.skip(reason="Requires GCP credentials")
class TestGCPIntegration:
    """Integration tests with real GCP Secret Manager"""
    
    def test_real_gcp_secret_retrieval(self):
        """End-to-end test with real GCP (manual verification only)"""
        config = SecretsConfig(
            provider='gcp_secret_manager',
            project_id=os.getenv('GCP_PROJECT_ID'),
            rotation_interval_days=90
        )
        
        # This would retrieve a real secret
        # value = config.get_secret('test-secret-name')
        # assert len(value) > 0
        pass
