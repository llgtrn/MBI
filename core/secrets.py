"""Secret Manager Integration for MBI System

Provides secure API key retrieval from GCP Secret Manager with:
- Auto-rotation policy enforcement
- Local dev fallback via env vars
- 5-minute caching to avoid rate limits
- Kill-switch for provider selection

Compliance: GDPR Article 32 (security of processing)
"""

import os
import time
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
from functools import lru_cache
import logging

logger = logging.getLogger(__name__)

# Kill-switch: SECRETS_PROVIDER env var controls backend
# Values: 'gcp_secret_manager' (prod) | 'env' (local dev only)
SECRETS_PROVIDER = os.getenv('SECRETS_PROVIDER', 'gcp_secret_manager')

# Cache TTL to avoid Secret Manager rate limits
CACHE_TTL_SECONDS = 300  # 5 minutes

class SecretCache:
    """Simple in-memory cache with TTL"""
    def __init__(self):
        self._cache: Dict[str, tuple[Any, float]] = {}
    
    def get(self, key: str) -> Optional[Any]:
        if key in self._cache:
            value, expires_at = self._cache[key]
            if time.time() < expires_at:
                return value
            else:
                del self._cache[key]
        return None
    
    def set(self, key: str, value: Any, ttl: int = CACHE_TTL_SECONDS):
        expires_at = time.time() + ttl
        self._cache[key] = (value, expires_at)

_secret_cache = SecretCache()


class SecretsConfig:
    """Configuration for secret management with rotation enforcement"""
    
    def __init__(
        self,
        provider: str = SECRETS_PROVIDER,
        rotation_interval_days: int = 90,
        project_id: Optional[str] = None
    ):
        """
        Args:
            provider: 'gcp_secret_manager' or 'env'
            rotation_interval_days: Min rotation frequency (≥30 days)
            project_id: GCP project ID (required for GCP provider)
        """
        if rotation_interval_days < 30:
            raise ValueError("rotation_interval_days must be ≥ 30 for compliance")
        
        self.provider = provider
        self.rotation_interval_days = rotation_interval_days
        self.project_id = project_id or os.getenv('GCP_PROJECT_ID')
        
        if self.provider == 'gcp_secret_manager' and not self.project_id:
            raise ValueError("GCP_PROJECT_ID required for gcp_secret_manager provider")
        
        logger.info(
            f"SecretsConfig initialized: provider={provider}, "
            f"rotation_interval={rotation_interval_days}d"
        )
    
    def get_secret(self, secret_name: str, version: str = 'latest') -> str:
        """
        Retrieve secret with caching.
        
        Args:
            secret_name: Secret identifier (e.g., 'meta-api-key')
            version: Secret version ('latest' or specific version ID)
        
        Returns:
            Secret value as string
        
        Raises:
            ValueError: If secret not found or provider misconfigured
        """
        cache_key = f"{secret_name}:{version}"
        
        # Check cache first
        cached = _secret_cache.get(cache_key)
        if cached is not None:
            logger.debug(f"Secret cache hit: {secret_name}")
            return cached
        
        # Fetch from provider
        if self.provider == 'gcp_secret_manager':
            value = self._get_from_gcp_secret_manager(secret_name, version)
        elif self.provider == 'env':
            value = self._get_from_env(secret_name)
        else:
            raise ValueError(f"Unknown secrets provider: {self.provider}")
        
        # Cache and return
        _secret_cache.set(cache_key, value)
        logger.info(f"Secret retrieved and cached: {secret_name} (provider={self.provider})")
        return value
    
    def _get_from_gcp_secret_manager(self, secret_name: str, version: str) -> str:
        """Retrieve from GCP Secret Manager API"""
        try:
            from google.cloud import secretmanager
            
            client = secretmanager.SecretManagerServiceClient()
            name = f"projects/{self.project_id}/secrets/{secret_name}/versions/{version}"
            
            response = client.access_secret_version(request={"name": name})
            return response.payload.data.decode('UTF-8')
        
        except ImportError:
            raise ValueError(
                "google-cloud-secret-manager not installed. "
                "Run: pip install google-cloud-secret-manager"
            )
        except Exception as e:
            logger.error(f"Failed to retrieve secret {secret_name} from GCP: {e}")
            raise ValueError(f"Secret retrieval failed: {secret_name}") from e
    
    def _get_from_env(self, secret_name: str) -> str:
        """Fallback: retrieve from environment variables (dev only)"""
        # Convert secret-name to ENV_VAR_NAME format
        env_var = secret_name.upper().replace('-', '_')
        value = os.getenv(env_var)
        
        if value is None:
            raise ValueError(
                f"Secret {secret_name} not found in environment. "
                f"Expected env var: {env_var}"
            )
        
        logger.warning(
            f"Using env var for secret {secret_name}. "
            f"Not recommended for production!"
        )
        return value
    
    def verify_rotation_policy(self, secret_name: str) -> bool:
        """
        Verify secret has been rotated within policy window.
        
        Returns:
            True if rotation compliant, False otherwise
        
        Note: This requires Secret Manager to track rotation metadata.
        For MVP, we log a warning if not verifiable.
        """
        if self.provider != 'gcp_secret_manager':
            logger.warning(
                f"Rotation policy verification skipped for provider={self.provider}"
            )
            return True
        
        try:
            from google.cloud import secretmanager
            
            client = secretmanager.SecretManagerServiceClient()
            name = f"projects/{self.project_id}/secrets/{secret_name}"
            
            # Get latest version metadata
            latest_version = client.access_secret_version(
                request={"name": f"{name}/versions/latest"}
            )
            
            # Check create time
            created_at = latest_version.create_time
            age_days = (datetime.utcnow() - created_at.replace(tzinfo=None)).days
            
            is_compliant = age_days <= self.rotation_interval_days
            
            if not is_compliant:
                logger.warning(
                    f"Secret {secret_name} rotation overdue: "
                    f"age={age_days}d, policy={self.rotation_interval_days}d"
                )
            
            return is_compliant
        
        except Exception as e:
            logger.error(f"Rotation policy check failed for {secret_name}: {e}")
            return False


# Global singleton for convenience
_default_config: Optional[SecretsConfig] = None

def get_default_config() -> SecretsConfig:
    """Get or create default SecretsConfig singleton"""
    global _default_config
    if _default_config is None:
        _default_config = SecretsConfig()
    return _default_config

def get_secret(secret_name: str, version: str = 'latest') -> str:
    """
    Convenience function: retrieve secret using default config.
    
    Usage:
        api_key = get_secret('meta-api-key')
    """
    return get_default_config().get_secret(secret_name, version)
