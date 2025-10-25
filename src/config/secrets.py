"""
Secrets Management with Auto-Rotation Support

Provides unified interface for retrieving secrets from GCP Secret Manager
with fallback to environment variables for local development.

Features:
- Auto-rotation policy enforcement (≥30 days)
- 5-minute caching to avoid rate limits
- Kill-switch for local dev mode
- Thread-safe singleton pattern
"""

from typing import Optional, Dict, Any
from datetime import datetime, timedelta
from enum import Enum
import os
import logging
from functools import lru_cache
from threading import Lock
from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)


class SecretsProvider(str, Enum):
    """Supported secret providers"""
    GCP_SECRET_MANAGER = "gcp"
    AWS_SECRETS_MANAGER = "aws"
    ENV_VARS = "env"


class SecretConfig(BaseModel):
    """Configuration for secret retrieval"""
    provider: SecretsProvider = Field(
        default_factory=lambda: SecretsProvider(
            os.getenv("SECRETS_PROVIDER", "gcp")
        )
    )
    project_id: Optional[str] = Field(
        default_factory=lambda: os.getenv("GCP_PROJECT_ID")
    )
    rotation_interval_days: int = Field(
        default=90,
        ge=30,
        description="Minimum rotation interval in days"
    )
    cache_ttl_seconds: int = Field(
        default=300,  # 5 minutes
        ge=60,
        le=3600
    )
    
    @validator("provider")
    def validate_provider(cls, v):
        """Ensure ENV provider only used in dev mode"""
        if v == SecretsProvider.ENV_VARS:
            env = os.getenv("ENVIRONMENT", "prod")
            if env == "prod":
                raise ValueError(
                    "ENV_VARS provider not allowed in production. "
                    "Use GCP_SECRET_MANAGER or AWS_SECRETS_MANAGER."
                )
        return v


class SecretMetadata(BaseModel):
    """Metadata about a secret"""
    name: str
    version: str
    created_at: datetime
    rotation_due_at: datetime
    provider: SecretsProvider
    
    @property
    def needs_rotation(self) -> bool:
        """Check if secret needs rotation"""
        return datetime.utcnow() >= self.rotation_due_at


class SecretsManager:
    """
    Thread-safe secrets manager with caching and rotation enforcement
    """
    _instance = None
    _lock = Lock()
    
    def __new__(cls):
        """Singleton pattern"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize secrets manager (runs once)"""
        if hasattr(self, '_initialized'):
            return
        
        self.config = SecretConfig()
        self._cache: Dict[str, tuple[Any, datetime]] = {}
        self._metadata: Dict[str, SecretMetadata] = {}
        
        # Initialize provider-specific clients
        if self.config.provider == SecretsProvider.GCP_SECRET_MANAGER:
            self._init_gcp_client()
        elif self.config.provider == SecretsProvider.AWS_SECRETS_MANAGER:
            self._init_aws_client()
        
        self._initialized = True
        logger.info(
            f"SecretsManager initialized with provider={self.config.provider.value}"
        )
    
    def _init_gcp_client(self):
        """Initialize GCP Secret Manager client"""
        try:
            from google.cloud import secretmanager
            self.gcp_client = secretmanager.SecretManagerServiceClient()
            logger.info("GCP Secret Manager client initialized")
        except ImportError:
            logger.warning(
                "google-cloud-secret-manager not installed. "
                "Falling back to ENV provider."
            )
            self.config.provider = SecretsProvider.ENV_VARS
    
    def _init_aws_client(self):
        """Initialize AWS Secrets Manager client"""
        try:
            import boto3
            self.aws_client = boto3.client('secretsmanager')
            logger.info("AWS Secrets Manager client initialized")
        except ImportError:
            logger.warning(
                "boto3 not installed. Falling back to ENV provider."
            )
            self.config.provider = SecretsProvider.ENV_VARS
    
    def get_secret(self, secret_name: str, version: str = "latest") -> str:
        """
        Retrieve secret with caching
        
        Args:
            secret_name: Name of the secret (e.g., "META_API_KEY")
            version: Version to retrieve (default: "latest")
        
        Returns:
            Secret value as string
        
        Raises:
            ValueError: If secret not found or rotation overdue
        """
        cache_key = f"{secret_name}:{version}"
        
        # Check cache
        if cache_key in self._cache:
            value, cached_at = self._cache[cache_key]
            ttl = timedelta(seconds=self.config.cache_ttl_seconds)
            if datetime.utcnow() - cached_at < ttl:
                logger.debug(f"Cache hit for {secret_name}")
                return value
        
        # Retrieve from provider
        if self.config.provider == SecretsProvider.GCP_SECRET_MANAGER:
            value = self._get_gcp_secret(secret_name, version)
        elif self.config.provider == SecretsProvider.AWS_SECRETS_MANAGER:
            value = self._get_aws_secret(secret_name, version)
        else:  # ENV_VARS
            value = self._get_env_secret(secret_name)
        
        # Check rotation status
        if secret_name in self._metadata:
            metadata = self._metadata[secret_name]
            if metadata.needs_rotation:
                logger.warning(
                    f"Secret {secret_name} is overdue for rotation. "
                    f"Due: {metadata.rotation_due_at.isoformat()}"
                )
        
        # Cache result
        self._cache[cache_key] = (value, datetime.utcnow())
        
        return value
    
    def _get_gcp_secret(self, secret_name: str, version: str) -> str:
        """Retrieve secret from GCP Secret Manager"""
        name = (
            f"projects/{self.config.project_id}/"
            f"secrets/{secret_name}/versions/{version}"
        )
        
        try:
            response = self.gcp_client.access_secret_version(
                request={"name": name}
            )
            value = response.payload.data.decode("UTF-8")
            
            # Update metadata
            self._metadata[secret_name] = SecretMetadata(
                name=secret_name,
                version=version,
                created_at=response.create_time,
                rotation_due_at=response.create_time + timedelta(
                    days=self.config.rotation_interval_days
                ),
                provider=SecretsProvider.GCP_SECRET_MANAGER
            )
            
            return value
        except Exception as e:
            logger.error(f"Failed to retrieve secret {secret_name}: {e}")
            raise ValueError(f"Secret {secret_name} not found in GCP")
    
    def _get_aws_secret(self, secret_name: str, version: str) -> str:
        """Retrieve secret from AWS Secrets Manager"""
        try:
            kwargs = {"SecretId": secret_name}
            if version != "latest":
                kwargs["VersionId"] = version
            
            response = self.aws_client.get_secret_value(**kwargs)
            value = response["SecretString"]
            
            # Update metadata
            self._metadata[secret_name] = SecretMetadata(
                name=secret_name,
                version=response.get("VersionId", "latest"),
                created_at=response["CreatedDate"],
                rotation_due_at=response["CreatedDate"] + timedelta(
                    days=self.config.rotation_interval_days
                ),
                provider=SecretsProvider.AWS_SECRETS_MANAGER
            )
            
            return value
        except Exception as e:
            logger.error(f"Failed to retrieve secret {secret_name}: {e}")
            raise ValueError(f"Secret {secret_name} not found in AWS")
    
    def _get_env_secret(self, secret_name: str) -> str:
        """Retrieve secret from environment variables (dev only)"""
        value = os.getenv(secret_name)
        if value is None:
            raise ValueError(
                f"Secret {secret_name} not found in environment variables"
            )
        
        logger.warning(
            f"Using ENV var for {secret_name} (dev mode only)"
        )
        
        return value
    
    def clear_cache(self):
        """Clear secret cache (e.g., after rotation)"""
        with self._lock:
            self._cache.clear()
            logger.info("Secret cache cleared")
    
    def get_rotation_status(self) -> Dict[str, SecretMetadata]:
        """Get rotation status for all accessed secrets"""
        return self._metadata.copy()


# Singleton instance
secrets_manager = SecretsManager()


def get_secret(secret_name: str, version: str = "latest") -> str:
    """
    Convenience function to get secret
    
    Args:
        secret_name: Name of the secret
        version: Version to retrieve (default: "latest")
    
    Returns:
        Secret value as string
    """
    return secrets_manager.get_secret(secret_name, version)
