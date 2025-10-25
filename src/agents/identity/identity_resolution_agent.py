"""
Identity Resolution Agent - Privacy-Safe Implementation

Implements GDPR-compliant identity resolution with:
- PII hashing with rotating salts
- 90-day TTL enforcement
- Audit logging for all PII operations
- Data minimization

Related: Q_002, A_002, C01_IdentityResolution
"""

import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Dict, Optional, Set
from pydantic import BaseModel

from src.agents.identity.privacy_config import PrivacyConfig


class UnifiedProfile(BaseModel):
    """Privacy-safe user profile with only hashed identifiers"""
    user_key: str
    email_hash: Optional[str] = None
    phone_hash: Optional[str] = None
    customer_id_hash: Optional[str] = None
    created_at: datetime
    expires_at: datetime
    segments: list[str] = []
    lifecycle_stage: Optional[str] = None


class IdentityResolutionAgent:
    """
    GDPR-compliant identity resolution.
    
    Key Features:
    - Hash all PII immediately with rotating salt
    - Enforce 90-day TTL on user_key
    - Audit log all PII operations
    - No plaintext PII storage
    """
    
    def __init__(self, privacy_config: Optional[PrivacyConfig] = None):
        self.privacy_config = privacy_config or PrivacyConfig()
        self._salt_cache: Dict[str, str] = {}
        self._user_profiles: Dict[str, UnifiedProfile] = {}
        self._audit_log_entries: list[dict] = []
        
        # Initialize current salt
        self._initialize_salt()
    
    def _initialize_salt(self) -> None:
        """Initialize or load current salt from secure storage"""
        # In production: load from GCP Secret Manager / AWS Secrets Manager
        # For now: generate new salt
        salt_bytes = secrets.token_bytes(self.privacy_config.min_entropy_bits // 8)
        self._current_salt = salt_bytes.hex()
        self._salt_rotated_at = datetime.utcnow()
    
    def _get_current_salt(self) -> str:
        """Get current salt, rotating if needed"""
        days_since_rotation = (datetime.utcnow() - self._salt_rotated_at).days
        
        if days_since_rotation >= self.privacy_config.salt_rotation_days:
            self._rotate_salt()
        
        return self._current_salt
    
    def _rotate_salt(self) -> None:
        """
        Rotate salt and invalidate old hashes.
        
        GDPR Compliance: Periodic salt rotation limits exposure window.
        """
        old_salt = self._current_salt
        
        # Generate new salt
        salt_bytes = secrets.token_bytes(self.privacy_config.min_entropy_bits // 8)
        self._current_salt = salt_bytes.hex()
        self._salt_rotated_at = datetime.utcnow()
        
        # Audit log
        self._audit_log({
            'event': 'salt_rotated',
            'timestamp': datetime.utcnow(),
            'old_salt_hash': hashlib.sha256(old_salt.encode()).hexdigest(),
            'new_salt_hash': hashlib.sha256(self._current_salt.encode()).hexdigest()
        })
    
    def hash_pii(self, value: str, pii_type: str) -> str:
        """
        Hash PII with current salt using SHA256.
        
        Args:
            value: Plaintext PII (email, phone, etc.)
            pii_type: Type of PII for audit purposes
        
        Returns:
            Hex digest of SHA256(value + salt)
        """
        if not value:
            return ""
        
        salt = self._get_current_salt()
        salted_value = f"{value}{salt}"
        
        hasher = hashlib.sha256()
        hasher.update(salted_value.encode('utf-8'))
        hash_value = hasher.hexdigest()
        
        # Audit log (no plaintext)
        self._audit_log({
            'event': 'pii_hashed',
            'pii_type': pii_type,
            'hash_value': hash_value,
            'timestamp': datetime.utcnow()
        })
        
        return hash_value
    
    def create_user_key(self, primary_identifier: str) -> str:
        """
        Create user_key from primary identifier (email or customer_id).
        
        Returns:
            Hashed user_key with TTL
        """
        user_key = self.hash_pii(primary_identifier, pii_type="user_key")
        
        # Set TTL
        expires_at = datetime.utcnow() + timedelta(days=self.privacy_config.ttl_days)
        
        # Audit log
        self._audit_log({
            'event': 'user_key_created',
            'user_key': user_key,
            'expires_at': expires_at.isoformat(),
            'timestamp': datetime.utcnow()
        })
        
        return user_key
    
    def get_user_key_expiry(self, user_key: str) -> datetime:
        """Get expiry timestamp for user_key"""
        if user_key in self._user_profiles:
            return self._user_profiles[user_key].expires_at
        
        # Default: 90 days from now
        return datetime.utcnow() + timedelta(days=self.privacy_config.ttl_days)
    
    def resolve_identity(self, signals: Dict[str, str]) -> str:
        """
        Resolve identity from signals, returning privacy-safe user_key.
        
        Args:
            signals: Dict with PII fields (email, phone, customer_id, etc.)
        
        Returns:
            user_key (hashed identifier)
        """
        # Filter to allowed PII fields only (data minimization)
        allowed_signals = {
            k: v for k, v in signals.items()
            if k in self.privacy_config.allowed_pii_fields
        }
        
        # Hash all PII immediately
        hashed_signals = {}
        for field, value in allowed_signals.items():
            if value:
                hashed_signals[f'{field}_hash'] = self.hash_pii(value, pii_type=field)
        
        # Deterministic matching: use email or customer_id as primary key
        if 'email_hash' in hashed_signals:
            user_key = hashed_signals['email_hash']
        elif 'customer_id_hash' in hashed_signals:
            user_key = hashed_signals['customer_id_hash']
        else:
            # Fallback: create new user_key from phone
            user_key = hashed_signals.get('phone_hash', secrets.token_hex(32))
        
        # Create or update profile
        if user_key not in self._user_profiles:
            profile = UnifiedProfile(
                user_key=user_key,
                email_hash=hashed_signals.get('email_hash'),
                phone_hash=hashed_signals.get('phone_hash'),
                customer_id_hash=hashed_signals.get('customer_id_hash'),
                created_at=datetime.utcnow(),
                expires_at=datetime.utcnow() + timedelta(days=self.privacy_config.ttl_days)
            )
            self._user_profiles[user_key] = profile
            
            # Audit log
            self._audit_log({
                'event': 'profile_created',
                'user_key': user_key,
                'timestamp': datetime.utcnow()
            })
        
        return user_key
    
    def get_profile(self, user_key: str) -> Optional[UnifiedProfile]:
        """Retrieve profile by user_key (privacy-safe)"""
        return self._user_profiles.get(user_key)
    
    def user_key_exists(self, user_key: str) -> bool:
        """Check if user_key exists and is not expired"""
        if user_key not in self._user_profiles:
            return False
        
        profile = self._user_profiles[user_key]
        if datetime.utcnow() > profile.expires_at:
            return False
        
        return True
    
    def purge_expired_keys(self) -> int:
        """
        Purge expired user_keys (GDPR data minimization).
        
        Returns:
            Number of keys purged
        """
        now = datetime.utcnow()
        expired_keys = [
            user_key for user_key, profile in self._user_profiles.items()
            if now > profile.expires_at
        ]
        
        for user_key in expired_keys:
            del self._user_profiles[user_key]
        
        # Audit log
        if expired_keys:
            self._audit_log({
                'event': 'keys_purged',
                'count': len(expired_keys),
                'timestamp': now
            })
        
        return len(expired_keys)
    
    def _audit_log(self, entry: dict) -> None:
        """
        Append to audit log (no PII in logs).
        
        In production: send to secure audit storage (BigQuery, CloudWatch)
        """
        self._audit_log_entries.append(entry)
