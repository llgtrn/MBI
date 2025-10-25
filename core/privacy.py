"""
Privacy Module - Salt Management with Dual-Key Fallback
Component: C01_IdentityResolution (CRITICAL)
Purpose: Secure PII hashing with graceful salt rotation
"""
import hashlib
from datetime import datetime, timedelta
from typing import Dict, Optional
from pydantic import BaseModel, Field, validator
from prometheus_client import Gauge, Counter
import logging

logger = logging.getLogger(__name__)

# Metrics
salt_rotation_age_days = Gauge(
    'salt_rotation_age_days',
    'Days since last salt rotation'
)

salt_dual_key_fallback_total = Counter(
    'salt_dual_key_fallback_total',
    'Total dual-key fallback attempts'
)

salt_grace_period_rejections_total = Counter(
    'salt_grace_period_rejections_total',
    'Total rejections after grace period expiration'
)


class PrivacyConfig(BaseModel):
    """
    Privacy configuration with dual-salt support
    
    Implements:
    - Q_002: Salt rotation with 7-day grace period
    - Q_402: Dual-key strategy for zero data loss
    """
    current_salt: str = Field(..., min_length=32)
    old_salt: str = Field(..., min_length=32)
    rotation_date: datetime
    grace_period_days: int = Field(default=7, ge=1, le=30)
    rotation_schedule_days: int = Field(default=90, ge=30, le=365)
    
    @validator('old_salt')
    def old_salt_must_differ(cls, v, values):
        """Ensure old_salt differs from current_salt"""
        if 'current_salt' in values and v == values['current_salt']:
            raise ValueError("old_salt must differ from current_salt")
        return v
    
    @validator('rotation_date')
    def rotation_date_not_future(cls, v):
        """Rotation date cannot be in the future"""
        if v > datetime.utcnow():
            raise ValueError("rotation_date cannot be in the future")
        return v


class PrivacyHasher:
    """
    PII hasher with salt rotation and dual-key fallback
    
    Implements Q_002/Q_402 dual-key strategy:
    1. Always hash with current_salt for new data
    2. For verification, try current_salt first
    3. If current fails AND within grace period, try old_salt
    4. Track fallback usage for monitoring
    """
    
    def __init__(self, config: PrivacyConfig = None):
        self.config = config or self._load_config_from_env()
        self.stats = {
            'dual_key_fallback_attempts': 0,
            'grace_period_rejections': 0,
            'successful_verifications': 0
        }
    
    def hash_pii(self, pii_data: Dict) -> Dict:
        """
        Hash all PII fields immediately
        
        Args:
            pii_data: Dict with email, phone, etc.
            
        Returns:
            Dict with *_hash fields only
        """
        hashed = {}
        
        for field, value in pii_data.items():
            if value is None:
                continue
            
            # Always use current_salt for new hashes
            hash_value = self._hash_with_salt(value, self.config.current_salt)
            hashed[f'{field}_hash'] = hash_value
        
        return hashed
    
    def verify_hash(self, pii_value: str, hash_to_verify: str) -> bool:
        """
        Verify a hash with dual-key fallback strategy
        
        Q_002/Q_402 implementation:
        1. Try current_salt first
        2. If fail AND within grace → try old_salt
        3. If fail AND beyond grace → reject
        
        Args:
            pii_value: Plain PII value
            hash_to_verify: Hash to check
            
        Returns:
            True if hash matches (with either salt within grace period)
        """
        # Step 1: Try current_salt
        current_hash = self._hash_with_salt(pii_value, self.config.current_salt)
        
        if current_hash == hash_to_verify:
            self.stats['successful_verifications'] += 1
            return True
        
        # Step 2: Check if within grace period
        days_since_rotation = (datetime.utcnow() - self.config.rotation_date).days
        
        if days_since_rotation > self.config.grace_period_days:
            # Beyond grace period - reject without trying old_salt
            salt_grace_period_rejections_total.inc()
            self.stats['grace_period_rejections'] += 1
            logger.debug(
                f"Hash verification failed: beyond grace period "
                f"({days_since_rotation} > {self.config.grace_period_days} days)"
            )
            return False
        
        # Step 3: Within grace - try old_salt fallback
        old_hash = self._hash_with_salt(pii_value, self.config.old_salt)
        
        if old_hash == hash_to_verify:
            # Q_402: Successful dual-key fallback
            salt_dual_key_fallback_total.inc()
            self.stats['dual_key_fallback_attempts'] += 1
            logger.info(
                f"Dual-key fallback successful "
                f"(rotation: {days_since_rotation} days ago, "
                f"grace: {self.config.grace_period_days} days)"
            )
            self.stats['successful_verifications'] += 1
            return True
        
        # Neither salt worked
        return False
    
    def _hash_with_salt(self, value: str, salt: str) -> str:
        """
        Hash a value with a specific salt
        
        Args:
            value: Plain value
            salt: Salt to use
            
        Returns:
            SHA256 hex digest
        """
        return hashlib.sha256((value + salt).encode('utf-8')).hexdigest()
    
    def update_rotation_metrics(self):
        """
        Q_002 metric: Update salt_rotation_age_days gauge
        """
        days_since_rotation = (datetime.utcnow() - self.config.rotation_date).days
        salt_rotation_age_days.set(days_since_rotation)
    
    def is_rotation_overdue(self) -> bool:
        """
        Check if salt rotation is overdue (>90 days)
        
        Returns:
            True if rotation is overdue
        """
        days_since_rotation = (datetime.utcnow() - self.config.rotation_date).days
        return days_since_rotation >= self.config.rotation_schedule_days
    
    def check_rotation_during_operation(self, started_at: datetime) -> bool:
        """
        Q_424: Detect if salt rotation occurred during an operation
        
        Args:
            started_at: When the operation started
            
        Returns:
            True if rotation occurred after operation start
        """
        # Reload config to detect any rotation
        current_config = self._load_config_from_env()
        
        # Check if rotation_date changed and is after operation start
        if current_config.rotation_date > started_at:
            logger.warning(
                f"Salt rotation detected during operation "
                f"(started: {started_at}, rotated: {current_config.rotation_date})"
            )
            return True
        
        return False
    
    def should_rollback_merge(
        self,
        started_at: datetime,
        rotation_detected: bool
    ) -> bool:
        """
        Q_424: Determine if merge should be rolled back due to rotation
        
        Args:
            started_at: Operation start time
            rotation_detected: Whether rotation was detected
            
        Returns:
            True if rollback should occur
        """
        if not rotation_detected:
            return False
        
        # If rotation occurred during merge, hashes may be inconsistent
        # Rollback to prevent data corruption
        logger.error(
            f"Merge rollback triggered: salt rotation during operation "
            f"(started: {started_at})"
        )
        return True
    
    def reload_config(self) -> PrivacyConfig:
        """Reload configuration from storage"""
        return self._load_config_from_env()
    
    def _load_config_from_env(self) -> PrivacyConfig:
        """Load privacy config from environment/secrets"""
        # In production, load from KMS/Secrets Manager
        # For now, return test config
        import os
        
        return PrivacyConfig(
            current_salt=os.getenv('PRIVACY_CURRENT_SALT', 'default_current_salt_32_chars_min'),
            old_salt=os.getenv('PRIVACY_OLD_SALT', 'default_old_salt_32_chars_minimum'),
            rotation_date=datetime.fromisoformat(
                os.getenv('PRIVACY_ROTATION_DATE', datetime.utcnow().isoformat())
            ),
            grace_period_days=int(os.getenv('PRIVACY_GRACE_PERIOD_DAYS', '7')),
            rotation_schedule_days=int(os.getenv('PRIVACY_ROTATION_SCHEDULE_DAYS', '90'))
        )


class SaltRotationManager:
    """
    Manages scheduled salt rotations
    
    Schedule: Every 90 days (configurable)
    Process:
    1. Generate new salt
    2. Set current → old
    3. Set new → current
    4. Store rotation timestamp
    5. Start grace period
    """
    
    def generate_new_salt(self, length: int = 64) -> str:
        """
        Generate cryptographically secure salt
        
        Args:
            length: Salt length in characters
            
        Returns:
            Random hex string
        """
        import secrets
        return secrets.token_hex(length // 2)
    
    def rotate_salt(
        self,
        current_salt: str,
        new_salt: str
    ) -> Dict:
        """
        Perform salt rotation
        
        Args:
            current_salt: Current salt (becomes old_salt)
            new_salt: New salt (becomes current_salt)
            
        Returns:
            Rotation result with timestamps
        """
        rotation_date = datetime.utcnow()
        
        result = {
            'old_salt': current_salt,
            'old_salt_stored_at': rotation_date,
            'new_salt': new_salt,
            'rotation_date': rotation_date,
            'grace_period_ends_at': rotation_date + timedelta(days=7)
        }
        
        logger.info(
            f"Salt rotation completed at {rotation_date.isoformat()}, "
            f"grace period ends {result['grace_period_ends_at'].isoformat()}"
        )
        
        # In production, persist to secrets manager
        self._persist_rotation_config(result)
        
        return result
    
    def calculate_next_rotation(self, last_rotation: datetime) -> datetime:
        """
        Calculate next scheduled rotation (90 days)
        
        Args:
            last_rotation: Last rotation date
            
        Returns:
            Next rotation date
        """
        return last_rotation + timedelta(days=90)
    
    def _persist_rotation_config(self, config: Dict):
        """Persist rotation config to secrets manager"""
        # In production: write to KMS/Secrets Manager
        logger.info(f"Persisting rotation config: {config['rotation_date']}")
        pass


# Cron job for automated rotation (pseudo-code)
"""
# In deployment/cron/rotate_salt.py
from core.privacy import SaltRotationManager, PrivacyConfig

def scheduled_salt_rotation():
    manager = SaltRotationManager()
    
    # Load current config
    current_config = PrivacyConfig.load_from_secrets()
    
    # Check if rotation is due
    days_since = (datetime.utcnow() - current_config.rotation_date).days
    
    if days_since >= 90:
        # Generate and rotate
        new_salt = manager.generate_new_salt()
        result = manager.rotate_salt(
            current_salt=current_config.current_salt,
            new_salt=new_salt
        )
        
        # Update secrets manager
        PrivacyConfig.save_to_secrets(result)
        
        # Alert team
        send_alert(
            subject="Salt Rotation Completed",
            details=f"Rotated at {result['rotation_date']}, grace ends {result['grace_period_ends_at']}"
        )
"""
