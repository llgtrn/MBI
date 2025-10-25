"""
Unit tests for Privacy module
Component: C01_IdentityResolution (CRITICAL priority)
Coverage: Salt rotation, dual-key fallback, grace period, rollback detection
"""
import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import hashlib
import time

from core.privacy import (
    PrivacyHasher,
    PrivacyConfig,
    SaltRotationManager
)


class TestSaltRotationGracePeriod:
    """Test salt rotation with dual-key fallback strategy (Q_002, Q_402)"""
    
    @pytest.fixture
    def privacy_config(self):
        """Create test privacy configuration"""
        return PrivacyConfig(
            current_salt="salt_v2_abc123",
            old_salt="salt_v1_xyz789",
            rotation_date=datetime.utcnow() - timedelta(days=3),
            grace_period_days=7,
            rotation_schedule_days=90
        )
    
    @pytest.fixture
    def hasher(self, privacy_config):
        """Create hasher with test config"""
        return PrivacyHasher(config=privacy_config)
    
    def test_salt_rotation_grace_period_within_7_days(self, hasher, privacy_config):
        """
        Q_002 acceptance: old_salt still valid within 7 days
        Test that hashes created with old_salt can be verified during grace period
        """
        # Arrange
        pii_value = "test@example.com"
        
        # Create hash with old salt (simulating pre-rotation data)
        old_hash = hashlib.sha256(
            (pii_value + privacy_config.old_salt).encode()
        ).hexdigest()
        
        # Mock current time to be 6 days after rotation (within grace)
        with patch('core.privacy.datetime') as mock_datetime:
            mock_datetime.utcnow.return_value = (
                privacy_config.rotation_date + timedelta(days=6)
            )
            
            # Act: Verify hash using dual-key strategy
            is_valid = hasher.verify_hash(pii_value, old_hash)
            
            # Assert: Should succeed with old_salt fallback
            assert is_valid is True
            
            # Verify dual-key attempt was logged
            assert hasher.stats['dual_key_fallback_attempts'] >= 1
    
    def test_old_salt_rejected_after_grace_period(self, hasher, privacy_config):
        """
        Q_002 acceptance: >7d old salt → rejection
        Test that old_salt is rejected after grace period expires
        """
        # Arrange
        pii_value = "test@example.com"
        
        # Create hash with old salt
        old_hash = hashlib.sha256(
            (pii_value + privacy_config.old_salt).encode()
        ).hexdigest()
        
        # Mock current time to be 8 days after rotation (beyond grace)
        with patch('core.privacy.datetime') as mock_datetime:
            mock_datetime.utcnow.return_value = (
                privacy_config.rotation_date + timedelta(days=8)
            )
            
            # Act: Try to verify hash
            is_valid = hasher.verify_hash(pii_value, old_hash)
            
            # Assert: Should fail - grace period expired
            assert is_valid is False
            
            # Verify rejection was logged
            assert hasher.stats['grace_period_rejections'] >= 1
    
    def test_current_salt_always_preferred(self, hasher):
        """
        Test that current_salt is always tried first before fallback
        """
        # Arrange
        pii_value = "priority@example.com"
        
        # Mock hash_with_salt to track call order
        call_order = []
        
        def mock_hash_with_salt(value, salt):
            call_order.append(salt)
            return hashlib.sha256((value + salt).encode()).hexdigest()
        
        with patch.object(hasher, '_hash_with_salt', side_effect=mock_hash_with_salt):
            # Act
            hasher.hash_pii({'email': pii_value})
            
            # Assert: current_salt was used (not old_salt)
            assert len(call_order) == 1
            assert call_order[0] == hasher.config.current_salt
    
    def test_dual_key_fallback_logic_order(self, hasher, privacy_config):
        """
        Q_402 acceptance: Code validates dual-key strategy with correct fallback order
        """
        # Arrange
        pii_value = "fallback@example.com"
        
        # Create hash with old salt
        old_hash = hashlib.sha256(
            (pii_value + privacy_config.old_salt).encode()
        ).hexdigest()
        
        # Mock time within grace period (5 days)
        with patch('core.privacy.datetime') as mock_datetime:
            mock_datetime.utcnow.return_value = (
                privacy_config.rotation_date + timedelta(days=5)
            )
            
            # Track which salts were attempted
            with patch.object(hasher, '_hash_with_salt', wraps=hasher._hash_with_salt) as mock_hash:
                # Act
                result = hasher.verify_hash(pii_value, old_hash)
                
                # Assert
                assert result is True
                
                # Verify call order: current_salt first, then old_salt
                calls = mock_hash.call_args_list
                assert len(calls) >= 2  # At least 2 attempts
                assert calls[0][0][1] == privacy_config.current_salt  # First: current
                assert calls[1][0][1] == privacy_config.old_salt  # Second: old (fallback)


class TestPrivacyConfigSchema:
    """Test PrivacyConfig schema validation (Q_402 contract)"""
    
    def test_privacy_config_required_fields(self):
        """
        Q_402 contract: PrivacyConfig has current_salt, old_salt, rotation_date, grace_period_days
        """
        config = PrivacyConfig(
            current_salt="test_current",
            old_salt="test_old",
            rotation_date=datetime.utcnow(),
            grace_period_days=7
        )
        
        assert hasattr(config, 'current_salt')
        assert hasattr(config, 'old_salt')
        assert hasattr(config, 'rotation_date')
        assert hasattr(config, 'grace_period_days')
        
        assert config.current_salt == "test_current"
        assert config.old_salt == "test_old"
        assert config.grace_period_days == 7
    
    def test_privacy_config_validation(self):
        """Test schema validation rules"""
        # Grace period must be positive
        with pytest.raises(ValueError):
            PrivacyConfig(
                current_salt="test",
                old_salt="test",
                rotation_date=datetime.utcnow(),
                grace_period_days=-1  # Invalid
            )
        
        # Current and old salt must differ
        with pytest.raises(ValueError):
            PrivacyConfig(
                current_salt="same_salt",
                old_salt="same_salt",  # Must be different
                rotation_date=datetime.utcnow(),
                grace_period_days=7
            )


class TestSaltRotationMetrics:
    """Test salt rotation age tracking (Q_002 metric)"""
    
    def test_salt_rotation_age_gauge(self):
        """
        Q_002 acceptance: metric salt_rotation_age_days tracks days since last rotation
        """
        from prometheus_client import REGISTRY
        
        # Arrange
        rotation_date = datetime.utcnow() - timedelta(days=15)
        config = PrivacyConfig(
            current_salt="test_current",
            old_salt="test_old",
            rotation_date=rotation_date,
            grace_period_days=7
        )
        
        hasher = PrivacyHasher(config=config)
        
        # Act: Update metric
        hasher.update_rotation_metrics()
        
        # Assert: Metric should show ~15 days
        metric_value = REGISTRY.get_sample_value('salt_rotation_age_days')
        assert metric_value is not None
        assert 14 <= metric_value <= 16  # Allow 1 day tolerance
    
    def test_salt_rotation_age_alert_threshold(self):
        """
        Test that rotation age alert triggers at 90 days (rotation schedule)
        """
        # Arrange: Salt is 91 days old (overdue)
        rotation_date = datetime.utcnow() - timedelta(days=91)
        config = PrivacyConfig(
            current_salt="test_current",
            old_salt="test_old",
            rotation_date=rotation_date,
            grace_period_days=7,
            rotation_schedule_days=90
        )
        
        hasher = PrivacyHasher(config=config)
        
        # Act
        is_overdue = hasher.is_rotation_overdue()
        
        # Assert
        assert is_overdue is True


class TestIdentityMergeRollback:
    """Test identity merge rollback on rotation (Q_424)"""
    
    def test_detect_salt_rotation_during_merge(self):
        """
        Q_424 acceptance: detect salt rotation during merge → rollback
        """
        # Arrange
        initial_config = PrivacyConfig(
            current_salt="salt_v1",
            old_salt="salt_v0",
            rotation_date=datetime.utcnow() - timedelta(days=30),
            grace_period_days=7
        )
        
        hasher = PrivacyHasher(config=initial_config)
        
        # Start merge operation
        merge_started_at = datetime.utcnow()
        user_key_1 = "uhash_user1"
        user_key_2 = "uhash_user2"
        
        # Simulate rotation happening during merge
        rotated_config = PrivacyConfig(
            current_salt="salt_v2",  # New salt
            old_salt="salt_v1",
            rotation_date=datetime.utcnow(),  # Just rotated
            grace_period_days=7
        )
        
        # Mock config reload
        with patch.object(hasher, 'reload_config', return_value=rotated_config):
            # Act: Check if rotation occurred during merge
            rotation_detected = hasher.check_rotation_during_operation(
                started_at=merge_started_at
            )
            
            # Assert: Should detect rotation and trigger rollback
            assert rotation_detected is True
            
            # Verify rollback would be initiated
            assert hasher.should_rollback_merge(
                started_at=merge_started_at,
                rotation_detected=rotation_detected
            ) is True


class TestSaltRotationSchedule:
    """Test 90-day rotation schedule (from AGENT.md requirement)"""
    
    def test_rotation_schedule_90_days(self):
        """
        Verify salt rotation occurs every 90 days as per AGENT.md spec
        """
        manager = SaltRotationManager()
        
        # Current rotation
        rotation_1 = datetime.utcnow()
        
        # Next scheduled rotation
        next_rotation = manager.calculate_next_rotation(rotation_1)
        
        # Assert: Should be 90 days later
        expected = rotation_1 + timedelta(days=90)
        assert abs((next_rotation - expected).days) <= 1  # Allow 1 day tolerance
    
    def test_rotation_stores_old_salt_with_timestamp(self):
        """
        Verify rotation process stores old_salt with timestamp for audit
        """
        manager = SaltRotationManager()
        
        # Perform rotation
        old_salt = "salt_v1_abc"
        new_salt = manager.generate_new_salt()
        
        rotation_result = manager.rotate_salt(
            current_salt=old_salt,
            new_salt=new_salt
        )
        
        # Assert: Old salt stored with timestamp
        assert rotation_result['old_salt'] == old_salt
        assert rotation_result['old_salt_stored_at'] is not None
        assert rotation_result['new_salt'] == new_salt
        assert rotation_result['rotation_date'] is not None
