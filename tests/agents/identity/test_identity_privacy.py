"""
Identity Privacy Tests - GDPR Compliance

Tests for salt rotation, TTL enforcement, and PII handling.
Related: Q_002, A_002
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
from src.agents.identity.identity_resolution_agent import IdentityResolutionAgent
from src.agents.identity.privacy_config import PrivacyConfig


class TestGDPRCompliance:
    """Test GDPR Article 32 compliance requirements"""
    
    @pytest.fixture
    def privacy_config(self):
        return PrivacyConfig(
            salt_rotation_days=90,
            ttl_days=90,
            hash_algorithm="sha256"
        )
    
    @pytest.fixture
    def agent(self, privacy_config):
        return IdentityResolutionAgent(privacy_config=privacy_config)
    
    def test_salt_rotation_schedule_enforced(self, agent):
        """ACCEPTANCE: Salt rotation policy â‰¥30 days configured"""
        assert agent.privacy_config.salt_rotation_days >= 30
        assert agent.privacy_config.salt_rotation_days == 90
    
    def test_ttl_enforcement_on_hashed_identifiers(self, agent):
        """ACCEPTANCE: 90-day TTL enforced on user_key"""
        assert agent.privacy_config.ttl_days == 90
        
        # Simulate user_key creation
        user_key = agent.create_user_key("test@example.com")
        expiry = agent.get_user_key_expiry(user_key)
        
        expected_expiry = datetime.utcnow() + timedelta(days=90)
        assert abs((expiry - expected_expiry).total_seconds()) < 60
    
    def test_pii_hashing_with_salt(self, agent):
        """ACCEPTANCE: PII hashed with rotating salt"""
        email = "user@example.com"
        
        # Hash should include salt
        hash1 = agent.hash_pii(email, pii_type="email")
        hash2 = agent.hash_pii(email, pii_type="email")
        
        # Same input, same salt → same hash
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA256 hex digest
    
    def test_salt_rotation_changes_hashes(self, agent):
        """ACCEPTANCE: Salt rotation invalidates old hashes"""
        email = "user@example.com"
        
        # Hash with current salt
        old_salt = agent._get_current_salt()
        hash_before = agent.hash_pii(email, pii_type="email")
        
        # Rotate salt
        agent._rotate_salt()
        new_salt = agent._get_current_salt()
        hash_after = agent.hash_pii(email, pii_type="email")
        
        # Hashes should differ after rotation
        assert old_salt != new_salt
        assert hash_before != hash_after
    
    @patch('src.agents.identity.identity_resolution_agent.datetime')
    def test_expired_user_keys_purged(self, mock_datetime, agent):
        """ACCEPTANCE: User keys auto-purged after TTL"""
        # Set current time
        now = datetime(2025, 10, 19, 12, 0, 0)
        mock_datetime.utcnow.return_value = now
        
        # Create user key
        user_key = agent.create_user_key("test@example.com")
        
        # Fast-forward 91 days
        future = now + timedelta(days=91)
        mock_datetime.utcnow.return_value = future
        
        # Purge expired keys
        agent.purge_expired_keys()
        
        # User key should be deleted
        assert not agent.user_key_exists(user_key)
    
    def test_audit_logging_for_pii_access(self, agent):
        """ACCEPTANCE: All PII operations logged for audit"""
        email = "audit@example.com"
        
        with patch.object(agent, '_audit_log') as mock_audit:
            user_key = agent.create_user_key(email)
            
            # Verify audit log called
            mock_audit.assert_called_once()
            log_entry = mock_audit.call_args[0][0]
            
            assert log_entry['event'] == 'user_key_created'
            assert log_entry['user_key'] == user_key
            assert 'email' not in log_entry  # PII not logged
            assert log_entry['timestamp'] is not None
    
    def test_no_plaintext_pii_in_storage(self, agent):
        """ACCEPTANCE: No plaintext PII stored anywhere"""
        email = "plaintext@example.com"
        phone = "+1234567890"
        
        user_key = agent.resolve_identity({
            'email': email,
            'phone': phone
        })
        
        # Retrieve stored profile
        profile = agent.get_profile(user_key)
        
        # Verify no plaintext PII
        profile_str = str(profile)
        assert email not in profile_str
        assert phone not in profile_str
        assert 'email_hash' in profile
        assert 'phone_hash' in profile


class TestPrivacyConfig:
    """Test privacy configuration schema"""
    
    def test_privacy_config_schema_validation(self):
        """ACCEPTANCE: PrivacyConfig validates rotation_interval_days ≥ 30"""
        # Valid config
        config = PrivacyConfig(salt_rotation_days=90, ttl_days=90)
        assert config.salt_rotation_days == 90
        
        # Invalid: rotation too frequent
        with pytest.raises(ValueError, match="Salt rotation must be ≥ 30 days"):
            PrivacyConfig(salt_rotation_days=15, ttl_days=90)
    
    def test_privacy_config_defaults(self):
        """ACCEPTANCE: Secure defaults for privacy settings"""
        config = PrivacyConfig()
        
        assert config.salt_rotation_days >= 30
        assert config.ttl_days >= 30
        assert config.hash_algorithm in ['sha256', 'sha512']
        assert config.min_entropy_bits >= 256


class TestDataMinimization:
    """Test GDPR data minimization principles"""
    
    def test_only_necessary_fields_collected(self):
        """ACCEPTANCE: Only collect PII necessary for identity resolution"""
        agent = IdentityResolutionAgent()
        
        # Define allowed PII fields
        allowed_fields = {'email', 'phone', 'customer_id'}
        
        # Attempt to store extra PII
        signals = {
            'email': 'test@example.com',
            'phone': '+1234567890',
            'ssn': '123-45-6789',  # Not allowed
            'credit_card': '1234-5678-9012-3456'  # Not allowed
        }
        
        user_key = agent.resolve_identity(signals)
        profile = agent.get_profile(user_key)
        
        # Verify only allowed fields stored (as hashes)
        assert 'email_hash' in profile
        assert 'phone_hash' in profile
        assert 'ssn' not in profile
        assert 'ssn_hash' not in profile
        assert 'credit_card' not in profile
