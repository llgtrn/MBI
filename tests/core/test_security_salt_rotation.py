"""
Unit tests for PII salt rotation with 24h dual-valid Secret Manager
Validates Q_002: Zero hash mismatches during rotation window
"""
import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import hashlib


class TestSecuritySaltRotation:
    """Test suite for 24h dual-valid salt rotation (Q_002)"""
    
    def test_dual_valid_24h(self):
        """
        Acceptance: Both current and previous salt resolve same user_key within 24h window
        Contract: salt_current AND salt_previous both valid
        Metric: zero hash_mismatch_count
        """
        from core.security import SecurityManager
        
        # Setup: mock Secret Manager with current and previous salt
        mock_secret_manager = Mock()
        mock_secret_manager.get_secret.side_effect = lambda key: {
            'salt_current': 'new_salt_v2',
            'salt_previous': 'old_salt_v1'
        }.get(key)
        
        security = SecurityManager(secret_manager=mock_secret_manager)
        
        # Test data
        email = "user@example.com"
        
        # Generate hash with current salt
        hash_current = security.hash_pii(email, salt_key='salt_current')
        
        # Generate hash with previous salt
        hash_previous = security.hash_pii(email, salt_key='salt_previous')
        
        # Both should be valid and resolve to same user_key
        user_key_current = security.resolve_user_key(email, try_previous=False)
        user_key_previous = security.resolve_user_key(email, try_previous=True)
        
        # Assertions
        assert user_key_current is not None, "Current salt must resolve"
        assert user_key_previous is not None, "Previous salt must resolve"
        assert user_key_current == user_key_previous, "Both salts must resolve to same user_key"
        
        # Verify both salts were fetched
        assert mock_secret_manager.get_secret.call_count >= 2
        
        # Check metrics: zero mismatches
        metrics = security.get_rotation_metrics()
        assert metrics['hash_mismatch_count'] == 0, "Must have zero hash mismatches in 24h window"
    
    def test_salt_rotation_window_expiry(self):
        """
        Test that previous salt is rejected after 24h rotation window
        """
        from core.security import SecurityManager
        
        mock_secret_manager = Mock()
        mock_secret_manager.get_secret.side_effect = lambda key: {
            'salt_current': 'new_salt_v2',
            'salt_previous': 'old_salt_v1',
            'rotation_timestamp': (datetime.utcnow() - timedelta(hours=25)).isoformat()
        }.get(key)
        
        security = SecurityManager(secret_manager=mock_secret_manager)
        
        email = "user@example.com"
        
        # After 24h, previous salt should be invalid
        with pytest.raises(ValueError, match="Previous salt expired"):
            security.resolve_user_key(email, salt_key='salt_previous', enforce_window=True)
    
    def test_fallback_on_secret_manager_failure(self):
        """
        Risk gate: Fallback to cached salt on Secret Manager timeout
        """
        from core.security import SecurityManager
        
        mock_secret_manager = Mock()
        mock_secret_manager.get_secret.side_effect = TimeoutError("Secret Manager timeout")
        
        # Should fallback to cached salt (not fail)
        security = SecurityManager(
            secret_manager=mock_secret_manager,
            cached_salt='cached_salt_v1'
        )
        
        email = "user@example.com"
        
        # Should succeed using cached salt
        user_key = security.resolve_user_key(email, use_cache_fallback=True)
        assert user_key is not None, "Must fallback to cached salt on timeout"
    
    def test_kill_switch_disables_rotation(self):
        """
        Risk gate: ENABLE_SALT_ROTATION kill switch
        """
        from core.security import SecurityManager
        
        mock_secret_manager = Mock()
        security = SecurityManager(
            secret_manager=mock_secret_manager,
            enable_salt_rotation=False  # Kill switch OFF
        )
        
        email = "user@example.com"
        
        # When kill switch is off, should use single cached salt only
        user_key = security.resolve_user_key(email)
        
        # Verify Secret Manager was NOT called (using cached salt only)
        assert mock_secret_manager.get_secret.call_count == 0, "Kill switch should prevent Secret Manager calls"
    
    def test_rotation_metrics_emitted(self):
        """
        Verify prometheus metrics are emitted for rotation events
        """
        from core.security import SecurityManager
        
        mock_secret_manager = Mock()
        mock_secret_manager.get_secret.side_effect = lambda key: {
            'salt_current': 'new_salt_v2',
            'salt_previous': 'old_salt_v1'
        }.get(key)
        
        security = SecurityManager(secret_manager=mock_secret_manager)
        
        email = "user@example.com"
        security.resolve_user_key(email)
        
        metrics = security.get_rotation_metrics()
        
        # Required metrics
        assert 'rotation_success_total' in metrics
        assert 'hash_mismatch_count' in metrics
        assert 'secret_manager_fetch_duration_seconds' in metrics
        assert metrics['hash_mismatch_count'] == 0
