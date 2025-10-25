"""
Test suite for MTA agent privacy enforcement: >=10 users per path requirement
Component: C05_MTA
Capsule: Q_007
Priority: P0 (CRITICAL - Privacy violation risk)
"""

import pytest
from agents.mta_agent import MTAAgent, ConversionPath
from core.exceptions import PrivacyViolationError


class TestMTAPrivacyEnforcement:
    """Test MTA privacy threshold enforcement for conversion paths"""
    
    @pytest.fixture
    def mta_agent(self):
        """Create MTA agent instance with privacy enforcement enabled"""
        return MTAAgent(min_users_per_path=10, privacy_mode=True)
    
    def test_path_with_10_users_accepted(self, mta_agent):
        """Test: Path with exactly 10 users should be accepted"""
        path = ConversionPath(
            path_id="hash_abc123",
            touchpoints=[
                {"channel": "meta", "position": 1},
                {"channel": "google", "position": 2}
            ],
            converted=True,
            conversions=10,  # Exactly at threshold
            revenue=50000.0
        )
        
        result = mta_agent.validate_path_privacy(path)
        assert result.is_valid is True
        assert result.violation is None
    
    def test_path_with_9_users_rejected(self, mta_agent):
        """Test: Path with 9 users should raise PrivacyViolationError"""
        path = ConversionPath(
            path_id="hash_def456",
            touchpoints=[
                {"channel": "meta", "position": 1},
                {"channel": "google", "position": 2}
            ],
            converted=True,
            conversions=9,  # Below threshold
            revenue=45000.0
        )
        
        with pytest.raises(PrivacyViolationError) as exc_info:
            mta_agent.validate_path_privacy(path)
        
        assert "minimum 10 users" in str(exc_info.value).lower()
        assert "9 users" in str(exc_info.value)
    
    def test_aggregated_paths_all_valid(self, mta_agent):
        """Test: All aggregated paths must meet 10-user threshold"""
        paths = [
            ConversionPath(path_id=f"hash_{i}", touchpoints=[{"channel": "meta", "position": 1}],
                          converted=True, conversions=10 + i, revenue=50000.0)
            for i in range(5)
        ]
        
        result = mta_agent.compute_attribution(paths, lookback_days=30)
        
        assert result.privacy_compliant is True
        assert all(p.conversions >= 10 for p in paths)
    
    def test_aggregated_paths_with_violation(self, mta_agent):
        """Test: Aggregated paths with any <10 user path should fail"""
        paths = [
            ConversionPath(path_id="hash_ok1", touchpoints=[{"channel": "meta", "position": 1}],
                          converted=True, conversions=12, revenue=60000.0),
            ConversionPath(path_id="hash_violation", touchpoints=[{"channel": "google", "position": 1}],
                          converted=True, conversions=8, revenue=40000.0),  # Violation
            ConversionPath(path_id="hash_ok2", touchpoints=[{"channel": "tiktok", "position": 1}],
                          converted=True, conversions=15, revenue=75000.0)
        ]
        
        with pytest.raises(PrivacyViolationError) as exc_info:
            mta_agent.compute_attribution(paths, lookback_days=30)
        
        assert "privacy threshold violated" in str(exc_info.value).lower()
        assert "8 users" in str(exc_info.value)
    
    def test_privacy_metrics_emitted(self, mta_agent, mocker):
        """Test: Privacy validation should emit prometheus metrics"""
        mock_counter = mocker.patch('agents.mta_agent.privacy_violation_counter')
        
        path_valid = ConversionPath(path_id="hash_valid", touchpoints=[{"channel": "meta", "position": 1}],
                                   converted=True, conversions=10, revenue=50000.0)
        path_invalid = ConversionPath(path_id="hash_invalid", touchpoints=[{"channel": "google", "position": 1}],
                                     converted=True, conversions=5, revenue=25000.0)
        
        # Valid path - no metric
        mta_agent.validate_path_privacy(path_valid)
        assert mock_counter.inc.call_count == 0
        
        # Invalid path - metric incremented
        with pytest.raises(PrivacyViolationError):
            mta_agent.validate_path_privacy(path_invalid)
        
        mock_counter.inc.assert_called_once()
    
    def test_kill_switch_bypass_privacy(self, mta_agent):
        """Test: Kill switch ENABLE_MTA_PRIVACY_CHECK=false bypasses validation"""
        mta_agent.config.kill_switches['ENABLE_MTA_PRIVACY_CHECK'] = False
        
        path = ConversionPath(path_id="hash_bypass", touchpoints=[{"channel": "meta", "position": 1}],
                             converted=True, conversions=1, revenue=5000.0)  # Would normally fail
        
        # Should not raise when kill switch disabled
        result = mta_agent.validate_path_privacy(path)
        assert result.is_valid is True
        assert result.bypass_reason == "kill_switch_disabled"


class TestMTAPathAggregation:
    """Test MTA path aggregation maintains privacy thresholds"""
    
    def test_rare_paths_filtered_out(self):
        """Test: Paths with <10 users are filtered during aggregation"""
        agent = MTAAgent(min_users_per_path=10, privacy_mode=True)
        
        raw_paths = [
            {"path_hash": "hash_A", "user_count": 15, "converted": True},
            {"path_hash": "hash_B", "user_count": 8, "converted": True},   # Filtered
            {"path_hash": "hash_C", "user_count": 12, "converted": True},
            {"path_hash": "hash_D", "user_count": 5, "converted": False},  # Filtered
            {"path_hash": "hash_E", "user_count": 20, "converted": True}
        ]
        
        aggregated = agent.aggregate_paths(raw_paths)
        
        assert len(aggregated) == 3  # Only A, C, E
        assert all(p.conversions >= 10 for p in aggregated)
        assert "hash_B" not in [p.path_id for p in aggregated]
        assert "hash_D" not in [p.path_id for p in aggregated]
