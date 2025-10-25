"""
Tests for PacingAgent - Asset-level Pause Enforcement & Idempotency
"""
import pytest
from datetime import date, datetime
from unittest.mock import Mock, patch, call
from agents.pacing_agent import PacingAgent, PacingStatus


class TestPacingAssetLevel:
    """Test asset-level selective pause enforcement"""
    
    @patch('agents.pacing_agent.get_feature_store')
    @patch('agents.pacing_agent.get_ad_platform_api')
    def test_pacing_asset_level_selective_pause(self, mock_api, mock_fs):
        """Asset A 120%, B 80% → only A paused"""
        agent = PacingAgent()
        
        # Mock asset spend data
        mock_fs.return_value.get_asset_pacing.return_value = {
            'asset_a': {
                'spend_today': 12000,
                'budget_daily': 10000,
                'pacing_percent': 120.0
            },
            'asset_b': {
                'spend_today': 8000,
                'budget_daily': 10000,
                'pacing_percent': 80.0
            }
        }
        
        # Execute pacing check
        results = agent.check_asset_pacing(campaign_id='camp_123', date=date.today())
        
        # Assertions
        assert len(results) == 2
        
        # Asset A should be paused
        asset_a_result = next(r for r in results if r.asset_id == 'asset_a')
        assert asset_a_result.action == 'pause'
        assert asset_a_result.pacing_percent == 120.0
        
        # Asset B should not be paused
        asset_b_result = next(r for r in results if r.asset_id == 'asset_b')
        assert asset_b_result.action == 'none'
        assert asset_b_result.pacing_percent == 80.0
        
        # Verify only asset_a pause API call
        mock_api.return_value.pause_asset.assert_called_once_with('asset_a')
    
    @patch('agents.pacing_agent.get_feature_store')
    @patch('agents.pacing_agent.get_ad_platform_api')
    def test_pacing_resume_when_below_threshold(self, mock_api, mock_fs):
        """Asset previously paused at 120%, now 90% → resume"""
        agent = PacingAgent()
        
        # Mock asset was paused
        mock_fs.return_value.get_asset_status.return_value = {
            'asset_a': 'paused'
        }
        
        # Mock current pacing below resume threshold
        mock_fs.return_value.get_asset_pacing.return_value = {
            'asset_a': {
                'spend_today': 9000,
                'budget_daily': 10000,
                'pacing_percent': 90.0
            }
        }
        
        results = agent.check_asset_pacing(
            campaign_id='camp_123',
            date=date.today(),
            resume_threshold=95.0
        )
        
        asset_a = results[0]
        assert asset_a.action == 'resume'
        assert asset_a.pacing_percent == 90.0
        
        mock_api.return_value.resume_asset.assert_called_once_with('asset_a')
    
    @patch('agents.pacing_agent.get_feature_store')
    def test_pacing_multiple_assets_selective(self, mock_fs):
        """3 assets: 110%, 120%, 80% → pause top 2"""
        agent = PacingAgent()
        
        mock_fs.return_value.get_asset_pacing.return_value = {
            'asset_a': {'spend_today': 11000, 'budget_daily': 10000, 'pacing_percent': 110.0},
            'asset_b': {'spend_today': 12000, 'budget_daily': 10000, 'pacing_percent': 120.0},
            'asset_c': {'spend_today': 8000, 'budget_daily': 10000, 'pacing_percent': 80.0}
        }
        
        results = agent.check_asset_pacing(
            campaign_id='camp_123',
            date=date.today(),
            pause_threshold=110.0
        )
        
        paused = [r for r in results if r.action == 'pause']
        assert len(paused) == 2
        assert set(r.asset_id for r in paused) == {'asset_a', 'asset_b'}


class TestPacingIdempotency:
    """Test idempotency of pacing actions"""
    
    @patch('agents.pacing_agent.redis_client')
    @patch('agents.pacing_agent.get_feature_store')
    @patch('agents.pacing_agent.get_ad_platform_api')
    def test_pacing_agent_idempotency(self, mock_api, mock_fs, mock_redis):
        """Same pause command → single API call (deduped)"""
        agent = PacingAgent()
        
        # Mock asset pacing
        mock_fs.return_value.get_asset_pacing.return_value = {
            'asset_a': {'spend_today': 12000, 'budget_daily': 10000, 'pacing_percent': 120.0}
        }
        
        # Mock Redis cache miss on first call
        mock_redis.get.side_effect = [None, b'processed']  # First miss, then hit
        
        # First call - should execute
        results1 = agent.check_asset_pacing(campaign_id='camp_123', date=date.today())
        
        # Second call with same state - should be deduped
        results2 = agent.check_asset_pacing(campaign_id='camp_123', date=date.today())
        
        # API should be called only once
        assert mock_api.return_value.pause_asset.call_count == 1
        mock_api.return_value.pause_asset.assert_called_with('asset_a')
        
        # Redis should have stored dedup key
        assert mock_redis.setex.call_count == 1
    
    @patch('agents.pacing_agent.redis_client')
    def test_pause_action_id_deterministic(self, mock_redis):
        """pause_action_id is SHA256(asset_id + action + date)"""
        agent = PacingAgent()
        
        action_id1 = agent._compute_action_id(
            asset_id='asset_a',
            action='pause',
            date=date(2025, 10, 19)
        )
        
        action_id2 = agent._compute_action_id(
            asset_id='asset_a',
            action='pause',
            date=date(2025, 10, 19)
        )
        
        # Same inputs → same hash
        assert action_id1 == action_id2
        assert len(action_id1) == 64  # SHA256 hex
        
        # Different date → different hash
        action_id3 = agent._compute_action_id(
            asset_id='asset_a',
            action='pause',
            date=date(2025, 10, 20)
        )
        assert action_id1 != action_id3


class TestPacingContract:
    """Test PacingStatus schema contract"""
    
    def test_pacing_status_schema_fields(self):
        """PacingStatus has required fields"""
        status = PacingStatus(
            asset_id='asset_a',
            campaign_id='camp_123',
            date=date.today(),
            pacing_percent=120.0,
            action='pause',
            timestamp=datetime.utcnow()
        )
        
        assert hasattr(status, 'asset_id')
        assert hasattr(status, 'pacing_percent')
        assert hasattr(status, 'action')
        assert hasattr(status, 'timestamp')
        assert status.action in ['pause', 'resume', 'none']
    
    def test_pacing_status_action_enum(self):
        """PacingStatus.action is enum (pause|resume|none)"""
        # Valid actions
        for action in ['pause', 'resume', 'none']:
            status = PacingStatus(
                asset_id='a',
                campaign_id='c',
                date=date.today(),
                pacing_percent=100.0,
                action=action,
                timestamp=datetime.utcnow()
            )
            assert status.action == action
        
        # Invalid action should raise
        with pytest.raises(ValueError):
            PacingStatus(
                asset_id='a',
                campaign_id='c',
                date=date.today(),
                pacing_percent=100.0,
                action='invalid',
                timestamp=datetime.utcnow()
            )


class TestPacingMetrics:
    """Test Prometheus metrics emission"""
    
    @patch('agents.pacing_agent.pacing_asset_paused_counter')
    @patch('agents.pacing_agent.pacing_asset_resumed_counter')
    @patch('agents.pacing_agent.get_feature_store')
    @patch('agents.pacing_agent.get_ad_platform_api')
    def test_pacing_metrics_emit_correctly(
        self,
        mock_api,
        mock_fs,
        mock_resumed,
        mock_paused
    ):
        """Metrics pacing_asset_paused_total, pacing_asset_resumed_total emit"""
        agent = PacingAgent()
        
        # Mock pause scenario
        mock_fs.return_value.get_asset_pacing.return_value = {
            'asset_a': {'spend_today': 12000, 'budget_daily': 10000, 'pacing_percent': 120.0}
        }
        mock_fs.return_value.get_asset_status.return_value = {'asset_a': 'active'}
        
        agent.check_asset_pacing(campaign_id='camp_123', date=date.today())
        
        # Paused counter should increment
        mock_paused.labels.assert_called_with(
            campaign_id='camp_123',
            asset_id='asset_a',
            reason='overpacing'
        )
        mock_paused.labels.return_value.inc.assert_called_once()
        
        # Mock resume scenario
        mock_fs.return_value.get_asset_pacing.return_value = {
            'asset_a': {'spend_today': 9000, 'budget_daily': 10000, 'pacing_percent': 90.0}
        }
        mock_fs.return_value.get_asset_status.return_value = {'asset_a': 'paused'}
        
        agent.check_asset_pacing(campaign_id='camp_123', date=date.today(), resume_threshold=95.0)
        
        # Resumed counter should increment
        mock_resumed.labels.assert_called_with(
            campaign_id='camp_123',
            asset_id='asset_a'
        )
        mock_resumed.labels.return_value.inc.assert_called_once()
