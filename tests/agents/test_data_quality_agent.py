# Test: Data Quality Agent - Data Age Auto-Pause

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
from agents.data_quality_agent import DataQualityAgent

class TestDataAgePause:
    """Test data staleness detection and auto-pause for MMM/MTA"""
    
    @pytest.fixture
    def agent(self):
        return DataQualityAgent(
            feature_flag_data_age_pause=True,
            staleness_threshold_hours=6.0,
            circuit_breaker_max_triggers=3,
            circuit_breaker_window_hours=1
        )
    
    def test_data_age_pause_mmm_when_stale(self, agent):
        """Test: 7h stale data triggers MMM pause"""
        # Arrange
        now = datetime.utcnow()
        stale_time = now - timedelta(hours=7)
        
        mock_data = Mock()
        mock_data.last_updated = stale_time
        mock_data.source = "ad_spend_daily"
        
        # Act
        with patch.object(agent, 'get_latest_data', return_value=mock_data):
            result = agent.check_data_age_and_pause()
        
        # Assert
        assert result['pause_mmm'] is True
        assert result['pause_mta'] is True
        assert result['data_age_hours'] == 7.0
        assert result['reason'] == "data_staleness_threshold_exceeded"
        assert 'data_quality.stale_data_paused' in result['metrics_emitted']
    
    def test_data_age_no_pause_when_fresh(self, agent):
        """Test: 5h fresh data does not trigger pause"""
        # Arrange
        now = datetime.utcnow()
        fresh_time = now - timedelta(hours=5)
        
        mock_data = Mock()
        mock_data.last_updated = fresh_time
        mock_data.source = "ad_spend_daily"
        
        # Act
        with patch.object(agent, 'get_latest_data', return_value=mock_data):
            result = agent.check_data_age_and_pause()
        
        # Assert
        assert result['pause_mmm'] is False
        assert result['pause_mta'] is False
        assert result['data_age_hours'] == 5.0
    
    def test_circuit_breaker_triggers_escalation(self, agent):
        """Test: >3 pauses in 1h triggers circuit breaker escalation"""
        # Arrange
        now = datetime.utcnow()
        
        # Simulate 4 pause triggers within 1 hour
        for i in range(4):
            stale_time = now - timedelta(hours=7)
            mock_data = Mock()
            mock_data.last_updated = stale_time
            
            with patch.object(agent, 'get_latest_data', return_value=mock_data):
                with patch.object(agent, 'get_current_time', return_value=now + timedelta(minutes=i*10)):
                    result = agent.check_data_age_and_pause()
        
        # Assert - 4th trigger should escalate
        assert result['circuit_breaker_triggered'] is True
        assert result['escalate_to_human'] is True
        assert 'data_quality.circuit_breaker_tripped' in result['metrics_emitted']
    
    def test_idempotency_key_prevents_duplicate_pauses(self, agent):
        """Test: Same check_id does not trigger duplicate pauses"""
        # Arrange
        check_id = "data_quality_check_20251019_090000"
        now = datetime.utcnow()
        stale_time = now - timedelta(hours=7)
        
        mock_data = Mock()
        mock_data.last_updated = stale_time
        
        # Act - First call
        with patch.object(agent, 'get_latest_data', return_value=mock_data):
            result1 = agent.check_data_age_and_pause(check_id=check_id)
        
        # Act - Second call with same check_id
        with patch.object(agent, 'get_latest_data', return_value=mock_data):
            result2 = agent.check_data_age_and_pause(check_id=check_id)
        
        # Assert
        assert result1['pause_mmm'] is True
        assert result2['pause_mmm'] is False  # Idempotent - no duplicate action
        assert result2['reason'] == "duplicate_check_id_skipped"
    
    def test_kill_switch_disables_pause(self, agent):
        """Test: Feature flag off disables pause logic"""
        # Arrange
        agent_disabled = DataQualityAgent(feature_flag_data_age_pause=False)
        now = datetime.utcnow()
        stale_time = now - timedelta(hours=7)
        
        mock_data = Mock()
        mock_data.last_updated = stale_time
        
        # Act
        with patch.object(agent_disabled, 'get_latest_data', return_value=mock_data):
            result = agent_disabled.check_data_age_and_pause()
        
        # Assert
        assert result['pause_mmm'] is False
        assert result['pause_mta'] is False
        assert result['reason'] == "feature_flag_disabled"
