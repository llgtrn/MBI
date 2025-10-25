"""
Crisis Detection Agent Tests - Tier-1 Sources and Velocity Baseline
Tests for tier-1 domain validation, risk score capping, and 24h baseline computation
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict
from agents.crisis_detection_agent import (
    CrisisDetectionAgent,
    CrisisConfig,
    CrisisSource,
    CrisisBrief,
    VelocityBaseline,
    TierLevel
)


class TestCrisisTier1Sources:
    """Test tier-1 source validation and risk capping (Q_018, A_022)"""
    
    @pytest.fixture
    def crisis_config(self):
        return CrisisConfig(
            tier1_domains=["reuters.com", "apnews.com", "bloomberg.com"],
            tier1_risk_cap=0.6,
            baseline_window_hours=24
        )
    
    @pytest.fixture
    def crisis_agent(self, crisis_config):
        return CrisisDetectionAgent(config=crisis_config)
    
    @pytest.mark.asyncio
    async def test_crisis_tier1_sources_risk_cap(self, crisis_agent):
        """Test: 2+ tier-1 sources → risk_score ≤ 0.6 (Q_018)"""
        # Sources: 2 tier-1, 1 tier-2
        sources = [
            CrisisSource(
                id="src_001",
                url="https://reuters.com/article/crisis",
                title="Crisis report",
                text="Detailed crisis information",
                tier=TierLevel.TIER1,
                domain="reuters.com"
            ),
            CrisisSource(
                id="src_002",
                url="https://bloomberg.com/news/crisis",
                title="Crisis analysis",
                text="Bloomberg analysis",
                tier=TierLevel.TIER1,
                domain="bloomberg.com"
            ),
            CrisisSource(
                id="src_003",
                url="https://techcrunch.com/crisis",
                title="Crisis coverage",
                text="Tech coverage",
                tier=TierLevel.TIER2,
                domain="techcrunch.com"
            )
        ]
        
        # Compute risk score with tier-1 cap
        brief = await crisis_agent.verify_crisis(
            topic_id="crisis_001",
            sources=sources,
            velocity=5.2  # High velocity would normally give high risk
        )
        
        # Assertions: risk capped at 0.6 due to 2 tier-1 sources
        assert brief.risk_score <= 0.6
        assert brief.tier1_source_count == 2
        assert brief.risk_cap_applied is True
        assert "tier1_sources" in brief.metadata
    
    @pytest.mark.asyncio
    async def test_crisis_no_tier1_sources_no_cap(self, crisis_agent):
        """Test: 0 tier-1 sources → no risk cap applied"""
        sources = [
            CrisisSource(
                id="src_001",
                url="https://techcrunch.com/crisis",
                title="Crisis report",
                text="Coverage",
                tier=TierLevel.TIER2,
                domain="techcrunch.com"
            ),
            CrisisSource(
                id="src_002",
                url="https://reddit.com/r/crisis",
                title="Discussion",
                text="Reddit discussion",
                tier=TierLevel.TIER3,
                domain="reddit.com"
            )
        ]
        
        brief = await crisis_agent.verify_crisis(
            topic_id="crisis_002",
            sources=sources,
            velocity=5.2
        )
        
        # No cap: risk can exceed 0.6
        assert brief.tier1_source_count == 0
        assert brief.risk_cap_applied is False
        assert brief.risk_score > 0.6  # High velocity with no tier-1 cap
    
    @pytest.mark.asyncio
    async def test_crisis_tier1_boundary_one_source(self, crisis_agent):
        """Test: exactly 1 tier-1 source → no cap (need >=2)"""
        sources = [
            CrisisSource(
                id="src_001",
                url="https://reuters.com/article/crisis",
                title="Reuters report",
                text="Coverage",
                tier=TierLevel.TIER1,
                domain="reuters.com"
            ),
            CrisisSource(
                id="src_002",
                url="https://twitter.com/user/status",
                title="Tweet",
                text="Social media",
                tier=TierLevel.TIER3,
                domain="twitter.com"
            )
        ]
        
        brief = await crisis_agent.verify_crisis(
            topic_id="crisis_003",
            sources=sources,
            velocity=5.0
        )
        
        # Only 1 tier-1: no cap
        assert brief.tier1_source_count == 1
        assert brief.risk_cap_applied is False
    
    @pytest.mark.asyncio
    async def test_crisis_tier1_domain_classification(self, crisis_agent):
        """Test: tier-1 domains correctly classified from config"""
        sources = [
            CrisisSource(
                id="src_001",
                url="https://reuters.com/article",
                domain="reuters.com"
            ),
            CrisisSource(
                id="src_002",
                url="https://apnews.com/article",
                domain="apnews.com"
            ),
            CrisisSource(
                id="src_003",
                url="https://bloomberg.com/news",
                domain="bloomberg.com"
            ),
            CrisisSource(
                id="src_004",
                url="https://cnn.com/article",
                domain="cnn.com"  # Not in tier-1 list
            )
        ]
        
        classified = crisis_agent.classify_source_tiers(sources)
        
        tier1_count = sum(1 for s in classified if s.tier == TierLevel.TIER1)
        assert tier1_count == 3  # reuters, apnews, bloomberg


class TestCrisisVelocityBaseline:
    """Test velocity baseline computation (Q_019, A_022)"""
    
    @pytest.fixture
    def crisis_config(self):
        return CrisisConfig(
            tier1_domains=["reuters.com", "apnews.com"],
            baseline_window_hours=24
        )
    
    @pytest.fixture
    def crisis_agent(self, crisis_config):
        return CrisisDetectionAgent(config=crisis_config)
    
    @pytest.mark.asyncio
    async def test_crisis_velocity_baseline_24h(self, crisis_agent):
        """Test: baseline mean/std computed over 24h window (Q_019)"""
        # Generate 24 hours of velocity data (hourly)
        now = datetime.utcnow()
        velocity_data = []
        
        for i in range(24):
            timestamp = now - timedelta(hours=23-i)
            velocity = np.random.normal(loc=2.0, scale=0.5)  # Mean=2, std=0.5
            velocity_data.append({
                "timestamp": timestamp,
                "velocity": velocity,
                "topic_id": "topic_001"
            })
        
        # Compute baseline
        baseline = await crisis_agent.compute_velocity_baseline(
            topic_id="topic_001",
            velocity_data=velocity_data
        )
        
        # Assertions
        assert baseline is not None
        assert isinstance(baseline, VelocityBaseline)
        assert baseline.window_hours == 24
        assert baseline.data_points == 24
        
        # Check mean/std are reasonable (near 2.0, 0.5)
        assert 1.5 <= baseline.mean <= 2.5
        assert 0.2 <= baseline.std <= 0.8
        assert baseline.timestamp is not None
    
    @pytest.mark.asyncio
    async def test_crisis_velocity_baseline_insufficient_data(self, crisis_agent):
        """Test: baseline handles insufficient data gracefully"""
        # Only 5 hours of data
        now = datetime.utcnow()
        velocity_data = [
            {"timestamp": now - timedelta(hours=i), "velocity": 2.0}
            for i in range(5)
        ]
        
        baseline = await crisis_agent.compute_velocity_baseline(
            topic_id="topic_002",
            velocity_data=velocity_data
        )
        
        # Should return baseline but flag insufficient data
        assert baseline.data_points == 5
        assert baseline.insufficient_data is True
    
    @pytest.mark.asyncio
    async def test_crisis_velocity_normalized_by_baseline(self, crisis_agent):
        """Test: velocity normalized using baseline for risk calculation"""
        # Set baseline
        baseline = VelocityBaseline(
            topic_id="topic_001",
            mean=2.0,
            std=0.5,
            window_hours=24,
            data_points=24,
            timestamp=datetime.utcnow()
        )
        
        crisis_agent.set_baseline("topic_001", baseline)
        
        # Current velocity
        current_velocity = 4.0  # 4 std deviations above mean
        
        # Compute normalized score
        normalized = crisis_agent.normalize_velocity(
            velocity=current_velocity,
            baseline=baseline
        )
        
        # Assertions
        assert normalized.z_score == pytest.approx(4.0, abs=0.1)
        assert normalized.sigma_above_mean == 4.0
    
    @pytest.mark.asyncio
    async def test_crisis_baseline_metrics_emission(self, crisis_agent, mocker):
        """Test: baseline metrics emitted (Q_019)"""
        mock_gauge_mean = mocker.patch('agents.crisis_detection_agent.crisis_velocity_baseline_mean')
        mock_gauge_std = mocker.patch('agents.crisis_detection_agent.crisis_velocity_baseline_std')
        
        # Compute baseline
        now = datetime.utcnow()
        velocity_data = [
            {"timestamp": now - timedelta(hours=i), "velocity": 2.0 + np.random.random()}
            for i in range(24)
        ]
        
        baseline = await crisis_agent.compute_velocity_baseline(
            topic_id="topic_001",
            velocity_data=velocity_data
        )
        
        # Verify metrics emitted
        mock_gauge_mean.labels.assert_called_with(topic_id="topic_001")
        mock_gauge_std.labels.assert_called_with(topic_id="topic_001")
        
        mock_gauge_mean.labels().set.assert_called_once()
        mock_gauge_std.labels().set.assert_called_once()


class TestCrisisConfig:
    """Test CrisisConfig schema (A_022)"""
    
    def test_crisis_config_tier1_domains(self):
        """Test: config includes tier1_domains list"""
        config = CrisisConfig(
            tier1_domains=["reuters.com", "apnews.com", "bloomberg.com"]
        )
        
        assert config.tier1_domains == ["reuters.com", "apnews.com", "bloomberg.com"]
        assert len(config.tier1_domains) == 3
    
    def test_crisis_config_baseline_window(self):
        """Test: config includes baseline_window_hours"""
        config = CrisisConfig(
            baseline_window_hours=24
        )
        
        assert config.baseline_window_hours == 24
    
    def test_crisis_config_validation(self):
        """Test: config validation"""
        # Valid config
        config = CrisisConfig(
            tier1_domains=["reuters.com"],
            tier1_risk_cap=0.6,
            baseline_window_hours=24
        )
        
        assert config.tier1_risk_cap == 0.6
        
        # Invalid: risk cap > 1.0
        with pytest.raises(ValueError):
            CrisisConfig(tier1_risk_cap=1.5)


class TestCrisisIntegration:
    """Integration tests for crisis detection with tier-1 and baseline"""
    
    @pytest.fixture
    def crisis_agent(self):
        config = CrisisConfig(
            tier1_domains=["reuters.com", "apnews.com", "bloomberg.com"],
            tier1_risk_cap=0.6,
            baseline_window_hours=24
        )
        return CrisisDetectionAgent(config=config)
    
    @pytest.mark.asyncio
    async def test_crisis_end_to_end_with_tier1_and_baseline(self, crisis_agent):
        """Test: end-to-end crisis detection with tier-1 sources and baseline normalization"""
        # 1. Set up baseline
        now = datetime.utcnow()
        velocity_data = [
            {"timestamp": now - timedelta(hours=i), "velocity": 1.0 + np.random.random() * 0.5}
            for i in range(24)
        ]
        
        baseline = await crisis_agent.compute_velocity_baseline(
            topic_id="crisis_topic",
            velocity_data=velocity_data
        )
        
        crisis_agent.set_baseline("crisis_topic", baseline)
        
        # 2. Detect crisis with tier-1 sources
        sources = [
            CrisisSource(
                id="src_001",
                url="https://reuters.com/article",
                title="Major crisis",
                text="Crisis details",
                tier=TierLevel.TIER1,
                domain="reuters.com"
            ),
            CrisisSource(
                id="src_002",
                url="https://bloomberg.com/news",
                title="Crisis analysis",
                text="Analysis",
                tier=TierLevel.TIER1,
                domain="bloomberg.com"
            )
        ]
        
        # High velocity (5 sigma above baseline)
        current_velocity = baseline.mean + (5 * baseline.std)
        
        brief = await crisis_agent.verify_crisis(
            topic_id="crisis_topic",
            sources=sources,
            velocity=current_velocity
        )
        
        # Assertions
        assert brief.tier1_source_count == 2
        assert brief.risk_score <= 0.6  # Capped
        assert brief.risk_cap_applied is True
        assert brief.velocity_normalized is not None
        assert brief.velocity_normalized.z_score >= 4.5
    
    @pytest.mark.asyncio
    async def test_crisis_high_velocity_no_tier1_sources(self, crisis_agent):
        """Test: high velocity + no tier-1 → high risk (no cap)"""
        # Set baseline
        baseline = VelocityBaseline(
            topic_id="crisis_topic",
            mean=1.0,
            std=0.2,
            window_hours=24,
            data_points=24,
            timestamp=datetime.utcnow()
        )
        crisis_agent.set_baseline("crisis_topic", baseline)
        
        # No tier-1 sources
        sources = [
            CrisisSource(
                id="src_001",
                url="https://reddit.com/r/topic",
                title="Discussion",
                text="Reddit thread",
                tier=TierLevel.TIER3,
                domain="reddit.com"
            )
        ]
        
        # Very high velocity (10 sigma)
        current_velocity = baseline.mean + (10 * baseline.std)
        
        brief = await crisis_agent.verify_crisis(
            topic_id="crisis_topic",
            sources=sources,
            velocity=current_velocity
        )
        
        # High risk, no cap
        assert brief.tier1_source_count == 0
        assert brief.risk_score > 0.8  # High risk
        assert brief.risk_cap_applied is False


class TestCrisisMetrics:
    """Test crisis detection metrics"""
    
    @pytest.mark.asyncio
    async def test_baseline_metrics_labels(self, mocker):
        """Test: baseline metrics use topic_id labels"""
        mock_mean = mocker.patch('agents.crisis_detection_agent.crisis_velocity_baseline_mean')
        mock_std = mocker.patch('agents.crisis_detection_agent.crisis_velocity_baseline_std')
        
        config = CrisisConfig(baseline_window_hours=24)
        agent = CrisisDetectionAgent(config=config)
        
        now = datetime.utcnow()
        velocity_data = [
            {"timestamp": now - timedelta(hours=i), "velocity": 2.0}
            for i in range(24)
        ]
        
        await agent.compute_velocity_baseline("test_topic", velocity_data)
        
        # Verify labels
        mock_mean.labels.assert_called_once_with(topic_id="test_topic")
        mock_std.labels.assert_called_once_with(topic_id="test_topic")
