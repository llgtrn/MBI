# MTA Agent Test Suite
"""
Tests for Multi-Touch Attribution Agent
Priority: P0 (Critical Path)
Capsule Refs: Q_010, Q_405, Q_406, Q_407, A_019
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
from agents.mta_agent import MTAAgent, ConversionPath, MTAResult, Touchpoint
from core.metrics import get_metric
import prometheus_client


class TestMTAKAnonymityEnforcement:
    """Q_010, A_019: Verify paths with k<10 are dropped"""
    
    def test_mta_k_anonymity_enforcement(self):
        """Given paths with k<10 → dropped; k>=10 → included"""
        agent = MTAAgent()
        
        # Arrange: Create paths with k=5 (below threshold) and k=15 (above)
        paths = [
            ConversionPath(
                path_id="p1",
                touchpoints=[
                    Touchpoint(channel="meta", timestamp_bucket="2025-10-19T10", position=1),
                    Touchpoint(channel="google", timestamp_bucket="2025-10-19T11", position=2)
                ],
                converted=True,
                conversions=5,  # k=5 < 10 → should be dropped
                revenue=500.0
            ),
            ConversionPath(
                path_id="p2",
                touchpoints=[
                    Touchpoint(channel="tiktok", timestamp_bucket="2025-10-19T09", position=1),
                    Touchpoint(channel="youtube", timestamp_bucket="2025-10-19T10", position=2)
                ],
                converted=True,
                conversions=15,  # k=15 >= 10 → should be included
                revenue=1500.0
            )
        ]
        
        # Act
        result = agent.compute_attribution(paths, lookback_days=30)
        
        # Assert
        assert result.dropped_paths_count == 1, "Expected 1 path dropped (p1 with k=5)"
        assert result.k_anonymity_violations == 1, "Expected 1 k-anonymity violation"
        
        # Verify p1 not in attribution
        assert "p1" not in result.attribution, "p1 should be dropped"
        
        # Verify p2 is in attribution
        assert len(result.attribution) > 0, "p2 should be included"
        
        # Verify metric emission
        metric_value = prometheus_client.REGISTRY.get_sample_value(
            'mta_k_anonymity_violations_total'
        )
        assert metric_value >= 1, "Metric mta_k_anonymity_violations_total should be incremented"
    
    def test_k_anonymity_threshold_configurable(self):
        """Verify k threshold is configurable"""
        agent = MTAAgent()
        assert agent.k_threshold == 10, "Default k threshold should be 10"
        
        # Test with different threshold
        agent_k20 = MTAAgent(k_threshold=20)
        assert agent_k20.k_threshold == 20


class TestBloomFilterCollisionSuppression:
    """Q_405: Hash collisions detected and suppressed"""
    
    def test_mta_bloom_filter_collision_suppression(self):
        """Given bloom filter collision → path suppressed"""
        agent = MTAAgent()
        
        # Arrange: Create paths
        path1 = ConversionPath(
            path_id="p1",
            touchpoints=[
                Touchpoint(channel="meta", timestamp_bucket="2025-10-19T10", position=1)
            ],
            converted=True,
            conversions=10,
            revenue=1000.0
        )
        path2 = ConversionPath(
            path_id="p2",
            touchpoints=[
                Touchpoint(channel="meta", timestamp_bucket="2025-10-19T10", position=1)
            ],
            converted=True,
            conversions=10,
            revenue=1000.0
        )
        
        # Mock bloom filter to return True (collision detected) on second add
        with patch.object(agent.bloom_filter, 'test') as mock_test:
            mock_test.side_effect = [False, True]  # First test False, second True (collision)
            
            # Act
            result = agent.compute_attribution([path1, path2])
            
            # Assert: Second path suppressed
            assert result.collision_suppressed_count == 1, "Expected 1 collision suppressed"
            assert result.dropped_paths_count >= 1, "At least 1 path should be dropped"
    
    def test_bloom_filter_false_positive_rate(self):
        """Verify bloom filter FPR < 1%"""
        agent = MTAAgent()
        
        false_positives = 0
        total_tests = 1000
        
        # Process unique paths
        for i in range(total_tests):
            path = ConversionPath(
                path_id=f"unique_path_{i}",
                touchpoints=[
                    Touchpoint(
                        channel="meta" if i % 2 == 0 else "google",
                        timestamp_bucket=f"2025-10-19T{i % 24:02d}",
                        position=1
                    )
                ],
                converted=True,
                conversions=10,
                revenue=100.0 * (i + 1)  # Unique revenue
            )
            
            # Check if bloom filter incorrectly flags as duplicate
            path_hash = agent._compute_path_hash(path)
            if agent.bloom_filter.test(path_hash):
                false_positives += 1
            agent.bloom_filter.add(path_hash)
        
        fpr = false_positives / total_tests
        assert fpr < 0.01, f"Bloom filter FPR must be <1%, was {fpr*100:.2f}%"


class TestMTAPathsDroppedMetric:
    """Q_010: Metric mta_paths_dropped_total emitted"""
    
    def test_mta_paths_dropped_metric_emission(self):
        """Given path with k<10 → metric mta_paths_dropped_total incremented"""
        agent = MTAAgent()
        
        # Arrange: Path with k=3 (below threshold)
        paths = [
            ConversionPath(
                path_id="p_low_k",
                touchpoints=[
                    Touchpoint(channel="meta", timestamp_bucket="2025-10-19T10", position=1)
                ],
                converted=True,
                conversions=3,  # k=3 < 10
                revenue=300.0
            )
        ]
        
        # Act
        agent.compute_attribution(paths)
        
        # Assert: Prometheus counter incremented
        metric_value = prometheus_client.REGISTRY.get_sample_value(
            'mta_paths_dropped_total',
            labels={'reason': 'k_anonymity'}
        )
        assert metric_value >= 1, "Metric mta_paths_dropped_total with reason=k_anonymity should be incremented"
        
        # Also check collision metric exists
        collision_metric = prometheus_client.REGISTRY.get_sample_value(
            'mta_paths_dropped_total',
            labels={'reason': 'collision'}
        )
        assert collision_metric is not None, "Collision metric should be registered"
    
    def test_paths_dropped_by_reason_breakdown(self):
        """Verify dropped paths are broken down by reason"""
        agent = MTAAgent()
        
        # Path dropped due to k-anonymity
        path_k = ConversionPath(
            path_id="p_k",
            touchpoints=[Touchpoint(channel="meta", timestamp_bucket="2025-10-19T10", position=1)],
            converted=True,
            conversions=2,
            revenue=200.0
        )
        
        agent.compute_attribution([path_k])
        
        # Verify both metrics exist
        k_anon_drops = prometheus_client.REGISTRY.get_sample_value(
            'mta_paths_dropped_total',
            labels={'reason': 'k_anonymity'}
        )
        
        collision_drops = prometheus_client.REGISTRY.get_sample_value(
            'mta_paths_dropped_total',
            labels={'reason': 'collision'}
        )
        
        assert k_anon_drops >= 1, "K-anonymity drops should be tracked"
        assert collision_drops is not None, "Collision drops should be tracked (may be 0)"


class TestMTAPathSuccessMetric:
    """Q_406: Path success metric p95<1% failed"""
    
    def test_path_success_rate_tracking(self):
        """Given 1000 paths, 5 failures → success_rate >= 99%"""
        agent = MTAAgent()
        
        # Simulate 1000 paths with 5 failures
        for i in range(1000):
            path = ConversionPath(
                path_id=f"path_{i}",
                touchpoints=[
                    Touchpoint(channel="meta", timestamp_bucket="2025-10-19T10", position=1),
                    Touchpoint(channel="google", timestamp_bucket="2025-10-19T11", position=2)
                ],
                converted=True,
                conversions=15,  # k>=10
                revenue=100.0
            )
            if i < 5:
                # Simulate failure (network timeout)
                with pytest.raises(Exception):
                    agent.process_path_with_failure(path)
            else:
                agent.process_path(path)
        
        # Check metric
        success_rate = get_metric("mta_path_success_rate")
        assert success_rate >= 0.99, f"Expected ≥99%, got {success_rate*100:.2f}%"
    
    def test_path_failure_types(self):
        """Test different failure modes are tracked"""
        agent = MTAAgent()
        
        failures = {
            "network_timeout": 0,
            "redis_unavailable": 0,
            "invalid_path": 0
        }
        
        # Simulate failures
        agent.record_failure("network_timeout")
        agent.record_failure("redis_unavailable")
        
        metrics = agent.get_failure_breakdown()
        assert "network_timeout" in metrics
        assert "redis_unavailable" in metrics


class TestRedisBackupRecovery:
    """Q_407: Redis AOF + S3 backup RTO<30min"""
    
    @pytest.mark.integration
    def test_redis_aof_enabled(self):
        """Verify AOF persistence is enabled"""
        import redis
        r = redis.Redis(host='localhost', port=6379, decode_responses=True)
        
        config = r.config_get('appendonly')
        assert config['appendonly'] == 'yes', "AOF must be enabled"
    
    @pytest.mark.integration
    def test_s3_backup_schedule(self):
        """Verify S3 backup cron is configured"""
        from config.redis import get_backup_schedule
        
        schedule = get_backup_schedule()
        assert schedule['enabled'] is True
        assert schedule['interval_seconds'] == 900  # 15 minutes
        assert 's3_bucket' in schedule
    
    @pytest.mark.slow
    def test_recovery_time_objective(self):
        """Simulate Redis failure and measure recovery time"""
        agent = MTAAgent()
        start_time = datetime.utcnow()
        
        # Simulate Redis failure
        agent.simulate_redis_failure()
        
        # Trigger recovery from S3
        agent.recover_from_backup()
        
        recovery_time = (datetime.utcnow() - start_time).total_seconds()
        assert recovery_time < 1800, f"RTO must be <30min, was {recovery_time/60:.1f}min"


class TestKAnonSuppression:
    """A_019: k-anonymity suppression k>=10"""
    
    def test_suppress_small_paths(self):
        """Given path with user_count<10 → suppressed"""
        agent = MTAAgent()
        
        # Path with only 5 users (k<10)
        path = ConversionPath(
            path_id="rare_path",
            touchpoints=[
                Touchpoint(channel="tiktok", timestamp_bucket="2025-10-19T09", position=1),
                Touchpoint(channel="youtube", timestamp_bucket="2025-10-19T10", position=2)
            ],
            converted=True,
            conversions=5,  # k=5 < 10
            revenue=500.0
        )
        
        attribution = agent.compute_attribution([path], lookback_days=30)
        
        # Path should be suppressed
        assert attribution.dropped_paths_count >= 1, "Path with k<10 should be dropped"
        assert attribution.k_anonymity_violations >= 1, "K-anonymity violation should be recorded"
    
    def test_keep_sufficient_paths(self):
        """Given path with user_count>=10 → included"""
        agent = MTAAgent()
        
        path = ConversionPath(
            path_id="common_path",
            touchpoints=[
                Touchpoint(channel="meta", timestamp_bucket="2025-10-19T09", position=1),
                Touchpoint(channel="google", timestamp_bucket="2025-10-19T10", position=2)
            ],
            converted=True,
            conversions=50,  # k=50 >= 10
            revenue=5000.0
        )
        
        attribution = agent.compute_attribution([path], lookback_days=30)
        
        assert len(attribution.attribution) > 0, "Path with k>=10 should be included"
        assert attribution.dropped_paths_count == 0, "No paths should be dropped with k>=10"
    
    def test_aggregate_revenue_privacy(self):
        """Q_426: Suppressed paths aggregate revenue without PII"""
        agent = MTAAgent()
        
        paths = [
            ConversionPath(
                path_id=f"path_{i}",
                touchpoints=[Touchpoint(channel="meta", timestamp_bucket="2025-10-19T10", position=1)],
                converted=True,
                conversions=2,  # k<10
                revenue=200.0
            )
            for i in range(5)
        ]
        
        attribution = agent.compute_attribution(paths, lookback_days=30)
        
        # All paths suppressed individually
        assert attribution.dropped_paths_count >= 5, "All paths with k<10 should be dropped"


class TestMTAMetrics:
    """Overall MTA metrics and observability"""
    
    def test_processing_latency(self):
        """Verify p95 latency < 500ms"""
        agent = MTAAgent()
        
        latencies = []
        for i in range(100):
            start = datetime.utcnow()
            path = ConversionPath(
                path_id=f"path_{i}",
                touchpoints=[Touchpoint(channel="meta", timestamp_bucket="2025-10-19T10", position=1)],
                converted=True,
                conversions=10,
                revenue=100.0
            )
            agent.process_path(path)
            latency = (datetime.utcnow() - start).total_seconds() * 1000
            latencies.append(latency)
        
        p95 = sorted(latencies)[94]  # 95th percentile
        assert p95 < 500, f"p95 latency must be <500ms, was {p95:.1f}ms"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
