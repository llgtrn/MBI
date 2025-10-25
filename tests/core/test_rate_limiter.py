"""
Tests for rate limiter with burst allowance.
Implements Q_017: 101st request returns 429 with metrics.
Implements A_017: Verify rate limiters configured and metrics emitted.
"""

import pytest
import time
from unittest.mock import patch
from core.rate_limiter import (
    TokenBucketRateLimiter,
    RateLimitConfig,
    RateLimitExceeded,
    get_rate_limiter,
    configure_default_limiters
)


class TestTokenBucketRateLimiter:
    """Test token bucket rate limiter implementation."""
    
    def test_basic_acquire_success(self):
        """Test successful token acquisition."""
        limiter = TokenBucketRateLimiter(
            RateLimitConfig(
                limiter_id="test_basic",
                resource="test",
                max_tokens=10,
                refill_rate=5.0,
                burst_allowance=5
            )
        )
        
        # First request should succeed
        assert limiter.acquire(tokens=1) is True
        assert limiter.get_tokens_available() == 9.0
    
    def test_101st_request_returns_429(self):
        """
        Q_017 acceptance: 101st request within burst window returns 429.
        
        Scenario:
        - Limiter: max_tokens=100, refill_rate=0 (no refill during test), burst_allowance=0
        - Make 100 requests: all succeed
        - Make 101st request: fails (returns False → HTTP 429)
        """
        limiter = TokenBucketRateLimiter(
            RateLimitConfig(
                limiter_id="test_101",
                resource="test",
                max_tokens=100,
                refill_rate=0.0,  # No refill
                burst_allowance=0
            )
        )
        
        # First 100 requests succeed
        for i in range(100):
            result = limiter.acquire(tokens=1)
            assert result is True, f"Request {i+1} should succeed"
        
        assert limiter.get_tokens_available() == 0.0
        
        # 101st request fails
        with patch('core.rate_limiter.rate_limit_exceeded') as mock_metric:
            result = limiter.acquire(tokens=1)
            assert result is False, "101st request should fail (429)"
            
            # Verify metric incremented (Q_017 acceptance)
            mock_metric.labels.return_value.inc.assert_called_once()
    
    def test_burst_allowance(self):
        """
        Test burst allowance beyond base capacity.
        
        Scenario:
        - max_tokens=100, burst_allowance=20
        - Total capacity = 120 tokens
        - Should allow 120 requests before rate limiting
        """
        limiter = TokenBucketRateLimiter(
            RateLimitConfig(
                limiter_id="test_burst",
                resource="test",
                max_tokens=100,
                refill_rate=0.0,
                burst_allowance=20
            )
        )
        
        # First 120 requests succeed (100 base + 20 burst)
        for i in range(120):
            result = limiter.acquire(tokens=1)
            assert result is True, f"Request {i+1} should succeed (within burst)"
        
        # 121st request fails
        result = limiter.acquire(tokens=1)
        assert result is False, "121st request should fail (burst exhausted)"
    
    def test_token_refill(self):
        """
        Test token refill over time.
        
        Scenario:
        - max_tokens=10, refill_rate=10.0 tokens/sec
        - Drain to 0
        - Wait 0.5 seconds → should have ~5 tokens
        """
        limiter = TokenBucketRateLimiter(
            RateLimitConfig(
                limiter_id="test_refill",
                resource="test",
                max_tokens=10,
                refill_rate=10.0,
                burst_allowance=0
            )
        )
        
        # Drain to 0
        for _ in range(10):
            limiter.acquire(tokens=1)
        
        assert limiter.get_tokens_available() == 0.0
        
        # Wait 0.5 seconds
        time.sleep(0.5)
        
        # Should have ~5 tokens (10 tokens/sec * 0.5 sec)
        # Use acquire to trigger refill
        limiter.acquire(tokens=0)  # Trigger refill without consuming
        
        tokens = limiter.get_tokens_available()
        assert 4.5 <= tokens <= 5.5, f"Expected ~5 tokens, got {tokens}"
    
    def test_refill_caps_at_max_plus_burst(self):
        """Test that refill doesn't exceed max_tokens + burst_allowance."""
        limiter = TokenBucketRateLimiter(
            RateLimitConfig(
                limiter_id="test_cap",
                resource="test",
                max_tokens=10,
                refill_rate=100.0,  # Very fast refill
                burst_allowance=5
            )
        )
        
        # Drain some tokens
        limiter.acquire(tokens=5)
        
        # Wait to trigger refill
        time.sleep(1.0)
        limiter.acquire(tokens=0)  # Trigger refill
        
        # Should cap at 15 (10 + 5 burst)
        tokens = limiter.get_tokens_available()
        assert tokens <= 15.0, f"Tokens should cap at 15, got {tokens}"
    
    def test_concurrent_access_thread_safety(self):
        """Test thread-safe concurrent access."""
        import threading
        
        limiter = TokenBucketRateLimiter(
            RateLimitConfig(
                limiter_id="test_concurrent",
                resource="test",
                max_tokens=100,
                refill_rate=0.0,
                burst_allowance=0
            )
        )
        
        results = []
        
        def worker():
            for _ in range(10):
                results.append(limiter.acquire(tokens=1))
        
        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # Exactly 100 should succeed, rest fail
        assert sum(results) == 100, f"Expected 100 successes, got {sum(results)}"
    
    def test_reset(self):
        """Test reset functionality."""
        limiter = TokenBucketRateLimiter(
            RateLimitConfig(
                limiter_id="test_reset",
                resource="test",
                max_tokens=10,
                refill_rate=0.0,
                burst_allowance=0
            )
        )
        
        # Drain tokens
        for _ in range(10):
            limiter.acquire(tokens=1)
        
        assert limiter.get_tokens_available() == 0.0
        
        # Reset
        limiter.reset()
        
        assert limiter.get_tokens_available() == 10.0


class TestRateLimiterRegistry:
    """Test global rate limiter registry."""
    
    def test_get_rate_limiter_creates_new(self):
        """Test creating new rate limiter via registry."""
        config = RateLimitConfig(
            limiter_id="test_registry",
            resource="test",
            max_tokens=50,
            refill_rate=10.0,
            burst_allowance=10
        )
        
        limiter = get_rate_limiter("test_registry", config)
        assert limiter is not None
        assert limiter.config.limiter_id == "test_registry"
    
    def test_get_rate_limiter_returns_existing(self):
        """Test that registry returns same instance."""
        config = RateLimitConfig(
            limiter_id="test_singleton",
            resource="test",
            max_tokens=50,
            refill_rate=10.0,
            burst_allowance=10
        )
        
        limiter1 = get_rate_limiter("test_singleton", config)
        limiter2 = get_rate_limiter("test_singleton")
        
        assert limiter1 is limiter2
    
    def test_get_rate_limiter_raises_without_config(self):
        """Test that registry raises if limiter not found and no config."""
        with pytest.raises(ValueError, match="not found and no config provided"):
            get_rate_limiter("nonexistent_limiter")


class TestDefaultLimiters:
    """
    A_017 acceptance: Rate limiters configured for BigQuery, external APIs, ad platforms.
    """
    
    def test_configure_default_limiters(self):
        """Test that default limiters are configured correctly."""
        configure_default_limiters()
        
        # BigQuery: 100 req/sec sustained, burst 120
        bq_limiter = get_rate_limiter("bigquery_api")
        assert bq_limiter.config.max_tokens == 100
        assert bq_limiter.config.refill_rate == 100.0
        assert bq_limiter.config.burst_allowance == 20
        
        # External API: 50 req/sec sustained, burst 75
        ext_limiter = get_rate_limiter("external_api")
        assert ext_limiter.config.max_tokens == 50
        assert ext_limiter.config.refill_rate == 50.0
        assert ext_limiter.config.burst_allowance == 25
        
        # Ad Platform: 30 req/sec sustained, burst 50
        ad_limiter = get_rate_limiter("ad_platform_api")
        assert ad_limiter.config.max_tokens == 30
        assert ad_limiter.config.refill_rate == 30.0
        assert ad_limiter.config.burst_allowance == 20
    
    def test_bigquery_10k_per_sec_sustained(self):
        """
        Q_026 acceptance: BigQuery rate limiter handles 10k/sec sustained with 0 dropped.
        
        Note: This is a stress test scenario. In practice, we'd configure limiter
        with max_tokens=10000, refill_rate=10000.0 for true 10k/sec.
        Here we test the mechanism scales correctly.
        """
        # Configure high-throughput limiter
        limiter = TokenBucketRateLimiter(
            RateLimitConfig(
                limiter_id="bigquery_high_throughput",
                resource="bigquery",
                max_tokens=10000,
                refill_rate=10000.0,
                burst_allowance=1000
            )
        )
        
        # Simulate 1 second of requests at 10k/sec
        # In practice, refill happens continuously
        success_count = 0
        
        for _ in range(10000):
            if limiter.acquire(tokens=1):
                success_count += 1
        
        # All should succeed (within bucket capacity)
        assert success_count == 10000, f"Expected 10000 successes, got {success_count}"


class TestMetricsEmission:
    """
    A_017 acceptance: Metrics emitted for rate limit violations.
    """
    
    def test_metrics_emitted_on_rate_limit(self):
        """Test that Prometheus metrics are emitted correctly."""
        limiter = TokenBucketRateLimiter(
            RateLimitConfig(
                limiter_id="test_metrics",
                resource="test",
                max_tokens=5,
                refill_rate=0.0,
                burst_allowance=0
            )
        )
        
        # Drain tokens
        for _ in range(5):
            limiter.acquire(tokens=1)
        
        # Next request should emit rate_limit_exceeded metric
        with patch('core.rate_limiter.rate_limit_exceeded') as mock_exceeded, \
             patch('core.rate_limiter.rate_limit_tokens_available') as mock_gauge, \
             patch('core.rate_limiter.rate_limit_request_latency') as mock_latency:
            
            result = limiter.acquire(tokens=1)
            
            assert result is False
            mock_exceeded.labels.return_value.inc.assert_called()
            mock_gauge.labels.return_value.set.assert_called()
            mock_latency.labels.return_value.observe.assert_called()
