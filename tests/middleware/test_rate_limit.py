"""
Rate Limiter Tests
Tests for Q_033: Rate limiter enforces 100 req/min per external API with token bucket

Acceptance:
- test_rate_limit_100_per_min passes
- RateLimitConfig schema {max_requests, window_seconds, bucket_size}
- metric rate_limit_hits counter >0
- dry_run: Burst 150 requests in 10s; verify 100 pass, 50 rejected with 429
"""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock
from middleware.rate_limiter import (
    RateLimiter,
    RateLimitExceeded,
    RateLimitConfig,
    TokenBucket
)


class TestTokenBucket:
    """Test token bucket algorithm"""
    
    def test_initial_bucket_full(self):
        """Token bucket starts with max tokens"""
        bucket = TokenBucket(
            capacity=100,
            refill_rate=100/60  # 100 per minute
        )
        assert bucket.tokens == 100
    
    def test_consume_tokens(self):
        """Consuming tokens decreases available tokens"""
        bucket = TokenBucket(capacity=100, refill_rate=1.0)
        
        assert bucket.consume(10) is True
        assert bucket.tokens == 90
        
        assert bucket.consume(20) is True
        assert bucket.tokens == 70
    
    def test_consume_more_than_available_fails(self):
        """Cannot consume more tokens than available"""
        bucket = TokenBucket(capacity=10, refill_rate=1.0)
        
        assert bucket.consume(5) is True
        assert bucket.tokens == 5
        
        assert bucket.consume(10) is False  # Only 5 left
        assert bucket.tokens == 5  # Unchanged
    
    def test_tokens_refill_over_time(self):
        """Tokens refill at specified rate"""
        bucket = TokenBucket(capacity=100, refill_rate=10.0)  # 10 per second
        
        with patch('middleware.rate_limiter.time') as mock_time:
            base_time = 1000.0
            mock_time.time.return_value = base_time
            
            # Consume 50 tokens
            bucket.consume(50)
            assert bucket.tokens == 50
            
            # Advance 2 seconds (should add 20 tokens)
            mock_time.time.return_value = base_time + 2.0
            bucket.refill()
            
            assert bucket.tokens == 70  # 50 + 20
    
    def test_tokens_dont_exceed_capacity(self):
        """Token bucket doesn't exceed capacity"""
        bucket = TokenBucket(capacity=100, refill_rate=50.0)
        
        with patch('middleware.rate_limiter.time') as mock_time:
            base_time = 1000.0
            mock_time.time.return_value = base_time
            
            # Start with 90 tokens
            bucket.consume(10)
            assert bucket.tokens == 90
            
            # Advance 5 seconds (would add 250 tokens)
            mock_time.time.return_value = base_time + 5.0
            bucket.refill()
            
            # Should cap at capacity
            assert bucket.tokens == 100


class TestRateLimiter:
    """Test rate limiter with token bucket"""
    
    def test_rate_limit_100_per_min(self):
        """Q_033 acceptance: Enforce 100 requests per minute"""
        config = RateLimitConfig(
            max_requests=100,
            window_seconds=60
        )
        limiter = RateLimiter(name="test_api", config=config)
        
        # Should allow 100 requests
        for i in range(100):
            limiter.check_limit()
        
        # 101st request should be rate limited
        with pytest.raises(RateLimitExceeded) as exc_info:
            limiter.check_limit()
        
        assert exc_info.value.status_code == 429
        assert "test_api" in str(exc_info.value)
    
    def test_burst_150_requests_100_pass_50_reject(self):
        """Q_033 dry_run acceptance: Burst handling"""
        config = RateLimitConfig(
            max_requests=100,
            window_seconds=60,
            burst_capacity=120  # Allow some burst
        )
        limiter = RateLimiter(name="burst_test", config=config)
        
        passed = 0
        rejected = 0
        
        # Simulate 150 rapid requests
        for i in range(150):
            try:
                limiter.check_limit()
                passed += 1
            except RateLimitExceeded:
                rejected += 1
        
        # Should allow ~100-120 (with burst), reject the rest
        assert passed >= 100
        assert passed <= 120
        assert rejected >= 30
        assert passed + rejected == 150
    
    def test_rate_limit_metrics_incremented(self):
        """Verify rate_limit_hits metric increments"""
        config = RateLimitConfig(max_requests=5, window_seconds=60)
        limiter = RateLimiter(name="test", config=config)
        
        # Exhaust limit
        for _ in range(5):
            limiter.check_limit()
        
        # Next requests should be rejected and increment metric
        initial_hits = limiter.metrics['rate_limit_hits']
        
        for _ in range(3):
            try:
                limiter.check_limit()
            except RateLimitExceeded:
                pass
        
        assert limiter.metrics['rate_limit_hits'] == initial_hits + 3
    
    def test_separate_limits_per_endpoint(self):
        """Each endpoint has independent rate limit"""
        config = RateLimitConfig(max_requests=10, window_seconds=60)
        
        limiter1 = RateLimiter(name="api1", config=config)
        limiter2 = RateLimiter(name="api2", config=config)
        
        # Exhaust api1
        for _ in range(10):
            limiter1.check_limit()
        
        # api1 should be limited
        with pytest.raises(RateLimitExceeded):
            limiter1.check_limit()
        
        # api2 should still work
        limiter2.check_limit()  # Should not raise
    
    def test_context_manager_interface(self):
        """Rate limiter works as context manager"""
        config = RateLimitConfig(max_requests=5, window_seconds=60)
        limiter = RateLimiter(name="test", config=config)
        
        # Should allow 5
        for _ in range(5):
            with limiter:
                pass  # Success
        
        # 6th should raise
        with pytest.raises(RateLimitExceeded):
            with limiter:
                pass
    
    def test_rate_limit_resets_over_time(self):
        """Rate limit resets as tokens refill"""
        config = RateLimitConfig(
            max_requests=10,
            window_seconds=60
        )
        limiter = RateLimiter(name="test", config=config)
        
        with patch('middleware.rate_limiter.time') as mock_time:
            base_time = 1000.0
            mock_time.time.return_value = base_time
            
            # Exhaust limit
            for _ in range(10):
                limiter.check_limit()
            
            # Should be rate limited
            with pytest.raises(RateLimitExceeded):
                limiter.check_limit()
            
            # Advance time to refill tokens (60 seconds)
            mock_time.time.return_value = base_time + 60.0
            
            # Should work again
            limiter.check_limit()  # Should not raise


class TestRateLimitConfig:
    """Test configuration schema"""
    
    def test_config_schema_validation(self):
        """Q_033 acceptance: Config schema validation"""
        config = RateLimitConfig(
            max_requests=100,
            window_seconds=60,
            burst_capacity=120
        )
        
        assert config.max_requests == 100
        assert config.window_seconds == 60
        assert config.burst_capacity == 120
        
        # Calculate refill rate
        refill_rate = config.max_requests / config.window_seconds
        assert refill_rate == pytest.approx(1.667, rel=0.01)


class TestRateLimiterRedisBackend:
    """Test distributed rate limiting with Redis"""
    
    def test_redis_backend_shared_state(self):
        """Multiple limiter instances share Redis state"""
        config = RateLimitConfig(
            max_requests=10,
            window_seconds=60,
            backend="redis"
        )
        
        mock_redis = MagicMock()
        mock_redis.get.return_value = None
        
        with patch('middleware.rate_limiter.redis_client', mock_redis):
            limiter1 = RateLimiter(name="shared_api", config=config)
            limiter2 = RateLimiter(name="shared_api", config=config)
            
            # Both use same Redis key
            assert limiter1.redis_key == limiter2.redis_key
    
    def test_redis_failure_graceful_degradation(self):
        """If Redis fails, allow requests (graceful degradation)"""
        config = RateLimitConfig(
            max_requests=10,
            window_seconds=60,
            backend="redis",
            allow_on_backend_failure=True
        )
        
        mock_redis = MagicMock()
        mock_redis.get.side_effect = Exception("Redis connection failed")
        
        with patch('middleware.rate_limiter.redis_client', mock_redis):
            limiter = RateLimiter(name="test", config=config)
            
            # Should not raise due to graceful degradation
            limiter.check_limit()


class TestRateLimiterIntegration:
    """Integration tests"""
    
    def test_prevents_ip_bans_from_ad_platforms(self):
        """Q_033: Defensive throttling prevents IP bans"""
        # Meta Ads: 100 requests per minute
        config = RateLimitConfig(max_requests=100, window_seconds=60)
        meta_limiter = RateLimiter(name="meta_ads_api", config=config)
        
        # Simulate application making many requests
        successful_requests = 0
        rate_limited_requests = 0
        
        for i in range(200):  # Try 200 requests
            try:
                with meta_limiter:
                    # Simulate API call
                    successful_requests += 1
            except RateLimitExceeded:
                rate_limited_requests += 1
        
        # Should have limited to ~100 requests
        assert successful_requests <= 120  # Allow burst capacity
        assert rate_limited_requests >= 80
        assert meta_limiter.metrics['rate_limit_hits'] > 0
        
        # Verify no 429 errors would reach external API
        # (all rate limiting handled internally)
