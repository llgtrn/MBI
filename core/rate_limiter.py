"""
Rate Limiter with burst allowance, token bucket algorithm, Prometheus metrics.
Implements Q_017: 101st request returns 429 with metrics.
Implements A_017: Rate limiters configured with metrics emitted.
Supports Q_026: BigQuery rate limit exhaustion handling.
"""

import time
import threading
from typing import Dict, Optional
from dataclasses import dataclass
from prometheus_client import Counter, Gauge, Histogram

# Prometheus metrics
rate_limit_exceeded = Counter(
    'rate_limit_exceeded_total',
    'Total number of rate limit violations',
    ['limiter_id', 'resource']
)

rate_limit_tokens_available = Gauge(
    'rate_limit_tokens_available',
    'Current available tokens in bucket',
    ['limiter_id', 'resource']
)

rate_limit_request_latency = Histogram(
    'rate_limit_request_latency_seconds',
    'Latency of rate limit check',
    ['limiter_id', 'resource']
)


@dataclass
class RateLimitConfig:
    """Rate limiter configuration."""
    limiter_id: str
    resource: str
    max_tokens: int  # Bucket capacity
    refill_rate: float  # Tokens per second
    burst_allowance: int  # Extra tokens beyond refill_rate


class TokenBucketRateLimiter:
    """
    Token bucket rate limiter with burst allowance.
    
    Acceptance criteria (Q_017):
    - 101st request within burst window returns 429
    - Prometheus metric rate_limit_exceeded_total increments
    - Tokens refill at configured rate
    
    Example:
        limiter = TokenBucketRateLimiter(
            config=RateLimitConfig(
                limiter_id="api_gateway",
                resource="bigquery_api",
                max_tokens=100,
                refill_rate=10.0,  # 10 tokens/sec
                burst_allowance=20  # Allow bursts up to 120 total
            )
        )
        
        if limiter.acquire():
            # Process request
            pass
        else:
            # Return 429 Too Many Requests
            raise RateLimitExceeded()
    """
    
    def __init__(self, config: RateLimitConfig):
        self.config = config
        self.tokens = float(config.max_tokens)
        self.last_refill = time.time()
        self.lock = threading.Lock()
        
        # Emit initial gauge
        rate_limit_tokens_available.labels(
            limiter_id=config.limiter_id,
            resource=config.resource
        ).set(self.tokens)
    
    def acquire(self, tokens: int = 1) -> bool:
        """
        Attempt to acquire tokens from bucket.
        
        Returns:
            True if tokens acquired (request allowed)
            False if insufficient tokens (rate limit exceeded, return 429)
        """
        start = time.time()
        
        with self.lock:
            # Refill tokens based on elapsed time
            now = time.time()
            elapsed = now - self.last_refill
            refill_amount = elapsed * self.config.refill_rate
            
            # Cap at max_tokens + burst_allowance
            max_capacity = self.config.max_tokens + self.config.burst_allowance
            self.tokens = min(max_capacity, self.tokens + refill_amount)
            self.last_refill = now
            
            # Check if sufficient tokens available
            if self.tokens >= tokens:
                self.tokens -= tokens
                allowed = True
            else:
                allowed = False
                # Emit rate limit exceeded metric
                rate_limit_exceeded.labels(
                    limiter_id=self.config.limiter_id,
                    resource=self.config.resource
                ).inc()
            
            # Emit gauge
            rate_limit_tokens_available.labels(
                limiter_id=self.config.limiter_id,
                resource=self.config.resource
            ).set(self.tokens)
        
        # Emit latency
        latency = time.time() - start
        rate_limit_request_latency.labels(
            limiter_id=self.config.limiter_id,
            resource=self.config.resource
        ).observe(latency)
        
        return allowed
    
    def get_tokens_available(self) -> float:
        """Get current token count (for testing/monitoring)."""
        with self.lock:
            return self.tokens
    
    def reset(self):
        """Reset to full capacity (for testing)."""
        with self.lock:
            self.tokens = float(self.config.max_tokens)
            self.last_refill = time.time()


class RateLimitExceeded(Exception):
    """Raised when rate limit is exceeded (HTTP 429)."""
    pass


# Global rate limiter registry
_rate_limiters: Dict[str, TokenBucketRateLimiter] = {}
_registry_lock = threading.Lock()


def get_rate_limiter(
    limiter_id: str,
    config: Optional[RateLimitConfig] = None
) -> TokenBucketRateLimiter:
    """
    Get or create rate limiter by ID.
    
    Args:
        limiter_id: Unique identifier for this limiter
        config: Configuration (required if limiter doesn't exist)
    
    Returns:
        TokenBucketRateLimiter instance
    """
    with _registry_lock:
        if limiter_id not in _rate_limiters:
            if config is None:
                raise ValueError(f"Rate limiter {limiter_id} not found and no config provided")
            _rate_limiters[limiter_id] = TokenBucketRateLimiter(config)
        return _rate_limiters[limiter_id]


def configure_default_limiters():
    """
    Configure default rate limiters for common resources.
    Implements A_017: Rate limiters configured.
    """
    # BigQuery API: 100 req/sec sustained, burst up to 120
    get_rate_limiter(
        "bigquery_api",
        RateLimitConfig(
            limiter_id="bigquery_api",
            resource="bigquery",
            max_tokens=100,
            refill_rate=100.0,  # 100 tokens/sec = 100 req/sec
            burst_allowance=20
        )
    )
    
    # External APIs: 50 req/sec sustained, burst up to 75
    get_rate_limiter(
        "external_api",
        RateLimitConfig(
            limiter_id="external_api",
            resource="external",
            max_tokens=50,
            refill_rate=50.0,
            burst_allowance=25
        )
    )
    
    # Ad Platform APIs: 30 req/sec sustained, burst up to 50
    get_rate_limiter(
        "ad_platform_api",
        RateLimitConfig(
            limiter_id="ad_platform_api",
            resource="ad_platform",
            max_tokens=30,
            refill_rate=30.0,
            burst_allowance=20
        )
    )
