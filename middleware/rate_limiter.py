"""
Rate Limiter Middleware
Implements Q_033: Token bucket rate limiter enforces 100 req/min per external API

Algorithm: Token Bucket
- Bucket starts with max_requests tokens
- Tokens refill at rate = max_requests / window_seconds
- Each request consumes 1 token
- Request rejected if no tokens available

Acceptance:
- Enforces max_requests per window_seconds
- Burst capacity allows temporary spikes
- Separate limits per API endpoint
- Redis backend for distributed rate limiting
- Graceful degradation on Redis failure
"""

from dataclasses import dataclass
from typing import Optional, Dict
import time
import logging
from threading import Lock

logger = logging.getLogger(__name__)

# Optional Redis support (gracefully degrades if unavailable)
try:
    import redis
    redis_client = redis.Redis(
        host='localhost',
        port=6379,
        decode_responses=True,
        socket_timeout=1
    )
except (ImportError, Exception) as e:
    logger.warning(f"Redis not available for rate limiting: {e}")
    redis_client = None


class RateLimitExceeded(Exception):
    """Raised when rate limit is exceeded"""
    def __init__(self, message: str, retry_after: int = 60):
        super().__init__(message)
        self.status_code = 429  # HTTP 429 Too Many Requests
        self.retry_after = retry_after


@dataclass
class RateLimitConfig:
    """Configuration for rate limiter"""
    max_requests: int  # Maximum requests per window
    window_seconds: int  # Time window in seconds
    burst_capacity: Optional[int] = None  # Allow burst above max (default: max_requests * 1.2)
    backend: str = "memory"  # "memory" or "redis"
    allow_on_backend_failure: bool = True  # Allow requests if backend fails


class TokenBucket:
    """
    Token bucket algorithm for rate limiting
    
    Tokens are added at a constant rate (refill_rate per second).
    Each request consumes 1 token. Request fails if no tokens available.
    """
    
    def __init__(self, capacity: int, refill_rate: float):
        """
        Args:
            capacity: Maximum number of tokens
            refill_rate: Tokens added per second
        """
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = float(capacity)  # Start full
        self.last_refill = time.time()
        self.lock = Lock()
    
    def refill(self):
        """Refill tokens based on elapsed time"""
        now = time.time()
        elapsed = now - self.last_refill
        
        # Add tokens based on elapsed time
        tokens_to_add = elapsed * self.refill_rate
        self.tokens = min(self.capacity, self.tokens + tokens_to_add)
        self.last_refill = now
    
    def consume(self, tokens: int = 1) -> bool:
        """
        Attempt to consume tokens
        
        Args:
            tokens: Number of tokens to consume
            
        Returns:
            True if tokens were consumed, False if insufficient tokens
        """
        with self.lock:
            self.refill()
            
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            else:
                return False
    
    def get_available_tokens(self) -> int:
        """Get current available tokens"""
        with self.lock:
            self.refill()
            return int(self.tokens)


class RateLimiter:
    """
    Rate limiter using token bucket algorithm
    
    Features:
    - Per-endpoint rate limiting
    - Burst capacity support
    - Memory or Redis backend
    - Graceful degradation
    
    Usage:
        limiter = RateLimiter(name="meta_ads_api", config=config)
        
        try:
            with limiter:
                response = api_call()
        except RateLimitExceeded as e:
            logger.warning(f"Rate limited: {e}")
            time.sleep(e.retry_after)
    """
    
    def __init__(
        self,
        name: str,
        config: RateLimitConfig
    ):
        self.name = name
        self.config = config
        
        # Set burst capacity
        if config.burst_capacity is None:
            self.burst_capacity = int(config.max_requests * 1.2)
        else:
            self.burst_capacity = config.burst_capacity
        
        # Calculate refill rate (tokens per second)
        self.refill_rate = config.max_requests / config.window_seconds
        
        # Initialize token bucket
        self.bucket = TokenBucket(
            capacity=self.burst_capacity,
            refill_rate=self.refill_rate
        )
        
        # Redis key for distributed rate limiting
        self.redis_key = f"rate_limit:{name}"
        
        # Metrics
        self.metrics: Dict[str, int] = {
            'rate_limit_hits': 0,
            'requests_allowed': 0,
            'requests_rejected': 0
        }
        
        logger.info(
            f"Rate limiter '{name}' initialized: "
            f"{config.max_requests} req/{config.window_seconds}s, "
            f"burst={self.burst_capacity}, "
            f"backend={config.backend}"
        )
    
    def check_limit(self):
        """
        Check if request is allowed under rate limit
        
        Raises:
            RateLimitExceeded: If rate limit exceeded
        """
        if self.config.backend == "redis" and redis_client:
            allowed = self._check_limit_redis()
        else:
            allowed = self._check_limit_memory()
        
        if allowed:
            self.metrics['requests_allowed'] += 1
        else:
            self.metrics['requests_rejected'] += 1
            self.metrics['rate_limit_hits'] += 1
            
            logger.warning(
                f"Rate limit exceeded for '{self.name}': "
                f"{self.config.max_requests} req/{self.config.window_seconds}s"
            )
            
            raise RateLimitExceeded(
                f"Rate limit exceeded for '{self.name}'. "
                f"Limit: {self.config.max_requests} requests per "
                f"{self.config.window_seconds} seconds.",
                retry_after=self.config.window_seconds
            )
    
    def _check_limit_memory(self) -> bool:
        """Check rate limit using in-memory token bucket"""
        return self.bucket.consume(1)
    
    def _check_limit_redis(self) -> bool:
        """Check rate limit using Redis backend (distributed)"""
        try:
            # Use Redis sliding window counter
            now = time.time()
            window_start = now - self.config.window_seconds
            
            pipe = redis_client.pipeline()
            
            # Remove old entries
            pipe.zremrangebyscore(self.redis_key, 0, window_start)
            
            # Count current window
            pipe.zcard(self.redis_key)
            
            # Add current request
            pipe.zadd(self.redis_key, {str(now): now})
            
            # Set expiry
            pipe.expire(self.redis_key, self.config.window_seconds + 10)
            
            results = pipe.execute()
            current_count = results[1]
            
            # Check against burst capacity
            if current_count < self.burst_capacity:
                return True
            else:
                return False
                
        except Exception as e:
            logger.error(f"Redis rate limit check failed: {e}")
            
            # Graceful degradation
            if self.config.allow_on_backend_failure:
                logger.warning(
                    f"Allowing request for '{self.name}' due to backend failure"
                )
                return True
            else:
                return False
    
    def __enter__(self):
        """Context manager entry: check rate limit"""
        self.check_limit()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        return False
    
    def get_remaining_tokens(self) -> int:
        """Get remaining tokens (requests) available"""
        if self.config.backend == "redis" and redis_client:
            try:
                now = time.time()
                window_start = now - self.config.window_seconds
                current_count = redis_client.zcount(
                    self.redis_key,
                    window_start,
                    now
                )
                return max(0, self.burst_capacity - current_count)
            except Exception:
                return 0
        else:
            return self.bucket.get_available_tokens()
    
    def reset(self):
        """Reset rate limiter (admin/testing operation)"""
        if self.config.backend == "redis" and redis_client:
            try:
                redis_client.delete(self.redis_key)
            except Exception as e:
                logger.error(f"Failed to reset Redis rate limiter: {e}")
        
        # Reset memory bucket
        self.bucket.tokens = float(self.bucket.capacity)
        self.bucket.last_refill = time.time()
        
        logger.info(f"Rate limiter '{self.name}' reset")


# Global registry for rate limiters by endpoint
_rate_limiters: Dict[str, RateLimiter] = {}


def get_rate_limiter(
    name: str,
    config: Optional[RateLimitConfig] = None
) -> RateLimiter:
    """
    Get or create rate limiter for named endpoint
    
    Args:
        name: Unique identifier (e.g., 'meta_ads_api')
        config: Configuration (uses default if not provided)
    
    Returns:
        RateLimiter instance
    """
    if name not in _rate_limiters:
        if config is None:
            # Default: 100 requests per minute
            config = RateLimitConfig(
                max_requests=100,
                window_seconds=60
            )
        _rate_limiters[name] = RateLimiter(name, config)
    
    return _rate_limiters[name]


def reset_all_rate_limiters():
    """Reset all rate limiters (testing/admin operation)"""
    for limiter in _rate_limiters.values():
        limiter.reset()
