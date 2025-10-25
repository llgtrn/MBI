"""
Redis client configuration
"""

import redis.asyncio as redis
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)

# Create Redis client
redis_client = redis.from_url(
    settings.REDIS_URL,
    encoding="utf-8",
    decode_responses=True,
    max_connections=50
)


async def get_redis():
    """
    Dependency to get Redis client
    
    Usage:
        @app.get("/items")
        async def get_items(redis: Redis = Depends(get_redis)):
            ...
    """
    return redis_client


class RedisCache:
    """Redis cache helper class"""
    
    def __init__(self, ttl: int = settings.REDIS_CACHE_TTL):
        self.ttl = ttl
        self.client = redis_client
    
    async def get(self, key: str) -> str | None:
        """Get value from cache"""
        try:
            return await self.client.get(key)
        except Exception as e:
            logger.error(f"Redis GET error: {e}")
            return None
    
    async def set(self, key: str, value: str, ttl: int | None = None) -> bool:
        """Set value in cache"""
        try:
            ttl = ttl or self.ttl
            await self.client.setex(key, ttl, value)
            return True
        except Exception as e:
            logger.error(f"Redis SET error: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete key from cache"""
        try:
            await self.client.delete(key)
            return True
        except Exception as e:
            logger.error(f"Redis DELETE error: {e}")
            return False
    
    async def exists(self, key: str) -> bool:
        """Check if key exists"""
        try:
            return await self.client.exists(key) > 0
        except Exception as e:
            logger.error(f"Redis EXISTS error: {e}")
            return False
    
    async def increment(self, key: str, amount: int = 1) -> int | None:
        """Increment counter"""
        try:
            return await self.client.incrby(key, amount)
        except Exception as e:
            logger.error(f"Redis INCR error: {e}")
            return None


# Global cache instance
cache = RedisCache()
