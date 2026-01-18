import logging
from typing import Optional, Any
import redis.asyncio as redis
from app.core.config import settings

logger = logging.getLogger(__name__)

class RedisClient:
    """
    Async Redis client wrapper with connection pooling and error handling.
    """
    _instance: Optional['RedisClient'] = None
    _client: Optional[redis.Redis] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(RedisClient, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        # Singleton initialization check
        if self._client is not None:
            return

        try:
            # redis.asyncio.from_url handles connection pooling automatically
            logger.info(f"Connecting to Redis at {settings.REDIS_HOST}:{settings.REDIS_PORT}")
            self._client = redis.from_url(
                settings.REDIS_URL, 
                encoding="utf-8", 
                decode_responses=True,
                socket_connect_timeout=5,
                socket_keepalive=True
            )
        except Exception as e:
            logger.error(f"Failed to initialize Redis client: {e}")
            self._client = None

    async def get_client(self) -> redis.Redis:
        """Get the underlying redis client instance."""
        if self._client is None:
             # Re-try initialization if it failed previously
             self.__init__()
        return self._client

    async def close(self):
        """Close the redis connection."""
        if self._client:
            await self._client.close()
            logger.info("Redis connection closed")

    async def ping(self) -> bool:
        """Check redis connection health."""
        if not self._client:
            return False
        try:
            return await self._client.ping()
        except Exception as e:
            logger.error(f"Redis ping failed: {e}")
            return False

    async def get(self, key: str) -> Optional[str]:
        """Get a value by key."""
        if not self._client: return None
        try:
            return await self._client.get(key)
        except Exception as e:
            logger.error(f"Redis get error for key {key}: {e}")
            return None

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set a value with optional TTL (seconds)."""
        if not self._client: return False
        try:
            return await self._client.set(key, value, ex=ttl)
        except Exception as e:
            logger.error(f"Redis set error for key {key}: {e}")
            return False

    async def delete(self, key: str) -> bool:
        """Delete a key."""
        if not self._client: return False
        try:
            return await self._client.delete(key) > 0
        except Exception as e:
            logger.error(f"Redis delete error for key {key}: {e}")
            return False

# Global instance
redis_client = RedisClient()

async def get_redis_async() -> redis.Redis:
    """Compatibility helper for legacy code."""
    return await redis_client.get_client()