"""Redis client configuration and utilities."""

import redis
from redis.asyncio import Redis as AsyncRedis
from typing import Optional, Any
import json
import logging
from app.core.config import settings

logger = logging.getLogger(__name__)

class RedisClient:
    """Redis client wrapper for caching operations."""
    
    def __init__(self):
        self.redis: Optional[redis.Redis] = None
        self.async_redis: Optional[AsyncRedis] = None
    
    def connect(self) -> redis.Redis:
        """Connect to Redis server."""
        if self.redis is None:
            self.redis = redis.Redis(
                host=settings.REDIS_HOST,
                port=settings.REDIS_PORT,
                db=settings.REDIS_DB,
                password=settings.REDIS_PASSWORD,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True,
                max_connections=20
            )
            # Test connection
            try:
                self.redis.ping()
                logger.info("Redis connection established successfully")
            except Exception as e:
                logger.error(f"Failed to connect to Redis: {e}")
                raise
        return self.redis
    
    async def connect_async(self) -> AsyncRedis:
        """Connect to Redis server asynchronously."""
        if self.async_redis is None:
            self.async_redis = AsyncRedis(
                host=settings.REDIS_HOST,
                port=settings.REDIS_PORT,
                db=settings.REDIS_DB,
                password=settings.REDIS_PASSWORD,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True,
                max_connections=20
            )
            # Test connection
            try:
                await self.async_redis.ping()
                logger.info("Async Redis connection established successfully")
            except Exception as e:
                logger.error(f"Failed to connect to async Redis: {e}")
                raise
        return self.async_redis
    
    def set_json(self, key: str, value: Any, ex: Optional[int] = None) -> bool:
        """Set JSON value in Redis."""
        try:
            client = self.connect()
            return client.set(key, json.dumps(value), ex=ex)
        except Exception as e:
            logger.error(f"Failed to set JSON in Redis: {e}")
            return False
    
    def get_json(self, key: str) -> Optional[Any]:
        """Get JSON value from Redis."""
        try:
            client = self.connect()
            value = client.get(key)
            return json.loads(value) if value else None
        except Exception as e:
            logger.error(f"Failed to get JSON from Redis: {e}")
            return None
    
    async def set_json_async(self, key: str, value: Any, ex: Optional[int] = None) -> bool:
        """Set JSON value in Redis asynchronously."""
        try:
            client = await self.connect_async()
            return await client.set(key, json.dumps(value), ex=ex)
        except Exception as e:
            logger.error(f"Failed to set JSON in async Redis: {e}")
            return False
    
    async def get_json_async(self, key: str) -> Optional[Any]:
        """Get JSON value from Redis asynchronously."""
        try:
            client = await self.connect_async()
            value = await client.get(key)
            return json.loads(value) if value else None
        except Exception as e:
            logger.error(f"Failed to get JSON from async Redis: {e}")
            return None
    
    def close(self):
        """Close Redis connections."""
        if self.redis:
            self.redis.close()
            self.redis = None
        if self.async_redis:
            # Note: async redis doesn't have close method, connection pool handles it
            self.async_redis = None

# Global Redis client instance
redis_client = RedisClient()

def get_redis() -> redis.Redis:
    """Dependency to get Redis client."""
    return redis_client.connect()

async def get_redis_async() -> AsyncRedis:
    """Dependency to get async Redis client."""
    return await redis_client.connect_async()