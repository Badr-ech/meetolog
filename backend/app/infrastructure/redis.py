"""
Redis connection pool management for Meetolog v2.

Provides async Redis client with connection pooling for:
- Job state persistence (RedisJobStore)
- ARQ background queue
- Transcript/artifact caching

Uses redis-py async client with hiredis for performance.
"""

import logging
from functools import lru_cache
from urllib.parse import urlparse

from arq.connections import RedisSettings
from redis.asyncio import Redis, ConnectionPool

from ..config import get_settings

logger = logging.getLogger(__name__)

# Global connection pool (singleton)
_redis_pool: ConnectionPool | None = None
_redis_client: Redis | None = None


def _parse_redis_url(url: str) -> dict:
    parsed = urlparse(url)
    
    params = {
        "host": parsed.hostname or "localhost",
        "port": parsed.port or 6379,
        "password": parsed.password,
        "db": int(parsed.path[1:]) if parsed.path and len(parsed.path) > 1 else 0,
    }
    
    if parsed.scheme == "rediss":
        import ssl as ssl_module
        params["ssl_cert_reqs"] = ssl_module.CERT_NONE
    
    return params


async def get_redis_pool() -> Redis:
    """
    Get or create the global Redis async client.
    
    Uses connection pooling for efficiency. The pool is created
    on first access and reused for all subsequent calls.
    
    Returns:
        Async Redis client instance
        
    Raises:
        ConnectionError: If Redis is unavailable
    """
    global _redis_pool, _redis_client
    
    if _redis_client is not None:
        return _redis_client
    
    settings = get_settings()
    params = _parse_redis_url(settings.redis_url)
    
    logger.info(f"Initializing Redis connection pool: {params['host']}:{params['port']}")
    
    pool_kwargs = {
        "host": params["host"],
        "port": params["port"],
        "db": params["db"],
        "max_connections": 20,
        "decode_responses": True,
        "socket_timeout": 5.0,
        "socket_connect_timeout": 5.0,
        "retry_on_timeout": True,
    }
    
    # Add optional parameters only if present
    if params.get("password"):
        pool_kwargs["password"] = params["password"]
    if "ssl_cert_reqs" in params:
        pool_kwargs["ssl_cert_reqs"] = params["ssl_cert_reqs"]
    
    _redis_pool = ConnectionPool(**pool_kwargs)
    
    _redis_client = Redis(connection_pool=_redis_pool)
    
    try:
        await _redis_client.ping()
        logger.info("Redis connection established successfully")
    except Exception as e:
        logger.error(f"Failed to connect to Redis: {e}")
        _redis_client = None
        _redis_pool = None
        raise ConnectionError(f"Cannot connect to Redis at {settings.redis_url}: {e}") from e
    
    return _redis_client


async def close_redis_pool() -> None:
    global _redis_pool, _redis_client
    
    if _redis_client is not None:
        logger.info("Closing Redis connection pool")
        await _redis_client.close()
        _redis_client = None
    
    if _redis_pool is not None:
        await _redis_pool.disconnect()
        _redis_pool = None


@lru_cache
def get_arq_redis_settings() -> RedisSettings:
    """
    Get ARQ-compatible Redis settings for worker configuration.
    
    ARQ uses its own RedisSettings class, so we parse our URL
    into that format.
    
    Returns:
        ARQ RedisSettings instance
    """
    settings = get_settings()
    params = _parse_redis_url(settings.redis_url)
    
    # Build ARQ settings - only include ssl if TLS is enabled
    arq_kwargs = {
        "host": params["host"],
        "port": params["port"],
        "database": params["db"],
        "conn_timeout": 10,
        "conn_retries": 5,
        "conn_retry_delay": 1,
    }
    
    # ARQ's RedisSettings expects boolean 'ssl' for TLS
    if "ssl_cert_reqs" in params:
        arq_kwargs["ssl"] = True
    
    if params.get("password"):
        arq_kwargs["password"] = params["password"]
    
    return RedisSettings(**arq_kwargs)


async def check_redis_health() -> dict:
    """
    Check Redis connection health.
    
    Returns:
        Dict with connection status and info
    """
    try:
        redis = await get_redis_pool()
        info = await redis.info("server")
        return {
            "status": "healthy",
            "redis_version": info.get("redis_version", "unknown"),
            "connected_clients": info.get("connected_clients", 0),
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
        }
