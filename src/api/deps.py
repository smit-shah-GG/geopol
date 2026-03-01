"""
FastAPI dependency injection providers.

Thin wrappers that adapt internal infrastructure (database sessions, settings,
Redis, forecast cache) into FastAPI-compatible ``Depends()`` callables. Keep
this module free of business logic -- it's pure plumbing.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator

import redis.asyncio as aioredis
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.services.cache_service import ForecastCache
from src.db.postgres import get_async_session
from src.settings import Settings, get_settings

logger = logging.getLogger(__name__)

# Module-level singletons -- lazy-initialized on first access.
_redis_client: aioredis.Redis | None = None
_forecast_cache: ForecastCache | None = None


class NullRedis:
    """Noop Redis stub used when the real Redis server is unreachable.

    Every read returns None; every write is silently discarded. This lets
    the cache service degrade to tier 1 + tier 3 without branching on
    ``Optional`` throughout the hot path.
    """

    async def get(self, _key: str) -> None:
        return None

    async def setex(self, _key: str, _ttl: int, _value: str) -> None:
        return None

    async def incr(self, _key: str) -> int:
        return 0

    async def expire(self, _key: str, _ttl: int) -> None:
        return None

    async def aclose(self) -> None:
        return None


async def get_redis() -> aioredis.Redis:
    """Return a lazily-initialized async Redis client.

    On first call, connects to ``Settings.redis_url``. If the connection
    fails, returns a ``NullRedis`` stub so callers degrade gracefully.

    Returns:
        ``redis.asyncio.Redis`` instance (or ``NullRedis`` stub).
    """
    global _redis_client  # noqa: PLW0603
    if _redis_client is not None:
        return _redis_client

    settings = get_settings()
    try:
        client = aioredis.from_url(
            settings.redis_url,
            decode_responses=True,
            socket_connect_timeout=2.0,
        )
        # Verify connectivity with a lightweight ping
        await client.ping()
        _redis_client = client
        logger.info("Redis connected: %s", settings.redis_url)
    except Exception as exc:
        logger.warning(
            "Redis unavailable (%s) -- degrading to memory-only cache: %s",
            settings.redis_url,
            exc,
        )
        _redis_client = NullRedis()  # type: ignore[assignment]

    return _redis_client  # type: ignore[return-value]


def get_cache() -> ForecastCache:
    """Return the ``ForecastCache`` singleton.

    If Redis is not yet initialized, creates the cache with a ``NullRedis``
    stub (tier 2 silently skipped). Once ``get_redis()`` is called and
    connects, subsequent ``get_cache()`` calls use the real client.

    Returns:
        ``ForecastCache`` singleton.
    """
    global _forecast_cache  # noqa: PLW0603
    if _forecast_cache is not None:
        return _forecast_cache

    # Use the existing Redis client if available, else NullRedis
    redis_client = _redis_client if _redis_client is not None else NullRedis()
    _forecast_cache = ForecastCache(redis_client)  # type: ignore[arg-type]
    return _forecast_cache


async def _close_redis() -> None:
    """Gracefully close the Redis connection (call from app lifespan shutdown)."""
    global _redis_client, _forecast_cache  # noqa: PLW0603
    if _redis_client is not None:
        try:
            await _redis_client.aclose()
        except Exception as exc:
            logger.warning("Error closing Redis: %s", exc)
        _redis_client = None
    _forecast_cache = None


# -----------------------------------------------------------------------
# Existing providers -- untouched.
# -----------------------------------------------------------------------


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Yield an async database session with automatic commit/rollback.

    Wraps ``src.db.postgres.get_async_session`` for use as a FastAPI
    dependency. The session commits on success, rolls back on exception,
    and closes on exit.

    Usage::

        @router.get("/items")
        async def list_items(db: AsyncSession = Depends(get_db)):
            ...
    """
    async for session in get_async_session():
        yield session


def get_current_settings() -> Settings:
    """Return the cached settings singleton.

    Usage::

        @router.get("/info")
        async def info(settings: Settings = Depends(get_current_settings)):
            return {"env": settings.environment}
    """
    return get_settings()
