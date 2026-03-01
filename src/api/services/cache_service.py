"""
Three-tier forecast response cache.

Tier 1: In-memory TTLCache (100 entries, 10-min TTL) -- zero-latency hot path
Tier 2: Redis (configurable TTL per key type) -- survives process restart
Tier 3: PostgreSQL via ForecastService -- cold storage, handled by caller

The cache service owns tiers 1 and 2. When both miss, it returns None and
the caller falls through to tier 3 (ForecastService queries). This avoids
coupling cache internals to the ORM layer.

IMPORTANT: ``get()`` and ``set()`` use ``cache_key`` AS-IS for Redis
operations. They do NOT add any prefix. All prefixing is done in the
``cache_key_for_*`` helper functions. This prevents double-prefix bugs
(e.g. ``forecast:forecast:{id}``) that cause permanent cache misses.
"""

from __future__ import annotations

import json
import logging

import redis.asyncio as aioredis
from cachetools import TTLCache

logger = logging.getLogger(__name__)

# TTL constants -- seconds
SUMMARY_TTL: int = 3600  # 1 hour for country lists / top forecasts
FULL_FORECAST_TTL: int = 21600  # 6 hours for single forecast detail

# Tier 1 defaults
_MEMORY_MAXSIZE: int = 100
_MEMORY_TTL: int = 600  # 10 minutes


class ForecastCache:
    """Three-tier forecast response cache.

    Tier 1: ``cachetools.TTLCache`` -- in-process, sub-microsecond access.
    Tier 2: Redis -- shared across workers, survives restart.
    Tier 3: PostgreSQL -- caller responsibility (returns None on miss).

    Args:
        redis_client: An ``redis.asyncio.Redis`` instance. If the client
            is a ``NullRedis`` stub, tier 2 is silently skipped.
    """

    def __init__(self, redis_client: aioredis.Redis) -> None:
        self._memory: TTLCache[str, dict] = TTLCache(
            maxsize=_MEMORY_MAXSIZE, ttl=_MEMORY_TTL
        )
        self._redis = redis_client

    async def get(self, cache_key: str) -> dict | None:
        """Look up a cached forecast response by key.

        Check order: memory -> Redis -> None (caller falls through to PG).

        Args:
            cache_key: Fully-qualified cache key. Used as-is for Redis --
                no additional prefix is applied.

        Returns:
            Deserialized dict if found in tier 1 or tier 2, else None.
        """
        # Tier 1: in-memory
        hit = self._memory.get(cache_key)
        if hit is not None:
            return hit

        # Tier 2: Redis
        try:
            raw: bytes | None = await self._redis.get(cache_key)
            if raw is not None:
                data: dict = json.loads(raw)
                # Promote to tier 1
                self._memory[cache_key] = data
                return data
        except Exception as exc:
            # Redis down -- degrade to tier 1 + tier 3 only
            logger.warning("Redis GET failed for key %s: %s", cache_key, exc)

        return None

    async def set(
        self, cache_key: str, data: dict, ttl: int = SUMMARY_TTL
    ) -> None:
        """Write a forecast response to tier 1 and tier 2.

        Args:
            cache_key: Fully-qualified cache key. Used as-is for Redis --
                no additional prefix is applied.
            data: JSON-serializable dict to cache.
            ttl: Time-to-live in seconds for the Redis entry. Defaults to
                ``SUMMARY_TTL`` (1 hour).
        """
        # Tier 1: in-memory (always succeeds)
        self._memory[cache_key] = data

        # Tier 2: Redis
        try:
            await self._redis.setex(cache_key, ttl, json.dumps(data))
        except Exception as exc:
            logger.warning("Redis SETEX failed for key %s: %s", cache_key, exc)


# -----------------------------------------------------------------------
# Cache key generators -- all prefixing happens HERE, nowhere else.
# -----------------------------------------------------------------------


def cache_key_for_forecast(forecast_id: str) -> str:
    """Generate cache key for a single forecast detail.

    Returns:
        ``"forecast:{forecast_id}"``
    """
    return f"forecast:{forecast_id}"


def cache_key_for_country(iso_code: str) -> str:
    """Generate cache key for a country's forecast list.

    The ISO code is uppercased for consistency.

    Returns:
        ``"forecast:country:{ISO_CODE}"``
    """
    return f"forecast:country:{iso_code.upper()}"


def cache_key_for_top(limit: int) -> str:
    """Generate cache key for the top-N forecasts endpoint.

    Returns:
        ``"forecast:top:{limit}"``
    """
    return f"forecast:top:{limit}"
