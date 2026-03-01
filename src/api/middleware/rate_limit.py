"""
Per-API-key daily rate limiting via Redis atomic counters.

Uses ``INCR`` + ``EXPIRE`` for lock-free, race-condition-free counting.
Each API key gets a daily budget (default 50 requests). When exceeded,
the request is rejected with HTTP 429.

Fail-open policy: if Redis is unreachable, the request is allowed. A
broken rate limiter must never kill the API.

Also provides Gemini budget tracking functions for the daily forecast
pipeline (Plan 04).

This is a FastAPI ``Depends()`` callable, NOT ASGI middleware.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Coroutine
from datetime import date
from typing import Any

import redis.asyncio as aioredis
from fastapi import Depends, HTTPException

from src.api.deps import get_redis
from src.api.middleware.auth import verify_api_key
from src.settings import get_settings

logger = logging.getLogger(__name__)

_RATE_LIMIT_TTL: int = 86400  # 24 hours -- auto-cleanup of daily keys


async def check_rate_limit(
    client_name: str,
    redis_client: aioredis.Redis,
    daily_limit: int = 50,
) -> None:
    """Check and increment the daily request count for an API key.

    Uses Redis key ``ratelimit:{client_name}:{YYYY-MM-DD}`` with atomic
    ``INCR`` + ``EXPIRE``. The EXPIRE is set only on the first increment
    (count == 1) to avoid resetting the TTL on subsequent requests.

    Args:
        client_name: Authenticated client identifier from ``verify_api_key``.
        redis_client: Async Redis connection.
        daily_limit: Maximum requests per day (default 50).

    Raises:
        HTTPException: 429 if the daily limit is exceeded.
    """
    key = f"ratelimit:{client_name}:{date.today().isoformat()}"
    try:
        count = await redis_client.incr(key)
        if count == 1:
            await redis_client.expire(key, _RATE_LIMIT_TTL)
        if count > daily_limit:
            logger.warning(
                "Rate limit exceeded for %s: %d/%d",
                client_name,
                count,
                daily_limit,
            )
            raise HTTPException(
                status_code=429,
                detail="Daily request limit exceeded. Try again tomorrow.",
            )
    except HTTPException:
        raise  # Re-raise our own 429 -- don't swallow it
    except Exception as exc:
        # Fail-open: Redis down -> allow the request
        logger.warning(
            "Rate limit check failed for %s (allowing request): %s",
            client_name,
            exc,
        )


def get_rate_limiter(
    daily_limit: int = 50,
) -> Callable[..., Coroutine[Any, Any, None]]:
    """Factory that returns a FastAPI dependency for rate limiting.

    The returned dependency resolves ``client_name`` from ``verify_api_key``
    and ``redis`` from ``get_redis``, then delegates to ``check_rate_limit``.

    Args:
        daily_limit: Maximum requests per day.

    Returns:
        An async callable suitable for ``Depends()``.
    """

    async def _dependency(
        client_name: str = Depends(verify_api_key),
        redis_client: aioredis.Redis = Depends(get_redis),
    ) -> None:
        await check_rate_limit(client_name, redis_client, daily_limit)

    return _dependency


# -----------------------------------------------------------------------
# Gemini budget tracking -- used by the daily forecast pipeline.
# -----------------------------------------------------------------------


async def gemini_budget_remaining(redis_client: aioredis.Redis) -> int:
    """Return the remaining Gemini API budget for today.

    Compares the current usage counter against
    ``Settings.gemini_daily_budget``.

    Args:
        redis_client: Async Redis connection.

    Returns:
        Remaining budget (>= 0). Returns the full budget if Redis is
        unreachable (fail-open for pipeline continuity).
    """
    settings = get_settings()
    key = f"gemini_budget:{date.today().isoformat()}"
    try:
        raw = await redis_client.get(key)
        used = int(raw) if raw is not None else 0
        remaining = max(0, settings.gemini_daily_budget - used)
        return remaining
    except Exception as exc:
        logger.warning("Gemini budget check failed (returning full budget): %s", exc)
        return settings.gemini_daily_budget


async def increment_gemini_usage(redis_client: aioredis.Redis) -> int:
    """Increment the daily Gemini usage counter and return current count.

    Uses ``INCR`` + ``EXPIRE`` for atomic, self-cleaning counters.

    Args:
        redis_client: Async Redis connection.

    Returns:
        Current usage count after increment. Returns 0 on Redis failure.
    """
    key = f"gemini_budget:{date.today().isoformat()}"
    try:
        count = await redis_client.incr(key)
        if count == 1:
            await redis_client.expire(key, _RATE_LIMIT_TTL)
        return int(count)
    except Exception as exc:
        logger.warning("Gemini usage increment failed: %s", exc)
        return 0
