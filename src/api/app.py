"""
FastAPI application factory.

``create_app()`` builds the fully configured application with:
- Lifespan: DB init/shutdown, logging setup, dev API key seeding
- CORS middleware
- RFC 9457 error handlers
- Versioned router at ``/api/v1``

Start with::

    uvicorn src.api.app:create_app --factory
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator

from fastapi import FastAPI
from sqlalchemy import select

from src.api.deps import _close_redis, get_redis
from src.api.errors import register_error_handlers
from src.api.middleware.cors import configure_cors
from src.db.models import ApiKey
from src.db.postgres import close_db, get_async_session, init_db
from src.logging_config import setup_logging
from src.settings import get_settings

logger = logging.getLogger(__name__)

_DEV_API_KEY = "dev-api-key-geopol-2026"
_DEV_CLIENT_NAME = "geopol-dev"


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan: startup and shutdown hooks.

    Startup:
      1. Configure structured logging
      2. Initialize PostgreSQL async engine
      3. Initialize Redis client (degrades to NullRedis if unavailable)
      4. Seed dev API key (development only)

    Shutdown:
      1. Close Redis connection
      2. Dispose PostgreSQL engine connection pool
    """
    settings = get_settings()

    # 1. Logging
    setup_logging(level=settings.log_level, json_format=settings.log_json)

    # Attach ring buffer to root logger for admin /logs endpoint
    from src.api.log_buffer import get_ring_buffer

    ring = get_ring_buffer()
    logging.getLogger().addHandler(ring)

    logger.info("Geopol API starting (env=%s)", settings.environment)

    # 2. Database
    init_db()
    logger.info("PostgreSQL engine initialized")

    # 3. Redis
    await get_redis()
    logger.info("Redis client initialized")

    # 4. Dev API key seeding
    if settings.environment == "development":
        await _seed_dev_api_key()

    # 5. Polymarket background matching cycle
    polymarket_task: asyncio.Task | None = None
    if settings.polymarket_enabled:
        polymarket_task = asyncio.create_task(_polymarket_loop(settings))
        logger.info("Polymarket matching loop started (interval=%ds)", settings.polymarket_poll_interval)

    yield

    # Shutdown
    logger.info("Geopol API shutting down")
    if polymarket_task is not None:
        polymarket_task.cancel()
        try:
            await polymarket_task
        except asyncio.CancelledError:
            pass
    await _close_redis()
    await close_db()


async def _seed_dev_api_key() -> None:
    """Insert the dev API key if it doesn't already exist.

    Only runs in development. Uses a direct session to avoid coupling
    to FastAPI dependency injection (lifespan runs outside request scope).
    """
    try:
        async for session in get_async_session():
            result = await session.execute(
                select(ApiKey).where(ApiKey.key == _DEV_API_KEY)
            )
            existing = result.scalar_one_or_none()
            if existing is None:
                session.add(
                    ApiKey(key=_DEV_API_KEY, client_name=_DEV_CLIENT_NAME)
                )
                await session.commit()
                logger.info("Seeded dev API key: %s", _DEV_API_KEY)
            else:
                logger.debug("Dev API key already exists")
    except Exception:
        # DB may not be migrated yet — don't crash startup
        logger.warning(
            "Could not seed dev API key (database may not be migrated). "
            "Run 'alembic upgrade head' first."
        )


async def _polymarket_loop(settings) -> None:
    """Periodic Polymarket matching + auto-forecast cycle.

    Runs run_matching_cycle() + capture_snapshots() + auto-forecaster
    on the configured interval. Re-forecasting of active comparisons
    runs at most once per UTC day. Errors are logged and swallowed --
    the loop must not crash.
    """
    import aiohttp

    from src.db.postgres import async_session_factory
    from src.forecasting.gemini_client import GeminiClient
    from src.polymarket.auto_forecaster import PolymarketAutoForecaster
    from src.polymarket.client import PolymarketClient
    from src.polymarket.comparison import PolymarketComparisonService
    from src.polymarket.matcher import PolymarketMatcher

    # Wait for first cycle to let the app finish booting
    await asyncio.sleep(30)

    try:
        gemini_client = GeminiClient()
    except Exception:
        logger.warning("Polymarket loop: Gemini client unavailable, matching disabled")
        return

    matcher = PolymarketMatcher(gemini_client, match_threshold=settings.polymarket_match_threshold)

    # Track last reforecast date to ensure at-most-once-daily
    _last_reforecast_date: str | None = None

    while True:
        try:
            async with aiohttp.ClientSession() as http_session:
                pm_client = PolymarketClient(session=http_session)
                service = PolymarketComparisonService(
                    async_session_factory=async_session_factory,
                    polymarket_client=pm_client,
                    matcher=matcher,
                    settings=settings,
                )
                result = await service.run_matching_cycle()
                logger.info("Polymarket matching: %s", result)
                await service.capture_snapshots()

                # Phase 18: Auto-forecast unmatched high-volume questions
                auto_forecaster = PolymarketAutoForecaster(
                    async_session_factory=async_session_factory,
                    gemini_client=gemini_client,
                    settings=settings,
                )
                geo_events = await pm_client.fetch_geopolitical_markets()
                auto_result = await auto_forecaster.run(
                    geo_events, tracked_ids=set()
                )
                logger.info("Polymarket auto-forecast: %s", auto_result)

                # Re-forecast active comparisons (at most once per UTC day)
                from datetime import date, timezone

                today_str = date.today().isoformat()
                if _last_reforecast_date != today_str:
                    reforecast_result = await auto_forecaster.reforecast_active()
                    logger.info("Polymarket re-forecast: %s", reforecast_result)
                    _last_reforecast_date = today_str

        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Polymarket matching cycle failed")

        await asyncio.sleep(settings.polymarket_poll_interval)


def create_app() -> FastAPI:
    """Build and return the configured FastAPI application.

    This is the factory function for uvicorn::

        uvicorn src.api.app:create_app --factory

    Returns:
        Fully configured FastAPI instance.
    """
    app = FastAPI(
        title="Geopol Forecast API",
        version="2.0.0-dev",
        description=(
            "AI-powered geopolitical forecasting combining Temporal Knowledge "
            "Graphs with LLM reasoning. Provides calibrated probabilistic "
            "forecasts with full explainability chains."
        ),
        lifespan=_lifespan,
    )

    # Error handlers — RFC 9457 Problem Details
    register_error_handlers(app)

    # CORS middleware
    configure_cors(app)

    # Versioned API routes
    from src.api.routes.v1.router import v1_router

    app.include_router(v1_router, prefix="/api/v1")

    return app
