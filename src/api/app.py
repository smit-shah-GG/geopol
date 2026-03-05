"""
FastAPI application factory.

``create_app()`` builds the fully configured application with:
- Lifespan: DB init/shutdown, logging setup, dev API key seeding,
  APScheduler startup with all 9 background jobs
- CORS middleware
- RFC 9457 error handlers
- Versioned router at ``/api/v1``

Start with::

    uvicorn src.api.app:create_app --factory
"""

from __future__ import annotations

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
from src.scheduler import create_scheduler, register_all_jobs, init_shared_deps, shutdown_scheduler
from src.scheduler.retry import JobFailureTracker
from src.settings import get_settings

logger = logging.getLogger(__name__)

_DEV_API_KEY = "dev-api-key-geopol-2026"
_DEV_CLIENT_NAME = "geopol-dev"


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan: startup and shutdown hooks.

    Startup:
      1. Configure structured logging
      2. Attach ring buffer to root logger
      3. Initialize PostgreSQL async engine
      4. Initialize Redis client (degrades to NullRedis if unavailable)
      5. Seed dev API key (development only)
      6. Initialize shared dependencies (EventStorage, TKG, GDELTPoller)
      7. Create, register, and start APScheduler with all 9 jobs

    Shutdown:
      1. Shut down APScheduler (30s graceful timeout)
      2. Close Redis connection
      3. Dispose PostgreSQL engine connection pool
    """
    settings = get_settings()

    # 1. Logging
    setup_logging(level=settings.log_level, json_format=settings.log_json)

    # 2. Ring buffer for admin /logs endpoint
    from src.api.log_buffer import get_ring_buffer

    ring = get_ring_buffer()
    logging.getLogger().addHandler(ring)

    logger.info("Geopol API starting (env=%s)", settings.environment)

    # 3. Database
    init_db()
    logger.info("PostgreSQL engine initialized")

    # 4. Redis
    await get_redis()
    logger.info("Redis client initialized")

    # 5. Dev API key seeding
    if settings.environment == "development":
        await _seed_dev_api_key()

    # 6. Shared dependencies (needs DB to be ready)
    init_shared_deps()
    logger.info("Shared dependencies initialized")

    # 7. APScheduler
    scheduler = create_scheduler()
    failure_tracker = JobFailureTracker(scheduler)
    register_all_jobs(scheduler, failure_tracker)
    scheduler.start()
    app.state.scheduler = scheduler
    app.state.failure_tracker = failure_tracker
    job_count = len(scheduler.get_jobs())
    logger.info("APScheduler started with %d jobs", job_count)

    yield

    # Shutdown -- order matters: scheduler first (may use DB), then Redis, then DB
    logger.info("Geopol API shutting down")

    # 1. Scheduler shutdown (30s graceful timeout)
    await shutdown_scheduler(app.state.scheduler, timeout=30.0)
    logger.info("APScheduler shut down")

    # 2. Redis
    await _close_redis()

    # 3. Database
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
