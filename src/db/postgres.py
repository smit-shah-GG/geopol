"""
Async PostgreSQL connection management via SQLAlchemy + asyncpg.

Provides a connection-pooled async engine and session factory. Use
``get_async_session()`` as a FastAPI dependency or async generator
in any async context.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from src.settings import get_settings

logger = logging.getLogger(__name__)

# Module-level engine and session factory, initialized lazily via init_db().
engine: AsyncEngine | None = None
async_session_factory: async_sessionmaker[AsyncSession] | None = None


def init_db(url: str | None = None) -> AsyncEngine:
    """Create the async engine and session factory.

    Called once at application startup (e.g., FastAPI lifespan).

    Args:
        url: Override DATABASE_URL. Defaults to settings value.

    Returns:
        The created AsyncEngine.
    """
    global engine, async_session_factory  # noqa: PLW0603

    settings = get_settings()
    database_url = url or settings.database_url

    engine = create_async_engine(
        database_url,
        pool_size=5,
        max_overflow=10,
        pool_timeout=30,
        pool_recycle=1800,
        echo=False,
    )

    async_session_factory = async_sessionmaker(
        engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autoflush=False,
    )

    logger.info("PostgreSQL async engine initialized (pool_size=5, max_overflow=10)")
    return engine


async def close_db() -> None:
    """Dispose of the engine connection pool.

    Called once at application shutdown.
    """
    global engine, async_session_factory  # noqa: PLW0603
    if engine is not None:
        await engine.dispose()
        logger.info("PostgreSQL engine disposed")
    engine = None
    async_session_factory = None


async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    """Yield an async session with automatic commit/rollback/close.

    Usage as FastAPI dependency::

        @router.get("/items")
        async def list_items(db: AsyncSession = Depends(get_async_session)):
            ...

    Or as async generator::

        async for session in get_async_session():
            await session.execute(...)
    """
    if async_session_factory is None:
        # Lazy init for scripts / tests that skip lifespan
        init_db()
    assert async_session_factory is not None

    async with async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
