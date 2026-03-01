"""
FastAPI dependency injection providers.

Thin wrappers that adapt internal infrastructure (database sessions, settings)
into FastAPI-compatible ``Depends()`` callables. Keep this module free of
business logic — it's pure plumbing.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession

from src.db.postgres import get_async_session
from src.settings import Settings, get_settings


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
