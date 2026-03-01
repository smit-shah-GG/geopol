"""
API key authentication dependency.

Validates the ``X-API-Key`` header against the ``api_keys`` table in
PostgreSQL. Keys that are revoked or don't exist result in a 401.

This is a FastAPI ``Depends()`` callable, NOT ASGI middleware — it only
runs on routes that explicitly declare it as a dependency.
"""

from __future__ import annotations

import logging

from fastapi import Depends, HTTPException, Security
from fastapi.security import APIKeyHeader
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.deps import get_db
from src.db.models import ApiKey

logger = logging.getLogger(__name__)

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(
    api_key: str | None = Security(_api_key_header),
    db: AsyncSession = Depends(get_db),
) -> str:
    """Validate the API key and return the associated client name.

    Args:
        api_key: Value from the X-API-Key header (None if absent).
        db: Async database session.

    Returns:
        The ``client_name`` associated with the API key.

    Raises:
        HTTPException: 401 if the key is missing, invalid, or revoked.
    """
    if api_key is None:
        raise HTTPException(
            status_code=401,
            detail="Missing API key. Provide X-API-Key header.",
        )

    result = await db.execute(
        select(ApiKey).where(ApiKey.key == api_key, ApiKey.revoked.is_(False))
    )
    key_record = result.scalar_one_or_none()

    if key_record is None:
        logger.warning("Rejected API key: %s...", api_key[:8] if len(api_key) >= 8 else api_key)
        raise HTTPException(
            status_code=401,
            detail="Invalid or revoked API key.",
        )

    logger.debug("Authenticated client: %s", key_record.client_name)
    return key_record.client_name
