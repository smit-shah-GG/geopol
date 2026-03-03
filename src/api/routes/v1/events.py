"""
Paginated event listing endpoint with full filter surface.

Serves unified GDELT + ACLED events from the SQLite events table.
Supports 9-parameter filtering (country, date range, CAMEO code, actor,
Goldstein range, text search, source discriminator) with cursor-based
keyset pagination via ``encode_keyset_cursor`` / ``decode_keyset_cursor``.

Default time window is 30 days when no date filters are provided.
SQLite access is synchronous -- wrapped in ``asyncio.to_thread()``.
"""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, Depends, HTTPException, Query

from src.api.middleware.auth import verify_api_key
from src.api.schemas.common import (
    PaginatedResponse,
    ProblemDetail,
    decode_keyset_cursor,
    encode_keyset_cursor,
)
from src.api.schemas.event import EventDTO
from src.database.storage import EventStorage
from src.settings import get_settings

logger = logging.getLogger(__name__)

router = APIRouter()

# ---------------------------------------------------------------------------
# EventStorage dependency
# ---------------------------------------------------------------------------

_event_storage: EventStorage | None = None


def _get_event_storage() -> EventStorage:
    """Lazily initialize and return the EventStorage singleton.

    Uses the GDELT DB path from settings. The singleton is safe because
    EventStorage opens a connection per-call (no shared cursor state).
    """
    global _event_storage  # noqa: PLW0603
    if _event_storage is None:
        settings = get_settings()
        _event_storage = EventStorage(db_path=settings.gdelt_db_path)
    return _event_storage


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


@router.get(
    "",
    response_model=PaginatedResponse[EventDTO],
    summary="List events with filters",
    description=(
        "Returns paginated events from the unified GDELT + ACLED events table. "
        "Supports filtering by country (ISO alpha-2), date range, CAMEO event "
        "code prefix, actor substring, Goldstein scale range, title text search, "
        "and source discriminator. Cursor-based keyset pagination on "
        "(event_date DESC, id DESC). Default time window: last 30 days."
    ),
    responses={
        400: {"model": ProblemDetail, "description": "Invalid cursor"},
    },
)
async def list_events(
    country: str | None = Query(None, description="ISO 3166-1 alpha-2 country code"),
    start_date: str | None = Query(None, description="Start date (YYYY-MM-DD, inclusive)"),
    end_date: str | None = Query(None, description="End date (YYYY-MM-DD, inclusive)"),
    cameo_code: str | None = Query(None, description="CAMEO event code prefix match"),
    actor: str | None = Query(None, description="Actor code substring match (either actor)"),
    goldstein_min: float | None = Query(None, ge=-10.0, le=10.0, description="Min Goldstein scale"),
    goldstein_max: float | None = Query(None, ge=-10.0, le=10.0, description="Max Goldstein scale"),
    text: str | None = Query(None, description="Title text substring search"),
    source: str | None = Query(None, description="Source discriminator: 'gdelt' or 'acled'"),
    cursor: str | None = Query(None, description="Opaque pagination cursor from previous page"),
    limit: int = Query(default=50, ge=1, le=200, description="Items per page"),
    _client: str = Depends(verify_api_key),
) -> PaginatedResponse[EventDTO]:
    """List events with full filter surface, cursor-based keyset pagination."""
    storage = _get_event_storage()

    # Decode cursor if provided
    cursor_id: int | None = None
    cursor_date: str | None = None
    if cursor:
        try:
            coords = decode_keyset_cursor(cursor, {"id", "event_date"})
            cursor_id = int(coords["id"])
            cursor_date = str(coords["event_date"])
        except (ValueError, KeyError, TypeError) as exc:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid pagination cursor: {exc}",
            ) from exc

    # Query SQLite via thread pool (sync driver)
    events = await asyncio.to_thread(
        storage.query_events,
        country=country,
        start_date=start_date,
        end_date=end_date,
        cameo_code=cameo_code,
        actor=actor,
        goldstein_min=goldstein_min,
        goldstein_max=goldstein_max,
        text=text,
        source=source,
        cursor_id=cursor_id,
        cursor_date=cursor_date,
        limit=limit,
    )

    # query_events returns limit+1 rows for has_more detection
    has_more = len(events) > limit
    page = events[:limit]

    next_cursor: str | None = None
    if has_more and page:
        last = page[-1]
        next_cursor = encode_keyset_cursor(id=last.id, event_date=last.event_date)

    items = [EventDTO.model_validate(e) for e in page]

    return PaginatedResponse(
        items=items,
        next_cursor=next_cursor,
        has_more=has_more,
    )
