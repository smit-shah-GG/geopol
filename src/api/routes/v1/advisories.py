"""
Government travel advisory endpoint.

Reads cached advisory data from the shared ``AdvisoryStore`` (populated by
``AdvisoryPoller``).  Returns normalised advisory records from US State
Department and UK FCDO.  If the cache is empty (poller hasn't run yet),
returns an empty list -- not an error.

Requires API key authentication.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, Query

from src.api.middleware.auth import verify_api_key
from src.api.schemas.advisory import AdvisoryDTO
from src.ingest.advisory_store import AdvisoryStore

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get(
    "",
    response_model=list[AdvisoryDTO],
    summary="List government travel advisories",
    description=(
        "Returns cached government travel advisories from US State Department "
        "and UK FCDO.  Data is refreshed daily by the advisory poller daemon.  "
        "Optionally filter by country ISO code.  If the cache is empty "
        "(poller not yet run), returns an empty list."
    ),
)
async def list_advisories(
    country: str | None = Query(None, description="ISO 3166-1 alpha-2 country code filter"),
    _client: str = Depends(verify_api_key),
) -> list[AdvisoryDTO]:
    """Return travel advisories from the in-memory cache."""
    raw = AdvisoryStore.get(country)

    advisories: list[AdvisoryDTO] = []
    for entry in raw:
        try:
            advisories.append(
                AdvisoryDTO(
                    source=entry.get("source", "unknown"),
                    country_iso=entry.get("country_iso"),
                    level=entry.get("level", 1),
                    level_description=entry.get("level_description", "Unknown"),
                    title=entry.get("title", ""),
                    summary=entry.get("summary", ""),
                    published_at=entry.get("published_at"),
                    updated_at=entry.get("updated_at"),
                    url=entry.get("url"),
                )
            )
        except Exception:
            logger.warning(
                "Skipping malformed advisory entry: %s",
                entry.get("title", "<no title>"),
                exc_info=True,
            )

    return advisories
