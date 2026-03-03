"""
Shared in-memory cache for government travel advisories.

This module is the canonical home for advisory data, imported by both the
``/api/v1/advisories`` route (reader) and ``AdvisoryPoller`` (writer).
By keeping the store free of Pydantic imports, we avoid circular
dependencies between the ingest and API layers.

Thread-safety: The GIL protects list/float assignment. For the replace-
all-at-once update pattern used here, no additional locking is needed.
"""

from __future__ import annotations

import time


class AdvisoryStore:
    """In-memory cache for government travel advisories, populated by poller.

    The poller calls ``update()`` with the full list of advisories after
    each successful fetch cycle.  The API route calls ``get()`` to read
    the current cache.  Data is stored as plain dicts so this module has
    zero Pydantic coupling.
    """

    _advisories: list[dict] = []
    _updated_at: float = 0.0

    @classmethod
    def update(cls, advisories: list[dict]) -> None:
        """Replace the cached advisory list atomically.

        Args:
            advisories: Full list of advisory dicts from both sources.
        """
        cls._advisories = list(advisories)  # defensive copy
        cls._updated_at = time.time()

    @classmethod
    def get(cls, country: str | None = None) -> list[dict]:
        """Return cached advisories, optionally filtered by country ISO code.

        Args:
            country: ISO 3166-1 alpha-2 code to filter by (case-insensitive).

        Returns:
            List of advisory dicts matching the filter (or all if no filter).
        """
        if country is None:
            return list(cls._advisories)

        country_upper = country.upper()
        return [
            a for a in cls._advisories
            if (a.get("country_iso") or "").upper() == country_upper
        ]

    @classmethod
    def last_updated(cls) -> float:
        """Return the UNIX timestamp of the last successful update.

        Returns 0.0 if never updated.
        """
        return cls._updated_at

    @classmethod
    def clear(cls) -> None:
        """Reset the cache (useful for testing)."""
        cls._advisories = []
        cls._updated_at = 0.0
