"""
Polymarket Gamma API client with circuit breaker and geopolitical tag filtering.

Fetches active prediction markets from gamma-api.polymarket.com, sorted by
volume, then filters client-side by event tag labels for geopolitical relevance.
Circuit breaker pattern prevents cascade failures when the upstream API is degraded.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import aiohttp
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)

# Circuit breaker recovery window (seconds)
_CIRCUIT_RECOVERY_SECONDS = 300  # 5 minutes
_REQUEST_TIMEOUT = aiohttp.ClientTimeout(total=15)


class PolymarketClient:
    """HTTP client for Polymarket Gamma API with circuit breaker.

    Fetches active events sorted by volume via a single API call, then
    filters client-side by event tag labels against GEO_INCLUDE keywords
    (excluding sports false positives via GEO_EXCLUDE).

    Circuit breaker opens after 5 consecutive failures. While open,
    all calls return empty results until the recovery window elapses,
    at which point a single probe request is allowed through.
    """

    GAMMA_API_BASE = "https://gamma-api.polymarket.com"

    # Inclusion keywords matched against lowercased tag labels
    GEO_INCLUDE: list[str] = [
        "politic",
        "geopolitic",
        "world",
        "war",
        "election",
        "international",
        "government",
        "military",
        "conflict",
        "sanction",
        "nuclear",
        "nato",
        "diplomacy",
        "economy",
        "fed",
        "foreign",
        "middle east",
        "iran",
        "israel",
        "china",
        "russia",
        "ukraine",
        "europe",
        "tariff",
        "trade",
        "greenland",
        "immigration",
    ]

    # Exclusion keywords to reject sports/entertainment false positives
    GEO_EXCLUDE: list[str] = [
        "sports",
        "nba",
        "nfl",
        "soccer",
        "hockey",
        "nhl",
        "mlb",
        "tennis",
        "mma",
        "boxing",
        "cricket",
        "fifa",
        "f1",
        "formula",
        "esports",
    ]

    _CIRCUIT_FAILURE_THRESHOLD = 5

    def __init__(self, session: aiohttp.ClientSession | None = None) -> None:
        self._external_session = session is not None
        self._session = session
        # Circuit breaker state
        self._consecutive_failures: int = 0
        self._circuit_open: bool = False
        self._last_failure_time: float = 0.0

    async def _get_session(self) -> aiohttp.ClientSession:
        """Return existing or lazily-created session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=_REQUEST_TIMEOUT)
            self._external_session = False
        return self._session

    def _is_circuit_open(self) -> bool:
        """Check circuit breaker state, allowing probe after recovery window."""
        if not self._circuit_open:
            return False
        elapsed = time.monotonic() - self._last_failure_time
        if elapsed >= _CIRCUIT_RECOVERY_SECONDS:
            # Allow one probe request (half-open state)
            logger.info(
                "Circuit breaker half-open after %.0fs, allowing probe request",
                elapsed,
            )
            return False
        return True

    def _record_success(self) -> None:
        """Reset circuit breaker on successful request."""
        if self._consecutive_failures > 0:
            logger.info(
                "Polymarket API recovered after %d consecutive failures",
                self._consecutive_failures,
            )
        self._consecutive_failures = 0
        self._circuit_open = False

    def _record_failure(self, error: Exception) -> None:
        """Increment failure count, open circuit after threshold."""
        self._consecutive_failures += 1
        self._last_failure_time = time.monotonic()
        logger.warning(
            "Polymarket API failure (%d/%d): %s",
            self._consecutive_failures,
            self._CIRCUIT_FAILURE_THRESHOLD,
            error,
        )
        if self._consecutive_failures >= self._CIRCUIT_FAILURE_THRESHOLD:
            self._circuit_open = True
            logger.error(
                "Circuit breaker OPEN after %d consecutive failures. "
                "Will retry after %ds.",
                self._consecutive_failures,
                _CIRCUIT_RECOVERY_SECONDS,
            )

    @retry(
        retry=retry_if_exception_type((aiohttp.ClientError, TimeoutError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=2, max=16),
        reraise=True,
    )
    async def _get_json(self, url: str, params: dict[str, Any] | None = None) -> Any:
        """GET request with retry. Raises on non-200 or transport error.

        Handles HTTP 429 specifically: reads the Retry-After header and
        sleeps for the indicated duration (capped at 60s) before raising
        so tenacity can retry with backoff.
        """
        import asyncio as _asyncio

        session = await self._get_session()
        async with session.get(url, params=params) as resp:
            if resp.status == 429:
                retry_after = min(
                    int(resp.headers.get("Retry-After", "30")), 60
                )
                logger.warning(
                    "Polymarket 429 rate limit on %s, retry after %ds",
                    url,
                    retry_after,
                )
                await _asyncio.sleep(retry_after)
                raise aiohttp.ClientResponseError(
                    resp.request_info,
                    resp.history,
                    status=429,
                    message=f"HTTP 429 rate limited from {url}",
                )
            if resp.status != 200:
                raise aiohttp.ClientResponseError(
                    resp.request_info,
                    resp.history,
                    status=resp.status,
                    message=f"HTTP {resp.status} from {url}",
                )
            return await resp.json()

    def _is_geo_event(self, event: dict) -> bool:
        """Check if an event's tags match geopolitical keywords.

        An event is geopolitical if any tag label contains a GEO_INCLUDE
        keyword AND no tag label contains a GEO_EXCLUDE keyword.
        """
        tags = event.get("tags")
        if not isinstance(tags, list):
            return False

        labels_lower: list[str] = []
        for tag in tags:
            if isinstance(tag, dict):
                label = tag.get("label", "")
                if isinstance(label, str):
                    labels_lower.append(label.lower())

        if not labels_lower:
            return False

        # Reject if any exclusion keyword matches
        for label in labels_lower:
            if any(ex in label for ex in self.GEO_EXCLUDE):
                return False

        # Accept if any inclusion keyword matches
        for label in labels_lower:
            if any(kw in label for kw in self.GEO_INCLUDE):
                return True

        return False

    async def fetch_top_geopolitical(
        self, limit: int = 10, fetch_limit: int = 100,
    ) -> tuple[list[dict], int]:
        """Fetch top geopolitical events by volume in a single API call.

        Makes one ``GET /events`` call sorted by volume descending, then
        filters client-side by tag labels for geopolitical relevance.

        Args:
            limit: Maximum geo events to return (top-N after filtering).
            fetch_limit: How many events to pull from the API before filtering.

        Returns:
            Tuple of (top geo events, total geo count from the fetched set).
            Never raises — returns ([], 0) on failure or circuit breaker open.
        """
        if self._is_circuit_open():
            logger.debug("Circuit breaker open, returning empty market list")
            return [], 0

        try:
            url = f"{self.GAMMA_API_BASE}/events"
            params = {
                "active": "true",
                "closed": "false",
                "order": "volume",
                "ascending": "false",
                "limit": str(fetch_limit),
            }
            all_events = await self._get_json(url, params=params)

            if not isinstance(all_events, list):
                logger.warning(
                    "Unexpected events response type: %s",
                    type(all_events).__name__,
                )
                self._record_failure(ValueError("events response not a list"))
                return [], 0

            # Filter to geopolitical events (already sorted by volume from API)
            geo_events: list[dict] = []
            for event in all_events:
                if not isinstance(event, dict):
                    continue
                if not event.get("id") or not event.get("title"):
                    continue
                if self._is_geo_event(event):
                    geo_events.append(event)

            total_geo = len(geo_events)
            self._record_success()
            logger.info(
                "Fetched %d geo events from %d total (returning top %d)",
                total_geo,
                len(all_events),
                min(limit, total_geo),
            )
            return geo_events[:limit], total_geo

        except Exception as exc:
            self._record_failure(exc)
            return [], 0

    async def fetch_geopolitical_markets(self, limit: int = 200) -> list[dict]:
        """Fetch all geopolitical events (convenience wrapper for matching cycle).

        Returns the full geo-filtered set without top-N slicing. Used by
        PolymarketComparisonService.run_matching_cycle() which needs all
        events for exhaustive prediction matching.

        Never raises. Returns empty list on failure.
        """
        events, _ = await self.fetch_top_geopolitical(
            limit=limit, fetch_limit=max(limit, 200),
        )
        return events

    async def fetch_event_prices(self, event_id: str) -> list[dict]:
        """Fetch current market prices for a specific event.

        Returns list of market dicts with outcomePrices data suitable
        for snapshot capture. Returns empty list on failure.

        Args:
            event_id: The Polymarket event identifier.

        Returns:
            List of market dicts (typically one per binary outcome).
        """
        if self._is_circuit_open():
            return []

        try:
            url = f"{self.GAMMA_API_BASE}/events/{event_id}"
            event_data = await self._get_json(url)

            if not isinstance(event_data, dict):
                logger.warning(
                    "Unexpected event response for %s: %s",
                    event_id,
                    type(event_data).__name__,
                )
                return []

            markets = event_data.get("markets", [])
            if not isinstance(markets, list):
                return []

            # Filter to markets with price data
            valid_markets: list[dict] = []
            for market in markets:
                if not isinstance(market, dict):
                    continue
                # outcomePrices or bestBid/bestAsk should be present
                if market.get("outcomePrices") or market.get("bestBid") is not None:
                    valid_markets.append(market)

            return valid_markets

        except Exception as exc:
            logger.warning("Failed to fetch prices for event %s: %s", event_id, exc)
            self._record_failure(exc)
            return []

    async def fetch_event_details(self, event_id: str) -> dict | None:
        """Fetch full event data including resolution metadata.

        Unlike fetch_event_prices() which filters to markets with price data,
        this returns the raw event dict with all market fields (closed,
        resolutionSource, automaticallyResolved, umaResolutionStatus, etc.).

        Returns None on failure or circuit breaker open.
        """
        if self._is_circuit_open():
            return None
        try:
            url = f"{self.GAMMA_API_BASE}/events/{event_id}"
            data = await self._get_json(url)
            self._record_success()
            return data if isinstance(data, dict) else None
        except Exception as exc:
            logger.warning("Failed to fetch event details for %s: %s", event_id, exc)
            self._record_failure(exc)
            return None

    @property
    def circuit_state(self) -> str:
        """Return current circuit breaker state: closed | open | half-open."""
        if not self._circuit_open:
            return "closed"
        elapsed = time.monotonic() - self._last_failure_time
        if elapsed >= _CIRCUIT_RECOVERY_SECONDS:
            return "half-open"
        return "open"

    @property
    def consecutive_failures(self) -> int:
        """Current consecutive failure count for admin status exposure."""
        return self._consecutive_failures

    async def close(self) -> None:
        """Close internally-created session. No-op for external sessions."""
        if self._session and not self._external_session:
            await self._session.close()
            self._session = None
