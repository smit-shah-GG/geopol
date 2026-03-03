"""
Polymarket Gamma API client with circuit breaker and tag-based geopolitical filtering.

Fetches active prediction markets from gamma-api.polymarket.com, filtering
for geopolitically relevant events via tag discovery. Circuit breaker pattern
prevents cascade failures when the upstream API is degraded.
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

    Discovers geopolitical markets by fetching tags, filtering against
    GEO_KEYWORDS, then pulling active events per matching tag. Events
    are deduplicated by ID before return.

    Circuit breaker opens after 5 consecutive failures. While open,
    all calls return empty results until the recovery window elapses,
    at which point a single probe request is allowed through.
    """

    GAMMA_API_BASE = "https://gamma-api.polymarket.com"

    GEO_KEYWORDS: list[str] = [
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
        """GET request with retry. Raises on non-200 or transport error."""
        session = await self._get_session()
        async with session.get(url, params=params) as resp:
            if resp.status != 200:
                raise aiohttp.ClientResponseError(
                    resp.request_info,
                    resp.history,
                    status=resp.status,
                    message=f"HTTP {resp.status} from {url}",
                )
            return await resp.json()

    async def fetch_geopolitical_markets(self, limit: int = 100) -> list[dict]:
        """Fetch active Polymarket events matching geopolitical keywords.

        Returns deduplicated list of event dicts. On failure or circuit
        breaker open, returns empty list (never raises).

        Args:
            limit: Maximum events per tag query.

        Returns:
            List of event dicts with at minimum 'id', 'title', 'markets' keys.
        """
        if self._is_circuit_open():
            logger.debug("Circuit breaker open, returning empty market list")
            return []

        try:
            # Phase 1: Discover geopolitical tags
            tags_url = f"{self.GAMMA_API_BASE}/tags"
            all_tags = await self._get_json(tags_url)

            if not isinstance(all_tags, list):
                logger.warning(
                    "Unexpected tags response type: %s", type(all_tags).__name__
                )
                self._record_failure(ValueError("tags response not a list"))
                return []

            geo_tags = []
            for tag in all_tags:
                label = tag.get("label", "") or tag.get("name", "")
                if not isinstance(label, str):
                    continue
                label_lower = label.lower()
                if any(kw in label_lower for kw in self.GEO_KEYWORDS):
                    tag_id = tag.get("id")
                    if tag_id is not None:
                        geo_tags.append(tag)

            logger.info(
                "Discovered %d geopolitical tags from %d total",
                len(geo_tags),
                len(all_tags),
            )

            # Phase 2: Fetch events per matching tag
            seen_ids: set[str] = set()
            events: list[dict] = []

            for tag in geo_tags:
                tag_id = tag.get("id")
                events_url = f"{self.GAMMA_API_BASE}/events"
                params = {
                    "tag_id": str(tag_id),
                    "active": "true",
                    "closed": "false",
                    "limit": str(limit),
                }

                try:
                    tag_events = await self._get_json(events_url, params=params)
                except (aiohttp.ClientError, TimeoutError) as exc:
                    # Individual tag fetch failure is non-fatal
                    logger.warning(
                        "Failed to fetch events for tag %s: %s",
                        tag.get("label", tag_id),
                        exc,
                    )
                    continue

                if not isinstance(tag_events, list):
                    continue

                for event in tag_events:
                    if not isinstance(event, dict):
                        continue
                    event_id = event.get("id")
                    if event_id is None:
                        logger.debug("Skipping event missing 'id' field")
                        continue
                    event_id_str = str(event_id)
                    if event_id_str in seen_ids:
                        continue
                    # Require at minimum title for downstream matching
                    if not event.get("title"):
                        logger.debug(
                            "Skipping event %s: missing 'title'", event_id_str
                        )
                        continue
                    seen_ids.add(event_id_str)
                    events.append(event)

            self._record_success()
            logger.info(
                "Fetched %d unique geopolitical markets from %d tags",
                len(events),
                len(geo_tags),
            )
            return events

        except Exception as exc:
            self._record_failure(exc)
            return []

    async def fetch_top_geopolitical(self, limit: int = 10) -> list[dict]:
        """Return the top geopolitical events sorted by volume descending.

        Fetches the full geo-filtered event set (up to 200), then sorts by
        ``volume`` (string -> float, default 0.0) descending with ``liquidity``
        as secondary sort key. Returns at most ``limit`` events.

        Never raises. Returns empty list on upstream failure or empty data.
        """
        try:
            events = await self.fetch_geopolitical_markets(limit=200)
        except Exception:
            # fetch_geopolitical_markets already swallows; belt-and-suspenders
            return []

        if not events:
            return []

        def _sort_key(e: dict) -> tuple[float, float]:
            try:
                vol = float(e.get("volume", "0") or "0")
            except (ValueError, TypeError):
                vol = 0.0
            try:
                liq = float(e.get("liquidity", "0") or "0")
            except (ValueError, TypeError):
                liq = 0.0
            return (vol, liq)

        events.sort(key=_sort_key, reverse=True)
        return events[:limit]

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

    async def close(self) -> None:
        """Close internally-created session. No-op for external sessions."""
        if self._session and not self._external_session:
            await self._session.close()
            self._session = None
