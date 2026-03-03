"""
Async ACLED armed conflict event polling daemon.

Fetches events from the ACLED API (Battles, Explosions/Remote violence,
Violence against civilians), maps them to the unified Event schema with
``source="acled"``, and inserts into SQLite via EventStorage.  Per-run
metrics are persisted to the PostgreSQL IngestRun table.

ACLED authentication uses ``key`` and ``email`` query parameters (NOT
OAuth2).  See https://acleddata.com/acleddatanew/wp-content/uploads/
dlm_uploads/2024/01/API-User-Guide_Jan2024.pdf

Designed for systemd: graceful SIGTERM shutdown, exponential backoff
on API failures, and IngestRun audit trail.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import signal
from datetime import datetime, timedelta, timezone
from typing import Optional

import aiohttp

from src.database.models import Event
from src.database.storage import EventStorage
from src.db.models import IngestRun
from src.db.postgres import get_async_session, init_db
from src.ingest.gdelt_poller import BackoffStrategy
from src.settings import get_settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ACLED API constants
# ---------------------------------------------------------------------------

ACLED_BASE_URL = "https://api.acleddata.com/acled/read"

# ACLED event_type -> approximate CAMEO event_code mapping
_EVENT_TYPE_TO_CAMEO: dict[str, str] = {
    "Battles": "19",                       # Use conventional military force
    "Explosions/Remote violence": "18",    # Assault
    "Violence against civilians": "20",    # Unconventional mass violence
}

# ---------------------------------------------------------------------------
# ISO 3166-1 alpha-3 to alpha-2 mapping for conflict-affected countries.
# ACLED provides iso3 (3-letter) codes; our Event model uses alpha-2.
# Coverage: ~90 countries with ACLED data presence.
# ---------------------------------------------------------------------------

ISO3_TO_ISO2: dict[str, str] = {
    "AFG": "AF", "ALB": "AL", "DZA": "DZ", "AGO": "AO", "ARG": "AR",
    "ARM": "AM", "AZE": "AZ", "BHR": "BH", "BGD": "BD", "BLR": "BY",
    "BEN": "BJ", "BOL": "BO", "BIH": "BA", "BWA": "BW", "BRA": "BR",
    "BFA": "BF", "BDI": "BI", "KHM": "KH", "CMR": "CM", "CAF": "CF",
    "TCD": "TD", "CHL": "CL", "CHN": "CN", "COL": "CO", "COD": "CD",
    "COG": "CG", "CRI": "CR", "CIV": "CI", "HRV": "HR", "CUB": "CU",
    "CYP": "CY", "CZE": "CZ", "DNK": "DK", "DJI": "DJ", "DOM": "DO",
    "ECU": "EC", "EGY": "EG", "SLV": "SV", "GNQ": "GQ", "ERI": "ER",
    "EST": "EE", "SWZ": "SZ", "ETH": "ET", "FRA": "FR", "GAB": "GA",
    "GMB": "GM", "GEO": "GE", "DEU": "DE", "GHA": "GH", "GRC": "GR",
    "GTM": "GT", "GIN": "GN", "GNB": "GW", "GUY": "GY", "HTI": "HT",
    "HND": "HN", "HUN": "HU", "IND": "IN", "IDN": "ID", "IRN": "IR",
    "IRQ": "IQ", "ISR": "IL", "ITA": "IT", "JAM": "JM", "JPN": "JP",
    "JOR": "JO", "KAZ": "KZ", "KEN": "KE", "XKX": "XK", "KWT": "KW",
    "KGZ": "KG", "LAO": "LA", "LVA": "LV", "LBN": "LB", "LSO": "LS",
    "LBR": "LR", "LBY": "LY", "LTU": "LT", "MKD": "MK", "MDG": "MG",
    "MWI": "MW", "MYS": "MY", "MLI": "ML", "MRT": "MR", "MEX": "MX",
    "MDA": "MD", "MNG": "MN", "MNE": "ME", "MAR": "MA", "MOZ": "MZ",
    "MMR": "MM", "NAM": "NA", "NPL": "NP", "NLD": "NL", "NZL": "NZ",
    "NIC": "NI", "NER": "NE", "NGA": "NG", "PRK": "KP", "NOR": "NO",
    "OMN": "OM", "PAK": "PK", "PSE": "PS", "PAN": "PA", "PNG": "PG",
    "PRY": "PY", "PER": "PE", "PHL": "PH", "POL": "PL", "PRT": "PT",
    "QAT": "QA", "ROU": "RO", "RUS": "RU", "RWA": "RW", "SAU": "SA",
    "SEN": "SN", "SRB": "RS", "SLE": "SL", "SGP": "SG", "SVK": "SK",
    "SVN": "SI", "SOM": "SO", "ZAF": "ZA", "KOR": "KR", "SSD": "SS",
    "ESP": "ES", "LKA": "LK", "SDN": "SD", "SUR": "SR", "SWE": "SE",
    "CHE": "CH", "SYR": "SY", "TWN": "TW", "TJK": "TJ", "TZA": "TZ",
    "THA": "TH", "TLS": "TL", "TGO": "TG", "TTO": "TT", "TUN": "TN",
    "TUR": "TR", "TKM": "TM", "UGA": "UG", "UKR": "UA", "ARE": "AE",
    "GBR": "GB", "USA": "US", "URY": "UY", "UZB": "UZ", "VEN": "VE",
    "VNM": "VN", "YEM": "YE", "ZMB": "ZM", "ZWE": "ZW",
}


# ---------------------------------------------------------------------------
# ACLED -> Event mapping
# ---------------------------------------------------------------------------


def _acled_to_event(acled_row: dict) -> Event:
    """Map a single ACLED API result dict to the unified Event schema.

    Args:
        acled_row: Dict from the ACLED API ``data`` array.

    Returns:
        Event dataclass with ``source="acled"``.
    """
    event_id = str(acled_row.get("event_id_cnty", ""))
    event_type = str(acled_row.get("event_type", ""))
    fatalities_raw = acled_row.get("fatalities", "0")
    try:
        fatalities = int(fatalities_raw)
    except (ValueError, TypeError):
        fatalities = 0

    # ACLED uses iso3 (3-letter alpha) for country code
    iso3 = str(acled_row.get("iso3", "")).strip().upper()
    country_iso = ISO3_TO_ISO2.get(iso3)
    if iso3 and country_iso is None:
        logger.warning(
            "Unmapped ACLED iso3 code: %s (event %s) -- setting country_iso=None",
            iso3, event_id,
        )

    # Goldstein scale: -10.0 for lethal, -8.0 for non-lethal conflict
    goldstein = -10.0 if fatalities > 0 else -8.0

    # Truncate notes to 200 chars for title
    notes = str(acled_row.get("notes", ""))[:200]

    return Event(
        gdelt_id=f"ACLED-{event_id}",
        content_hash=hashlib.sha256(f"acled|{event_id}".encode()).hexdigest()[:32],
        time_window=str(acled_row.get("event_date", "")),
        event_date=str(acled_row.get("event_date", "")),
        actor1_code=acled_row.get("actor1"),
        actor2_code=acled_row.get("actor2"),
        event_code=_EVENT_TYPE_TO_CAMEO.get(event_type, "19"),
        quad_class=4,  # All three selected types are Material Conflict
        goldstein_scale=goldstein,
        num_mentions=None,
        num_sources=None,
        tone=None,
        url=None,
        title=notes,
        domain=None,
        country_iso=country_iso,
        source="acled",
    )


# ---------------------------------------------------------------------------
# ACLED API Client
# ---------------------------------------------------------------------------


class ACLEDClient:
    """ACLED REST API client with key+email authentication.

    Authentication is via ``key`` and ``email`` query parameters on every
    request -- not OAuth2 (per ACLED API documentation).
    """

    def __init__(self, key: str, email: str) -> None:
        self._key = key
        self._email = email

    async def fetch_events(
        self,
        session: aiohttp.ClientSession,
        since_date: str,
        event_types: list[str] | None = None,
    ) -> list[dict]:
        """Fetch ACLED events since a given date, paginating through all results.

        Args:
            session: Active aiohttp session.
            since_date: Inclusive start date (YYYY-MM-DD).
            event_types: Filter by event types (default: all 3 conflict types).

        Returns:
            List of raw ACLED event dicts.
        """
        end_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        params: dict[str, str] = {
            "key": self._key,
            "email": self._email,
            "event_date": f"{since_date}|{end_date}",
            "event_date_where": "BETWEEN",
            "limit": "5000",
        }

        if event_types:
            params["event_type"] = "|".join(event_types)

        all_events: list[dict] = []
        page = 1

        while True:
            params["page"] = str(page)

            async with session.get(
                ACLED_BASE_URL,
                params=params,
                timeout=aiohttp.ClientTimeout(total=60),
            ) as resp:
                if resp.status == 401:
                    raise RuntimeError("ACLED authentication failed -- check ACLED_EMAIL and ACLED_PASSWORD settings")
                if resp.status == 403:
                    raise RuntimeError("ACLED API access forbidden -- verify API key permissions")
                if resp.status != 200:
                    raise RuntimeError(f"ACLED API returned HTTP {resp.status}")

                data = await resp.json()

            events = data.get("data", [])
            if not events:
                break

            all_events.extend(events)
            logger.debug("ACLED page %d: %d events", page, len(events))

            if len(events) < 5000:
                break  # Last page
            page += 1

        return all_events


# ---------------------------------------------------------------------------
# Poller daemon
# ---------------------------------------------------------------------------


class ACLEDPoller:
    """Daily ACLED conflict event ingestion daemon.

    Lifecycle:
      1. Fetch events since last successful run date via ACLEDClient.
      2. Map to unified Event schema via ``_acled_to_event()``.
      3. Insert via ``EventStorage.insert_events()`` (INSERT OR IGNORE on gdelt_id).
      4. Record IngestRun metrics to PostgreSQL with daemon_type="acled".
      5. On SIGTERM, exit cleanly.
    """

    def __init__(
        self,
        event_storage: EventStorage,
        poll_interval: int | None = None,
        backoff: BackoffStrategy | None = None,
    ) -> None:
        settings = get_settings()
        self.event_storage = event_storage
        self.poll_interval = poll_interval or settings.acled_poll_interval
        self.backoff = backoff or BackoffStrategy(base=300.0, max_delay=3600.0)

        self._shutdown = asyncio.Event()
        self._client = ACLEDClient(
            key=settings.acled_password,
            email=settings.acled_email,
        )
        self._event_types = settings.acled_event_types

        # Track last successful fetch date (defaults to 30 days ago)
        self._last_success_date: str = (
            datetime.now(timezone.utc) - timedelta(days=30)
        ).strftime("%Y-%m-%d")

    async def run(self) -> None:
        """Main event loop with graceful shutdown support."""
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, self._handle_shutdown)

        # Initialize PostgreSQL engine (needed for IngestRun persistence)
        init_db()

        logger.info(
            "ACLED poller starting (interval=%ds, event_types=%s)",
            self.poll_interval,
            self._event_types,
        )

        while not self._shutdown.is_set():
            run_start = datetime.now(timezone.utc)
            try:
                await self._poll_once()
                self.backoff.reset()
            except Exception as exc:
                logger.error("ACLED poll cycle failed: %s", exc, exc_info=True)
                await self._record_run(
                    started_at=run_start,
                    status="failed",
                    error_message=str(exc),
                )
                delay = self.backoff.next_delay()
                logger.info(
                    "ACLED backing off for %.0fs (failure #%d)",
                    delay, self.backoff.failures,
                )
                try:
                    await asyncio.wait_for(self._shutdown.wait(), timeout=delay)
                except asyncio.TimeoutError:
                    pass
                continue

            # Wait for next poll interval (or shutdown signal)
            try:
                await asyncio.wait_for(
                    self._shutdown.wait(), timeout=self.poll_interval,
                )
            except asyncio.TimeoutError:
                pass  # Normal timeout -- poll again

        logger.info("ACLED poller shut down cleanly")

    def _handle_shutdown(self) -> None:
        """Signal handler for SIGTERM/SIGINT."""
        logger.info("ACLED poller received shutdown signal")
        self._shutdown.set()

    async def _poll_once(self) -> None:
        """Execute a single poll cycle: fetch -> map -> insert -> record."""
        run_start = datetime.now(timezone.utc)

        async with aiohttp.ClientSession() as session:
            # Fetch events since last success
            raw_events = await self._client.fetch_events(
                session,
                since_date=self._last_success_date,
                event_types=self._event_types,
            )

            events_fetched = len(raw_events)

            if events_fetched == 0:
                logger.info("ACLED poll: no new events since %s", self._last_success_date)
                await self._record_run(
                    started_at=run_start,
                    status="success",
                    events_fetched=0,
                    events_new=0,
                    events_duplicate=0,
                )
                return

            # Map to unified Event schema
            events = [_acled_to_event(e) for e in raw_events]

            # Insert into SQLite (dedup via INSERT OR IGNORE on gdelt_id)
            events_new = await asyncio.to_thread(
                self.event_storage.insert_events, events,
            )
            events_duplicate = events_fetched - events_new

            logger.info(
                "ACLED poll complete: fetched=%d, new=%d, dup=%d",
                events_fetched, events_new, events_duplicate,
            )

            # Record metrics
            await self._record_run(
                started_at=run_start,
                status="success",
                events_fetched=events_fetched,
                events_new=events_new,
                events_duplicate=events_duplicate,
            )

            # Update last success date to today
            self._last_success_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    async def _record_run(
        self,
        started_at: datetime,
        status: str,
        events_fetched: int = 0,
        events_new: int = 0,
        events_duplicate: int = 0,
        error_message: Optional[str] = None,
    ) -> None:
        """Persist an IngestRun row to PostgreSQL."""
        try:
            async for session in get_async_session():
                run = IngestRun(
                    started_at=started_at,
                    completed_at=datetime.now(timezone.utc),
                    status=status,
                    daemon_type="acled",
                    events_fetched=events_fetched,
                    events_new=events_new,
                    events_duplicate=events_duplicate,
                    error_message=error_message,
                )
                session.add(run)
        except Exception as exc:
            # PostgreSQL may be down -- log but do not crash the daemon
            logger.warning("Failed to record ACLED IngestRun: %s", exc)
