"""
Async government travel advisory polling daemon.

Fetches travel advisories from two sources daily:
  - US State Department (cadataapi.state.gov JSON API, no auth)
  - UK FCDO (GOV.UK Content API, no auth)

EU EEAS was evaluated and dropped -- no structured API or machine-readable
feed exists as of 2026-03.  Revisit if a structured endpoint emerges.

Advisory data is pushed to the shared ``AdvisoryStore`` in-memory cache
(``src.ingest.advisory_store``) for consumption by the ``/api/v1/advisories``
route.  Per-run metrics are persisted to the PostgreSQL IngestRun table.

Designed for systemd: graceful SIGTERM shutdown, exponential backoff
on API failures, and IngestRun audit trail.
"""

from __future__ import annotations

import asyncio
import logging
import re
import signal
from datetime import datetime, timezone
from typing import Optional

import aiohttp

from src.db.models import IngestRun
from src.db.postgres import get_async_session, init_db
from src.ingest.advisory_store import AdvisoryStore
from src.ingest.gdelt_poller import BackoffStrategy
from src.settings import get_settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Country name -> ISO 3166-1 alpha-2 mapping
# Used by both State Dept (country names in JSON) and FCDO (country slugs).
# Lowercase, normalised (hyphens/spaces stripped for FCDO slug matching).
# ---------------------------------------------------------------------------

COUNTRY_NAME_TO_ISO: dict[str, str] = {
    "afghanistan": "AF",
    "albania": "AL",
    "algeria": "DZ",
    "andorra": "AD",
    "angola": "AO",
    "antigua and barbuda": "AG",
    "argentina": "AR",
    "armenia": "AM",
    "australia": "AU",
    "austria": "AT",
    "azerbaijan": "AZ",
    "bahamas": "BS",
    "the bahamas": "BS",
    "bahrain": "BH",
    "bangladesh": "BD",
    "barbados": "BB",
    "belarus": "BY",
    "belgium": "BE",
    "belize": "BZ",
    "benin": "BJ",
    "bhutan": "BT",
    "bolivia": "BO",
    "bosnia and herzegovina": "BA",
    "botswana": "BW",
    "brazil": "BR",
    "brunei": "BN",
    "bulgaria": "BG",
    "burkina faso": "BF",
    "burma": "MM",
    "myanmar": "MM",
    "burundi": "BI",
    "cabo verde": "CV",
    "cape verde": "CV",
    "cambodia": "KH",
    "cameroon": "CM",
    "canada": "CA",
    "central african republic": "CF",
    "chad": "TD",
    "chile": "CL",
    "china": "CN",
    "colombia": "CO",
    "comoros": "KM",
    "congo, democratic republic of the": "CD",
    "democratic republic of the congo": "CD",
    "congo, republic of the": "CG",
    "republic of the congo": "CG",
    "costa rica": "CR",
    "cote d'ivoire": "CI",
    "ivory coast": "CI",
    "croatia": "HR",
    "cuba": "CU",
    "cyprus": "CY",
    "czech republic": "CZ",
    "czechia": "CZ",
    "denmark": "DK",
    "djibouti": "DJ",
    "dominica": "DM",
    "dominican republic": "DO",
    "ecuador": "EC",
    "egypt": "EG",
    "el salvador": "SV",
    "equatorial guinea": "GQ",
    "eritrea": "ER",
    "estonia": "EE",
    "eswatini": "SZ",
    "swaziland": "SZ",
    "ethiopia": "ET",
    "fiji": "FJ",
    "finland": "FI",
    "france": "FR",
    "gabon": "GA",
    "gambia": "GM",
    "the gambia": "GM",
    "georgia": "GE",
    "germany": "DE",
    "ghana": "GH",
    "greece": "GR",
    "grenada": "GD",
    "guatemala": "GT",
    "guinea": "GN",
    "guinea-bissau": "GW",
    "guineabissau": "GW",
    "guyana": "GY",
    "haiti": "HT",
    "honduras": "HN",
    "hungary": "HU",
    "iceland": "IS",
    "india": "IN",
    "indonesia": "ID",
    "iran": "IR",
    "iraq": "IQ",
    "ireland": "IE",
    "israel": "IL",
    "italy": "IT",
    "jamaica": "JM",
    "japan": "JP",
    "jordan": "JO",
    "kazakhstan": "KZ",
    "kenya": "KE",
    "kiribati": "KI",
    "north korea": "KP",
    "korea, north": "KP",
    "south korea": "KR",
    "korea, south": "KR",
    "kosovo": "XK",
    "kuwait": "KW",
    "kyrgyzstan": "KG",
    "laos": "LA",
    "latvia": "LV",
    "lebanon": "LB",
    "lesotho": "LS",
    "liberia": "LR",
    "libya": "LY",
    "liechtenstein": "LI",
    "lithuania": "LT",
    "luxembourg": "LU",
    "madagascar": "MG",
    "malawi": "MW",
    "malaysia": "MY",
    "maldives": "MV",
    "mali": "ML",
    "malta": "MT",
    "marshall islands": "MH",
    "mauritania": "MR",
    "mauritius": "MU",
    "mexico": "MX",
    "micronesia": "FM",
    "moldova": "MD",
    "monaco": "MC",
    "mongolia": "MN",
    "montenegro": "ME",
    "morocco": "MA",
    "mozambique": "MZ",
    "namibia": "NA",
    "nauru": "NR",
    "nepal": "NP",
    "netherlands": "NL",
    "the netherlands": "NL",
    "new zealand": "NZ",
    "nicaragua": "NI",
    "niger": "NE",
    "nigeria": "NG",
    "north macedonia": "MK",
    "macedonia": "MK",
    "norway": "NO",
    "oman": "OM",
    "pakistan": "PK",
    "palau": "PW",
    "panama": "PA",
    "papua new guinea": "PG",
    "paraguay": "PY",
    "peru": "PE",
    "philippines": "PH",
    "poland": "PL",
    "portugal": "PT",
    "qatar": "QA",
    "romania": "RO",
    "russia": "RU",
    "russian federation": "RU",
    "rwanda": "RW",
    "saint kitts and nevis": "KN",
    "saint lucia": "LC",
    "saint vincent and the grenadines": "VC",
    "samoa": "WS",
    "san marino": "SM",
    "sao tome and principe": "ST",
    "saudi arabia": "SA",
    "senegal": "SN",
    "serbia": "RS",
    "seychelles": "SC",
    "sierra leone": "SL",
    "singapore": "SG",
    "slovakia": "SK",
    "slovenia": "SI",
    "solomon islands": "SB",
    "somalia": "SO",
    "south africa": "ZA",
    "south sudan": "SS",
    "spain": "ES",
    "sri lanka": "LK",
    "sudan": "SD",
    "suriname": "SR",
    "sweden": "SE",
    "switzerland": "CH",
    "syria": "SY",
    "syrian arab republic": "SY",
    "taiwan": "TW",
    "tajikistan": "TJ",
    "tanzania": "TZ",
    "thailand": "TH",
    "timor-leste": "TL",
    "timorleste": "TL",
    "east timor": "TL",
    "togo": "TG",
    "tonga": "TO",
    "trinidad and tobago": "TT",
    "tunisia": "TN",
    "turkey": "TR",
    "turkmenistan": "TM",
    "tuvalu": "TV",
    "uganda": "UG",
    "ukraine": "UA",
    "united arab emirates": "AE",
    "united kingdom": "GB",
    "united states": "US",
    "uruguay": "UY",
    "uzbekistan": "UZ",
    "vanuatu": "VU",
    "venezuela": "VE",
    "vietnam": "VN",
    "yemen": "YE",
    "zambia": "ZM",
    "zimbabwe": "ZW",
    # Territories / special cases seen in advisory data
    "hong kong": "HK",
    "macau": "MO",
    "palestinian territories": "PS",
    "west bank and gaza": "PS",
    "israel, the west bank and gaza": "IL",
    "occupied palestinian territories": "PS",
    "the occupied palestinian territories": "PS",
    "curacao": "CW",
    "bermuda": "BM",
    "cayman islands": "KY",
    "turks and caicos islands": "TC",
    "british virgin islands": "VG",
}


def _normalise_country_name(name: str) -> str:
    """Normalise a country name for dict lookup.

    Lowercases, strips whitespace, collapses multiple spaces.
    """
    return re.sub(r"\s+", " ", name.strip().lower())


def _slug_to_country_name(slug: str) -> str:
    """Convert a FCDO URL slug to a country name for lookup.

    FCDO slugs: ``afghanistan``, ``british-virgin-islands``, etc.
    We convert hyphens to spaces for dict lookup.
    """
    return slug.replace("-", " ").strip().lower()


def country_name_to_iso(name: str) -> str | None:
    """Look up ISO alpha-2 code from a country name.

    Tries exact match first, then normalised match.
    Returns None (with warning) if not found.
    """
    normalised = _normalise_country_name(name)
    iso = COUNTRY_NAME_TO_ISO.get(normalised)
    if iso is None:
        logger.debug("No ISO mapping for country name: %r", name)
    return iso


# ---------------------------------------------------------------------------
# US State Department client
# ---------------------------------------------------------------------------

# Level descriptions as used by State Dept
_US_LEVEL_DESCRIPTIONS: dict[int, str] = {
    1: "Exercise Normal Precautions",
    2: "Exercise Increased Caution",
    3: "Reconsider Travel",
    4: "Do Not Travel",
}

_LEVEL_PATTERN = re.compile(r"Level\s+(\d)", re.IGNORECASE)


class StateDeptClient:
    """Client for the US State Department Travel Advisory API.

    Endpoint: https://cadataapi.state.gov/api/TravelAdvisories
    No authentication required.
    """

    URL = "https://cadataapi.state.gov/api/TravelAdvisories"

    @staticmethod
    def _parse_level(title: str) -> int:
        """Extract advisory level (1-4) from title string.

        Expected format: ``"CountryName - Level N: Description"``
        Returns 1 if parsing fails.
        """
        match = _LEVEL_PATTERN.search(title)
        if match:
            level = int(match.group(1))
            return max(1, min(4, level))  # clamp to valid range
        return 1

    async def fetch(self, session: aiohttp.ClientSession) -> list[dict]:
        """Fetch all current State Dept travel advisories.

        Returns:
            List of advisory dicts ready for AdvisoryStore.
        """
        try:
            async with session.get(
                self.URL,
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                if resp.status != 200:
                    logger.warning(
                        "State Dept API returned HTTP %d", resp.status,
                    )
                    return []
                data = await resp.json(content_type=None)
        except Exception as exc:
            logger.warning("State Dept fetch failed: %s", exc)
            return []

        advisories: list[dict] = []
        # API returns a list of advisory objects
        items = data if isinstance(data, list) else data.get("data", data.get("value", []))
        if not isinstance(items, list):
            logger.warning("Unexpected State Dept response structure: %s", type(items))
            return []

        for item in items:
            title = str(item.get("advisory_text", item.get("title", "")))
            country_name = str(item.get("country", item.get("country_name", "")))
            level = self._parse_level(title)

            iso = country_name_to_iso(country_name)

            advisories.append({
                "source": "us_state_dept",
                "country_iso": iso,
                "level": level,
                "level_description": _US_LEVEL_DESCRIPTIONS.get(level, "Unknown"),
                "title": title[:300],
                "summary": str(item.get("advisory_text", title))[:500],
                "published_at": item.get("date_published"),
                "updated_at": item.get("date_updated"),
                "url": item.get("link"),
            })

        logger.info("State Dept: fetched %d advisories", len(advisories))
        return advisories


# ---------------------------------------------------------------------------
# UK FCDO client
# ---------------------------------------------------------------------------


class FCDOClient:
    """Client for the UK FCDO travel advice via GOV.UK Content API.

    Index: https://www.gov.uk/api/content/foreign-travel-advice
    Per-country: https://www.gov.uk/api/content{base_path}
    No authentication required.
    """

    INDEX_URL = "https://www.gov.uk/api/content/foreign-travel-advice"

    @staticmethod
    def _alert_to_level(alert_statuses: list[str]) -> int:
        """Map FCDO alert_status values to a normalised 1-4 level.

        Mapping:
          - avoid_all_travel_to_whole_country -> 4
          - avoid_all_travel_to_parts -> 3
          - avoid_all_but_essential_travel_to_whole_country -> 3
          - avoid_all_but_essential_travel_to_parts -> 2
          - default (no alerts / exercise caution) -> 1
        """
        if not alert_statuses:
            return 1

        # Check in severity order (highest first)
        status_set = set(alert_statuses)
        if "avoid_all_travel_to_whole_country" in status_set:
            return 4
        if "avoid_all_travel_to_parts" in status_set:
            return 3
        if "avoid_all_but_essential_travel_to_whole_country" in status_set:
            return 3
        if "avoid_all_but_essential_travel_to_parts" in status_set:
            return 2
        return 1

    async def fetch(
        self,
        session: aiohttp.ClientSession,
        semaphore: asyncio.Semaphore,
    ) -> list[dict]:
        """Fetch all FCDO travel advisories.

        Fetches the index page first, then individual country pages
        concurrently (bounded by semaphore to respect rate limits).

        Args:
            session: Active aiohttp session.
            semaphore: Concurrency limiter for per-country fetches.

        Returns:
            List of advisory dicts ready for AdvisoryStore.
        """
        # Step 1: Fetch index
        try:
            async with session.get(
                self.INDEX_URL,
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                if resp.status != 200:
                    logger.warning("FCDO index returned HTTP %d", resp.status)
                    return []
                index_data = await resp.json(content_type=None)
        except Exception as exc:
            logger.warning("FCDO index fetch failed: %s", exc)
            return []

        # Extract country links from the index
        links = index_data.get("links", {}).get("children", [])
        if not links:
            logger.warning("FCDO index has no children links")
            return []

        logger.info("FCDO: found %d countries in index, fetching details...", len(links))

        # Step 2: Fetch each country page concurrently (with rate limiting)
        tasks = [
            self._fetch_country(session, semaphore, link)
            for link in links
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        advisories: list[dict] = []
        for result in results:
            if isinstance(result, Exception):
                logger.debug("FCDO country fetch error: %s", result)
                continue
            if result is not None:
                advisories.append(result)

        logger.info("FCDO: fetched %d advisories", len(advisories))
        return advisories

    async def _fetch_country(
        self,
        session: aiohttp.ClientSession,
        semaphore: asyncio.Semaphore,
        link: dict,
    ) -> dict | None:
        """Fetch a single country's FCDO advice page.

        Args:
            session: Active aiohttp session.
            semaphore: Concurrency limiter.
            link: Link dict from the index page (has ``base_path``, ``title``).

        Returns:
            Advisory dict or None if fetch fails.
        """
        base_path = link.get("base_path", "")
        title = link.get("title", "")

        if not base_path:
            return None

        url = f"https://www.gov.uk/api/content{base_path}"

        async with semaphore:
            # Small delay between requests to avoid hammering GOV.UK
            await asyncio.sleep(0.3)

            try:
                async with session.get(
                    url,
                    timeout=aiohttp.ClientTimeout(total=15),
                ) as resp:
                    if resp.status != 200:
                        logger.debug(
                            "FCDO %s returned HTTP %d", title, resp.status,
                        )
                        return None
                    data = await resp.json(content_type=None)
            except Exception as exc:
                logger.debug("FCDO %s fetch failed: %s", title, exc)
                return None

        # Extract alert status from details
        details = data.get("details", {})
        alert_status = details.get("alert_status", [])
        if not isinstance(alert_status, list):
            alert_status = []

        level = self._alert_to_level(alert_status)

        # Extract country slug from base_path for ISO lookup
        # base_path looks like: /foreign-travel-advice/afghanistan
        slug = base_path.rstrip("/").rsplit("/", 1)[-1]
        country_name = _slug_to_country_name(slug)
        iso = COUNTRY_NAME_TO_ISO.get(country_name)
        if iso is None:
            # Try the title as fallback
            iso = country_name_to_iso(title)

        summary = str(details.get("summary", ""))[:500]
        updated_at = data.get("public_updated_at")
        first_published = data.get("first_published_at")

        return {
            "source": "uk_fcdo",
            "country_iso": iso,
            "level": level,
            "level_description": _US_LEVEL_DESCRIPTIONS.get(level, "Unknown"),
            "title": f"{title}",
            "summary": summary if summary else f"FCDO travel advice for {title}",
            "published_at": first_published,
            "updated_at": updated_at,
            "url": f"https://www.gov.uk{base_path}",
        }


# ---------------------------------------------------------------------------
# Poller daemon
# ---------------------------------------------------------------------------


class AdvisoryPoller:
    """Daily government travel advisory polling daemon.

    Lifecycle:
      1. Fetch US State Dept advisories (single JSON request).
      2. Fetch UK FCDO advisories (index + per-country, rate-limited).
      3. Merge both sources (keep all -- multiple per country is valid).
      4. Push to AdvisoryStore in-memory cache.
      5. Record IngestRun metrics to PostgreSQL.
      6. On SIGTERM, exit cleanly.

    If one source fails, the other is still processed.
    """

    def __init__(
        self,
        poll_interval: int | None = None,
        backoff: BackoffStrategy | None = None,
    ) -> None:
        settings = get_settings()
        self.poll_interval = poll_interval or settings.advisory_poll_interval
        self.backoff = backoff or BackoffStrategy(base=300.0, max_delay=3600.0)
        self._shutdown = asyncio.Event()

        self._state_dept_client = StateDeptClient()
        self._fcdo_client = FCDOClient()
        # Concurrency limiter for FCDO per-country fetches
        self._fcdo_semaphore = asyncio.Semaphore(5)

    async def run(self) -> None:
        """Main event loop with graceful shutdown support."""
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, self._handle_shutdown)

        # Initialise PostgreSQL engine for IngestRun persistence
        init_db()

        logger.info(
            "Advisory poller starting (interval=%ds)", self.poll_interval,
        )

        while not self._shutdown.is_set():
            run_start = datetime.now(timezone.utc)
            try:
                await self._poll_once()
                self.backoff.reset()
            except Exception as exc:
                logger.error(
                    "Advisory poll cycle failed: %s", exc, exc_info=True,
                )
                await self._record_run(
                    started_at=run_start,
                    status="failed",
                    error_message=str(exc),
                )
                delay = self.backoff.next_delay()
                logger.info(
                    "Advisory backing off for %.0fs (failure #%d)",
                    delay, self.backoff.failures,
                )
                try:
                    await asyncio.wait_for(
                        self._shutdown.wait(), timeout=delay,
                    )
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

        logger.info("Advisory poller shut down cleanly")

    def _handle_shutdown(self) -> None:
        """Signal handler for SIGTERM/SIGINT."""
        logger.info("Advisory poller received shutdown signal")
        self._shutdown.set()

    async def _poll_once(self) -> None:
        """Execute a single poll cycle: fetch both sources -> merge -> store."""
        run_start = datetime.now(timezone.utc)
        all_advisories: list[dict] = []

        async with aiohttp.ClientSession() as session:
            # Fetch US State Dept (resilient -- returns [] on failure)
            state_dept = await self._state_dept_client.fetch(session)
            all_advisories.extend(state_dept)

            # Fetch UK FCDO (resilient -- returns [] on failure)
            fcdo = await self._fcdo_client.fetch(session, self._fcdo_semaphore)
            all_advisories.extend(fcdo)

        # Push to shared in-memory cache
        AdvisoryStore.update(all_advisories)

        logger.info(
            "Advisory poll complete: state_dept=%d, fcdo=%d, total=%d",
            len(state_dept), len(fcdo), len(all_advisories),
        )

        # Record metrics
        await self._record_run(
            started_at=run_start,
            status="success",
            events_fetched=len(all_advisories),
            events_new=len(all_advisories),
            events_duplicate=0,
        )

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
                    daemon_type="advisory",
                    events_fetched=events_fetched,
                    events_new=events_new,
                    events_duplicate=events_duplicate,
                    error_message=error_message,
                )
                session.add(run)
        except Exception as exc:
            # PostgreSQL may be down -- log but do not crash the daemon
            logger.warning("Failed to record advisory IngestRun: %s", exc)
