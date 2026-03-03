# Phase 17: Live Data Feeds & Country Depth - Research

**Researched:** 2026-03-04
**Domain:** Backend API expansion (events/articles endpoints), ACLED conflict data ingestion, government travel advisory ingestion, frontend panel data wiring
**Confidence:** HIGH

## Summary

Phase 17 replaces mock data with live data across all frontend panels and adds two new ingestion pipelines (ACLED armed conflict events, government travel advisories). The core work splits into four domains: (1) new backend API endpoints for events and articles with full filter surfaces and cursor-based pagination, (2) ACLED poller daemon with unified event schema mapping, (3) government advisory poller for US State Dept / UK FCDO / EU EEAS feeds, (4) frontend panel wiring to replace mock data and populate country screen tabs.

The codebase already has well-established patterns for all of these. The GDELT poller (`src/ingest/gdelt_poller.py`) is a production-grade async daemon with backoff, graceful shutdown, and IngestRun audit logging. The cursor-based pagination infrastructure already exists in `src/api/schemas/common.py` (base64url-encoded JSON cursors with keyset pagination). The Panel base class and RefreshScheduler provide the frontend data push patterns. The primary technical challenge is the missing `country` column in the SQLite events table -- GDELT country data (ActionGeo_CountryCode) is buried in `raw_json`, requiring a schema migration before the events API can support country filtering.

**Primary recommendation:** Start with the SQLite schema migration (add `country_iso` column to events table, backfill from raw_json), then build the events/articles API endpoints, then the two ingestion pipelines in parallel, and finally wire the frontend panels.

## Standard Stack

The established libraries/tools for this domain:

### Core (already in project)
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| FastAPI | >=0.115 | API endpoints | Already in use, async native |
| Pydantic | >=2.0 | Request/response DTOs | Already in use for all schemas |
| SQLAlchemy | >=2.0 (asyncpg) | PostgreSQL queries | Already in use for predictions |
| sqlite3 (stdlib) | 3.x | GDELT event queries | Already in use via EventStorage |
| ChromaDB | >=1.4.0 | Article semantic search | Already in use for RSS indexing |
| aiohttp | >=3.9 | External API HTTP client | Already in use by GDELT/RSS daemons |
| feedparser | >=6.0 | RSS/Atom parsing | Already in use by RSS daemon |

### Supporting (new for Phase 17)
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| (none new) | -- | -- | All required libraries already in pyproject.toml |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Direct ACLED API calls | `acled` PyPI package | Unofficial wrapper, adds dependency for thin convenience; direct aiohttp is consistent with existing codebase pattern |
| `fastapi-pagination` | Manual cursor pagination | Library exists but codebase already has `encode_cursor`/`decode_cursor` in `common.py`; adding library for one more endpoint is overkill |

**Installation:**
```bash
# No new packages required -- all dependencies already in pyproject.toml
```

## Architecture Patterns

### Recommended Project Structure
```
src/
├── api/
│   ├── routes/v1/
│   │   ├── events.py         # NEW: GET /events with full filter surface
│   │   ├── articles.py       # NEW: GET /articles with keyword + semantic search
│   │   ├── sources.py        # NEW: GET /sources auto-discovery endpoint
│   │   ├── advisories.py     # NEW: GET /advisories for gov travel warnings
│   │   └── router.py         # MODIFIED: include new routers
│   └── schemas/
│       ├── event.py          # NEW: EventDTO, EventFilterParams
│       ├── article.py        # NEW: ArticleDTO, ArticleFilterParams
│       ├── advisory.py       # NEW: AdvisoryDTO
│       └── source.py         # NEW: SourceStatusDTO
├── database/
│   ├── schema.sql            # MODIFIED: add country_iso column, source column
│   ├── models.py             # MODIFIED: add country_iso, source to Event dataclass
│   └── storage.py            # MODIFIED: add country filter, cursor pagination
├── ingest/
│   ├── acled_poller.py       # NEW: ACLED conflict event poller
│   ├── advisory_poller.py    # NEW: Government advisory poller
│   └── gdelt_poller.py       # MODIFIED: extract country_iso into Event.country_iso
└── settings.py               # MODIFIED: add ACLED credentials, advisory config

frontend/src/
├── components/
│   ├── EventTimelinePanel.ts # MODIFIED: wire to /events API, remove mock data
│   ├── SourcesPanel.ts       # MODIFIED: wire to /sources API instead of health filter
│   └── CountryBriefPage.ts   # MODIFIED: populate tabs with real data
├── services/
│   └── forecast-client.ts    # MODIFIED: add getEvents(), getArticles(), getSources(), getAdvisories()
└── types/
    └── api.ts                # MODIFIED: add EventDTO, ArticleDTO, etc.
```

### Pattern 1: Daemon-per-source (follow existing GDELT/RSS pattern)
**What:** Each external data source gets its own async poller daemon with graceful shutdown, exponential backoff, and IngestRun audit logging.
**When to use:** All new ingestion pipelines (ACLED, advisories).
**Example:**
```python
# Source: existing src/ingest/gdelt_poller.py (the canonical pattern)
class ACLEDPoller:
    """Daily ACLED conflict event poller.

    Follows the same lifecycle as GDELTPoller:
    1. OAuth token acquisition (24h validity)
    2. Fetch events since last successful run
    3. Map to unified Event schema
    4. INSERT OR IGNORE into SQLite
    5. Record IngestRun to PostgreSQL
    """

    def __init__(self, event_storage: EventStorage, ...):
        self._shutdown = asyncio.Event()
        self.backoff = BackoffStrategy(base=300.0)  # 5min base for daily source

    async def _poll_once(self) -> None:
        token = await self._acquire_token()
        events = await self._fetch_events(token, since=self._last_success_date)
        mapped = [self._map_to_event(e) for e in events]
        new_count = await asyncio.to_thread(self.event_storage.insert_events, mapped)
        await self._record_run(...)
```

### Pattern 2: Cursor-based pagination for event/article APIs
**What:** Keyset pagination using the existing `encode_cursor`/`decode_cursor` functions from `src/api/schemas/common.py`. Cursor encodes `(id, event_date)` for events or `(chunk_id, indexed_at)` for articles.
**When to use:** All list endpoints returning potentially large result sets.
**Example:**
```python
# Source: existing src/api/schemas/common.py pattern
from src.api.schemas.common import PaginatedResponse, encode_cursor, decode_cursor

@router.get("/events", response_model=PaginatedResponse[EventDTO])
async def list_events(
    country: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    cameo_code: str | None = None,
    actor: str | None = None,
    goldstein_min: float | None = None,
    goldstein_max: float | None = None,
    text: str | None = None,
    source: str | None = None,  # "gdelt" | "acled" | None (both)
    cursor: str | None = None,
    limit: int = Query(default=50, le=200),
):
    # Decode cursor for keyset pagination
    if cursor:
        coords = decode_cursor(cursor)
        # WHERE (event_date, id) < (coords['ts'], coords['id'])
    ...
```

### Pattern 3: Unified Event Model (GDELT + ACLED merged)
**What:** ACLED events are mapped to the existing GDELT Event schema with a `source` discriminator field. Both query through the same `/events` endpoint.
**When to use:** All ACLED ingestion and all event queries.
**Example:**
```python
# ACLED-to-Event mapping (Claude's discretion per CONTEXT.md)
def _acled_to_event(acled_row: dict) -> Event:
    """Map ACLED fields to GDELT-compatible Event schema."""
    # ACLED event_type -> CAMEO quad_class mapping:
    #   "Battles" -> quad_class=4 (Material Conflict)
    #   "Explosions/Remote violence" -> quad_class=4
    #   "Violence against civilians" -> quad_class=4
    ACLED_QUAD_CLASS = 4  # All three selected types are Material Conflict

    # ACLED event_type -> approximate CAMEO event_code:
    #   "Battles" -> "19" (Use conventional military force)
    #   "Explosions/Remote violence" -> "18" (Assault)
    #   "Violence against civilians" -> "20" (Unconventional mass violence)
    EVENT_TYPE_TO_CAMEO = {
        "Battles": "19",
        "Explosions/Remote violence": "18",
        "Violence against civilians": "20",
    }

    # ACLED fatalities -> approximate Goldstein scale
    # Material conflict base is -10.0, adjust slightly by fatalities
    fatalities = int(acled_row.get("fatalities", 0))
    goldstein = -10.0 if fatalities > 0 else -8.0

    return Event(
        gdelt_id=f"ACLED-{acled_row['event_id_cnty']}",  # Prefixed to avoid collision
        content_hash=hashlib.sha256(
            f"acled|{acled_row['event_id_cnty']}".encode()
        ).hexdigest()[:32],
        time_window=acled_row["event_date"],
        event_date=acled_row["event_date"],
        actor1_code=acled_row.get("actor1"),
        actor2_code=acled_row.get("actor2"),
        event_code=EVENT_TYPE_TO_CAMEO.get(acled_row["event_type"], "19"),
        quad_class=ACLED_QUAD_CLASS,
        goldstein_scale=goldstein,
        num_mentions=None,  # ACLED has no mention count
        num_sources=None,
        tone=None,
        url=None,  # ACLED has no source URL per event
        title=acled_row.get("notes", "")[:200],  # ACLED "notes" serves as description
        domain=None,
        country_iso=acled_row.get("iso"),  # NEW FIELD
        source="acled",  # NEW FIELD: discriminator
    )
```

### Pattern 4: SourcesPanel Auto-Discovery
**What:** Backend `/sources` endpoint returns health/staleness for all data sources. Frontend renders dynamically -- adding a new source on the backend automatically appears in the UI.
**When to use:** SourcesPanel wiring (replaces current health-subsystem filtering approach).
**Example:**
```python
# Backend /sources endpoint
@router.get("/sources", response_model=list[SourceStatusDTO])
async def list_sources(db: AsyncSession = Depends(get_db)):
    """Auto-discover active data sources from IngestRun table + advisory table."""
    sources = []

    # Query last IngestRun per daemon_type
    for daemon_type in ["gdelt", "rss", "acled", "advisory"]:
        last_run = await db.execute(
            select(IngestRun)
            .where(IngestRun.daemon_type == daemon_type)
            .order_by(IngestRun.started_at.desc())
            .limit(1)
        )
        row = last_run.scalar_one_or_none()
        sources.append(SourceStatusDTO(
            name=daemon_type,
            healthy=row is not None and row.status == "success",
            last_update=row.completed_at if row else None,
            events_last_run=row.events_new if row else 0,
            detail=f"{row.status}: {row.events_new} new" if row else "Never run",
        ))

    return sources
```

### Anti-Patterns to Avoid
- **Country filtering via raw_json parsing at query time:** The events table stores GDELT country info only in `raw_json`. Querying `WHERE json_extract(raw_json, '$.ActionGeo_CountryCode') = 'UA'` on every request is catastrophically slow. Add a `country_iso` column and backfill it.
- **Separate event tables for ACLED vs GDELT:** Creates N+1 query complexity, breaks unified `/events` endpoint. Use single table with `source` discriminator.
- **Polling advisories every 30 seconds:** Government advisory feeds update at most once per day. Polling faster wastes bandwidth and may trigger rate limiting.
- **Fetching all ACLED events globally:** ACLED has millions of records. Filter to the three event types (Battles, Explosions/Remote violence, Violence against civilians) at API query time, and only fetch events from the last 30 days per poll.

## Don't Hand-Roll

Problems that look simple but have existing solutions:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Cursor pagination | Custom cursor encoding | Existing `encode_cursor`/`decode_cursor` in `common.py` | Already implemented, tested, base64url JSON keyset |
| OAuth2 token management | Manual token refresh logic | `aiohttp` + simple token cache with expiry check | ACLED tokens last 24h, refresh tokens 14d; a simple `_token_expires_at` timestamp comparison suffices |
| CAMEO code descriptions | Lookup table in Python | Reference `geopol.md` CAMEO taxonomy section | Already documented in the project reference |
| ISO 3166 country code mapping | Custom mapping dict | Existing `country-geometry.ts` (frontend) / GDELT ActionGeo_CountryCode (backend) | Both sides already use ISO 3166-1 alpha-2 |
| Async SQLite queries | asyncio.to_thread wrapper everywhere | Follow existing pattern in `gdelt_poller.py` line 325 | `asyncio.to_thread(self.event_storage.insert_events, events)` is the established pattern |
| HTML-to-text for advisories | Custom parser | GOV.UK Content API returns structured JSON; State Dept returns JSON with HTML summaries -- use existing `trafilatura` for HTML stripping if needed | Already in ingest dependencies |

**Key insight:** The codebase already solves most infrastructure problems. Phase 17 is primarily glue work: exposing existing data through new API endpoints, adding two new pollers following established patterns, and wiring frontend panels to real endpoints instead of mock data.

## Common Pitfalls

### Pitfall 1: SQLite Events Table Missing Country Column
**What goes wrong:** The events API needs `?country=UA` filtering, but the `events` table has no `country_iso` column. Country data is buried in `raw_json` as `ActionGeo_CountryCode`.
**Why it happens:** The original schema was designed for TKG graph construction, not for API-level querying by country.
**How to avoid:** Add `country_iso TEXT` column to events table. Backfill existing rows with `UPDATE events SET country_iso = json_extract(raw_json, '$.ActionGeo_CountryCode')`. Modify `_gdelt_row_to_event()` to populate the new field. Add index: `CREATE INDEX idx_country_iso ON events(country_iso)`.
**Warning signs:** Slow event queries, no results when filtering by country.

### Pitfall 2: SQLite Concurrent Write Contention
**What goes wrong:** The events API reads from SQLite while the GDELT/ACLED pollers write to it. SQLite allows only one writer at a time; WAL mode helps but concurrent read+write from different threads can still cause `SQLITE_BUSY`.
**Why it happens:** FastAPI runs in an async event loop; SQLite operations run via `asyncio.to_thread()`. Multiple threads hitting the same SQLite file simultaneously.
**How to avoid:** The existing `DatabaseConnection` already enables WAL mode and sets `PRAGMA busy_timeout`. For read-heavy API queries, use a separate `DatabaseConnection` instance (or even a read-only connection with `?mode=ro`) from the writer connection used by pollers. Keep queries fast (indexed columns only).
**Warning signs:** `sqlite3.OperationalError: database is locked`.

### Pitfall 3: ACLED OAuth Token Expiry Mid-Poll
**What goes wrong:** ACLED access tokens expire after 24 hours. If the token is acquired at startup and never refreshed, API calls start failing silently with 401.
**Why it happens:** Daily polling means the token may be acquired once and reused indefinitely.
**How to avoid:** Store `_token_acquired_at` timestamp. Before each API call, check if token age > 23 hours (1h safety margin). If expired, re-acquire. Use the refresh token (14-day validity) for seamless renewal.
**Warning signs:** ACLED poller returns 401 errors after first successful run.

### Pitfall 4: ACLED Event ID Collision with GDELT IDs
**What goes wrong:** Both GDELT and ACLED events go into the same `events` table. ACLED uses alphanumeric event IDs (e.g., `AFG12345`), GDELT uses numeric IDs. A collision in the `gdelt_id` UNIQUE column causes INSERT OR IGNORE to silently drop valid ACLED events.
**Why it happens:** The `gdelt_id` column was designed for GDELT only; ACLED IDs have a different format.
**How to avoid:** Prefix ACLED IDs with `ACLED-` (e.g., `ACLED-AFG12345`). This guarantees uniqueness against GDELT numeric IDs. Add the `source` column to the events table to distinguish origin.
**Warning signs:** ACLED events appear to be ingested (no errors) but never appear in queries.

### Pitfall 5: EU EEAS Has No Structured API
**What goes wrong:** The CONTEXT.md specifies three government advisory feeds: US State Dept, UK FCDO, EU EEAS. But research found no structured API for EU EEAS travel advisories.
**Why it happens:** EEAS publishes travel advice as unstructured web pages, not as machine-readable feeds.
**How to avoid:** Two options: (A) Scrape EEAS pages with trafilatura and parse advisory levels heuristically (fragile, maintenance burden). (B) Drop EEAS and use only the two feeds with structured APIs (US State Dept JSON API at `cadataapi.state.gov`, UK FCDO via GOV.UK Content API). **Recommendation: Option B** -- two reliable sources provide sufficient coverage. EU EEAS can be added later if a structured feed becomes available.
**Warning signs:** EEAS scraping breaks on page layout changes; maintenance cost exceeds value.

### Pitfall 6: ChromaDB Semantic Search Performance
**What goes wrong:** The articles endpoint with `?semantic=true` performs a ChromaDB similarity query. If the collection has hundreds of thousands of chunks, queries can be slow (>500ms).
**Why it happens:** ChromaDB is not designed for real-time low-latency queries at scale.
**How to avoid:** Set sensible defaults: `n_results=20`, `similarity_threshold=0.5`. Cache results in Redis with a 5-minute TTL (matching article refresh interval). Consider limiting semantic search to the most recent 30 days of articles via metadata filter.
**Warning signs:** Slow article search responses, ChromaDB memory pressure.

### Pitfall 7: EventTimelinePanel Losing Expanded State on Refresh
**What goes wrong:** The EventTimelinePanel refreshes every ~30 seconds. If the user has expanded a card to see details, the refresh re-renders all rows and collapses the expanded card.
**Why it happens:** `replaceChildren()` destroys and recreates all DOM nodes.
**How to avoid:** Follow the ForecastPanel's diff-based DOM update pattern (already implemented in Phase 16). Compare incoming event IDs with existing rendered IDs. Only add/remove changed rows. Preserve DOM nodes for unchanged events.
**Warning signs:** User complaints about losing their place in the event timeline.

## Code Examples

Verified patterns from the existing codebase:

### SQLite Events Table Migration
```sql
-- Migration: add country_iso and source columns to events table
ALTER TABLE events ADD COLUMN country_iso TEXT;
ALTER TABLE events ADD COLUMN source TEXT NOT NULL DEFAULT 'gdelt';

-- Backfill country_iso from raw_json for existing GDELT events
UPDATE events
SET country_iso = json_extract(raw_json, '$.ActionGeo_CountryCode')
WHERE raw_json IS NOT NULL
  AND country_iso IS NULL;

-- Index for country-based API queries
CREATE INDEX IF NOT EXISTS idx_events_country_iso ON events(country_iso);

-- Composite index for cursor-based pagination (country + date + id)
CREATE INDEX IF NOT EXISTS idx_events_country_date_id
ON events(country_iso, event_date DESC, id DESC);

-- Index for source filtering
CREATE INDEX IF NOT EXISTS idx_events_source ON events(source);
```

### Event API Endpoint (FastAPI)
```python
# Source: follows existing pattern from src/api/routes/v1/countries.py
from fastapi import APIRouter, Depends, Query
from src.api.deps import get_cache
from src.api.middleware.auth import verify_api_key
from src.api.schemas.common import PaginatedResponse, encode_cursor, decode_cursor
from src.api.schemas.event import EventDTO, EventFilterParams
from src.database.storage import EventStorage

router = APIRouter()

@router.get("", response_model=PaginatedResponse[EventDTO])
async def list_events(
    country: str | None = Query(None, description="ISO 3166-1 alpha-2 country code"),
    start_date: str | None = Query(None, description="YYYY-MM-DD start date"),
    end_date: str | None = Query(None, description="YYYY-MM-DD end date"),
    cameo_code: str | None = Query(None, description="CAMEO event code prefix"),
    actor: str | None = Query(None, description="Actor code substring match"),
    goldstein_min: float | None = Query(None, ge=-10.0, le=10.0),
    goldstein_max: float | None = Query(None, ge=-10.0, le=10.0),
    text: str | None = Query(None, description="Title text search"),
    source: str | None = Query(None, description="'gdelt' or 'acled'"),
    cursor: str | None = Query(None, description="Pagination cursor"),
    limit: int = Query(default=50, ge=1, le=200),
    _client: str = Depends(verify_api_key),
):
    """List events with full filter surface, cursor-based pagination."""
    # Default time range: 30 days
    if not start_date and not end_date:
        from datetime import datetime, timedelta
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

    # Build SQLite query (asyncio.to_thread for sync SQLite access)
    events = await asyncio.to_thread(
        _query_events, country, start_date, end_date, cameo_code,
        actor, goldstein_min, goldstein_max, text, source, cursor, limit + 1,
    )

    has_more = len(events) > limit
    page = events[:limit]
    next_cursor = None
    if has_more and page:
        last = page[-1]
        next_cursor = encode_cursor(str(last.id), last.event_date)

    return PaginatedResponse(
        items=[_event_to_dto(e) for e in page],
        next_cursor=next_cursor,
        has_more=has_more,
    )
```

### ACLED API Client
```python
# Source: ACLED API docs at https://acleddata.com/api-documentation/getting-started
class ACLEDClient:
    """ACLED API client with OAuth2 token management."""

    BASE_URL = "https://acleddata.com/api/acled/read"
    TOKEN_URL = "https://acleddata.com/oauth/token"

    def __init__(self, email: str, password: str):
        self._email = email
        self._password = password
        self._token: str | None = None
        self._token_expires_at: float = 0.0

    async def _ensure_token(self, session: aiohttp.ClientSession) -> str:
        """Acquire or refresh OAuth2 token."""
        import time
        if self._token and time.time() < self._token_expires_at:
            return self._token

        async with session.post(self.TOKEN_URL, data={
            "username": self._email,
            "password": self._password,
            "grant_type": "password",
            "client_id": "acled",
        }) as resp:
            if resp.status != 200:
                raise RuntimeError(f"ACLED token acquisition failed: HTTP {resp.status}")
            data = await resp.json()
            self._token = data["access_token"]
            # Token valid for 24h; refresh at 23h for safety
            self._token_expires_at = time.time() + 82800  # 23 hours

        return self._token

    async def fetch_events(
        self,
        session: aiohttp.ClientSession,
        since_date: str,  # YYYY-MM-DD
        event_types: list[str] | None = None,
    ) -> list[dict]:
        """Fetch ACLED events since a given date."""
        token = await self._ensure_token(session)

        params = {
            "_format": "json",
            "event_date": f"{since_date}|{datetime.now().strftime('%Y-%m-%d')}",
            "event_date_where": "BETWEEN",
            "limit": "5000",
        }

        if event_types:
            params["event_type"] = "|".join(event_types)

        headers = {"Authorization": f"Bearer {token}"}

        all_events = []
        page = 1
        while True:
            params["page"] = str(page)
            async with session.get(
                self.BASE_URL, params=params, headers=headers,
                timeout=aiohttp.ClientTimeout(total=60),
            ) as resp:
                if resp.status != 200:
                    raise RuntimeError(f"ACLED API returned HTTP {resp.status}")
                data = await resp.json()

            events = data.get("data", [])
            if not events:
                break
            all_events.extend(events)

            if len(events) < 5000:
                break  # Last page
            page += 1

        return all_events
```

### US State Dept Advisory Client
```python
# Source: https://cadataapi.state.gov/api/TravelAdvisories
class StateDeptAdvisoryClient:
    """US State Department travel advisory client.

    No authentication required. JSON API at cadataapi.state.gov.
    Advisory levels: 1 (Exercise Normal Precautions) to 4 (Do Not Travel).
    """

    BASE_URL = "https://cadataapi.state.gov/api/TravelAdvisories"

    async def fetch_all(self, session: aiohttp.ClientSession) -> list[dict]:
        """Fetch all current travel advisories."""
        async with session.get(
            self.BASE_URL,
            timeout=aiohttp.ClientTimeout(total=30),
        ) as resp:
            if resp.status != 200:
                raise RuntimeError(f"State Dept API returned HTTP {resp.status}")
            data = await resp.json()

        advisories = []
        for item in data:
            # Parse advisory level from title: "CountryName - Level N: Description"
            title = item.get("Title", "")
            level = self._parse_level(title)
            country_codes = item.get("Category", [])  # ISO alpha-2 codes

            advisories.append({
                "source": "us_state_dept",
                "country_iso": country_codes[0] if country_codes else None,
                "level": level,
                "level_description": self._level_to_text(level),
                "title": title,
                "summary": item.get("Summary", ""),
                "published_at": item.get("Published"),
                "updated_at": item.get("Updated"),
                "url": item.get("Link"),
            })

        return advisories

    @staticmethod
    def _parse_level(title: str) -> int:
        """Extract advisory level (1-4) from title string."""
        import re
        match = re.search(r"Level (\d)", title)
        return int(match.group(1)) if match else 0

    @staticmethod
    def _level_to_text(level: int) -> str:
        return {
            1: "Exercise Normal Precautions",
            2: "Exercise Increased Caution",
            3: "Reconsider Travel",
            4: "Do Not Travel",
        }.get(level, "Unknown")
```

### UK FCDO Advisory Client
```python
# Source: GOV.UK Content API at https://content-api.publishing.service.gov.uk
class FCDOAdvisoryClient:
    """UK FCDO travel advice client via GOV.UK Content API.

    No authentication required. JSON API returns structured content
    including alert_status array for risk classification.
    """

    BASE_URL = "https://www.gov.uk/api/content/foreign-travel-advice"
    INDEX_URL = "https://www.gov.uk/api/content/foreign-travel-advice"

    async def fetch_all(self, session: aiohttp.ClientSession) -> list[dict]:
        """Fetch FCDO travel advice for all countries."""
        # First, get the index page with links to all country pages
        async with session.get(
            self.INDEX_URL,
            timeout=aiohttp.ClientTimeout(total=30),
        ) as resp:
            if resp.status != 200:
                raise RuntimeError(f"FCDO index returned HTTP {resp.status}")
            index = await resp.json()

        # Extract country links from the index
        country_links = index.get("links", {}).get("children", [])

        advisories = []
        for link in country_links:
            base_path = link.get("base_path", "")
            country_name = link.get("title", "")

            # Fetch individual country advice
            try:
                async with session.get(
                    f"https://www.gov.uk/api/content{base_path}",
                    timeout=aiohttp.ClientTimeout(total=15),
                ) as resp:
                    if resp.status != 200:
                        continue
                    data = await resp.json()

                details = data.get("details", {})
                alert_status = details.get("alert_status", [])

                advisories.append({
                    "source": "uk_fcdo",
                    "country_name": country_name,
                    "country_iso": self._name_to_iso(country_name),
                    "alert_status": alert_status,
                    "level": self._alert_to_level(alert_status),
                    "updated_at": data.get("public_updated_at"),
                    "url": f"https://www.gov.uk{base_path}",
                })
            except Exception as exc:
                logger.warning("Failed to fetch FCDO advice for %s: %s", country_name, exc)

        return advisories

    @staticmethod
    def _alert_to_level(alert_status: list[str]) -> int:
        """Map FCDO alert_status to numeric level (1-4) for consistency."""
        if "avoid_all_travel_to_whole_country" in alert_status:
            return 4
        if "avoid_all_travel_to_parts" in alert_status:
            return 3
        if "avoid_all_but_essential_travel_to_parts" in alert_status:
            return 2
        return 1
```

### Frontend EventTimelinePanel Wired to API
```typescript
// Source: follows existing ForecastPanel diff-update pattern
export class EventTimelinePanel extends Panel {
  private currentEvents: EventDTO[] = [];
  private expandedId: string | null = null;

  public async refresh(): Promise<void> {
    const response = await forecastClient.getEvents({ limit: 50 });
    this.updateEvents(response.items);
  }

  /** Diff-based update preserving expanded card state. */
  private updateEvents(events: EventDTO[]): void {
    this.setCount(events.length);

    if (events.length === 0) {
      replaceChildren(this.content,
        h('div', { className: 'empty-state' },
          'No events in the last 30 days',
        ),
      );
      return;
    }

    // Diff against current: only re-render changed rows
    const existingIds = new Set(this.currentEvents.map(e => e.id));
    const newIds = new Set(events.map(e => e.id));

    // Remove departed events
    for (const id of existingIds) {
      if (!newIds.has(id)) {
        const el = this.content.querySelector(`[data-event-id="${id}"]`);
        el?.remove();
      }
    }

    // Add/update events
    for (const event of events) {
      if (!existingIds.has(event.id)) {
        // New event: insert at correct position
        const row = this.buildEventRow(event);
        this.content.appendChild(row);
      }
    }

    this.currentEvents = events;
  }
}
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Mock data in EventTimelinePanel | Live GDELT events via API | Phase 17 (this phase) | Events panel shows real data |
| SourcesPanel derives from health subsystems | Dedicated /sources endpoint with auto-discovery | Phase 17 (this phase) | New sources auto-appear without frontend changes |
| GDELT only for event data | GDELT + ACLED unified event model | Phase 17 (this phase) | Armed conflict coverage improved |
| No government advisory data | US State Dept + UK FCDO advisories | Phase 17 (this phase) | Official risk context alongside AI forecasts |

**Deprecated/outdated:**
- EventTimelinePanel `MOCK_EVENTS` array and `MockEvent` interface: replaced by live API data
- SourcesPanel `SOURCE_KEYWORDS` health subsystem filtering: replaced by `/sources` endpoint

## Open Questions

Things that could not be fully resolved:

1. **ACLED API Pagination Parameters**
   - What we know: ACLED defaults to 5000 rows per request. Documentation mentions pagination.
   - What's unclear: The exact pagination parameter names (page/offset/limit) are not explicitly documented in the API reference.
   - Recommendation: Test empirically with a registered account. Use `&page=N` (common pattern for REST APIs with row limits). Fall back to date-range chunking if needed (fetch by week instead of all-at-once).

2. **EU EEAS Advisory Feed**
   - What we know: No structured API or machine-readable feed exists for EU EEAS travel advisories.
   - What's unclear: Whether EEAS has an internal API that could be discovered through their website.
   - Recommendation: **Drop EEAS from Phase 17 scope.** Two reliable sources (US State Dept JSON API, UK FCDO GOV.UK Content API) provide adequate coverage. EEAS can be added in a future phase if a structured feed becomes available. The CONTEXT.md lists EU EEAS as a decision, so this should be surfaced to the user during planning.

3. **SQLite Events Table Country Backfill Completeness**
   - What we know: `raw_json` contains ActionGeo_CountryCode, but it may be NULL for events where GDELT did not geolocate.
   - What's unclear: What percentage of existing events have ActionGeo_CountryCode in their raw_json. Actor1CountryCode / Actor2CountryCode are alternative sources but semantically different (country of actor vs. country of event).
   - Recommendation: Use ActionGeo_CountryCode as primary, fall back to Actor1CountryCode if ActionGeo is NULL. Log the backfill stats to understand coverage.

4. **CountryBriefPage Tab Content Mapping**
   - What we know: 7 tabs exist (overview, forecasts, events, risk-signals, history, entities, calibration). CONTEXT.md says "keep all 6 existing tabs" (discrepancy: there are actually 7).
   - What's unclear: Exactly which tabs get data from which new endpoints.
   - Recommendation: Map as follows: `overview` -> country risk summary + advisory; `forecasts` -> existing country forecasts API; `events` -> new /events?country=XX; `risk-signals` -> advisory data; `history` -> existing forecast history; `entities` -> /events actor aggregation; `calibration` -> existing calibration data.

## Sources

### Primary (HIGH confidence)
- Existing codebase files: `src/database/schema.sql`, `src/database/storage.py`, `src/database/models.py`, `src/ingest/gdelt_poller.py`, `src/ingest/rss_daemon.py`, `src/api/routes/v1/countries.py`, `src/api/schemas/common.py`, `frontend/src/components/EventTimelinePanel.ts`, `frontend/src/components/CountryBriefPage.ts`, `frontend/src/components/SourcesPanel.ts`, `frontend/src/services/forecast-client.ts`, `frontend/src/screens/dashboard-screen.ts`
- ACLED API documentation: https://acleddata.com/api-documentation/getting-started, https://acleddata.com/api-documentation/acled-endpoint
- US State Department API: https://cadataapi.state.gov/api/TravelAdvisories (verified JSON endpoint, no auth required)
- UK FCDO GOV.UK Content API: https://content-api.publishing.service.gov.uk/reference.html (verified JSON endpoint at `https://www.gov.uk/api/content/foreign-travel-advice/{country}`)

### Secondary (MEDIUM confidence)
- ACLED codebook (event type definitions): https://acleddata.com/methodology/acled-codebook
- State Dept developer community: https://www.state.gov/developer/
- ACLED pagination behavior inferred from documentation mentioning 5000-row default limit

### Tertiary (LOW confidence)
- EU EEAS travel advisory feed existence: searched extensively, found no structured API. Confidence LOW that one exists.
- ACLED exact pagination parameter names: not explicitly documented in API reference pages fetched.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - all libraries already in use, no new dependencies
- Architecture: HIGH - all patterns follow existing codebase conventions
- External API integration (ACLED): MEDIUM - API docs verified but pagination details unclear
- External API integration (US State Dept): HIGH - JSON endpoint verified, no auth needed
- External API integration (UK FCDO): HIGH - GOV.UK Content API verified, structured JSON
- External API integration (EU EEAS): LOW - no structured API found
- SQLite migration: HIGH - straightforward ALTER TABLE + json_extract backfill
- Frontend wiring: HIGH - follows established Panel + RefreshScheduler patterns
- Pitfalls: HIGH - identified from direct codebase analysis

**Research date:** 2026-03-04
**Valid until:** 2026-04-04 (stable -- external APIs unlikely to change within 30 days)
