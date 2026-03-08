# Phase 24: Global Seeding & Globe Layers - Research

**Researched:** 2026-03-08
**Domain:** Geospatial risk computation, deck.gl layer data wiring, H3 hexagonal indexing
**Confidence:** HIGH

## Summary

This phase has two distinct deliverables: (1) a baseline risk computation engine that scores all ~195 countries from multi-source event data + travel advisories, and (2) wiring real data into the three currently no-op globe layers (heatmap, arcs, scenarios/risk-deltas).

The codebase is well-positioned for this work. The existing architecture provides: SQLite events table (1.43M GDELT events), PostgreSQL ORM + Alembic migrations, APScheduler job framework with heavy-job mutual exclusion, and a functional deck.gl DeckGLMap with all 5 layer slots already built (just lacking data for 3 of them). The frontend uses deck.gl 9.2.6 + maplibre-gl 5.16.0.

**Critical finding:** GDELT `country_iso` values in the events table are FIPS 10-4 codes, NOT ISO 3166-1 alpha-2. Examples: `UK` (should be `GB`), `IS` (should be `IL`), `NI` (should be `NG`), `AS` (should be `AU`). The FIPS-to-ISO mapping table is mandatory before any country-level aggregation will produce correct results. The advisory poller and ACLED poller already use proper ISO codes, but GDELT (1.43M events, the vast majority of data) does not.

**Primary recommendation:** The FIPS-to-ISO translation must be the first thing built and applied retroactively to all 1.43M existing GDELT events. Without it, baseline risk scores will be wrong, the choropleth will map to nonexistent countries, and arcs will be garbage.

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| h3 (Python) | 4.4.2 | Server-side H3 hex binning for heatmap aggregation | Uber's official H3 bindings; `h3.latlng_to_cell()` for point-to-hex, resolution-aware |
| @deck.gl/geo-layers | ^9.2.6 | H3HexagonLayer for frontend hex rendering | Official deck.gl geo layer, renders H3 hex indices directly |
| h3-js | ^4.1.0 | Frontend H3 dependency for @deck.gl/geo-layers | Required by H3HexagonLayer internally |
| SQLAlchemy 2.0 | existing | New PostgreSQL ORM models for baseline_country_risk, heatmap_hexbins, etc. | Already in use for all PostgreSQL tables |
| Alembic | existing | Migration 010 for 4 new tables | Already in use (9 migrations exist) |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| pypopulation | 7.0+ | Static population lookup by ISO alpha-2 for per-capita GDELT normalization | Claude's discretion item -- simple, zero-API-call library, JSON dict at import time |
| APScheduler 3.11.2 | existing | Hourly baseline risk recomputation job | Already registered and running |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| h3 (Python server-side) | Raw grid aggregation (0.5-degree) | H3 is uniform hexagonal (no lat distortion), but adds a ~6MB dependency. Worth it for visual quality. |
| pypopulation | Static dict in source | pypopulation is maintained and covers edge cases. A 200-line dict would work too but why maintain it. |
| @deck.gl/geo-layers H3HexagonLayer | HeatmapLayer (already imported) | Context decided on H3 hex binning. HeatmapLayer is simpler but produces blurry density blobs. H3 gives crisp hexagonal cells. |

### Installation

```bash
# Python (add to pyproject.toml core dependencies)
uv add h3

# Frontend (H3HexagonLayer requires @deck.gl/geo-layers + h3-js)
cd frontend && npm install @deck.gl/geo-layers h3-js
```

## Architecture Patterns

### Recommended Project Structure

```
src/
├── seeding/                    # NEW: baseline risk computation engine
│   ├── __init__.py
│   ├── baseline_risk.py        # Core computation: 4-input weighted score
│   ├── country_codes.py        # FIPS-to-ISO mapping + canonical country list
│   ├── population.py           # Per-capita normalization helper
│   ├── heatmap_binner.py       # H3 hex binning from SQLite events
│   ├── arc_builder.py          # Bilateral relationship extraction from KG
│   └── risk_delta.py           # 7-day risk change computation
├── api/
│   ├── routes/v1/
│   │   └── countries.py        # MODIFIED: dual-score model, new endpoints
│   ├── schemas/
│   │   └── country.py          # MODIFIED: add baseline/forecast/blended fields
│   └── ...
├── scheduler/
│   ├── job_wrappers.py         # MODIFIED: add heavy_baseline_risk wrapper
│   ├── registry.py             # MODIFIED: register baseline_risk hourly job
│   └── heavy_runner.py         # MODIFIED: add run_baseline_risk function
├── database/
│   ├── schema.sql              # MODIFIED: add lat, lon columns
│   └── models.py               # MODIFIED: add lat, lon to Event dataclass
├── db/
│   └── models.py               # MODIFIED: add 4 new PostgreSQL ORM models
├── ingest/
│   └── gdelt_poller.py         # MODIFIED: extract and store lat/lon from CSV
└── knowledge_graph/
    └── ...                     # READ ONLY: query graph edges for arc data
```

### Pattern 1: Pre-computed Layer Data (Write-Once, Read-Many)

**What:** All layer data (baseline risk, heatmap hexbins, arcs, risk deltas) is pre-computed by a single hourly heavy job and written to PostgreSQL tables. API endpoints simply read and return the latest computed data.

**When to use:** Always. The context document explicitly specifies this: "PostgreSQL tables for all pre-computed data", "always serve last computed data with a computed_at timestamp."

**Why:** The baseline risk computation requires scanning 1.4M+ SQLite events with time decay, querying advisory store, and aggregating per-country. Doing this on every API request would be catastrophic for latency. Pre-compute hourly, serve from PostgreSQL instantly.

```python
# The heavy job function structure (in heavy_runner.py)
def run_baseline_risk() -> int:
    """Compute all globe layer data in a single pass.

    1. Compute baseline_country_risk from GDELT + ACLED + advisories
    2. Compute heatmap_hexbins from geocoded events
    3. Compute country_arcs from knowledge graph edges
    4. Compute risk_deltas from baseline_risk history
    5. Write all to PostgreSQL in a single transaction
    """
    import asyncio
    asyncio.run(_compute_all_layers())
    return 0
```

### Pattern 2: FIPS-to-ISO Translation Layer

**What:** A static mapping dict that converts GDELT FIPS 10-4 country codes to ISO 3166-1 alpha-2 codes. Applied at two points: (1) retroactive migration of existing 1.43M events, (2) at ingestion time for all new GDELT events.

**When to use:** Every time a GDELT `country_iso` (actually FIPS) value is used.

```python
# src/seeding/country_codes.py
FIPS_TO_ISO: dict[str, str] = {
    "US": "US",  # Same in both systems
    "UK": "GB",  # United Kingdom
    "IS": "IL",  # Israel (NOT Iceland)
    "NI": "NG",  # Nigeria (NOT Nicaragua)
    "AS": "AU",  # Australia
    "CH": "CN",  # China
    "UP": "UA",  # Ukraine
    "RS": "RU",  # Russia
    "IN": "IN",  # India (same)
    "IR": "IR",  # Iran (same)
    "LE": "LB",  # Lebanon
    "EI": "IE",  # Ireland
    "GM": "DE",  # Germany
    "IZ": "IQ",  # Iraq
    "CA": "CA",  # Canada (same)
    "PK": "PK",  # Pakistan (same)
    "FR": "FR",  # France (same)
    # ... ~250 total entries
}

def fips_to_iso(fips_code: str) -> str | None:
    """Convert FIPS 10-4 code to ISO 3166-1 alpha-2.

    Returns None for unmapped codes (logged and skipped).
    """
    return FIPS_TO_ISO.get(fips_code.strip().upper())
```

### Pattern 3: Dual-Score API Response

**What:** The `GET /api/v1/countries` endpoint returns all ~195 countries with three risk fields: `baseline_risk`, `forecast_risk` (nullable), and `blended_risk`. Countries with active forecasts get the 70/30 blend; countries without show baseline only.

**When to use:** This replaces the current implementation which only returns countries with active predictions.

```python
# Schema change
class CountryRiskSummary(BaseModel):
    iso_code: str
    baseline_risk: float          # Always present (0-100)
    forecast_risk: float | None   # Only when active predictions exist
    blended_risk: float           # COALESCE(0.7*forecast + 0.3*baseline, baseline)
    risk_score: float             # Alias for blended_risk (backward compat)
    forecast_count: int
    top_forecast: str | None      # Nullable now (baseline-only countries have no forecast)
    top_probability: float | None
    trend: str
    last_updated: datetime
    disputed: bool = False        # For XK, TW, PS, EH
```

### Pattern 4: Heavy Job with Skip-if-Locked

**What:** The baseline risk job is a heavy job dispatched to ProcessPoolExecutor. It attempts to acquire the existing `_heavy_job_lock`. If the lock is held (daily pipeline, backtest, polymarket cycle are running), the job is skipped rather than queued. Next hour tries again.

**When to use:** This is the explicit decision from context: "Skip-if-locked: if the heavy job lock is held, skip this cycle."

```python
# In job_wrappers.py -- different from other heavy jobs
async def heavy_baseline_risk() -> None:
    """Compute baseline risk + layer data, skip if another heavy job is running."""
    if _heavy_job_lock.locked():
        logger.info("baseline_risk: skipping -- heavy job lock held")
        return  # Skip silently, next hour retries

    async with _heavy_job_lock:
        loop = asyncio.get_running_loop()
        returncode = await loop.run_in_executor(
            _get_process_executor(), run_baseline_risk
        )
        if returncode != 0:
            raise RuntimeError(f"baseline_risk exited with code {returncode}")
```

### Anti-Patterns to Avoid

- **Computing risk on API request:** The baseline risk computation scans 1.4M+ events. Never compute on demand. Always serve pre-computed data.
- **Storing FIPS codes and converting at read time:** Fix the data at the source. Storing FIPS and converting on read means every consumer must know about FIPS. Convert once in the migration, convert at ingestion going forward.
- **Separate jobs for each layer:** One job computes everything. Separate jobs would mean inconsistent timestamps between layers and more scheduling complexity.
- **Frontend H3 computation:** Server-side only. Sending 1.4M lat/lon pairs to the browser for client-side binning is insane. Pre-compute hex bins, send ~5K rows to the client.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| FIPS-to-ISO mapping | Manual 50-entry dict | Complete ~250-entry static dict from Wikipedia/CIA reference data | GDELT has ~260 unique FIPS codes. A partial mapping silently drops events for unmapped countries. |
| Population data | Scraped World Bank API | pypopulation library (static JSON, offline) | Zero API calls, no network dependency in the critical computation path. Updated annually. |
| H3 hex indexing | Custom grid binning (lat/lon buckets) | h3 library (`h3.latlng_to_cell()`) | Uniform hexagonal cells avoid latitude distortion. Industry standard from Uber. |
| Exponential time decay | Custom math per query | Reusable decay function with configurable half-life | Same pattern used in 3 places (baseline risk, heatmap weights, arc staleness). |

**Key insight:** The FIPS-to-ISO mapping is the single most important "don't hand-roll" item. Getting this wrong means ~50% of GDELT events map to the wrong country or no country at all. The existing country risk scores in production are currently wrong for any non-US country because `UK != GB`, `IS != IL`, etc.

## Common Pitfalls

### Pitfall 1: FIPS-to-ISO Code Collision

**What goes wrong:** FIPS `IS` = Israel, but ISO `IS` = Iceland. FIPS `AS` = Australia, but ISO `AS` = American Samoa. If you naively treat FIPS codes as ISO codes (which the current codebase does), Israel's 3,036 events show up under Iceland, Australia's 1,184 events show under American Samoa, etc.

**Why it happens:** GDELT inherited the CIA's FIPS 10-4 system (deprecated 2014). Many codes look like ISO but map to completely different countries. The GDELT poller stores `ActionGeo_CountryCode` directly without conversion.

**How to avoid:** Build and apply the FIPS-to-ISO mapping as the very first task. Run a retroactive UPDATE on all 1.43M existing events. Modify the GDELT poller to convert at ingestion time going forward.

**Warning signs:** Countries like Iceland, American Samoa, Nicaragua showing impossibly high event counts. Syria, Ukraine, Myanmar showing lower counts than expected.

### Pitfall 2: Lat/Lon Data Availability Gap

**What goes wrong:** The heatmap layer requires geocoded events with lat/lon. The existing 1.43M events have `raw_json = NULL` (the poller never stored it). `ActionGeo_Lat`/`ActionGeo_Long` are parsed from the CSV but discarded.

**Why it happens:** The `_gdelt_row_to_event()` function in `gdelt_poller.py` extracts country code but ignores lat/lon. The `Event` dataclass has no `lat`/`lon` fields. The SQLite schema has no `lat`/`lon` columns.

**How to avoid:** Add `lat`/`lon` columns to SQLite schema + Event dataclass. Modify `_gdelt_row_to_event()` to extract `ActionGeo_Lat`/`ActionGeo_Long`. Going forward, new events will have coordinates. For existing events, use country centroid coordinates (from the Natural Earth data already loaded via `countryGeometry`) as approximate hex assignment.

**Warning signs:** Heatmap layer shows no data despite 1.4M events existing. Only events ingested after the schema change appear.

### Pitfall 3: H3 Resolution Mismatch Between Server and Client

**What goes wrong:** If the server computes H3 indices at resolution X but the frontend expects resolution Y, the H3HexagonLayer silently renders nothing or renders malformed hexagons.

**Why it happens:** All hexagons in a single H3HexagonLayer MUST use the same resolution (deck.gl constraint). If the API mixes resolutions, rendering breaks.

**How to avoid:** Fix resolution at the server side. Store `h3_index` as a string column in `heatmap_hexbins`. The frontend reads and renders whatever the server provides. Recommended resolution: **3** (12,393 km^2 per hex, ~41,000 total hexagons on Earth -- reasonable for global-scale visualization, crisp enough for country-level detail).

**Warning signs:** H3HexagonLayer renders but hexagons are invisible or overlap weirdly.

### Pitfall 4: DeckGLMap Data Push API Mismatch

**What goes wrong:** The DeckGLMap class has no public methods for pushing heatmap data, arc data, or scenario zone data. It stores `heatData: HeatDatum[]`, `arcs: ArcDatum[]`, and `scenarioIsos: Set<string>` internally, but only `updateRiskScores()` and `updateForecasts()` are public. The heatmap/arc/scenario data is populated via internal methods only.

**Why it happens:** The DeckGLMap was built with the assumption that arc data comes from `buildArcsForCountry()` (internal, triggered by country selection), heatmap data is pushed externally (but no public method exists), and scenario data comes from `setSelectedForecast()` (entity ISOs from forecast scenarios).

**How to avoid:** Add public data-push methods: `updateHeatmapData(data: HeatDatum[])`, `updateArcData(data: ArcDatum[])`, `updateRiskDeltas(data: RiskDeltaDatum[])`. The globe screen wires these to API polling.

**Warning signs:** Layer pill bar shows "Heatmap" toggle but nothing renders when activated.

### Pitfall 5: Advisory Store In-Memory Cache

**What goes wrong:** The AdvisoryStore is an in-memory class-level cache (`_advisories: list[dict]`). The baseline risk job runs in a ProcessPoolExecutor worker (separate process). The worker can't read the main process's AdvisoryStore.

**Why it happens:** `ProcessPoolExecutor` forks a new process. Class-level state doesn't transfer. The advisory poller updates `AdvisoryStore._advisories` in the main process, but the heavy job worker sees an empty list.

**How to avoid:** The baseline risk job must fetch advisory data independently -- either query the advisory API endpoints directly, or persist advisories to a PostgreSQL table and read from there. Given the existing architecture, the cleanest approach is to call the State Dept and FCDO APIs directly within the heavy job (they're lightweight HTTP calls), or to persist the advisory cache to a `travel_advisories` table.

**Warning signs:** Baseline risk scores ignore advisory input entirely (advisory component always 0). Countries with Level 4 advisories (Syria, Afghanistan) show lower-than-expected risk.

## Code Examples

### H3 Hex Binning (Python Server-Side)

```python
# Source: h3 official docs + H3HexagonLayer API reference
import h3

HEATMAP_RESOLUTION = 3  # ~12,393 km^2 per hex

def bin_events_to_h3(
    events: list[dict],  # Must have 'lat', 'lon', 'goldstein_scale'
    resolution: int = HEATMAP_RESOLUTION,
    decay_days: int = 90,
) -> dict[str, float]:
    """Aggregate events into H3 hex bins with time-decayed weights.

    Returns dict of h3_index -> aggregated weight.
    """
    from datetime import datetime, timezone
    import math

    now = datetime.now(timezone.utc)
    bins: dict[str, float] = {}

    for event in events:
        lat, lon = event.get("lat"), event.get("lon")
        if lat is None or lon is None:
            continue

        h3_index = h3.latlng_to_cell(lat, lon, resolution)

        # Weight: severity * mention count * time decay
        severity = abs(event.get("goldstein_scale", 0)) / 10.0
        mentions = min(event.get("num_mentions", 1) or 1, 100) / 100.0

        event_date = datetime.fromisoformat(event["event_date"])
        age_days = (now - event_date).total_seconds() / 86400
        decay = math.exp(-0.693 * age_days / 30)  # 30-day half-life for heatmap

        weight = severity * mentions * decay
        bins[h3_index] = bins.get(h3_index, 0.0) + weight

    return bins
```

### H3HexagonLayer (TypeScript Frontend)

```typescript
// Source: deck.gl H3HexagonLayer API reference
import { H3HexagonLayer } from '@deck.gl/geo-layers';

interface HexBinDatum {
  h3_index: string;
  weight: number;
}

function buildHeatmapLayer(data: HexBinDatum[]): H3HexagonLayer<HexBinDatum> {
  return new H3HexagonLayer<HexBinDatum>({
    id: 'GDELTEventHeatmap',
    data,
    pickable: true,
    filled: true,
    extruded: false,
    getHexagon: (d) => d.h3_index,
    getFillColor: (d) => {
      const intensity = Math.min(1, d.weight / 50);
      // Orange-to-red gradient for event density
      return [
        255,
        Math.round(200 * (1 - intensity)),
        Math.round(50 * (1 - intensity)),
        Math.round(80 + 140 * intensity),
      ];
    },
    coverage: 0.9,
    highPrecision: 'auto',
  });
}
```

### Bilateral Arc Extraction from NetworkX Graph

```python
# Source: existing knowledge_graph/graph_builder.py edge structure
import networkx as nx
from collections import defaultdict

def extract_bilateral_arcs(
    graph: nx.MultiDiGraph,
    top_n: int = 20,
) -> list[dict]:
    """Extract top-N strongest bilateral country relationships.

    Aggregates all edges between country-entity node pairs,
    computing total event volume and average Goldstein score.
    """
    pair_stats: dict[tuple[str, str], dict] = defaultdict(
        lambda: {"count": 0, "goldstein_sum": 0.0}
    )

    for u, v, _key, data in graph.edges(data=True, keys=True):
        # Only country-level nodes (entity_type == 'country')
        u_type = graph.nodes.get(u, {}).get("entity_type")
        v_type = graph.nodes.get(v, {}).get("entity_type")
        if u_type != "country" or v_type != "country":
            continue

        # Canonical pair ordering (alphabetical) for aggregation
        pair = (min(u, v), max(u, v))
        goldstein = data.get("goldstein_scale", 0.0) or 0.0

        pair_stats[pair]["count"] += 1
        pair_stats[pair]["goldstein_sum"] += goldstein

    # Rank by event volume
    ranked = sorted(
        pair_stats.items(),
        key=lambda x: x[1]["count"],
        reverse=True,
    )[:top_n]

    return [
        {
            "source_iso": pair[0],
            "target_iso": pair[1],
            "event_count": stats["count"],
            "avg_goldstein": stats["goldstein_sum"] / max(stats["count"], 1),
        }
        for pair, stats in ranked
    ]
```

### Baseline Risk Computation

```python
# Core formula from context decisions
import math
from datetime import datetime, timezone

WEIGHTS = {
    "advisory": 0.35,
    "acled": 0.25,
    "gdelt": 0.25,
    "goldstein": 0.15,
}

ADVISORY_FLOORS = {4: 70.0, 3: 45.0}
DECAY_HALF_LIFE_DAYS = 30  # Within the 90-day window

def compute_baseline_risk(
    country_iso: str,
    gdelt_event_count: int,
    population: int,
    acled_fatalities: int,
    acled_event_count: int,
    advisory_level: int,  # 1-4
    avg_goldstein: float,  # -10 to +10
) -> float:
    """Compute 0-100 baseline risk score for a country.

    Components:
    1. GDELT density (per-capita, 0-100 normalized)
    2. ACLED intensity (fatality-weighted conflict, 0-100)
    3. Advisory level (1-4 mapped to 0-100, with hard floors)
    4. Goldstein severity (negative = conflict, 0-100)
    """
    # GDELT per-capita density (events per million population)
    pop_millions = max(population, 1) / 1_000_000
    gdelt_per_capita = gdelt_event_count / pop_millions
    gdelt_score = min(100.0, gdelt_per_capita * 2.0)  # Tunable scaling

    # ACLED intensity (fatalities + event count)
    acled_score = min(100.0, (acled_fatalities * 5.0 + acled_event_count * 2.0))

    # Advisory level
    advisory_map = {1: 10.0, 2: 35.0, 3: 60.0, 4: 90.0}
    advisory_score = advisory_map.get(advisory_level, 10.0)

    # Goldstein severity (flip sign: negative = more severe)
    goldstein_score = max(0.0, min(100.0, (10.0 - avg_goldstein) * 5.0))

    # Weighted composite
    composite = (
        WEIGHTS["gdelt"] * gdelt_score +
        WEIGHTS["acled"] * acled_score +
        WEIGHTS["advisory"] * advisory_score +
        WEIGHTS["goldstein"] * goldstein_score
    )

    # Apply advisory hard floors
    floor = ADVISORY_FLOORS.get(advisory_level, 0.0)
    composite = max(composite, floor)

    return round(min(100.0, max(0.0, composite)), 1)
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| FIPS 10-4 country codes in GDELT | Must convert to ISO 3166-1 alpha-2 | This phase | All country-level aggregation currently wrong for non-US countries |
| country_iso stored as FIPS | country_iso stored as ISO | This phase | Retroactive UPDATE on 1.43M events + ingestion-time conversion |
| Countries endpoint: only forecast-active | All ~195 countries with baseline risk | This phase | Globe goes from ~8-15 colored countries to 195 |
| HeatmapLayer with no data | H3HexagonLayer with pre-computed hex bins | This phase | `@deck.gl/geo-layers` added, `HeatmapLayer` import remains but layer implementation changes |
| Arcs from client-side scenario entities | Arcs from server-side KG bilateral edges | This phase | Real geopolitical relationship data instead of forecast-entity adjacency |
| ScenarioZones highlighting forecast entities | Risk delta visualization (7-day change) | This phase | Analytical value: "what changed" overlay |

**Deprecated/outdated:**
- The current `_COUNTRY_RISK_SQL` CTE in `countries.py` is forecast-only. It will be replaced or augmented with a query that merges `baseline_country_risk` with the prediction-derived scores.
- The `HeatmapLayer` import from `@deck.gl/aggregation-layers` remains valid but the layer implementation switches to `H3HexagonLayer` from `@deck.gl/geo-layers`. Both can coexist.
- The internal `buildArcsForCountry()` method in DeckGLMap uses forecast marker ISOs as arc targets. This will be replaced with server-provided bilateral relationship data.

## Open Questions

### 1. ACLED Lat/Lon Availability

**What we know:** ACLED events have `latitude` and `longitude` fields in their API response. The current `_acled_to_event()` mapping discards them (same problem as GDELT).

**What's unclear:** Whether the ACLED poller is actually running (0 ACLED events in the database currently -- likely need API credentials). If ACLED events start flowing, they'll also need lat/lon extraction.

**Recommendation:** Add lat/lon to the Event dataclass and extract from ACLED responses too. Even if ACLED isn't running yet, the schema should support it.

### 2. Knowledge Graph Node Types for Arc Data

**What we know:** The KG uses entity normalization (`EntityNormalizer`) that maps actor codes to entities. Entities have an `entity_type` field. Bilateral country arcs require filtering to `entity_type == "country"` nodes.

**What's unclear:** How reliably GDELT actor codes map to country entities. If most nodes are organizations or individuals rather than countries, the arc data may be sparse.

**Recommendation:** The planner should include a verification step that queries the graph for country-type node count before building the arc API. If insufficient, fall back to aggregating by `country_iso` from the events table directly (same-event bilateral: actor1_country + actor2_country pairs).

### 3. pypopulation Data Currency

**What we know:** pypopulation bundles World Bank 2020 data. Country populations are stable enough that 2020 data is adequate for per-capita normalization (population changes <3% per year for most countries).

**What's unclear:** Whether pypopulation covers all disputed territories (XK, TW, PS, EH) that the context document requires.

**Recommendation:** Use pypopulation as primary source. For any missing countries, hard-code population estimates from UN data. Test coverage during implementation.

### 4. Advisory Data Cross-Process Access

**What we know:** Advisory data lives in `AdvisoryStore._advisories` (class-level in-memory cache in the main process). The baseline risk job runs in a ProcessPoolExecutor worker (separate process). The worker cannot access the main process's memory.

**What's unclear:** Whether we should persist advisories to PostgreSQL, re-fetch from APIs in the worker, or use shared memory.

**Recommendation:** Persist advisory data to a `travel_advisories` PostgreSQL table (a simple insert-on-update pattern). The advisory poller already runs IngestRun audit rows via PostgreSQL -- adding a data table is consistent. The baseline risk worker reads from this table. This also provides advisory history for the risk delta computation.

## Sources

### Primary (HIGH confidence)
- Codebase analysis: `src/ingest/gdelt_poller.py` -- confirmed FIPS codes stored as country_iso
- Codebase analysis: `src/database/schema.sql` -- confirmed no lat/lon columns
- Codebase analysis: `frontend/src/components/DeckGLMap.ts` -- confirmed layer data interfaces and empty data stores
- SQLite query: events table -- 1,429,572 GDELT events, FIPS codes verified (UK=3401, IS=3036, NI=1618)
- deck.gl docs: H3HexagonLayer API reference at https://deck.gl/docs/api-reference/geo-layers/h3-hexagon-layer
- H3 docs: Resolution table at https://h3geo.org/docs/core-library/restable/

### Secondary (MEDIUM confidence)
- h3 PyPI: version 4.4.2, Python 3.12-3.14 wheels available
- @deck.gl/geo-layers npm: version 9.2.2 (compatible with existing 9.2.6)
- FIPS-to-ISO mapping: Wikipedia List of FIPS country codes (cross-referenced with actual DB values)
- pypopulation PyPI: World Bank 2020 population data bundled as JSON

### Tertiary (LOW confidence)
- pypopulation coverage of disputed territories -- needs runtime verification
- ACLED API lat/lon field availability -- needs verification when credentials are configured

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- all libraries verified via official docs and npm/PyPI
- Architecture: HIGH -- patterns follow existing codebase conventions exactly
- FIPS-to-ISO issue: HIGH -- verified empirically against 1.43M events in production DB
- Pitfalls: HIGH -- all derived from direct codebase analysis
- H3 resolution choice: MEDIUM -- theoretical (12,393 km^2/hex) but needs visual testing

**Research date:** 2026-03-08
**Valid until:** 2026-04-08 (stable domain, no fast-moving dependencies)
