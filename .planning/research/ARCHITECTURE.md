# Architecture Patterns: v3.0 Integration

**Project:** Geopol v3.0 -- Operational Command & Verification
**Researched:** 2026-03-04
**Mode:** Integration architecture for 7 new feature areas into existing system
**Confidence:** HIGH (based on direct codebase analysis of existing components + documented patterns)

## Executive Summary

v3.0 adds 7 capabilities to a well-structured FastAPI + vanilla TS system. The existing architecture is clean -- clear component boundaries, consistent patterns (IngestRun audit trail, async pollers, Panel-based UI). The integration challenge is consolidation (4 scattered daemons -> 1 unified scheduler) and data plumbing (3 globe layers with empty data arrays, global seeding with no pipeline).

The architecture below maps each v3.0 feature to specific integration points, new components, and data flow changes. Build order is derived from dependency analysis, not arbitrary grouping.

---

## 1. Current Architecture Inventory

### Backend Process Topology (as-is)

```
Process 1: uvicorn (FastAPI)
  |- src/api/app.py          -- HTTP routes, lifespan manages DB/Redis/Polymarket loop
  |- _polymarket_loop()      -- asyncio.Task inside lifespan (not APScheduler)
  |- PostgreSQL connection   -- async SQLAlchemy (asyncpg)
  |- Redis connection        -- forecast response caching

Process 2: scripts/gdelt_poller.py (systemd: geopol-gdelt-poller.service)
  |- GDELTPoller.run()       -- asyncio event loop, SIGTERM shutdown
  |- SQLite writes           -- EventStorage.insert_events()
  |- PostgreSQL writes       -- IngestRun audit

Process 3: scripts/rss_daemon.py (systemd: geopol-rss-daemon.service)
  |- RSSDaemon.start()       -- tiered polling (15min/60min)
  |- ChromaDB writes         -- article indexing
  |- PostgreSQL writes       -- IngestRun audit

Process 4: scripts/daily_forecast.py (systemd timer: 06:00 UTC)
  |- DailyPipeline.run_with_retry()  -- one-shot, 4-phase pipeline
  |- Gemini API calls        -- question generation + forecasting
  |- PostgreSQL writes       -- Prediction rows

Process 5: scripts/acled_poller.py (manual / cron -- no systemd unit in deploy/)
  |- ACLEDPoller             -- daily ACLED event fetch

Process 6: scripts/advisory_poller.py (manual / cron)
  |- AdvisoryPoller          -- daily government advisory fetch
```

### Frontend Route Topology (as-is)

```
main.ts
  |- Router (pushState)
  |- /dashboard  -- mountDashboard(container, ctx)
  |- /globe      -- mountGlobe(container, ctx)  [dynamic import: deck.gl]
  |- /forecasts  -- mountForecasts(container, ctx)
  |- NavBar      -- 3 route links
```

### Data Store Topology (as-is)

```
SQLite (data/events.db):    GDELT events + ACLED events (source discriminator)
PostgreSQL:                 predictions, outcomes, calibration, api_keys,
                            ingest_runs, pending_questions, forecast_requests,
                            polymarket_comparisons, polymarket_snapshots
ChromaDB (chroma_db/):      RSS article chunks (semantic search)
Redis (Upstash):            Forecast response cache (SUMMARY_TTL)
```

---

## 2. Integration Architecture by Feature

### 2.1 Admin Route (`/admin`)

**Decision (from MEMORY.md):** Same app, dynamic import code-split, route-level auth gating.

**Frontend Integration:**

The existing Router is a clean pushState implementation. Adding `/admin` requires:

1. **Route registration in `main.ts`** -- add a 4th route with dynamic import:

```typescript
// Lazy-load admin screen (code-split chunk)
router.addRoute({
  path: '/admin',
  mount: async (container) => {
    const { mountAdmin, unmountAdmin: ua } = await import('@/screens/admin-screen');
    adminUnmount = ua;
    await mountAdmin(container, ctx);
  },
  unmount: () => { if (adminUnmount) adminUnmount(ctx); },
});
```

2. **Auth gating pattern** -- NOT a router-level middleware (the Router class has no middleware concept). Two approaches:

   **Option A: Guard inside mount function** (recommended)
   ```typescript
   export async function mountAdmin(container: HTMLElement, ctx: GeoPolAppContext) {
     const authed = await forecastClient.checkAdminAuth();
     if (!authed) {
       container.innerHTML = '<div class="admin-denied">Unauthorized</div>';
       return;
     }
     // ... mount admin panels
   }
   ```

   **Option B: Router-level guard** (requires Router modification)
   ```typescript
   router.addRoute({
     path: '/admin',
     guard: () => forecastClient.checkAdminAuth(),  // New Router feature
     mount: ...,
     unmount: ...,
   });
   ```

   **Recommendation:** Option A. The Router is intentionally minimal. Adding a guard abstraction for exactly one route is over-engineering. The mount function already handles async initialization -- an auth check is just another async step.

3. **NavBar modification** -- The admin route should NOT appear in the main NavBar for public users. Either:
   - Conditionally render admin link based on auth state
   - Keep admin link always present but route guard handles unauthorized access (simpler, no conditional rendering)
   - Access via direct URL only (no nav link) -- most appropriate for single-operator system

4. **Code-splitting impact** -- Admin screen imports should be isolated from the public bundle:
   - Admin panels: `frontend/src/screens/admin-screen.ts` (new)
   - Admin components: `frontend/src/components/admin/` directory (new)
   - Zero bytes in dashboard/globe/forecasts chunks

**Backend integration:**

Admin auth is simple for a single-operator system. Options:
- Reuse existing API key with an `is_admin` flag on the `api_keys` table
- Separate admin key in settings (`ADMIN_API_KEY` env var)
- **Recommendation:** Add `is_admin: bool = False` column to `api_keys` table. Frontend stores admin key in localStorage. All admin API routes check `verify_admin_key` dependency.

**New API routes:**
```
POST /api/v1/admin/auth/verify     -- verify admin credentials
GET  /api/v1/admin/scheduler/jobs  -- list all scheduled jobs
POST /api/v1/admin/scheduler/jobs/{id}/pause
POST /api/v1/admin/scheduler/jobs/{id}/resume
POST /api/v1/admin/scheduler/jobs/{id}/trigger
GET  /api/v1/admin/sources         -- enhanced source management
POST /api/v1/admin/sources/feeds   -- add/remove/edit RSS feeds
GET  /api/v1/admin/backtesting     -- backtesting results
```

**New files:**
- `src/api/routes/v1/admin.py` -- admin router with auth dependency
- `src/api/middleware/admin_auth.py` -- admin key verification
- `frontend/src/screens/admin-screen.ts`
- `frontend/src/components/admin/SchedulerPanel.ts`
- `frontend/src/components/admin/SourceManagerPanel.ts`
- `frontend/src/components/admin/BacktestingPanel.ts`

---

### 2.2 APScheduler Daemon Consolidation

**Decision (from MEMORY.md):** APScheduler AsyncIOScheduler, single process in-process with FastAPI, pause/resume/trigger via admin API.

**Current state:** 4-6 separate processes, each with their own event loop, signal handling, and backoff logic. The Polymarket loop is already in-process with FastAPI (as an `asyncio.Task` in lifespan).

**Target state:** One `AsyncIOScheduler` instance created in FastAPI lifespan, all jobs registered as APScheduler jobs.

**Integration into `src/api/app.py` lifespan:**

```python
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.triggers.cron import CronTrigger

@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    # ... existing DB/Redis init ...

    # APScheduler with in-memory job store
    scheduler = AsyncIOScheduler(
        job_defaults={
            'coalesce': True,       # Collapse missed runs into one
            'max_instances': 1,     # Never overlap same job
            'misfire_grace_time': 300,  # 5 min grace for late runs
        },
    )

    # Register jobs
    scheduler.add_job(gdelt_poll_job, IntervalTrigger(seconds=settings.gdelt_poll_interval),
                      id='gdelt_poller', name='GDELT Event Ingestion', replace_existing=True)
    scheduler.add_job(rss_tier1_job, IntervalTrigger(seconds=settings.rss_poll_interval_tier1),
                      id='rss_tier1', name='RSS Tier-1 Feeds', replace_existing=True)
    scheduler.add_job(rss_tier2_job, IntervalTrigger(seconds=settings.rss_poll_interval_tier2),
                      id='rss_tier2', name='RSS Tier-2 Feeds', replace_existing=True)
    scheduler.add_job(acled_poll_job, IntervalTrigger(seconds=settings.acled_poll_interval),
                      id='acled_poller', name='ACLED Conflict Events', replace_existing=True)
    scheduler.add_job(advisory_poll_job, IntervalTrigger(seconds=settings.advisory_poll_interval),
                      id='advisory_poller', name='Government Advisories', replace_existing=True)
    scheduler.add_job(daily_pipeline_job, CronTrigger(hour=6, minute=0),
                      id='daily_pipeline', name='Daily Forecast Pipeline', replace_existing=True)
    scheduler.add_job(polymarket_job, IntervalTrigger(seconds=settings.polymarket_poll_interval),
                      id='polymarket', name='Polymarket Matching', replace_existing=True)

    scheduler.start()
    app.state.scheduler = scheduler

    yield

    scheduler.shutdown(wait=True)
    # ... existing cleanup ...
```

**Job store decision: PostgreSQL vs in-memory.**

- PostgreSQL job store (SQLAlchemyJobStore): Persists job state across restarts. Knows which jobs missed their window. Adds dependency on sync psycopg2 driver for the jobstore (APScheduler's SQLAlchemy jobstore doesn't support async).
- In-memory (MemoryJobStore): Simpler. Jobs are re-registered on every startup. No missed-run recovery.
- **Recommendation:** In-memory. The jobs are statically defined in code. There's no dynamic job creation from users. `coalesce=True` handles the "app was down" case by running immediately on startup. PostgreSQL jobstore adds a sync driver dependency for marginal benefit.

**Refactoring existing pollers:**

The existing `GDELTPoller`, `RSSDaemon`, `ACLEDPoller` classes manage their own event loops, signal handlers, and backoff. Under APScheduler, each job function wraps the single-cycle logic:

```python
# New: src/scheduler/jobs.py
async def gdelt_poll_job():
    """Single GDELT poll cycle. APScheduler manages interval/retry."""
    poller = _get_gdelt_poller()  # Cached singleton
    await poller._poll_once()

async def rss_tier1_job():
    """Single RSS tier-1 poll cycle."""
    daemon = _get_rss_daemon()
    await daemon.poll_feeds(TIER_1_FEEDS)

async def daily_pipeline_job():
    """Full daily pipeline run."""
    # Reuse existing DailyPipeline but without the retry wrapper
    # (APScheduler handles misfire/retry)
    ...
```

The key insight: existing pollers already separate "single cycle" (`_poll_once`, `poll_feeds`) from "loop management" (`run`, `start`). APScheduler replaces the loop management layer. The single-cycle functions stay intact.

**Signal handling removal:** APScheduler handles graceful shutdown via `scheduler.shutdown(wait=True)`. Remove SIGTERM/SIGINT handlers from individual pollers.

**Backoff replacement:** APScheduler has built-in `misfire_grace_time` and `coalesce`. For transient failures within a job, use try/except with logging (not backoff delay -- the next scheduled run will retry).

**Admin API exposure:**

```python
# src/api/routes/v1/admin.py
@router.get("/scheduler/jobs")
async def list_jobs(request: Request):
    scheduler: AsyncIOScheduler = request.app.state.scheduler
    return [
        {
            "id": job.id,
            "name": job.name,
            "next_run": job.next_run_time.isoformat() if job.next_run_time else None,
            "paused": job.next_run_time is None,
            "trigger": str(job.trigger),
        }
        for job in scheduler.get_jobs()
    ]

@router.post("/scheduler/jobs/{job_id}/pause")
async def pause_job(job_id: str, request: Request):
    scheduler: AsyncIOScheduler = request.app.state.scheduler
    scheduler.pause_job(job_id)

@router.post("/scheduler/jobs/{job_id}/resume")
async def resume_job(job_id: str, request: Request):
    scheduler: AsyncIOScheduler = request.app.state.scheduler
    scheduler.resume_job(job_id)

@router.post("/scheduler/jobs/{job_id}/trigger")
async def trigger_job(job_id: str, request: Request):
    scheduler: AsyncIOScheduler = request.app.state.scheduler
    scheduler.modify_job(job_id, next_run_time=datetime.now(timezone.utc))
```

**New files:**
- `src/scheduler/__init__.py`
- `src/scheduler/jobs.py` -- job functions wrapping existing single-cycle logic
- `src/scheduler/config.py` -- job registration definitions

**Modified files:**
- `src/api/app.py` -- lifespan: add scheduler init, remove `_polymarket_loop`
- `src/ingest/gdelt_poller.py` -- extract `_poll_once` as standalone, remove `run()` loop
- `src/ingest/rss_daemon.py` -- extract `poll_feeds` tier calls, remove `start()` loop

**Systemd impact:** `geopol-gdelt-poller.service` and `geopol-rss-daemon.service` become obsolete. Only `geopol-api.service` runs. The daily timer also becomes obsolete (APScheduler CronTrigger replaces it).

---

### 2.3 ICEWS + UCDP Data Integration

**Data source characteristics:**

| Source | Format | Access | Update Frequency | Event Schema |
|--------|--------|--------|------------------|-------------|
| ICEWS | TSV (tab-separated) | Harvard Dataverse API (token required) | Weekly since 2020 | Actor1, Actor2, EventCode (CAMEO), Date, Location |
| UCDP GED | CSV | RESTful API (token since Feb 2026) or bulk download | Annual + quarterly candidate events | Actor1, Actor2, EventType, Date, Lat/Lon, Country |
| Existing GDELT | CSV.zip (GDELT v2) | Public HTTP | Every 15 minutes | CAMEO coded, 61 columns |
| Existing ACLED | JSON API | Authenticated (email+key) | Daily | Event type, Actor1/2, Country, Lat/Lon |

**SQLite schema integration:**

Current `events` table schema already has a `source` discriminator field (`"gdelt"` or `"acled"`). ICEWS and UCDP events map to the same Event dataclass:

```python
# ICEWS events map directly -- ICEWS uses CAMEO coding system (same as GDELT)
Event(
    source="icews",
    gdelt_id=None,                    # ICEWS has its own event ID
    content_hash=sha256(f"{icews_event_id}"),
    event_date="2026-03-01",
    actor1_code="USA",
    actor2_code="CHN",
    event_code="042",                 # CAMEO code (same taxonomy as GDELT)
    quad_class=2,                     # Verbal conflict
    goldstein_scale=-4.0,
    country_iso="CN",
)

# UCDP events need CAMEO mapping (UCDP uses own event type taxonomy)
Event(
    source="ucdp",
    gdelt_id=None,
    content_hash=sha256(f"{ucdp_event_id}"),
    event_date="2026-02-15",
    actor1_code="GOV:SYR",           # Normalize to GDELT-style actor codes
    actor2_code="REB:SYR",
    event_code="19",                  # UCDP "Battle" -> CAMEO 19 (fight)
    quad_class=4,                     # Material conflict
    goldstein_scale=-10.0,
    country_iso="SY",
)
```

**Schema changes needed:** None for `events` table (source discriminator already supports it). Need to add:

1. **Content deduplication across sources**: GDELT event about "Syria battle on 2026-02-15" and UCDP event about the same battle should not create duplicate knowledge graph edges. The existing `content_hash` + `time_window` dedup only works within a source.

   **Approach:** Cross-source dedup via (actor1, actor2, event_date, country_iso) fuzzy matching at graph insertion time (not at event insertion -- keep raw events from all sources for audit).

2. **Source metadata table** (new PostgreSQL table):
   ```sql
   CREATE TABLE data_sources (
       id SERIAL PRIMARY KEY,
       name VARCHAR(20) UNIQUE NOT NULL,     -- 'gdelt', 'acled', 'icews', 'ucdp'
       enabled BOOLEAN DEFAULT TRUE,
       poll_interval_seconds INTEGER,
       last_poll_at TIMESTAMPTZ,
       last_success_at TIMESTAMPTZ,
       last_error TEXT,
       config_json JSONB DEFAULT '{}'::jsonb  -- source-specific config
   );
   ```

**New poller classes:**

```python
# src/ingest/icews_poller.py
class ICEWSPoller:
    """Weekly ICEWS event fetch from Harvard Dataverse.

    ICEWS provides weekly TSV dumps via Dataverse API.
    Uses CAMEO coding (same as GDELT) -- direct Event mapping.
    """
    async def poll_once(self) -> CycleMetrics:
        # 1. Check Dataverse for new weekly file
        # 2. Download TSV, parse rows
        # 3. Map to Event(source="icews", ...)
        # 4. Insert via EventStorage
        # 5. Record IngestRun

# src/ingest/ucdp_poller.py
class UCDPPoller:
    """Periodic UCDP GED event fetch.

    UCDP uses own event taxonomy, requires mapping to CAMEO.
    API requires auth token (introduced Feb 2026).
    Quarterly candidate events + annual finalized.
    """
    UCDP_TO_CAMEO = {
        1: "19",  # State-based violence -> CAMEO 19 (fight)
        2: "20",  # Non-state violence -> CAMEO 20 (unconventional mass violence)
        3: "18",  # One-sided violence -> CAMEO 18 (assault)
    }
```

**Entity resolution across sources:**

This is the hard problem. GDELT uses FIPS/ISO country codes and actor type codes. ACLED uses full actor names. ICEWS uses CAMEO actor codes. UCDP uses actor names with country identifiers.

**Approach for v3.0:** Country-level normalization only (all sources -> ISO alpha-2). Sub-national actor resolution is out of scope -- it requires a dedicated entity resolution pipeline.

```python
# src/ingest/entity_resolver.py
class CountryResolver:
    """Normalize country identifiers across GDELT/ACLED/ICEWS/UCDP to ISO alpha-2.

    Already partially implemented:
    - GDELT: ActionGeo_CountryCode (FIPS -> ISO via lookup)
    - ACLED: iso3 -> ISO2_TO_ISO2 mapping (in acled_poller.py)
    - ICEWS: Country field (full name -> ISO lookup)
    - UCDP: country_id (numeric) + country (name) -> ISO lookup
    """
```

**New files:**
- `src/ingest/icews_poller.py`
- `src/ingest/ucdp_poller.py`
- `src/ingest/entity_resolver.py` -- country normalization shared across sources

**Modified files:**
- `src/database/storage.py` -- add `query_events` source filter for `icews`, `ucdp`
- `src/api/routes/v1/sources.py` -- add `icews`, `ucdp` to `_KNOWN_DAEMON_TYPES`
- `src/settings.py` -- add ICEWS/UCDP credentials and poll intervals
- `src/scheduler/jobs.py` -- register ICEWS and UCDP poll jobs

---

### 2.4 Backtesting Architecture

**Decision (from MEMORY.md):** Isolated internal reporting system only. Walk-forward eval, model comparison (TiRGN vs RE-GCN), calibration audit.

**Architectural position:** Backtesting is a READ-ONLY analysis pipeline that queries historical predictions and outcomes. It does NOT interfere with live predictions.

**Data flow:**

```
PostgreSQL (read-only)
  |- predictions table (historical forecasts)
  |- outcome_records table (ground truth resolutions)
  |- calibration_weight_history table (weight audit trail)
  |- polymarket_snapshots table (Polymarket vs Geopol time series)
       |
       v
BacktestEngine (new)
  |- Walk-forward evaluation: sliding window, retrain-predict-evaluate
  |- Model comparison: TiRGN vs RE-GCN MRR/Hits@k on same windows
  |- Calibration audit: reliability diagrams over time buckets
  |- Polymarket accuracy: cumulative Brier score comparison curves
       |
       v
BacktestReport (stored in PostgreSQL)
  |- JSON blobs for each analysis type
  |- Served via /api/v1/admin/backtesting endpoints
```

**Why separate from production pipeline:**

The daily pipeline (`DailyPipeline`) creates forecasts. Backtesting analyzes historical forecasts. They share some components (BrierScorer, CalibrationMetrics) but have different lifecycles. Backtesting is triggered manually or on schedule (weekly), not per-forecast.

**Integration pattern:**

```python
# src/evaluation/backtest_engine.py
class BacktestEngine:
    """Walk-forward evaluation of forecast quality.

    Operates on historical data only. Does NOT create new predictions.
    """
    def __init__(self, async_session_factory):
        self.session_factory = async_session_factory

    async def run_walk_forward(
        self,
        window_days: int = 30,
        step_days: int = 7,
        start_date: date | None = None,
    ) -> WalkForwardReport:
        """Evaluate forecast accuracy using sliding windows.

        For each window:
        1. Collect predictions created in [start, start + window_days)
        2. Match with outcomes resolved by end of window
        3. Compute Brier score, calibration curve
        4. Step forward by step_days
        """

    async def compare_models(self) -> ModelComparisonReport:
        """Compare TiRGN vs RE-GCN across evaluation windows."""

    async def calibration_audit(self) -> CalibrationAuditReport:
        """Generate reliability diagrams by time period."""

    async def polymarket_accuracy(self) -> PolymarketAccuracyReport:
        """Cumulative Brier score: Geopol vs Polymarket."""
```

**APScheduler integration:**
```python
scheduler.add_job(
    backtest_job,
    CronTrigger(day_of_week='sun', hour=4),  # Weekly Sunday 4 AM
    id='backtesting',
    name='Weekly Backtest Report',
)
```

**New files:**
- `src/evaluation/backtest_engine.py` -- core analysis logic
- `src/evaluation/reports.py` -- report dataclasses
- `frontend/src/components/admin/BacktestingPanel.ts` -- results visualization

**New PostgreSQL table:**
```sql
CREATE TABLE backtest_reports (
    id SERIAL PRIMARY KEY,
    report_type VARCHAR(30) NOT NULL,  -- 'walk_forward', 'model_comparison', 'calibration', 'polymarket'
    created_at TIMESTAMPTZ DEFAULT NOW(),
    report_json JSONB NOT NULL,
    window_start DATE,
    window_end DATE
);
```

---

### 2.5 Global Seeding Pipeline

**Decision (from MEMORY.md):** All ~195 countries get baseline risk from GDELT event density + ACLED + ICEWS + UCDP + advisories. Active forecasts override base score.

**Current state:** Country risk comes from the `_COUNTRY_RISK_SQL` CTE in `src/api/routes/v1/countries.py`. It only produces risk scores for countries that have active predictions. Countries without forecasts show no risk data.

**Target state:** Every country has a risk score. Two sources:
1. **Forecast-derived risk** (existing) -- countries with active predictions
2. **Event-density baseline** (new) -- all countries based on event volume from all sources

**Architecture:**

```
Event Sources (SQLite events table)
  |- GDELT events (every 15 min)
  |- ACLED events (daily)
  |- ICEWS events (weekly)
  |- UCDP events (quarterly)
       |
       v
BaselineRiskComputer (new, scheduled job)
  |- Count events per country per 30-day window
  |- Weight by source reliability + event severity (Goldstein/QuadClass)
  |- Apply time decay (7-day half-life, matching forecast decay)
  |- Normalize to 0-100 scale
  |- Store in PostgreSQL baseline_risk table
       |
       v
Country Risk API (modified)
  |- Merge forecast-derived risk with baseline risk
  |- Active forecasts OVERRIDE baseline (not additive)
  |- Result: all ~195 countries have risk scores
```

**New PostgreSQL table:**
```sql
CREATE TABLE baseline_country_risk (
    country_iso VARCHAR(2) PRIMARY KEY,
    risk_score FLOAT NOT NULL,           -- 0-100
    event_count INTEGER NOT NULL,
    avg_severity FLOAT,                  -- Average Goldstein magnitude
    dominant_source VARCHAR(20),         -- Which source contributes most events
    advisory_level VARCHAR(20),          -- e.g., 'Do Not Travel', 'Exercise Caution'
    computed_at TIMESTAMPTZ NOT NULL
);
```

**Modified SQL in countries.py:**

The current `_COUNTRY_RISK_SQL` CTE queries only the `predictions` table. The modified version unions forecast-derived risk with baseline risk, using `COALESCE` to prefer forecast risk when available:

```sql
WITH forecast_risk AS (
    -- existing CTE (unchanged)
),
baseline AS (
    SELECT country_iso, risk_score AS baseline_score
    FROM baseline_country_risk
    WHERE computed_at > NOW() - INTERVAL '48 hours'
),
merged AS (
    SELECT
        COALESCE(f.country_iso, b.country_iso) AS country_iso,
        COALESCE(f.risk_score, b.baseline_score) AS risk_score,
        -- forecast fields (null for baseline-only countries)
        f.forecast_count,
        f.avg_probability,
        f.top_question
    FROM baseline b
    FULL OUTER JOIN forecast_risk f ON f.country_iso = b.country_iso
)
SELECT * FROM merged ORDER BY risk_score DESC;
```

**Advisory integration:**

Government travel advisories (already polled by `advisory_poller.py`, stored in `src/ingest/advisory_store.py`) provide a severity signal. Map advisory levels to a risk floor:

| Advisory Level | Risk Floor |
|---------------|------------|
| Do Not Travel | 70 |
| Reconsider Travel | 50 |
| Exercise Increased Caution | 30 |
| Exercise Normal Precautions | 10 |

The baseline risk score is `max(event_density_score, advisory_floor)`.

**APScheduler integration:**
```python
scheduler.add_job(
    global_seeding_job,
    CronTrigger(hour='*/6'),  # Every 6 hours
    id='global_seeding',
    name='Baseline Country Risk Computation',
)
```

**New files:**
- `src/evaluation/baseline_risk.py` -- BaselineRiskComputer
- Alembic migration for `baseline_country_risk` table

**Modified files:**
- `src/api/routes/v1/countries.py` -- merge baseline with forecast risk
- `src/scheduler/jobs.py` -- register global_seeding_job

---

### 2.6 Globe Layer Data Wiring

**Current state (from DeckGLMap.ts):**

The DeckGLMap has 5 layers. Only 2 receive data:

| Layer | Data Source | Status |
|-------|-----------|--------|
| ForecastRiskChoropleth | `updateRiskScores(CountryRiskSummary[])` | WIRED -- globe-screen pushes countries |
| ActiveForecastMarkers | `updateForecasts(ForecastResponse[])` | WIRED -- globe-screen pushes forecasts |
| KnowledgeGraphArcs | `buildArcsForCountry()` | PARTIAL -- builds from markers + scenarioIsos, but data is sparse |
| GDELTEventHeatmap | `this.heatData: HeatDatum[]` | NO-OP -- array is never populated |
| ScenarioZones | `this.scenarioIsos: Set<string>` | PARTIAL -- only populated when a forecast is selected |

**Data wiring needed:**

**Layer 3: KnowledgeGraphArcs** -- Currently derives arcs from forecast markers and scenario entities. This is fundamentally correct but the data is too sparse (only works when both the selected country AND other countries have forecasts). The fix: populate arcs from the knowledge graph directly.

New API endpoint:
```
GET /api/v1/countries/{iso}/relations
Response: { relations: [{ target_iso, relation_type, event_count, goldstein_avg }] }
```

Backend query: `SELECT DISTINCT country_iso_target, event_code, COUNT(*), AVG(goldstein_scale) FROM cross_country_events WHERE country_iso_source = ? AND event_date > ? GROUP BY country_iso_target, event_code` (requires cross-referencing actor country codes in the events table).

**Layer 4: GDELTEventHeatmap** -- Needs a new API endpoint serving geo-located events.

New API endpoint:
```
GET /api/v1/events/heatmap?days=7
Response: { points: [{ lat, lon, weight }] }
```

Backend: Query SQLite events table for events with lat/lon coordinates (ActionGeo_Lat, ActionGeo_Long in GDELT data). Problem: the current Event dataclass and SQLite schema do NOT store lat/lon. The GDELT poller (`_gdelt_row_to_event`) only extracts country_iso from `ActionGeo_CountryCode`, discarding coordinates.

**Fix required:** Add `lat` and `lon` columns to the events table. Modify `_gdelt_row_to_event` to extract `ActionGeo_Lat` and `ActionGeo_Long`. This is a breaking change to the Event dataclass but straightforward migration.

```python
# Modified Event dataclass
@dataclass
class Event:
    # ... existing fields ...
    lat: Optional[float] = None
    lon: Optional[float] = None
```

**Layer 5: ScenarioZones** -- Currently works correctly (populated from selected forecast's scenario entities). The issue is that the Arcs and Heatmap pills appear interactive but show nothing. Once those are wired, ScenarioZones becomes useful by comparison context.

**Frontend wiring in globe-screen.ts:**

```typescript
// Add to refresh scheduler registrations:
{
  name: 'globe-heatmap',
  fn: async () => {
    const heatmap = await forecastClient.getEventHeatmap(7);
    if (deckMap) deckMap.updateHeatmap(heatmap);  // New DeckGLMap method
  },
  intervalMs: 300_000,  // 5 minutes (event data doesn't change fast)
},
{
  name: 'globe-arcs',
  fn: async () => {
    if (!deckMap || !selectedCountryIso) return;
    const relations = await forecastClient.getCountryRelations(selectedCountryIso);
    if (deckMap) deckMap.updateArcs(relations);  // New DeckGLMap method
  },
  intervalMs: 120_000,
  condition: () => !!selectedCountryIso,  // Only when a country is selected
},
```

**New DeckGLMap methods:**
- `updateHeatmap(points: HeatDatum[])` -- populates `this.heatData`
- `updateArcs(relations: ArcData[])` -- replaces `buildArcsForCountry()`

**Heatmap performance concern:** Raw GDELT events can be 50K+ per week. Sending 50K lat/lon pairs to the frontend is wasteful. Server-side aggregation into grid cells (e.g., 0.5-degree bins) reduces payload to ~2K points while preserving visual density. The HeatmapLayer already handles interpolation.

**New files:**
- `src/api/routes/v1/heatmap.py` -- event heatmap endpoint with grid aggregation
- `src/api/routes/v1/relations.py` -- country relations endpoint

**Modified files:**
- `src/database/models.py` -- add `lat`, `lon` to Event
- `src/database/storage.py` -- add `query_heatmap()`, add lat/lon migration
- `src/ingest/gdelt_poller.py` -- extract lat/lon in `_gdelt_row_to_event()`
- `frontend/src/components/DeckGLMap.ts` -- add `updateHeatmap()`, `updateArcs()`
- `frontend/src/screens/globe-screen.ts` -- wire new endpoints to refresh scheduler
- `frontend/src/services/forecast-client.ts` -- add `getEventHeatmap()`, `getCountryRelations()`

---

### 2.7 Polymarket Hardening + Brier Score Tracking

**Current state:** Polymarket loop runs as `asyncio.Task` inside FastAPI lifespan. Auto-forecaster caps at 3 new + 5 reforecast per day. Snapshot time series stored in `polymarket_snapshots`.

**What needs hardening:**

1. **Daemon consolidation** -- Move from bare `asyncio.Task` to APScheduler job (covered in 2.2)
2. **Error resilience** -- The current `_polymarket_loop` has a broad try/except that swallows all errors. Need per-stage error reporting to IngestRun.
3. **Brier score tracking** -- Currently computed ad-hoc. Need persistent tracking:

**New PostgreSQL table:**
```sql
CREATE TABLE brier_score_history (
    id SERIAL PRIMARY KEY,
    computed_at TIMESTAMPTZ DEFAULT NOW(),
    geopol_cumulative_brier FLOAT NOT NULL,
    polymarket_cumulative_brier FLOAT NOT NULL,
    sample_size INTEGER NOT NULL,
    window_start DATE NOT NULL,
    window_end DATE NOT NULL,
    per_category_json JSONB DEFAULT '{}'::jsonb
);
```

**Integration with backtesting:** The `polymarket_accuracy()` method in BacktestEngine produces the comparison data. The Brier score history table persists it for time-series display.

**Admin panel:** Cumulative accuracy curve (Geopol vs Polymarket Brier score over time), rendered in BacktestingPanel.

**Modified files:**
- `src/polymarket/auto_forecaster.py` -- improved error reporting per stage
- `src/scheduler/jobs.py` -- Polymarket job with IngestRun recording
- Alembic migration for `brier_score_history`

---

### 2.8 Source Expansion (WM-Derived Feed Management)

**WM patterns observed:**

World Monitor's `src/config/feeds.ts` has:
- `SOURCE_TIERS` -- numeric tier assignment per source name
- `FEEDS` array -- complete feed list with URL, name, category
- `SOURCE_PROPAGANDA_RISK` -- per-source propaganda risk metadata

Geopol already adopted this pattern in `src/ingest/feed_config.py`:
- `FeedTier` enum (TIER_1 = 1, TIER_2 = 2)
- `FeedSource` frozen dataclass (name, url, tier, category, lang)
- `PROPAGANDA_RISK` dict

WM's `data-freshness.ts` tracks staleness per source with thresholds (15min fresh, 2h stale, 6h very_stale). Geopol's `src/api/routes/v1/sources.py` has a simpler version querying `ingest_runs`.

**Admin feed management architecture:**

Currently, feeds are hardcoded in `feed_config.py`. For admin management:

**Option A: Move feed config to PostgreSQL** -- feeds become DB rows, admin can CRUD.
**Option B: Keep feed config in code, admin can enable/disable** -- simpler, no migration.

**Recommendation:** Option B for v3.0. Feed URLs rarely change. The admin needs to enable/disable and adjust tiers, not add arbitrary URLs. Store overrides in a `feed_overrides` table:

```sql
CREATE TABLE feed_overrides (
    feed_name VARCHAR(100) PRIMARY KEY,  -- matches FeedSource.name
    enabled BOOLEAN DEFAULT TRUE,
    tier_override INTEGER,               -- null = use code default
    poll_interval_override INTEGER       -- null = use tier default
);
```

At poll time, merge code-defined feeds with DB overrides:
```python
def get_active_feeds(tier: FeedTier) -> list[FeedSource]:
    code_feeds = get_feeds_by_tier(tier)
    overrides = _load_overrides()  # cached, refreshed every 5 min
    return [f for f in code_feeds if overrides.get(f.name, {}).get('enabled', True)]
```

This preserves the simplicity of the existing feed_config.py while giving the admin control.

**New files:**
- `src/api/routes/v1/admin.py` (feeds section) -- CRUD for feed_overrides
- `frontend/src/components/admin/SourceManagerPanel.ts`
- Alembic migration for `feed_overrides`

---

## 3. Component Dependency Graph

```
                    APScheduler (core)
                    /    |    |    \
              GDELT   RSS  ACLED  Daily   Polymarket  ICEWS  UCDP  Backtesting  Seeding
              Poller  Daemon Poller Pipeline Matching  Poller Poller Engine      Computer
                |       |     |       |        |         |      |       |           |
                v       v     v       v        v         v      v       v           v
            SQLite  ChromaDB SQLite  Postgres  Postgres  SQLite SQLite Postgres   Postgres
            events  articles events  predict   poly      events events backtest   baseline
                |                     |  |      |                       |           |
                +---------------------+--+------+-----------------------+-----------+
                                      |
                                Forecast API
                               (countries, forecasts, heatmap, relations)
                                      |
                              Frontend (TypeScript)
                             /     |      |       \
                        /dashboard /globe /forecasts /admin
```

---

## 4. Suggested Build Order

Based on dependency analysis:

### Phase A: Daemon Consolidation (foundation -- everything else depends on this)
1. APScheduler integration into FastAPI lifespan
2. Refactor existing pollers to single-cycle functions
3. Remove standalone systemd services
4. Admin scheduler API endpoints (pause/resume/trigger)

**Rationale:** All subsequent features (ICEWS, UCDP, backtesting, seeding) need to register as scheduler jobs. Build the scheduler first.

### Phase B: Admin Dashboard (UI for everything else)
1. Admin route with auth gating
2. SchedulerPanel (visualize/control jobs)
3. API key admin management

**Rationale:** Once scheduler exists, admin needs visibility. Building admin UI early means all subsequent features get admin panels "for free" as they're built.

### Phase C: Source Expansion + Globe Data Wiring (parallel tracks)

**Track C1: New Sources**
1. Event table schema: add lat/lon columns
2. ICEWS poller + scheduler job
3. UCDP poller + scheduler job
4. Feed override table + admin SourceManagerPanel
5. Entity resolution (country-level normalization)

**Track C2: Globe Layers**
1. Heatmap endpoint + DeckGLMap.updateHeatmap()
2. Relations endpoint + DeckGLMap.updateArcs()
3. Globe screen wiring (refresh scheduler integration)

**Rationale:** C1 and C2 can proceed in parallel. Globe heatmap benefits from C1 (more events = denser heatmap) but can ship with GDELT-only data first.

### Phase D: Global Seeding
1. BaselineRiskComputer
2. Merge baseline + forecast risk in countries.py
3. Advisory level -> risk floor mapping
4. Scheduler job (every 6 hours)

**Rationale:** Depends on C1 (more event sources = better baseline) but can start with GDELT+ACLED data.

### Phase E: Polymarket Hardening + Backtesting
1. Polymarket: migrate from asyncio.Task to APScheduler job
2. Brier score history table + tracking
3. BacktestEngine (walk-forward, model comparison, calibration audit)
4. Polymarket accuracy curves
5. Admin BacktestingPanel

**Rationale:** Backtesting benefits from having more historical data. Running it last gives the most data to work with. Polymarket hardening is low-risk refactoring.

---

## 5. Data Flow Changes Summary

### New API Endpoints

| Endpoint | Method | Purpose | Auth |
|----------|--------|---------|------|
| `/api/v1/events/heatmap` | GET | Geo-located events for globe heatmap | API key |
| `/api/v1/countries/{iso}/relations` | GET | Country-to-country event relationships | API key |
| `/api/v1/admin/auth/verify` | POST | Admin credential verification | Admin key |
| `/api/v1/admin/scheduler/jobs` | GET | List scheduler jobs | Admin key |
| `/api/v1/admin/scheduler/jobs/{id}/pause` | POST | Pause a job | Admin key |
| `/api/v1/admin/scheduler/jobs/{id}/resume` | POST | Resume a job | Admin key |
| `/api/v1/admin/scheduler/jobs/{id}/trigger` | POST | Fire a job immediately | Admin key |
| `/api/v1/admin/sources/feeds` | GET/POST | Feed override management | Admin key |
| `/api/v1/admin/backtesting` | GET | Latest backtest reports | Admin key |
| `/api/v1/admin/backtesting/trigger` | POST | Trigger backtest run | Admin key |

### New PostgreSQL Tables

| Table | Purpose |
|-------|---------|
| `data_sources` | Source metadata and health tracking |
| `baseline_country_risk` | Baseline risk scores for all countries |
| `backtest_reports` | Persistent backtest analysis results |
| `brier_score_history` | Cumulative Brier score time series |
| `feed_overrides` | Admin feed enable/disable + tier overrides |

### Modified Tables

| Table | Change |
|-------|--------|
| `api_keys` | Add `is_admin BOOLEAN DEFAULT FALSE` |

### SQLite Schema Changes

| Change | Purpose |
|--------|---------|
| Add `lat REAL`, `lon REAL` to `events` | Geo-located events for heatmap |

---

## 6. Anti-Patterns to Avoid

### Anti-Pattern 1: Dual Scheduler
**What:** Running APScheduler AND keeping standalone systemd services "just in case."
**Why bad:** Two scheduling systems = race conditions on shared SQLite, double IngestRun entries, impossible to reason about state.
**Instead:** Full migration to APScheduler. Remove all standalone scripts/daemons. One scheduler, one truth.

### Anti-Pattern 2: Admin as Separate App
**What:** Building admin as a separate frontend application with its own build pipeline.
**Why bad:** Doubles build complexity, separate deployment, separate auth system, divergent component patterns.
**Instead:** Same app, same build, dynamic import code-split. Admin components are just TypeScript modules that happen to only load on `/admin`.

### Anti-Pattern 3: Real-Time Event Heatmap
**What:** WebSocket-pushing individual GDELT events to the globe heatmap in real-time.
**Why bad:** GDELT updates every 15 minutes (not real-time). Server-side aggregation is mandatory for payload size. WebSocket adds complexity for no latency benefit.
**Instead:** 5-minute polling with pre-aggregated grid cells. The refresh scheduler pattern already exists.

### Anti-Pattern 4: Cross-Source Entity Resolution at Ingest Time
**What:** Attempting to merge GDELT + ACLED + ICEWS + UCDP events into unified entities at ingest time.
**Why bad:** Different coding systems (CAMEO vs ACLED event types), different actor granularity, different temporal resolution. Perfect dedup is a research problem.
**Instead:** Insert all events with source discriminator. Deduplicate at graph-insertion or query time using country-level normalization only.

---

## 7. Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| APScheduler jobstore sync driver conflicts with asyncpg | LOW | MEDIUM | Use MemoryJobStore (recommended) |
| ICEWS Dataverse API rate limits or token issues | MEDIUM | LOW | Weekly poll, exponential backoff |
| UCDP API token procurement delay | MEDIUM | LOW | Fall back to bulk CSV download |
| Cross-source event dedup false positives | MEDIUM | MEDIUM | Country-level only, conservative matching |
| Admin auth bypass (single-operator system) | LOW | HIGH | API key + is_admin flag, no complex RBAC |
| Globe heatmap performance with 100K+ points | MEDIUM | MEDIUM | Server-side grid aggregation (0.5-degree bins) |
| Daily pipeline timeout under APScheduler | LOW | HIGH | Set generous `misfire_grace_time`, separate max_instances |

---

## Sources

- APScheduler + FastAPI integration patterns: [Sentry Guide](https://sentry.io/answers/schedule-tasks-with-fastapi/), [Medium Guide](https://ahaw021.medium.com/scheduled-jobs-with-fastapi-and-apscheduler-5a4c50580b0e), [GitHub Discussion](https://github.com/agronholm/apscheduler/discussions/830)
- ICEWS data access: [Harvard Dataverse](https://dataverse.harvard.edu/dataverse/icews), [icews R package](https://www.andybeger.com/icews/)
- UCDP GED: [Download Center](https://ucdp.uu.se/downloads/), [API Docs](https://ucdp.uu.se/apidocs/), [GED v25.1 Codebook](https://ucdp.uu.se/downloads/ged/ged251.pdf)
- Direct codebase analysis: `src/api/app.py`, `src/ingest/*.py`, `frontend/src/**/*.ts`, `deploy/systemd/*.service`
- World Monitor reference: `worldmonitor/src/config/feeds.ts`, `worldmonitor/src/services/data-freshness.ts`
