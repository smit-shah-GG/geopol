# Technology Stack: v3.0 Operational Command & Verification

**Project:** Geopolitical Forecasting Engine v3.0
**Researched:** 2026-03-04
**Constraint:** RTX 3060 12GB, Python 3.11+, single server, existing FastAPI + PostgreSQL + Redis + SQLite stack
**Overall Confidence:** MEDIUM-HIGH

---

## Executive Summary

v3.0 adds six capability dimensions: (1) daemon consolidation into a single APScheduler process in-process with FastAPI, (2) admin dashboard with job control, (3) UCDP + POLECAT source expansion, (4) walk-forward backtesting, (5) global country seeding, and (6) Polymarket operational hardening.

Three critical findings from this research:

**First:** APScheduler 4.0 is still alpha (v4.0.0a6, last release April 2025). It is a ground-up rewrite with incompatible API, unstable data store schema, and the author explicitly warns against production use. **Use APScheduler 3.11.2** (stable, released December 2025). The v3.x `AsyncIOScheduler` integrates cleanly with FastAPI's asyncio event loop, supports `pause_job()`/`resume_job()`, and `SQLAlchemyJobStore` persists schedules across restarts using the existing PostgreSQL connection.

**Second:** ICEWS was discontinued in April 2023. Its successor is POLECAT (POLitical Event Classification, Attributes, and Types), hosted on Harvard Dataverse, updated weekly, using the PLOVER ontology (not CAMEO). PLOVER-to-CAMEO mapping is non-trivial but feasible. Data is distributed as TSV files via Dataverse API, not a real-time REST API. This changes the integration pattern from "poll an API" to "download weekly file dumps."

**Third:** UCDP introduced mandatory token-based authentication in February 2026. The API remains free but requires contacting the maintainer for a token. Rate limit is 5,000 requests/day. The existing World Monitor codebase already integrates UCDP events and can serve as a reference pattern.

---

## New Dependencies for v3.0

### Daemon Consolidation: APScheduler 3.11.2

| Attribute | Detail |
|-----------|--------|
| **Package** | `APScheduler==3.11.2` |
| **Scheduler class** | `apscheduler.schedulers.asyncio.AsyncIOScheduler` |
| **Job store** | `apscheduler.jobstores.sqlalchemy.SQLAlchemyJobStore` |
| **Why 3.x, not 4.x** | 4.0 is alpha (v4.0.0a6). Complete API rewrite: `add_job()` becomes `add_schedule()`, "job stores" become "data stores" with incompatible schema, event broker system added, `BlockingScheduler`/`BackgroundScheduler` merged into single `Scheduler` class. The author states: "do NOT use this release in production." |
| **Why not Celery** | Celery requires a message broker (RabbitMQ/Redis) as a separate process, worker processes, and brings 15+ transitive dependencies. Total overkill for 4-5 scheduled jobs on a single server. |
| **Why not raw asyncio.create_task** | No persistence, no pause/resume, no retry semantics, no job overlap prevention (`max_instances=1`). Would require reimplementing half of APScheduler. |

**Integration pattern with FastAPI:**

```python
from contextlib import asynccontextmanager
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
from fastapi import FastAPI

scheduler = AsyncIOScheduler(
    jobstores={
        "default": SQLAlchemyJobStore(url=settings.postgres_dsn),
    },
    job_defaults={
        "coalesce": True,       # If multiple missed runs, execute once
        "max_instances": 1,     # Prevent overlapping executions
        "misfire_grace_time": 300,
    },
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    scheduler.start()
    yield
    scheduler.shutdown(wait=True)

app = FastAPI(lifespan=lifespan)
```

**Jobs to consolidate:**

| Current Mechanism | Job | Interval | APScheduler Trigger |
|-------------------|-----|----------|---------------------|
| Standalone `asyncio.run()` loop | GDELT micro-batch ingest | 15 min | `IntervalTrigger(minutes=15)` |
| Standalone `asyncio.run()` loop | RSS feed polling (T1) | 15 min | `IntervalTrigger(minutes=15)` |
| Standalone `asyncio.run()` loop | RSS feed polling (T2) | 60 min | `IntervalTrigger(minutes=60)` |
| systemd timer | Daily forecast pipeline | Daily 06:00 | `CronTrigger(hour=6)` |
| Called from daily pipeline | Polymarket auto-forecaster | Daily (after forecasts) | `CronTrigger(hour=7)` |
| systemd timer | ACLED polling | 6 hours | `IntervalTrigger(hours=6)` |
| New (v3.0) | UCDP polling | Daily | `CronTrigger(hour=3)` |
| New (v3.0) | POLECAT weekly download | Weekly | `CronTrigger(day_of_week='mon', hour=2)` |
| Existing `RetrainingScheduler` | TKG retraining | Weekly | `CronTrigger(day_of_week='sun', hour=2)` |

**Admin API for job control (APScheduler 3.x methods):**

```python
# Pause a job
scheduler.pause_job(job_id="gdelt_ingest")

# Resume a job
scheduler.resume_job(job_id="gdelt_ingest")

# Trigger immediate execution
scheduler.modify_job(job_id="daily_forecast", next_run_time=datetime.now())

# Get all jobs with status
jobs = scheduler.get_jobs()
# Each job exposes: id, name, next_run_time (None = paused), trigger, func_ref
```

**Known pitfall — APScheduler memory leak (GitHub #235):** v3.x retains references to tracebacks when jobs raise exceptions, preventing GC. Mitigation: wrap every job function in `try/except` that explicitly logs and discards the traceback. This is already documented in the existing `PITFALLS.md`.

**Confidence:** HIGH. APScheduler 3.11.2 is mature, well-documented, and the `AsyncIOScheduler` + `SQLAlchemyJobStore` pattern is a known-good combination for FastAPI.

---

### Source Expansion: UCDP API Client

| Attribute | Detail |
|-----------|--------|
| **No new dependency** | Raw `aiohttp` (already in `pyproject.toml`) against the REST API |
| **API base URL** | `https://ucdpapi.pcr.uu.se/api/{resource}/{version}` |
| **Authentication** | `x-ucdp-access-token` HTTP header (mandatory since Feb 2026) |
| **Rate limit** | 5,000 requests/day (errors count) |
| **Response format** | JSON with `TotalCount`, `TotalPages`, `Result[]` |
| **Key resources** | `gedevents/25.1` (georeferenced events), `ucdpprioconflict/25.1` (armed conflicts) |
| **Pagination** | `pagesize` + `page` query params |
| **Data freshness** | Candidate events updated monthly (v26.0.1), yearly datasets updated annually |

**Integration pattern:** Mirror the existing `ACLEDPoller` class structure. UCDP events map to the unified `Event` schema with `source="ucdp"`. The UCDP API returns CAMEO-adjacent conflict categorizations (state-based, non-state, one-sided violence) that need mapping to CAMEO event codes.

**WM reference:** `/home/kondraki/personal/worldmonitor/server/worldmonitor/conflict/v1/list-ucdp-events.ts` shows WM's UCDP integration: Redis-cached with 25h TTL, populated by a cron job. Geopol should follow the same pattern but with APScheduler instead of external cron.

**Action required before implementation:** Email UCDP API maintainer to request access token. Describe project and intended use.

**Confidence:** HIGH. API is documented, free, and the existing `ACLEDPoller` provides a direct template.

---

### Source Expansion: POLECAT (ICEWS Successor)

| Attribute | Detail |
|-----------|--------|
| **New dependency** | `pyDataverse>=0.3.1` (Harvard Dataverse API client) |
| **Dataset DOI** | `doi:10.7910/DVN/AJGVIT` (weekly data) |
| **Data format** | TSV files (tab-separated), one file per week |
| **Ontology** | PLOVER (not CAMEO) — different event taxonomy |
| **Update frequency** | Weekly (current year unzipped, historical years as ZIP archives) |
| **Access** | Dataverse API with API token (`X-Dataverse-key` header) |
| **Coverage** | 2018-present, 7 languages, 1000+ sources globally |

**Critical consideration — PLOVER-to-CAMEO mapping:**

POLECAT uses PLOVER, a fundamentally different event ontology from CAMEO. PLOVER uses an event-mode-context structure (e.g., PROTEST event with DEMONSTRATION mode) vs. CAMEO's flat numeric codes. A mapping layer is required. This is non-trivial but bounded:

- PLOVER has ~20 top-level event types vs. CAMEO's ~20 root codes
- Both encode actor-action-target triples
- The mapping will be lossy (PLOVER has modes that CAMEO doesn't distinguish)
- Recommendation: build a static `PLOVER_TO_CAMEO` dict covering the ~20 top-level types, accept precision loss on modes

**Integration pattern:** Unlike GDELT (15-min poll) and ACLED (6-hour poll), POLECAT is a weekly batch download. The APScheduler job should:
1. Query Dataverse API for latest weekly file
2. Download TSV to temp directory
3. Parse and map PLOVER events to CAMEO-coded `Event` schema
4. Bulk insert into SQLite event store
5. Clean up temp files

**Alternative considered: skip POLECAT, keep GDELT + ACLED + UCDP only.**
This is defensible. POLECAT adds a third event coding ontology (PLOVER) alongside CAMEO (GDELT) and ACLED's own categories. The mapping overhead may not justify the incremental signal, especially since GDELT already covers 500K-1M articles daily. Recommend implementing UCDP first and deferring POLECAT to v3.1 if the PLOVER mapping proves too lossy in prototyping.

**Confidence:** MEDIUM. Data format has known quirks (excess columns, quoting issues). PLOVER-to-CAMEO mapping is untested. pyDataverse is at v0.3.1 (not v1.0).

---

### Backtesting: Custom Implementation

| Attribute | Detail |
|-----------|--------|
| **No new dependency** | Custom implementation using existing `numpy`, `pandas`, `scipy` |
| **Why not skforecast** | skforecast targets scikit-learn regressors for numerical time-series. Geopol's forecasts are probabilistic event predictions (0-1 probabilities over discrete geopolitical events), not continuous value forecasts. |
| **Why not backtesting.py** | Financial instrument backtesting (candlestick data, buy/sell signals). Wrong domain entirely. |
| **Why not sktime** | Closest fit, but still oriented toward numerical time-series with `predict()` interface. Geopol's pipeline is: events -> KG -> TKG embedding -> LLM ensemble -> calibrated probability. None of these stages conform to scikit-learn's estimator API. |

**Recommendation: build a thin custom walk-forward harness.**

The backtesting system is an internal reporting tool, not a user-facing feature. It needs:

1. **Walk-forward splitter**: Given a date range, yield `(train_window, eval_window)` pairs with configurable step size
2. **Prediction collector**: For each eval window, run the full pipeline (or replay cached predictions) and collect `(question, predicted_probability, actual_outcome)` triples
3. **Scorer**: Compute Brier score, calibration curve (reliability diagram), and resolution/reliability decomposition per window
4. **Model comparator**: Side-by-side TiRGN vs RE-GCN performance over time
5. **Reporter**: Generate JSON + matplotlib plots for admin dashboard display

This is ~300-500 lines of focused code, not a framework adoption decision. The existing `src/evaluation/provisional_scorer.py` and `src/calibration/` modules already implement the scoring math.

**Confidence:** HIGH. Custom implementation is the correct call. The domain-specific pipeline doesn't fit any existing backtesting framework.

---

### Admin Dashboard: No New Backend Dependencies

| Attribute | Detail |
|-----------|--------|
| **Authentication** | Extend existing `ApiKey` model with `is_admin: bool` column |
| **Routing** | New `src/api/routes/v1/admin.py` router with `verify_admin_key` dependency |
| **Frontend** | Same TypeScript app, `/admin` route, dynamic import code-split |
| **No new dependency** | Existing FastAPI + Pydantic + SQLAlchemy handles everything |

**Admin auth pattern:**

```python
async def verify_admin_key(
    client_name: str = Depends(verify_api_key),
    db: AsyncSession = Depends(get_db),
) -> str:
    """Verify the API key belongs to an admin."""
    result = await db.execute(
        select(ApiKey).where(
            ApiKey.client_name == client_name,
            ApiKey.is_admin.is_(True),
        )
    )
    if result.scalar_one_or_none() is None:
        raise HTTPException(status_code=403, detail="Admin access required.")
    return client_name
```

**Admin API endpoints needed:**

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/v1/admin/jobs` | GET | List all scheduled jobs with status |
| `/api/v1/admin/jobs/{id}/pause` | POST | Pause a job |
| `/api/v1/admin/jobs/{id}/resume` | POST | Resume a job |
| `/api/v1/admin/jobs/{id}/trigger` | POST | Trigger immediate execution |
| `/api/v1/admin/feeds` | GET | List RSS feeds with health |
| `/api/v1/admin/feeds` | POST | Add a new RSS feed |
| `/api/v1/admin/feeds/{id}` | DELETE | Remove an RSS feed |
| `/api/v1/admin/sources` | GET | Source health overview (GDELT, ACLED, UCDP, POLECAT) |
| `/api/v1/admin/backtest` | POST | Trigger backtesting run |
| `/api/v1/admin/backtest/results` | GET | Retrieve latest backtest results |

**WM reference for feed management:** WM uses a static TypeScript config (`src/config/feeds.ts`) with `VARIANT_FEEDS` dict organized by category. Feed health is tracked via in-memory `feedFailures` Map with cooldown logic (2 failures -> 5-min cooldown). Geopol should persist feed config in PostgreSQL (not hardcoded) to allow admin CRUD, while keeping the cooldown/health tracking pattern from WM's `rss.ts`.

**RSS feed config migration path:** Move `src/ingest/feed_config.py` feed definitions to a `feeds` PostgreSQL table. Seed with current hardcoded feeds via Alembic migration. Admin API then does CRUD against this table. `RSSDaemon` queries the table on each cycle instead of importing from `feed_config.py`.

**Confidence:** HIGH. Straightforward extension of existing patterns.

---

### Global Country Seeding: No New Dependencies

| Attribute | Detail |
|-----------|--------|
| **Data sources for baseline risk** | GDELT event density, ACLED event counts, UCDP conflict data, travel advisories |
| **No new dependency** | Existing `aiohttp`, `sqlalchemy`, advisory system covers this |
| **Storage** | New `country_risk_baseline` table in PostgreSQL |

The global seeding feature computes a baseline risk score (0-100) for all ~195 countries using aggregated event data. No new libraries needed — this is a SQL aggregation + periodic recomputation task, scheduled via APScheduler.

**Confidence:** HIGH.

---

### Polymarket Hardening: No New Dependencies

| Attribute | Detail |
|-----------|--------|
| **Existing** | `src/polymarket/auto_forecaster.py` (627 lines), `src/polymarket/client.py` |
| **Changes needed** | Retry logic, Brier score tracking, cap enforcement |
| **No new dependency** | `tenacity` (already installed) for retries, existing `scipy` for scoring |

**Confidence:** HIGH.

---

## Full Dependency Additions

```toml
# In pyproject.toml [project.dependencies]
"APScheduler>=3.11,<4.0",     # Daemon consolidation

# In [project.optional-dependencies] ingest
"pyDataverse>=0.3.1",         # POLECAT weekly download (defer to v3.1 if PLOVER mapping is too lossy)
```

That's it. Two new packages. Everything else uses existing dependencies.

---

## Alembic Migrations Required

| Migration | Table | Change |
|-----------|-------|--------|
| Add admin flag | `api_keys` | `ADD COLUMN is_admin BOOLEAN DEFAULT FALSE` |
| Feed config table | `feeds` (new) | `name, url, tier, category, lang, enabled, created_at, updated_at` |
| Feed health table | `feed_health` (new) | `feed_id, last_success, last_failure, failure_count, cooldown_until` |
| Country risk baseline | `country_risk_baseline` (new) | `country_code, risk_score, gdelt_density, acled_count, ucdp_count, advisory_level, computed_at` |
| Backtest results | `backtest_runs` (new) | `id, started_at, completed_at, config_json, results_json, model_type` |
| UCDP source tracking | `ingest_runs` (existing) | No schema change — `daemon_type='ucdp'` uses existing `IngestRun` model |
| APScheduler job store | `apscheduler_jobs` (auto-created) | SQLAlchemyJobStore creates this automatically |

---

## What NOT to Add

| Rejected | Why |
|----------|-----|
| APScheduler 4.x | Alpha (v4.0.0a6). Ground-up rewrite with unstable API. Author warns against production use. |
| Celery | Message broker + worker processes. Overkill for 8-9 scheduled jobs on a single server. |
| `schedule` library | Already in `pyproject.toml` (ingest extras). Single-threaded, no persistence, no pause/resume. Inferior to APScheduler for daemon consolidation. |
| skforecast / backtesting.py / sktime | Wrong domain. These target numerical time-series or financial instruments. Geopol's probabilistic event forecasting pipeline doesn't conform to scikit-learn estimator APIs. |
| FastAPI-Scheduler wrapper | Thin wrapper around APScheduler adding FastAPI route decorators. Adds indirection without value — direct APScheduler integration is cleaner. |
| Dataverse bulk download tools | `pyDataverse` is sufficient for weekly file retrieval. No need for heavy ETL frameworks. |
| ICEWS | **Discontinued April 2023.** Replaced by POLECAT. Do not attempt to access historical ICEWS data for ongoing event ingestion. |

---

## Alternatives Considered

| Decision | Chosen | Alternative | Rationale |
|----------|--------|-------------|-----------|
| Scheduler | APScheduler 3.11.2 | APScheduler 4.x | 4.x is alpha, API unstable, author warns against production |
| Scheduler | APScheduler 3.11.2 | Celery | Requires broker + workers, 15+ transitive deps, overkill |
| ICEWS replacement | POLECAT via Dataverse | Skip entirely | POLECAT provides additional signal; defer if PLOVER mapping is lossy |
| UCDP client | Raw aiohttp | Dedicated UCDP library | No maintained Python UCDP client exists; only R packages (`ucdp.api`) |
| Backtesting | Custom ~400 LOC | skforecast / sktime | Wrong domain; geopol pipeline doesn't fit scikit-learn estimator API |
| Admin auth | `is_admin` flag on `ApiKey` | Separate admin user model | Single-server, single-admin. Role flag is sufficient. RBAC is overengineering. |
| Feed management | PostgreSQL table | Keep hardcoded `feed_config.py` | Admin CRUD requires persistent, mutable config. Python source code is not a database. |

---

## Integration Points with Existing Stack

| Existing Component | v3.0 Integration |
|-------------------|------------------|
| FastAPI lifespan | APScheduler `start()`/`shutdown()` in lifespan context manager |
| PostgreSQL (asyncpg) | SQLAlchemyJobStore uses same connection string; new tables via Alembic |
| Redis (Upstash) | Cache UCDP responses (same pattern as existing forecast caching) |
| `ACLEDPoller` | Template for `UcdpPoller` class structure |
| `RSSDaemon` | Moves from standalone `asyncio.run()` to APScheduler-managed job |
| `GDELTPoller` | Same — standalone loop becomes APScheduler job |
| `verify_api_key` | Extended with `verify_admin_key` wrapper |
| `IngestRun` model | Reused for UCDP/POLECAT audit trail (`daemon_type='ucdp'`, `daemon_type='polecat'`) |
| `EventStorage` | UCDP/POLECAT events stored via same `insert_events()` path |
| `RetrainingScheduler` | Wrapped as APScheduler job instead of standalone scheduling logic |
| `src/calibration/` | Backtesting reuses Brier score computation |
| `src/evaluation/provisional_scorer.py` | Backtesting reuses scoring logic |

---

## Sources

### HIGH Confidence
- [APScheduler PyPI](https://pypi.org/project/APScheduler/) — v3.11.2 stable (Dec 2025), v4.0.0a6 alpha (Apr 2025)
- [APScheduler 3.x User Guide](https://apscheduler.readthedocs.io/en/3.x/userguide.html) — AsyncIOScheduler, SQLAlchemyJobStore, pause/resume API
- [APScheduler Migration Guide](https://apscheduler.readthedocs.io/en/master/migration.html) — v4.0 breaking changes documented
- [UCDP API Documentation](https://ucdp.uu.se/apidocs/) — endpoints, auth, rate limits, pagination
- Existing codebase: `src/ingest/acled_poller.py`, `src/ingest/rss_daemon.py`, `src/api/middleware/auth.py`
- WM codebase: `server/_shared/acled.ts`, `server/worldmonitor/conflict/v1/list-ucdp-events.ts`, `src/services/rss.ts`, `src/config/feeds.ts`

### MEDIUM Confidence
- [POLECAT Weekly Data (Harvard Dataverse)](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/AJGVIT) — dataset location confirmed, format details sparse
- [PLOVER Ontology (GitHub)](https://github.com/openeventdata/PLOVER) — event type structure, JSONL recommendation
- [pyDataverse Documentation](https://pydataverse.readthedocs.io/en/latest/) — v0.3.1, Dataverse API client
- [POLECAT Event Data Analysis](https://www.andybeger.com/blog/2024-05-21-polecat-event-data/) — format quirks (TSV, excess columns, quoting issues)
- [ICEWS Discontinuation](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/28075) — confirmed discontinued April 2023

### LOW Confidence
- PLOVER-to-CAMEO mapping feasibility — no published mapping exists; assessed from ontology structure comparison only
- pyDataverse stability — v0.3.1 is not v1.0; API may have quirks with large file downloads
