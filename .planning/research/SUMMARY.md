# v3.0 Research Summary: Operational Command & Verification

**Project:** Geopolitical Forecasting Engine v3.0
**Domain:** Operational hardening — admin layer, daemon consolidation, source expansion, backtesting, global seeding
**Researched:** 2026-03-04
**Confidence:** MEDIUM-HIGH

---

## Executive Summary

v3.0 adds the operational command layer that v2.0/v2.1 deliberately deferred: a single unified scheduler replacing four isolated processes, an admin dashboard for system control without SSH access, two new event data sources (UCDP + POLECAT), walk-forward backtesting, global country risk seeding, Polymarket hardening, and data wiring for the three globe layers that currently render empty. The research reveals a system that is architecturally clean and well-structured, with consistent patterns (IngestRun audit trail, async pollers, Panel-based TypeScript UI) that extend naturally into v3.0 features. The dominant engineering challenge is consolidation without regression: collapsing four isolated processes into one APScheduler instance eliminates systemd's process isolation guarantees, and any memory leak, OOM, or event loop block in a single job now takes down the entire API.

Three findings with direct impact on sequencing: (1) **APScheduler 4.x is production-unsuitable** — use 3.11.2 (`AsyncIOScheduler` + in-memory `MemoryJobStore`). The 4.x rewrite has unstable API, incompatible job store schema, and the author explicitly warns against production use. (2) **ICEWS was discontinued April 2023** — the intended data source no longer exists. The successor is POLECAT (Harvard Dataverse, weekly TSV, PLOVER ontology), but PLOVER-to-CAMEO mapping is non-trivial and untested. UCDP is a safer first integration: well-documented REST API, CAMEO-adjacent coding, directly templatable from the existing `ACLEDPoller`. (3) **UCDP requires a token** (mandatory auth since February 2026) — this must be requested from the UCDP team before Phase 23 begins. Email-gated token procurement takes days to weeks and is a pre-phase blocker.

The recommended approach is to build in five phases following the natural dependency chain: admin dashboard and daemon consolidation first (foundation for everything else), then source expansion and globe data wiring in parallel, then global seeding, then Polymarket hardening and backtesting last (requires the most accumulated historical data to be meaningful). The single non-negotiable architectural constraint: APScheduler runs `--workers 1` on uvicorn — running multiple workers creates duplicate job execution with no coordination.

---

## Key Findings

### Recommended Stack

**Core conclusion:** v3.0 requires exactly two new Python dependencies: `APScheduler>=3.11,<4.0` and `pyDataverse>=0.3.1` (the latter only if POLECAT is implemented; otherwise zero new production dependencies). Everything else extends the existing FastAPI + PostgreSQL + Redis + SQLite stack. No new frontend frameworks, no new databases, no new auth systems.

**Technology decisions (prescriptive):**

| Technology | Decision | Rationale |
|-----------|----------|-----------|
| `APScheduler==3.11.2` | Use this, lock the minor version | Stable since Dec 2025. 4.x alpha breaks all APIs. |
| `AsyncIOScheduler` class | Use this, not `BackgroundScheduler` | FastAPI is async-native; must share the same event loop |
| `MemoryJobStore` | Use this, not `SQLAlchemyJobStore` | Static job definitions don't need persistence. SQLAlchemy jobstore requires sync psycopg2 driver, adding a second Postgres driver alongside asyncpg. |
| `uvicorn --workers 1` | Hard constraint | APScheduler in-process is not safe with multiple workers — each worker creates its own scheduler instance, running every job N times |
| `pyDataverse>=0.3.1` | Optional, defer to v3.1 | POLECAT integration. Only if PLOVER-to-CAMEO mapping proves tractable |
| POLECAT | Defer to v3.1 | ICEWS is dead (April 2023). POLECAT is its successor but uses PLOVER ontology. Mapping layer is non-trivial and untested. UCDP is the safer v3.0 source. |
| UCDP | Implement in v3.0, Phase 23 | Documented API, CAMEO-adjacent coding, direct ACLEDPoller template |
| Admin auth | `is_admin` flag on `api_keys` table | Single operator. RBAC is overengineering. Separate admin router with router-level dependency. |
| Feed config | `feed_overrides` table (enable/disable/tier) | Keep code-defined feeds as canonical source. DB stores overrides only. Avoids migration of 50+ feed definitions. |
| Backtesting framework | Custom ~400 LOC | `skforecast`, `sktime`, `backtesting.py` all target scikit-learn estimator APIs. Geopol's probabilistic event pipeline doesn't conform. |

**What NOT to add:** APScheduler 4.x, Celery, ICEWS, any backtesting framework, FastAPI-Scheduler wrapper, `schedule` library, multi-worker uvicorn setup.

**Confidence:** HIGH for APScheduler 3.x, UCDP, admin auth decisions — all verified against official docs and existing codebase. MEDIUM for POLECAT/PLOVER — format quirks documented, mapping untested.

### Expected Features

**Must have (table stakes for v3.0):**
- Daemon status overview in admin — green/yellow/red per job, last run time, error count
- Manual trigger / pause / resume per daemon — foundational reason for the admin dashboard
- Ingest run history table — last N runs per daemon type, sortable
- UCDP event source integration — new conflict data, weekly cadence
- Walk-forward backtesting report — calibration curve (reliability diagram) + cumulative Brier score
- Global baseline risk for all ~195 countries — globe choropleth requires non-zero data everywhere
- Polymarket cap tracking fix — `reforecasted_at` column, unified `BudgetTracker`, eliminates known drift
- Globe heatmap + arcs data wiring — three layers currently render empty; data is available, just unwired

**Should have (differentiators, v3.0):**
- TKG training status + on-demand retrain trigger in admin
- Per-CAMEO calibration audit (reliability diagrams per event type over time)
- Model comparison: TiRGN vs RE-GCN on resolved predictions
- Feed health table with auto-disable after N consecutive failures
- Risk trend arrows (7-day delta) on globe country layer
- Advisory-level risk floor (Do Not Travel -> 70 minimum, etc.)
- Active forecast override for baseline risk (prediction-derived risk beats event-density baseline)

**Defer (v3.1+):**
- POLECAT integration (PLOVER-to-CAMEO mapping is untested; validate first)
- Feed CRUD via admin UI — add/remove arbitrary URLs (current: enable/disable only)
- Public-facing accuracy page (internal admin tooling only in v3.0)
- Animated temporal arcs on globe (DeckGL `TripLayer` — visually compelling, high complexity)
- Sub-national risk scores
- Multi-platform accuracy comparison (Kalshi, Manifold — Polymarket only for v3.0)
- Walk-forward eval with full TKG retraining per window (computationally expensive batch job; start with static-weight backtesting of resolved predictions)

**Feature dependency chain (hard ordering constraints):**
```
Phase 19 (Admin Dashboard) ─── no dependencies, start immediately
    ↓
Phase 20 (Daemon Consolidation) ─── enables admin pause/resume to actually work
    ↓ (parallel tracks diverge here)
Phase 21 (Polymarket Hardening) ──────────── Phase 23 (Source Expansion: UCDP)
    ↓                                              ↓
Phase 22 (Backtesting) ─────────────────── Phase 24 (Global Seeding)
                                                   ↓
                                           Phase 25 (Globe Data Wiring)
```

### Architecture Approach

v3.0 adds 7 capabilities to a clean, well-structured system. The existing architecture has consistent patterns that extend naturally: `IngestRun` audit trail reuses for UCDP/POLECAT without schema changes; `ACLEDPoller` is a direct template for `UcdpPoller`; `verify_api_key` extends to `verify_admin_key` with minimal code; the TypeScript Router adds a 4th route with the same dynamic import pattern used for DeckGL in v2.1. The central structural change is migration from 4-6 separate processes (3 systemd units + 1 asyncio.Task + 2 ad-hoc scripts) to a single `AsyncIOScheduler` in-process with FastAPI, with all jobs registered at startup via the lifespan context manager.

**Major components being added:**

1. **`src/scheduler/`** (new module) — `AsyncIOScheduler` integration, job definitions wrapping existing single-cycle poller methods, admin job control API endpoints. All existing pollers keep their `_poll_once()`/`poll_feeds()` logic intact; APScheduler replaces only the loop-management layer.

2. **`src/api/routes/v1/admin.py`** (new router) — separate `APIRouter` with `dependencies=[Depends(verify_admin_key)]` at router level. Never per-endpoint — this is the anti-pattern that gets admin endpoints accidentally left unprotected. Endpoints: scheduler CRUD, feed override management, backtest trigger/results.

3. **`src/evaluation/backtest_engine.py`** (new) — `BacktestEngine` with `run_walk_forward()`, `compare_models()`, `calibration_audit()`, `polymarket_accuracy()`. READ-ONLY against historical predictions and outcomes. Does not interfere with live pipeline. New `backtest_reports` PostgreSQL table for persistence.

4. **`src/evaluation/baseline_risk.py`** (new) — `BaselineRiskComputer` aggregating events per country with time decay and source weighting. New `baseline_country_risk` PostgreSQL table. Countries API modified to `COALESCE(forecast_risk, baseline_risk)` — active forecasts always override baseline.

5. **`src/ingest/ucdp_poller.py`** (new) — `UCDPPoller` using existing `aiohttp`, following `ACLEDPoller` structure. Requires auth token (contact UCDP before Phase 23). Events mapped to `Event(source="ucdp", ...)` with UCDP-to-CAMEO type mapping.

6. **`frontend/src/screens/admin-screen.ts`** (new) — dynamically imported at `/admin` route, zero bytes in public bundle. Auth check inside `mountAdmin()` (not router-level guard — the Router has no middleware concept). Contains `SchedulerPanel`, `SourceManagerPanel`, `BacktestingPanel`.

7. **Globe data wiring** — `lat`/`lon` columns added to SQLite `events` table; GDELT poller extracts `ActionGeo_Lat`/`ActionGeo_Long`; new `/api/v1/events/heatmap` endpoint with server-side 0.5-degree grid aggregation; new `/api/v1/countries/{iso}/relations` endpoint from event cross-references. `DeckGLMap` gets `updateHeatmap()` and `updateArcs()` methods.

**New Alembic migrations required:**
- `api_keys`: add `is_admin BOOLEAN DEFAULT FALSE`
- New tables: `feed_overrides`, `data_sources`, `baseline_country_risk`, `backtest_reports`, `brier_score_history`
- SQLite: add `lat REAL`, `lon REAL` to `events`

**Systemd impact:** `geopol-gdelt-poller.service` and `geopol-rss-daemon.service` become **obsolete**. Only `geopol-api.service` remains. The daily forecast systemd timer also becomes obsolete (CronTrigger replaces it). Do not run both APScheduler and the old systemd services simultaneously — dual scheduling creates race conditions and duplicate `IngestRun` entries.

### Critical Pitfalls

Ranked by severity and probability of occurrence:

**1. Process isolation loss during daemon consolidation (C1) — CRITICAL**
Four systemd services currently have `MemoryMax=512M` and `Restart=on-failure` each. APScheduler in-process has none of that. A memory leak in trafilatura article extraction or GDELT ZIP parsing accumulates indefinitely; an OOM kill takes down the API, not just the ingest daemon. Prevention: run heavy jobs (daily forecast, TKG retraining) via `ProcessPoolExecutor` to retain OS-level memory isolation; wrap every job in a memory-monitoring decorator; set `max_instances=1` on all jobs; implement a circuit breaker — after 3 consecutive failures, pause the job and surface the error in admin dashboard. Monitor RSS monotonically via `psutil.Process().memory_info().rss` in each job wrapper.

**2. AsyncIOScheduler event loop starvation (C2) — CRITICAL**
APScheduler's `AsyncIOScheduler` runs jobs on the FastAPI event loop. Any synchronous call inside a job function that is NOT offloaded to `asyncio.to_thread()` blocks all HTTP request handling. The existing codebase has correct patterns (`run_in_executor` for feedparser, `asyncio.to_thread` for SQLite writes) but also gaps (synchronous `GeminiClient()` constructor in job setup code). Prevention: ALL job functions must be async; ALL sync work must use `asyncio.to_thread()`; bounded `ThreadPoolExecutor(max_workers=4)` dedicated to job offloading, not shared with FastAPI's default executor; start scheduler after event loop is running (inside lifespan, not at module level).

**3. Admin routes without authentication (C3) — CRITICAL**
Admin mutation endpoints that ship without the auth dependency will stay unprotected. Prevention: place `dependencies=[Depends(verify_admin_key)]` on the admin `APIRouter` — impossible to add an unprotected endpoint by forgetting a `Depends()` on individual functions. Add a pytest that discovers all `/admin` routes and asserts they return 401/403 without credentials. Use `HttpOnly + Secure + SameSite=Strict` cookies for admin session token, not localStorage. CSRF double-submit cookie on all POST/PUT/DELETE admin endpoints.

**4. Cross-source event duplication breaking TKG and risk scores (C4) — CRITICAL**
The same real-world event (e.g., Syria battle 2026-02-15) appearing in GDELT, ACLED, and UCDP creates three separate triples in the knowledge graph. TiRGN's `history_rate` parameter weights by recent event frequency — duplicates inflate frequency and distort learned patterns. Country risk scores use event count as a signal — duplicates inflate risk. Prevention: cross-source dedup layer BEFORE knowledge graph insertion using `(event_date, country_iso, event_type_canonical, actor1_canonical)` fingerprint hash; source priority hierarchy ACLED > UCDP > GDELT (ACLED is human-verified); keep raw events from all sources in SQLite for audit, deduplicate only at graph insertion.

**5. Backtesting look-ahead bias from calibration weights (M2) — MODERATE**
Walk-forward evaluation that applies current per-CAMEO calibration weights to evaluate historical predictions uses future information to calibrate past predictions. Prevention: store calibration snapshots every time `WeightLoader` recalculates — new `calibration_snapshots` PostgreSQL table with timestamp. Backtesting loads the snapshot closest to (but not after) each prediction's timestamp. Alternative: backtest raw pre-calibration ensemble output and analyze calibration separately.

**6. RAG temporal contamination in backtesting (M3) — MODERATE**
ChromaDB contains ALL indexed articles with no temporal partitioning. RAG queries return semantically similar articles regardless of publication date, including articles from after the evaluation cutoff. Prevention for v3.0: backtest TKG-only predictions (skip RAG component entirely) to avoid temporal contamination. Future option: per-window ChromaDB index rebuild with `published_at <= T` filter.

**7. Polymarket cap tracking drift (M5) — MODERATE (already documented, must fix in Phase 21)**
`reforecast_active()` overwrites `created_at`, causing re-forecasts to count against the new-forecast cap. `increment_gemini_usage` is called at three scatter points, not atomically. Prevention: add `reforecasted_at` column to `Prediction` (never overwrite `created_at`); build `BudgetTracker` class that atomically increments and checks; daily reconciliation job comparing Redis counters against actual DB counts.

---

## Implications for Roadmap

The dependency chain is clear. Admin dashboard and daemon consolidation are the prerequisite foundation — every other feature registers new APScheduler jobs or adds admin UI panels. Source expansion and globe data wiring can proceed in parallel after consolidation. Global seeding depends on source expansion (more sources = better baseline). Backtesting depends on Polymarket hardening (needs reliable resolution data).

### Phase 19: Admin Dashboard

**Rationale:** No dependencies. Start immediately. Building the admin UI early means all subsequent phases get admin panels "for free" as new features are added. The admin route is the observation layer for everything that follows.

**Delivers:**
- `/admin` route with dynamic import code-split (zero bytes in public bundle)
- Auth gating inside `mountAdmin()` — `verify_admin_key` dependency on admin `APIRouter`
- `SchedulerPanel` — job list, status, pause/resume/trigger buttons (read-only until Phase 20 wires APScheduler)
- `SourceManagerPanel` — feed health grid with status dots, staleness detection
- Ingest run history table (last 50 runs, sortable)
- System config display (non-secret settings, read-only)
- Auth security: router-level dependency, HttpOnly cookie, CSRF on mutations, pytest coverage of all admin routes

**Addresses features:** Admin daemon status, manual trigger, ingest history (table stakes).

**Avoids pitfalls:** C3 (admin auth bypass) — router-level dependency, automated auth test.

**Research flag:** Standard patterns, no research-phase needed. UI: reuse WM `status-dot` + relative timestamp pattern (`formatTime`: "just now", "5m ago", "2:30 PM"). WM's `UnifiedSettings` three-tab pattern (General / Sources / Status) maps directly to admin screens.

---

### Phase 20: Daemon Consolidation

**Rationale:** Everything else registers APScheduler jobs. Build the scheduler first. Pause/resume buttons in Phase 19's `SchedulerPanel` become functional only when this phase wires `app.state.scheduler`.

**Delivers:**
- `src/scheduler/` module with `AsyncIOScheduler` in FastAPI lifespan
- All existing pollers refactored to single-cycle functions (`_poll_once`, `poll_feeds`) — loop management removed
- `_polymarket_loop` asyncio.Task replaced by APScheduler job
- All 8-9 existing jobs registered with correct `IntervalTrigger`/`CronTrigger`
- Admin scheduler API endpoints (`GET /jobs`, `POST /jobs/{id}/pause`, `/resume`, `/trigger`)
- `geopol-gdelt-poller.service` and `geopol-rss-daemon.service` retired
- Daily forecast systemd timer retired (replaced by `CronTrigger(hour=6)`)

**Implements architecture:** Single-process scheduler topology. `uvicorn --workers 1` documented as hard constraint.

**Avoids pitfalls:** C1 (process isolation) — `ProcessPoolExecutor` for daily forecast + TKG retraining; memory-monitoring job wrapper; circuit breaker after 3 failures. C2 (event loop starvation) — all sync work in `asyncio.to_thread()`, dedicated `ThreadPoolExecutor(max_workers=4)`. M6 (shutdown race) — shutdown order: `scheduler.shutdown(wait=True)` → Redis → PostgreSQL. N3 (multi-worker duplication) — documented constraint.

**Research flag:** APScheduler 3.11.2 integration is well-documented. The refactoring is mechanical (extract single-cycle from existing pollers). No research-phase needed. PRIMARY RISK: the `misfire_grace_time` + `coalesce=True` combination — test that the daily pipeline doesn't silently skip if the API restarts within the grace window.

---

### Phase 21: Polymarket Hardening

**Rationale:** Fix known reliability bugs before adding Brier score tracking. Tracking is meaningless if the underlying data (predictions, resolution, cap counts) is corrupt. Fix the foundation, then measure it.

**Delivers:**
- `reforecasted_at` column on `Prediction` — stop overwriting `created_at` on re-forecast
- `BudgetTracker` class — atomic increment + check, replaces 3 scattered `increment_gemini_usage` calls
- Daily budget reconciliation job — Redis counters vs DB counts, alerts on discrepancy
- Polymarket daemon migrated to APScheduler job (from Phase 20, but Polymarket-specific error handling here)
- Per-stage error reporting to `IngestRun` (was: broad `try/except` swallowing all stages)
- `brier_score_history` PostgreSQL table — cumulative Geopol vs Polymarket Brier score time series
- `BacktestingPanel` in admin: accuracy comparison table (Geopol vs Polymarket, win/loss count, head-to-head table)

**Addresses features:** Polymarket resolution tracking, Brier score per prediction, cumulative accuracy curve (table stakes for v3.0).

**Avoids pitfalls:** M5 (cap tracking drift), M7 (Brier score interpretation — require N>=30, show calibration plots alongside).

**Research flag:** No research-phase needed. The bugs are already identified in `auto_forecaster.py`. The fix is surgical.

---

### Phase 22: Backtesting

**Rationale:** Backtesting uses historical predictions and outcomes. More resolved predictions = more meaningful results. Running this after Polymarket hardening ensures the resolution data is clean. Also benefits from Phase 20 (scheduler provides the weekly backtesting cron job).

**Delivers:**
- `src/evaluation/backtest_engine.py` — `BacktestEngine` with read-only access to historical data
- `run_walk_forward()` — sliding window evaluation (configurable window/step size)
- `compare_models()` — TiRGN vs RE-GCN MRR, Hits@K, Brier on resolved predictions
- `calibration_audit()` — reliability diagrams per CAMEO category over time
- `polymarket_accuracy()` — cumulative Brier comparison (feeds `brier_score_history` from Phase 21)
- `backtest_reports` PostgreSQL table with `report_json` JSONB blobs
- `BacktestingPanel` upgrade — reliability diagram visualization, model comparison table
- Weekly APScheduler job (`CronTrigger(day_of_week='sun', hour=4)`)

**Addresses features:** Brier score over time, calibration curve, model comparison (table stakes for v3.0), per-CAMEO calibration audit (differentiator).

**Avoids pitfalls:** M2 (look-ahead bias) — calibration snapshots table; backtest against snapshot at prediction time. M3 (RAG contamination) — initial implementation backtests TKG-only predictions; RAG-inclusive backtesting is v3.1. M7 (Brier interpretation) — show calibration plots alongside scores, N>=30 gate.

**Research flag:** No research-phase needed. Walk-forward evaluation is standard ML methodology. The custom implementation is ~400 LOC using existing `BrierScorer` and `CalibrationMetrics` classes. The look-ahead bias fix (calibration snapshots) is the only non-trivial design decision.

---

### Phase 23: Source Expansion (UCDP + Feed Management)

**Rationale:** New data sources add event signal that improves baseline risk seeding (Phase 24) and eventually TKG training quality. UCDP first (safer integration, CAMEO-adjacent coding). POLECAT deferred pending PLOVER-to-CAMEO mapping validation.

**PRE-PHASE BLOCKER:** Email UCDP API team for access token before writing any code. Token procurement can take days to weeks.

**Delivers:**
- `src/ingest/ucdp_poller.py` — `UCDPPoller` with `x-ucdp-access-token` header, `aiohttp` (no new dependency), `ACLEDPoller` pattern
- UCDP-to-CAMEO type mapping dict (state-based conflict -> CAMEO 19, one-sided violence -> CAMEO 18, etc.)
- UCDP APScheduler job: `CronTrigger(hour=3)` daily
- `src/ingest/entity_resolver.py` — `CountryResolver` normalizing all sources to ISO alpha-2
- Cross-source dedup layer before knowledge graph insertion (fingerprint hash on date + country + event_type_canonical)
- `feed_overrides` PostgreSQL table — enable/disable/tier-override per feed name
- `feed_health` PostgreSQL table — per-feed consecutive failures, last success, auto-disable threshold
- Admin `SourceManagerPanel` upgrade: feed health grid, enable/disable toggles, staleness alerts
- WM feed validation logic ported as weekly scheduled job

**Addresses features:** UCDP source health monitoring (table stakes), per-source staleness alerts, feed enable/disable (differentiator).

**Avoids pitfalls:** M1 (UCDP auth blocker — pre-phase action). C4 (cross-source duplication — dedup layer). M8 (zombie feeds — feed_health table + auto-disable). N4 (ICEWS lag — note: UCDP also has publication lag; use `ingested_at` vs `event_date` tracking, exclude from real-time risk).

**Research flag:** POLECAT integration should have a dedicated spike before implementation. Specifically: download one sample weekly TSV from Harvard Dataverse, attempt PLOVER-to-CAMEO mapping on the top-20 event types, and assess precision loss. If >30% of events lose semantic meaning in translation, defer POLECAT to v4.0.

---

### Phase 24: Global Seeding

**Rationale:** Globe choropleth needs non-zero data for all ~195 countries. This phase wires the `BaselineRiskComputer` output to the countries API, merging baseline with forecast-derived risk.

**Delivers:**
- `src/evaluation/baseline_risk.py` — `BaselineRiskComputer` aggregating events per country
- Composite risk formula: GDELT event density (0.30) + ACLED event severity (0.25) + advisory level (0.20) + UCDP fatality rate (0.10) + FSI baseline normalized (0.15 — static CSV import, updated annually)
- `baseline_country_risk` PostgreSQL table
- Advisory-level risk floor: Do Not Travel → 70 minimum, Reconsider Travel → 50, etc.
- Active forecast override: `COALESCE(forecast_risk, baseline_risk)` in countries API query
- APScheduler job: `CronTrigger(hour='*/6')` every 6 hours
- Risk trend arrows: 7-day delta comparison per country

**Addresses features:** Baseline risk for all countries (table stakes), active forecast override (differentiator), risk trend (differentiator).

**Avoids pitfalls:** M4 (GDELT coverage bias) — ACLED is primary signal for conflict, GDELT is secondary; UCDP minimum floor for countries with active armed conflicts (UCDP says conflict -> minimum risk score 60 regardless of GDELT count); travel advisories as calibration signal.

**Research flag:** FSI (Fragile States Index) data import needs a one-time data pull and normalization script. Published annually in CSV format at fragilestatesindex.org. Straightforward but needs sourcing before implementation.

---

### Phase 25: Globe Data Wiring

**Rationale:** Three of five DeckGL layers (KnowledgeGraphArcs, GDELTEventHeatmap, ScenarioZones) are no-ops because data is never pushed. This phase wires them. Benefits from Phase 23 (more event sources = denser heatmap). Can ship Heatmap and Relations with GDELT-only data; full multi-source density comes after Phase 23.

**Delivers:**
- SQLite schema change: `lat REAL`, `lon REAL` columns in `events` table
- `_gdelt_row_to_event()` modified to extract `ActionGeo_Lat`/`ActionGeo_Long`
- `GET /api/v1/events/heatmap?days=7` — server-side 0.5-degree grid aggregation (50K raw events → ~2K grid cells)
- `GET /api/v1/countries/{iso}/relations` — country-to-country event relationships from GDELT actor codes
- `DeckGLMap.updateHeatmap(points)` and `DeckGLMap.updateArcs(relations)` methods
- Globe screen refresh scheduler integration: heatmap every 5 minutes, arcs on country selection
- All 3 globe layer pills now render actual data

**Addresses features:** Arcs layer, heatmap layer, scenario zones (all table stakes for a useful globe view).

**Avoids pitfalls:** Anti-pattern 3 (real-time heatmap via WebSocket) — 5-minute polling with pre-aggregated grid cells. Heatmap performance with 100K+ events — server-side aggregation is mandatory, not optional.

**Research flag:** No research-phase needed. The data exists in GDELT. The wiring is mechanical. The only non-trivial decision is grid cell size (0.5 degrees is a reasonable default; can tune after observing visual density).

---

### Phase Ordering Rationale

**Why this order:**

1. **Admin Dashboard first (Phase 19)** — No dependencies. Establishes the observation layer. Every subsequent phase adds panels to an already-working admin screen.

2. **Daemon Consolidation second (Phase 20)** — Makes the admin scheduler controls functional. All subsequent features register as APScheduler jobs — build the scheduler before writing jobs that depend on it.

3. **Polymarket Hardening before Backtesting (Phase 21 before 22)** — Backtesting needs clean resolution data. The known cap-tracking bugs in Phase 21 corrupt exactly the data that backtesting analyzes. Fix before measuring.

4. **Source Expansion before Global Seeding (Phase 23 before 24)** — Global seeding's composite formula includes UCDP fatality data. Can start Phase 24 with GDELT+ACLED only, but UCDP improves it significantly. Preferred order: build the sources, then build the consumer.

5. **Globe Data Wiring last (Phase 25)** — Depends on: SQLite lat/lon schema (can be done earlier, but the heatmap density is better after Phase 23 adds more event sources). Arcs benefit from Phase 24's country risk context. No strong blocker, but maximum value when other phases complete.

**Two independent tracks after Phase 20:**
- Track A: Phases 21-22 (Polymarket + Backtesting) — accuracy verification track
- Track B: Phases 23-25 (Source Expansion + Seeding + Globe) — data enrichment track

These tracks are independent after Phase 20 and can be built in parallel by separate contributors if that's desirable.

### Research Flags

**Phases likely needing deeper research or spikes before planning:**

- **Phase 23 (Source Expansion):** POLECAT/PLOVER-to-CAMEO mapping feasibility needs a dedicated spike (download sample TSV, attempt mapping, measure precision loss) before deciding whether to implement POLECAT in v3.0 or defer to v3.1. The UCDP portion is well-understood and needs no additional research.

- **Phase 22 (Backtesting) — calibration snapshots design:** The look-ahead bias fix requires `calibration_snapshots` table design. The schema is non-trivial: must store the full `WeightLoader` state (all per-CAMEO alpha weights + global alpha) per timestamp. Worth a planning-phase design review before implementation.

**Phases with standard patterns (skip research-phase):**

- **Phase 19 (Admin Dashboard):** TypeScript route + dynamic import + auth check. Well-documented pattern, implemented for DeckGL in v2.1.
- **Phase 20 (Daemon Consolidation):** APScheduler 3.11.2 integration is thoroughly documented. The refactoring is mechanical.
- **Phase 21 (Polymarket Hardening):** Bugs are identified, fixes are straightforward.
- **Phase 24 (Global Seeding):** Composite risk formula is defined in FEATURES.md with clear weights. FSI CSV import is a one-time data pull. SQL `COALESCE` merge is trivial.
- **Phase 25 (Globe Data Wiring):** Mechanical wiring of existing data to existing DeckGL layers.

---

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | APScheduler 3.11.2 verified from PyPI + readthedocs. APScheduler 4.0 alpha status verified from author's own documentation. ICEWS discontinuation verified from Dataverse. UCDP auth change verified from API docs (Feb 2026). POLECAT format verified from Dataverse + blog analysis. Two new dependencies only — confidence is high because scope is minimal. |
| Features | MEDIUM-HIGH | Admin dashboard patterns: HIGH (WM codebase examined directly, MLflow patterns surveyed). Backtesting: MEDIUM (Metaculus/Brier.fyi patterns documented but walk-forward implementation for probabilistic event forecasting has no reference). Global seeding composite formula weights: MEDIUM (FSI methodology published, weight assignments are informed estimates not measured empirically). |
| Architecture | HIGH | Based on direct codebase analysis of all relevant files (app.py, pollers, admin auth middleware, DeckGLMap.ts, globe-screen.ts). Integration patterns are extensions of existing code, not novel designs. APScheduler + FastAPI lifespan integration is a documented pattern from Sentry, Medium, and APScheduler GitHub discussions. |
| Pitfalls | HIGH | All critical pitfalls verified: C1-C2 against APScheduler GitHub issues (#600, #235, #484, #304). C3 against existing auth.py and FastAPI security docs. C4 against ACLED comparison paper and existing poller code. M2 against calibration code (per-CAMEO weights verified in Phase 13 implementation). M3 against RAG pipeline code (ChromaDB usage verified). M5 against direct `auto_forecaster.py` code analysis (bugs confirmed, line numbers cited). |

**Overall confidence:** MEDIUM-HIGH

Research is strong on stack decisions (verified against official sources) and architecture (derived from codebase analysis). The primary uncertainty areas:

1. **PLOVER-to-CAMEO mapping feasibility** — no published mapping exists. Assessed from ontology structure comparison only. This uncertainty gates the POLECAT integration decision. Resolution: run the POLECAT spike in Phase 23 planning before committing.

2. **Walk-forward backtesting with TKG retraining** — per-window TKG retraining is computationally expensive (training time unknown for the current dataset size post-v2.x). Initial v3.0 implementation uses static weights (existing model) for backtesting, sidestepping this uncertainty. Full walk-forward retraining is v3.1.

3. **Cross-source deduplication precision** — the fingerprint hash approach (date + country + event_type_canonical) will have false negatives (events that are truly duplicates but don't match exactly) and possibly false positives (coincidental same-day same-country different events). The actual error rate is unknown until UCDP data is ingested and compared. Risk mitigation: keep all raw events from all sources; dedup only at graph insertion; audit by querying (SELECT date, country, COUNT(DISTINCT source) FROM events WHERE date='...' AND country='...'` and inspecting samples).

4. **Baseline risk composite weights** — the formula (GDELT 0.30, ACLED 0.25, advisory 0.20, FSI 0.15, UCDP 0.10) is informed by FSI methodology but the specific weights are not empirically validated against ground truth. Start with these values, track against travel advisory ground truth during Phase 24, recalibrate in v3.1.

### Gaps to Address During Planning

- **Gap 1: UCDP token procurement** — must happen before Phase 23. Not a planning gap, but a calendar blocker. Assign someone to email the UCDP team immediately. If token doesn't arrive before Phase 23 starts, implement bulk CSV fallback first (datasets at ucdp.uu.se/downloads/).

- **Gap 2: POLECAT spike** — decide POLECAT inclusion/exclusion before Phase 23 planning. The spike is: download one weekly TSV, map 20 PLOVER event types to CAMEO, measure how many events lose semantic meaning. Expected outcome: implement UCDP in Phase 23, defer POLECAT to v3.1.

- **Gap 3: Calibration snapshots schema** — design the `calibration_snapshots` table before Phase 22 planning. Must store: timestamp, full per-CAMEO alpha weight vector, global fallback alpha, sample counts per category. The `WeightLoader` class shape determines what needs to be serialized.

- **Gap 4: FSI static baseline** — before Phase 24, download the latest FSI CSV (fragilestatesindex.org), normalize to ISO alpha-2 + 0-100 scale, commit as `data/fsi_baseline.csv`. This is a one-time manual step, not a polling job.

---

## Sources

### Primary — HIGH Confidence (directly verified)

**Codebase analysis:**
- `/home/kondraki/personal/geopol/src/api/app.py` — lifespan management, Polymarket loop integration
- `/home/kondraki/personal/geopol/src/ingest/gdelt_poller.py` — poll_once structure, ZIP parsing, backoff
- `/home/kondraki/personal/geopol/src/ingest/rss_daemon.py` — tiered polling, executor usage
- `/home/kondraki/personal/geopol/src/ingest/acled_poller.py` — template for UCDPPoller
- `/home/kondraki/personal/geopol/src/polymarket/auto_forecaster.py` — cap tracking bugs (lines cited in M5)
- `/home/kondraki/personal/geopol/src/api/middleware/auth.py` — existing API key auth pattern
- `/home/kondraki/personal/geopol/deploy/systemd/*.service` — current process isolation configuration
- `/home/kondraki/personal/geopol/src/ingest/feed_config.py` — existing FeedSource/FeedTier model
- `/home/kondraki/personal/worldmonitor/src/components/UnifiedSettings.ts` — admin UI patterns (three-tab)
- `/home/kondraki/personal/worldmonitor/src/config/feeds.ts` — 4-tier source system, region mapping
- `/home/kondraki/personal/worldmonitor/scripts/validate-rss-feeds.mjs` — feed health validation approach
- `/home/kondraki/personal/worldmonitor/server/worldmonitor/conflict/v1/list-ucdp-events.ts` — UCDP with Redis caching (25h TTL)

**Official documentation:**
- [APScheduler 3.x User Guide](https://apscheduler.readthedocs.io/en/3.x/userguide.html) — AsyncIOScheduler, SQLAlchemyJobStore, pause/resume
- [APScheduler Migration Guide](https://apscheduler.readthedocs.io/en/master/migration.html) — 4.0 breaking changes
- [APScheduler PyPI](https://pypi.org/project/APScheduler/) — 3.11.2 stable (Dec 2025), 4.0.0a6 alpha (Apr 2025)
- [UCDP API Documentation](https://ucdp.uu.se/apidocs/) — endpoints, auth, rate limits, pagination
- [Fragile States Index Methodology](https://fragilestatesindex.org/methodology/) — 12-indicator composite score

**APScheduler GitHub Issues:**
- [Issue #600](https://github.com/agronholm/apscheduler/issues/600) — memory not cleared in Docker
- [Issue #235](https://github.com/agronholm/apscheduler/issues/235) — memory leak on worker exception
- [Issue #484](https://github.com/agronholm/apscheduler/issues/484) — AsyncIOScheduler must start after event loop
- [Issue #304](https://github.com/agronholm/apscheduler/issues/304) — AsyncIOScheduler ThreadPoolExecutor default

### Secondary — MEDIUM Confidence (community consensus, multiple sources)

- [POLECAT Weekly Data (Harvard Dataverse)](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/AJGVIT) — dataset location, format
- [PLOVER Ontology (GitHub)](https://github.com/openeventdata/PLOVER) — event type structure
- [POLECAT Event Data Analysis (andybeger.com)](https://www.andybeger.com/blog/2024-05-21-polecat-event-data/) — format quirks (TSV, excess columns)
- [Brier.fyi About](https://brier.fyi/about/) — cross-platform prediction accuracy methodology
- [Polymarket Accuracy](https://polymarket.com/accuracy) — Brier score 0.058 at 12h benchmark
- [Metaculus Track Record](https://www.metaculus.com/questions/track-record/) — calibration curve visualization patterns
- [ONS GDELT Data Quality Note](https://www.ons.gov.uk/...) — 55% accuracy, English-language bias
- [ACLED Comparison Analysis](https://acleddata.com/...) — cross-source discrepancy rates
- APScheduler + FastAPI integration patterns: [Sentry Guide](https://sentry.io/answers/schedule-tasks-with-fastapi/), [Medium Guide](https://ahaw021.medium.com/...)

### Tertiary — LOW Confidence (single source or inference)

- PLOVER-to-CAMEO mapping feasibility — no published mapping; assessed from ontology structure comparison
- Baseline risk composite formula weights — informed by FSI methodology, not empirically validated against ground truth
- Cross-source dedup false positive/negative rates — theoretical; actual rates unknown until UCDP data is ingested
- pyDataverse v0.3.1 stability — pre-1.0; API quirks possible on large file downloads

---

*Research completed: 2026-03-04*
*Ready for roadmap: YES*
