# Pitfalls Research: v3.0 Operational Command & Verification

**Domain:** Adding admin dashboard, daemon consolidation, source expansion, backtesting, global seeding, Polymarket hardening to existing geopolitical forecasting engine
**Researched:** 2026-03-04
**Confidence:** HIGH (verified against codebase analysis, APScheduler GitHub issues, Polymarket API docs, WM codebase patterns, GDELT/UCDP data quality research)

## Executive Summary

v3.0 adds operational infrastructure to a system that already works but runs fragile. The pitfalls are fundamentally different from v2.0: instead of "make it work publicly" risks, v3.0 faces "consolidate without breaking what already works" risks. Seven categories:

1. **Daemon consolidation pitfalls** that crash the entire system when one job fails (process isolation loss, memory accumulation, event loop starvation)
2. **Admin dashboard security pitfalls** that expose control surfaces to unauthenticated users (route-level auth bypass, CSRF on mutation endpoints, admin bundle leaking to public)
3. **Source expansion pitfalls** that produce duplicate/contradictory events and break entity resolution (GDELT vs ACLED vs ICEWS vs UCDP schema mismatch, deduplication across sources, coverage bias compounding)
4. **Backtesting validity pitfalls** that produce misleading accuracy metrics (look-ahead bias from calibration data, temporal leakage in walk-forward splits, survivorship bias in question selection)
5. **Global seeding pitfalls** that produce nonsensical risk scores for sparse-data countries (GDELT English-language coverage bias, composite score normalization artifacts, denominator-zero edge cases)
6. **Polymarket reliability pitfalls** that silently degrade forecast quality tracking (cap tracking inconsistency already observed, API schema drift, Brier score interpretation traps)
7. **APScheduler + FastAPI integration pitfalls** that cause event loop deadlocks and memory leaks (AsyncIOScheduler initialization timing, blocking jobs starving HTTP handlers, graceful shutdown races)

The single most dangerous pattern: **collapsing 4 isolated processes into 1 APScheduler process means a memory leak, OOM, or event loop block in any job takes down the entire system** -- including the API that serves the frontend.

---

## Critical Pitfalls (Block Progress or Cause Outages)

### C1: Loss of Process Isolation During Daemon Consolidation

**What goes wrong:** Currently, GDELT poller, RSS daemon, daily forecast pipeline, and Polymarket loop run as separate processes (3 systemd units + 1 asyncio.create_task in app.py). Consolidating into a single APScheduler process means:
- A memory leak in RSS feed parsing (trafilatura, feedparser) accumulates indefinitely instead of being cleaned up by process restart
- An OOM kill takes down the API server, not just the ingest daemon
- A blocking `asyncio.to_thread()` call (e.g., TKG predictor inference) can starve HTTP request handling

**Why it happens:** The current systemd services have `MemoryMax=512M` and `Restart=on-failure` -- systemd isolates failures. APScheduler in-process has no such isolation. The existing `_polymarket_loop` in `app.py` already demonstrates this risk: it runs as an asyncio task inside the API process with only `try/except` guarding.

**Existing evidence from codebase:**
- `src/ingest/rss_daemon.py` uses `loop.run_in_executor(None, extract_article_text, url)` -- trafilatura is synchronous and can hang on malformed HTML
- `src/ingest/gdelt_poller.py` downloads ZIP files up to several MB and does pandas DataFrame parsing -- memory-intensive
- `scripts/daily_forecast.py` initializes `TKGPredictor`, `GeminiClient`, `RAGPipeline` -- heavy objects created per run
- `src/polymarket/auto_forecaster.py` creates `EnsemblePredictor()` fresh per prediction (line 413) -- intentional but memory-hungry

**Consequences:** If consolidated naively, a single stuck trafilatura extraction or GDELT download OOM kills the API, dashboard, and all scheduled jobs simultaneously. No graceful degradation.

**Prevention:**
1. Use `max_instances=1` on every APScheduler job to prevent overlapping runs
2. Wrap each job function in a memory-monitoring decorator that logs RSS before/after and aborts if delta exceeds threshold
3. Run heavy jobs (daily forecast, TKG retraining) via `ProcessPoolExecutor` instead of `ThreadPoolExecutor` to get OS-level memory isolation
4. Set `misfire_grace_time` appropriately per job (300s for polling, 3600s for daily forecast) -- APScheduler's default creates backlog calculations that themselves leak memory (see [litellm PR #15846](https://github.com/BerriAI/litellm/pull/15846))
5. Implement a circuit breaker on the scheduler itself: if 3 consecutive runs of any job fail, pause that job and alert via admin dashboard

**Detection:** Monitor process RSS memory over time. If it grows monotonically across job cycles without plateau, you have a leak. Check `psutil.Process().memory_info().rss` in each job wrapper.

**Phase:** Phase 20 (daemon consolidation)

**Confidence:** HIGH -- verified against APScheduler GitHub issues [#600](https://github.com/agronholm/apscheduler/issues/600), [#235](https://github.com/agronholm/apscheduler/issues/235), [#223](https://github.com/agronholm/apscheduler/issues/223) and existing codebase patterns.

---

### C2: AsyncIOScheduler + FastAPI Event Loop Starvation

**What goes wrong:** APScheduler's `AsyncIOScheduler` runs jobs on the same event loop as FastAPI's ASGI server. If a scheduled job blocks the event loop (even briefly via a forgotten `await` or a synchronous call not offloaded to `to_thread`), all HTTP request handling stalls.

**Why it happens:** The existing codebase has multiple patterns that block the event loop:
- `feedparser.parse(body)` in `rss_daemon.py` is already offloaded to `run_in_executor` (correct)
- `EventStorage.insert_events()` is synchronous SQLite (offloaded via `asyncio.to_thread` -- correct)
- `EnsemblePredictor.predict()` is synchronous (offloaded via `asyncio.to_thread` -- correct)
- BUT: `GeminiClient()` constructor and `init_db()` are synchronous and called in job setup code, not in the thread-offloaded section

The trap is that APScheduler with `AsyncIOExecutor` expects async job functions but does NOT automatically offload blocking code. If you define `async def gdelt_poll_job()` and call a sync function without `to_thread`, the event loop blocks.

**Existing evidence:**
- `src/api/app.py` line 140: `gemini_client = GeminiClient()` is called synchronously inside `_polymarket_loop` -- this works because `_polymarket_loop` has `await asyncio.sleep(30)` before it, but in APScheduler there's no such buffer
- APScheduler issue [#484](https://github.com/agronholm/apscheduler/issues/484): `AsyncIOScheduler` must be started AFTER the event loop is running
- APScheduler issue [#304](https://github.com/agronholm/apscheduler/issues/304): `AsyncIOScheduler` defaults to `ThreadPoolExecutor` for sync jobs, but the executor is shared

**Consequences:** API becomes unresponsive during job execution. Dashboard shows loading spinners. Health checks fail. Reverse proxy marks the backend as down.

**Prevention:**
1. Initialize the scheduler in FastAPI's `_lifespan` context manager AFTER `yield` is not an option -- initialize BEFORE yield but start AFTER event loop is running (use `scheduler.start()` inside lifespan)
2. ALL job functions MUST be async and MUST offload sync work to `asyncio.to_thread()` -- enforce this with a decorator or linter rule
3. Set `job_defaults={'coalesce': True, 'max_instances': 1}` on the scheduler to prevent job pile-up
4. Add a dedicated `ThreadPoolExecutor` with bounded workers (e.g., 4) for sync job offloading -- don't share with FastAPI's default executor
5. Use `replace_existing=True` when adding jobs to prevent duplicates on FastAPI reload (uvicorn `--reload`)

**Detection:** Monitor event loop lag. Add a periodic heartbeat that measures `asyncio.get_event_loop().time()` delta. If lag exceeds 100ms, a job is blocking.

**Phase:** Phase 20 (daemon consolidation)

**Confidence:** HIGH -- verified against APScheduler issues and existing `app.py` patterns.

---

### C3: Admin Route Exposure Without Authentication

**What goes wrong:** The v3.0 plan places admin routes at `/admin` in the same FastAPI app. The existing auth is API key-based (`X-API-Key` header via `verify_api_key` dependency). But:
- The frontend is a TypeScript SPA that doesn't send API keys (it's a public-facing dashboard)
- Adding `/admin` routes to the same router means they're accessible from the same origin
- If admin routes use cookie-based auth (for browser access) while public routes use API keys, the auth model becomes split and error-prone
- CSRF attacks against admin mutation endpoints (pause job, add feed, trigger reforecast) if using cookies

**Existing auth architecture:**
- `src/api/middleware/auth.py`: `verify_api_key()` checks `X-API-Key` header against PostgreSQL `api_keys` table
- `src/api/routes/v1/router.py`: routes are under `/api/v1` with selective auth dependency application
- Frontend at `/dashboard`, `/globe`, `/forecasts` -- no auth required
- No session management, no cookie auth, no JWT

**Why it's dangerous:** Adding admin routes means adding a second auth mechanism (session/JWT for browser-based admin) to a system that only has one (API key for programmatic access). The likely mistake is:
1. Add admin routes under `/api/v1/admin/` with `verify_api_key` dependency
2. Realize the admin SPA can't send API keys from the browser conveniently
3. Add cookie/JWT auth as a second mechanism
4. Forget to apply it consistently -- leave one admin endpoint unprotected
5. Or worse: make admin routes "temporarily" unprotected during development and forget to add auth

**Prevention:**
1. Admin auth MUST be a separate dependency (`verify_admin_session`) that is applied at the router level, not per-endpoint -- use `dependencies=[Depends(verify_admin_session)]` on the admin `APIRouter`
2. Use environment variable (`ADMIN_PASSWORD_HASH`) for a single admin user -- no need for multi-user admin on a single-server system
3. Admin login returns a signed JWT stored in an HttpOnly, Secure, SameSite=Strict cookie -- no localStorage JWT
4. All admin mutation endpoints (POST/PUT/DELETE) must validate a CSRF token (double-submit cookie pattern)
5. Admin routes under a separate prefix (`/api/admin/`) with a separate router instance that has the auth dependency at the router level -- impossible to add an unprotected endpoint by forgetting `Depends()`
6. Test: write a pytest that hits every admin endpoint without auth and asserts 401

**Detection:** Automated test that discovers all admin routes and verifies they return 401/403 without credentials.

**Phase:** Phase 19 (admin dashboard)

**Confidence:** HIGH -- verified against existing `auth.py` and FastAPI security docs.

---

### C4: Entity Resolution Chaos Across 4+ Data Sources

**What goes wrong:** Adding ICEWS and UCDP alongside existing GDELT and ACLED means the same real-world event can appear in multiple sources with different:
- Entity names: "Russian Federation" (UCDP) vs "RUS" (GDELT Actor1Code) vs "Russia" (ACLED) vs "RUSSIA" (ICEWS)
- Event coding: CAMEO codes (GDELT/ICEWS) vs UCDP's own violence typology vs ACLED's event_type taxonomy
- Temporal granularity: GDELT has 15-minute updates, ACLED is weekly, UCDP is monthly/yearly, ICEWS is daily
- Geographic precision: GDELT gives lat/lon per event, UCDP may give only admin1 region

When the knowledge graph ingests the "same" event from 4 sources, it creates 4 separate triples. TKG training sees them as 4 independent events, inflating event counts and distorting learned patterns.

**Why it's especially dangerous for this system:**
- The TiRGN model's `history_rate` parameter weights recent event frequency -- duplicate events inflate frequency
- Country risk scoring uses event counts (`4-CTE SQL query, 0-100 composite score (count + probability + severity + decay)` from memory) -- duplicates inflate risk
- Calibration data gets polluted: if the same event resolves as "happened" from GDELT and separately from ACLED, it counts twice in Brier score calculations

**Prevention:**
1. Cross-source deduplication layer that runs BEFORE knowledge graph insertion
2. Canonical entity mapping table: map all source-specific actor codes to a shared entity ID (use Wikidata QIDs or ISO country codes as canonical keys)
3. Event fingerprinting: hash (date, country, event_type_canonical, actor1_canonical, actor2_canonical) to detect duplicates across sources
4. Source priority hierarchy for duplicates: ACLED > UCDP > GDELT (ACLED has human-verified coding; GDELT is automated)
5. Add `source` column to Event model (already exists -- `source="gdelt"` in `_gdelt_row_to_event`) and ensure all sources populate it
6. Start with ACLED + UCDP as read-through supplements (query-time enrichment, not ingest-time duplication), then graduate to full ingest after dedup is proven

**WM lesson learned:** World Monitor's `list-acled-events.ts` and `list-ucdp-events.ts` keep sources completely separate -- they don't attempt cross-source dedup. This works for a display system but NOT for a forecasting system that needs consistent event counts.

**Phase:** Phase 22 (source expansion)

**Confidence:** HIGH -- verified against ACLED comparison paper, UCDP API docs, and existing `_gdelt_row_to_event` code.

---

## Moderate Pitfalls (Cause Delays or Technical Debt)

### M1: UCDP API Authentication Change (February 2026)

**What goes wrong:** UCDP introduced token-based authentication in February 2026. The API now requires contacting the UCDP team to request an access token. This is a manual, human-in-the-loop process -- cannot be automated.

**Why it matters:** If the implementation plan assumes UCDP is a freely accessible REST API (like GDELT), the team will hit a blocker when they try to integrate. The token request process may take days or weeks.

**Prevention:**
1. Request UCDP API token BEFORE starting Phase 22 implementation
2. Fall back to UCDP bulk CSV downloads if token access is delayed (datasets available at ucdp.uu.se/downloads/)
3. Abstract the UCDP client behind an interface so bulk-CSV and API-based implementations are swappable

**Detection:** Try `curl https://ucdp.uu.se/api/gedevents/24.1` -- if it returns 401, you need a token.

**Phase:** Phase 22 (source expansion) -- BLOCKER if not addressed pre-phase

**Confidence:** MEDIUM -- based on WebSearch finding; verify by attempting an unauthenticated API call.

---

### M2: Backtesting Look-Ahead Bias from Calibration Weights

**What goes wrong:** The backtesting system evaluates historical forecast accuracy. But the current system uses per-CAMEO dynamic calibration weights (`WeightLoader` with L-BFGS-B optimization). If backtesting uses current calibration weights to evaluate historical predictions, it's using future information to calibrate past predictions -- textbook look-ahead bias.

**Concrete example:**
- Jan 2026: System makes prediction with `alpha=0.6` (fixed weight at that time)
- Feb 2026: Per-CAMEO calibration shipped, `alpha` for conflict events becomes `0.72`
- Mar 2026: Backtest evaluates Jan 2026 prediction using current `0.72` weight -- prediction quality appears different than it actually was

**Prevention:**
1. Walk-forward evaluation MUST reconstruct the calibration state that existed at the time of each prediction
2. Store calibration snapshots: every time `WeightLoader` recalculates weights, persist the full weight vector with timestamp to a `calibration_snapshots` table
3. Backtesting framework must load calibration weights from the snapshot closest to (but not after) each prediction timestamp
4. Alternative: backtest only the raw ensemble output (pre-calibration) and apply calibration analysis separately -- simpler but less representative

**Detection:** If backtested Brier scores are significantly better than live Brier scores, suspect look-ahead bias.

**Phase:** Phase 21 (backtesting)

**Confidence:** HIGH -- this is a well-known problem in quantitative forecasting. The calibration system was shipped in v2.0 Phase 13.

---

### M3: Walk-Forward Split Contamination via RAG Pipeline

**What goes wrong:** Walk-forward evaluation splits data temporally: train on events before time T, evaluate predictions at time T. But the RAG pipeline (`src/forecasting/rag_pipeline.py`) retrieves from ChromaDB, which contains ALL indexed articles -- including articles published after time T.

**Why it happens:** ChromaDB is a single collection with no temporal partitioning. The RAG pipeline queries by semantic similarity, not by publication date. Even if you filter by `published_before` metadata, the embedding model has been trained/fine-tuned on the full corpus.

**Consequences:** Backtested predictions get information from "future" articles, making them appear more accurate than they would have been at the time.

**Prevention:**
1. For backtesting, build a separate ChromaDB index per time window, containing only articles published before the evaluation timestamp
2. Or: add strict `published_at <= T` filter to RAG queries during backtesting and verify it's actually filtering correctly (test with a known future article)
3. Simplest approach: backtest TKG-only predictions (skip RAG/LLM component entirely) -- this tests the core model without temporal contamination

**Detection:** Check if RAG-augmented backtested predictions are suspiciously better than TKG-only predictions -- the improvement should be consistent with live performance, not dramatically better.

**Phase:** Phase 21 (backtesting)

**Confidence:** HIGH -- verified against RAG pipeline code and ChromaDB usage patterns in the codebase.

---

### M4: GDELT Coverage Bias Corrupting Global Seeding Risk Scores

**What goes wrong:** Global seeding computes baseline risk for ~195 countries using GDELT event density. But GDELT has documented coverage bias:
- GDELT processes English-language media primarily (claims 98.4% translation coverage, but coded event accuracy is ~55% per ONS data quality note)
- US, UK, and Western European events are massively overrepresented
- Countries with low English-media coverage (Central Asia, Pacific Islands, Central Africa) show artificially low event counts
- This creates a systematic bias where Western-aligned countries appear more "at risk" simply because they have more media coverage

**Concrete impact:**
- Chad may have active armed conflict but low GDELT event count (few English-language sources covering it)
- UK may have high GDELT event count from routine political coverage (Brexit debates, elections) but low actual conflict risk
- The composite risk score would rank UK higher than Chad -- objectively wrong

**Prevention:**
1. ACLED data (human-verified, covers Africa and Asia comprehensively) should be the primary signal for conflict risk, not GDELT
2. Normalize GDELT event counts by expected media coverage per country (use GDELT's own source-country crossreferencing dataset)
3. Use UCDP as ground truth for active armed conflicts -- if UCDP says there's a conflict, the risk score must reflect it regardless of GDELT event count
4. Implement minimum risk floors: countries with active UCDP conflicts get minimum risk score of 60 regardless of GDELT data
5. Separate "media attention" from "actual risk" in the composite score -- report both, but rank by risk
6. Travel advisories (already ingested via `advisory_poller.py`) should be a calibration signal: if State Dept says "Do Not Travel" but risk score is low, something is wrong

**WM lesson learned:** WM's `get-risk-scores.ts` in the intelligence module already handles this by combining multiple signals. The key insight: never derive risk from a single automated source.

**Phase:** Phase 23 (global seeding)

**Confidence:** HIGH -- verified against ONS data quality note, GDELT coverage research, and ACLED comparison analysis.

---

### M5: Polymarket Cap Tracking Inconsistency (Already Observed)

**What goes wrong:** The existing Polymarket auto-forecaster has known reliability issues documented in project memory: "sometimes fails silently, cap tracking inconsistent." The specific mechanisms:

1. `count_today_reforecasts()` in `auto_forecaster.py` (lines 190-213) uses `Prediction.created_at` to count re-forecasts, but `reforecast_active()` (line 601) overwrites `created_at` with `datetime.now()` -- so a prediction created yesterday and re-forecasted today looks like it was created today, inflating the "new forecast" count
2. The daily cap split (`3 new + 5 reforecast`) is checked against `count_today_new_forecasts()` which counts `provenance='polymarket_driven'` created today -- but re-forecasts of `polymarket_driven` predictions also update `created_at`, making them count against the new forecast cap
3. Gemini budget checks happen at two points (lines 373 and 401) but `increment_gemini_usage` is called at three points (lines 389, 393, 456) -- the budget increment for country extraction LLM call happens even when the subsequent predict() call is skipped due to budget exhaustion

**Consequences:** Budget tracking drifts from reality. Some days produce 0 forecasts when budget is available. Other days exceed the intended cap because re-forecasts are miscounted as new forecasts.

**Prevention:**
1. Add a `reforecasted_at` column to Prediction -- do NOT overwrite `created_at` on re-forecast
2. Use `reforecasted_at IS NOT NULL AND reforecasted_at >= today_start` for re-forecast counting
3. Unify budget tracking into a single `BudgetTracker` class that atomically increments and checks -- no scattered `increment_gemini_usage` calls
4. Add a daily budget reconciliation job that compares Redis budget counters against actual DB prediction counts and logs discrepancies
5. Add `provenance_detail` to distinguish "polymarket_driven_new" from "polymarket_driven_reforecast"

**Detection:** Query: `SELECT DATE(created_at), provenance, COUNT(*) FROM predictions WHERE provenance LIKE 'polymarket%' GROUP BY 1, 2 ORDER BY 1 DESC` -- look for days with suspiciously high or zero counts.

**Phase:** Phase 24 (Polymarket hardening)

**Confidence:** HIGH -- verified directly from codebase analysis of `auto_forecaster.py`.

---

### M6: APScheduler Graceful Shutdown Race with FastAPI

**What goes wrong:** When uvicorn receives SIGTERM, FastAPI's lifespan shutdown runs. If the scheduler is shutting down while a job is mid-execution, the job's database session/Redis connection may be torn out from under it.

**Existing pattern:** The current `_polymarket_loop` shutdown (app.py lines 81-86) cancels the task and catches `CancelledError`. But APScheduler's `scheduler.shutdown(wait=True)` blocks until running jobs complete -- which can deadlock if a job is waiting on a database session that's already been closed by `close_db()` in the lifespan shutdown.

**Shutdown order matters:**
```
BAD:  close_db() -> scheduler.shutdown(wait=True)  # job hangs waiting for DB
BAD:  scheduler.shutdown(wait=False) -> close_db()  # job crashes mid-transaction
GOOD: scheduler.shutdown(wait=True) -> close_redis() -> close_db()
```

**Prevention:**
1. Shutdown order: scheduler first (with timeout), then Redis, then PostgreSQL
2. Set `scheduler.shutdown(wait=True)` with a timeout wrapper: `asyncio.wait_for(scheduler.shutdown(), timeout=30)`
3. Each job function should check a shutdown flag and exit early if set
4. Jobs must use their own session factory calls, not rely on module-level state that may be torn down

**Detection:** SIGTERM followed by hanging process that never exits. Check for `TimeoutStopSec` kills in systemd journal.

**Phase:** Phase 20 (daemon consolidation)

**Confidence:** HIGH -- verified against FastAPI lifespan shutdown order in `app.py`.

---

### M7: Brier Score Interpretation Traps in Polymarket Comparison

**What goes wrong:** v3.0 plans "rigorous Brier score tracking (Geopol vs market cumulative accuracy curve)." But comparing Brier scores between Geopol and Polymarket has fundamental methodological traps:

1. **Temporal mismatch:** Geopol predicts once (or re-forecasts daily). Polymarket prices update continuously. Comparing Geopol's prediction at time T with Polymarket's price at time T is valid, but which T? The snapshot closest to resolution? The earliest prediction? The time-weighted average?

2. **Selection bias:** Geopol only generates forecasts for high-volume Polymarket questions (volume threshold). These are questions the market has strong opinions on -- precisely the questions where market prices are most efficient. Comparing against efficient markets makes Geopol look worse than it would on less-liquid questions.

3. **Resolution ambiguity:** Polymarket resolves binary markets as 0 or 1. But many geopolitical questions have ambiguous outcomes. If Geopol predicted "70% chance of ceasefire" and a temporary ceasefire was declared then broken, did it happen? Polymarket may resolve differently than Geopol's outcome resolver.

4. **Calibration vs. resolution:** A well-calibrated 70% prediction that resolves as 0 (didn't happen) contributes 0.49 to the Brier score. This is correct but feels like a bad prediction to human observers. Presenting raw Brier scores to admin users without context invites misinterpretation.

**Prevention:**
1. Always present Brier scores with confidence intervals (bootstrap the score over N predictions)
2. Show calibration plots (reliability diagrams) alongside Brier scores -- a Brier score alone is meaningless without N
3. For Polymarket comparison, use the market price at the time of Geopol's prediction (snapshot from `polymarket_snapshots` table), not the current price
4. Require minimum N=30 predictions per category before showing comparative Brier scores
5. Stratify by question difficulty: compare Geopol vs Polymarket separately for high/medium/low volume questions

**Phase:** Phase 24 (Polymarket hardening) and Phase 21 (backtesting)

**Confidence:** HIGH -- standard forecasting methodology.

---

### M8: RSS Feed Administration Without Health Monitoring Creates Zombie Feeds

**What goes wrong:** v3.0 adds admin ability to add/remove/categorize RSS feeds. Without automated health monitoring, feeds silently die:
- Feed URL returns 404 (site migrated, RSS endpoint deprecated)
- Feed returns 200 but XML is empty or malformed
- Feed returns content but hasn't been updated in 60+ days (stale)
- Feed is behind Cloudflare bot protection and returns captcha pages

**WM lesson learned:** World Monitor built a dedicated `validate-rss-feeds.mjs` script (at `/home/kondraki/personal/worldmonitor/scripts/validate-rss-feeds.mjs`) that checks every feed for:
- HTTP accessibility (15s timeout)
- Parseable XML with dates
- Staleness (newest item > 30 days old)
- Status categories: OK, STALE, DEAD, EMPTY

This is a batch validation tool, not runtime monitoring. The gap: feeds can die between validation runs and silently produce no articles for days.

**Existing geopol pattern:** `src/ingest/rss_daemon.py` logs feed failures (`logger.warning("Feed %s returned HTTP %d"...)`) but has no persistent tracking -- you'd have to grep logs to find dead feeds.

**Prevention:**
1. Add a `feed_health` table: (feed_id, last_success, last_failure, consecutive_failures, last_article_date, status)
2. Update feed health on every poll cycle -- this data already exists in `CycleMetrics` but isn't persisted per-feed
3. Admin dashboard shows feed health status with color coding (green/yellow/red)
4. Auto-disable feeds after N consecutive failures (configurable, default 10) with admin notification
5. Periodic staleness check: if a feed hasn't produced a new article in 7 days, flag as STALE in admin dashboard
6. Port WM's feed validation logic as a scheduled job (weekly) that proactively checks all feeds

**Phase:** Phase 22 (source expansion -- RSS feed management)

**Confidence:** HIGH -- verified against WM's `validate-rss-feeds.mjs` and existing `rss_daemon.py` metrics.

---

## Minor Pitfalls (Cause Annoyance but Fixable)

### N1: APScheduler Job Logging Noise

**What goes wrong:** APScheduler logs at INFO level for every job execution, including "skipped" runs (when `coalesce=True` collapses misfired jobs). With 4+ jobs running at various intervals, the log becomes dominated by scheduler noise, making it harder to find actual application events.

**Prevention:**
1. Set APScheduler logger to WARNING level: `logging.getLogger('apscheduler').setLevel(logging.WARNING)`
2. Log job execution in your own wrapper with structured fields (job_name, duration, result)
3. Use structured logging (JSON format -- already available via `settings.log_json`) so scheduler events can be filtered

**Phase:** Phase 20

---

### N2: Admin Dashboard Bundle Size Leaking to Public Routes

**What goes wrong:** The admin dashboard is built as part of the same frontend app with dynamic imports at the route level. If the code-splitting boundary is wrong, admin-only dependencies (charts, data tables, form components) get included in the main bundle, increasing load time for public users.

**Prevention:**
1. Verify admin route chunk is truly separate: `npx source-map-explorer dist/assets/*.js` -- admin chunk must not appear in non-admin entry points
2. Use `React.lazy(() => import('./admin/AdminLayout'))` at the route level (consistent with existing DeckGLMap pattern from v2.1)
3. Test: load `/dashboard` in browser, check Network tab -- no admin-prefixed chunks should load

**Phase:** Phase 19

---

### N3: Multiple Workers Break APScheduler Job Uniqueness

**What goes wrong:** If the API is run with `uvicorn --workers N` (N > 1), each worker process creates its own APScheduler instance. This means each job runs N times per interval -- N GDELT polls, N RSS cycles, N Polymarket loops.

**Prevention:**
1. Use `--workers 1` when APScheduler is in-process (single-server deployment -- this is appropriate for v3.0)
2. If multi-worker is needed later (v4.0 Docker): move scheduler to a separate process or use APScheduler's `SQLAlchemyJobStore` with `replace_existing=True` to ensure single execution
3. Document this constraint prominently in deployment docs

**Phase:** Phase 20

**Confidence:** HIGH -- APScheduler FAQ explicitly documents this issue.

---

### N4: ICEWS Data Lag

**What goes wrong:** ICEWS (Integrated Crisis Early Warning System) data has a publication lag of approximately 1-2 days. If the system treats ICEWS events as "current" without accounting for this lag, recent event counts appear artificially low, then spike when delayed data arrives.

**Prevention:**
1. Track `ingested_at` vs `event_date` per source to measure lag
2. Don't include ICEWS in real-time risk calculations -- use it for historical enrichment and model training only
3. Clearly label ICEWS data freshness in admin dashboard

**Phase:** Phase 22

---

## Phase-Specific Warnings

| Phase | Topic | Likely Pitfall | Severity | Mitigation |
|-------|-------|---------------|----------|------------|
| 19 | Admin dashboard | Auth bypass on mutation endpoints | CRITICAL | Router-level dependency, not per-endpoint |
| 19 | Admin dashboard | CSRF on state-changing admin actions | CRITICAL | Double-submit cookie pattern, SameSite=Strict |
| 19 | Admin dashboard | Admin bundle in public bundle | MINOR | Verify code-splitting with source-map-explorer |
| 20 | Daemon consolidation | Process isolation loss | CRITICAL | ProcessPoolExecutor for heavy jobs |
| 20 | Daemon consolidation | Event loop starvation | CRITICAL | All sync work in to_thread, dedicated executor |
| 20 | Daemon consolidation | Graceful shutdown race | MODERATE | Shutdown scheduler before DB/Redis |
| 20 | Daemon consolidation | Multi-worker job duplication | MINOR | --workers 1 constraint |
| 21 | Backtesting | Look-ahead bias from calibration | MODERATE | Calibration snapshots table |
| 21 | Backtesting | RAG temporal contamination | MODERATE | Backtest TKG-only or temporal ChromaDB |
| 21 | Backtesting | Brier score misinterpretation | MODERATE | Require N>=30, show calibration plots |
| 22 | Source expansion | Cross-source event duplication | CRITICAL | Dedup layer before KG insertion |
| 22 | Source expansion | UCDP API auth blocker | MODERATE | Request token pre-phase |
| 22 | Source expansion | RSS zombie feeds | MODERATE | feed_health table + auto-disable |
| 23 | Global seeding | GDELT coverage bias | MODERATE | ACLED primary, GDELT secondary |
| 23 | Global seeding | Sparse-data denominator-zero | MINOR | Minimum observation threshold |
| 24 | Polymarket | Cap tracking bug | MODERATE | Separate reforecasted_at column |
| 24 | Polymarket | Brier comparison methodology | MODERATE | Snapshot-time prices, N>=30 |

## Sources

### Codebase Analysis (PRIMARY -- HIGH confidence)
- `/home/kondraki/personal/geopol/src/api/app.py` -- Polymarket loop integration, lifespan management
- `/home/kondraki/personal/geopol/src/ingest/rss_daemon.py` -- RSS daemon architecture, concurrency model
- `/home/kondraki/personal/geopol/src/ingest/gdelt_poller.py` -- GDELT poller with backoff strategy
- `/home/kondraki/personal/geopol/src/polymarket/auto_forecaster.py` -- Cap tracking, budget management
- `/home/kondraki/personal/geopol/src/polymarket/client.py` -- Circuit breaker pattern
- `/home/kondraki/personal/geopol/src/api/middleware/auth.py` -- API key auth dependency
- `/home/kondraki/personal/geopol/deploy/systemd/geopol-gdelt-poller.service` -- Process isolation via systemd

### World Monitor Patterns (HIGH confidence)
- `/home/kondraki/personal/worldmonitor/scripts/validate-rss-feeds.mjs` -- Feed health validation approach
- `/home/kondraki/personal/worldmonitor/server/_shared/acled.ts` -- ACLED API integration with caching
- `/home/kondraki/personal/worldmonitor/server/worldmonitor/conflict/v1/list-ucdp-events.ts` -- UCDP data with staleness detection (25h MAX_AGE_MS)
- `/home/kondraki/personal/worldmonitor/src/app/refresh-scheduler.ts` -- Smart polling with backoff and visibility-aware scheduling

### APScheduler GitHub Issues (HIGH confidence)
- [Issue #600: Memory not cleared in Docker](https://github.com/agronholm/apscheduler/issues/600)
- [Issue #235: Memory leak on worker exception](https://github.com/agronholm/apscheduler/issues/235)
- [Issue #484: AsyncIOScheduler must start after event loop](https://github.com/agronholm/apscheduler/issues/484)
- [Issue #304: AsyncIOScheduler defaults to ThreadPoolExecutor](https://github.com/agronholm/apscheduler/issues/304)
- [LiteLLM PR #15846: APScheduler memory leak from jitter + misfire_grace_time](https://github.com/BerriAI/litellm/pull/15846)

### External Research (MEDIUM confidence)
- [Polymarket API Rate Limits](https://docs.polymarket.com/quickstart/introduction/rate-limits) -- 60 req/min, Cloudflare throttling
- [UCDP API Documentation](https://ucdp.uu.se/apidocs/) -- Token auth introduced Feb 2026
- [ONS GDELT Data Quality Note](https://www.ons.gov.uk/peoplepopulationandcommunity/birthsdeathsandmarriages/deaths/methodologies/globaldatabaseofeventslanguageandtonegdeltdataqualitynote) -- 55% accuracy rate, 20% redundancy
- [ACLED Comparison Analysis](https://acleddata.com/sites/default/files/wp-content-archive/uploads/2022/02/ACLED_WorkingPaper_ComparisonAnalysis_2019.pdf) -- Cross-source discrepancies
- [FastAPI CSRF Protection](https://www.stackhawk.com/blog/csrf-protection-in-fastapi/) -- Double-submit cookie pattern
