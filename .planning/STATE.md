# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-04)

**Core value:** Explainability -- every forecast must provide clear, traceable reasoning paths
**Current focus:** v3.0 Operational Command & Verification -- Phase 25 Frontend Finalization

## Current Position

Milestone: v3.0 Operational Command & Verification
Phase: 25 of 25 (Frontend Finalization) -- In progress
Plan: 01 of 03
Status: In progress
Last activity: 2026-03-08 -- Completed 25-01-PLAN.md (Shared Infrastructure & Skeleton Builder)

Progress: [########################################################] 100% (90/80+ plans lifetime)
v3.0:    [###########################-] 96% (27/28 plans in v3.0 -- 7/7 phases)

## Performance Metrics

**Velocity:**
- Total plans completed: 88
- Average duration: 10 minutes
- Total execution time: 13.3 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 01-data-foundation | 3 | 30min | 10min |
| 02-knowledge-graph | 3 | 180min | 60min |
| 03-hybrid-forecasting | 4 | 95min | 24min |
| 04-calibration-evaluation | 2 | 44min | 22min |
| 05-tkg-training | 4 | 64min | 16min |
| 06-networkx-fix | 1 | 2min | 2min |
| 07-bootstrap-pipeline | 2 | 12min | 6min |
| 08-graph-partitioning | 2 | 12min | 6min |
| 09-api-foundation | 6 | 33min | 6min |
| 10-ingest-forecast-pipeline | 4 | 27min | 7min |
| 11-tkg-predictor-replacement | 3 | 21min | 7min |
| 12-wm-derived-frontend | 7 | 36min | 5min |
| 13-calibration-monitoring-hardening | 7 | 27min | 4min |
| 14-backend-api-hardening | 4 | 17min | 4min |
| 15-url-routing-dashboard | 3 | 20min | 7min |
| 16-globe-forecasts-screens | 3 | 16min | 5min |
| 17-live-data-feeds-country-depth | 3 | 26min | 9min |
| 18-polymarket-driven-forecasting | 3 | 14min | 5min |
| 19-admin-dashboard-foundation | 3 | 20min | 7min |
| 20-daemon-consolidation | 3 | 25min | 8min |
| 21-source-expansion-feed-mgmt | 5 | 34min | 7min |
| 22-polymarket-hardening | 3 | 23min | 8min |
| 23-historical-backtesting | 3 | 31min | 10min |
| 24-global-seeding-globe-layers | 6 | 23min | 4min |
| 25-frontend-finalization | 1/3 | 3min | 3min |

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Key decisions affecting current work:

- Admin dashboard at `/admin` route, same app, dynamic import code-split, route-level auth gating (2026-03-04)
- APScheduler 3.11.2 for daemon consolidation -- single process, in-process with FastAPI, AsyncIOScheduler, MemoryJobStore (2026-03-04)
- Backtesting: isolated internal reporting only -- walk-forward eval, model comparison, calibration audit (2026-03-04)
- Polymarket: created_at overwrite bug FIXED in 22-01; reforecasted_at column tracks reforecast activity (2026-03-06)
- PolymarketAccuracy table: append-only cumulative Brier score ledger (2026-03-06)
- Source expansion: UCDP (not ICEWS -- dead since April 2023), RSS feed management from admin, per-source health (2026-03-04)
- Global seeding: all ~195 countries get baseline risk from GDELT + ACLED + UCDP + advisories (2026-03-04)
- Globe layer pills: Arcs/Heatmap/Scenarios are no-ops -- data-wiring fix, not toggle fix (2026-03-04)
- uvicorn --workers 1 is a hard constraint with APScheduler in-process (2026-03-04)
- POLECAT/ICEWS deferred to v3.1+ (PLOVER-to-CAMEO mapping untested) (2026-03-04)
- Admin auth: X-Admin-Key header separate from X-API-Key, router-level dependency (2026-03-05)
- trigger_job now fires via APScheduler modify_job(next_run_time=now) -- 501 stub deleted (2026-03-05)
- Scheduler shutdown order: scheduler (30s) -> Redis -> DB (in-flight jobs may use DB) (2026-03-05)
- RSS daemon aggregates 3 APScheduler sub-jobs; paused only when ALL paused (2026-03-05)
- system_config values wrapped as {"v": value} for type-preserving JSON round-trip (2026-03-05)
- AdminLayout exposes adminKey property for Plan 03 panel AdminClient construction (2026-03-05)
- admin-screen.ts is the static/dynamic import boundary -- only import type at top level (2026-03-05)
- asyncio.Lock for heavy job mutual exclusion -- FIFO queue prevents concurrent subprocess work (2026-03-05)
- subprocess.run for scripts/ (not importable), in-process for src.polymarket.* (2026-03-05)
- Singleton GDELTPoller in SharedDeps -- _last_url persists across poll cycles (2026-03-05)
- RSS feeds: DB-backed with feed_config.py fallback, auto-disable after 5 failures, callback-based health updates (2026-03-06)
- Feed CRUD: soft-delete default, hard delete via ?purge=true, 409 on duplicate name (2026-03-06)
- Cross-source dedup: conservative (date, country, coarse_type) fingerprint, ACLED > UCDP > GDELT priority (2026-03-06)
- Actor1 code used as country_iso proxy for dedup fingerprinting (2026-03-06)
- Source tier/propaganda maps are frontend-only static data, ported from WM (2026-03-06)
- Jaccard threshold 0.5 for article clustering; probability spike threshold 0.15 for breaking alerts (2026-03-06)
- Alert sound default: off (opt-in via localStorage geopol-alert-sound) (2026-03-06)
- YouTube IFrame API: lazy-loaded via Promise wrapper, onYouTubeIframeAPIReady set before script injection (2026-03-06)
- Exclusive unmute: only one YouTube player can produce audio at a time (2026-03-06)
- SettingsModal sources are static catalog (45 entries, 10 categories), not DB-driven (2026-03-06)
- geopol:sources-changed CustomEvent dispatches on toggle; consumer wired in Plan 05 (2026-03-06)
- Accuracy panel: live-computed from polymarket_comparisons, not from polymarket_accuracy ledger (2026-03-06)
- Winner = lower Brier score; client-side sorting with state preservation across 30s refresh (2026-03-06)
- Resolution uses Gamma API closed/resolutionSource/umaResolutionStatus, not price convergence (2026-03-06)
- Voided markets: three-heuristic detection (ambiguous prices, source keywords, UMA status) (2026-03-06)
- Cycle order: match -> snapshot -> resolve -> forecast -> reforecast (race-safe) (2026-03-06)
- Top-10 active set: volume threshold filter removed from auto_forecaster.run(), caller provides pre-filtered set (2026-03-06)
- reforecast_active() scoped to top-10 event IDs; None parameter maintains backward compat (2026-03-06)
- 429 handling: sleep(min(Retry-After, 60)) before raising for tenacity retry (2026-03-06)
- Backtesting: DB-based cancellation (not threading.Event) -- process-safe, polls status between windows (2026-03-07)
- Backtesting: Python-side date filtering for temporal ChromaDB index (not $lte string comparison) (2026-03-07)
- Backtesting: MRR = None in window results (TKG ranking data unavailable from re-prediction path) (2026-03-07)
- Backtesting: fire-and-forget dispatch via asyncio.create_task(heavy_backtest(config_json)) (2026-03-08)
- Backtesting: heavy_backtest() NOT registered as APScheduler job -- on-demand only via admin API (2026-03-08)
- Backtesting: import-by-value bug fixed in run_polymarket_cycle (bare names -> module reference) (2026-03-08)
- Backtesting admin panel: expandable d3 chart sections with lazy render, 10s refresh, scoped CSS injection (2026-03-08)
- TravelAdvisory: UniqueConstraint on (country_iso, source) for UPSERT semantics -- cross-process advisory persistence (2026-03-08)
- HeatmapHexbin: String(20) for H3 index, all resolutions covered. Pre-computed layer data pattern (2026-03-08)
- SQLite events lat/lon: nullable REAL columns for geocoding, backward-compatible with 1.43M existing events (2026-03-08)
- FIPS CSV under src/seeding/data/ (not top-level data/) due to gitignore; negation rule added (2026-03-08)
- fips_to_iso() called at GDELT ingestion boundary -- all stored country_iso values are now ISO alpha-2 (2026-03-08)
- Retroactive _migrate_fips_to_iso() runs on every EventStorage startup, idempotent (2026-03-08)
- Advisory dual-write: in-memory AdvisoryStore + PostgreSQL travel_advisories, DB write is non-critical (2026-03-08)
- heavy_baseline_risk uses skip-if-locked (not queue) -- silently skips when heavy_job_lock is held, retries next hour (2026-03-08)
- compute_all_layers: full table replace (DELETE all + INSERT new) in single transaction -- simpler than UPSERT (2026-03-08)
- Heatmap uses 30-day window (vs 90-day for baseline risk) -- shows recent hotspots, not historical spread (2026-03-08)
- Two-query Python-side merge: baseline table scan + forecast CTE for countries endpoint (2026-03-08)
- Layer endpoints return envelope objects with computed_at for staleness display (2026-03-08)
- GET /countries/{iso} no longer 404 for baseline-only countries (returns baseline risk with forecast_risk=None) (2026-03-08)
- DeckGLMap layers 3/4/5 are dual-mode: global bilateral arcs vs per-country, H3HexagonLayer vs HeatmapLayer, risk deltas vs scenario highlights (2026-03-08)
- Globe screen all 5 layers default ON, 5-minute refresh cycle for layer data (2026-03-08)
- WCAG AA contrast: --text-muted #6a7a8c, --accent #4080dd (was #506070, #3a7bd5) (2026-03-08)
- Panel constructor calls showSkeleton() instead of showLoading() -- shimmer replaces radar sweep (2026-03-08)
- Skeleton builder: 7 shapes, PANEL_SKELETON_MAP for 10 panel IDs, role=status + aria-busy a11y (2026-03-08)
- prefers-reduced-motion: skeleton shimmer disabled (static bg), view transitions near-instant (2026-03-08)
- buildLayers() return type changed to Layer[] (deck.gl base class) for polymorphic layer container (2026-03-08)

### Deferred Issues

- Docker daemon requires sudo to start -- user must `sudo systemctl start docker` before running containers or Alembic migrations
- PostgreSQL tests (8 tests in test_forecast_persistence.py and test_concurrent_db.py) skip until Docker is running
- TKG entity resolution mismatch: GDELT actor codes (USA, GOV) vs LLM entity names -- structural gap requiring normalization layer
- Pre-existing test failure: test_default_is_regcn expects "regcn" but default is "tirgn" since Phase 11
- UCDP API token must be requested before Phase 21 implementation (email-gated, days to weeks)

### Blockers/Concerns

- Polyglot tax: Python + TypeScript -- two ecosystems for single developer
- UCDP token procurement -- calendar blocker for Phase 21; fallback is bulk CSV download

### Quick Tasks Completed

| # | Description | Date | Commit | Directory |
|---|-------------|------|--------|-----------|
| 002 | Post-Phase 16 UX fixes (Polymarket top-10, map resize, dashboard nav, ensemble removal, expandable cards, QUEUED status) | 2026-03-03 | fa00903 | [002-post-phase16-ux-fixes](./quick/002-post-phase16-ux-fixes/) |

## Session Continuity

Last session: 2026-03-08
Stopped at: Completed 25-01-PLAN.md (Shared Infrastructure & Skeleton Builder)
Resume file: None
Next: 25-02-PLAN.md (Per-Panel UX Overhaul -- ForecastPanel, RiskIndex, NewsFeed, SystemHealth)
