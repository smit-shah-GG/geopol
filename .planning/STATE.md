# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-04)

**Core value:** Explainability -- every forecast must provide clear, traceable reasoning paths
**Current focus:** v3.0 Operational Command & Verification -- Phase 21 Source Expansion & Feed Management

## Current Position

Milestone: v3.0 Operational Command & Verification
Phase: 21 of 25 (Source Expansion & Feed Management) -- In progress
Plan: 03 of 05
Status: In progress
Last activity: 2026-03-06 -- Completed 21-03-PLAN.md (dashboard NewsFeedPanel + BreakingNewsBanner)

Progress: [########################################################] 100% (74/74+ plans lifetime)
v3.0:    [#########           ] 43% (9/21 plans in v3.0 -- 2/7 phases + 3 plans)

## Performance Metrics

**Velocity:**
- Total plans completed: 74
- Average duration: 10 minutes
- Total execution time: 11.9 hours

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
| 21-source-expansion-feed-mgmt | 3 | 21min | 7min |

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Key decisions affecting current work:

- Admin dashboard at `/admin` route, same app, dynamic import code-split, route-level auth gating (2026-03-04)
- APScheduler 3.11.2 for daemon consolidation -- single process, in-process with FastAPI, AsyncIOScheduler, MemoryJobStore (2026-03-04)
- Backtesting: isolated internal reporting only -- walk-forward eval, model comparison, calibration audit (2026-03-04)
- Polymarket: fix created_at overwrite + add rigorous Brier score tracking (2026-03-04)
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

Last session: 2026-03-06
Stopped at: Completed 21-03-PLAN.md (dashboard NewsFeedPanel + BreakingNewsBanner)
Resume file: None
Next: 21-04-PLAN.md
