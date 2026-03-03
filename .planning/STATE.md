# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-02)

**Core value:** Explainability -- every forecast must provide clear, traceable reasoning paths
**Current focus:** Phase 18 in progress (Polymarket-Driven Forecasting)

## Current Position

Milestone: v2.1 Production UX & Live Data Integration
Phase: 18 of 18 (Polymarket-Driven Forecasting)
Plan: 01 of 03
Status: In progress
Last activity: 2026-03-04 -- Completed 18-01-PLAN.md (Auto-Forecast Pipeline)

Progress: [####################################################] 100% (63/65 plans lifetime)
v2.1:    [##############------] 70% (14/20 plans)

## Performance Metrics

**Velocity:**
- Total plans completed: 63
- Average duration: 10 minutes
- Total execution time: 10.9 hours

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
| 18-polymarket-driven-forecasting | 1 | 5min | 5min |

**Recent Trend:**
- Last 4 plans: 17-02 (12min), 17-03 (7min), 18-01 (5min)
- Trend: Fast execution on well-scoped orchestration plans

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Key decisions affecting current work:

- Three-screen URL routing (/dashboard, /globe, /forecasts) for bookmarkability -- not tab state (2026-03-02)
- Question submission queue model, NOT chatbot -- 2-3 minute forecast generation precludes conversational UX (2026-03-02)
- Progressive disclosure: click-expand inline first, "View Full Analysis" for ScenarioExplorer modal (2026-03-02)
- Kill mock fixture fallback in forecasts.py -- return empty results when no data (2026-03-02)
- Real country risk from PostgreSQL aggregation of predictions table -- not hardcoded list (2026-03-02)
- Scenario tree node text: short label (~40 chars) + tooltip, not multiline text boxes (2026-03-02)
- use_fixtures=False by default; production never sees fixture data (2026-03-03)
- question_tsv uses GENERATED ALWAYS AS ... STORED rather than trigger-maintained (2026-03-03)
- Bare except blocks removed from GET production paths; PostgreSQL errors propagate to 500 (2026-03-03)
- plainto_tsquery over to_tsquery for search -- safe natural-language input, no injection risk (2026-03-03)
- Nullable suggestions field in SearchResponse -- prevents breaking DTO change when LLM suggestions added later (2026-03-03)
- sqlalchemy.text() for CTE country risk query -- 4-CTE analytical query unreadable as Core expressions (2026-03-03)
- top_question renamed to top_forecast -- completed in 15-01 (2026-03-03)
- asyncio.Semaphore(3) over Celery/Redis queue for submission worker -- single-server deployment (2026-03-03)
- Two-phase submit/confirm flow -- user reviews LLM-parsed interpretation before committing API budget (2026-03-03)
- SELECT FOR UPDATE SKIP LOCKED for worker request claiming -- no blocking, no double-pickup (2026-03-03)
- Graceful LLM parse fallback to defaults -- failed parse must never block submission (2026-03-03)
- View Transition API with sync fallback for screen switches (2026-03-03)
- Module-scoped state for screen mount/unmount lifecycle (2026-03-03)
- DeckGLMap dynamic import at route level for code-splitting (2026-03-03)
- SourcesPanel receives data via push from health refresh -- no independent /health calls (2026-03-03)
- getRequests() shares health circuit breaker group -- low-priority polling endpoint (2026-03-03)
- Mutations (submitQuestion, confirmSubmission) bypass dedup and circuit breaker for immediate feedback (2026-03-03)
- Diff-based DOM updates in ForecastPanel via cardElements Map -- preserves expanded state across 60s refresh (2026-03-03)
- Mini d3 tree limited to 2 depth levels with 20-char labels -- preview only, full detail in ScenarioExplorer (2026-03-03)
- SearchBar as standalone class (not Panel subclass) -- no header/resize/badge overhead for inline control (2026-03-03)
- ScenarioExplorer tooltip uses HTML div positioned via pageX/pageY -- not SVG title (2026-03-03)
- forecast-selected CustomEvent dispatches on window for cross-screen listening (2026-03-03)
- DeckGLMap layer defaults unchanged (all true); globe screen calls setLayerDefaults() post-construction (2026-03-03)
- Globe drill-down uses country-brief-requested event (not country-selected) to open CountryBriefPage (2026-03-03)
- LayerPillBar in separate Vite chunk (1.13 kB) -- only loaded on /globe route (2026-03-03)
- getTopForecasts(50) for globe (vs 10 for dashboard) -- more markers for scatter layer (2026-03-03)
- Generic keyset cursor (encode_keyset_cursor/decode_keyset_cursor) alongside existing forecast cursor -- no breaking changes (2026-03-04)
- Event backfill yields 0 results (1.37M rows have NULL raw_json) -- country_iso populates on new ingestion only (2026-03-04)
- query_top_actors uses UNION ALL actor1+actor2 with GROUP BY dedup for bilateral actor coverage (2026-03-04)
- AdvisoryStore uses classmethod in-memory cache -- no Pydantic coupling, import-safe for both route and poller (2026-03-04)
- ACLED uses key+email query params (NOT OAuth2) per ACLED API docs (2026-03-04)
- EU EEAS dropped from advisory sources -- no structured API exists (2026-03-04)
- FCDO per-country fetches bounded by Semaphore(5) + 0.3s delay to respect GOV.UK rate limits (2026-03-04)
- Events breaker independent from forecast breaker -- different failure modes (high-freq events vs low-freq forecasts) (2026-03-04)
- SourcesPanel self-refreshes via /sources at 60s -- decoupled from health push (2026-03-04)
- EventTimelinePanel renamed to EVENT FEED (multi-source: GDELT + ACLED) (2026-03-04)
- CountryBriefPage lazy tab loading with null sentinel pattern (null=not loaded, []=loaded empty) (2026-03-04)
- Client-side actor aggregation from 200-event window -- avoids dedicated backend endpoint (2026-03-04)
- SourcesPanel push from health refresh replaced by independent /sources self-refresh (supersedes 2026-03-03 decision) (2026-03-04)
- Fresh EnsemblePredictor instance per prediction -- holds mutable _forecast_output state (2026-03-04)
- Reforecast overwrites existing Prediction row -- historical values preserved in polymarket_snapshots (2026-03-04)
- Split daily caps: 3 new + 5 reforecast (8 total within 25 Gemini budget) (2026-03-04)
- Country extraction tiered: heuristic COUNTRY_NAME_TO_ISO first (zero cost), Gemini LLM fallback (2026-03-04)
- CAMEO extraction always via LLM -- per-CAMEO calibration weights require accuracy (2026-03-04)
- Daily reforecast guard via _last_reforecast_date flag -- simpler than DB count per cycle (2026-03-04)

### Deferred Issues

- Docker daemon requires sudo to start -- user must `sudo systemctl start docker` before running containers or Alembic migrations
- PostgreSQL tests (8 tests in test_forecast_persistence.py and test_concurrent_db.py) skip until Docker is running
- TKG entity resolution mismatch: GDELT actor codes (USA, GOV) vs LLM entity names (US Department of the Treasury) -- structural gap requiring normalization layer
- Mobile/responsive layout deferred -- three screens + globe = poor mobile experience
- Migration 004 must be applied before persistence tests pass: `uv run alembic upgrade head`
- Migration 005 must be applied for Polymarket auto-forecasting: `uv run alembic upgrade head`

### Blockers/Concerns

- Polyglot tax: Python + TypeScript -- two ecosystems for single developer (Rust/Tauri dropped, web-only)
- Docker daemon not auto-started -- verification of PostgreSQL/Redis containers and Alembic migration deferred

### Quick Tasks Completed

| # | Description | Date | Commit | Directory |
|---|-------------|------|--------|-----------|
| 002 | Post-Phase 16 UX fixes (Polymarket top-10, map resize, dashboard nav, ensemble removal, expandable cards, QUEUED status) | 2026-03-03 | fa00903 | [002-post-phase16-ux-fixes](./quick/002-post-phase16-ux-fixes/) |

## Session Continuity

Last session: 2026-03-04
Stopped at: Completed 18-01-PLAN.md (Auto-Forecast Pipeline)
Resume file: None
Next: Execute 18-02-PLAN.md (Badges + Inline Comparison)
