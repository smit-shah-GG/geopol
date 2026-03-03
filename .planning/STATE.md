# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-02)

**Core value:** Explainability -- every forecast must provide clear, traceable reasoning paths
**Current focus:** Phase 16 in progress -- Globe & Forecasts Screens

## Current Position

Milestone: v2.1 Production UX & Live Data Integration
Phase: 16 of 18 (Globe & Forecasts Screens)
Plan: 02 of 03
Status: In progress
Last activity: 2026-03-03 -- Completed 16-02-PLAN.md (Globe Screen + Layer Pill Bar + Drill-Down)

Progress: [#################################################...] 97% (58/60 plans lifetime)
v2.1:    [#########.] 75% (9/12 plans)

## Performance Metrics

**Velocity:**
- Total plans completed: 58
- Average duration: 11 minutes
- Total execution time: 10.30 hours

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

| 16-globe-forecasts-screens | 2 | 11min | 6min |

**Recent Trend:**
- Last 4 plans: 15-03 (4min), 16-01 (5min), 16-02 (6min)
- Trend: Consistent 4-6min -- UI component construction on established codebase

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

### Deferred Issues

- Docker daemon requires sudo to start -- user must `sudo systemctl start docker` before running containers or Alembic migrations
- PostgreSQL tests (8 tests in test_forecast_persistence.py and test_concurrent_db.py) skip until Docker is running
- TKG entity resolution mismatch: GDELT actor codes (USA, GOV) vs LLM entity names (US Department of the Treasury) -- structural gap requiring normalization layer
- Mobile/responsive layout deferred -- three screens + globe = poor mobile experience
- Migration 004 must be applied before persistence tests pass: `uv run alembic upgrade head`

### Blockers/Concerns

- Polyglot tax: Python + TypeScript -- two ecosystems for single developer (Rust/Tauri dropped, web-only)
- Docker daemon not auto-started -- verification of PostgreSQL/Redis containers and Alembic migration deferred

## Session Continuity

Last session: 2026-03-03
Stopped at: Completed 16-02-PLAN.md (Globe Screen + Layer Pill Bar + Drill-Down)
Resume file: None
Next: 16-03-PLAN.md -- Forecasts Screen (submission form + queue)
