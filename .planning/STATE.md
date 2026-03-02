# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-02)

**Core value:** Explainability -- every forecast must provide clear, traceable reasoning paths
**Current focus:** Defining requirements for v2.1

## Current Position

Milestone: v2.1 Production UX & Live Data Integration
Phase: Not started (defining requirements)
Plan: —
Status: Defining requirements
Last activity: 2026-03-02 — Milestone v2.1 started

Progress: [##########################################] 100% (16/16 phases lifetime)
v2.1:    [..........] 0% (0/? phases — roadmap pending)

## Performance Metrics

**Velocity:**
- Total plans completed: 49
- Average duration: 12 minutes
- Total execution time: 9.50 hours

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

**Recent Trend:**
- Last 4 plans: 13-04 (4min), 13-05 (3min), 13-07 (4min), 13-06 (6min)
- Trend: Phase 13 complete -- all calibration/monitoring/integration plans delivered

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

### Deferred Issues

- Docker daemon requires sudo to start -- user must `sudo systemctl start docker` before running containers or Alembic migrations
- PostgreSQL tests (8 tests in test_forecast_persistence.py and test_concurrent_db.py) skip until Docker is running
- TKG entity resolution mismatch: GDELT actor codes (USA, GOV) vs LLM entity names (US Department of the Treasury) -- structural gap requiring normalization layer
- Mobile/responsive layout deferred -- three screens + globe = poor mobile experience

### Blockers/Concerns

- Polyglot tax: Python + TypeScript -- two ecosystems for single developer (Rust/Tauri dropped, web-only)
- Docker daemon not auto-started -- verification of PostgreSQL/Redis containers and Alembic migration deferred

## Session Continuity

Last session: 2026-03-02
Stopped at: Defining v2.1 requirements
Resume file: None
Next: Complete requirements definition and roadmap creation
