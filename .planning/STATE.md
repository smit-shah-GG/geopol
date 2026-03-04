# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-04)

**Core value:** Explainability -- every forecast must provide clear, traceable reasoning paths
**Current focus:** v3.0 Operational Command & Verification

## Current Position

Milestone: v3.0 Operational Command & Verification
Phase: Not started (defining requirements)
Plan: —
Status: Defining requirements
Last activity: 2026-03-04 — Milestone v3.0 started

Progress: [########################################################] 100% (65/65 plans lifetime)
v3.0:    [                    ] 0% (0/? plans)

## Performance Metrics

**Velocity:**
- Total plans completed: 65
- Average duration: 10 minutes
- Total execution time: 11.1 hours

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

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Key decisions affecting current work:

- Admin dashboard at `/admin` route, same app, dynamic import code-split, route-level auth gating (2026-03-04)
- APScheduler for daemon consolidation — single process, in-process with FastAPI, AsyncIOScheduler (2026-03-04)
- Backtesting: isolated internal reporting only — walk-forward eval, model comparison, calibration audit (2026-03-04)
- Polymarket: fix operational reliability + add rigorous Brier score tracking (2026-03-04)
- Source expansion: WM RSS feed management, ICEWS + UCDP, per-source health/admin controls (2026-03-04)
- Global seeding: all ~195 countries get baseline risk from event density + ACLED + ICEWS + advisories (2026-03-04)
- Globe layer pills: Arcs/Heatmap/Scenarios are no-ops because data arrays never populated — data-wiring fix (2026-03-04)
- Dockerization deferred to v4.0 (gates on daemon consolidation) (2026-03-04)
- TiRGN training quality: weight_decay=0.001, warmup_epochs=3, label_smoothing=0.1 (2026-03-04)

### Deferred Issues

- Docker daemon requires sudo to start -- user must `sudo systemctl start docker` before running containers or Alembic migrations
- PostgreSQL tests (8 tests in test_forecast_persistence.py and test_concurrent_db.py) skip until Docker is running
- TKG entity resolution mismatch: GDELT actor codes (USA, GOV) vs LLM entity names (US Department of the Treasury) -- structural gap requiring normalization layer
- Mobile/responsive layout deferred -- three screens + globe = poor mobile experience
- Migration 004 must be applied before persistence tests pass: `uv run alembic upgrade head`
- Migration 005 must be applied for Polymarket auto-forecasting: `uv run alembic upgrade head`
- Pre-existing test failure: test_default_is_regcn expects "regcn" but default is "tirgn" since Phase 11

### Blockers/Concerns

- Polyglot tax: Python + TypeScript -- two ecosystems for single developer (Rust/Tauri dropped, web-only)
- Docker daemon not auto-started -- verification of PostgreSQL/Redis containers and Alembic migration deferred

### Quick Tasks Completed

| # | Description | Date | Commit | Directory |
|---|-------------|------|--------|-----------|
| 002 | Post-Phase 16 UX fixes (Polymarket top-10, map resize, dashboard nav, ensemble removal, expandable cards, QUEUED status) | 2026-03-03 | fa00903 | [002-post-phase16-ux-fixes](./quick/002-post-phase16-ux-fixes/) |

## Session Continuity

Last session: 2026-03-04
Stopped at: v3.0 milestone initialization — defining requirements
Resume file: None
Next: Define v3.0 requirements and create roadmap
