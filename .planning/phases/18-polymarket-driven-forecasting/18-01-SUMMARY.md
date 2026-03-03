---
phase: 18-polymarket-driven-forecasting
plan: 01
subsystem: api, database, polymarket
tags: [polymarket, ensemble-predictor, auto-forecast, alembic, gemini, redis]

# Dependency graph
requires:
  - phase: 13-calibration-monitoring-hardening
    provides: PolymarketClient, PolymarketMatcher, PolymarketComparisonService, comparison/snapshot tables
  - phase: 14-backend-api-hardening
    provides: submission_worker pattern (EnsemblePredictor + ForecastService + Redis budget)
provides:
  - PolymarketAutoForecaster orchestration class for auto-forecasting unmatched markets
  - Prediction.provenance + Prediction.polymarket_event_id columns with migration
  - Comparison service snapshot query for sparkline data
  - Unified get_all_comparisons() query for ComparisonPanel
  - Auto-forecaster wired into _polymarket_loop with daily reforecast guard
affects: [18-02 (badges + inline comparison), 18-03 (ComparisonPanel + API routes)]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "PolymarketAutoForecaster: volume-filtered, budget-gated, cap-limited background forecasting"
    - "Tiered extraction: heuristic COUNTRY_NAME_TO_ISO first, Gemini LLM fallback for country"
    - "Reforecast overwrites existing Prediction row via ORM (historical preserved in snapshots)"
    - "Daily reforecast guard via _last_reforecast_date module-level flag"

key-files:
  created:
    - src/polymarket/auto_forecaster.py
    - alembic/versions/20260304_005_polymarket_provenance.py
  modified:
    - src/db/models.py
    - src/settings.py
    - src/polymarket/comparison.py
    - src/api/app.py

key-decisions:
  - "Fresh EnsemblePredictor instance per prediction (holds mutable _forecast_output state)"
  - "Reforecast overwrites existing Prediction row -- historical values preserved in polymarket_snapshots"
  - "Split daily caps: 3 new forecasts + 5 reforecasts (8 total, within 25 Gemini budget)"
  - "Country extraction: heuristic first (zero API cost), LLM fallback only when needed"
  - "CAMEO extraction always via LLM (per-CAMEO calibration weights require accuracy)"
  - "Application-level snapshot sampling (every Nth) for sparkline data portability"
  - "Daily reforecast guard via _last_reforecast_date flag (simpler than DB count check)"

patterns-established:
  - "Auto-forecaster pattern: filter -> extract -> predict -> persist -> compare"
  - "Budget-gated pipeline: check gemini_budget_remaining before each API call"
  - "Dedup via polymarket_event_id column on Prediction + comparisons table"

# Metrics
duration: 5min
completed: 2026-03-04
---

# Phase 18 Plan 01: Auto-Forecast Pipeline Summary

**PolymarketAutoForecaster with volume-filtered candidate selection, tiered country/CAMEO extraction, budget-gated EnsemblePredictor execution, and daily reforecast of active comparisons**

## Performance

- **Duration:** 5 min
- **Started:** 2026-03-03T21:44:11Z
- **Completed:** 2026-03-03T21:49:02Z
- **Tasks:** 3
- **Files modified:** 6

## Accomplishments
- PolymarketAutoForecaster.run() filters unmatched events by volume threshold ($100K), extracts country+CAMEO+horizon, generates forecasts via EnsemblePredictor, persists with provenance="polymarket_driven", creates comparison rows
- PolymarketAutoForecaster.reforecast_active() overwrites existing Prediction rows for tracked comparisons (historical values in polymarket_snapshots)
- Daily caps (3 new + 5 reforecast) and Gemini budget checks via Redis prevent runaway API costs
- Comparison service extended with sparkline-ready snapshot queries and unified all-comparisons query

## Task Commits

Each task was committed atomically:

1. **Task 1: DB schema extensions + settings + Alembic migration** - `94cb649` (feat)
2. **Task 2a: PolymarketAutoForecaster class** - `c657ffb` (feat)
3. **Task 2b: comparison.py snapshot/panel queries + app.py wiring** - `4bb4fd9` (feat)

## Files Created/Modified
- `src/polymarket/auto_forecaster.py` - 627-line orchestration class: volume filter, country/CAMEO extraction, EnsemblePredictor pipeline, budget gating, daily caps, reforecast logic
- `src/db/models.py` - Prediction gains provenance (VARCHAR(30), indexed) and polymarket_event_id (VARCHAR(100), indexed)
- `src/settings.py` - polymarket_volume_threshold (100K), polymarket_daily_new_forecast_cap (3), polymarket_daily_reforecast_cap (5)
- `alembic/versions/20260304_005_polymarket_provenance.py` - Migration 005: two columns, two indexes, clean downgrade
- `src/polymarket/comparison.py` - get_snapshots_for_comparison() for sparklines, get_all_comparisons() for ComparisonPanel
- `src/api/app.py` - _polymarket_loop extended with auto-forecaster + daily reforecast guard

## Decisions Made
- Fresh EnsemblePredictor instance per prediction -- holds mutable `_forecast_output` state, cannot be reused across predictions
- Reforecast overwrites existing Prediction row via direct ORM attribute assignment -- no new rows, historical values preserved in `polymarket_snapshots` table
- Split daily caps (3 new + 5 reforecast) rather than shared cap -- ensures active comparisons always get refreshed while allowing new discoveries
- Country extraction uses tiered approach: heuristic (zero cost) first from COUNTRY_NAME_TO_ISO dict, Gemini LLM fallback only when heuristic fails
- CAMEO extraction always via LLM -- per-CAMEO calibration weights require accurate categorization
- Daily reforecast guard via `_last_reforecast_date` module-level flag rather than DB count check per cycle
- Application-level snapshot sampling (every Nth row) for sparkline data -- portable across DB backends

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None.

## User Setup Required
Migration 005 must be applied before auto-forecasting works: `uv run alembic upgrade head`

## Next Phase Readiness
- Auto-forecast pipeline complete and wired into _polymarket_loop
- Comparison service has all queries needed for ComparisonPanel (Plan 03)
- Snapshot data served via get_snapshots_for_comparison() -- ready for sparkline rendering (Plan 02/03)
- Provenance column enables badge rendering on forecast cards (Plan 02)
- Plan 02 (badges + inline comparison) can proceed immediately
- Plan 03 (ComparisonPanel + API routes) can proceed immediately

---
*Phase: 18-polymarket-driven-forecasting*
*Completed: 2026-03-04*
