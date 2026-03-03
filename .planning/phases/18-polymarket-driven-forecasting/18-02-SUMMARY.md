---
phase: 18-polymarket-driven-forecasting
plan: 02
subsystem: api, schemas
tags: [polymarket, forecast-response, dto, comparison-panel, sparkline, enrichment]

# Dependency graph
requires:
  - phase: 18-polymarket-driven-forecasting
    plan: 01
    provides: PolymarketComparison/Snapshot tables, get_all_comparisons(), get_snapshots_for_comparison()
  - phase: 13-calibration-monitoring-hardening
    provides: Calibration router, Polymarket comparison endpoints

provides:
  - PolymarketComparisonData DTO on ForecastResponse
  - GET /calibration/polymarket/comparisons endpoint
  - GET /calibration/polymarket/comparisons/{id}/snapshots endpoint
  - ForecastService.enrich_with_comparisons() batch enrichment

affects:
  - phase: 18-polymarket-driven-forecasting
    plan: 03
    impact: Frontend can now consume polymarket_comparison from /forecasts/top and dedicated panel endpoints

# Tech tracking
tech-stack:
  added: []
  patterns:
    - Batch IN-clause enrichment (ForecastService.enrich_with_comparisons)
    - model_copy(update={}) for immutable DTO enrichment
    - Session-wrapper pattern for PolymarketComparisonService in route handlers

# File tracking
key-files:
  modified:
    - src/api/schemas/forecast.py
    - src/api/services/forecast_service.py
    - src/api/routes/v1/forecasts.py
    - src/api/routes/v1/calibration.py

# Decisions
decisions:
  - id: D-1802-01
    description: "Batch IN-clause for forecast enrichment -- single query for all forecast IDs, not N+1"
    rationale: "Comparison table is small (tens of rows) but N+1 would still be wasteful pattern"
  - id: D-1802-02
    description: "Provenance determined from Prediction.provenance field, fallback to 'polymarket_tracked'"
    rationale: "Polymarket-driven predictions have explicit provenance; tracked ones lack it but have comparison row"
  - id: D-1802-03
    description: "Divergence computed as geopol_probability - polymarket_price (positive = geopol higher)"
    rationale: "Consistent sign convention -- positive divergence means geopol is more bullish"

# Metrics
metrics:
  duration: 3min
  completed: 2026-03-04
---

# Phase 18 Plan 02: API Layer for Polymarket Comparison Data Summary

**One-liner:** ForecastResponse DTO extended with optional polymarket_comparison field, two new calibration endpoints for ComparisonPanel and sparkline snapshots, batch enrichment via JOIN on forecast routes.

## What Was Built

### Task 1: DTO Extensions + Forecast Enrichment
- **PolymarketComparisonData** Pydantic model added to `forecast.py` -- carries comparison_id, polymarket_event_id, title, prices, divergence, provenance, status, Brier scores
- **ForecastResponse.polymarket_comparison** -- Optional field, None when no comparison exists, populated via enrichment
- **ForecastService.enrich_with_comparisons()** -- Single batch query with `SELECT * FROM polymarket_comparisons JOIN predictions WHERE geopol_prediction_id IN (:ids)`, builds PolymarketComparisonData, attaches via model_copy
- Wired into all three GET forecast routes: `/forecasts/top`, `/forecasts/country/{iso}`, `/forecasts/{id}`

### Task 2: Comparison Panel + Snapshot Endpoints
- **GET /calibration/polymarket/comparisons** -- Returns all active+resolved comparisons with computed divergence and provenance, ordered by created_at DESC
- **GET /calibration/polymarket/comparisons/{id}/snapshots** -- Returns sampled time-series data for sparkline rendering, with configurable limit (default 30, max 100)
- Both endpoints use API key auth (Depends(verify_api_key))
- Both use the session-wrapper pattern for PolymarketComparisonService (same as existing `/polymarket` endpoint)

## Deviations from Plan

None -- plan executed exactly as written.

## Decisions Made

| ID | Decision | Rationale |
|----|----------|-----------|
| D-1802-01 | Batch IN-clause for enrichment | No N+1; single query for all forecast IDs |
| D-1802-02 | Provenance fallback to polymarket_tracked | Prediction.provenance null + comparison exists = tracked |
| D-1802-03 | Divergence = geopol - polymarket | Positive = geopol more bullish, consistent sign convention |

## Task Completion

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | DTO extensions + forecast enrichment | 4876c49 | forecast.py, forecast_service.py, forecasts.py |
| 2 | Comparison panel + snapshot endpoints | c90d623 | calibration.py |

## Verification Results

- ForecastResponse validates with/without polymarket_comparison (Optional field)
- JSON round-trip serialization confirmed
- 6 GET routes in calibration.py (4 existing + 2 new)
- Both new endpoints registered under /polymarket/comparisons paths
- All endpoints require API key auth
- No N+1 query patterns

## Next Phase Readiness

Plan 18-03 (Frontend ComparisonPanel + Badges) can proceed. The API contract is fully defined:
- `/forecasts/top` responses include `polymarket_comparison` field
- `/calibration/polymarket/comparisons` serves the full comparison list
- `/calibration/polymarket/comparisons/{id}/snapshots` serves sparkline data
