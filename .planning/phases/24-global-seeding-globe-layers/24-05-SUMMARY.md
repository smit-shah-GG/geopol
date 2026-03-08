---
phase: 24-global-seeding-globe-layers
plan: 05
subsystem: api
tags: [dual-score, baseline-risk, forecast-risk, blended-risk, globe-layers, heatmap, arcs, deltas, h3, pydantic]

dependency_graph:
  requires:
    - "24-03 (FIPS-to-ISO at ingestion, advisory persistence)"
    - "24-04 (seeding computation engine: baseline_country_risk, heatmap_hexbins, country_arcs, risk_deltas tables populated)"
  provides:
    - "Dual-score CountryRiskSummary API (baseline + forecast + blended for all ~195 countries)"
    - "Globe layer endpoints: /globe/heatmap, /globe/arcs, /globe/deltas"
    - "Backward-compatible risk_score alias for blended_risk"
  affects:
    - "24-06 (frontend globe layer wiring consumes these endpoints)"
    - "Frontend choropleth coloring (now receives all ~195 countries, not just ~8-15)"

tech_stack:
  added: []
  patterns:
    - "Two-query Python-side merge: baseline table scan + forecast CTE, merged in Python"
    - "LayerEnvelope wrapper: computed_at timestamp for frontend staleness display"
    - "Nullable DTO fields for optional forecast data (top_forecast, top_probability, forecast_risk)"

key_files:
  created:
    - src/api/routes/v1/layers.py
  modified:
    - src/api/schemas/country.py
    - src/api/routes/v1/countries.py
    - src/api/routes/v1/router.py

decisions:
  - id: "two-query-merge"
    description: "Baseline table scan (~195 rows) + forecast CTE, merged in Python rather than massive SQL JOIN"
    rationale: "Both queries are well-tested independently; Python merge is readable and the row counts are trivially small"
  - id: "envelope-wrapper"
    description: "Layer endpoints return envelope objects (HeatmapEnvelope, ArcsEnvelope, DeltasEnvelope) with computed_at timestamp"
    rationale: "Frontend needs staleness display ('Updated Xh ago'); flat list response lacks metadata"
  - id: "no-404-baseline-only"
    description: "GET /countries/{iso} returns baseline-only data for countries without forecasts instead of 404"
    rationale: "Baseline-only countries are valid entries -- 404 should only fire for truly unknown ISO codes"

metrics:
  duration: "~4 min"
  completed: "2026-03-08"
---

# Phase 24 Plan 05: API Endpoints (Dual-Score Countries + Globe Layers) Summary

**One-liner:** Dual-score CountryRiskSummary (baseline/forecast/blended) for all ~195 countries + 3 globe layer endpoints (heatmap hexbins, arcs, risk deltas) reading pre-computed PostgreSQL data.

## Performance

- **Duration:** 4 min
- **Started:** 2026-03-08T12:42:58Z
- **Completed:** 2026-03-08T12:46:39Z
- **Tasks:** 2
- **Files created:** 1
- **Files modified:** 3

## Accomplishments

### Task 1: Dual-Score Schema + Merged Countries Endpoint

**CountryRiskSummary schema:**
- Added `baseline_risk` (float, 0-100, always present)
- Added `forecast_risk` (float | None, only when active predictions exist)
- Added `blended_risk` (float, 0-100, = 0.7*forecast + 0.3*baseline or baseline alone)
- Changed `risk_score` to backward-compat alias for `blended_risk`
- Made `top_forecast` and `top_probability` nullable (baseline-only countries have None)
- Added `disputed` (bool, default False, for XK/TW/PS/EH)

**Countries endpoint rewrite:**
- `list_countries()` now queries `baseline_country_risk` for ALL ~195 countries, then merges with forecast CTE
- Returns every country from baseline table, not just the ~8-15 with active predictions
- `get_country_risk()` returns baseline-only data instead of 404 for countries without forecasts
- Blend formula: `_blend_risk(baseline, forecast)` = 0.7*forecast + 0.3*baseline when both, baseline alone otherwise
- Fixture data updated with new schema fields (baseline_risk, blended_risk, disputed)
- Cache layer preserved (same keys, same TTL)

### Task 2: Globe Layer Endpoints

**layers.py (3 endpoints):**
- `GET /globe/heatmap` -> `HeatmapEnvelope` with H3 hexbin data from `heatmap_hexbins` table
- `GET /globe/arcs` -> `ArcsEnvelope` with bilateral relationship arcs from `country_arcs` table
- `GET /globe/deltas` -> `DeltasEnvelope` with significant risk changes from `risk_deltas` table
- All endpoints return empty envelope (not error) when seeding hasn't run yet
- All endpoints require X-API-Key authentication
- Response models defined inline as simple Pydantic DTOs

**router.py:**
- Layers router registered at `/globe` prefix with `globe-layers` tag
- Full paths: `/api/v1/globe/heatmap`, `/api/v1/globe/arcs`, `/api/v1/globe/deltas`

## Deviations from Plan

None -- plan executed exactly as written.

## Verification Results

| Check | Result |
|-------|--------|
| CountryRiskSummary nullable fields (baseline-only) | PASS |
| CountryRiskSummary with forecast (70/30 blend) | PASS |
| Disputed territory flag | PASS |
| _blend_risk(60, 82) = 75.4 | PASS |
| _blend_risk(80, None) = 80.0 | PASS |
| layers router has 3 routes | PASS |
| HexbinResponse model validates | PASS |
| ArcResponse model validates | PASS |
| RiskDeltaResponse model validates | PASS |
| Globe mount in router.py | PASS |
| baseline_country_risk query in countries.py | PASS |
| blended_risk merge logic in countries.py | PASS |

## Commits

| # | Hash | Message |
|---|------|---------|
| 1 | f52696f | feat(24-05): dual-score CountryRiskSummary schema + merged countries endpoint |
| 2 | e7545a3 | feat(24-05): globe layer endpoints (heatmap, arcs, deltas) |

## Next Phase Readiness

- Frontend globe layer wiring (Plan 06) can consume all 4 API endpoints
- Choropleth now receives ~195 countries instead of ~8-15 (visual coverage is universal)
- Heatmap, arcs, and deltas layers have data endpoints ready for deck.gl layer consumption
- computed_at timestamp enables "Updated Xh ago" staleness display in frontend

---
*Phase: 24-global-seeding-globe-layers*
*Completed: 2026-03-08*
