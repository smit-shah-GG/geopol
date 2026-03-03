---
phase: 14-backend-api-hardening
plan: 02
subsystem: api
tags: [postgresql, cte, sqlalchemy, risk-score, cameo, exponential-decay]

# Dependency graph
requires:
  - phase: 14-01
    provides: "Prediction model with country_iso, cameo_root_code, expires_at; ix_predictions_country_created composite index"
  - phase: 09-api-foundation
    provides: "CountryRiskSummary DTO, countries router, ForecastCache service"
provides:
  - "CTE-based country risk aggregation from predictions table (0-100 composite score)"
  - "CountryRiskSummary DTO with 0-100 risk_score scale and top_forecast field"
  - "Exponential time decay (7-day half-life) on probability and severity aggregation"
  - "CAMEO root code -> Goldstein severity SQL CASE mapping"
  - "7-day delta trend computation (rising/stable/falling)"
affects: [15-url-routing-dashboard, 16-globe-forecasts-screens]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "CTE-based SQL aggregation via sqlalchemy.text() for complex analytical queries"
    - "CAMEO-to-severity SQL CASE expression for Goldstein severity weighting"
    - "Exponential time decay in SQL: EXP(-0.693 * age_seconds / half_life)"
    - "DISTINCT ON for per-group top-N selection (PostgreSQL-specific)"

key-files:
  created: []
  modified:
    - "src/api/schemas/country.py"
    - "src/api/routes/v1/countries.py"
    - "src/api/fixtures/factory.py"

key-decisions:
  - "Used sqlalchemy.text() over Core expressions for CTE query readability -- 4-CTE query would be unreadable as chained Core selects"
  - "Kept USE_FIXTURES fallback inline (not in separate module) -- same pattern as forecasts.py for consistency"
  - "Fixture scores scaled to 0-100 range (e.g. SY=87.0 not 0.87) to match new DTO contract"
  - "top_question renamed to top_forecast per BAPI-02 requirement spec -- breaking change for frontend TypeScript types (Phase 15)"

patterns-established:
  - "countries: namespace for cache keys (countries:all, countries:{ISO})"
  - "Row-to-DTO mapping via _row_to_dto() for text() query result translation"

# Metrics
duration: 5min
completed: 2026-03-03
---

# Phase 14 Plan 02: Country Risk Aggregation Endpoint Summary

**CTE-based PostgreSQL aggregation replacing hardcoded mock countries with composite risk scores (0-100) from predictions table, using exponential time decay and CAMEO severity weighting**

## Performance

- **Duration:** 5 min
- **Started:** 2026-03-03T07:36:33Z
- **Completed:** 2026-03-03T07:41:33Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments
- Rewrote GET /countries and GET /countries/{iso_code} to query PostgreSQL predictions table via a 4-CTE SQL aggregation query
- Composite risk score (0-100) combines: forecast count (0-20 pts, caps at 5), decay-weighted probability (0-50 pts), CAMEO Goldstein severity (0-30 pts)
- Exponential time decay (7-day half-life) ensures recent predictions dominate aggregation
- Trend computed from 7-day delta comparison: >5pt change = rising/falling, else stable
- Updated CountryRiskSummary DTO from 0-1 to 0-100 scale, renamed top_question to top_forecast

## Task Commits

Each task was committed atomically:

1. **Task 1: Update CountryRiskSummary DTO to 0-100 scale** - `12992b8` (feat)
2. **Task 2: Rewrite countries.py with PostgreSQL CTE aggregation** - `12d08fa` (feat)

## Files Created/Modified
- `src/api/schemas/country.py` - CountryRiskSummary DTO: risk_score 0-100, top_forecast field, PostgreSQL-aware descriptions
- `src/api/routes/v1/countries.py` - Complete rewrite: 4-CTE SQL aggregation, cache integration, fixture fallback gated
- `src/api/fixtures/factory.py` - Updated create_mock_country_risk to use top_forecast field name

## Decisions Made
- Used `sqlalchemy.text()` for the CTE query rather than SQLAlchemy Core expressions -- the 4-CTE analytical query with CASE expressions, window functions, and cross-CTE joins would be far less readable in Core syntax
- Renamed `top_question` to `top_forecast` per BAPI-02 spec -- this is a breaking change for the frontend TypeScript type `CountryRiskSummary` in `frontend/src/types/api.ts` and `frontend/src/components/RiskIndexPanel.ts`, deferred to Phase 15
- Fixture data scaled to 0-100 range (SY=87.0, UA=82.0, etc.) to match the new DTO contract

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Updated fixture factory for renamed field**
- **Found during:** Task 1 (CountryRiskSummary DTO update)
- **Issue:** `create_mock_country_risk` in `factory.py` used `top_question=` which would fail validation against the renamed `top_forecast` field
- **Fix:** Changed `top_question=` to `top_forecast=` in the factory function
- **Files modified:** `src/api/fixtures/factory.py`
- **Verification:** Factory function produces valid DTOs
- **Committed in:** `12992b8` (Task 1 commit)

---

**Total deviations:** 1 auto-fixed (1 bug)
**Impact on plan:** Essential fix to keep fixture factory consistent with DTO rename. No scope creep.

## Issues Encountered
None -- plan executed cleanly. Pre-existing test failures in `test_concurrent_db.py` (asyncio loop reuse) and `test_tkg_integration.py` (TKGPredictor API mismatch) are unrelated to this plan.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Country risk endpoints now serve real aggregated data from PostgreSQL
- Frontend TypeScript types need updating in Phase 15 (`top_question` -> `top_forecast`, `risk_score` scale change)
- The risk score formula and CAMEO severity mapping are tunable constants in the SQL -- can be adjusted based on real data feedback
- `ix_predictions_country_created` composite index covers the GROUP BY + ORDER BY access pattern

---
*Phase: 14-backend-api-hardening*
*Completed: 2026-03-03*
