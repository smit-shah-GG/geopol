---
phase: 14-backend-api-hardening
plan: 01
subsystem: database, api
tags: [alembic, postgresql, tsvector, gin-index, fixtures, orm, forecast-requests]

# Dependency graph
requires:
  - phase: 13-calibration-monitoring-hardening
    provides: PostgreSQL schema (revision 003), calibration_weight_history, polymarket tables
provides:
  - ForecastRequest ORM model for queue-based forecast submission
  - predictions.question_tsv TSVECTOR column with GIN index for full-text search
  - Alembic migration 004 (forecast_requests + tsvector)
  - Fixture-free GET endpoints (PostgreSQL-only by default)
  - USE_FIXTURES dev flag in settings.py
affects: [14-02, 14-03, 14-04, 15-url-routing-dashboard, 16-globe-forecasts]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "USE_FIXTURES env flag for dev-only fixture fallback"
    - "PostgreSQL GENERATED ALWAYS AS ... STORED for computed tsvector columns"
    - "GIN index for full-text search on predictions"

key-files:
  created:
    - alembic/versions/20260303_004_forecast_requests_and_tsvector.py
  modified:
    - src/db/models.py
    - src/settings.py
    - src/api/routes/v1/forecasts.py

key-decisions:
  - "use_fixtures=False by default; production never sees fixture data"
  - "question_tsv uses Computed(persisted=True) generated column rather than trigger-maintained"
  - "Bare except blocks removed from GET production paths; PostgreSQL errors propagate to 500"

patterns-established:
  - "Feature flag pattern: env var gates dev-only behavior via pydantic-settings"
  - "Raw SQL via op.execute() for PostgreSQL-specific DDL in Alembic migrations"

# Metrics
duration: 5min
completed: 2026-03-03
---

# Phase 14 Plan 01: Schema Foundation + Fixture Removal Summary

**ForecastRequest ORM + tsvector search column + fixture-free GET endpoints gated by USE_FIXTURES flag**

## Performance

- **Duration:** 5 min
- **Started:** 2026-03-03T07:27:30Z
- **Completed:** 2026-03-03T07:32:55Z
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments
- ForecastRequest model with full lifecycle tracking (pending -> confirmed -> processing -> complete | failed)
- predictions.question_tsv TSVECTOR computed column with GIN index for full-text search
- All three GET forecast endpoints return PostgreSQL-only results by default -- Myanmar-under-Syria bleed-through bug is now impossible
- USE_FIXTURES=1 env var preserves fixture fallback for local development

## Task Commits

Each task was committed atomically:

1. **Task 1: Alembic migration 004 + ORM model updates** - `248634b` (feat)
2. **Task 2: Remove fixture fallback + add USE_FIXTURES flag** - `a0fde13` (feat)

## Files Created/Modified
- `alembic/versions/20260303_004_forecast_requests_and_tsvector.py` - Migration 004: forecast_requests table + question_tsv generated column + GIN index
- `src/db/models.py` - ForecastRequest model + Prediction.question_tsv Computed column
- `src/settings.py` - use_fixtures bool flag (default False)
- `src/api/routes/v1/forecasts.py` - Fixture fallback gated behind use_fixtures; bare except blocks removed from GET paths

## Decisions Made
- Used PostgreSQL GENERATED ALWAYS AS ... STORED for question_tsv rather than a trigger -- simpler maintenance, automatic consistency
- Removed bare `except Exception` blocks from all three GET endpoints' production paths -- PostgreSQL errors now propagate as 500s instead of being silently swallowed with fixture fallback
- Preserved all fixture code (`_get_fixture_cache`, `_guess_country_iso`, imports) behind the USE_FIXTURES flag rather than deleting -- matches CONTEXT decision

## Deviations from Plan
None -- plan executed exactly as written.

## Issues Encountered
- `test_forecast_persistence.py` (2 tests) now fail with `column predictions.question_tsv does not exist` because the ORM includes the new column but migration 004 hasn't been run on the live database yet. These will pass after `alembic upgrade head` (requires Docker daemon running).
- Pre-existing failures in `test_concurrent_db.py` (event loop), `test_tkg_predictor.py` (CUDA OOM), and `test_tkg_integration.py` (attribute error) are unrelated.

## User Setup Required
None -- no external service configuration required. Migration 004 will be applied when Docker/PostgreSQL is running via `uv run alembic upgrade head`.

## Next Phase Readiness
- Schema foundation in place: Plans 02-04 can build on forecast_requests table and question_tsv index
- GET endpoints are now PostgreSQL-only -- Plan 02 (country risk aggregation) and Plan 03 (POST endpoint hardening) can proceed
- Migration must be applied to PostgreSQL before integration tests pass: `sudo systemctl start docker && uv run alembic upgrade head`

---
*Phase: 14-backend-api-hardening*
*Completed: 2026-03-03*
