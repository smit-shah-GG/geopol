---
phase: 24-global-seeding-globe-layers
plan: 02
subsystem: database
tags: [sqlalchemy, alembic, postgresql, sqlite, orm, h3, geospatial]

# Dependency graph
requires:
  - phase: 23-historical-backtesting
    provides: Alembic migration chain (009), existing ORM models
provides:
  - 5 PostgreSQL ORM models (BaselineCountryRisk, HeatmapHexbin, CountryArc, RiskDelta, TravelAdvisory)
  - Alembic migration 010 creating 5 new tables
  - SQLite events schema with lat/lon columns
  - Event dataclass with lat/lon fields
affects: [24-03 FIPS conversion, 24-04 seeding engine, 24-05 API endpoints, 24-06 globe layer wiring]

# Tech tracking
tech-stack:
  added: []
  patterns: [pre-computed layer data tables, cross-process advisory persistence via PostgreSQL]

key-files:
  created:
    - alembic/versions/20260308_010_seeding_tables.py
  modified:
    - src/db/models.py
    - src/database/schema.sql
    - src/database/models.py

key-decisions:
  - "TravelAdvisory uses UniqueConstraint on (country_iso, source) for UPSERT semantics"
  - "HeatmapHexbin uses String(20) for H3 index -- accommodates all H3 resolutions"
  - "lat/lon columns on SQLite events are nullable to preserve existing data"

patterns-established:
  - "Pre-computed layer data: heavy job writes to PostgreSQL, API reads latest computed_at rows"
  - "Cross-process data access: advisory data persisted to PostgreSQL for ProcessPoolExecutor workers"

# Metrics
duration: 3min
completed: 2026-03-08
---

# Phase 24 Plan 02: Database Schema for Globe Seeding Summary

**5 PostgreSQL ORM models + Alembic migration 010 for baseline risk, heatmap hexbins, country arcs, risk deltas, and travel advisories; SQLite lat/lon geocoding columns added to events**

## Performance

- **Duration:** 3 min
- **Started:** 2026-03-08T12:27:37Z
- **Completed:** 2026-03-08T12:30:18Z
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments
- 5 new SQLAlchemy ORM models with proper indexes, constraints, and UniqueConstraint for UPSERT semantics
- Alembic migration 010 with full upgrade/downgrade, correct revision chain (down_revision=009)
- SQLite events schema extended with lat/lon REAL columns + composite index for geocoded queries
- Event dataclass extended with optional lat/lon fields preserving backward compatibility

## Task Commits

Each task was committed atomically:

1. **Task 1: Add 5 PostgreSQL ORM models** - `c8ec8d7` (feat)
2. **Task 2: Alembic migration 010 + SQLite lat/lon + Event dataclass** - `ee6e451` (feat)

## Files Created/Modified
- `src/db/models.py` - Added BaselineCountryRisk, HeatmapHexbin, CountryArc, RiskDelta, TravelAdvisory ORM models
- `alembic/versions/20260308_010_seeding_tables.py` - Migration 010 creating 5 PostgreSQL tables with indexes and constraints
- `src/database/schema.sql` - Added lat/lon REAL columns and idx_events_lat_lon index to events table
- `src/database/models.py` - Added lat/lon Optional[float] fields to Event dataclass

## Decisions Made
- TravelAdvisory uses UniqueConstraint on (country_iso, source) pair for UPSERT semantics -- each advisory source (us_state_dept, uk_fcdo) has exactly one row per country
- HeatmapHexbin h3_index column is String(20) -- sufficient for all H3 resolution levels (max 16 chars at res 15)
- SQLite lat/lon columns are nullable REAL -- existing 1.43M events lack coordinates, only new GDELT ingestions populate them
- BaselineCountryRisk has unique constraint on country_iso -- one row per country, overwritten on each hourly recompute
- Added UniqueConstraint import to sqlalchemy imports (was not previously needed)

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None.

## User Setup Required
None - no external service configuration required. Migration 010 will be applied when Docker + PostgreSQL are running (`alembic upgrade head`).

## Next Phase Readiness
- All 5 PostgreSQL tables ready for the seeding engine (Plan 04) to write pre-computed data
- API endpoints (Plan 05) can read from these tables
- lat/lon columns ready for GDELT poller modification (Plan 03) to extract ActionGeo_Lat/Long
- Event dataclass backward-compatible -- existing code that creates Events without lat/lon continues to work

---
*Phase: 24-global-seeding-globe-layers*
*Completed: 2026-03-08*
