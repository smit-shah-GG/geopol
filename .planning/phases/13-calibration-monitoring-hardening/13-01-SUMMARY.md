---
phase: 13-calibration-monitoring-hardening
plan: 01
subsystem: database
tags: [sqlalchemy, alembic, postgresql, pydantic-settings, calibration, polymarket]

# Dependency graph
requires:
  - phase: 09-api-foundation
    provides: Base ORM models (Prediction, CalibrationWeight, etc.), Settings singleton
provides:
  - CalibrationWeightHistory ORM model for versioned weight audit
  - PolymarketComparison and PolymarketSnapshot ORM models for market benchmarking
  - Prediction.cameo_root_code column for per-CAMEO weight lookup
  - CalibrationWeight.cameo_code widened to 30 chars for hierarchical keys
  - 19 new settings fields (calibration, monitoring, polymarket, logging)
  - Alembic migration 003 for all schema changes
affects: [13-02 (calibration weight recompute), 13-03 (monitoring alerts), 13-04 (Polymarket poller), 13-05 (log rotation), 13-06 (hardening), 13-07 (integration)]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Hierarchical CAMEO keys in calibration_weights: root codes (01-20), super-categories (super:verbal_coop), global"
    - "Alembic alter_column for in-place VARCHAR widening"
    - "ForeignKey constraint from polymarket_snapshots -> polymarket_comparisons"

key-files:
  created:
    - alembic/versions/20260302_001_phase13_schema.py
  modified:
    - src/db/models.py
    - src/settings.py

key-decisions:
  - "CalibrationWeight.cameo_code widened to String(30) -- supports hierarchical keys without separate table"
  - "PolymarketSnapshot uses ForeignKey to polymarket_comparisons.id -- cascade semantics at DB level"
  - "smtp_password as plain str, not SecretStr -- consistent with existing extra=ignore pattern"

patterns-established:
  - "Phase 13 settings grouped by subsystem with comment headers: calibration, monitoring, polymarket, logging"
  - "Alembic migration numbering: 003 continues sequential from 001/002"

# Metrics
duration: 3min
completed: 2026-03-02
---

# Phase 13 Plan 01: Schema & Settings Foundation Summary

**3 new ORM models (CalibrationWeightHistory, PolymarketComparison, PolymarketSnapshot), widened CAMEO key column, 19 new settings fields, and Alembic migration 003**

## Performance

- **Duration:** 3 min
- **Started:** 2026-03-02T06:12:34Z
- **Completed:** 2026-03-02T06:16:03Z
- **Tasks:** 3
- **Files modified:** 3

## Accomplishments
- Extended ORM with 3 new tables and 2 column additions (cameo_root_code on Prediction, widened cameo_code on CalibrationWeight)
- Added 19 settings fields across 4 subsystems (calibration, monitoring/alerting, Polymarket, log rotation)
- Created Alembic migration 003 with full upgrade/downgrade covering all schema changes

## Task Commits

Each task was committed atomically:

1. **Task 1: Extend ORM models with Phase 13 tables and columns** - `7f5baef` (feat)
2. **Task 2: Extend settings with Phase 13 configuration fields** - `b902321` (feat)
3. **Task 3: Create Alembic migration for schema changes** - `64a8938` (feat)

## Files Created/Modified
- `src/db/models.py` - Added CalibrationWeightHistory, PolymarketComparison, PolymarketSnapshot ORM models; widened CalibrationWeight.cameo_code to String(30); added Prediction.cameo_root_code
- `src/settings.py` - 19 new fields: calibration (min_samples, max_deviation, recompute_day), monitoring (SMTP, alert cooldown, staleness/drift/disk thresholds), Polymarket (enabled, poll_interval, match_threshold), logging (log_dir, log_retention_days)
- `alembic/versions/20260302_001_phase13_schema.py` - Migration 003: alter calibration_weights.cameo_code, add predictions.cameo_root_code, create 3 new tables with indexes and FK

## Decisions Made
- CalibrationWeight.cameo_code widened to String(30) in-place rather than introducing a separate hierarchy table -- keys like "super:verbal_coop" and "global" fit within 30 chars and avoid join overhead
- PolymarketSnapshot.comparison_id uses a real ForeignKey constraint to polymarket_comparisons.id for referential integrity (first FK in the schema)
- smtp_password kept as plain `str` (not `SecretStr`) for consistency with existing `extra="ignore"` pydantic-settings pattern; value sourced from SMTP_PASSWORD env var in production

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required. All new settings have sensible development defaults.

## Next Phase Readiness
- All 3 ORM models importable and verified
- All 19 settings fields accessible with defaults
- Alembic migration ready for execution when Docker/PostgreSQL is available
- Plans 02-07 can now reference these models and settings directly

---
*Phase: 13-calibration-monitoring-hardening*
*Completed: 2026-03-02*
