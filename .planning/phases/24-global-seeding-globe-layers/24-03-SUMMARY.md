---
phase: 24-global-seeding-globe-layers
plan: 03
subsystem: ingest
tags: [fips, iso, gdelt, lat-lon, geocoding, advisory, postgresql, upsert]

# Dependency graph
requires:
  - phase: 24-01
    provides: src/seeding/fips.py with FIPS_TO_ISO mapping and fips_to_iso() converter
  - phase: 24-02
    provides: TravelAdvisory ORM model, SQLite lat/lon columns in schema, Event dataclass lat/lon fields
provides:
  - FIPS-to-ISO conversion at GDELT ingestion time (no more wrong country codes)
  - lat/lon coordinate extraction from ActionGeo fields
  - Retroactive FIPS-to-ISO migration for all existing events on startup
  - Advisory persistence to PostgreSQL travel_advisories table via UPSERT
affects: [24-04 seeding engine reads ISO codes + lat/lon + travel_advisories, 24-05 API endpoints, 24-06 globe heatmap layer]

# Tech tracking
tech-stack:
  added: []
  patterns: [FIPS-to-ISO at ingestion boundary, retroactive migration on startup, dual-write advisory pattern]

key-files:
  created: []
  modified:
    - src/ingest/gdelt_poller.py
    - src/database/storage.py
    - src/ingest/advisory_poller.py

key-decisions:
  - "fips_to_iso() called on raw FIPS code before assignment to country_iso -- conversion at the ingestion boundary"
  - "Retroactive migration runs on every EventStorage startup but is idempotent (already-converted rows skip)"
  - "3-letter alpha-3 codes handled via pycountry in _migrate_fips_to_iso() -- 64 unique in existing DB"
  - "Advisory DB persistence is non-critical -- wrapped in try/except, in-memory store still works if PG is down"
  - "pg_insert().on_conflict_do_update() uses named constraint uq_travel_advisory_country_source"

patterns-established:
  - "Dual-write advisory pattern: in-memory AdvisoryStore for same-process API, PostgreSQL for cross-process heavy jobs"

# Metrics
duration: 2min
completed: 2026-03-08
---

# Phase 24 Plan 03: FIPS-to-ISO Conversion at Ingestion + Advisory Persistence Summary

**GDELT poller converts FIPS codes to ISO at ingestion, extracts lat/lon, retroactively migrates existing events; advisory poller dual-writes to PostgreSQL travel_advisories table**

## Performance

- **Duration:** 2 min
- **Started:** 2026-03-08T12:34:23Z
- **Completed:** 2026-03-08T12:36:36Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments
- GDELT poller now imports `fips_to_iso` from `src.seeding.fips` and converts raw FIPS country codes to ISO alpha-2 at ingestion time -- 142 of 251 FIPS codes that previously mapped to the wrong country (e.g., IS = Israel not Iceland) are now correct
- ActionGeo_Lat and ActionGeo_Long extracted from GDELT CSV rows into Event lat/lon fields, enabling the heatmap globe layer
- EventStorage._migrate_columns() adds lat/lon columns to existing SQLite databases on startup
- EventStorage._migrate_fips_to_iso() retroactively converts all existing event country codes: 2-letter FIPS via FIPS_TO_ISO dict, 3-letter alpha-3 via pycountry
- Advisory poller dual-writes: in-memory AdvisoryStore (unchanged) + PostgreSQL travel_advisories table via UPSERT on (country_iso, source)

## Task Commits

Each task was committed atomically:

1. **Task 1: FIPS conversion + lat/lon extraction** - `8a01609` (feat)
2. **Task 2: Advisory persistence to PostgreSQL** - `abff9ac` (feat)

## Files Created/Modified
- `src/ingest/gdelt_poller.py` - Import fips_to_iso, convert country codes at ingestion, extract ActionGeo_Lat/Long
- `src/database/storage.py` - Add lat/lon column migration, add _migrate_fips_to_iso() retroactive conversion
- `src/ingest/advisory_poller.py` - Add _persist_advisories_to_db() with UPSERT, call after AdvisoryStore.update()

## Decisions Made
- fips_to_iso() is called at the ingestion boundary (in _gdelt_row_to_event) rather than at query time -- ensures all stored data uses ISO codes consistently
- Retroactive migration is idempotent and runs on every startup -- already-converted rows won't match FIPS codes
- Advisory DB write is non-critical: try/except prevents PostgreSQL issues from crashing the advisory poller
- Advisories with None country_iso (unmapped country names) are filtered out before DB write
- COUNTRY_NAME_TO_ISO dict (lines 44-271) in advisory_poller.py left completely untouched per constraint

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None.

## User Setup Required
None - changes are backward-compatible. lat/lon columns auto-migrate on startup. Advisory persistence requires PostgreSQL to be running (non-critical if not).

## Next Phase Readiness
- Seeding engine (Plan 04) can now read ISO-correct country codes from SQLite events
- Seeding engine can read lat/lon for H3 hex-binning (heatmap layer)
- Seeding engine can read travel_advisories from PostgreSQL (cross-process access)
- All existing events will be retroactively converted on next startup

---
*Phase: 24-global-seeding-globe-layers*
*Completed: 2026-03-08*
