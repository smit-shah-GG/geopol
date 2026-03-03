# Phase 17 Plan 01: Data Layer Foundation Summary

---
phase: 17-live-data-feeds-country-depth
plan: 01
subsystem: data-layer
tags: [sqlite, migration, pydantic, dto, cursor-pagination, event-model, settings]
dependency_graph:
  requires: [phase-16]
  provides: [event-query-api-surface, cursor-pagination-generic, dto-contracts, acled-settings]
  affects: [17-02, 17-03, 17-04]
tech_stack:
  added: []
  patterns: [keyset-cursor-pagination, dynamic-sql-filter-builder, alter-table-migration-with-backfill]
key_files:
  created:
    - src/api/schemas/event.py
    - src/api/schemas/article.py
    - src/api/schemas/advisory.py
    - src/api/schemas/source.py
  modified:
    - src/database/schema.sql
    - src/database/models.py
    - src/database/storage.py
    - src/api/schemas/common.py
    - src/ingest/gdelt_poller.py
    - src/settings.py
decisions:
  - Generic keyset cursor (encode_keyset_cursor/decode_keyset_cursor) added alongside existing forecast-specific cursor -- no breaking changes
  - Event backfill produces 0 results because existing 1.37M rows have NULL raw_json -- country_iso will populate only on new ingestion runs
  - query_top_actors uses UNION ALL of actor1_code + actor2_code with GROUP BY dedup -- captures bilateral actor involvement
metrics:
  duration: 7min
  completed: 2026-03-04
---

**One-liner:** SQLite schema migration (country_iso + source columns with composite indexes), 9-parameter EventStorage query surface with keyset pagination, four new Pydantic DTOs (event/article/advisory/source), and ACLED/advisory settings.

## What Was Done

### Task 1: Schema Migration + Event Model + EventStorage Queries + Generic Cursor (8460c03)

**Schema changes:**
- Added `country_iso TEXT` and `source TEXT NOT NULL DEFAULT 'gdelt'` columns to events table
- Created 3 new indexes: `idx_events_country_iso`, `idx_events_country_date_id` (composite for cursor pagination), `idx_events_source`
- Migration runs automatically in `EventStorage.init_database()` via PRAGMA table_info introspection
- Backfill logic extracts country_iso from `json_extract(raw_json, '$.ActionGeo_CountryCode')` with Actor1CountryCode fallback -- backfill yielded 0 results because all 1.37M existing rows have NULL raw_json (GDELT poller never stored raw_json for CSV-ingested events)

**Event model:**
- Added `country_iso: Optional[str] = None` and `source: str = "gdelt"` fields to Event dataclass
- Fields placed in "Multi-source metadata" section before raw_json
- Backward compatible -- both fields have defaults

**EventStorage query methods:**
- `query_events()`: 9-parameter filter surface (country, start_date, end_date, cameo_code, actor, goldstein_min, goldstein_max, text, source) + cursor_id/cursor_date for keyset pagination. Returns limit+1 rows for has_more detection. Default 30-day window.
- `query_events_count()`: Aggregate count for country brief stats
- `query_top_actors()`: UNION ALL of actor1_code + actor2_code with GROUP BY dedup, sorted by frequency

**Generic cursor:**
- `encode_keyset_cursor(**kwargs)`: Accepts arbitrary keyword args, JSON-encodes to base64url
- `decode_keyset_cursor(cursor, required_keys)`: Decodes and validates presence of required keys
- Existing `encode_cursor`/`decode_cursor` preserved untouched for forecast route backward compatibility

**GDELT poller:**
- `_gdelt_row_to_event()` now extracts `country_iso` from ActionGeo_CountryCode with Actor1CountryCode fallback
- Explicitly sets `source="gdelt"` on Event construction

### Task 2: Pydantic DTOs + Settings (33ea5b4)

**DTOs created:**
- `EventDTO`: Full event field surface including country_iso and source discriminator
- `ArticleDTO`: RSS article with chunk_id, source_feed, snippet, optional relevance_score
- `AdvisoryDTO`: Government advisory with normalised 1-4 risk level, source/country/summary
- `SourceStatusDTO`: Ingestion pipeline health with name, healthy flag, last_update, events_last_run, detail

**Settings additions:**
- `acled_email`, `acled_password`: ACLED API credentials (env: ACLED_EMAIL, ACLED_PASSWORD)
- `acled_poll_interval`: Daily polling interval (default 86400s)
- `acled_event_types`: Filtered event types (default: Battles, Explosions/Remote violence, Violence against civilians)
- `advisory_poll_interval`: Advisory polling interval (default 86400s)

## Deviations from Plan

None -- plan executed exactly as written.

## Verification Results

- All 5 modified + 4 created Python files import cleanly
- Event model constructs with new fields: `Event(content_hash='x', time_window='t', event_date='2026-01-01', country_iso='US', source='gdelt')`
- EventStorage.query_events() returns results with parameterized filters and default 30-day window
- encode_keyset_cursor/decode_keyset_cursor roundtrip verified: `{id: 42, event_date: "2026-01-15"}`
- Legacy encode_cursor/decode_cursor unchanged and verified
- All 4 DTO JSON schemas generated successfully
- Settings reflect ACLED defaults (empty email, 86400s poll, 3 event types)
- SQLite PRAGMA table_info confirms country_iso and source columns present
- Existing test suite: 304 passed, 20 failed (all pre-existing failures unrelated to this plan), 9 skipped

## Next Phase Readiness

This plan provides the complete data layer for all subsequent Phase 17 plans:
- Plan 02 (API routes): Can use EventStorage.query_events(), EventDTO, PaginatedResponse, encode_keyset_cursor for GET /events endpoint
- Plan 03 (Ingestion daemons): ACLED settings and Event model with source="acled" ready
- Plan 04 (Frontend wiring): EventDTO schema matches the API contract
