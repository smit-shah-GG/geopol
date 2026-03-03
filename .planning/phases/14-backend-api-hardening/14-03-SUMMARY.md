---
phase: 14-backend-api-hardening
plan: 03
subsystem: api
tags: [postgresql, tsvector, full-text-search, gin-index, fastapi, pydantic]

# Dependency graph
requires:
  - phase: 14-01
    provides: "Prediction.question_tsv TSVECTOR column + GIN index (migration 004)"
provides:
  - "GET /api/v1/forecasts/search endpoint with tsvector full-text search"
  - "SearchResult DTO wrapping ForecastResponse with relevance score"
  - "SearchResponse DTO with total count, query echo, nullable suggestions field"
affects: [15-url-routing-dashboard, 16-globe-forecasts-screens]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "PostgreSQL plainto_tsquery + ts_rank for full-text search (no LIKE/ILIKE)"
    - "SearchResult wrapper DTO pattern for augmenting existing DTOs with metadata"

key-files:
  created:
    - src/api/schemas/search.py
  modified:
    - src/api/routes/v1/forecasts.py
    - src/api/schemas/__init__.py

key-decisions:
  - "plainto_tsquery over to_tsquery -- safe for natural language input, no injection risk"
  - "Separate count query before pagination rather than window function -- simpler, acceptable at v2.1 scale"
  - "suggestions field nullable/None by default -- prevents breaking DTO change when LLM suggestions added later"
  - "Route order: /search before /{forecast_id} to prevent path capture"

patterns-established:
  - "Search endpoint pattern: tsvector match + ts_rank ordering + count subquery + limit"
  - "Reserved nullable field pattern: add field with None default for planned-but-unimplemented features"

# Metrics
duration: 3min
completed: 2026-03-03
---

# Phase 14 Plan 03: Full-Text Search Endpoint Summary

**GET /forecasts/search using PostgreSQL tsvector + GIN index with ts_rank relevance ordering and optional country/category filters**

## Performance

- **Duration:** 3 min
- **Started:** 2026-03-03T07:37:22Z
- **Completed:** 2026-03-03T07:40:00Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments
- SearchResult and SearchResponse Pydantic V2 DTOs with proper from_attributes config
- GET /api/v1/forecasts/search endpoint using plainto_tsquery + ts_rank over question_tsv GIN-indexed column
- Optional country and category query parameter filters
- Total count returned for frontend pagination UI
- Nullable suggestions field reserved for future LLM search suggestions
- Route ordering prevents /{forecast_id} from capturing /search

## Task Commits

Each task was committed atomically:

1. **Task 1: Create SearchResult DTO** - `6b47d86` (feat)
2. **Task 2: Add search endpoint to forecasts router** - `ccab167` (feat)

## Files Created/Modified
- `src/api/schemas/search.py` - SearchResult (forecast + relevance) and SearchResponse (results + total + query + suggestions) DTOs
- `src/api/routes/v1/forecasts.py` - GET /search endpoint with tsvector full-text search, country/category filters, ts_rank ordering
- `src/api/schemas/__init__.py` - Re-exports for SearchResult and SearchResponse

## Decisions Made
- Used `plainto_tsquery` over `to_tsquery` -- safe for arbitrary natural-language input, no tsquery syntax injection risk, handles stop-words gracefully (returns empty results, not errors)
- Separate count query via subquery rather than window function -- simpler implementation, acceptable performance at v2.1 scale (thousands of predictions, not millions)
- `suggestions` field is `Optional[list[str]]` defaulting to `None` -- placeholder for future LLM-powered search suggestions, prevents breaking DTO change when implemented
- Search route registered before `/{forecast_id}` in router definition order to prevent FastAPI path parameter capture

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- Search endpoint ready for frontend integration (Phase 15/16)
- GIN index from migration 004 (Plan 01) provides the performance foundation
- Plan 04 (pagination + filtering) can build on this search infrastructure
- All 95 existing tests pass; 1 pre-existing Docker-dependent test skipped

---
*Phase: 14-backend-api-hardening*
*Completed: 2026-03-03*
