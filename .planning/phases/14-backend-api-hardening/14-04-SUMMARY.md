---
phase: 14-backend-api-hardening
plan: 04
subsystem: api
tags: [fastapi, gemini, asyncio, submission-queue, llm-parsing, background-worker]

# Dependency graph
requires:
  - phase: 14-01
    provides: ForecastRequest ORM model, forecast_requests table
  - phase: 14-03
    provides: Full-text search patterns, route ordering under /forecasts prefix
provides:
  - POST /submit endpoint with LLM question parsing
  - POST /submit/{id}/confirm with background worker trigger
  - GET /requests endpoint for per-user queue visibility
  - Async submission worker with bounded parallelism (Semaphore(3))
  - Question parser service with ISO code validation
affects: [15-url-routing-dashboard, 16-globe-forecasts-screens]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Two-phase submit/confirm flow for user-facing forecast generation"
    - "Async background worker via asyncio.create_task + Semaphore (no Celery)"
    - "SELECT FOR UPDATE SKIP LOCKED for race-condition-free request claiming"
    - "Graceful LLM parse fallback (never crash on Gemini failure)"

key-files:
  created:
    - src/api/schemas/submission.py
    - src/api/services/question_parser.py
    - src/api/services/submission_worker.py
    - src/api/routes/v1/submissions.py
  modified:
    - src/api/routes/v1/router.py
    - src/api/schemas/__init__.py

key-decisions:
  - "asyncio.Semaphore(3) over Celery/Redis queue -- complexity not justified for single-server deployment"
  - "Two-phase submit/confirm flow -- user reviews LLM-parsed interpretation before committing API budget"
  - "SELECT FOR UPDATE SKIP LOCKED for worker -- prevents double-pickup without blocking"
  - "Graceful parse fallback to XX/30d/GENERAL -- failed LLM parse must never block submission"
  - "Budget check before processing, not before parsing -- parsing is cheap, forecasting is expensive"

patterns-established:
  - "Background worker pattern: schedule_processing() -> asyncio.create_task -> Semaphore-gated coroutine"
  - "Retry with exponential backoff: call_later + ensure_future bridge for delayed re-scheduling"

# Metrics
duration: 4min
completed: 2026-03-03
---

# Phase 14 Plan 04: Question Submission Queue Summary

**Two-phase submit/confirm flow with LLM parsing, bounded async worker (Semaphore(3)), and exponential backoff retry -- the core BAPI-03 submission pipeline**

## Performance

- **Duration:** 4 min
- **Started:** 2026-03-03T07:46:12Z
- **Completed:** 2026-03-03T07:49:53Z
- **Tasks:** 3
- **Files modified:** 6

## Accomplishments
- Three new submission endpoints: POST /submit (LLM parsing), POST /confirm (queue trigger), GET /requests (status listing)
- Gemini-powered question parser extracts country ISO codes, horizon days, and CAMEO category from natural language
- Async background worker processes confirmed requests with bounded parallelism (max 3 concurrent via asyncio.Semaphore)
- Retry logic: 3 attempts with exponential backoff (30s, 2min, 10min) before permanent failure
- Gemini budget enforcement: worker checks remaining budget before consuming API calls

## Task Commits

Each task was committed atomically:

1. **Task 1: DTOs + question parser service** - `8f4c79d` (feat)
2. **Task 2: Submission endpoints + router registration** - `03fe4e1` (feat)
3. **Task 3: Async submission worker with bounded parallelism** - `eb113d5` (feat)

**Schema re-exports:** `c19f28e` (chore)

## Files Created/Modified
- `src/api/schemas/submission.py` - SubmitQuestionRequest, ParsedQuestionResponse, ForecastRequestStatus, ConfirmSubmissionResponse DTOs
- `src/api/services/question_parser.py` - Gemini LLM question parsing with ISO validation and graceful fallback
- `src/api/services/submission_worker.py` - Async background worker with Semaphore(3), retry logic, budget checks
- `src/api/routes/v1/submissions.py` - POST /submit, POST /submit/{id}/confirm, GET /requests endpoints
- `src/api/routes/v1/router.py` - Submissions router registered under /forecasts prefix
- `src/api/schemas/__init__.py` - Re-export submission DTOs

## Decisions Made
- asyncio.Semaphore(3) for bounded parallelism instead of Celery/Redis queue -- appropriate for single-server deployment
- Budget check placed after parsing but before forecast execution -- parsing is cheap, forecasting is expensive
- Worker gets its own DB session via get_async_session() since the HTTP handler's session is closed by worker execution time
- SELECT FOR UPDATE SKIP LOCKED prevents race conditions without blocking concurrent workers on different requests
- Graceful parse fallback: on any Gemini failure, returns {country: ["XX"], horizon: 30, category: "GENERAL"} rather than crashing

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Missing Critical] Re-exported submission DTOs from schemas __init__**
- **Found during:** Post-task verification
- **Issue:** New DTOs not re-exported from src/api/schemas/__init__.py, breaking the established convention
- **Fix:** Added imports and __all__ entries for all 4 submission DTOs
- **Files modified:** src/api/schemas/__init__.py
- **Verification:** `from src.api.schemas import SubmitQuestionRequest` succeeds
- **Committed in:** c19f28e

---

**Total deviations:** 1 auto-fixed (1 missing critical)
**Impact on plan:** Convention consistency fix. No scope creep.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Phase 14 (Backend API Hardening) is now complete: all 4 plans delivered
- All BAPI-01..04 requirements implemented
- Phase 15 (URL Routing & Dashboard) can begin -- frontend has all backend endpoints it needs
- Migration 004 must be applied for forecast_requests table before submission endpoints work in production

---
*Phase: 14-backend-api-hardening*
*Completed: 2026-03-03*
