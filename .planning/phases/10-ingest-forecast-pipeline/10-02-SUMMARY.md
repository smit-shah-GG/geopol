---
phase: 10-ingest-forecast-pipeline
plan: 02
subsystem: api
tags: [redis, cachetools, rate-limiting, input-sanitization, prompt-injection, fastapi-depends]

# Dependency graph
requires:
  - phase: 09-api-foundation
    provides: FastAPI app factory, deps.py, auth middleware, Settings with redis_url
provides:
  - Three-tier forecast cache service (memory -> Redis -> PostgreSQL)
  - Per-API-key daily rate limiting via Redis counters
  - Input sanitization with injection blocklist and geopolitical keyword validation
  - Gemini budget tracking functions for daily pipeline
  - NullRedis stub for graceful Redis unavailability
affects: [10-03, 10-04, 12-frontend, 13-calibration]

# Tech tracking
tech-stack:
  added: [cachetools 7.0.1, pytest-asyncio 1.3.0]
  patterns: [three-tier cache hierarchy, fail-open rate limiting, NullRedis stub for degradation, centralized cache key generation]

key-files:
  created:
    - src/api/services/cache_service.py
    - src/api/middleware/rate_limit.py
    - src/api/middleware/sanitize.py
    - tests/test_api_hardening.py
  modified:
    - src/api/deps.py
    - pyproject.toml

key-decisions:
  - "Cache key generators centralize all prefixing; get/set use keys as-is (prevents double-prefix bugs)"
  - "NullRedis stub enables graceful degradation without Optional branching in hot path"
  - "Rate limiter fail-open: Redis failure allows request through (rate limiter must never kill the API)"
  - "Sanitization uses substring blocklist (not regex) for injection detection -- simpler, no ReDoS risk"
  - "pytest asyncio_mode=auto configured globally in pyproject.toml"

patterns-established:
  - "Three-tier cache: TTLCache(100, 600) -> Redis(configurable TTL) -> PostgreSQL(caller handles)"
  - "NullRedis pattern: stub that satisfies the Redis interface with noop operations"
  - "Fail-open rate limiting: Redis errors allow requests through with warning log"
  - "Error response sanitization: strip paths, API keys, model names before client exposure"

# Metrics
duration: 5min
completed: 2026-03-01
---

# Phase 10 Plan 02: API Hardening Summary

**Three-tier forecast cache (cachetools TTLCache -> Redis -> PG), per-key Redis rate limiter with fail-open degradation, and prompt injection blocklist with geopolitical keyword validation**

## Performance

- **Duration:** 5 min
- **Started:** 2026-03-01T13:56:38Z
- **Completed:** 2026-03-01T14:01:38Z
- **Tasks:** 3
- **Files modified:** 6

## Accomplishments
- ForecastCache with memory (10min, 100 entries) -> Redis (configurable TTL) -> PostgreSQL fallthrough, with graceful degradation on Redis failure
- Per-API-key daily rate limiting via Redis INCR+EXPIRE with fail-open policy; Gemini budget tracking for daily pipeline
- Input sanitization: 14-phrase injection blocklist, 68-keyword geopolitical relevance check, 10-500 char length enforcement, error response sanitization stripping file paths/API keys/model names
- 16 tests covering all three components with mocked Redis (no external dependencies)

## Task Commits

Each task was committed atomically:

1. **Task 1: Three-tier cache service and Redis dependency** - `7e7abfe` (feat)
2. **Task 2: Rate limiting and input sanitization middleware** - `b8fa329` (feat)
3. **Task 3: Tests for API hardening components** - `8cd71be` (test)

## Files Created/Modified
- `src/api/services/cache_service.py` - ForecastCache class with three-tier hierarchy, TTL constants, cache key generators
- `src/api/middleware/rate_limit.py` - check_rate_limit, get_rate_limiter factory, gemini_budget_remaining, increment_gemini_usage
- `src/api/middleware/sanitize.py` - validate_forecast_question (blocklist + keywords + length), sanitize_error_response
- `src/api/deps.py` - Added get_redis(), get_cache(), _close_redis(), NullRedis stub
- `tests/test_api_hardening.py` - 16 tests: cache (6), rate limit (4), sanitization (6)
- `pyproject.toml` - Added pytest-asyncio dependency, asyncio_mode=auto

## Decisions Made
- Cache key generators produce fully-qualified keys (e.g., `forecast:{id}`); get/set use keys as-is with no internal prefix. Double-prefix bug explicitly guarded by test.
- NullRedis stub implements the Redis interface with noop returns. This avoids `Optional` branching throughout the cache/rate-limit hot path while providing clean degradation.
- Rate limiter uses fail-open policy: if Redis is unreachable, requests are allowed with a warning log. A broken rate limiter must never kill the API.
- Input sanitization uses substring matching (not regex) for injection detection to eliminate ReDoS risk.
- Enabled `asyncio_mode = "auto"` in pytest config and installed pytest-asyncio, which was declared as a dev dependency but not actually installed.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] pytest-asyncio not installed, asyncio_mode not configured**
- **Found during:** Task 3 (test execution)
- **Issue:** pytest-asyncio was listed in pyproject.toml dev dependencies but not installed. Additionally, `asyncio_mode` was not set in pytest config, causing all async tests to fail with "async def functions are not natively supported."
- **Fix:** Ran `uv add --dev pytest-asyncio` and added `asyncio_mode = "auto"` to `[tool.pytest.ini_options]` in pyproject.toml.
- **Files modified:** pyproject.toml
- **Verification:** All 16 async tests pass.
- **Committed in:** 8cd71be (Task 3 commit)

---

**Total deviations:** 1 auto-fixed (1 blocking)
**Impact on plan:** Required to run async tests. No scope creep.

## Issues Encountered
None beyond the pytest-asyncio blocking issue documented above.

## User Setup Required
None - no external service configuration required. Redis is optional (system degrades gracefully without it).

## Next Phase Readiness
- Cache, rate limiter, and sanitizer are ready as FastAPI Depends() for Plan 04's real forecast endpoints
- Gemini budget tracking functions (gemini_budget_remaining, increment_gemini_usage) are ready for Plan 03/04's daily pipeline
- NullRedis pattern established for any future Redis-dependent code that must degrade gracefully

---
*Phase: 10-ingest-forecast-pipeline*
*Completed: 2026-03-01*
