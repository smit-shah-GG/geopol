---
phase: 10-ingest-forecast-pipeline
plan: 04
subsystem: pipeline
tags: [gemini, ensemble-predictor, fastapi, redis, caching, rate-limiting, systemd, gdelt]

# Dependency graph
requires:
  - phase: 10-01
    provides: GDELT poller, PendingQuestion ORM model, IngestRun.daemon_type
  - phase: 10-02
    provides: ForecastCache, check_rate_limit, gemini_budget_remaining, validate_forecast_question, NullRedis
  - phase: 10-03
    provides: RSS feed daemon, ArticleIndexer with ChromaDB rss_articles collection
  - phase: 09-06
    provides: ForecastService (persist_forecast, get_forecast_by_id, get_forecasts_by_country)
provides:
  - Daily autonomous forecast pipeline (question gen -> predict -> persist -> resolve)
  - Budget-aware Gemini usage with PendingQuestion carryover queue
  - Outcome resolver comparing expired predictions to GDELT ground truth
  - Real API endpoints with 3-tier cache, rate limiting, input sanitization
  - ForecastService.get_top_forecasts() for top-N query
  - systemd timer for daily 06:00 UTC execution
affects: [11-frontend, 12-llm-enhancements, 13-calibration]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "4-phase daily pipeline cycle: dequeue -> generate -> predict+persist -> resolve"
    - "Budget exhaustion queues overflow to PendingQuestion for next-day priority"
    - "LLM-based outcome resolution with heuristic (actor+event_code) fallback"
    - "3-tier cache on all GET endpoints: memory -> Redis -> PostgreSQL -> fixture fallback"
    - "Rate limiter + sanitizer as FastAPI Depends on POST /forecasts"

key-files:
  created:
    - src/pipeline/__init__.py
    - src/pipeline/question_generator.py
    - src/pipeline/budget_tracker.py
    - src/pipeline/outcome_resolver.py
    - src/pipeline/daily_forecast.py
    - scripts/daily_forecast.py
    - deploy/systemd/geopol-daily-forecast.service
    - deploy/systemd/geopol-daily-forecast.timer
    - tests/test_daily_pipeline.py
  modified:
    - src/api/routes/v1/forecasts.py
    - src/api/app.py
    - src/api/services/forecast_service.py

key-decisions:
  - "LLM-based outcome resolution primary, heuristic (actor+event_code matching) as fallback"
  - "Carryover questions get priority=1 and are processed before fresh generation"
  - "Consecutive failure alerting at >= 2 failures emits CRITICAL log"
  - "POST /forecasts creates EnsemblePredictor per-request (stateless, no shared predictor state)"
  - "Fixture cache preserved as development fallback on all GET endpoints"
  - "Redis lifecycle added to app.py lifespan (init on startup, close on shutdown)"

patterns-established:
  - "Pipeline orchestrator pattern: DailyPipeline with run_daily() + run_with_retry()"
  - "Budget-gated generation: check budget -> generate -> queue overflow"
  - "Mock factory for async_sessionmaker in tests: custom class instead of AsyncMock"

# Metrics
duration: 10min
completed: 2026-03-01
---

# Phase 10 Plan 04: Daily Forecast Pipeline Summary

**Autonomous daily forecast pipeline with Gemini question generation, EnsemblePredictor integration, budget-aware carryover queue, GDELT outcome resolution, and real API endpoints with 3-tier cache + rate limiting + sanitization**

## Performance

- **Duration:** 10 min
- **Started:** 2026-03-01T14:07:14Z
- **Completed:** 2026-03-01T14:17:14Z
- **Tasks:** 4
- **Files modified:** 12

## Accomplishments
- Daily pipeline generates yes/no forecast questions from high-significance GDELT events via Gemini, runs EnsemblePredictor.predict() per question, persists via ForecastService
- Budget exhaustion mid-pipeline queues remaining questions to PendingQuestion table; next day prioritizes carryover before fresh generation
- Outcome resolver compares expired predictions against GDELT events with LLM-based resolution (Gemini) and heuristic fallback
- All API endpoints (GET/POST) now serve real PostgreSQL data with 3-tier cache, rate limiting, input sanitization, and fixture fallback
- 12 tests covering pipeline components and route wiring -- all pass

## Task Commits

Each task was committed atomically:

1. **Task 1: Question generator, budget tracker, and outcome resolver** - `af836ca` (feat)
2. **Task 2: Daily pipeline orchestrator, entry point, and systemd timer** - `f5ef326` (feat)
3. **Task 3: Wire real API endpoints with cache, rate limit, and sanitization** - `d72c62e` (feat)
4. **Task 4: Tests for daily pipeline and route wiring** - `3fe52b7` (test)
5. **Cleanup: Restore pipeline __init__.py exports** - `41de139` (chore)

## Files Created/Modified
- `src/pipeline/__init__.py` - Package init with all pipeline exports
- `src/pipeline/question_generator.py` (248 lines) - LLM-based forecast question generation from high-significance GDELT events
- `src/pipeline/budget_tracker.py` (184 lines) - Gemini budget tracking with Redis primary / PostgreSQL fallback, PendingQuestion queue management
- `src/pipeline/outcome_resolver.py` (340 lines) - Outcome resolution comparing expired predictions to GDELT ground truth (LLM + heuristic)
- `src/pipeline/daily_forecast.py` (290 lines) - 4-phase daily pipeline orchestrator with retry and consecutive failure alerting
- `scripts/daily_forecast.py` (209 lines) - Entry point for systemd timer with --max-questions, --skip-outcomes, --dry-run
- `deploy/systemd/geopol-daily-forecast.service` - oneshot service unit
- `deploy/systemd/geopol-daily-forecast.timer` - Daily 06:00 UTC timer (Persistent=true, RandomizedDelaySec=300)
- `src/api/routes/v1/forecasts.py` (368 lines) - Real forecast endpoints with cache + rate limit + sanitize + budget check
- `src/api/app.py` - Redis lifecycle added to lifespan (startup init, shutdown close)
- `src/api/services/forecast_service.py` - Added get_top_forecasts() for top-N query
- `tests/test_daily_pipeline.py` (490 lines) - 12 tests for pipeline and route wiring

## Decisions Made
- **LLM-based outcome resolution primary**: Gemini assesses whether predicted outcomes occurred (0.0-1.0 float). Heuristic (actor+event_code matching against GDELT events) as fallback when Gemini unavailable.
- **Carryover priority**: PendingQuestion rows from budget exhaustion get priority=1 and are dequeued before fresh generation begins, ensuring no question is silently dropped.
- **Consecutive failure alerting**: At >= 2 consecutive pipeline failures, a CRITICAL log is emitted. This relies on log-based alerting infrastructure (journald, CloudWatch, etc.) rather than adding a notification dependency.
- **POST /forecasts creates EnsemblePredictor per-request**: Avoids shared mutable state across concurrent requests. The predictor is lightweight (no model loading -- delegates to GeminiClient and TKGPredictor).
- **Redis lifecycle in app.py lifespan**: get_redis() called on startup to eagerly initialize (or degrade to NullRedis). _close_redis() called on shutdown for clean connection disposal.
- **Mock factory for async_sessionmaker in tests**: Custom `_MockSessionCtx` class because AsyncMock wraps `__call__` in a coroutine, breaking the `async with factory() as session:` pattern that async_sessionmaker uses.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Fixed __init__.py circular import during incremental development**
- **Found during:** Task 1 (before daily_forecast.py existed)
- **Issue:** `__init__.py` imported `DailyPipeline` from `daily_forecast.py` which didn't exist yet
- **Fix:** Deferred __init__.py exports until all modules existed, then restored full exports after Task 2
- **Files modified:** src/pipeline/__init__.py
- **Committed in:** af836ca (Task 1), 41de139 (cleanup)

---

**Total deviations:** 1 auto-fixed (1 blocking)
**Impact on plan:** Necessary for incremental development. No scope creep.

## Issues Encountered
- AsyncMock cannot correctly emulate `async_sessionmaker` behavior: `async_sessionmaker()` returns an async context manager directly (not a coroutine), but `AsyncMock.__call__` wraps the return in a coroutine. Resolved by creating a custom `_MockSessionCtx` test helper class.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Phase 10 complete: all 4 plans delivered
- Daily pipeline is operational (requires GEMINI_API_KEY env var and running PostgreSQL + Redis)
- systemd timer ready for deployment
- Phase 11 (Frontend), Phase 12 (LLM Enhancements), Phase 13 (Calibration) can proceed in parallel
- Phase 13 needs accumulated outcome data from this pipeline before calibration weights can be computed

---
*Phase: 10-ingest-forecast-pipeline*
*Completed: 2026-03-01*
