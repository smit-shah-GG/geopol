---
phase: 10-ingest-forecast-pipeline
plan: 01
subsystem: ingest
tags: [gdelt, asyncio, aiohttp, systemd, backoff, sigterm, polling, sqlite, postgresql]

# Dependency graph
requires:
  - phase: 09-api-foundation
    provides: PostgreSQL ORM models (IngestRun, Base), async session factory, Settings
provides:
  - GDELT micro-batch polling daemon with URL-dedup, backoff, SIGTERM handling
  - PendingQuestion ORM model for daily pipeline budget carryover
  - IngestRun.daemon_type column distinguishing gdelt vs rss runs
  - Alembic migration 002 (pending_questions + daemon_type)
  - Backfill module for gap recovery on startup
  - systemd unit file for production deployment
affects: [10-02, 10-03, 10-04, 11-tkg-predictor]

# Tech tracking
tech-stack:
  added: [aiohttp, cachetools, feedparser, trafilatura]
  patterns: [async-daemon-with-sigterm, exponential-backoff, url-dedup-fast-path, asyncio-to-thread-bridge]

key-files:
  created:
    - src/ingest/__init__.py
    - src/ingest/gdelt_poller.py
    - src/ingest/backfill.py
    - scripts/gdelt_poller.py
    - deploy/systemd/geopol-gdelt-poller.service
    - alembic/versions/20260301_002_pending_questions_and_daemon_type.py
    - tests/test_gdelt_poller.py
  modified:
    - pyproject.toml
    - src/settings.py
    - src/db/models.py

key-decisions:
  - "URL-dedup fast path: track last-seen URL from lastupdate.txt, skip download if unchanged, record events_fetched=0 as valid success"
  - "Exponential backoff 1min -> 30min with 10% jitter on feed failures"
  - "asyncio.to_thread() bridge for synchronous EventStorage and TemporalKnowledgeGraph calls"
  - "Only quad_class 1 (diplomatic) and 4 (material conflict) events added to graph incrementally"
  - "IngestRun recording tolerates PostgreSQL downtime -- logs warning, does not crash daemon"

patterns-established:
  - "Async daemon pattern: asyncio.Event for shutdown + loop.add_signal_handler for SIGTERM/SIGINT"
  - "BackoffStrategy class: reusable exponential backoff with jitter_fraction param and reset()"
  - "GDELT CSV parsing: tab-separated 61-column format via pandas, _gdelt_row_to_event converter"

# Metrics
duration: 6min
completed: 2026-03-01
---

# Phase 10 Plan 01: GDELT Poller Summary

**Async GDELT micro-batch polling daemon with exponential backoff, URL-dedup fast path, incremental graph updates via asyncio.to_thread(), and PendingQuestion ORM model**

## Performance

- **Duration:** 6 min
- **Started:** 2026-03-01T13:55:45Z
- **Completed:** 2026-03-01T14:01:37Z
- **Tasks:** 3
- **Files modified:** 13

## Accomplishments
- Production-grade GDELT polling daemon with graceful SIGTERM shutdown and exponential backoff
- PendingQuestion ORM model and daemon_type column for daily pipeline integration
- Alembic migration 002 for schema evolution
- 17 unit tests covering all critical poller behaviors (parse, dedup, backoff, record, shutdown, graph)
- systemd unit file with security hardening (NoNewPrivileges, ProtectSystem, MemoryMax)

## Task Commits

Each task was committed atomically:

1. **Task 1: Dependencies, ORM schema, Alembic migration** - `1c46044` (feat)
2. **Task 2: GDELT poller daemon + backfill + systemd** - `30e8372` (feat)
3. **Task 3: Unit tests for GDELT poller** - `53a67ee` (test)

## Files Created/Modified
- `src/ingest/gdelt_poller.py` - Async GDELT micro-batch polling daemon (464 lines)
- `src/ingest/backfill.py` - Gap recovery from last successful IngestRun (80 lines)
- `src/ingest/__init__.py` - Ingest package init
- `scripts/gdelt_poller.py` - CLI entry point for systemd (79 lines)
- `deploy/systemd/geopol-gdelt-poller.service` - Production systemd unit
- `src/db/models.py` - Added PendingQuestion model, daemon_type on IngestRun
- `src/settings.py` - Added GDELT/RSS/pipeline settings
- `pyproject.toml` - Added aiohttp, cachetools, feedparser, trafilatura
- `alembic/versions/20260301_002_pending_questions_and_daemon_type.py` - Migration 002
- `tests/test_gdelt_poller.py` - 17 unit tests (417 lines)

## Decisions Made
- URL-dedup fast path: events_fetched=0 is a valid successful run, not a failure
- BackoffStrategy uses 10% jitter to avoid thundering-herd synchronisation across instances
- asyncio.to_thread() wraps all synchronous v1.0 code (EventStorage, TemporalKnowledgeGraph)
- IngestRun recording swallows PostgreSQL errors to keep the daemon running even if metrics DB is down
- Graph incremental updates filter to quad_class 1/4 only, matching the graph builder's existing filter

## Deviations from Plan

None -- plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None -- no external service configuration required.

## Next Phase Readiness
- GDELT poller is ready to run (`scripts/gdelt_poller.py` or via systemd)
- PendingQuestion model ready for Plan 04 (daily forecast pipeline)
- IngestRun.daemon_type ready for Plan 02 (RSS daemon) to distinguish run types
- Docker/PostgreSQL needed to run Alembic migration 002

---
*Phase: 10-ingest-forecast-pipeline*
*Completed: 2026-03-01*
