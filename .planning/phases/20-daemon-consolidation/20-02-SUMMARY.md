---
phase: 20-daemon-consolidation
plan: 02
subsystem: scheduler-integration
tags: [apscheduler, fastapi, lifespan, admin-api, daemon-control]
depends_on: ["20-01"]
provides: ["scheduler-mounted", "admin-job-control", "polymarket-loop-deleted"]
affects: ["20-03", "22-polymarket-fixes"]
tech_stack:
  added: []
  patterns: ["lifespan-scheduler-mount", "app-state-injection", "daemon-to-job-id-mapping"]
key_files:
  created: []
  modified:
    - src/api/app.py
    - src/api/services/admin_service.py
    - src/api/schemas/admin.py
    - src/api/routes/v1/admin.py
decisions:
  - "Scheduler shutdown order: scheduler (30s) -> Redis -> DB (scheduler may use DB during in-flight jobs)"
  - "RSS daemon aggregates 3 sub-jobs -- paused only when ALL sub-jobs paused, next_run is earliest"
  - "_get_scheduler_or_none() pattern for graceful fallback when scheduler not initialized (tests)"
  - "trigger fires ALL sub-jobs for multi-job daemons (RSS tier1+tier2+prune)"
metrics:
  duration: "6 minutes"
  completed: "2026-03-05"
---

# Phase 20 Plan 02: APScheduler-FastAPI Integration Summary

**One-liner:** APScheduler mounted in FastAPI lifespan, Polymarket asyncio loop deleted, admin API delegates pause/resume/trigger to scheduler with live job state

## What Was Done

### Task 1: Mount scheduler in lifespan (4e46abc)
Rewrote `_lifespan` to create and start APScheduler after DB/Redis init. Deleted the entire `_polymarket_loop()` function (72 lines of asyncio.create_task + while-True loop). Shutdown order: scheduler (30s graceful timeout) -> Redis -> DB. Startup order preserved: logging -> ring buffer -> DB -> Redis -> dev key -> shared deps -> scheduler.

### Task 2: Extend schema + rewire admin service (3ae634f)
Added 4 fields to ProcessInfo: `last_duration`, `last_error`, `consecutive_failures`, `paused`. Rewrote `get_processes()` to merge APScheduler live state (next_run_time, paused) with DB aggregates (success/fail counts) and failure tracker stats. RSS daemon aggregates state from 3 sub-jobs. Deleted the 501 stub. Added `trigger_job()` via `scheduler.modify_job(next_run_time=now)`, `pause_job()`, `resume_job()`, `reschedule_job()`, and `get_jobs()`.

### Task 3: Admin routes + dependency injection (4dcb190)
Added `POST /processes/{daemon_type}/pause`, `POST /processes/{daemon_type}/resume`, and `GET /jobs` endpoints. Updated `_get_service` to inject `failure_tracker` from `request.app.state`. All job control operations delegate through AdminService to APScheduler.

## Key Implementation Details

- **Daemon-to-job mapping:** `_DAEMON_TO_JOB_IDS` maps each daemon_type to its APScheduler job IDs. RSS has 3 (tier1, tier2, prune), all others have 1. Pause/resume/trigger operate on all sub-jobs.
- **Graceful fallback:** `_get_scheduler_or_none()` catches RuntimeError from uninitialized scheduler, returning None. Admin service falls back to DB-only process info (existing behavior for tests).
- **Shutdown order:** Scheduler shuts down first with 30s timeout (in-flight jobs may need DB access), then Redis, then DB.

## Deviations from Plan

None -- plan executed exactly as written.

## Verification

- `_polymarket_loop` deleted from app.py (0 grep matches)
- `create_scheduler` mounted in lifespan (2 grep matches: import + call)
- `shutdown_scheduler` in shutdown path (2 grep matches: import + call)
- 501 stub deleted from admin_service.py (0 grep matches)
- ProcessInfo has all 4 new fields
- Admin router has `/processes/{daemon_type}/pause`, `/resume`, and `/jobs`

## Requirements Coverage

| Requirement | Status |
|---|---|
| DAEMON-01: APScheduler starts with FastAPI | Complete |
| DAEMON-02: All 9 jobs registered at startup | Complete |
| DAEMON-03: Polymarket asyncio loop deleted | Complete |
| DAEMON-04: Admin pause/resume via APScheduler | Complete |
| DAEMON-05: Admin trigger via APScheduler | Complete |
| DAEMON-06: Graceful shutdown with timeout | Complete |
