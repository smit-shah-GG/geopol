---
phase: 20-daemon-consolidation
plan: 01
subsystem: scheduler
tags: [apscheduler, daemon, background-jobs, process-pool, failure-tracking]
dependency-graph:
  requires: [phase-10, phase-17, phase-18]
  provides: [scheduler-package, job-wrappers, failure-tracking, job-registry]
  affects: [20-02, 20-03]
tech-stack:
  added: [APScheduler-3.11.2]
  patterns: [AsyncIOScheduler, ProcessPoolExecutor-isolation, asyncio-Lock-mutual-exclusion, singleton-dependency-injection]
key-files:
  created:
    - src/scheduler/__init__.py
    - src/scheduler/core.py
    - src/scheduler/dependencies.py
    - src/scheduler/job_wrappers.py
    - src/scheduler/heavy_runner.py
    - src/scheduler/retry.py
    - src/scheduler/registry.py
  modified:
    - pyproject.toml
    - uv.lock
decisions:
  - id: heavy-lock
    description: "asyncio.Lock for heavy job mutual exclusion (FIFO queue)"
    rationale: "Prevents concurrent subprocess/process-pool work from overwhelming max_workers=1"
  - id: subprocess-for-scripts
    description: "subprocess.run() for daily_pipeline and tkg_retrain"
    rationale: "scripts/ is not a Python package; retrain_tkg uses argparse.parse_args()"
  - id: inprocess-polymarket
    description: "In-process asyncio.run() for polymarket cycle in ProcessPoolExecutor"
    rationale: "src.polymarket.* is a proper package; needs fine-grained async control"
  - id: singleton-gdelt-poller
    description: "GDELTPoller singleton in SharedDeps"
    rationale: "_last_url state must persist between poll cycles for URL-dedup fast path"
metrics:
  duration: 4min
  completed: 2026-03-05
---

# Phase 20 Plan 01: Scheduler Foundation Summary

APScheduler 3.11.2 infrastructure with dual executors, 9 job wrappers, failure tracking, and job registry.

## What Was Built

### src/scheduler/core.py
AsyncIOScheduler factory with dual executors:
- `default`: AsyncIOExecutor for light async wrappers (6 jobs)
- `processpool`: ProcessPoolExecutor(max_workers=1) for heavy jobs (3 jobs)
- Job defaults: coalesce=True, max_instances=1, misfire_grace_time=900s
- Graceful shutdown: pause -> wait_for(shutdown, timeout) -> force

### src/scheduler/dependencies.py
SharedDeps dataclass holding singleton instances:
- Settings, EventStorage, TemporalKnowledgeGraph, GDELTPoller
- GDELT poller is singleton so `_last_url` persists across poll cycles

### src/scheduler/job_wrappers.py
9 async wrappers:
- Light (6): gdelt_poll_cycle, rss_poll_tier1, rss_poll_tier2, rss_prune, acled_poll_cycle, advisory_poll_cycle
- Heavy (3): heavy_daily_pipeline, heavy_polymarket_cycle, heavy_tkg_retrain
- Heavy jobs acquire `_heavy_job_lock` (asyncio.Lock) then dispatch to ProcessPoolExecutor
- All wrappers catch/log/re-raise exceptions for failure tracker

### src/scheduler/heavy_runner.py
Module-level functions (pickleable by ProcessPoolExecutor):
- `run_daily_pipeline()`: subprocess.run(["uv", "run", "python", "scripts/daily_forecast.py"])
- `run_tkg_retrain()`: subprocess.run(["uv", "run", "python", "scripts/retrain_tkg.py", "--force"])
- `run_polymarket_cycle()`: in-process asyncio.run() with src.polymarket.* imports

### src/scheduler/retry.py
- JobFailureTracker: consecutive failure tracking per job, auto-pause at 5 failures
- get_job_stats() / get_all_stats() for admin API consumption
- reset_failures() for manual resume
- JOB_DEPENDENCIES: daily_pipeline depends on gdelt_poller + rss_tier1
- check_upstream_health() for dependency cascade

### src/scheduler/registry.py
register_all_jobs() with correct triggers:
- GDELT: IntervalTrigger(gdelt_poll_interval)
- RSS tier1/tier2: IntervalTrigger(rss_poll_interval_tier1/tier2)
- RSS prune: IntervalTrigger(86400s)
- ACLED: IntervalTrigger(acled_poll_interval)
- Advisory: IntervalTrigger(advisory_poll_interval)
- Daily pipeline: CronTrigger(hour=6, minute=0)
- Polymarket: IntervalTrigger(polymarket_poll_interval), conditional on polymarket_enabled
- TKG retrain: CronTrigger(day_of_week="sun", hour=2)

## Decisions Made

1. **asyncio.Lock mutual exclusion** -- heavy jobs queue in FIFO order on a single lock, preventing concurrent subprocess work from overwhelming the single-worker process pool.
2. **subprocess.run for scripts/** -- scripts/ is not a Python package (no __init__.py, not in pyproject.toml packages), and retrain_tkg.main() uses argparse.parse_args() which reads sys.argv.
3. **In-process polymarket** -- src.polymarket.* is a proper package needing fine-grained async control (aiohttp sessions, DB transactions). Uses asyncio.run() inside ProcessPoolExecutor worker.
4. **Singleton GDELTPoller** -- _last_url must persist between poll cycles to enable URL-dedup fast path.

## Deviations from Plan

None -- plan executed exactly as written.

## Next Phase Readiness

Plan 20-02 (FastAPI lifespan wiring) can proceed. All exports are in place:
- `create_scheduler()`, `get_scheduler()`, `shutdown_scheduler()`
- `init_shared_deps()`, `get_shared_deps()`
- `register_all_jobs(scheduler, failure_tracker)`
- `JobFailureTracker(scheduler, max_consecutive_failures=5)`
