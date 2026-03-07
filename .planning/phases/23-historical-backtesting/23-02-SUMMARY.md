---
phase: 23
plan: 02
subsystem: backtesting-api
tags: [backtesting, admin-api, scheduler, heavy-job, checkpoint-discovery]
dependency-graph:
  requires: [phase-23-plan-01-engine]
  provides: [backtesting-api-endpoints, heavy-job-backtest, checkpoint-discovery]
  affects: [phase-23-plan-03-admin-panel]
tech-stack:
  added: []
  patterns: [fire-and-forget-create-task, heavy-job-mutual-exclusion, checkpoint-scan]
key-files:
  created: []
  modified:
    - src/api/schemas/admin.py
    - src/api/services/admin_service.py
    - src/api/routes/v1/admin.py
    - src/scheduler/heavy_runner.py
    - src/scheduler/job_wrappers.py
decisions:
  - Backtest dispatch via asyncio.create_task(heavy_backtest(config_json)) -- fire-and-forget, endpoint returns immediately
  - run_backtest() is in-process (not subprocess.run) -- src.backtesting.* is a proper package
  - heavy_backtest() NOT registered as APScheduler job -- on-demand only via admin API
  - Checkpoint discovery scans data/models/ for .npz/.pt/.json files with metadata extraction
  - Import-by-value bug fix in run_polymarket_cycle -- was importing bare names, now uses module reference
metrics:
  duration: 8min
  completed: 2026-03-08
---

# Phase 23 Plan 02: API + Scheduler Integration Summary

Admin API endpoints for backtesting CRUD, ProcessPoolExecutor dispatch via heavy job infrastructure, and checkpoint discovery for model comparison.

## What Was Done

### Task 1: Admin API Endpoints + Pydantic Schemas + Service Methods (b5de93d)

**src/api/schemas/admin.py** -- Added 6 Pydantic DTOs:
- `CheckpointInfo`: model name, type (tirgn/regcn), path, metrics, created_at
- `StartBacktestRequest`: label, description, window params with validation (ge/le constraints), checkpoints dict
- `BacktestRunDTO`: Full ORM model serialization with from_attributes=True
- `BacktestResultDTO`: Per-window result serialization
- `BacktestRunDetailDTO`: Run + results drill-down response
- `BacktestExportDTO`: Export format and run ID selection

**src/api/services/admin_service.py** -- Added 6 service methods:
- `get_backtest_runs()`: All runs ordered by created_at desc
- `start_backtest_run()`: Creates pending DB row, dispatches via asyncio.create_task(heavy_backtest(config_json))
- `get_backtest_run_detail()`: Run + all window results, 404 if not found
- `cancel_backtest_run()`: Sets status='cancelling', 409 if not running/pending
- `export_backtest_run()`: Delegates to export_run_csv/json, CSV returns with Content-Disposition header
- `get_checkpoints()`: Scans data/models/ for checkpoint files with metadata extraction

**src/api/routes/v1/admin.py** -- Added 6 endpoints under /backtesting/:
- `GET /backtesting/runs` -- list all runs
- `POST /backtesting/runs` (201) -- start new run
- `GET /backtesting/runs/{run_id}` -- drill-down detail
- `POST /backtesting/runs/{run_id}/cancel` -- cancel running/pending
- `GET /backtesting/runs/{run_id}/export?format=csv|json` -- download results
- `GET /backtesting/checkpoints` -- list available model checkpoints

### Task 2: Heavy Job Wiring (42b3591)

**src/scheduler/heavy_runner.py** -- Added `run_backtest(config_json: str) -> int`:
- In-process pattern (like run_polymarket_cycle, not subprocess.run)
- Creates own asyncio event loop, initializes DB in subprocess
- Deserializes BacktestRunConfig from JSON, creates BacktestRunner, runs evaluation
- Also fixed import-by-value bug in run_polymarket_cycle (bare `async_session_factory` -> `_pg.async_session_factory`)

**src/scheduler/job_wrappers.py** -- Added `heavy_backtest(config_json: str) -> None`:
- Acquires _heavy_job_lock (mutual exclusion with daily pipeline, polymarket, TKG retrain)
- Dispatches to ProcessPoolExecutor via run_in_executor
- NOT registered as APScheduler interval job (on-demand only)

## Full Dispatch Chain

```
POST /backtesting/runs (admin endpoint)
  -> AdminService.start_backtest_run()
    -> Creates BacktestRun row (status='pending')
    -> asyncio.create_task(heavy_backtest(config_json))
      -> Acquires _heavy_job_lock
      -> loop.run_in_executor(_get_process_executor(), run_backtest, config_json)
        -> run_backtest() in subprocess worker
          -> _pg.init_db() (own DB engine)
          -> BacktestRunConfig.from_json(config_json)
          -> BacktestRunner(config, session_factory).run()
```

## Verification Results

- All 6 backtesting routes registered in admin router
- BacktestRunDTO, StartBacktestRequest, CheckpointInfo import cleanly
- run_backtest() has correct signature (config_json parameter)
- heavy_backtest() imports cleanly from job_wrappers
- AdminService imports cleanly

## Next Phase Readiness

Plan 23-03 (BacktestingPanel frontend) can proceed. The API layer is complete:
1. TypeScript types need to mirror BacktestRunDTO, BacktestResultDTO, CheckpointInfo
2. AdminClient needs 6 new methods matching the endpoints
3. BacktestingPanel consumes these endpoints for run list, drill-down, and export
