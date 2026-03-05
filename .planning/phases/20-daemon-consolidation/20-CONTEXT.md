# Phase 20: Daemon Consolidation - Context

**Gathered:** 2026-03-05
**Status:** Ready for planning

<domain>
## Phase Boundary

All background jobs run under a single APScheduler 3.11.2 instance inside the FastAPI process, replacing scattered systemd timers and standalone daemon processes. Admin dashboard's pause/resume/trigger controls become functional. Heavy jobs retain OS-level memory isolation via ProcessPoolExecutor. uvicorn --workers 1 is a hard constraint.

</domain>

<decisions>
## Implementation Decisions

### Job failure & recovery
- Retry immediately up to 3 times with short backoff before marking as failed; next scheduled run still fires normally
- Stuck job detection: alert only (log warning + show "overrunning" in admin), do NOT forcibly kill — operator decides
- Auto-pause after N consecutive failures — job requires manual resume from admin dashboard; prevents log spam from persistently broken jobs
- Dependency-aware cascading: track job dependencies; if upstream job (e.g., GDELT poller) failed today, downstream jobs (e.g., daily pipeline) skip execution with a warning rather than running on stale data

### Migration & cutover
- Big-bang cutover — all systemd timers retired at once, APScheduler takes over all jobs simultaneously. No dual-scheduling period
- Catch up on missed jobs at startup — use APScheduler's misfire_grace_time so jobs missed during downtime run immediately on server restart
- Systemd unit files moved to `.planning/archive/` for reference, not deleted entirely
- Existing standalone daemon scripts preserved as importable functions — code remains callable via CLI (`uv run python scripts/xxx.py`) AND from APScheduler. No refactor to job-only callables

### Job concurrency rules
- Self-overlap: never allowed — max_instances=1 for all jobs. If a job is still running when the next trigger fires, skip the new execution
- Heavy jobs (daily forecast pipeline, TKG retraining, Polymarket auto-forecaster) are mutually exclusive with each other but CAN run alongside light jobs (GDELT poller, RSS poller, advisory poller, baseline risk)
- When a heavy job is triggered but another heavy job is running: queue and wait (FIFO). No missed executions for heavy jobs
- ProcessPoolExecutor with 1 worker for heavy jobs — enforces mutual exclusion at OS level, serializes the heavy job queue

### Admin control granularity
- Schedule editing from dashboard — admin can change cron/interval for any job, persists to system_config table, takes effect on next trigger cycle
- Kill button for running jobs — admin can force-stop an in-flight job. For ProcessPoolExecutor jobs, terminates the worker process
- Job display (extended): status, last run time, next scheduled run, success/fail counts, last duration, last error message, consecutive failure count, queue position for heavy jobs
- No job history UI — admin shows current/live state only. Past run data lives in logs. No new job_runs table

### Claude's Discretion
- Misfire grace time values per job type
- Exact consecutive failure threshold for auto-pause (likely 5-10)
- Job dependency graph definition (which jobs depend on which)
- Heavy job queue implementation (asyncio.Queue vs APScheduler listener pattern)
- Backoff timing for retries (exponential vs fixed)

</decisions>

<specifics>
## Specific Ideas

- Standalone daemon entry points must remain invocable from CLI for manual debugging outside the server process
- Heavy job classification: daily pipeline + TKG retrain + Polymarket auto-forecaster (all touch GPU or Gemini API heavily)
- Light job classification: GDELT poller, RSS poller, advisory poller, and any future baseline risk job

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 20-daemon-consolidation*
*Context gathered: 2026-03-05*
