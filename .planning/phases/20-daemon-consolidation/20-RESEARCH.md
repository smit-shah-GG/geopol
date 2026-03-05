# Phase 20: Daemon Consolidation - Research

**Researched:** 2026-03-05
**Domain:** APScheduler 3.x in-process job scheduling with FastAPI, ProcessPoolExecutor isolation
**Confidence:** HIGH

## Summary

Phase 20 replaces 4+ scattered systemd services and standalone `asyncio.run()` loops with a single APScheduler 3.11.2 `AsyncIOScheduler` mounted in the FastAPI lifespan. The research examined: (1) the existing daemon codebase -- 5 pollers + 1 pipeline + 1 retrainer, each with their own event loops and SIGTERM handlers, (2) APScheduler 3.x API for AsyncIOScheduler, executor configuration, and job control, (3) ProcessPoolExecutor constraints for heavy job isolation, and (4) the admin API surface that currently returns 501 for all trigger operations.

Key findings: Every existing poller follows the same pattern -- a class with a `run()` method that installs signal handlers, runs `while not shutdown` with `asyncio.wait_for`, and records `IngestRun` rows. These need to be refactored into single-invocation callables (one poll cycle) that APScheduler triggers on schedule. The Polymarket loop in `app.py` (lines 126-198) is already an `asyncio.create_task` -- it gets absorbed. ProcessPoolExecutor jobs MUST use module-level pickleable functions (not bound methods), which constrains the architecture for heavy jobs.

**Primary recommendation:** Create a `src/scheduler/` module with a `JobRegistry` that wraps each poller's `_poll_once()` / `poll_feeds()` method in a top-level async function, registers all jobs with APScheduler at startup, and exposes the scheduler instance for admin API job control. Heavy jobs use a ProcessPoolExecutor with `max_workers=1` to serialize GPU/Gemini-heavy work.

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| APScheduler | 3.11.2 | Job scheduling | Mature, AsyncIOScheduler integrates with FastAPI event loop, pause/resume/trigger API |
| ProcessPoolExecutor | stdlib | Heavy job isolation | OS-level memory isolation, serializes heavy work with max_workers=1 |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| apscheduler.executors.pool.ProcessPoolExecutor | 3.11.2 | APScheduler executor wrapper | Register as secondary executor for heavy jobs |
| apscheduler.executors.asyncio.AsyncIOExecutor | 3.11.2 | Default async executor | Light jobs (pollers) run directly in event loop |
| apscheduler.jobstores.memory.MemoryJobStore | 3.11.2 | In-memory job store | Per CONTEXT.md decision: MemoryJobStore, not SQLAlchemy |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| MemoryJobStore | SQLAlchemyJobStore | Persistence across restarts, but CONTEXT.md specifies MemoryJobStore with `coalesce=True` for simplicity -- schedules are defined in code, not user-created |
| ProcessPoolExecutor(1) | ThreadPoolExecutor | No OS-level memory isolation -- heavy jobs can OOM the FastAPI process |
| APScheduler 4.x | APScheduler 3.x | v4.0.0a6 is alpha, incompatible API rewrite -- explicitly ruled out |

**Installation:**
```bash
uv add "APScheduler==3.11.2"
```

## Architecture Patterns

### Recommended Project Structure
```
src/
├── scheduler/
│   ├── __init__.py          # Module init, exports get_scheduler()
│   ├── core.py              # AsyncIOScheduler factory, startup/shutdown
│   ├── registry.py          # Job registration: all jobs defined here
│   ├── job_wrappers.py      # Top-level async functions wrapping poller classes
│   ├── heavy_runner.py      # ProcessPoolExecutor entry points (pickleable)
│   ├── retry.py             # Retry logic, failure tracking, auto-pause
│   └── dependencies.py      # Shared dependency factory (EventStorage, TKG, etc.)
├── api/
│   ├── app.py               # Lifespan mounts scheduler
│   ├── services/
│   │   └── admin_service.py # trigger/pause/resume now delegate to scheduler
│   └── schemas/
│       └── admin.py         # ProcessInfo extended with APScheduler fields
```

### Pattern 1: Scheduler Singleton in FastAPI Lifespan
**What:** Create the AsyncIOScheduler in lifespan startup, store on `app.state`, shut down in lifespan teardown.
**When to use:** Always -- the scheduler must share the FastAPI event loop.
**Example:**
```python
# src/scheduler/core.py
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.executors.asyncio import AsyncIOExecutor
from apscheduler.executors.pool import ProcessPoolExecutor as APProcessPoolExecutor

_scheduler: AsyncIOScheduler | None = None

def create_scheduler() -> AsyncIOScheduler:
    global _scheduler
    _scheduler = AsyncIOScheduler(
        executors={
            "default": AsyncIOExecutor(),
            "processpool": APProcessPoolExecutor(max_workers=1),
        },
        job_defaults={
            "coalesce": True,
            "max_instances": 1,
            "misfire_grace_time": 900,  # 15 min default
        },
    )
    return _scheduler

def get_scheduler() -> AsyncIOScheduler:
    if _scheduler is None:
        raise RuntimeError("Scheduler not initialized")
    return _scheduler
```

```python
# src/api/app.py (lifespan modification)
from src.scheduler.core import create_scheduler
from src.scheduler.registry import register_all_jobs

@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    # ... existing init (DB, Redis, logging) ...

    scheduler = create_scheduler()
    register_all_jobs(scheduler)
    scheduler.start()
    app.state.scheduler = scheduler

    yield

    # Shutdown: scheduler first, then DB/Redis
    scheduler.shutdown(wait=True)  # waits for in-flight jobs
    await _close_redis()
    await close_db()
```

### Pattern 2: Poller Wrapping -- Single-Cycle Callable
**What:** Each existing poller class has a `_poll_once()` method that does one cycle. The APScheduler job wraps construction + single-cycle invocation.
**When to use:** For all light jobs (GDELT, RSS, ACLED, Advisory).
**Example:**
```python
# src/scheduler/job_wrappers.py
async def gdelt_poll_cycle() -> None:
    """One GDELT poll cycle. APScheduler calls this on interval."""
    from src.scheduler.dependencies import get_shared_deps
    deps = get_shared_deps()

    poller = GDELTPoller(
        event_storage=deps.event_storage,
        graph=deps.graph,
    )
    await poller._poll_once()

async def rss_poll_tier1() -> None:
    """One RSS tier-1 poll cycle."""
    from src.scheduler.dependencies import get_shared_deps
    deps = get_shared_deps()

    daemon = RSSDaemon(config=DaemonConfig(
        tier1_interval=deps.settings.rss_poll_interval_tier1,
    ))
    metrics = await daemon.poll_feeds(TIER_1_FEEDS)
    await daemon._record_ingest_run(metrics, "tier-1")
```

### Pattern 3: ProcessPoolExecutor for Heavy Jobs
**What:** Heavy jobs (daily pipeline, TKG retrain, Polymarket auto-forecast) run in ProcessPoolExecutor to isolate memory.
**When to use:** Any job that loads ML models, calls Gemini API extensively, or uses >512MB RAM.
**Critical constraint:** ProcessPoolExecutor requires **pickleable, module-level functions**. No bound methods, no lambdas, no closures.
**Example:**
```python
# src/scheduler/heavy_runner.py
# MUST be top-level functions -- ProcessPoolExecutor pickles by name

def run_daily_pipeline() -> int:
    """Entry point for daily forecast pipeline in subprocess.

    Returns exit code (0=success, 1=failure).
    Runs synchronously -- ProcessPoolExecutor handles it.
    """
    import asyncio
    from scripts.daily_forecast import _run

    # Create a fresh event loop in the subprocess
    args = _make_pipeline_args()
    return asyncio.run(_run(args))

def run_tkg_retrain() -> int:
    """Entry point for TKG retraining in subprocess."""
    from scripts.retrain_tkg import main
    return main()
```

```python
# In registry.py
scheduler.add_job(
    run_daily_pipeline,
    trigger=CronTrigger(hour=6, minute=0),
    id="daily_pipeline",
    name="Daily Forecast Pipeline",
    executor="processpool",
    misfire_grace_time=3600,
    max_instances=1,
)
```

### Pattern 4: Job Failure Tracking via Listener
**What:** APScheduler event listeners track failures, implement retry, and auto-pause after N consecutive failures.
**When to use:** All jobs -- this is the central failure management system.
**Example:**
```python
# src/scheduler/retry.py
from apscheduler.events import EVENT_JOB_EXECUTED, EVENT_JOB_ERROR, EVENT_JOB_MISSED

class JobFailureTracker:
    def __init__(self, scheduler, max_consecutive_failures: int = 5):
        self._scheduler = scheduler
        self._max_failures = max_consecutive_failures
        self._consecutive_failures: dict[str, int] = {}
        self._retry_counts: dict[str, int] = {}  # per-execution retries

    def on_job_event(self, event):
        if event.code == EVENT_JOB_EXECUTED:
            self._consecutive_failures[event.job_id] = 0
            self._retry_counts.pop(event.job_id, None)
        elif event.code == EVENT_JOB_ERROR:
            count = self._consecutive_failures.get(event.job_id, 0) + 1
            self._consecutive_failures[event.job_id] = count

            logger.error(
                "Job %s failed (consecutive=%d): %s",
                event.job_id, count, event.exception,
            )

            # Auto-pause after threshold
            if count >= self._max_failures:
                self._scheduler.pause_job(event.job_id)
                logger.warning(
                    "Job %s auto-paused after %d consecutive failures",
                    event.job_id, count,
                )
```

### Pattern 5: Heavy Job Mutual Exclusion via asyncio.Lock
**What:** Heavy jobs are mutually exclusive (FIFO queue). Only one heavy job at a time.
**When to use:** Daily pipeline, TKG retrain, Polymarket auto-forecaster.
**Example:**
```python
# src/scheduler/job_wrappers.py
import asyncio

_heavy_job_lock = asyncio.Lock()

async def _run_heavy_job(func, job_name: str) -> None:
    """Acquire heavy job lock, then run in processpool."""
    async with _heavy_job_lock:
        logger.info("Heavy job %s acquired lock, executing...", job_name)
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None,  # Use scheduler's processpool
            func,
        )
        if result != 0:
            raise RuntimeError(f"{job_name} exited with code {result}")
```

**IMPORTANT NOTE on ProcessPoolExecutor + AsyncIOScheduler interaction:** When a job is registered with `executor="processpool"`, APScheduler runs the function directly in the ProcessPoolExecutor -- the function must be synchronous and pickleable. For the heavy job mutual exclusion pattern, there are two approaches:

**Approach A (Recommended):** Use the default AsyncIOExecutor for heavy jobs, but have the async wrapper acquire a lock then manually dispatch to `loop.run_in_executor(processpool_executor, sync_func)`. This gives you async-level mutual exclusion + process isolation.

**Approach B:** Use APScheduler's `processpool` executor directly with `max_workers=1` -- this serializes at the executor level, but you lose the ability to use `asyncio.Lock` for FIFO queuing since the functions are sync.

Given the CONTEXT.md requirement for "ProcessPoolExecutor with 1 worker" AND "queue and wait (FIFO)", **Approach A is correct**: register heavy jobs with the default async executor, wrap them with an `asyncio.Lock`, and manually dispatch to a shared `ProcessPoolExecutor(max_workers=1)`.

### Pattern 6: Admin Trigger via modify_job
**What:** Triggering immediate execution uses `scheduler.modify_job(job_id, next_run_time=datetime.now(tz))`.
**When to use:** POST /api/v1/admin/jobs/{id}/trigger endpoint.
**Example:**
```python
# In admin_service.py
from datetime import datetime, timezone

async def trigger_job(self, job_id: str) -> None:
    scheduler = get_scheduler()
    job = scheduler.get_job(job_id)
    if job is None:
        raise HTTPException(404, f"Unknown job: {job_id}")
    scheduler.modify_job(job_id, next_run_time=datetime.now(timezone.utc))
```

### Anti-Patterns to Avoid
- **Running poller's `run()` method as APScheduler job:** The `run()` methods contain their own `while not shutdown` loops, signal handlers, and backoff. APScheduler replaces ALL of that. Use `_poll_once()` only.
- **Using SQLAlchemyJobStore with AsyncSession:** APScheduler 3.x SQLAlchemyJobStore uses synchronous SQLAlchemy. If used, it needs a separate sync engine URL, not the async one.
- **Signal handlers in APScheduler jobs:** Jobs must NOT install SIGTERM/SIGINT handlers. APScheduler owns the signal lifecycle. The pollers' `_handle_shutdown` methods become dead code.
- **Creating new event loops in async jobs:** Never call `asyncio.run()` inside an async APScheduler job. Use `await` directly.
- **Storing mutable state on poller instances between cycles:** Each cycle should construct fresh poller state or share via the dependency container. The `_last_url` in GDELTPoller needs to persist between cycles (move to module-level or dependency container).

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Job scheduling | Custom `while True` + `asyncio.sleep` loops | APScheduler `IntervalTrigger` / `CronTrigger` | Handles misfires, coalescing, overlap prevention |
| Job overlap prevention | Manual `asyncio.Lock` per-job | APScheduler `max_instances=1` | Built-in, tested, handles edge cases |
| Missed job recovery | Custom "check last run time" logic | APScheduler `misfire_grace_time` + `coalesce=True` | Fires once on startup if missed during downtime |
| Pause/resume jobs | Custom flag + conditional in job body | `scheduler.pause_job()` / `resume_job()` | Sets `next_run_time=None` atomically |
| Job event tracking | Manual try/except + logging in every job | `scheduler.add_listener(EVENT_JOB_EXECUTED | EVENT_JOB_ERROR)` | Centralizes failure tracking |

**Key insight:** The existing pollers hand-roll 80% of what APScheduler provides: scheduling, overlap prevention, backoff, graceful shutdown, and missed-run recovery. Phase 20 deletes all of that loop infrastructure and replaces it with declarative job registration.

## Common Pitfalls

### Pitfall 1: ProcessPoolExecutor Pickle Failures
**What goes wrong:** Job functions registered with `executor="processpool"` must be pickleable (module-level, not bound methods, no closures). Attempting to use a lambda, inner function, or class method causes `PicklingError` at runtime.
**Why it happens:** ProcessPoolExecutor serializes the function reference via pickle to send to the subprocess. Pickle only works with globally importable names.
**How to avoid:** Define all heavy job entry points as top-level functions in `src/scheduler/heavy_runner.py`. Each function imports its dependencies inside the function body (the subprocess has a fresh Python environment).
**Warning signs:** `_pickle.PicklingError: Can't pickle <function ...>` in job error logs.

### Pitfall 2: GDELTPoller._last_url State Loss Between Cycles
**What goes wrong:** The GDELT poller uses `self._last_url` to deduplicate polls (skip if same URL as last time). When APScheduler calls `_poll_once()` on a fresh `GDELTPoller` instance each cycle, this state is lost, causing redundant downloads.
**Why it happens:** Each cycle creates a new poller instance for clean state.
**How to avoid:** Two options: (A) persist `_last_url` in the dependency container (singleton poller), or (B) persist it in the database (an `IngestRun` metadata field). Option A is simpler -- keep one `GDELTPoller` instance alive across cycles via the dependency container.
**Warning signs:** GDELT poller processing the same CSV every cycle, high duplicate counts in IngestRun.

### Pitfall 3: APScheduler Memory Leak on Job Exceptions (GitHub #235)
**What goes wrong:** APScheduler 3.x retains references to tracebacks when jobs raise exceptions, preventing garbage collection.
**Why it happens:** Known bug in APScheduler 3.x -- the `JobExecutionEvent` holds a reference to the traceback.
**How to avoid:** Wrap every job function in `try/except` that catches all exceptions, logs them, and explicitly clears the traceback via `event.traceback = None` in the listener, or by using `sys.exc_info()` and clearing it.
**Warning signs:** Gradually increasing memory usage, especially with frequently failing jobs.

### Pitfall 4: Blocking the Event Loop with Sync Code in Async Jobs
**What goes wrong:** Light job wrappers call synchronous code (e.g., `EventStorage.insert_events()`) directly without `asyncio.to_thread()`, blocking the FastAPI event loop.
**Why it happens:** The existing pollers already use `asyncio.to_thread()` for sync calls, but when extracting `_poll_once()`, it's tempting to skip this.
**How to avoid:** Preserve all existing `asyncio.to_thread()` calls within `_poll_once()`. The pollers already handle this correctly -- don't remove it during refactoring.
**Warning signs:** API response times spike during poller execution.

### Pitfall 5: RSSDaemon Tier-2 Scheduling Complexity
**What goes wrong:** The RSS daemon has two tiers with different intervals (15min and 60min), managed internally via `_last_tier1_poll` / `_last_tier2_poll` timestamps. Naively creating one APScheduler job loses this tiered behavior.
**Why it happens:** The tier logic is embedded in `_tick()`.
**How to avoid:** Register **two separate APScheduler jobs**: `rss_poll_tier1` (IntervalTrigger 15min) and `rss_poll_tier2` (IntervalTrigger 60min). Each calls `poll_feeds()` with the appropriate feed list. The pruning check becomes a third daily job.
**Warning signs:** Only tier-1 feeds being polled, or tier-2 feeds being polled at tier-1 frequency.

### Pitfall 6: Graceful Shutdown Timeout
**What goes wrong:** `scheduler.shutdown(wait=True)` blocks indefinitely if a heavy job (e.g., daily pipeline taking 30+ minutes) is running.
**Why it happens:** `shutdown(wait=True)` waits for ALL executors to complete ALL submitted jobs.
**How to avoid:** Implement a timeout wrapper around `scheduler.shutdown()`. If the scheduler doesn't stop within 30 seconds, terminate the ProcessPoolExecutor's worker process and proceed with shutdown. APScheduler 3.x does NOT have a `shutdown(wait_timeout=N)` parameter -- you must implement this manually.
**Warning signs:** Server hangs indefinitely on restart during a heavy job.

### Pitfall 7: Polymarket Loop Already in Lifespan
**What goes wrong:** The Polymarket matching loop is currently an `asyncio.create_task` in the lifespan (app.py lines 79-82). If not removed, it runs alongside the new APScheduler-scheduled Polymarket job, causing duplicate matching cycles and double API usage.
**Why it happens:** Incomplete migration -- forgetting to remove the old task.
**How to avoid:** Delete the `polymarket_task` code from `_lifespan()` when registering the Polymarket job with APScheduler.
**Warning signs:** Double IngestRun entries for `polymarket` daemon_type.

## Code Examples

### Job Registration (All Jobs)
```python
# src/scheduler/registry.py
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.triggers.cron import CronTrigger

def register_all_jobs(scheduler) -> None:
    """Register all background jobs with the scheduler."""
    settings = get_settings()

    # --- Light jobs (default AsyncIOExecutor) ---

    scheduler.add_job(
        gdelt_poll_cycle,
        trigger=IntervalTrigger(seconds=settings.gdelt_poll_interval),
        id="gdelt_poller",
        name="GDELT Poller",
        max_instances=1,
        coalesce=True,
        misfire_grace_time=settings.gdelt_poll_interval,  # 900s
    )

    scheduler.add_job(
        rss_poll_tier1,
        trigger=IntervalTrigger(seconds=settings.rss_poll_interval_tier1),
        id="rss_tier1",
        name="RSS Poller (Tier 1)",
        max_instances=1,
        coalesce=True,
        misfire_grace_time=settings.rss_poll_interval_tier1,
    )

    scheduler.add_job(
        rss_poll_tier2,
        trigger=IntervalTrigger(seconds=settings.rss_poll_interval_tier2),
        id="rss_tier2",
        name="RSS Poller (Tier 2)",
        max_instances=1,
        coalesce=True,
        misfire_grace_time=settings.rss_poll_interval_tier2,
    )

    scheduler.add_job(
        acled_poll_cycle,
        trigger=IntervalTrigger(seconds=settings.acled_poll_interval),
        id="acled_poller",
        name="ACLED Poller",
        max_instances=1,
        coalesce=True,
        misfire_grace_time=3600,
    )

    scheduler.add_job(
        advisory_poll_cycle,
        trigger=IntervalTrigger(seconds=settings.advisory_poll_interval),
        id="advisory_poller",
        name="Advisory Poller",
        max_instances=1,
        coalesce=True,
        misfire_grace_time=3600,
    )

    # --- Heavy jobs (async wrapper -> ProcessPoolExecutor) ---

    scheduler.add_job(
        heavy_daily_pipeline,  # async wrapper with lock
        trigger=CronTrigger(hour=6, minute=0),
        id="daily_pipeline",
        name="Daily Forecast Pipeline",
        max_instances=1,
        coalesce=True,
        misfire_grace_time=3600,
    )

    scheduler.add_job(
        heavy_polymarket_cycle,  # async wrapper with lock
        trigger=IntervalTrigger(seconds=settings.polymarket_poll_interval),
        id="polymarket",
        name="Polymarket Auto-Forecaster",
        max_instances=1,
        coalesce=True,
        misfire_grace_time=1800,
    )

    scheduler.add_job(
        heavy_tkg_retrain,  # async wrapper with lock
        trigger=CronTrigger(day_of_week="sun", hour=2),
        id="tkg_retrain",
        name="TKG Retrainer",
        max_instances=1,
        coalesce=True,
        misfire_grace_time=7200,
    )
```

### Admin API Integration
```python
# src/api/services/admin_service.py (modified)
from src.scheduler.core import get_scheduler

async def get_processes(self) -> list[ProcessInfo]:
    """Query APScheduler for live job state + IngestRun history."""
    scheduler = get_scheduler()
    jobs = scheduler.get_jobs()

    processes = []
    for job in jobs:
        # Get IngestRun stats from DB (existing query)
        # ...
        processes.append(ProcessInfo(
            name=job.name,
            daemon_type=job.id,
            status="paused" if job.next_run_time is None else "scheduled",
            last_run=last_run_from_db,
            next_run=job.next_run_time,
            success_count=ok_count,
            fail_count=fail_count,
        ))
    return processes

async def trigger_job(self, job_id: str) -> None:
    scheduler = get_scheduler()
    job = scheduler.get_job(job_id)
    if job is None:
        raise HTTPException(404, f"Unknown job: {job_id}")
    scheduler.modify_job(job_id, next_run_time=datetime.now(timezone.utc))

async def pause_job(self, job_id: str) -> None:
    scheduler = get_scheduler()
    scheduler.pause_job(job_id)

async def resume_job(self, job_id: str) -> None:
    scheduler = get_scheduler()
    scheduler.resume_job(job_id)
```

### Shared Dependency Container
```python
# src/scheduler/dependencies.py
from dataclasses import dataclass
from src.database.storage import EventStorage
from src.knowledge_graph.graph_builder import TemporalKnowledgeGraph
from src.settings import Settings, get_settings

@dataclass
class SharedDeps:
    """Lazily-initialized shared dependencies for job wrappers."""
    settings: Settings
    event_storage: EventStorage
    graph: TemporalKnowledgeGraph
    # Add more as needed

_deps: SharedDeps | None = None

def init_shared_deps() -> SharedDeps:
    """Initialize once at startup. Called from lifespan."""
    global _deps
    settings = get_settings()
    _deps = SharedDeps(
        settings=settings,
        event_storage=EventStorage(db_path=settings.gdelt_db_path),
        graph=TemporalKnowledgeGraph(),
    )
    return _deps

def get_shared_deps() -> SharedDeps:
    if _deps is None:
        raise RuntimeError("SharedDeps not initialized")
    return _deps
```

### Graceful Shutdown with Timeout
```python
# src/scheduler/core.py
import signal
from concurrent.futures import ProcessPoolExecutor

async def shutdown_scheduler(scheduler, timeout: float = 30.0) -> None:
    """Shut down scheduler with timeout for in-flight jobs."""
    # First, prevent new jobs from starting
    scheduler.pause()

    # Wait for in-flight jobs with timeout
    try:
        await asyncio.wait_for(
            asyncio.to_thread(scheduler.shutdown, wait=True),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        logger.warning(
            "Scheduler shutdown timed out after %.0fs, forcing...", timeout
        )
        # Force-shutdown executors
        scheduler.shutdown(wait=False)
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Standalone `asyncio.run()` daemons + systemd | In-process APScheduler | This phase | Eliminates 4+ separate processes |
| `asyncio.create_task()` for Polymarket | APScheduler IntervalTrigger | This phase | Gains pause/resume/trigger control |
| systemd timers for daily pipeline | APScheduler CronTrigger | This phase | Gains admin dashboard visibility |
| Custom backoff in each poller | APScheduler misfire_grace_time + listener retry | This phase | Centralizes failure handling |

## Job Inventory

Complete list of jobs to register, with their current implementation location and refactoring needs:

| Job ID | Source File | Current Entry Point | Trigger | Light/Heavy | Refactoring Needed |
|--------|------------|---------------------|---------|-------------|-------------------|
| `gdelt_poller` | `src/ingest/gdelt_poller.py` | `GDELTPoller._poll_once()` | Interval 15min | Light | Extract from `run()` loop; persist `_last_url` state |
| `rss_tier1` | `src/ingest/rss_daemon.py` | `RSSDaemon.poll_feeds(TIER_1_FEEDS)` | Interval 15min | Light | Split from `_tick()` tier logic |
| `rss_tier2` | `src/ingest/rss_daemon.py` | `RSSDaemon.poll_feeds(TIER_2_FEEDS)` | Interval 60min | Light | Split from `_tick()` tier logic |
| `rss_prune` | `src/ingest/rss_daemon.py` | `RSSDaemon._maybe_prune()` | Interval 24h | Light | Extract as standalone |
| `acled_poller` | `src/ingest/acled_poller.py` | `ACLEDPoller._poll_once()` | Interval (settings) | Light | Extract from `run()` loop |
| `advisory_poller` | `src/ingest/advisory_poller.py` | `AdvisoryPoller._poll_once()` | Interval (settings) | Light | Extract from `run()` loop |
| `daily_pipeline` | `scripts/daily_forecast.py` | `_run(args)` | Cron 06:00 | **Heavy** | Wrap in ProcessPoolExecutor runner |
| `polymarket` | `src/api/app.py` + `src/polymarket/auto_forecaster.py` | `_polymarket_loop()` | Interval (settings) | **Heavy** | Extract from `app.py` lifespan task |
| `tkg_retrain` | `scripts/retrain_tkg.py` | `main()` | Cron weekly Sun 02:00 | **Heavy** | Wrap in ProcessPoolExecutor runner |

## Dependency Construction Analysis

Critical constraint: each poller has different constructor requirements. This table maps what each job needs at construction time:

| Job | Dependencies Needed | Notes |
|-----|---------------------|-------|
| GDELT Poller | `EventStorage`, `TemporalKnowledgeGraph`, settings | Also needs `init_db()` for IngestRun persistence |
| RSS Daemon | `DaemonConfig` (from settings), `ArticleIndexer` | Self-contained; ArticleIndexer created internally |
| ACLED Poller | `EventStorage`, settings (for ACLED credentials) | Also needs `init_db()` |
| Advisory Poller | settings (for interval) | Lightest dependencies; no EventStorage needed |
| Daily Pipeline | Everything: EventStorage, GeminiClient, TKGPredictor, RAGPipeline, session_factory, Redis | Heaviest; currently wired in `scripts/daily_forecast.py:_run()` |
| Polymarket | `async_session_factory`, `GeminiClient`, `PolymarketClient`, settings | Currently self-wires in `_polymarket_loop()` |
| TKG Retrain | `RetrainingScheduler` (from config YAML) | Synchronous; wraps training loop |

## Open Questions

1. **Polymarket as heavy vs. light job**
   - CONTEXT.md explicitly marks it as heavy (uses EnsemblePredictor)
   - But the matching cycle itself is I/O-bound (HTTP calls to Polymarket API)
   - Only the auto-forecast sub-step is CPU/Gemini-heavy
   - Recommendation: treat the entire Polymarket cycle as heavy (conservative), matching the CONTEXT.md decision
   - Confidence: HIGH (user-decided)

2. **Job dependency graph implementation**
   - CONTEXT.md says: "if upstream job (e.g., GDELT poller) failed today, downstream jobs (e.g., daily pipeline) skip execution with a warning"
   - APScheduler 3.x has no built-in job dependency mechanism
   - Implementation: use the `JobFailureTracker` to check upstream status before running downstream jobs. The daily pipeline wrapper checks if GDELT and RSS ran successfully today before proceeding.
   - This is discretionary per CONTEXT.md. Recommend a simple dict of `{downstream_id: [upstream_ids]}` checked in the job wrapper.
   - Confidence: MEDIUM (implementation approach is custom, not library-provided)

3. **Schedule editing from admin dashboard persistence**
   - CONTEXT.md says schedule changes persist to `system_config` table
   - Implementation: on startup, read schedule overrides from `system_config` before registering jobs. `scheduler.reschedule_job(job_id, trigger=new_trigger)` for runtime changes.
   - The `reschedule_job` method exists on APScheduler 3.x BaseScheduler
   - Confidence: HIGH

4. **Kill button for ProcessPoolExecutor jobs**
   - CONTEXT.md says: "For ProcessPoolExecutor jobs, terminates the worker process"
   - This requires access to the subprocess PID, which ProcessPoolExecutor doesn't directly expose
   - Implementation: use `concurrent.futures.ProcessPoolExecutor` directly (not APScheduler's wrapper) so we can call `process.terminate()` on the worker
   - Alternative: store the `Future` object and cancel it, then restart the executor
   - Confidence: LOW (needs prototyping -- killing a process pool worker is messy)

## Sources

### Primary (HIGH confidence)
- APScheduler 3.11.2 User Guide: https://apscheduler.readthedocs.io/en/3.x/userguide.html
- APScheduler 3.11.2 AsyncIOScheduler docs: https://apscheduler.readthedocs.io/en/3.x/modules/schedulers/asyncio.html
- APScheduler 3.11.2 Base Scheduler docs: https://apscheduler.readthedocs.io/en/3.x/modules/schedulers/base.html
- APScheduler 3.11.2 Events docs: https://apscheduler.readthedocs.io/en/3.x/modules/events.html
- Existing codebase: `src/ingest/gdelt_poller.py`, `src/ingest/rss_daemon.py`, `src/ingest/acled_poller.py`, `src/ingest/advisory_poller.py`, `src/api/app.py`, `scripts/daily_forecast.py`, `scripts/retrain_tkg.py`

### Secondary (MEDIUM confidence)
- `.planning/research/STACK_V3.md` -- v3.0 stack research (APScheduler selection rationale)
- APScheduler GitHub Issue #235 (memory leak on exceptions)
- APScheduler GitHub Issue #304 (AsyncIOScheduler executor defaults)

### Tertiary (LOW confidence)
- ProcessPoolExecutor kill semantics -- needs prototyping

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- APScheduler 3.11.2 is locked by prior decision, API verified against official docs
- Architecture: HIGH -- patterns derived from official docs + codebase analysis of all 7 pollers/scripts
- Pitfalls: HIGH -- derived from codebase-specific analysis (GDELTPoller._last_url, RSS tier split, etc.)
- ProcessPoolExecutor kill: LOW -- no clean mechanism found, needs prototyping

**Research date:** 2026-03-05
**Valid until:** 2026-04-05 (APScheduler 3.x is stable, no breaking changes expected)
