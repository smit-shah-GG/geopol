"""
APScheduler-based daemon consolidation for all background jobs.

Provides a single AsyncIOScheduler with dual executors (async default +
process pool for heavy jobs), shared dependency injection, failure tracking
with auto-pause, and a job registry for all 9 background tasks.
"""

from src.scheduler.core import create_scheduler, get_scheduler, shutdown_scheduler
from src.scheduler.dependencies import get_shared_deps, init_shared_deps
from src.scheduler.registry import register_all_jobs

__all__ = [
    "create_scheduler",
    "get_scheduler",
    "shutdown_scheduler",
    "init_shared_deps",
    "get_shared_deps",
    "register_all_jobs",
]
