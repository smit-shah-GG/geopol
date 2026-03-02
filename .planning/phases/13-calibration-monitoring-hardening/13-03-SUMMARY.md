---
phase: 13-calibration-monitoring-hardening
plan: 03
subsystem: monitoring
tags: [smtp, psutil, brier-score, alerting, disk-monitoring, feed-staleness]

# Dependency graph
requires:
  - phase: 13-01
    provides: "Settings fields (smtp_*, feed_staleness_hours, drift_threshold_pct, disk_*_pct) and ORM models (IngestRun, Prediction, OutcomeRecord)"
provides:
  - "AlertManager with per-type cooldown and async-safe SMTP"
  - "FeedMonitor for GDELT staleness detection"
  - "DriftMonitor for rolling Brier score drift detection"
  - "BudgetMonitor for read-only Gemini API usage reporting"
  - "DiskMonitor with emergency cleanup at critical threshold"
affects: ["13-04", "13-06", "13-07"]

# Tech tracking
tech-stack:
  added: ["psutil>=5.9"]
  patterns: ["async-safe SMTP via asyncio.to_thread()", "check + check_and_alert monitor pattern", "fire-and-forget alert delivery"]

key-files:
  created:
    - "src/monitoring/__init__.py"
    - "src/monitoring/alert_manager.py"
    - "src/monitoring/feed_monitor.py"
    - "src/monitoring/drift_monitor.py"
    - "src/monitoring/budget_monitor.py"
    - "src/monitoring/disk_monitor.py"
  modified:
    - "pyproject.toml"

key-decisions:
  - "monitoring.py converted to monitoring/ package -- legacy DataQualityMonitor preserved in __init__.py, import path unchanged"
  - "DriftMonitor uses PostgreSQL-backed rolling Brier score, replacing legacy JSON-file DriftDetector"
  - "BudgetMonitor is read-only reporter, NOT the budget enforcer (that remains BudgetTracker)"
  - "DiskMonitor emergency_cleanup targets only .csv/.zip/.gz/.CSV files in data/ dir -- never touches .db files"
  - "Minimum 20 resolved predictions required before drift detection triggers (prevents noisy small-sample alerts)"

patterns-established:
  - "Monitor pattern: check_X() returns status dict, check_and_alert() wraps check + conditional alert"
  - "All monitors designed for periodic invocation, no background threads"
  - "AlertManager is_enabled=False graceful degradation when SMTP unconfigured"

# Metrics
duration: 4min
completed: 2026-03-02
---

# Phase 13 Plan 03: Monitoring Package Summary

**SMTP alert manager with per-type cooldown + feed/drift/budget/disk monitors using PostgreSQL-backed queries and psutil**

## Performance

- **Duration:** 4 min
- **Started:** 2026-03-02T06:19:28Z
- **Completed:** 2026-03-02T06:23:35Z
- **Tasks:** 2/2
- **Files modified:** 7

## Accomplishments
- AlertManager with per-type cooldown, asyncio.to_thread() for non-blocking SMTP, fire-and-forget error handling
- FeedMonitor detects GDELT staleness by querying most recent successful IngestRun
- DriftMonitor computes rolling 30-day Brier score vs all-time baseline with 20-sample minimum
- BudgetMonitor reports Gemini API budget utilisation from PostgreSQL (read-only, no enforcement)
- DiskMonitor checks root partition via psutil, warning at 80%, critical at 90% with emergency cleanup

## Task Commits

Each task was committed atomically:

1. **Task 1: Create alert manager with SMTP sending and rate limiting** - `63fbf71` (feat)
2. **Task 2: Create feed, drift, budget, and disk monitors** - `89f2a26` (feat)

## Files Created/Modified
- `src/monitoring/__init__.py` - Package init (legacy DataQualityMonitor preserved, import path fixed)
- `src/monitoring/alert_manager.py` - SMTP alerting with per-type cooldown and async-safe sending
- `src/monitoring/feed_monitor.py` - GDELT feed staleness detection from ingest_runs
- `src/monitoring/drift_monitor.py` - Rolling Brier score computation and drift detection
- `src/monitoring/budget_monitor.py` - Gemini API usage tracking for health reporting
- `src/monitoring/disk_monitor.py` - Disk usage monitoring with emergency cleanup
- `pyproject.toml` - Added psutil>=5.9 dependency

## Decisions Made
- Converted `src/monitoring.py` to `src/monitoring/` package: legacy DataQualityMonitor preserved in `__init__.py`, relative import `from .constants` changed to `from src.constants` to fix package resolution. Existing imports in `pipeline.py` and `run_pipeline.py` continue to work unchanged.
- DriftMonitor replaces the legacy DriftDetector conceptually (JSON file-backed, synchronous) with PostgreSQL-backed async implementation. Legacy DriftDetector not removed -- it serves a different purpose (ECE-based drift from in-memory prediction lists).
- BudgetMonitor is strictly read-only. Budget enforcement remains in BudgetTracker (`src/pipeline/budget_tracker.py`).
- DiskMonitor emergency_cleanup only targets `.csv`, `.zip`, `.gz`, `.CSV` suffix files in the data directory -- never touches `.db` files to prevent accidental SQLite deletion.
- Minimum sample threshold of 20 resolved predictions for drift detection prevents false alarms from small windows.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Fixed relative import in monitoring package __init__.py**
- **Found during:** Task 1 (package conversion)
- **Issue:** Converting `src/monitoring.py` to `src/monitoring/__init__.py` broke `from .constants import GDELT100_THRESHOLD` -- relative import now resolves within `monitoring/` package instead of `src/`
- **Fix:** Changed to `from src.constants import GDELT100_THRESHOLD`
- **Files modified:** `src/monitoring/__init__.py`
- **Verification:** `from src.monitoring import DataQualityMonitor` imports successfully
- **Committed in:** `63fbf71` (Task 1 commit)

---

**Total deviations:** 1 auto-fixed (1 blocking)
**Impact on plan:** Import fix necessary for package conversion to work. No scope creep.

## Issues Encountered
None beyond the deviation above.

## User Setup Required
None - no external service configuration required. SMTP settings are optional (AlertManager gracefully disables when smtp_host is empty).

## Next Phase Readiness
- All five monitoring modules importable and verified
- Monitors ready for integration into health endpoint (Plan 06) and scheduler (Plan 07)
- AlertManager ready for use by all monitors and the daily pipeline
- psutil dependency declared and resolved

---
*Phase: 13-calibration-monitoring-hardening*
*Completed: 2026-03-02*
