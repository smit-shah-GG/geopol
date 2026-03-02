---
phase: 13-calibration-monitoring-hardening
plan: 04
subsystem: infra
tags: [logging, systemd, rotation, structured-logs, daemon-supervision]

# Dependency graph
requires:
  - phase: 13-01
    provides: "log_dir and log_retention_days settings fields"
provides:
  - "Daily-rotated JSON log files via TimedRotatingFileHandler"
  - "setup_logging_from_settings() convenience wrapper"
  - "systemd service units for GDELT ingest, RSS poller, daily forecast"
  - "systemd timer for daily 06:00 UTC forecast pipeline"
affects: ["13-05 (alerting)", "13-06 (health dashboard)", "deployment documentation"]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "TimedRotatingFileHandler with midnight UTC rotation and configurable retention"
    - "systemd oneshot + timer pattern for scheduled pipelines"
    - "Security hardening directives in all service units (NoNewPrivileges, ProtectSystem)"

key-files:
  created:
    - "systemd/geopol-ingest.service"
    - "systemd/geopol-rss.service"
    - "systemd/geopol-daily-forecast.service"
    - "systemd/geopol-daily-forecast.timer"
  modified:
    - "src/logging_config.py"

key-decisions:
  - "File handler always uses JSON format regardless of stderr json_format setting"
  - "File handler level set to DEBUG (captures everything); stderr handler respects configured level"
  - "systemd units use ProtectSystem=strict + NoNewPrivileges for defense-in-depth"
  - "Daily forecast timer uses Persistent=true + RandomizedDelaySec=300 for resilient scheduling"

patterns-established:
  - "Additive handler pattern: file handler supplements stderr, never replaces it"
  - "Template service files with install instructions as comments"

# Metrics
duration: 4min
completed: 2026-03-02
---

# Phase 13 Plan 04: Logging & Systemd Summary

**Daily-rotated JSON structured log files via TimedRotatingFileHandler with 30-day retention, plus 4 systemd units for unattended daemon supervision**

## Performance

- **Duration:** 4 min
- **Started:** 2026-03-02T06:19:45Z
- **Completed:** 2026-03-02T06:24:14Z
- **Tasks:** 2
- **Files modified:** 5

## Accomplishments
- Extended setup_logging() with optional TimedRotatingFileHandler that writes JSON logs to daily-rotated files
- Added setup_logging_from_settings() convenience wrapper that reads log_dir/log_retention_days from Settings
- Created systemd service units for GDELT ingest (Restart=on-failure, 30s), RSS poller (Restart=on-failure, 60s), and daily forecast (oneshot, 1h timeout)
- Created systemd timer triggering daily forecast at 06:00 UTC with Persistent=true for missed-run catchup

## Task Commits

Each task was committed atomically:

1. **Task 1: Extend logging config with file rotation** - `f6be947` (feat)
2. **Task 2: Create systemd service and timer units** - `0d5cf2c` (feat)

## Files Created/Modified
- `src/logging_config.py` - Added TimedRotatingFileHandler, setup_logging_from_settings()
- `systemd/geopol-ingest.service` - GDELT daemon with restart supervision and security hardening
- `systemd/geopol-rss.service` - RSS poller daemon with restart supervision
- `systemd/geopol-daily-forecast.service` - Oneshot forecast pipeline, 1h timeout
- `systemd/geopol-daily-forecast.timer` - Daily 06:00 UTC trigger with jitter and persistence

## Decisions Made
- File handler always uses _JSONFormatter regardless of stderr json_format setting -- structured logs on disk are mandatory for machine parsing, human-readable is only for interactive stderr
- File handler level set to DEBUG unconditionally -- disk captures everything for post-incident analysis, stderr filters at configured level
- All systemd units include NoNewPrivileges=true, ProtectSystem=strict, ProtectHome=true, PrivateTmp=true for defense-in-depth
- Timer uses RandomizedDelaySec=300 to avoid thundering-herd if multiple instances exist
- StartLimitIntervalSec=600, StartLimitBurst=5 on daemon services to prevent restart loops

## Deviations from Plan
None - plan executed exactly as written.

## Issues Encountered
None.

## User Setup Required
None - systemd units are templates requiring path editing for deployment, but no external service configuration needed.

## Next Phase Readiness
- Logging infrastructure ready for alerting system (13-05) to use structured log analysis
- systemd units ready for deployment documentation
- Pre-existing test failures (test_auto_load_graceful_when_missing, flaky test_fit_on_synthetic_graph) are unrelated to this plan

---
*Phase: 13-calibration-monitoring-hardening*
*Completed: 2026-03-02*
