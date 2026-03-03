---
phase: 15-url-routing-dashboard
plan: 03
subsystem: ui
tags: [typescript, panels, forecast-client, status-badges, polling, css]

# Dependency graph
requires:
  - phase: 14-backend-api-hardening
    provides: submission endpoints (POST /forecasts/submit, GET /forecasts/requests)
  - phase: 15-url-routing-dashboard
    provides: 4-column dashboard layout, Panel base class, RefreshScheduler
provides:
  - MyForecastsPanel with status badges and click-to-open completed forecasts
  - SourcesPanel with health-derived staleness indicators
  - forecast-client submission methods (submitQuestion, confirmSubmission, getRequests)
  - TypeScript DTOs for submission flow (ParsedQuestionResponse, ForecastRequestStatus, ConfirmSubmissionResponse)
  - Dashboard Col 3 fully populated
affects: [16-globe-forecasts-screens]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Push-based panel updates from dashboard data load and refresh scheduler"
    - "Status badges with CSS animation (pulse for processing state)"
    - "Staleness color-coding derived from HealthResponse subsystem timestamps"

key-files:
  created:
    - frontend/src/components/MyForecastsPanel.ts
    - frontend/src/components/SourcesPanel.ts
  modified:
    - frontend/src/types/api.ts
    - frontend/src/services/forecast-client.ts
    - frontend/src/screens/dashboard-screen.ts
    - frontend/src/styles/panels.css

key-decisions:
  - "SourcesPanel receives data via push from health refresh, not independent fetch -- avoids duplicate /health calls"
  - "getRequests() uses health circuit breaker -- low-priority polling shares existing breaker group"
  - "Mutations (submitQuestion, confirmSubmission) bypass dedup and circuit breaker -- user expects immediate feedback"

patterns-established:
  - "Push-based SourcesPanel: piggybacks on existing health refresh rather than adding a new endpoint poll"
  - "Status badge CSS pattern: .status-{state} classes with consistent color scheme matching semantic variables"

# Metrics
duration: 4min
completed: 2026-03-03
---

# Phase 15 Plan 03: Dashboard Panels (MyForecasts + Sources) Summary

**MyForecastsPanel with 5-state status badges and SourcesPanel with staleness indicators wired into dashboard Col 3, forecast-client extended with submission flow methods**

## Performance

- **Duration:** 4 min
- **Started:** 2026-03-03T12:06:46Z
- **Completed:** 2026-03-03T12:11:21Z
- **Tasks:** 2
- **Files modified:** 6

## Accomplishments
- Extended forecast-client with submitQuestion(), confirmSubmission(), getRequests() methods matching Phase 14 backend DTOs
- Created MyForecastsPanel showing user submissions with 5-state status badges (pending/confirmed/processing/complete/failed) and click-to-open completed forecasts via ScenarioExplorer
- Created SourcesPanel displaying data source health with color-coded staleness indicators (green < 30min, yellow 30min-2h, red > 2h)
- Wired both panels into dashboard Col 3 with 30s polling via RefreshScheduler

## Task Commits

Each task was committed atomically:

1. **Task 1: Submission/request types + forecast-client methods** - `5ebdd5c` (feat)
2. **Task 2: MyForecastsPanel + SourcesPanel + dashboard wiring + CSS** - `a206c4f` (feat)

## Files Created/Modified
- `frontend/src/types/api.ts` - Added ParsedQuestionResponse, ForecastRequestStatus, ConfirmSubmissionResponse interfaces
- `frontend/src/services/forecast-client.ts` - Added submitQuestion(), confirmSubmission(), getRequests() methods with appropriate dedup/breaker config
- `frontend/src/components/MyForecastsPanel.ts` - User submission tracking panel with status badges and completed forecast click handler
- `frontend/src/components/SourcesPanel.ts` - Data source health/staleness indicators derived from HealthResponse subsystems
- `frontend/src/screens/dashboard-screen.ts` - Wired MyForecastsPanel and SourcesPanel into Col 3, added refresh tasks
- `frontend/src/styles/panels.css` - CSS for my-forecast rows, status badges, pulse animation, source rows, staleness colors

## Decisions Made
- SourcesPanel receives data via push from health refresh rather than making independent /health calls -- eliminates duplicate requests
- getRequests() shares the health circuit breaker group (low-priority polling endpoint)
- Mutations (submitQuestion, confirmSubmission) bypass dedup and circuit breaker to ensure immediate user feedback
- Completed forecast click dispatches window-level `forecast-selected` event for ScenarioExplorer compatibility

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Dashboard Col 3 fully populated with MyForecastsPanel and SourcesPanel
- Submission flow frontend-to-backend wiring complete (types, client methods, panel)
- All 3 plans of Phase 15 now complete -- ready for Phase 16 (Globe & Forecasts Screens)

---
*Phase: 15-url-routing-dashboard*
*Completed: 2026-03-03*
