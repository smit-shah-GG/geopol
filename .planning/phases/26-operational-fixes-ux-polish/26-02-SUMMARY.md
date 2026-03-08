---
phase: 26-operational-fixes-ux-polish
plan: 02
subsystem: ui
tags: [router, circuit-breaker, cache-invalidation, expandable-cards, sessionStorage, polymarket]

# Dependency graph
requires:
  - phase: 12-wm-derived-frontend
    provides: Router, Panel system, circuit breaker architecture
  - phase: 25-frontend-finalization
    provides: expandable-card shared system, skeleton loading, Panel.showSkeleton()
  - phase: 18-polymarket-driven-forecasting
    provides: ComparisonPanel, forecastClient.getComparisons(), ComparisonPanelItem type
provides:
  - Route refresh with cache bust on every navigation (same-route + cross-route)
  - CircuitBreaker.invalidateCache() and ForecastServiceClient.bustAllCaches() API
  - ComparisonPanel expandable cards with lazy forecast fetch and progressive disclosure
  - SubmissionForm draft persistence via sessionStorage
affects:
  - 26-03 (route refresh behavior available for clickable forecasts)
  - 27-globe (globe remount/reset guaranteed on navigation)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Cache bust on navigation: bustAllCaches() before unmount/remount"
    - "replaceState for same-route navigation (no history pollution)"
    - "sessionStorage for ephemeral form draft persistence across remounts"
    - "Lazy forecast fetch with in-component cache (forecastCache Map)"

key-files:
  modified:
    - frontend/src/app/router.ts
    - frontend/src/utils/circuit-breaker.ts
    - frontend/src/services/forecast-client.ts
    - frontend/src/components/ComparisonPanel.ts
    - frontend/src/components/SubmissionForm.ts

key-decisions:
  - "replaceState for same-route clicks -- prevents back-button history pollution"
  - "bustAllCaches clears inFlight dedup map in addition to breaker caches"
  - "ComparisonPanel uses buildExpandedContent (not buildExpandableCard) -- custom dual-bar collapsed view preserved"
  - "Forecast cache scoped to ComparisonPanel instance -- cleared on destroy()"
  - "sessionStorage draft key 'geopol-submission-draft' -- survives remount, cleared on successful submission"
  - "expandedIds typed as Set<number> (comp.id) not Set<string> (forecast_id) -- matches ComparisonPanelItem.id"

patterns-established:
  - "Cache bust pattern: every navigate() call triggers bustAllCaches() before resolve()"
  - "Lazy expand pattern: expanded section fetches on first open, caches, re-renders on subsequent toggles"

# Metrics
duration: 4min
completed: 2026-03-08
---

# Phase 26 Plan 02: Route Refresh + ComparisonPanel Expandable Cards Summary

**Router forces full unmount/remount with cache invalidation on every navigation; ComparisonPanel entries expand to show lazy-fetched forecast details with ensemble, calibration, evidence, and Polymarket comparison data**

## Performance

- **Duration:** 4 min
- **Started:** 2026-03-08T18:45:02Z
- **Completed:** 2026-03-08T18:48:40Z
- **Tasks:** 2
- **Files modified:** 5

## Accomplishments
- Every navigation (including same-route clicks) triggers full cache bust + unmount/remount cycle -- no stale data
- ComparisonPanel entries are clickable with keyboard-accessible expand/collapse revealing full forecast details
- SubmissionForm textarea draft survives /forecasts same-route refresh via sessionStorage
- Globe resets to default view on /globe navigation (DeckGLMap destroyed and recreated)

## Task Commits

Each task was committed atomically:

1. **Task 1: Route refresh with cache bust** - `79e003e` (feat)
2. **Task 2: ComparisonPanel expandable cards with lazy forecast fetch** - `f611f09` (feat)

## Files Created/Modified
- `frontend/src/utils/circuit-breaker.ts` - Added invalidateCache() method clearing cache + resetting data state
- `frontend/src/services/forecast-client.ts` - Added bustAllCaches() clearing all 4 breakers + inFlight map
- `frontend/src/app/router.ts` - Removed same-route guard, added replaceState + bustAllCaches before remount
- `frontend/src/components/SubmissionForm.ts` - sessionStorage draft persistence on input, restore on mount, clear on submit
- `frontend/src/components/ComparisonPanel.ts` - Full rewrite: expandable entries with lazy forecast fetch, keyboard a11y, Polymarket-specific expanded section

## Decisions Made
- **replaceState for same-route** -- pushState would create duplicate history entries; replaceState preserves clean back-button behavior
- **bustAllCaches clears inFlight map** -- stale in-flight promises could return cached results from previous navigation cycle
- **Custom collapsed view retained** -- ComparisonPanel's dual GP/PM bars are its signature visual; buildExpandableCard would replace them with generic forecast card layout
- **expandedIds as Set<number>** -- ComparisonPanelItem.id is a number (database PK), distinct from forecast_id (string UUID)
- **Forecast cache scoped to instance** -- cleared on destroy() to prevent cross-mount stale data; rebuild is cheap (lazy-fetch on expand)

## Deviations from Plan

None -- plan executed exactly as written.

## Issues Encountered
None.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Route refresh infrastructure ready for Plan 03 (clickable forecasts navigate to /forecasts with pre-populated question)
- Globe remount behavior verified -- Plan 27 (3D globe) will get clean DeckGLMap lifecycle on every /globe navigation
- ComparisonPanel progressive disclosure parity achieved with Active Forecasts and My Forecasts panels

---
*Phase: 26-operational-fixes-ux-polish*
*Completed: 2026-03-08*
