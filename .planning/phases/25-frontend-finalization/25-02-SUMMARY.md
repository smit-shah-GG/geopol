---
phase: 25-frontend-finalization
plan: 02
subsystem: ui
tags: [skeleton, error-handling, empty-state, lazy-loading, sparkline, code-splitting, debounce]

# Dependency graph
requires:
  - phase: 25-01
    provides: Panel base class showSkeleton/showErrorWithRetry/showRefreshToast/dismissToast methods, skeleton.ts builder, timing.ts debounce
provides:
  - All 10 Panel subclasses use skeleton/error/retry/toast/empty pattern
  - GlobeDrillDown sparkline wired to real GDELT event data
  - CountryBriefPage lazy-loaded on dashboard (separate chunk)
  - ScenarioExplorer lazy-loaded on forecasts screen
  - SearchBar uses shared debounce from timing.ts
  - CAMEO trend stub removed from CountryBriefPage
affects: [25-03]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "hasData field pattern: track whether panel has rendered real content at least once"
    - "isTransientError() classifier: timeout/503/502/504/ECONNREFUSED/network/fetch -> amber toast, else red"
    - "Lazy-load modal pattern: proxy event listener -> dynamic import -> cache instance -> remove proxy"
    - "SVG sparkline via createElementNS: polyline + filled polygon area"

key-files:
  created: []
  modified:
    - "frontend/src/components/ForecastPanel.ts"
    - "frontend/src/components/RiskIndexPanel.ts"
    - "frontend/src/components/ComparisonPanel.ts"
    - "frontend/src/components/SystemHealthPanel.ts"
    - "frontend/src/components/PolymarketPanel.ts"
    - "frontend/src/components/MyForecastsPanel.ts"
    - "frontend/src/components/NewsFeedPanel.ts"
    - "frontend/src/components/EventTimelinePanel.ts"
    - "frontend/src/components/SourcesPanel.ts"
    - "frontend/src/components/LiveStreamsPanel.ts"
    - "frontend/src/components/SearchBar.ts"
    - "frontend/src/components/GlobeDrillDown.ts"
    - "frontend/src/components/CountryBriefPage.ts"
    - "frontend/src/screens/dashboard-screen.ts"
    - "frontend/src/screens/forecasts-screen.ts"
    - "frontend/src/styles/panels.css"

key-decisions:
  - "isTransientError regex matches timeout, 503, 502, 504, ECONNREFUSED, network, fetch -- amber toast for transient, red for persistent"
  - "EventTimelinePanel: removed currentEvents field entirely (hasData replaces the length check)"
  - "ComparisonPanel: constructor no longer calls showEmpty() -- skeleton from base class shows instead"
  - "GlobeDrillDown sparkline: fetches 500 events for 30-day window via getEvents({country, start_date, limit})"
  - "CountryBriefPage CAMEO trend: removed entirely rather than faking -- no historical data available for real trends"
  - "CountryBriefPage lazy-load: proxy event listener on country-brief-requested, cached instance, cleanup on unmount"
  - "ScenarioExplorer lazy-load: proxy listener removed after first load since SE registers its own listener in constructor"
  - "ScenarioExplorer stays statically imported on dashboard (core interaction) -- only lazy on forecasts + globe screens"

patterns-established:
  - "Panel hasData pattern: private hasData = false; skeleton on !hasData, toast on hasData+error, errorWithRetry on !hasData+error"
  - "Enhanced empty state: .empty-state-enhanced with icon (unicode 32px) + title + desc + optional CTA button"
  - "Lazy modal pattern: module-scoped handler -> dynamic import() -> cache -> remove proxy -> cleanup on unmount"

# Metrics
duration: 12min
completed: 2026-03-08
---

# Phase 25 Plan 02: Per-Panel UX Overhaul Summary

**Skeleton/error/retry/toast/empty patterns applied to all 10 Panel subclasses; GlobeDrillDown sparkline wired to real GDELT data; CountryBriefPage lazy-loaded (30.6 kB chunk); CAMEO trend stub removed**

## Performance

- **Duration:** 12 min
- **Started:** 2026-03-08T16:39:49Z
- **Completed:** 2026-03-08T16:52:10Z
- **Tasks:** 2
- **Files modified:** 16

## Accomplishments
- All 10 Panel subclasses upgraded with three-tier loading pattern: skeleton shimmer on initial load, error+retry block on initial failure, amber/red toast on refresh failure with stale data preserved
- 12 contextual empty states with unicode icons, titles, and descriptions (including filtered vs unfiltered variants for ForecastPanel and NewsFeedPanel)
- MyForecastsPanel empty state includes "Submit a Forecast" CTA button that navigates to /forecasts
- GlobeDrillDown sparkline renders real 30-day GDELT event data as SVG polyline with filled area
- CountryBriefPage CAMEO trend column removed (was hardcoded rising/stable stub)
- CountryBriefPage lazy-loaded on dashboard via dynamic import (30.6 kB separate chunk, loaded on first country click)
- ScenarioExplorer lazy-loaded on forecasts screen via dynamic import with proxy listener pattern
- SearchBar debounce refactored to use shared utility from @/utils/timing.ts

## Task Commits

1. **Task 1: Upgrade all Panel subclasses with skeleton/error/empty patterns** - `c0fd007` (feat)
2. **Task 2: Fix stale placeholders, lazy-load CountryBriefPage and ScenarioExplorer** - `3833ff5` (feat)

## Files Created/Modified
- `frontend/src/components/ForecastPanel.ts` - hasData field, skeleton/error/retry/toast in refresh(), enhanced empty states
- `frontend/src/components/RiskIndexPanel.ts` - hasData field, skeleton/error/retry/toast, globe icon empty state
- `frontend/src/components/ComparisonPanel.ts` - hasData field, skeleton/error/retry/toast, scale icon empty state
- `frontend/src/components/SystemHealthPanel.ts` - hasData field, skeleton/error/retry/toast
- `frontend/src/components/PolymarketPanel.ts` - hasData field, chart icon empty state (was bare text)
- `frontend/src/components/MyForecastsPanel.ts` - hasData field, skeleton/error/retry/toast, CTA empty state
- `frontend/src/components/NewsFeedPanel.ts` - hasData field, skeleton/error/retry/toast, dual empty states
- `frontend/src/components/EventTimelinePanel.ts` - hasData field (replaces currentEvents), skeleton/error/retry/toast
- `frontend/src/components/SourcesPanel.ts` - hasData field, skeleton/error/retry/toast, database icon empty state
- `frontend/src/components/LiveStreamsPanel.ts` - error+retry on YouTube API failure
- `frontend/src/components/SearchBar.ts` - shared debounce import, inline debounce removed
- `frontend/src/components/GlobeDrillDown.ts` - real SVG sparkline, EventDTO import, sparkline rendering
- `frontend/src/components/CountryBriefPage.ts` - CAMEO trend column/cells/stub removed
- `frontend/src/screens/dashboard-screen.ts` - CountryBriefPage lazy-load via dynamic import
- `frontend/src/screens/forecasts-screen.ts` - ScenarioExplorer lazy-load via dynamic import
- `frontend/src/styles/panels.css` - sparkline CSS (replaced placeholder styles)

## Decisions Made
- Used `isTransientError()` regex to classify error severity (amber for transient network issues, red for persistent errors)
- CountryBriefPage CAMEO trend removed entirely rather than faked -- no historical comparison data available for real trend computation
- ScenarioExplorer remains statically imported on dashboard screen (core interaction) but lazily on forecasts + globe screens
- Sparkline fetches up to 500 events for 30-day window to ensure adequate data for daily bucketing

## Deviations from Plan

None -- plan executed exactly as written.

## Issues Encountered
- Linter auto-fixed several unrelated files during editing (DeckGLMap, GlobeHud, NavBar, etc.) -- these were excluded from Task 1/2 commits to keep commits atomic
- ScenarioExplorer can't be code-split because dashboard-screen.ts statically imports it -- this is by design per the plan's explicit instruction

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- All panels have consistent loading/error/empty UX patterns
- CountryBriefPage and ScenarioExplorer properly code-split
- Ready for Plan 03 (CSS polish, animation, a11y final pass)
- No blockers

---
*Phase: 25-frontend-finalization*
*Completed: 2026-03-08*
