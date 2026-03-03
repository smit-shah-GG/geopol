---
phase: 15-url-routing-dashboard
plan: 02
subsystem: ui
tags: [typescript, d3, search, progressive-disclosure, forecast-ux, abort-controller]

# Dependency graph
requires:
  - phase: 15-url-routing-dashboard/01
    provides: Dashboard 4-column layout, screen mount/unmount lifecycle, router
  - phase: 14-backend-api-hardening/04
    provides: GET /forecasts/search endpoint (BAPI-04)
provides:
  - Expandable forecast cards with two-column progressive disclosure layout
  - SearchBar component with debounced search and country/category filters
  - ScenarioExplorer node tooltip on hover
  - SearchResponse/SearchResult TypeScript types
  - ForecastServiceClient.search() method with AbortController
affects: [16-globe-forecasts-screens]

# Tech tracking
tech-stack:
  added: []
  patterns: [diff-based-dom-updates, abort-controller-race-prevention, debounced-search]

key-files:
  created:
    - frontend/src/components/SearchBar.ts
  modified:
    - frontend/src/components/ForecastPanel.ts
    - frontend/src/components/ScenarioExplorer.ts
    - frontend/src/services/forecast-client.ts
    - frontend/src/types/api.ts
    - frontend/src/screens/dashboard-screen.ts
    - frontend/src/styles/panels.css

key-decisions:
  - "Diff-based DOM updates in ForecastPanel -- cardElements Map preserves expanded state across 60s refresh"
  - "Mini d3 tree limited to 2 levels deep with 20-char labels -- preview only, full detail in ScenarioExplorer"
  - "SearchBar is a standalone class, not a Panel subclass -- no header/resize/badge overhead for an inline control"
  - "Tooltip is HTML div positioned via pageX/pageY, not SVG title -- better styling and multiline support"

patterns-established:
  - "Diff-based DOM updates: store cardElements Map, update in-place, preserve UI state across data refresh"
  - "AbortController race prevention: abort previous in-flight request before starting new one"
  - "Debounced search: 300ms delay on text input, immediate on dropdown change"

# Metrics
duration: 7min
completed: 2026-03-03
---

# Phase 15 Plan 02: Forecast UX -- Progressive Disclosure + Search + Tooltip Summary

**Expandable forecast cards with inline two-column layout (ensemble/calibration + mini d3 tree/evidence), debounced SearchBar with AbortController race prevention, ScenarioExplorer node tooltip**

## Performance

- **Duration:** 7 min
- **Started:** 2026-03-03T12:05:44Z
- **Completed:** 2026-03-03T12:12:26Z
- **Tasks:** 2
- **Files modified:** 7

## Accomplishments
- ForecastPanel rewritten with diff-based DOM updates -- expanded card state survives data refresh
- Two-column expanded layout: left (probability bar, ensemble weights stacked bar, calibration metadata grid), right (mini d3 scenario tree, top 3 evidence summaries, "View Full Analysis" button)
- SearchBar component with 300ms debounced text input, country/category dropdowns, and AbortController-based race condition prevention
- ScenarioExplorer tooltip showing full scenario description on node hover
- SearchResponse/SearchResult types and ForecastServiceClient.search() method added

## Task Commits

Each task was committed atomically:

1. **Task 1: SearchBar + search API types + forecast-client search method** - `308cac0` (feat)
2. **Task 2: ForecastPanel expandable cards + ScenarioExplorer tooltip + CSS + wiring** - `c97b1a1` (feat)

## Files Created/Modified
- `frontend/src/components/SearchBar.ts` - Debounced search with country/category dropdowns, AbortController
- `frontend/src/components/ForecastPanel.ts` - Rewritten with expandable cards, diff-based DOM updates, mini d3 tree
- `frontend/src/components/ScenarioExplorer.ts` - HTML tooltip on node hover
- `frontend/src/services/forecast-client.ts` - search() method with AbortSignal
- `frontend/src/types/api.ts` - SearchResponse, SearchResult interfaces
- `frontend/src/screens/dashboard-screen.ts` - SearchBar wired into Col 2
- `frontend/src/styles/panels.css` - CSS for search-bar, expanded cards, mini tree, tooltip, view-full button

## Decisions Made
- Diff-based DOM updates via cardElements Map rather than full re-render -- required for expanded state persistence across 60s refresh cycles
- Mini d3 tree limited to 2 depth levels with 20-char labels -- it's a preview, full interaction in ScenarioExplorer
- SearchBar as standalone class (not Panel subclass) -- avoids header/resize/badge overhead for an inline search control
- ScenarioExplorer tooltip uses HTML div (not SVG `<title>`) positioned via pageX/pageY -- enables proper styling and multiline text

## Deviations from Plan

None -- plan executed exactly as written.

## Issues Encountered
- Concurrent plan 15-03 execution modified dashboard-screen.ts and api.ts/forecast-client.ts in parallel, adding MyForecastsPanel/SourcesPanel and submission types. The SearchBar wiring in dashboard-screen.ts was folded into the 15-03 commit. No conflicts -- changes were orthogonal.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- ForecastPanel progressive disclosure complete (FUX-01)
- Search with filtering complete (FUX-03)
- ScenarioExplorer tooltip complete (FUX-02)
- Ready for Phase 16 (Globe & Forecasts screens)

---
*Phase: 15-url-routing-dashboard*
*Completed: 2026-03-03*
