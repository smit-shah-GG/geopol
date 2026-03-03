---
phase: 15-url-routing-dashboard
plan: 01
subsystem: ui
tags: [router, pushstate, flexbox, dark-theme, code-splitting, view-transitions]

# Dependency graph
requires:
  - phase: 12-wm-derived-frontend
    provides: Panel components, DeckGLMap, app-context, theme-manager
  - phase: 14-backend-api-hardening
    provides: top_forecast field in CountryRiskSummary API response
provides:
  - Client-side Router (pushState/popstate with View Transition API)
  - NavBar component with active link highlighting
  - 3-screen SPA routing (/dashboard, /globe, /forecasts)
  - 4-column flexbox dashboard layout replacing 3-column grid
  - Screen mount/unmount lifecycle pattern
  - DeckGLMap lazy-loaded via dynamic import (code-splitting)
  - Dark-only theme (light theme removed entirely)
  - top_question -> top_forecast rename complete
affects: [15-url-routing-dashboard, 16-globe-forecasts-screens]

# Tech tracking
tech-stack:
  added: []
  patterns: [pushState SPA routing, screen mount/unmount lifecycle, View Transition API, dynamic import code-splitting]

key-files:
  created:
    - frontend/src/app/router.ts
    - frontend/src/components/NavBar.ts
    - frontend/src/screens/dashboard-screen.ts
    - frontend/src/screens/globe-screen.ts
    - frontend/src/screens/forecasts-screen.ts
  modified:
    - frontend/src/types/api.ts
    - frontend/src/components/RiskIndexPanel.ts
    - frontend/src/app/panel-layout.ts
    - frontend/src/app/app-context.ts
    - frontend/src/main.ts
    - frontend/src/utils/theme-manager.ts
    - frontend/src/components/DeckGLMap.ts
    - frontend/src/styles/main.css
    - frontend/src/styles/panels.css
    - frontend/index.html

key-decisions:
  - "View Transition API with sync fallback -- 150ms crossfade, no polyfill"
  - "Module-scoped state for screen lifecycle (not class instances)"
  - "DeckGLMap dynamic import splits 1.9MB of deck.gl+maplibre to /globe only"
  - "country-filter-changed dispatched directly from RiskIndexPanel click handler"

patterns-established:
  - "Screen lifecycle: mountX(container, ctx) / unmountX(ctx) exported from screens/"
  - "Router dispatches 'route-changed' CustomEvent for NavBar sync"
  - "Dynamic import() for heavy visualization bundles at route boundaries"

# Metrics
duration: 9min
completed: 2026-03-03
---

# Phase 15 Plan 01: Screen Routing & Dashboard Infrastructure Summary

**pushState SPA router with 3 URL-routed screens, 4-column flexbox dashboard, dark-only theme, DeckGLMap code-split to /globe**

## Performance

- **Duration:** 9 min
- **Started:** 2026-03-03T11:53:12Z
- **Completed:** 2026-03-03T12:01:43Z
- **Tasks:** 3
- **Files modified:** 15

## Accomplishments
- Three-screen SPA with bookmarkable URLs (/dashboard, /globe, /forecasts) and browser history support
- 4-column flexbox dashboard layout (15/35/30/20%) replacing 3-column CSS Grid
- DeckGLMap lazy-loaded only on /globe route via dynamic import -- deck.gl+maplibre chunks not loaded on dashboard
- Light theme completely removed (CSS, theme-manager, DeckGLMap, index.html)
- top_question -> top_forecast breaking rename fixed across all frontend files
- CSS bundle reduced 4.1% (33.37KB -> 31.99KB) from light theme removal

## Task Commits

Each task was committed atomically:

1. **Task 1: Breaking change fix + Router + NavBar + Theme simplification** - `8d3390d` (feat)
2. **Task 2: 4-column layout, screen lifecycle, and main.ts rewrite** - `dc6f2ef` (feat)
3. **Task 3: CSS overhaul -- nav bar, 4-column layout, light theme removal, View Transitions** - `6a03750` (style)

## Files Created/Modified
- `frontend/src/app/router.ts` - Client-side Router class with pushState, popstate, View Transition API
- `frontend/src/components/NavBar.ts` - Navigation bar with 3 route links and active state
- `frontend/src/screens/dashboard-screen.ts` - Dashboard mount/unmount with all 6 panels + scheduler
- `frontend/src/screens/globe-screen.ts` - Globe screen with dynamic DeckGLMap import
- `frontend/src/screens/forecasts-screen.ts` - Placeholder for Phase 16 submission UI
- `frontend/src/app/panel-layout.ts` - 4-column flexbox layout (replaced 3-column grid)
- `frontend/src/app/app-context.ts` - Added activeCountryFilter + setCountryFilter
- `frontend/src/main.ts` - Router-driven bootstrap (removed all direct panel imports)
- `frontend/src/types/api.ts` - top_question -> top_forecast rename
- `frontend/src/components/RiskIndexPanel.ts` - top_forecast + country-filter-changed dispatch
- `frontend/src/utils/theme-manager.ts` - Dark-only (removed toggleTheme, light paths)
- `frontend/src/components/DeckGLMap.ts` - Removed light theme paths, unused getCurrentTheme import
- `frontend/src/styles/main.css` - Nav bar styles, dashboard-columns, view transitions, removed light theme + grid
- `frontend/src/styles/panels.css` - Removed 4 light theme overrides, added screen-placeholder
- `frontend/index.html` - Simplified inline theme script to dark-only

## Decisions Made
- View Transition API used with synchronous fallback -- no polyfill needed, graceful degradation
- Module-scoped state pattern for screen lifecycle (let variables at module scope rather than class instances)
- DeckGLMap dynamic import at route level provides meaningful code-splitting (8.65KB DeckGLMap chunk separate from 1.9MB deck.gl+maplibre)
- ScenarioExplorer and CountryBriefPage properly destroyed on dashboard unmount (prevents global listener leaks)

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] DeckGLMap light theme code caused TypeScript errors**
- **Found during:** Task 1 (theme simplification)
- **Issue:** Narrowing Theme type to literal 'dark' made DeckGLMap's light-theme comparisons unreachable, triggering TS2367
- **Fix:** Removed light theme code paths from DeckGLMap (basemap switch, accentColor, defaultFill, countryStroke, arc colors), removed unused getCurrentTheme import, removed LIGHT_STYLE constant
- **Files modified:** frontend/src/components/DeckGLMap.ts
- **Verification:** npx tsc --noEmit passes
- **Committed in:** 8d3390d (Task 1 commit)

**2. [Rule 3 - Blocking] index.html inline theme script had light branch**
- **Found during:** Task 1 (theme simplification)
- **Issue:** The FOUC-prevention inline script still checked for 'light' theme in localStorage
- **Fix:** Simplified to single `document.documentElement.dataset.theme = 'dark'`
- **Files modified:** frontend/index.html
- **Verification:** File content verified
- **Committed in:** 8d3390d (Task 1 commit)

---

**Total deviations:** 2 auto-fixed (2 blocking)
**Impact on plan:** Both fixes were direct consequences of the planned theme simplification. No scope creep.

## Issues Encountered
None -- all three tasks executed cleanly.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Router and screen lifecycle infrastructure ready for Plan 02 (ForecastPanel refactor + search integration)
- Col 3 empty and waiting for Plan 03 (MyForecastsPanel + SourcesPanel)
- Globe screen functional with DeckGLMap, ready for Phase 16 full-viewport expansion

---
*Phase: 15-url-routing-dashboard*
*Completed: 2026-03-03*
