---
phase: 27-3d-globe
plan: 03
subsystem: ui
tags: [globe.gl, geojson, h3-js, typescript, layer-sync]

# Dependency graph
requires:
  - phase: 27-3d-globe (plans 01, 02)
    provides: GlobeMap renderer, MapContainer wrapper, LayerPillBar
provides:
  - Fixed polygon rendering (removed incorrect ring winding reversal)
  - Working forecast markers via scenario entity ISO extraction
  - Race-safe heatmap flush via debounced path
  - View-synced layer pill bar via globe-mode-changed listener
affects: []

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Scenario entity ISO extraction: recursive collectIsos() over child_scenarios for 2-char uppercase country codes"
    - "Controller sync pattern: LayerPillBar.syncFromController() resyncs from active renderer on view toggle"

key-files:
  modified:
    - frontend/src/components/GlobeMap.ts
    - frontend/src/components/DeckGLMap.ts
    - frontend/src/components/LayerPillBar.ts

key-decisions:
  - "globe.gl handles GeoJSON ring winding internally -- manual CW->CCW reversal was the bug, not the fix"
  - "Marker ISO extraction uses scenarios[].entities[] recursive walk, not calibration.category (which holds CAMEO strings)"
  - "h3-js async completion routes through scheduleFlush() debounced path, not standalone flushPoints()"
  - "LayerPillBar syncs via globe-mode-changed CustomEvent -- same event MapContainer already dispatches"

patterns-established:
  - "Entity ISO extraction: /^[A-Z]{2}$/ regex on entity strings from recursive scenario tree walk"

# Metrics
duration: 3min
completed: 2026-03-10
---

# Phase 27 Plan 03: Globe Layer Fixes Summary

**Fixed 4 UAT bugs: polygon winding reversal removal, marker ISO extraction from scenario entities, heatmap debounced flush, and LayerPillBar view-mode sync**

## Performance

- **Duration:** 3 min
- **Started:** 2026-03-10T08:43:09Z
- **Completed:** 2026-03-10T08:46:03Z
- **Tasks:** 3
- **Files modified:** 3

## Accomplishments
- Removed incorrect GeoJSON ring winding reversal that corrupted polygon rendering and click detection on the 3D globe
- Fixed forecast markers in both GlobeMap and DeckGLMap -- markers now extract country ISOs from scenario entity trees instead of the CAMEO calibration.category field
- Eliminated heatmap flush race condition by routing h3-js async completion through the debounced scheduleFlush() path
- Added LayerPillBar syncFromController() method with globe-mode-changed event listener so pill active states track the active renderer

## Task Commits

Each task was committed atomically:

1. **Task 1: fix-polygon-winding** - `d28db74` (fix)
2. **Task 2: fix-marker-iso-extraction** - `1a29740` (fix)
3. **Task 3: fix-heatmap-flush-and-pillbar-sync** - `292652d` (fix)

## Files Created/Modified
- `frontend/src/components/GlobeMap.ts` - Removed getReversedFeature() and reversedFeatureCache; rewrote updateForecasts() ISO extraction; fixed flushPoints() async callback
- `frontend/src/components/DeckGLMap.ts` - Rewrote updateForecasts() ISO extraction (same logic, position-based MarkerDatum)
- `frontend/src/components/LayerPillBar.ts` - Added syncFromController() public method, globe-mode-changed listener, cleanup in destroy()

## Decisions Made
- globe.gl handles GeoJSON ring winding internally -- the manual CW->CCW reversal in getReversedFeature() was corrupting polygons, not fixing them
- calibration.category contains CAMEO domain strings (e.g. "conflict"), not ISO country codes -- markers have never rendered since initial implementation
- Marker extraction uses the same recursive entity walk pattern already used in setSelectedForecast() -- consistent approach across both methods
- h3-js async callback must go through scheduleFlush() to maintain the single-atomic-flush invariant that prevents globe.gl data channel interference

## Deviations from Plan

None -- plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- All 5 UAT polygon/marker/heatmap/pillbar issues addressed
- TypeScript compiles clean, production build succeeds
- Phase 27 gap closure plan complete

---
*Phase: 27-3d-globe*
*Completed: 2026-03-10*
