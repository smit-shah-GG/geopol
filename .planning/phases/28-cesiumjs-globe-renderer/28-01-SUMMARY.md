---
phase: 28-cesiumjs-globe-renderer
plan: 01
subsystem: ui
tags: [cesium, vite, vite-plugin-static-copy, navbar, scene-mode]

# Dependency graph
requires:
  - phase: 27-3d-globe
    provides: "globe.gl + deck.gl dual-renderer architecture being replaced"
provides:
  - "CesiumJS npm package installed and importable"
  - "Vite config with CESIUM_BASE_URL define and static asset copying"
  - "NavBar 3-pill segmented control dispatching globe-view-toggle with mode payload"
  - "Old rendering packages (globe.gl, three, deck.gl, maplibre-gl) removed"
affects: [28-02-PLAN, 28-03-PLAN]

# Tech tracking
tech-stack:
  added: [cesium@1.139.1, vite-plugin-static-copy@3.2.0]
  patterns: [viteStaticCopy for CesiumJS Workers/Assets/ThirdParty/Widgets, segmented control with CSS-injected styles]

key-files:
  modified:
    - frontend/package.json
    - frontend/vite.config.ts
    - frontend/src/components/NavBar.ts

key-decisions:
  - "vite-plugin-static-copy over abandoned vite-plugin-cesium for CesiumJS Vite integration"
  - "cesiumStatic as CESIUM_BASE_URL path segment"
  - "Module-scope <style> injection for scene-pill CSS (same pattern as admin styles)"

patterns-established:
  - "CESIUM_BASE_URL define in vite.config.ts for CesiumJS worker/asset resolution"
  - "globe-view-toggle CustomEvent now carries { mode: '3d' | 'columbus' | '2d' } payload"

# Metrics
duration: 3min
completed: 2026-03-12
---

# Phase 28 Plan 01: CesiumJS Build Infrastructure Summary

**CesiumJS installed with Vite static-copy integration, old dual-renderer deps purged, NavBar 3-pill scene mode control [3D][CV][2D] wired**

## Performance

- **Duration:** 3 min
- **Started:** 2026-03-12T08:53:44Z
- **Completed:** 2026-03-12T08:56:33Z
- **Tasks:** 2
- **Files modified:** 3 (+ package-lock.json)

## Accomplishments
- Removed 10 old rendering packages (globe.gl, three, @types/three, deck.gl, @deck.gl/core, @deck.gl/layers, @deck.gl/geo-layers, @deck.gl/aggregation-layers, @deck.gl/mapbox, maplibre-gl) -- net -3,059 lines in package-lock.json
- Installed cesium@^1.139.1 and vite-plugin-static-copy@^3.2.0
- Rewrote vite.config.ts with CESIUM_BASE_URL define, viteStaticCopy plugin (4 targets), cesium manual chunk
- Replaced NavBar single toggle button with 3-pill segmented control dispatching mode-aware CustomEvents

## Task Commits

Each task was committed atomically:

1. **Task 1: Swap npm dependencies and configure Vite for CesiumJS** - `a176fdb` (chore)
2. **Task 2: Replace NavBar toggle with 3-pill segmented control** - `8c7e870` (feat)

## Files Created/Modified
- `frontend/package.json` - cesium + vite-plugin-static-copy added; 10 old packages removed
- `frontend/vite.config.ts` - CESIUM_BASE_URL define, viteStaticCopy plugin with 4 targets, cesium manual chunk
- `frontend/src/components/NavBar.ts` - 3-pill segmented control [3D][CV][2D] with mode payload on globe-view-toggle

## Decisions Made
- Used `vite-plugin-static-copy` (official Cesium Vite example pattern) over abandoned `vite-plugin-cesium`
- `cesiumStatic` as the base URL path segment for Cesium Workers/Assets/ThirdParty/Widgets
- Module-scope `<style>` injection for `.nav-scene-mode` and `.scene-pill` CSS (same pattern used by admin styles elsewhere in codebase)
- Immediate pill state update on click for responsive feel, with `globe-mode-changed` listener for authoritative sync from CesiumMap

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
- Vite build fails on `maplibre-gl/dist/maplibre-gl.css` import in `globe-screen.ts` -- expected, resolved in Plan 03 when globe-screen.ts is rewired. Not a Plan 01 issue.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- CesiumJS is importable (`import { Viewer } from 'cesium'` resolves)
- Vite config copies all required CesiumJS static assets to build output
- NavBar dispatches `globe-view-toggle` with `{ mode: '3d' | 'columbus' | '2d' }` -- ready for CesiumMap (Plan 02)
- Plan 02 can proceed immediately to create CesiumMap.ts
- Plan 03 will delete GlobeMap.ts, DeckGLMap.ts, MapContainer.ts and rewire globe-screen.ts

---
*Phase: 28-cesiumjs-globe-renderer*
*Completed: 2026-03-12*
