---
phase: 28-cesiumjs-globe-renderer
plan: 02
subsystem: ui
tags: [cesium, webgl, geojson, choropleth, billboard, polyline, scene-mode, morph]

# Dependency graph
requires:
  - phase: 28-cesiumjs-globe-renderer
    provides: "CesiumJS npm package installed, Vite configured with CESIUM_BASE_URL and viteStaticCopy"
  - phase: 27-3d-globe
    provides: "country-geometry.ts service with getFeatureByIso/getCentroid/getNameByIso/getFeatureCollection"
provides:
  - "CesiumMap.ts: single CesiumJS globe renderer with 5 analytic layers"
  - "Exported types: CesiumMap, LAYER_IDS, LayerId, HexBinDatum, BilateralArcDatum, RiskDeltaDatum"
  - "Public API: updateRiskScores/updateForecasts/updateHeatmapData/updateArcData/updateRiskDeltas"
  - "Scene mode: setSceneMode/getSceneMode with morph animation and flyTo queueing"
  - "Layer visibility: setLayerVisible/getLayerVisible/setLayerDefaults (LayerController interface)"
  - "Selection: setSelectedCountry with per-country arc filtering, setSelectedForecast with scenarioIsos"
  - "Events: country-selected CustomEvent on click, globe-mode-changed on morph complete"
affects: [28-03-PLAN]

# Tech tracking
tech-stack:
  added: []
  patterns: [GeoJsonDataSource choropleth with entityIsoMap for O(1) re-coloring, PolylineCollection parabolic arcs with 4t(1-t) formula, CustomDataSource for independently-togglable scenario zones, requestRenderMode with explicit requestRender after every data update, morphStart/morphComplete listener pair for morph-safe camera operations]

key-files:
  created:
    - frontend/src/components/CesiumMap.ts

key-decisions:
  - "CustomDataSource for scenario zones -- independent .show toggle from choropleth GeoJsonDataSource"
  - "GeoJsonDataSource.load for choropleth, then entityIsoMap for O(1) re-coloring on subsequent updateRiskScores calls"
  - "morphStart disables requestRenderMode temporarily to ensure morph animation plays; morphComplete re-enables it"
  - "Entity properties tagged with _cesiumIso and _cesiumLayerId for pick identification (avoids getValue on every hover)"
  - "Dual-mode scenario zones: accent highlights when forecast selected, red/green risk deltas when not"
  - "flattenRing helper for GeoJSON-to-PolygonHierarchy conversion (avoids .flat() type narrowing issues)"

patterns-established:
  - "CesiumJS entity tagging via PropertyBag.addProperty('_cesiumIso', iso) for O(1) pick identification"
  - "_rebuildArcs pattern: filter allArcData by selectedCountryIso for per-country view, full dataset when null"
  - "Lazy h3-js import cached at instance level for async heatmap rendering"

# Metrics
duration: 8min
completed: 2026-03-12
---

# Phase 28 Plan 02: CesiumMap.ts Globe Renderer Summary

**1156-line CesiumJS Viewer with 5 analytic layers (choropleth, markers, arcs, heatmap, scenario zones), ScreenSpaceEventHandler click/hover, scene mode morphing (3D/CV/2D), and requestRenderMode GPU efficiency**

## Performance

- **Duration:** 8 min
- **Started:** 2026-03-12T09:01:43Z
- **Completed:** 2026-03-12T09:10:13Z
- **Tasks:** 1
- **Files created:** 1 (1156 lines)

## Accomplishments
- Single CesiumJS renderer replacing GlobeMap.ts (1112 lines) + DeckGLMap.ts (800 lines) + MapContainer.ts (327 lines) -- net reduction of ~1083 lines
- All 5 analytic layers implemented with correct primitive types (GeoJsonDataSource for choropleth, BillboardCollection for markers, PolylineCollection for arcs, PointPrimitiveCollection for heatmap, CustomDataSource for scenario zones)
- Full public API matching CONTEXT.md contract: 5 data push methods, 2 selection methods, 2 camera methods, 3 layer visibility methods, 2 scene mode methods, destroy
- Per-country arc filtering via _rebuildArcs with selectedCountryIso filter on allArcData
- Morph-safe camera operations: pendingFlyTo queue executes on morphComplete, requestRenderMode temporarily disabled during morph animation

## Task Commits

Each task was committed atomically:

1. **Task 1: CesiumMap.ts -- Viewer, layers, events, scene modes** - `e338aa2` (feat)

## Files Created/Modified
- `frontend/src/components/CesiumMap.ts` - Complete CesiumJS globe renderer with 5 layers, ScreenSpaceEventHandler, scene mode morphing, layer visibility, selection state, camera flyTo

## Decisions Made
- Used CustomDataSource (not second GeoJsonDataSource) for scenario zones -- provides independent `.show` toggle and `.entities.removeAll()` for clean rebuild, without interfering with choropleth GeoJsonDataSource
- Entity properties tagged with `_cesiumIso` and `_cesiumLayerId` via `PropertyBag.addProperty()` for efficient pick identification -- avoids calling `.getValue(JulianDate)` and `normalizeCode()` on every MOUSE_MOVE
- `morphStart` event disables `requestRenderMode` during morph animation, `morphComplete` re-enables it -- solves RESEARCH.md open question #2 about requestRenderMode interaction with morph animations
- Parabolic arc peak height formula: `max(100k, min(2M, surfaceDist * 50k))` meters -- scales with geographic distance, capped for visual consistency
- Canvas-rendered marker pins with outer glow + core circle + rim -- more visually distinct than plain dots on the 3D globe

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Missing Critical] Dual-mode scenario zones**
- **Found during:** Task 1
- **Issue:** Initial implementation only showed risk deltas. DeckGLMap has dual-mode: accent highlights when forecast selected, risk deltas otherwise.
- **Fix:** Added `selectedForecast` check in `_rebuildScenarioZones`: mode A (accent highlights for scenarioIsos) when forecast selected, mode B (red/green risk deltas from riskDeltaMap) otherwise. Added `_rebuildScenarioZones()` call from `setSelectedForecast()`.
- **Files modified:** frontend/src/components/CesiumMap.ts
- **Verification:** TypeScript compiles clean, selectedForecast field is now read (no TS6133)
- **Committed in:** e338aa2

---

**Total deviations:** 1 auto-fixed (1 missing critical)
**Impact on plan:** Parity with existing DeckGLMap dual-mode behavior. Essential for correct operation.

## Issues Encountered
- GeoJSON Position type `number[]` caused TypeScript `noUncheckedIndexedAccess` errors when accessing `pos[0]`, `pos[1]` -- resolved with non-null assertions (`pos[0]!`, `pos[1]!`) since GeoJSON Position always has at least 2 elements
- `skyAtmosphere` property is nullable in CesiumJS types -- added null guard
- `ConstantProperty` was missing from initial import list -- added to cesium imports

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- CesiumMap.ts is complete and type-checks cleanly against cesium@1.139.x
- Full project build will still fail because globe-screen.ts, LayerPillBar.ts still import from old files (GlobeMap.ts, DeckGLMap.ts, MapContainer.ts)
- Plan 03 will delete old renderer files, rewire globe-screen.ts to use CesiumMap, and repoint LayerPillBar imports

---
*Phase: 28-cesiumjs-globe-renderer*
*Completed: 2026-03-12*
