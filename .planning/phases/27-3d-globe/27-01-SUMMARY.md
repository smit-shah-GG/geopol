---
phase: 27-3d-globe
plan: 01
subsystem: frontend-globe
tags: [globe.gl, three.js, 3d-rendering, webgl, code-splitting]
completed: 2026-03-09
duration: 5min
dependency-graph:
  requires: [phase-24, phase-25]
  provides: [GlobeMap-class, globe-dependencies, earth-texture, vite-globe-chunk]
  affects: [27-02, 27-03]
tech-stack:
  added: [globe.gl@2.45.0, three@0.183.2, "@types/three@0.183.1"]
  patterns: [debounced-flush, dual-timer-coalesce, dynamic-import-lazy-load, ring-winding-reversal]
key-files:
  created:
    - frontend/src/components/GlobeMap.ts
    - frontend/public/textures/earth-topo-bathy.jpg
  modified:
    - frontend/package.json
    - frontend/package-lock.json
    - frontend/vite.config.ts
decisions:
  - "h3-js dynamically imported and cached in GlobeMap (not top-level) -- avoids polluting module scope"
  - "GeoJSON ring reversal cached per feature ISO code -- prevents repeated array copies on every flush"
  - "Three.js dynamically imported inside applyAtmosphereGlow() only -- avoids 600KB parse-time load"
  - "Layers 1 and 5 share single polygonsData channel -- scenario polygons at altitude 0.004 above choropleth at 0.002"
  - "Heatmap degrades from H3 hexagons to point markers on 3D (globe.gl has no native H3 support)"
  - "HTML marker click handlers include stopPropagation to prevent globe polygon click bubbling"
metrics:
  tasks-completed: 2/2
  lines-added: 1112
  type-errors: 0
---

# Phase 27 Plan 01: Core GlobeMap 3D Renderer Summary

GlobeMap.ts (1112 lines) wrapping globe.gl with identical 13-method public API as DeckGLMap, plus npm deps, earth texture, and Vite code-split chunk.

## What Was Done

### Task 1: Install dependencies and copy texture asset
- Installed `globe.gl ^2.45.0`, `three ^0.183.2`, `@types/three ^0.183.1`
- Copied `earth-topo-bathy.jpg` (715KB) from WM to `frontend/public/textures/`
- Added `globe: ['globe.gl', 'three']` manual chunk to `frontend/vite.config.ts`
- Commit: `4c1643e`

### Task 2: Create GlobeMap.ts -- 3D globe renderer with DeckGLMap-compatible API
- 1112-line class with all 13 public methods matching DeckGLMap signatures exactly
- 5 analytic layers mapped to 4 globe.gl data channels:
  - L1 ForecastRiskChoropleth -> polygonsData (risk-colored country fills)
  - L2 ActiveForecastMarkers -> htmlElementsData (DOM dots at centroids)
  - L3 KnowledgeGraphArcs -> arcsData (sentiment-colored great-circle arcs)
  - L4 GDELTEventHeatmap -> pointsData (colored dots from H3 hex centers)
  - L5 ScenarioZones -> polygonsData (overlaid at higher altitude)
- Custom atmosphere: dual SphereGeometry BackSide meshes with Geopol blue (#4080dd)
- MeshStandardMaterial globe surface (roughness 0.8, metalness 0.1, emissive #0a1f2e)
- Debounced flush: 100ms trailing + 300ms max wait (WM dual-timer pattern)
- Auto-rotate with 120s idle timeout (longer than WM's 60s for analytical use)
- GeoJSON ring winding reversal with feature cache (Natural Earth CW -> Three.js CCW)
- Country click dispatches `country-selected` CustomEvent (both polygon and marker clicks)
- 8 VIEW_POVS region presets exported for GlobeHud integration (Plan 03)
- Commit: `fb6bd31`

## Deviations from Plan

None -- plan executed exactly as written.

## Decisions Made

1. **h3-js dynamic import with cache**: Rather than a top-level import or window global hack, h3-js is dynamically imported on first heatmap flush and cached as `{ cellToLatLng }`. This avoids polluting the module scope and is cleanly typed.

2. **Feature-level ring reversal cache**: Instead of caching per-ring (WM's approach with zone+country+ringIdx keys), cache entire reversed Features keyed by ISO code. Simpler for Geopol's use case where each country has one canonical Feature.

3. **Shared polygonsData channel**: Layers 1 and 5 both use `polygonsData`. Combined into a single array with discriminated `_kind` field. Scenario/delta polygons use altitude 0.004 to visually float above choropleth at 0.002.

4. **Marker click stopPropagation**: HTML marker elements call `e.stopPropagation()` on click to prevent the click from also firing on the underlying polygon, which would dispatch duplicate `country-selected` events.

## Verification Results

| Check | Result |
|-------|--------|
| `npx tsc --noEmit` | 0 errors |
| `npm run build` | Success (5.19s), `globe` chunk in output |
| 13 public methods present | All 13 confirmed via grep |
| globe.gl + three in package.json | globe.gl ^2.45.0, three ^0.183.2 |
| earth-topo-bathy.jpg in public/textures/ | 715KB present |
| Production build succeeds | Yes, with globe chunk |

## Next Phase Readiness

Plan 02 (MapContainer wrapper + globe-screen integration) can proceed immediately. GlobeMap exports:
- `GlobeMap` class (same constructor pattern as DeckGLMap: `new GlobeMap(container)`)
- `VIEW_POVS` object (8 region presets for GlobeHud selector)
- Type re-exports via DeckGLMap imports (LayerId, HexBinDatum, BilateralArcDatum, RiskDeltaDatum)

No blockers. No deferred issues.
