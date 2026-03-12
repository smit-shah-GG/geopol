---
status: complete
phase: 28-cesiumjs-globe-renderer
source: [28-01-SUMMARY.md, 28-02-SUMMARY.md, 28-03-SUMMARY.md]
started: 2026-03-12T10:00:00Z
updated: 2026-03-12T10:30:00Z
---

## Current Test

[testing complete]

## Tests

### 1. CesiumJS Globe Renders
expected: Navigate to /globe. A 3D globe renders with dark basemap tiles and visible atmosphere glow. No blank screen, no WebGL errors in console. The globe should be interactive (drag to rotate, scroll to zoom).
result: issue
reported: "It's not scaled correctly — globe rendering in tiny ~250px area in top-left corner, CesiumJS navigation widgets showing with white background"
severity: blocker
fix: Added `import 'cesium/Build/Cesium/Widgets/widgets.css'` to CesiumMap.ts and `fullscreenButton: false` to Viewer options. Verified fixed during testing.

### 2. NavBar 3-Pill Scene Mode Control
expected: On /globe, the NavBar shows a segmented control with three pills: [3D] [CV] [2D]. The "3D" pill should be active (highlighted) by default. The pills should NOT be visible on /dashboard or /forecasts.
result: pass

### 3. Scene Mode Toggle (3D to 2D)
expected: Click the [2D] pill. The globe morphs (animated transition) from 3D sphere to flat 2D map view. The [2D] pill becomes active. Click [3D] to morph back to sphere. Click [CV] for Columbus View (a flat projection that preserves depth).
result: issue
reported: "pass, but after ~10 seconds crashes with RangeError: Too many properties to enumerate in computeRhumbLineSubdivision. Also: 2D reload has weird stretch zoom, Antarctica tearing in 2D, visual defects near Russia in 3D"
severity: major
fix: Set `arcType: GEODESIC`, `granularity: Math.PI`, `outline: false` on choropleth polygon entities to prevent rhumb subdivision overflow. Verified morph works. Visual defects remain (cosmetic).

### 4. Risk Choropleth Layer
expected: Countries on the globe are colored by risk intensity (red/orange shading). Countries with higher risk scores appear more intensely colored. Countries with no data appear neutral/uncolored.
result: pass

### 5. Country Click Opens Drill-Down
expected: Click on a country polygon on the globe. The GlobeDrillDown slide-in panel appears showing that country's name, risk score, active forecasts, and event sparkline. The globe flies to center on the clicked country.
result: issue
reported: "pass, but nothing works to deselect a country — arcs stay filtered to that country, have to reload to get rid of it"
severity: minor

### 6. Layer Toggle Pills
expected: Five layer toggle pills are visible (Risk, Markers, Arcs, Heatmap, Scenarios). Clicking a pill toggles that layer's visibility on/off. The pill visual state (active/inactive) reflects the current layer state.
result: pass

### 7. No Old Renderer Console Errors
expected: Open browser DevTools console on /globe. There should be zero errors mentioning Three.js, deck.gl, MapboxOverlay, maplibre-gl, globe.gl, DrawLayersPass, or NaN bounding sphere. CesiumJS may log informational messages but no errors.
result: pass

### 8. Scene Mode Persists Across Navigation
expected: On /globe, switch to [2D] mode. Navigate to /dashboard, then back to /globe. The globe should remember and restore 2D mode (not reset to 3D).
result: pass

## Summary

total: 8
passed: 5
issues: 3
pending: 0
skipped: 0

## Gaps

- truth: "Globe fills viewport and renders without unstyled widget chrome"
  status: failed
  reason: "User reported: globe rendering in tiny area, navigation widgets with white background"
  severity: blocker
  test: 1
  root_cause: "CesiumJS widgets.css never imported — .cesium-widget has no width:100%;height:100%, fullscreenButton not disabled"
  artifacts:
    - path: "frontend/src/components/CesiumMap.ts"
      issue: "Missing CSS import and fullscreenButton:false"
  missing:
    - "import 'cesium/Build/Cesium/Widgets/widgets.css' at top of CesiumMap.ts"
    - "fullscreenButton: false in Viewer options"
  debug_session: ""
  fixed_during_uat: true

- truth: "Scene mode morph completes without errors"
  status: failed
  reason: "User reported: RangeError Too many properties to enumerate in computeRhumbLineSubdivision after morph"
  severity: major
  test: 3
  root_cause: "GeoJsonDataSource choropleth polygons use default ArcType.RHUMB which triggers rhumb-line subdivision in workers — complex country polygons (Russia ~10K vertices) overflow subdivision buffer"
  artifacts:
    - path: "frontend/src/components/CesiumMap.ts"
      issue: "Default rhumb subdivision on choropleth polygon entities"
  missing:
    - "Set arcType:GEODESIC, granularity:Math.PI, outline:false on choropleth entities after GeoJsonDataSource.load"
  debug_session: ""
  fixed_during_uat: true

- truth: "Clicking empty space or same country deselects current country"
  status: failed
  reason: "User reported: nothing works to deselect a country, arcs stay filtered, have to reload"
  severity: minor
  test: 5
  root_cause: "ScreenSpaceEventHandler click handler only handles entity picks, no logic for clicking empty space to clear selection"
  artifacts:
    - path: "frontend/src/components/CesiumMap.ts"
      issue: "Click handler missing deselect-on-empty-click logic"
  missing:
    - "When click pick returns no entity, call setSelectedCountry(null) to clear selection and dispatch country-deselected event"
  debug_session: ""

## Remaining Cosmetic Issues (not gaps)

- 2D mode has weird stretch zoom on initial load/navigate-back (camera setup issue)
- Antarctica polygon tearing in 2D projection
- Russia polygon visual defects in 3D (antimeridian crossing)
- All three are in the choropleth Risk layer (confirmed by toggle test)
