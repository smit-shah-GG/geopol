---
status: diagnosed
phase: 27-3d-globe
source: 27-01-SUMMARY.md, 27-02-SUMMARY.md
started: 2026-03-09T21:00:00Z
updated: 2026-03-09T21:25:00Z
---

## Current Test

[testing complete]

## Tests

### 1. 3D Globe Default View
expected: Navigate to /globe. The screen shows a 3D spherical globe (not the flat 2D map) with a dark topographic texture, a blue atmosphere glow ring around the edge, and a black background. Countries should be visible on the sphere surface.
result: pass

### 2. 3D/2D Toggle Button
expected: On /globe, the NavBar shows a "2D" toggle button (indicating you can switch to 2D). Clicking it switches to the flat deck.gl map. The button label changes to "3D". Clicking again returns to the 3D globe.
result: pass

### 3. Risk Choropleth on 3D Globe
expected: Countries on the 3D globe are colored by risk score — high-risk countries (e.g., conflict zones) appear in warmer/brighter colors, low-risk or no-data countries appear neutral/dark. Not all countries the same color.
result: issue
reported: "Risk is entirely fucked, so is scenarios. Layering most likely is fucked."
severity: blocker

### 4. Country Click on 3D Globe
expected: Click any visible country polygon on the 3D globe. The GlobeDrillDown panel slides in from the right showing that country's active forecasts, risk score, and event sparkline — same panel as on the 2D map.
result: issue
reported: "Wrong country on click, probably related to layering fuck"
severity: blocker

### 5. Forecast Markers on 3D Globe
expected: Small dot markers appear at country centroids for countries with active forecasts. These are the same "ActiveForecastMarkers" layer as 2D. Clicking a marker should also open GlobeDrillDown for that country.
result: issue
reported: "markers layer does entirely nothing on both 2d and 3d"
severity: blocker

### 6. Region Presets in GlobeHud
expected: The GlobeHud (top-left overlay) shows 8 region preset buttons (Global, Americas, Europe, MENA, Asia, LatAm, Africa, Oceania). Clicking one (e.g., "MENA") smoothly flies the globe to center on that region. The clicked button shows an active/highlighted state.
result: pass

### 7. Layer Toggle on 3D Globe
expected: The layer pill bar at the bottom of /globe works while in 3D mode. Toggle the "Heatmap" layer off — heatmap points disappear from the globe. Toggle it back on — they reappear. Same for other layers.
result: issue
reported: "pills work absolutely, but heatmap and markers do fucking nothing, and risk and scenarios are fucked as mentioned previously. Heatmap has loading race: arcs load first, heatmap appears later when h3-js loads, arcs disappear when heatmap appears."
severity: major

### 8. Auto-Rotate
expected: Leave the 3D globe alone for a few seconds. It should slowly auto-rotate. Click/drag the globe to interact — rotation pauses. After releasing and waiting ~2 minutes, it should resume rotating.
result: pass

### 9. Mode Persistence
expected: While on /globe, switch to 2D view using the toggle. Navigate away (e.g., /dashboard), then return to /globe. It should load in 2D mode (your preference was saved). Switch back to 3D, reload the page — it should load in 3D.
result: pass

### 10. Independent Layer State
expected: In 3D view, toggle the "Arcs" layer OFF. Switch to 2D view — the Arcs layer should still be ON in 2D (each view has its own layer state). Toggle Arcs off in 2D, switch back to 3D — Arcs should still be off in 3D from your earlier toggle.
result: issue
reported: "fail, changes persist bilaterally"
severity: major

## Summary

total: 10
passed: 5
issues: 5
pending: 0
skipped: 0

## Gaps

- truth: "Countries on the 3D globe are colored by risk score with visible differentiation"
  status: failed
  reason: "User reported: Risk is entirely fucked, so is scenarios. Layering most likely is fucked."
  severity: blocker
  test: 3
  root_cause: "GlobeMap.getReversedFeature() reverses GeoJSON ring winding (CW->CCW), but globe.gl handles winding internally via its d3-geo pipeline. The reversal inverts polygon topology, causing corrupted rendering (inside-out polygons, wrong fills, visual artifacts). Scenario polygons use the same reversed features."
  artifacts:
    - path: "frontend/src/components/GlobeMap.ts"
      issue: "getReversedFeature() at line 1120 reverses ring coordinates. flushPolygons() at line 830 calls getReversedFeature() for every polygon."
  missing:
    - "Remove ring winding reversal — pass GeoJSON features directly to globe.gl polygonsData without coordinate modification"
    - "Clear reversedFeatureCache usage"
  debug_session: ""

- truth: "Clicking a country polygon on the 3D globe opens GlobeDrillDown for that country"
  status: failed
  reason: "User reported: Wrong country on click, probably related to layering fuck"
  severity: blocker
  test: 4
  root_cause: "Consequence of ring winding reversal (same root cause as Test 3). Reversed polygon geometry causes Three.js raycasting to hit wrong mesh faces, resolving to incorrect country ISO on click. Fix polygons → fix click detection."
  artifacts:
    - path: "frontend/src/components/GlobeMap.ts"
      issue: "onPolygonClick handler at line 579 correctly reads normalizeCode from polygon properties, but the polygon being clicked is wrong due to corrupted geometry"
  missing:
    - "Same fix as Test 3 — remove ring reversal"
  debug_session: ""

- truth: "Forecast markers appear at country centroids on both 2D and 3D views"
  status: failed
  reason: "User reported: markers layer does entirely nothing on both 2d and 3d"
  severity: blocker
  test: 5
  root_cause: "PRE-EXISTING BUG since Phase 12. Both DeckGLMap.updateForecasts() and GlobeMap.updateForecasts() extract country ISO via `f.calibration.category.length === 2`. But CalibrationDTO.category contains CAMEO category STRINGS like 'conflict', 'diplomatic', 'economic' — NOT 2-character ISO codes. The length check fails for word-length strings, and even when CAMEO root codes like '14' pass the length check, getCentroid('14') returns null. Result: zero markers ever created."
  artifacts:
    - path: "frontend/src/components/GlobeMap.ts"
      issue: "updateForecasts() at line 241: `f.calibration.category.length === 2` — wrong field for ISO extraction"
    - path: "frontend/src/components/DeckGLMap.ts"
      issue: "updateForecasts() at line 229: identical broken logic"
    - path: "frontend/src/types/api.ts"
      issue: "ForecastResponse has no country_iso field — must extract from scenarios[].entities[]"
  missing:
    - "Extract country ISOs from scenarios[].entities[] (2-char uppercase strings matching /^[A-Z]{2}$/)"
    - "Fix in BOTH DeckGLMap and GlobeMap"
    - "One forecast may map to multiple countries — create a marker per unique ISO"
  debug_session: ""

- truth: "Heatmap points and layer data render correctly when toggled on in 3D"
  status: failed
  reason: "User reported: pills work but heatmap does nothing initially, loads late after h3-js, arcs disappear when heatmap appears"
  severity: major
  test: 7
  root_cause: "h3-js is dynamically imported in flushPoints() (line 1023-1036). First call triggers async import and returns without rendering. When h3-js loads, flushPoints() calls itself directly (line 1029) outside the debounced flush cycle. This standalone pointsData() update may trigger a globe.gl internal re-render that interferes with arcsData. The fix is to call scheduleFlush() instead of flushPoints() after h3-js loads, so all layers flush atomically."
  artifacts:
    - path: "frontend/src/components/GlobeMap.ts"
      issue: "Line 1029: this.flushPoints() called directly from h3-js import callback instead of this.scheduleFlush()"
  missing:
    - "Replace this.flushPoints() with this.scheduleFlush() at line 1029 to batch the heatmap update with all other layer data"
  debug_session: ""

- truth: "Each view (3D and 2D) maintains independent layer visibility state"
  status: failed
  reason: "User reported: fail, changes persist bilaterally"
  severity: major
  test: 10
  root_cause: "LayerPillBar maintains a single `states: Record<LayerId, boolean>` cache (line 37) constructed once from controller.getLayerVisible() in the constructor (lines 43-46). No listener for `globe-mode-changed`, no resync mechanism. MapContainer correctly maintains independent layerState3d/layerState2d and syncs renderers via syncLayerVisibility(), but LayerPillBar's stale cache + DOM become the corruption vector: pill clicks use the old view's state to compute !states[id], then write the wrong value to the new view via setLayerVisible()."
  artifacts:
    - path: "frontend/src/components/LayerPillBar.ts"
      issue: "this.states at line 37 is view-agnostic; no globe-mode-changed listener; no syncFromController() method"
    - path: "frontend/src/components/MapContainer.ts"
      issue: "No defect — per-view state logic correct. But dispatches globe-mode-changed without ensuring LayerPillBar resyncs."
  missing:
    - "Add syncFromController() method to LayerPillBar: re-reads getLayerVisible() for all 5 layers, updates states and pill DOM"
    - "LayerPillBar listens for globe-mode-changed in constructor, calls syncFromController() in handler"
  debug_session: ".planning/debug/layer-state-not-independent.md"
