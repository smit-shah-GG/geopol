---
phase: 12-wm-derived-frontend
plan: 03
subsystem: frontend-map
tags: [deck.gl, maplibre, geojson, choropleth, globe]
depends_on:
  requires: ["12-01", "12-02"]
  provides: ["DeckGLMap component", "country-geometry service", "countries.geojson"]
  affects: ["12-07"]
tech_stack:
  added: []
  patterns: ["MapboxOverlay interleaved rendering", "GeoJsonLayer choropleth", "CustomEvent dispatch for component communication"]
key_files:
  created:
    - frontend/src/components/DeckGLMap.ts
  modified:
    - frontend/src/styles/panels.css
  already_committed_by_parallel:
    - frontend/src/services/country-geometry.ts (committed by 12-05)
    - frontend/public/data/countries.geojson (committed by 12-04)
    - .gitignore (committed by 12-04)
decisions:
  - "DeckGLMap is NOT a Panel subclass -- standalone component for center grid area"
  - "Natural Earth 110m GeoJSON slimmed to 258KB (5 properties vs 150+ in original)"
  - "LABEL_X/LABEL_Y from Natural Earth used as centroid coordinates (more visually accurate than computed centroids)"
  - "Risk color scale: blue [70,130,180] -> gray [128,128,128] -> red [220,50,50] diverging"
  - "5 countries with missing ISO_A2 fixed: Norway->NO, France->FR, N.Cyprus->CY, Somaliland->SO, Kosovo->XK"
  - "Layer toggle panel positioned top-right with checkboxes, not a dropdown"
  - "Tooltip handled via custom DOM element, not deck.gl's built-in HTML tooltip (more control)"
  - "Arcs built from selected country to all other forecast markers + scenario entities"
metrics:
  duration: "9min"
  completed: "2026-03-02"
---

# Phase 12 Plan 03: DeckGLMap Globe Component Summary

**One-liner:** deck.gl globe with 5 analytic layers (risk choropleth, forecast markers, KG arcs, GDELT heatmap, scenario zones) over CARTO dark-matter basemap with layer toggle UI and theme switching.

## What Was Built

### Task 1: Country Geometry Service + GeoJSON Data

**Note:** Both `country-geometry.ts` and `countries.geojson` were already committed by parallel plan executions (12-04 and 12-05 respectively). The files on disk match what this plan specified -- no duplicate commit needed.

**countries.geojson** (258KB, 177 features):
- Source: Natural Earth 110m admin-0 countries
- Slimmed from 839KB by stripping 145+ unused property fields
- Retained: ISO_A2, ISO_A3, NAME, LABEL_X, LABEL_Y
- Fixed 5 countries with missing ISO_A2: Norway (NO), France (FR), N. Cyprus (CY), Somaliland (SO), Kosovo (XK)

**country-geometry.ts** (328 lines):
- Adapted from WM's 303-line service
- Kept: normalizeCode(), GeoJSON loading, centroid calculation, ISO/name lookups
- Removed: Tauri cache, WM country metadata, name-in-text matching
- Added: getCentroid(), getFeatureByIso(), getNameByIso(), isLoaded()
- Singleton: `export const countryGeometry`

### Task 2: DeckGLMap Component (680 lines)

**Initialization:**
- maplibre-gl Map with CARTO dark-matter basemap
- MapboxOverlay in interleaved mode (deck.gl layers composited into maplibre)
- Center: [30, 20], zoom: 1.8, renderWorldCopies: false
- WebGL context loss/restore handlers

**5 Layers:**

| Layer | Type | Data Source | Visibility |
|-------|------|-------------|------------|
| ForecastRiskChoropleth | GeoJsonLayer | countries.geojson + riskScores map | Always (toggleable) |
| ActiveForecastMarkers | ScatterplotLayer | Forecast centroids from countryGeometry | When forecasts loaded |
| KnowledgeGraphArcs | ArcLayer | Built from selected country | When country selected |
| GDELTEventHeatmap | HeatmapLayer | Empty (no GDELT endpoint yet) | When data pushed |
| ScenarioZones | GeoJsonLayer | Filtered countries from scenario entities | When forecast selected |

**Layer Toggle UI:** Floating panel top-right with 5 labeled checkboxes.

**Event Handling:**
- Country click dispatches `country-selected` CustomEvent with `{ iso }` payload
- Marker click dispatches same event
- Hover shows tooltip with country name + risk score (choropleth) or question + probability (markers)

**Theme Integration:**
- Listens for `theme-changed` event
- Swaps basemap: dark-matter <-> positron
- Rebuilds all layers with theme-aware colors

**Public API:**
- `updateRiskScores(summaries)` -- push CountryRiskSummary[] to choropleth
- `updateForecasts(forecasts)` -- push ForecastResponse[] to markers
- `setSelectedCountry(iso)` -- show KG arcs for country
- `setSelectedForecast(forecast)` -- highlight scenario zone countries
- `destroy()` -- cleanup map, overlay, event listeners

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Removed unused BreakerDataMode import in ForecastPanel.ts**
- Found during: Task 1 verification (tsc --noEmit)
- Issue: Pre-existing error from 12-04; `BreakerDataMode` imported but never used
- Fix: Already corrected in HEAD by parallel plan execution (12-05)

**2. [Rule 1 - Bug] Removed dead code in ScenarioExplorer.ts**
- Found during: Task 1 verification (tsc --noEmit)
- Issue: Pre-existing errors from 12-05; `currentForecast`, `selectedNode` fields write-only, `selectedCircle` unused variable, dead `findNodeElement` method
- Fix: Already corrected in HEAD by parallel plan execution (12-05)

**3. [Rule 3 - Blocking] Task 1 deliverables already committed by parallel plans**
- Found during: Task 1 commit
- Issue: country-geometry.ts committed by 12-05 (1ce1336), countries.geojson + .gitignore by 12-04 (4f1988f)
- Fix: Verified file contents match plan spec, skipped duplicate commit
- Impact: Task 1 has no dedicated commit hash; deliverables confirmed present

## Decisions Made

1. **Standalone component, not Panel subclass** -- DeckGLMap occupies the center grid area and doesn't need panel header/resize/badge infrastructure.
2. **Custom tooltip DOM** -- More control over positioning and content than deck.gl's built-in getTooltip HTML return.
3. **Arc generation strategy** -- Selected country connects to all other forecast markers + scenario entities. Future enhancement: use actual KG edges from backend.
4. **Empty HeatmapLayer data** -- No GDELT event endpoint exists yet; layer structure is ready for when data becomes available.

## Verification Results

| Check | Status |
|-------|--------|
| `tsc --noEmit` | PASS |
| `vite build` | PASS |
| DeckGLMap.ts contains `class DeckGLMap` | PASS |
| country-geometry.ts contains `normalizeCode` | PASS |
| countries.geojson exists | PASS |
| 5 layer IDs defined | PASS |
| Layer toggle UI created | PASS |
| Theme-changed listener wired | PASS |
| country-selected event dispatched | PASS |

## Next Phase Readiness

Plan 07 (main.ts wiring) will import DeckGLMap, instantiate it in the center grid area, and wire the public API to ForecastServiceClient data. All public methods are documented and typed.
