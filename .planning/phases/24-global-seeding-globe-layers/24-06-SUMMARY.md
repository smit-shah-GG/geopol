---
phase: 24-global-seeding-globe-layers
plan: 06
subsystem: frontend-globe
tags: [deck.gl, h3, hexagon, arcs, globe, visualization, heatmap]
depends_on:
  requires: ["24-04", "24-05"]
  provides: ["frontend-globe-layer-wiring", "h3-hexagon-rendering", "bilateral-arc-visualization", "risk-delta-overlay"]
  affects: []
tech-stack:
  added: ["@deck.gl/geo-layers", "h3-js (explicit)"]
  patterns: ["dual-mode-layer (global vs selected)", "API-to-datum type conversion", "parallel layer data loading"]
key-files:
  created: []
  modified:
    - frontend/src/components/DeckGLMap.ts
    - frontend/src/screens/globe-screen.ts
    - frontend/src/services/forecast-client.ts
    - frontend/src/types/api.ts
    - frontend/src/components/RiskIndexPanel.ts
    - frontend/package.json
decisions:
  - id: dual-mode-arcs
    description: "Layer 3 (KnowledgeGraphArcs) operates in two modes: country-selected shows scenario arcs (existing), no selection shows global bilateral arcs with sentiment coloring"
  - id: dual-mode-scenarios
    description: "Layer 5 (ScenarioZones) operates in two modes: forecast-selected shows scenario entity highlights (existing), no forecast shows risk delta regions (red=worsening, green=improving)"
  - id: layer-type-to-layer
    description: "Replaced HeatmapLayer (aggregation-layers) with H3HexagonLayer (geo-layers) for discrete hexagonal rendering of pre-computed server-side H3 bins"
  - id: all-layers-default-on
    description: "Globe screen enables all 5 layers by default (was: only choropleth + markers). Data now exists for all layers."
  - id: layer-refresh-interval
    description: "Layer data (heatmap, arcs, deltas) refreshes every 5 minutes on the globe screen. Server computes hourly."
metrics:
  duration: 6min
  completed: 2026-03-08
---

# Phase 24 Plan 06: Frontend Globe Layer Wiring Summary

**One-liner:** H3HexagonLayer replaces HeatmapLayer, bilateral arcs with Goldstein sentiment coloring, risk delta overlay, 5-min refresh cycle for all globe layer data.

## What Was Done

### Task 1: Frontend deps + TypeScript types + forecast client
- Installed `@deck.gl/geo-layers` and `h3-js` as explicit package.json dependencies (both were already transitive deps of `deck.gl`, now explicitly tracked)
- Updated `CountryRiskSummary` interface with dual-score fields: `baseline_risk`, `forecast_risk` (nullable), `blended_risk`, `disputed`. Made `top_forecast` and `top_probability` nullable for baseline-only countries.
- Added 3 new TypeScript interfaces: `HexbinData`, `ArcData`, `RiskDeltaData`
- Added 3 new forecast client methods: `getHeatmapData()`, `getArcData()`, `getRiskDeltas()` -- all use the `eventsBreaker` circuit breaker with empty-array fallbacks
- Fixed null-safety bug in `RiskIndexPanel.buildRow()` where `c.top_forecast.length` would crash for countries with no forecasts (now nullable)

### Task 2: DeckGLMap + globe-screen layer data loading
- **Layer 3 (KnowledgeGraphArcs):** Dual-mode rendering. When a country is selected, shows per-country scenario arcs (unchanged). When no country selected, renders global bilateral arcs from server data with sentiment-based coloring (red = negative Goldstein / conflictual, blue = positive Goldstein / cooperative) and event-count-proportional width (clamped 1-5px).
- **Layer 4 (GDELTEventHeatmap):** Replaced `HeatmapLayer` from `@deck.gl/aggregation-layers` with `H3HexagonLayer` from `@deck.gl/geo-layers`. Renders pre-computed H3 hexagonal bins with yellow-to-red weight-based fill color and 0.9 coverage.
- **Layer 5 (ScenarioZones):** Dual-mode rendering. When a forecast is selected, shows accent-colored scenario entity highlights (unchanged). When no forecast selected, renders risk delta regions: red fill for worsening countries, green fill for improving countries, alpha proportional to |delta|/50.
- **Tooltips:** Added tooltip content for H3 hexagons (event count + weight), bilateral arcs (countries, sentiment, event count), and risk delta regions (country name, direction, magnitude).
- **Globe screen data loading:** `loadInitialData()` now loads heatmap, arcs, and deltas via `Promise.all` alongside countries and forecasts. Three converter functions (`pushHeatmap`, `pushArcs`, `pushDeltas`) map API types to DeckGLMap datum types, resolving country centroids for arcs.
- **Refresh scheduling:** New `globe-layers` job at 300s (5 minutes) interval refreshes all three layer data sources.
- **Layer defaults:** All 5 layers enabled by default on globe screen (was: only choropleth + markers).
- **Build type system:** Changed `buildLayers()` return type from union of specific layer classes to `Layer[]` (deck.gl base class) to accommodate the new H3HexagonLayer without maintaining a growing union type.

## Decisions Made

| # | Decision | Rationale |
|---|----------|-----------|
| 1 | Dual-mode arcs (global bilateral vs per-country scenario) | Provides global context when browsing, focused context when drilling into a country |
| 2 | Dual-mode scenarios (risk deltas vs forecast entity highlights) | Repurposes existing layer slot for "what changed" overlay without adding a 6th layer |
| 3 | H3HexagonLayer instead of HeatmapLayer | Server pre-computes H3 bins; discrete hexagons are more analytically useful than continuous heatmap |
| 4 | All layers default ON | All layers now have real data sources; hiding them by default defeats the purpose |
| 5 | 5-minute refresh for layers | Layer data is computed hourly server-side; polling faster than 5min wastes bandwidth |

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed null-safety in RiskIndexPanel.buildRow()**
- **Found during:** Task 1
- **Issue:** `c.top_forecast.length` crashes when `top_forecast` is `null` (now possible for baseline-only countries without active forecasts)
- **Fix:** Added null coalescing: `const raw = c.top_forecast ?? '';`
- **Files modified:** `frontend/src/components/RiskIndexPanel.ts`
- **Commit:** 841d6cf

**2. [Rule 3 - Blocking] Changed buildLayers() return type to Layer[]**
- **Found during:** Task 2
- **Issue:** Adding `H3HexagonLayer` (a `CompositeLayer` subclass) to the union type `(GeoJsonLayer | ScatterplotLayer | ArcLayer | HeatmapLayer)[]` would require importing and listing every possible layer class
- **Fix:** Changed to `Layer[]` (base class from `@deck.gl/core`), which is the intended polymorphic container
- **Files modified:** `frontend/src/components/DeckGLMap.ts`
- **Commit:** 8167e5e

## Verification

- TypeScript compilation: `cd frontend && npx tsc --noEmit` passes cleanly
- H3HexagonLayer present in DeckGLMap (3 occurrences: header comment, import, usage)
- HeatmapLayer removed completely (0 occurrences)
- 3 public methods exported: updateHeatmapData, updateArcData, updateRiskDeltas
- CountryRiskSummary has baseline_risk, forecast_risk (nullable), blended_risk, disputed
- forecast-client has getHeatmapData, getArcData, getRiskDeltas
- globe-screen loads layer data on mount and registers 5-min refresh
- All 5 layers enabled by default on globe screen

## Next Phase Readiness

Phase 24 is now complete (all 6 plans executed). The globe renders all ~195 countries with risk-based coloring, H3 hexagonal heatmap cells, bilateral arcs with sentiment coloring, and risk delta regions. Phase 25 (final v3.0 phase) can proceed.
