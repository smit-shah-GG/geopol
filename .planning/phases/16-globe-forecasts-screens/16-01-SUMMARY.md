---
phase: 16-globe-forecasts-screens
plan: 01
subsystem: frontend-components
tags: [expandable-card, DeckGLMap, refactor, DRY, progressive-disclosure]
depends_on:
  requires: [15-03]
  provides: [shared expandable-card utility, DeckGLMap public API for globe screen]
  affects: [16-02, 16-03]
tech-stack:
  added: []
  patterns: [shared utility extraction, public API facade]
key-files:
  created:
    - frontend/src/components/expandable-card.ts
  modified:
    - frontend/src/components/ForecastPanel.ts
    - frontend/src/components/DeckGLMap.ts
decisions:
  - forecast-selected event dispatches on window (not element) for cross-screen listening
  - DeckGLMap defaults unchanged (all layers true) -- globe screen calls setLayerDefaults() post-construction
metrics:
  duration: 5min
  completed: 2026-03-03
---

# Phase 16 Plan 01: Shared Expandable Card + DeckGLMap Public API Summary

Extracted 300+ lines of progressive disclosure card rendering from ForecastPanel into a shared expandable-card.ts utility, and added flyToCountry/setLayerVisible/setLayerDefaults public API to DeckGLMap while removing the built-in checkbox toggle panel.

## What Was Done

### Task 1: Extract expandable-card.ts + Refactor ForecastPanel

Extracted the following from ForecastPanel.ts into `frontend/src/components/expandable-card.ts`:

**Utility functions (8):** `severityClass`, `isoToFlag`, `relativeTime`, `truncate`, `pctLabel`, `extractCountryIso`, `sourceClass`, `sourceLabel`

**Mini tree types and functions:** `MiniTreeDatum` interface, `buildMiniTreeData` function

**Card building functions (5 exports):**
- `buildExpandableCard(f, opts)` -- collapsed card with click-to-expand wiring
- `buildExpandedContent(f)` -- two-column expanded layout (ensemble + calibration + mini tree + evidence)
- `renderMiniTree(container, f)` -- d3 scenario tree preview (~150px)
- `buildMiniEvidenceCard(ev)` -- compact evidence source badge + description
- `updateCardInPlace(card, f, expandedIds)` -- diff-based DOM update preserving expansion

ForecastPanel now imports and delegates to these shared functions. The d3 import moved to expandable-card.ts. ForecastPanel retains class definition, state management, search event wiring, and diff-based render orchestration.

One behavioral change: `forecast-selected` CustomEvent now dispatches on `window` instead of `this.element`. This enables globe drill-down and forecasts queue screens to listen for ScenarioExplorer open without requiring element-level bubbling.

### Task 2: DeckGLMap Public API + Toggle Panel Removal

**Added public methods:**
- `flyToCountry(iso)` -- maplibre-gl `flyTo` to country centroid, zoom 4.5, 800ms duration
- `setLayerVisible(layerId, visible)` -- toggle individual layer, triggers rebuild
- `setLayerDefaults(defaults)` -- batch-set layer visibility (partial map)
- `getLayerVisible(layerId)` -- query current layer visibility
- `getMap()` -- access underlying maplibre-gl Map instance

**Exported types:** `LayerId` type and `LAYER_IDS` constant for external consumers (LayerPillBar in Plan 02).

**Removed:** `createLayerToggles()` method, `togglePanel` field, constructor call to `createLayerToggles()`, and `togglePanel` cleanup in `destroy()`. The CSS classes (`.map-layer-toggles`, etc.) remain in panels.css -- they're dead code now but harmless; can be cleaned up when LayerPillBar CSS is added.

## Deviations from Plan

None -- plan executed exactly as written.

## Decisions Made

| Decision | Rationale |
|----------|-----------|
| forecast-selected dispatches on window | Cross-screen listening: globe drill-down and forecasts queue need this event without element-level bubbling chains |
| DeckGLMap layer defaults unchanged (all true) | Dashboard screen expects all layers visible by default; globe screen will call setLayerDefaults() post-construction to override |
| Toggle panel CSS not removed from panels.css | Dead CSS is harmless; will be replaced when LayerPillBar styles are added in Plan 02 |

## Verification Results

| Check | Result |
|-------|--------|
| `npx tsc --noEmit` | Pass -- zero errors |
| `npx vite build` | Pass -- 4.16s, DeckGLMap chunk 8.15 kB (down from 8.65 kB) |
| ForecastPanel imports buildExpandableCard | Confirmed |
| ForecastPanel has no d3 import | Confirmed |
| ForecastPanel has no severityClass/renderMiniTree/buildMiniEvidenceCard | Confirmed |
| expandable-card.ts exports all 5 card functions | Confirmed (14 total exports including utilities) |
| DeckGLMap has flyToCountry/setLayerVisible/setLayerDefaults/getLayerVisible/getMap | Confirmed |
| DeckGLMap exports LayerId type and LAYER_IDS const | Confirmed |
| createLayerToggles removed from DeckGLMap | Confirmed |
| togglePanel references removed from DeckGLMap | Confirmed |

## Commits

| Hash | Message |
|------|---------|
| `64acc78` | feat(16-01): extract expandable-card.ts shared utility from ForecastPanel |
| `fe67d3c` | feat(16-01): add DeckGLMap public API and remove built-in toggle panel |

## Next Phase Readiness

Plan 16-02 (Globe Screen + Layer Pill Bar) can proceed immediately:
- `expandable-card.ts` provides `buildExpandableCard()` for globe drill-down panel forecast cards
- `DeckGLMap` exposes `flyToCountry()` for country click camera animation
- `DeckGLMap` exposes `setLayerVisible()`/`setLayerDefaults()`/`getLayerVisible()` for LayerPillBar
- `DeckGLMap` exports `LayerId` type and `LAYER_IDS` for pill bar construction
- No blockers or concerns.
