---
phase: 12-wm-derived-frontend
plan: 05
subsystem: frontend-panels
tags: [d3, hierarchy, scenario-tree, modal, evidence, explainability]
depends_on: [12-01, 12-02]
provides: [ScenarioExplorer, FE-04-scenario-exploration]
affects: [12-07]
tech-stack:
  added: []
  patterns: [d3-hierarchy-tree, custom-event-modal, svg-node-selection]
key-files:
  created:
    - frontend/src/components/ScenarioExplorer.ts
  modified:
    - frontend/src/styles/panels.css
decisions:
  - Node highlight uses coordinate-matching on SVG transform attribute (avoids d3 selection dependency)
  - Pruned indicator shows "+N deeper" text below node when exceeding MAX_DEPTH=4
  - Evidence sidebar scrolls independently; tree container scrolls independently
  - Source badges classified by substring match (gdelt/tkg/rag) for flexibility
metrics:
  duration: 3min
  completed: 2026-03-02
---

# Phase 12 Plan 05: ScenarioExplorer Summary

**One-liner:** Full-screen d3-hierarchy scenario tree modal with evidence sidebar, probability-sized nodes, and source provenance cards.

## What Was Built

ScenarioExplorer (FE-04): a standalone modal component that opens when a forecast card dispatches a `forecast-selected` CustomEvent. Visualizes the forecast's scenario tree as a vertical top-down SVG using `d3.hierarchy()` + `d3.tree()`, with an evidence sidebar showing source cards per selected node.

### Key Components

1. **Modal shell** -- Full-screen backdrop (rgba(0,0,0,0.85)), fixed-inset modal with flex-row layout (70% tree, 30% sidebar), header with question text + probability badge + close button. Closes on Escape, backdrop click, or X.

2. **d3 tree rendering** -- Converts `ForecastResponse.scenarios` (recursive `ScenarioDTO.child_scenarios`) to d3 hierarchy data. `d3.tree().nodeSize([200, 100])` for vertical layout. SVG with computed viewBox from node bounds. Curved links via `d3.linkVertical()`. Max 4 depth levels with pruned-count indicator.

3. **Node visualization** -- Circles sized by probability (`8 + p*20`, capped at 28px radius). Affirmative nodes: semantic-critical (red). Negative nodes: accent (blue). Opacity 0.8. Percentage label above, truncated description (40 chars) below. Hover glow via SVG filter.

4. **Evidence sidebar** -- Default placeholder text. On node click: full description, probability, entity badges, ordered timeline, evidence source cards. Each card shows source type badge (GDELT/TKG/RAG with distinct colors), description, confidence bar, optional timestamp, optional GDELT Event ID in monospace.

5. **CSS** -- 200+ lines added to panels.css: modal positioning, sidebar layout, source badges, entity badges, confidence bars, evidence cards, node hover glow, responsive breakpoint (mobile stacks vertically), light theme overrides.

## Verification

- `npx tsc --noEmit` passes clean
- ScenarioExplorer exports `open()`, `close()`, `destroy()` public methods
- d3 imports compile (`d3.hierarchy`, `d3.tree`, `d3.linkVertical`)
- 491 lines (min_lines: 200 requirement met)
- `class ScenarioExplorer` present in export
- `forecast-selected` event listener wired in constructor

## Decisions Made

| # | Decision | Rationale |
|---|----------|-----------|
| 1 | Node selection by transform coordinate matching | Avoids maintaining node-to-element map; SVG nodes already have unique x,y positions from d3 layout |
| 2 | MAX_DEPTH=4 with pruned indicator | Prevents unbounded tree depth from consuming viewport; shows count of hidden descendants |
| 3 | Source badge classification by substring match | Flexible against source string variations (e.g., "GDELT v2", "tkg_pattern", "RAG-enriched") |
| 4 | Independent scroll for tree and sidebar | Tree can be large; evidence sidebar always accessible without scrolling tree back to viewport |

## Deviations from Plan

### Minor Issue

**1. [Accidental inclusion] country-geometry.ts staged by parallel plan**
- **Found during:** Commit staging
- **Issue:** `frontend/src/services/country-geometry.ts` from 12-04 (GlobeRiskPanel) was pre-staged in git index by a parallel execution
- **Impact:** None -- file compiles, doesn't affect this plan's functionality
- **Commit:** 1ce1336

## Commits

| Hash | Message |
|------|---------|
| 1ce1336 | feat(12-05): ScenarioExplorer modal with d3 tree + evidence sidebar |

## Next Phase Readiness

- ScenarioExplorer is ready for integration in 12-07 (app shell wiring)
- ForecastPanel (12-03) can dispatch `forecast-selected` CustomEvent to trigger this modal
- No blockers
