---
phase: 12
plan: 06
subsystem: frontend-components
tags: [typescript, d3-force, svg, modal, tabs, country-brief, cameo, calibration]
depends_on:
  requires: [12-01, 12-02, 12-05]
  provides: [CountryBriefPage full-screen modal with 7 tabs]
  affects: [12-07]
tech-stack:
  added: []
  patterns: [full-screen tabbed modal, d3-force entity graph, SVG reliability diagram, CAMEO category breakdown]
key-files:
  created:
    - frontend/src/components/CountryBriefPage.ts
  modified:
    - frontend/src/styles/panels.css
decisions:
  - Entity graph runs d3.forceSimulation synchronously (200 ticks) for layout stability without animation flicker
  - CAMEO frequency extraction walks evidence_sources recursively looking for 2-digit codes and CAMEO references
  - entityViewMode as class field enables toggle between graph/table without full tab re-render teardown
metrics:
  duration: 6min
  completed: 2026-03-02
---

# Phase 12 Plan 06: Country Brief Page Summary

Full-screen tabbed modal (CountryBriefPage) with 7 tabs for country-level drill-down. 1231 lines of TypeScript, 615 lines of new CSS. Opens via 'country-selected' CustomEvent from DeckGLMap globe clicks.

## What Was Built

### CountryBriefPage (1231 lines)

**Modal structure:**
- Full-screen overlay with dark backdrop (reuses ScenarioExplorer pattern)
- Header: flag emoji + country name + ISO code + risk score badge + close button
- Horizontal tab bar with 7 tabs, active state via `tab-active` class
- Close on: Escape key, backdrop click, X button
- Listens for `country-selected` CustomEvent on window

**Data loading:**
- `forecastClient.getForecastsByCountry(iso)` + `forecastClient.getCountryRisk(iso)` in parallel
- Stale response guard (checks `currentIso` matches after await)
- Loading/empty states for each tab

**Tab 1 - Overview:**
- 2x2 CSS grid of summary cards
- Risk Score card: large number with severity class + trend arrow
- Top Forecast card: question text + probability bar + confidence
- Recent Events card: placeholder for GDELT event feed
- Calibration Snapshot card: Brier score + historical accuracy

**Tab 2 - Active Forecasts:**
- Expanded forecast detail cards with full question, probability bar, ensemble breakdown (LLM/TKG), dates
- Click dispatches `forecast-selected` CustomEvent (opens ScenarioExplorer)

**Tab 3 - GDELT Events:**
- Notice banner: "GDELT event data will be available when the ingest daemon is running"
- 6 mock event rows demonstrating layout: timestamp, actor1 -> action -> actor2, CAMEO badge, Goldstein value
- Ready for real data when events endpoint ships

**Tab 4 - Risk Signals:**
- Full CAMEO root category table (codes 01-20)
- Columns: code, category name, frequency, Goldstein bar + value, trend indicator
- Sorted by frequency descending (non-zero first)
- Goldstein bar with color coding (green for cooperation, red for conflict)
- Legend: "Goldstein scale: -10 (conflict) to +10 (cooperation)"

**Tab 5 - Forecast History:**
- SVG line chart (600x250): probability points plotted by creation date
- Polyline connecting points when multiple forecasts exist
- Y-axis: probability 0-1, X-axis: forecast creation dates
- Below chart: scrollable list of all forecasts (same detail card format)

**Tab 6 - Entity Relations:**
- Toggle between Graph View and Table View
- Graph: d3.forceSimulation with forceLink, forceManyBody, forceCenter, forceCollide
  - Nodes sized by entity frequency, labels below
  - Links width proportional to co-occurrence weight
  - 200 synchronous ticks for stable layout
- Table: entity pairs sorted by frequency, columns: Entity, Co-occurs With, Frequency, Avg Probability

**Tab 7 - Calibration:**
- Reliability diagram SVG (300x300): predicted vs observed with perfect calibration diagonal
- Brier decomposition table: category, Brier score (color-coded), sample size, plus overall summary row
- Per-CAMEO category accuracy table: category, accuracy percentage, sample size

### CSS (panels.css, +615 lines)

All styles for the modal, header, tab bar, tab content, and each tab's specific components. Responsive breakpoints for mobile (single-column grid, smaller tabs). Light theme overrides.

## Deviations from Plan

None -- plan executed exactly as written.

## Next Phase Readiness

CountryBriefPage is a standalone component. It needs to be:
1. Instantiated in main.ts (Plan 12-07)
2. Wired to receive `country-selected` events from DeckGLMap (already listens on window)

No blockers for Plan 12-07.
