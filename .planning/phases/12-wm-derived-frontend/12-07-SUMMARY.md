---
phase: 12
plan: 07
subsystem: frontend-integration
tags: [typescript, main.ts, panel-layout, grid, wiring, integration, bootstrap]
depends_on:
  requires: [12-01, 12-02, 12-03, 12-04, 12-05, 12-06]
  provides: [Complete integrated Geopol dashboard frontend]
  affects: [Phase 13 deployment]
tech-stack:
  added: []
  patterns: [app bootstrap, panel grid layout manager, event-driven data flow, circuit breaker integration]
key-files:
  created:
    - frontend/src/app/panel-layout.ts
  modified:
    - frontend/src/main.ts
    - frontend/src/styles/main.css
    - frontend/src/styles/panels.css
decisions:
  - Panel grid layout uses CSS Grid with named areas for stable positioning without flexbox complexity
  - Initial data load via update(data) to push to multiple consumers; RefreshScheduler uses refresh() for independent periodic fetches
  - Event wiring via CustomEvent dispatch (country-selected, forecast-selected, theme-changed) with window-level listeners
  - All panels mount via appendChild(el) to createPanelLayout slots after creation
metrics:
  duration: 3min
  completed: 2026-03-02
---

# Phase 12 Plan 07: Integration Wiring Summary

Complete dashboard bootstrap and wiring. All 6 panels, globe, 2 modals, refresh scheduler, and theme toggle connected and functional. Human verification APPROVED.

## What Was Built

### panel-layout.ts (97 lines)

**Purpose:** Creates the fixed 3-column grid container and returns HTMLElement slots for each panel region.

**Implementation:**
- `createPanelLayout(container: HTMLElement): Record<string, HTMLElement>`
- CSS Grid with named areas:
  ```
  "forecasts    map         risk-index"
  "forecasts    map         risk-index"
  "ensemble     map         health"
  "calibration  events      health"
  ```
- Grid proportions: left 25%, center 50%, right 25%
- Returns: `{ forecasts, map, riskIndex, ensemble, calibration, events, health }`
- Each slot is a div with id matching the key

### main.ts (210 lines)

**Complete rewrite from stub.** Full bootstrap sequence:

1. **Imports:** All styles, components, services, utilities
2. **Theme initialization:** Call applyStoredTheme() (FOUC prevention)
3. **DOM Ready:** Wait for DOMContentLoaded
4. **Container setup:** Get #app, create panel layout via createPanelLayout()
5. **AppContext:** Create global context with panels, map, forecasts, countries
6. **Service initialization:** Load countryGeometry (GeoJSON for DeckGLMap)
7. **Panel instantiation:** Create all 6 panels, mount to slots
8. **DeckGLMap:** Create in center slot with container ref
9. **Modal creation:** Create ScenarioExplorer and CountryBriefPage (append to body)
10. **Initial data load:**
    - Fetch forecasts, countries, health in parallel
    - Push to panels via `update(data)` (no panel self-fetch, main.ts distributes)
    - Update map risk scores and forecast markers
11. **Event wiring:**
    - 'forecast-selected' → update EnsembleBreakdownPanel + CalibrationPanel + open ScenarioExplorer
    - 'country-selected' → open CountryBriefPage (listens on window)
    - 'theme-changed' → rebuild DeckGLMap layers
12. **RefreshScheduler:** Register periodic tasks (forecasts 60s, countries 120s, health 30s)
13. **Theme toggle:** Add button to header, call setTheme() on click
14. **Ready:** Log "Geopol dashboard ready"

**Key distinction:**
- `update(data)`: One-time push from main.ts to multiple consumers (initial load, event-driven)
- `refresh()`: Each panel self-fetches on timer (scheduler callbacks)

### CSS Updates (main.css + panels.css)

**main.css:**
- Grid container: grid-display, grid-template-areas
- Header bar: thin top bar with "GEOPOL" logo + theme toggle button
- Responsive breakpoints: <1200px stacks panels vertically
- Panel spacing and shadows

**panels.css:**
- Grid item sizing (25%, 50%, 25% columns)
- Panel borders and background colors
- Modal z-index layering
- maplibre-gl CSS import for globe rendering

## Verification PASSED

Human checkpoint approved after verification of:
1. Dashboard loads with globe in center
2. All 6 panels arrange in 3-column grid around globe
3. Countries colored by risk score (blue to red)
4. Forecast cards clickable → ScenarioExplorer modal opens
5. Country clicks on globe → Country Brief modal opens
6. Theme toggle switches dark/light across all components
7. Layer control on globe works
8. No TypeScript errors; `npx tsc --noEmit` and `npx vite build` pass
9. API failure resilience: "unavailable" badge shown instead of crash

## Deviations from Plan

None -- plan executed exactly as written. All wiring completed successfully.

## Technical Decisions

1. **Grid layout:** Named areas over flexbox for explicit positioning and resize support readiness
2. **Event dispatch:** CustomEvent on window for decoupling (country-selected from DeckGLMap, forecast-selected from panels)
3. **Update vs Refresh:** Clear protocol to avoid redundant API calls on initial load
4. **Panel lifecycle:** Create all, then mount to DOM, then update with data (not mount→update sequence)

## Next Phase Readiness

Phase 12 COMPLETE. All 6 component plans (01-06) + integration plan (07) delivered.

Frontend fully wired and ready for Phase 13 (deployment + API integration):
- Dashboard boots successfully
- All event flows connected
- Theme persistence via localStorage working
- Circuit breaker resilience functional
- Ready for Phase 13 backend routing and deployment
