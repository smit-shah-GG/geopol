---
phase: 12-wm-derived-frontend
plan: 01
subsystem: frontend
tags: [typescript, vite, panel, theme, dom-utils, deck.gl, dto]
status: complete

dependency-graph:
  requires: [09-api-foundation]
  provides: [frontend-scaffold, panel-base-class, api-types, theme-system, dom-utils]
  affects: [12-02, 12-03, 12-04, 12-05, 12-06, 12-07]

tech-stack:
  added: [vite-6, typescript-5.7, deck.gl-9.2.6, maplibre-gl-5.16, d3-7.9]
  patterns: [hyperscript-dom, css-custom-properties, singleton-context, visibility-aware-scheduling]

key-files:
  created:
    - frontend/package.json
    - frontend/vite.config.ts
    - frontend/tsconfig.json
    - frontend/index.html
    - frontend/src/types/api.ts
    - frontend/src/types/index.ts
    - frontend/src/components/Panel.ts
    - frontend/src/utils/dom-utils.ts
    - frontend/src/utils/theme-manager.ts
    - frontend/src/utils/theme-colors.ts
    - frontend/src/utils/sanitize.ts
    - frontend/src/app/app-context.ts
    - frontend/src/app/refresh-scheduler.ts
    - frontend/src/styles/main.css
    - frontend/src/styles/panels.css
    - frontend/src/main.ts
  modified:
    - .gitignore

decisions:
  - id: geopol-theme-key
    description: "localStorage key 'geopol-theme' (not 'worldmonitor-theme')"
    rationale: "Namespace isolation from WM"
  - id: geopol-panel-spans-key
    description: "localStorage key 'geopol-panel-spans' for resize persistence"
    rationale: "Namespace isolation from WM"
  - id: no-i18n-no-tauri
    description: "Removed all t(), isDesktopRuntime, invokeTauri, and analytics from Panel"
    rationale: "Geopol is web-only, English-only; no Tauri desktop runtime"
  - id: focused-appcontext
    description: "GeoPolAppContext has ~8 fields vs WM's 100+"
    rationale: "Geopol needs forecasts/panels/scheduler, not 40 data caches"
  - id: monospace-analyst-aesthetic
    description: "SF Mono primary font, #0a0e14 dark bg, terminal-inspired"
    rationale: "Intelligence analyst dashboard aesthetic per CONTEXT.md"
  - id: snake-case-dto-fields
    description: "TypeScript interfaces use snake_case matching JSON wire format"
    rationale: "Eliminates camelCase<->snake_case transform layer"

metrics:
  duration: 7min
  completed: 2026-03-01
---

# Phase 12 Plan 01: Frontend Scaffold + Build System Summary

**One-liner:** Vite + strict TypeScript scaffold with WM-derived Panel class, h() hyperscript, dark/light theme system, and all Pydantic DTO mirrors.

## What Was Built

The complete `frontend/` directory scaffold that every subsequent plan in Phase 12 depends on. This is the "fork that isn't a fork" -- WM's architectural patterns (Panel lifecycle, h() DOM helper, theme-manager, refresh-scheduler) carried wholesale with all WM-specific content (Tauri, i18n, analytics, 40+ data caches) stripped.

### Task 1: Project Scaffold + Build System + TypeScript Types
- **Vite config**: es2020 target, sourcemaps, manual chunks (deckgl/maplibre/d3), `/api` proxy to FastAPI
- **TypeScript strict mode**: noUncheckedIndexedAccess, bundler module resolution, `@/*` path alias
- **API types**: 11 TypeScript interfaces mirroring all Pydantic DTOs across 4 schema files
- **package.json**: deck.gl ^9.2.6, maplibre-gl ^5.16.0, d3 ^7.9.0 (pinned to WM's versions)
- **.gitignore**: Added node_modules/ and frontend/dist/

### Task 2: Panel Base Class + DOM Utils + Theme + AppContext + main.ts
- **Panel.ts**: Full WM lifecycle (resize handles with touch, loading/error/retrying states, data badges, content debounce, AbortController cleanup, tooltip system) minus Tauri/i18n/analytics
- **dom-utils.ts**: h(), fragment(), replaceChildren(), rawHtml(), safeHtml(), text() -- wholesale from WM
- **theme-manager.ts**: dark/light toggle with localStorage persistence, CSS variable swap, meta theme-color update
- **theme-colors.ts**: getCSSColor() with per-theme cache invalidation
- **sanitize.ts**: escapeHtml() and sanitizeUrl() for XSS prevention
- **app-context.ts**: GeoPolAppContext (container, panels, scheduler, inFlight, destroy)
- **refresh-scheduler.ts**: Visibility-aware polling with 4x background throttle, jitter, in-flight dedup
- **main.css**: ~350 lines of CSS custom properties (dark + light), reset, header, grid, loading animations
- **panels.css**: Panel headers, data badges, resize handles, span heights, info tooltips
- **main.ts**: Boot sequence (applyStoredTheme -> header with toggle -> 6 placeholder panels)

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] .gitignore missing node_modules/**
- **Found during:** Task 1
- **Issue:** `git add` would stage 370 npm packages into the repository
- **Fix:** Added `node_modules/` and `frontend/dist/` to .gitignore
- **Files modified:** .gitignore
- **Commit:** 5b72dcb

## Verification Results

| Check | Result |
|-------|--------|
| `npx tsc --noEmit` | Pass (0 errors) |
| `npx vite build` | Pass (dist/ produced, 6 assets) |
| api.ts mirrors all Pydantic fields | Pass (11 interfaces, snake_case) |
| Panel base class lifecycle | Pass (showLoading/showError/setDataBadge/destroy) |
| Dark theme CSS variables | Pass (--bg, --text-primary, severity colors) |
| Theme toggle persistence | Pass (localStorage geopol-theme) |

## Next Phase Readiness

All 7 remaining plans in Phase 12 can now build on this scaffold:
- **12-02** (Globe panel): imports Panel, deck.gl, maplibre-gl, api types
- **12-03** (Forecast list): imports Panel, h(), ForecastResponse
- **12-04** (Scenario explorer): imports Panel, ScenarioDTO
- **12-05** (Evidence chain): imports Panel, EvidenceDTO
- **12-06** (Country risk): imports Panel, CountryRiskSummary
- **12-07** (Health dashboard): imports Panel, HealthResponse

No blockers. No architectural concerns. The scaffold compiles, builds, and runs.
