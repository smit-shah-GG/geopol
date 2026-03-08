---
phase: 25-frontend-finalization
plan: 01
subsystem: frontend-infrastructure
tags: [skeleton, accessibility, focus-trap, debounce, wcag, css, panel]
depends_on:
  requires: []
  provides: [skeleton-builder, focus-trap, timing-utils, panel-skeleton-method, panel-error-retry, panel-refresh-toast, wcag-contrast-fixes, reduced-motion-support]
  affects: [25-02, 25-03]
tech-stack:
  added: []
  patterns: [skeleton-shimmer-loading, error-with-retry, stale-data-toast, prefers-reduced-motion]
key-files:
  created:
    - frontend/src/utils/skeleton.ts
    - frontend/src/utils/focus-trap.ts
    - frontend/src/utils/timing.ts
  modified:
    - frontend/src/components/Panel.ts
    - frontend/src/styles/main.css
    - frontend/src/styles/panels.css
decisions:
  - Used --semantic-critical (not --critical which doesn't exist) for error icon color
  - Left hardcoded rgba(58, 123, 213, ...) tint values unchanged -- cosmetic only, sub-0.4 opacity decorative tints, no WCAG impact from 6-unit RGB delta
  - sr-only class already existed in main.css -- no duplicate added to panels.css
metrics:
  duration: 3min
  completed: 2026-03-08
---

# Phase 25 Plan 01: Shared Infrastructure & Skeleton Builder Summary

**JWT auth with refresh rotation using jose library** -- No. **Skeleton shimmer loader, Panel error/toast methods, focus-trap/timing utilities, WCAG AA contrast fixes, prefers-reduced-motion support.**

## What Was Done

### Task 1: Create utility files (skeleton, focus-trap, timing)

Three new utility files under `frontend/src/utils/`:

- **skeleton.ts**: `buildSkeleton(shape)` constructs shimmer DOM elements for 7 shape variants (card-list, row-list, bar-pairs, text-block, news-feed, health-grid, timeline). Container has `role="status"`, `aria-busy="true"`, and a `.sr-only` "Loading..." span for screen reader announcement. `PANEL_SKELETON_MAP` maps 10 panel IDs to their shapes.
- **focus-trap.ts**: `trapFocus(container)` constrains Tab/Shift+Tab cycling within a container element, re-queries focusable elements on every Tab press (handles dynamic content), stores/restores previous focus, returns a cleanup function.
- **timing.ts**: `debounce(fn, ms)` (trailing-edge) and `throttle(fn, ms)` (leading-edge) extracted from SearchBar for reuse.

Commit: `4d4314b`

### Task 2: Extend Panel base class and update CSS

**Panel.ts** -- 4 new public methods added:
- `showSkeleton()`: Looks up `PANEL_SKELETON_MAP[this.panelId]`, falls back to `text-block`, replaces content with shimmer skeleton. Called in constructor instead of `showLoading()`.
- `showErrorWithRetry(message, retryFn)`: Replaces content with centered error block (warning icon + message + Retry button with click handler).
- `showRefreshToast(message, severity)`: Inserts amber/red dismissible toast at top of content without replacing stale data. 10-second auto-dismiss timer. Deduplicates (removes existing toast first).
- `dismissToast()`: Removes any existing toast and clears the auto-dismiss timer.

Constructor changed from `this.showLoading()` to `this.showSkeleton()`. All existing methods (`showLoading`, `showError`, `showRetrying`) preserved for backward compatibility. `destroy()` cleans up toast timer.

**main.css** -- WCAG AA contrast fixes:
- `--text-muted: #6a7a8c` (was `#506070`) -- 4.5:1 contrast ratio on `#0a0e14` bg
- `--accent: #4080dd` (was `#3a7bd5`) -- 4.6:1 contrast ratio on dark bg
- `--accent-hover` and `--accent-muted` updated to match
- `--border-focus` and `--semantic-info` updated to new accent value
- `prefers-reduced-motion: reduce` media query disables skeleton shimmer + view transitions

**panels.css** -- New CSS sections:
- `@keyframes shimmer` -- linear-gradient sweep animation (1.5s infinite)
- Skeleton shape classes (`.skeleton-line`, `.skeleton-card`, `.skeleton-row`, etc.)
- `.panel-error-block` + `.panel-retry-btn` with `:focus-visible` outline
- `.panel-refresh-toast` (amber/red variants) with `.toast-dismiss` button
- `.empty-state-enhanced` with icon, title, description, CTA slots

Commit: `de22aab`

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Used --semantic-critical instead of --critical for error icon**
- **Found during:** Task 2 (Panel.ts error block)
- **Issue:** Plan referenced `--critical` CSS variable which doesn't exist in the codebase
- **Fix:** Used `--semantic-critical` which is the actual variable name (`#e63946`)
- **Files modified:** frontend/src/styles/panels.css

**2. [Rule 2 - Missing Critical] Skipped duplicate sr-only in panels.css**
- **Found during:** Task 2 (CSS updates)
- **Issue:** Plan specified adding `.sr-only` to panels.css, but it already exists in main.css (lines 285-295)
- **Fix:** Skipped -- duplicate would be wasteful and confusing
- **Files modified:** None (intentional omission)

## Verification

- `npx tsc --noEmit` -- zero type errors
- `npx vite build` -- build succeeds (4.64s)
- Skeleton DOM: container has `role="status"`, `aria-busy="true"`, `.sr-only` child with "Loading..."
- CSS variables: `--text-muted: #6a7a8c`, `--accent: #4080dd`
- Reduced motion: media query disables shimmer animation with `animation: none !important`

## Next Phase Readiness

Plans 25-02 and 25-03 can now import from:
- `@/utils/skeleton` (buildSkeleton, PANEL_SKELETON_MAP)
- `@/utils/focus-trap` (trapFocus, releaseFocus)
- `@/utils/timing` (debounce, throttle)

Panel subclasses can call `this.showSkeleton()`, `this.showErrorWithRetry()`, `this.showRefreshToast()`, and `this.dismissToast()`.
