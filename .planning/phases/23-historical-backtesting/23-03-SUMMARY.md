---
phase: 23
plan: 03
subsystem: backtesting-admin-panel
tags: [backtesting, admin-panel, d3-charts, brier-score, calibration, polymarket]
dependency-graph:
  requires: [phase-23-plan-01-engine, phase-23-plan-02-api]
  provides: [backtesting-admin-ui, brier-curve-charts, calibration-diagrams, backtest-export]
  affects: []
tech-stack:
  added: []
  patterns: [expandable-chart-sections, d3-svg-responsive-viewbox, scoped-css-injection, blob-download]
key-files:
  created:
    - frontend/src/admin/panels/BacktestingPanel.ts
  modified:
    - frontend/src/admin/admin-types.ts
    - frontend/src/admin/admin-client.ts
    - frontend/src/admin/admin-layout.ts
    - frontend/src/admin/components/AdminSidebar.ts
decisions:
  - "Expandable chart sections: click-to-reveal d3 charts (lazy render on first expand)"
  - "Scoped CSS injection via style element with #bt-panel-styles guard"
  - "10s auto-refresh (faster than AccuracyPanel 30s) for running backtest progress"
  - "Area fill color on PM comparison chart based on aggregate winner (green/red)"
metrics:
  duration: 5min
  completed: 2026-03-08
---

# Phase 23 Plan 03: Backtesting Admin Panel Summary

**BacktestingPanel with run list, d3 Brier/calibration/hit-rate/PM-comparison charts, start/cancel/export controls, wired into admin sidebar**

## What Was Done

### Task 1: TypeScript types, AdminClient extension, sidebar + layout registration

- Added `BacktestRun`, `BacktestResult`, `BacktestRunDetail`, `CheckpointInfo`, `StartBacktestRequest` interfaces to `admin-types.ts`
- Extended `AdminSection` union type to include `'backtesting'`
- Added 6 methods to `AdminClient`: `getBacktestRuns`, `startBacktestRun`, `getBacktestRun`, `cancelBacktestRun`, `exportBacktestRun` (raw fetch for Blob), `getCheckpoints`
- Added Backtesting nav item to `AdminSidebar` NAV_ITEMS array (stopwatch unicode icon)
- Added `backtesting` case to `createPanel` switch in `admin-layout.ts` and `SECTION_TITLES` record

### Task 2: BacktestingPanel -- run list, drill-down, d3 charts, export, start/cancel

Created 1423-line panel implementing `AdminPanel` interface with two views:

**View A: Run List**
- Toolbar with "Start New Run" button toggling inline form
- Inline start form: label input, description textarea, checkpoint checkboxes grouped by model type, window parameter inputs (size/slide/min), API call warning, validation
- Run table: label, status badge (color-coded), checkpoints, windows (with progress bar for running), Brier, MRR, created, duration, cancel button
- Empty state message for no runs

**View B: Drill-Down**
- Back navigation button
- Run metadata header: label, status badge, description, window config, duration, error message
- 4 summary stat cards: Brier Score, Calibration (mean absolute cal error), Hit Rate, vs Polymarket record
- 4 expandable chart sections (lazy d3 render on first click):
  1. **Brier Score Curves**: d3.scaleTime x d3.scaleLinear, multi-checkpoint color-coded lines, 0.25 random baseline dashed reference, dot tooltips with window details
  2. **Calibration Reliability Diagram**: predicted vs observed with perfect-calibration diagonal, faint count bar chart behind curve
  3. **Hit Rate by Checkpoint**: d3.scaleBand grouped bars with percentage labels (or simple stat display for single checkpoint)
  4. **Geopol vs Polymarket Brier**: dual-line time chart with area fill (green when Geopol outperforms, red when underperforms), win/loss/draw summary
- Export buttons: CSV and JSON via `URL.createObjectURL(blob)` + programmatic anchor click
- Cancel button for running/pending runs

## Deviations from Plan

None -- plan executed exactly as written.

## Verification Results

- TypeScript compilation: 0 errors (clean)
- Vite build: succeeds in 4.29s
- NAV_ITEMS: 6 entries (was 5)
- AdminSection includes 'backtesting'
- AdminClient has all 6 backtest methods
- createPanel handles 'backtesting' case
- BacktestingPanel implements AdminPanel with mount/destroy lifecycle
- d3 imported as namespace import (`import * as d3 from 'd3'`)
- File size: 1423 lines (exceeds 300 minimum)

## Commits

| Hash | Message |
|------|---------|
| 6a18311 | feat(23-03): add backtesting types, client methods, sidebar + layout registration |
| d76c11f | feat(23-03): implement BacktestingPanel with run list, drill-down, d3 charts, export |
