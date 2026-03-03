---
phase: quick
plan: 002
subsystem: frontend
tags: [expandable-card, status-merge, progressive-disclosure, queued-status]
tech-stack:
  patterns: [expandable-card reuse, forecast caching, status consolidation]
key-files:
  modified:
    - frontend/src/components/MyForecastsPanel.ts
    - frontend/src/components/SubmissionQueue.ts
    - frontend/src/styles/panels.css
decisions:
  - id: QUEUED-MERGE
    title: Merge pending+confirmed into QUEUED
    rationale: Users don't need to distinguish between pending and confirmed -- both mean "waiting in queue"
  - id: MF-EXPANDABLE
    title: Reuse expandable-card pattern in MyForecastsPanel
    rationale: Identical progressive disclosure across all three screens (dashboard, globe, forecasts)
metrics:
  completed: 2026-03-03
---

# Quick 002 Task 3: MyForecastsPanel Expandable Cards + QUEUED Status Merge

**One-liner:** Completed forecasts in MyForecastsPanel now use shared expandable-card pattern with cached forecast data; pending/confirmed merged to QUEUED across both components.

## Task Summary

### Task 3: MyForecastsPanel expandable cards + QUEUED status merge (8dc0dd1)

**A. Expandable Cards in MyForecastsPanel:**
- Replaced click-to-open-ScenarioExplorer pattern with shared `buildExpandableCard()` progressive disclosure
- Added `forecastCache` (Map<string, ForecastResponse>) to avoid re-fetching on refresh
- Added `expandedIds` (Set<string>) to preserve expansion state across re-renders
- Complete row now shows COMPLETE badge + completion time header, then expandable forecast card
- Loading placeholder (question text) shown while forecast fetches in background
- Removed `openCompletedForecast()` method -- "View Full Analysis" button in expanded content handles `forecast-selected` event dispatch
- Removed local `relativeTime()` and `truncate()` -- imports from shared `expandable-card.ts`
- Override `destroy()` clears cache and expanded state

**B. QUEUED Status Merge:**
- MyForecastsPanel: `pending` and `confirmed` both map to `status-queued` CSS class and `QUEUED` label
- SubmissionQueue: `pending` and `confirmed` both map to `sq-status-queued` CSS class and `QUEUED` label
- Added `.mf-status-badge.status-queued` and `.sq-status-queued` CSS rules (neutral muted style)
- Removed dead `.status-pending`, `.status-confirmed`, `.sq-status-pending`, `.sq-status-confirmed` CSS rules

## Deviations from Plan

None -- plan executed exactly as written.

## Verification Results

- TypeScript compiles cleanly (zero errors from Task 3 files; pre-existing Task 2 dashboard-screen error unrelated)
- No old status class references (`status-pending`, `status-confirmed`, `sq-status-pending`, `sq-status-confirmed`) in any TS file
- Both components show QUEUED label
- `buildExpandableCard` imported and used in MyForecastsPanel
- `openCompletedForecast` removed
- Local `relativeTime`/`truncate` removed (using shared imports)
