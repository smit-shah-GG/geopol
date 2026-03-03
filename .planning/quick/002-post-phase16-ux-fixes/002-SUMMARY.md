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

# Quick 002 Task 2: Frontend Polymarket Top-10 + EnsembleBreakdownPanel Removal + Dashboard Cleanup

**One-liner:** CalibrationPanel now renders top-10 Polymarket geo events by volume with optional Geopol match; EnsembleBreakdownPanel deleted entirely (file, imports, DOM, CSS, ctx registration).

## Task Summary

### Task 2: Frontend Polymarket top-10 + EnsembleBreakdownPanel removal + dashboard cleanup (04de6b2)

**A. New Types (api.ts):**
- Added `PolymarketTopEvent` interface (event_id, title, slug, volume, liquidity, optional geopol match fields)
- Added `PolymarketTopResponse` interface (events array + total_geo_markets count)

**B. New Client Method (forecast-client.ts):**
- Added `getPolymarketTop()` method with dedup + circuit breaker wrapping
- Added `FALLBACK_POLYMARKET_TOP` constant for breaker fallback
- Imported `PolymarketTopResponse` type

**C. CalibrationPanel Rewrite:**
- Replaced `updatePolymarket(data: PolymarketComparisonResponse)` with `updatePolymarketTop(data: PolymarketTopResponse)`
- Replaced `buildPolymarketTable(comparisons: PolymarketComparison[])` with `buildPolymarketTable(events: PolymarketTopEvent[])`
- New table columns: QUESTION (linked to polymarket.com, truncated 60 chars), VOL (K/M suffix), GEOPOL (probability or "--"), MATCH (confidence or empty)
- Added `formatVolume()` helper (K/M suffix formatting), replaced `formatDelta()` which is no longer needed
- Matched probabilities highlighted with `.polymarket-matched` CSS class
- Footer shows "Showing top N of M geopolitical markets"
- Preserved `this.polymarketContainer` in `showPlaceholder()` (existing bug fix preserved)

**D. Dashboard Cleanup (dashboard-screen.ts):**
- Removed `EnsembleBreakdownPanel` import, instantiation, DOM append, ctx registration
- Removed `ensemblePanel.update(forecast)` from `forecastSelectedHandler`
- Updated `loadInitial()`: `getPolymarket()` -> `getPolymarketTop()`, `updatePolymarket()` -> `updatePolymarketTop()`
- Updated scheduler polymarket registration: same method renames
- Updated file header comment (removed EnsembleBreakdown from Col 4 list)

**E. File Deletion:**
- Deleted `frontend/src/components/EnsembleBreakdownPanel.ts` entirely

**F. CSS Cleanup (panels.css):**
- Removed entire Ensemble Breakdown CSS section (10 rules: `.ensemble-section`, `.ensemble-label`, `.ensemble-bar`, `.ensemble-segment`, `.ensemble-values`, `.ensemble-value`, `.ensemble-dot`, `.ensemble-pct`, `.ensemble-weights-text`, `.ensemble-temp`)
- Added Polymarket table CSS section (`.polymarket-section`, `.polymarket-section-inner`, `.polymarket-table`, `.polymarket-row`, `.polymarket-header`, `.polymarket-cell`, `.polymarket-cell-question`, `.polymarket-cell-vol`, `.polymarket-cell-prob`, `.polymarket-cell-conf`, `.polymarket-matched`, `.polymarket-link`, `.polymarket-footer`, `.polymarket-seeking`)

## Deviations from Plan

**[Rule 2 - Missing Critical] Added Polymarket table CSS**
- The old CalibrationPanel used polymarket CSS classes that had no corresponding CSS rules anywhere in the codebase
- Added complete polymarket table CSS section to panels.css (row layout, cell sizing, link hover, matched highlight, footer)
- Without these rules the table renders as unstyled inline elements

## Verification Results

- TypeScript compiles cleanly (zero errors)
- `EnsembleBreakdownPanel.ts` deleted (confirmed not in filesystem)
- No dead references to `EnsembleBreakdownPanel`, `ensemble-section`, `ensemble-bar`, `ensemble-segment`, or `ensemble-label` in any TS/CSS file (only `expanded-ensemble-*` in expandable-card.ts which is correct)

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
