---
status: complete
phase: 26-operational-fixes-ux-polish
source: 26-01-SUMMARY.md, 26-02-SUMMARY.md, 26-03-SUMMARY.md
started: 2026-03-09T19:15:00Z
updated: 2026-03-09T19:35:00Z
---

## Current Test
<!-- OVERWRITE each test - shows where we are -->

[testing complete]

## Tests

### 1. Same-route refresh shows skeletons
expected: On /dashboard, click the "Dashboard" nav link again. All panels should briefly show skeleton shimmer loading states, then re-render with fresh data from the API. No stale data should persist from the previous view.
result: pass

### 2. Cross-route refresh shows fresh data
expected: Navigate /dashboard → /globe → /dashboard. The dashboard should show skeleton loading on return and fetch fresh data — not display stale cached panels from the first visit.
result: pass

### 3. Globe resets on navigation
expected: Navigate to /globe. Pan and zoom the globe to a non-default view. Navigate to /dashboard, then back to /globe. The globe should be at its default position and zoom level (not where you left it).
result: pass

### 4. Forecasts form draft preserved
expected: Navigate to /forecasts. Type some text in the submission textarea (don't submit). Click the "Forecasts" nav link again (same-route refresh). The text you typed should still be in the textarea after the page refreshes.
result: pass

### 5. ComparisonPanel entries are expandable
expected: On /dashboard, find a Polymarket comparison entry in the ComparisonPanel (Col 2, below Active Forecasts). Click it. The entry should expand to reveal forecast details: ensemble weights, calibration metadata, evidence summaries — plus Polymarket-specific data (market price, divergence in pp). Click again to collapse.
result: issue
reported: "failed to load forecast — but no 500 or anything other than 200 in the server logs"
severity: major

### 6. ComparisonPanel keyboard accessibility
expected: Use Tab to focus a ComparisonPanel entry header. Press Enter or Space. The entry should expand/collapse just like clicking. The header should have role="button" and aria-expanded toggling.
result: pass
note: Expand/collapse works via keyboard. Forecast content fails to load (same issue as test 5).

### 7. Scenario tree shows multiline text
expected: Open any forecast via "View Full Analysis" to open the ScenarioExplorer modal. Tree nodes should display 2-3 line text blocks (~80-120 characters of the scenario description) instead of the old 40-character truncated single-line labels.
result: pass

### 8. Scenario tree alternating sides layout
expected: In the ScenarioExplorer tree, nodes in the left subtree should have their text extending to the LEFT of the node circle. Nodes in the right subtree should have text extending to the RIGHT. This creates a balanced dendrogram look.
result: pass
note: User wants to revisit node circle size (make them larger) — cosmetic tweak, not a blocker.

### 9. Scenario tree pan/zoom on dense trees
expected: Open a forecast with 5+ scenario branches. The tree should be pannable (click-drag) and zoomable (scroll wheel). On a forecast with fewer than 5 scenarios, the scroll wheel should scroll the modal body normally (not zoom the tree).
result: pass

### 10. Root node shows narrative summary
expected: In the ScenarioExplorer, click the root node (the forecast question at the top of the tree). The sidebar should show a "Situation Overview" section with a 2-3 sentence analytical narrative. If the forecast predates the narrative_summary migration, a muted fallback message should appear instead.
result: pass

### 11. Root node shows related articles
expected: Below the narrative summary in the root node sidebar, related news articles should appear (fetched via semantic search). 2 articles visible by default with title, source badge, and snippet. A "Show more" link should reveal additional articles. A loading spinner should appear briefly while articles are fetching.
result: skipped
reason: Needs new forecasts with narrative_summary populated to test article retrieval

## Summary

total: 11
passed: 9
issues: 1
pending: 0
skipped: 1

## Gaps

- truth: "ComparisonPanel entries expand to reveal forecast details (ensemble, calibration, evidence, Polymarket data)"
  status: failed
  reason: "User reported: failed to load forecast — but no 500 or anything other than 200 in server logs"
  severity: major
  test: 5
  root_cause: "forecastBreaker single-slot cache poison: getForecastById() routes through forecastBreaker.execute() which returns cached ForecastResponse[] array from getTopForecasts(), not a single ForecastResponse"
  artifacts:
    - path: "frontend/src/services/forecast-client.ts"
      issue: "getForecastById uses forecastBreaker.execute() instead of bypassing like getForecastsByCountry and getCountryRisk"
  missing:
    - "getForecastById must bypass forecastBreaker, matching existing pattern at lines 176 and 223"
  debug_session: ".planning/debug/comparison-panel-forecast-fetch.md"
