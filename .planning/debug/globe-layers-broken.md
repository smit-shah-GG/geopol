---
status: investigating
trigger: "3D globe layers broken: choropleth, scenarios, wrong country click, markers, heatmap"
created: 2026-03-09T00:00:00Z
updated: 2026-03-09T00:00:00Z
---

## Current Focus

hypothesis: Multiple independent bugs across the 5 globe layers
test: Trace data flow for each reported symptom
expecting: Identify distinct root causes per symptom
next_action: Continue investigating remaining bugs after markers root cause confirmed

## Symptoms

expected: All 5 layers render correctly on 3D globe; markers render on both views; polygon click resolves correct country
actual: Choropleth colors wrong, scenarios broken, wrong country on click, markers empty on both views, heatmap empty on 3D
errors: No console errors reported
reproduction: Load /globe page, observe layers
started: After Phase 27 shipped (3D globe implementation)

## Eliminated

(none yet)

## Evidence

- timestamp: 2026-03-09T00:01
  checked: calibration.category field in ForecastResponse
  found: Contains event category strings ("conflict", "diplomatic", "economic") -- NEVER 2 chars
  implication: ROOT CAUSE for Bug 4 (markers). `f.calibration.category.length === 2` always false; no markers created on either view.

- timestamp: 2026-03-09T00:02
  checked: GeoJSON ISO_A2 values and normalizeCode function
  found: All 177 features have valid ISO_A2; normalizeCode returns uppercase; risk_scores map uses uppercase keys
  implication: ISO code normalization is NOT the cause of choropleth issues

- timestamp: 2026-03-09T00:03
  checked: risk_score range and division by 100
  found: API returns 0-100, both DeckGLMap and GlobeMap divide by 100 for color mapping
  implication: Risk score scaling is correct

## Resolution

root_cause: (multiple bugs -- see diagnosis below)
fix:
verification:
files_changed: []
