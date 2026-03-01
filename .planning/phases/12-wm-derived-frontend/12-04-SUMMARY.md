---
phase: 12-wm-derived-frontend
plan: 04
subsystem: frontend-panels
tags: [typescript, panels, svg, calibration, forecast, risk-index, health]
completed: 2026-03-02
duration: 6min
dependency-graph:
  requires: [12-01, 12-02]
  provides:
    - "6 dashboard panels: ForecastPanel, RiskIndexPanel, EventTimelinePanel, EnsembleBreakdownPanel, SystemHealthPanel, CalibrationPanel"
    - "Dual API pattern: refresh() + update() on all panels"
    - "forecast-selected and country-selected CustomEvent dispatch"
    - "SVG reliability diagram + Brier table + track record sparkline"
  affects: [12-06, 12-07]
tech-stack:
  added: []
  patterns:
    - "Dual API panel pattern: refresh() for self-fetch, update(data) for injection"
    - "Hand-rolled SVG via createElementNS (no D3 dependency)"
    - "BreakerDataState-driven data badges per panel"
key-files:
  created:
    - frontend/src/components/ForecastPanel.ts
    - frontend/src/components/RiskIndexPanel.ts
    - frontend/src/components/EventTimelinePanel.ts
    - frontend/src/components/EnsembleBreakdownPanel.ts
    - frontend/src/components/SystemHealthPanel.ts
    - frontend/src/components/CalibrationPanel.ts
  modified:
    - frontend/src/styles/panels.css
decisions:
  - "isoToFlag via regional indicator symbol codepoint math (no lookup table)"
  - "CalibrationPanel uses temperature as predicted-bin proxy on x-axis, historical_accuracy as observed on y-axis"
  - "EventTimelinePanel renders mock data (6 hardcoded events) since no GDELT events endpoint exists yet"
  - "EnsembleBreakdownPanel and CalibrationPanel are update-driven (refresh is no-op) -- they respond to forecast selection, not polling"
  - "Severity thresholds: probability >0.8 critical, >0.6 high, >0.4 elevated, >0.2 normal, else low; risk score uses same tiers at 80/60/40/20"
  - "Track record sparkline requires >= 3 data points before rendering polyline; fewer shows placeholder"
metrics:
  tasks: 2/2
  duration: 6min
  commits: 3
---

# Phase 12 Plan 04: Dashboard Panels Summary

All 6 dashboard panels implemented with dual API pattern, SVG calibration visualizations, and analyst-grade aesthetic CSS.

## What Was Done

### Task 1: ForecastPanel + RiskIndexPanel + EventTimelinePanel

**ForecastPanel** (~130 lines):
- Fetches top N forecasts via `forecastClient.getTopForecasts(10)`
- Renders cards with question text (truncated 100 chars), probability bar (severity-colored fill), probability badge, country flag+ISO, confidence, scenario count, relative timestamp
- Sorted by probability descending
- Dispatches `forecast-selected` CustomEvent on click (ScenarioExplorer listens)
- Circuit breaker data badge integration

**RiskIndexPanel** (~100 lines):
- Fetches country risk data via `forecastClient.getCountries()`
- Renders rows with flag+ISO, risk score (color-coded severity), trend arrow (rising/stable/falling unicode), top question (truncated 60 chars)
- Sorted by risk_score descending
- Dispatches `country-selected` CustomEvent on click

**EventTimelinePanel** (~110 lines):
- Renders 6 mock GDELT events demonstrating expected layout
- CAMEO code badges with category coloring (cooperative/neutral/conflictual/hostile)
- Goldstein scale indicators (positive green, negative red)
- Alternating row backgrounds
- Notice text: "GDELT event feed connects when ingest daemon is running"

### Task 2: EnsembleBreakdownPanel + SystemHealthPanel + CalibrationPanel

**EnsembleBreakdownPanel** (~80 lines):
- Update-driven (refresh is no-op)
- Shows LLM vs TKG as horizontal stacked bar (LLM blue #3388ff, TKG orange #ff8800)
- Weight percentages, temperature_applied value
- Placeholder when no forecast selected

**SystemHealthPanel** (~100 lines):
- Fetches health via `forecastClient.getHealth()` (public endpoint, no API key)
- Aggregate status badge: HEALTHY (green) / DEGRADED (yellow) / UNHEALTHY (red)
- 8 subsystem rows with status dot (green/red with glow), name, detail, checked_at
- Circuit breaker data badge integration

**CalibrationPanel** (~250 lines):
- Update-driven (refresh is no-op), accepts `CalibrationDTO[]`
- Three visualizations:
  1. **Reliability diagram** (SVG 300x300): predicted vs observed scatter, dashed diagonal reference, grid lines at 0.2 intervals, dot radius proportional to sqrt(sample_size)
  2. **Brier score table**: category/score/N with color-coded severity (<0.1 green, <0.25 yellow, >=0.25 red), zebra striping
  3. **Track record sparkline** (SVG 300x80): polyline of historical_accuracy values, placeholder with dashed line at 0.5 when <3 data points
- Placeholder when no calibration data available

## CSS Added to panels.css

- Forecast cards: `.forecast-card`, `.probability-bar`, `.probability-fill`, `.forecast-meta`, severity-* color classes
- Risk rows: `.risk-row`, `.risk-score`, `.risk-country`, trend arrow colors
- Event timeline: `.event-row`, `.cameo-badge` with category colors, `.goldstein-indicator`
- Ensemble: `.ensemble-bar`, `.ensemble-segment`, `.ensemble-dot`, `.ensemble-pct`
- Health: `.health-status-badge`, `.health-subsystem`, `.status-dot` with glow effects
- Calibration: `.reliability-diagram`, `.brier-table`, `.brier-row`, `.track-record-sparkline`, brier severity classes

## Deviations from Plan

None -- plan executed exactly as written.

## Commits

| Hash | Message |
|------|---------|
| abe4455 | feat(12-04): ForecastPanel + RiskIndexPanel + EventTimelinePanel |
| d76fd6e | feat(12-04): add EnsembleBreakdownPanel + SystemHealthPanel + CalibrationPanel |

## Requirements Coverage

- **FE-03** (5 dashboard panels): ForecastPanel, RiskIndexPanel, EventTimelinePanel, EnsembleBreakdownPanel, SystemHealthPanel
- **FE-08** (CalibrationPanel with track record): Reliability diagram + Brier decomposition + track record sparkline

## Next Phase Readiness

All 6 panels ready for:
- **12-06** (main.ts wiring): Panels can be instantiated and wired via update()/refresh()
- **12-07** (RefreshScheduler): All panels expose refresh() for periodic polling
