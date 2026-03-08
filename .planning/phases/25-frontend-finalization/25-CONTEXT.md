# Phase 25: Frontend Finalization - Context

**Gathered:** 2026-03-08
**Status:** Ready for planning

<domain>
## Phase Boundary

Polish pass across the entire frontend: every panel and screen handles loading, error, and empty states gracefully. Heavy components lazy-load. Interactive elements are keyboard-accessible. Stale placeholders and stub code from prior phases are cleaned up. The frontend is ready for external users who encounter edge cases, slow connections, and assistive technology.

</domain>

<decisions>
## Implementation Decisions

### Skeleton/Loading Design
- Shimmer animation (left-to-right gradient sweep, YouTube/LinkedIn style)
- Panel-specific skeleton shapes matching each panel's layout (ForecastPanel shows card outlines, RiskIndex shows row lines, etc.) — zero layout shift when real content arrives
- Dynamically calculate how many skeleton items fit in available panel height
- Skeletons appear on initial page load only; on refresh failures where data is >2x the refresh interval old, show a subtle "updating" indicator — NOT full skeleton replacement

### Error & Retry Behavior
- Two-tier error handling based on data availability:
  - **Initial load failure** (no stale data): full-panel centered error message with manual "Retry" button
  - **Refresh failure** (stale data exists): toast banner at top of panel, stale data stays visible below
- Toast auto-dismisses after 10 seconds; if next refresh also fails, toast reappears
- Severity-graded error colors: amber for transient errors (timeout, 503), red for persistent errors (500, 404)
- Each panel's error boundary is independent — one panel failing does not affect others

### Empty State Content
- Contextual/helpful tone: explain what the panel shows AND how to populate it
- Include CTAs where a user action exists (e.g., ForecastPanel empty → "Go to Forecasts" link)
- No CTAs where data appears automatically (e.g., ComparisonPanel → "Polymarket comparisons appear automatically when markets are matched")
- Subtle muted monochrome icons above empty state text (empty chart, globe outline, etc.)

### Stale Placeholder Cleanup
- GlobeDrillDown sparkline: wire to real `/api/v1/events` data — render sparkline (event count per day, last 30 days) instead of "Event data available in Phase 17" placeholder
- CountryBriefPage CAMEO trend stub (line 869): currently hardcoded `count > 2 ? 'rising' : 'stable'` — fix or remove
- Globe sizing: initial view center/zoom fine-tuning, MapLibre container fill on first load

### Accessibility
- Keyboard navigation: core interactions only — tab through nav, modal open/close, forecast card expand/collapse, layer toggles, buttons. No arrow-key list navigation or keyboard map panning.
- Focus trapping in modals: Tab/Shift+Tab cycles within ScenarioExplorer, CountryBriefPage, SettingsModal. Focus returns to trigger element on close.
- `prefers-reduced-motion` support: shimmer animations and view transitions wrapped in `@media (prefers-reduced-motion: no-preference)`. Static fallbacks for vestibular-sensitive users.
- ARIA labels on map controls and buttons: layer toggle pills, zoom controls, close buttons, nav links. Minimum for screen reader navigation.

### Claude's Discretion
- Exact shimmer animation timing and gradient colors
- Which panels get which skeleton shapes (specific block layouts per panel)
- Icon selection for empty states
- How to implement the "updating" indicator on stale refreshes
- Exact error message wording per panel
- Whether to auto-retry on initial load failure or wait for manual retry

</decisions>

<specifics>
## Specific Ideas

- The ROADMAP.md Phase 25 plans section has copy-paste artifacts referencing Phase 20 plans — needs cleanup before planning
- Current circuit breaker already handles stale-while-revalidate — error toasts should integrate with existing CircuitBreaker state rather than duplicating failure detection

</specifics>

<deferred>
## Deferred Ideas

- **Phase 26: Globe Enhancement & Map Filters** — 3D globe rendering (MapLibre globe projection or deck.gl GlobeView) where arcs render as true great-circle arcs on a sphere. Port WM-style filter surfaces (event type, date range, severity, country) to both 2D and 3D map views. Added to v3.0 after Phase 25.

</deferred>

---

*Phase: 25-frontend-finalization*
*Context gathered: 2026-03-08*
