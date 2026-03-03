# Phase 16: Globe & Forecasts Screens - Context

**Gathered:** 2026-03-03
**Status:** Ready for planning

<domain>
## Phase Boundary

Deliver the Globe screen (/globe) with full-viewport geospatial exploration and contextual country drill-down, and the Forecasts screen (/forecasts) with question submission workflow and queue status display. These are the remaining two of three URL-routed screens (Dashboard delivered in Phase 15). All three screens complete after this phase.

</domain>

<decisions>
## Implementation Decisions

### Globe viewport
- Minimal HUD overlay in corner when nothing is selected: total active forecasts, countries with forecasts, last data update timestamp
- Full-viewport globe underneath all overlays -- no permanent sidebar
- Static camera -- no auto-rotate on idle
- Country click triggers simultaneous camera fly-to (center on country) AND slide-in panel open

### Globe drill-down panel
- Slide-in overlay from right edge -- overlaps the globe, globe stays full viewport underneath
- Panel contains three sections: active forecasts list, risk score with trend arrow, GDELT event sparkline
- Empty country: panel still opens, displays "No active forecasts for [Country]" with risk score 0 or N/A
- Forecast click in drill-down uses same progressive disclosure as dashboard: inline expand with mini d3 tree, ensemble weights, evidence preview. "View Full Analysis" opens ScenarioExplorer modal
- "View Details" link at bottom opens CountryBrief modal (Phase 12 six-tab full-screen)

### Layer toggle controls
- Floating pill bar -- compact horizontal bar of toggle pills, floating over the globe
- Default layers ON: choropleth (risk coloring) + forecast markers
- Default layers OFF: conflict arcs, heatmap, scenario zones
- All 5 layers independently toggleable (choropleth, markers, arcs, heatmap, scenario zones)
- Fade transition (~200ms) on layer toggle

### Forecasts submission screen
- Two-column split layout: left column is the submission form, right column is scrollable queue of past/active submissions
- LLM confirmation step: form transforms inline (replaces itself) to show parsed fields (country, horizon, category) with Edit/Confirm buttons -- no modal, no navigation
- Processing state: status card in queue with badge (processing), elapsed time counter, subtle animation. No fake progress bar
- Completed forecast: same progressive disclosure as dashboard/globe -- click expands inline with mini tree preview, "View Full Analysis" for ScenarioExplorer modal
- Consistent expand pattern across all three screens (dashboard, globe drill-down, forecasts queue)

### Claude's Discretion
- HUD overlay exact positioning and content layout
- Pill bar positioning relative to drill-down panel (avoid overlap)
- Camera fly-to animation easing and duration
- Submission form field layout and placeholder text
- Elapsed time display format during processing

</decisions>

<specifics>
## Specific Ideas

- Progressive disclosure is consistent everywhere: click-expand inline first (mini tree + ensemble + evidence), "View Full Analysis" for full ScenarioExplorer modal. Same pattern on dashboard cards, globe drill-down forecasts, and forecasts queue results.
- The globe should feel like a data exploration tool, not a screensaver -- static camera, contextual overlays, information appears on interaction.

</specifics>

<deferred>
## Deferred Ideas

None -- discussion stayed within phase scope. Event sparkline real data and additional data sources are Phase 17 scope.

</deferred>

---

*Phase: 16-globe-forecasts-screens*
*Context gathered: 2026-03-03*
