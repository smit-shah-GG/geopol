# Phase 12: WM-Derived Frontend - Context

**Gathered:** 2026-03-02
**Status:** Ready for planning

<domain>
## Phase Boundary

TypeScript dashboard consuming Geopol's FastAPI backend. Architecturally derived from World Monitor's vanilla TypeScript patterns (Panel class, AppContext, h() helper, Vite build) but with a completely new visual identity. Delivers: deck.gl globe with map layers, forecast panels, interactive scenario explorer, country brief pages, and system health panel. Develops against Phase 9 mock fixtures initially, real API data when Phase 10 endpoints are available.

Tauri desktop wrap, PWA offline support, and command palette are NOT in scope (future phases or backlog).

</domain>

<decisions>
## Implementation Decisions

### Globe & map layers
- All 5 layers visible on initial load: ForecastRiskChoropleth, ActiveForecastMarkers, KnowledgeGraphArcs, GDELTEventHeatmap, ScenarioZones
- Risk choropleth uses blue-to-red diverging color scale (cool blue = stable, neutral gray = moderate, hot red = high risk). Perceptually uniform, colorblind-accessible.
- Knowledge graph arcs appear on country hover/select only — not a persistent global layer. Shows actor-to-actor relations for the selected country.
- GDELT event heatmap has user-selectable time range: 24h / 7d / 30d / 90d (dropdown or slider)
- All layers toggleable via UI controls

### Forecast panel & scenario explorer
- ForecastPanel sorts by probability (highest first)
- Each forecast card shows: question text, probability bar/badge, country flag/code, confidence level, scenario count, last-updated timestamp
- ScenarioExplorer renders as vertical top-down tree (root question at top, branches expand downward)
- Evidence displayed in a fixed sidebar panel to the right of the scenario tree — click a branch node to populate sidebar with evidence for that branch
- ScenarioExplorer opens as a modal when clicking a forecast card

### Country brief pages
- Opens as full-screen modal (WM's CountryBriefPage structure)
- Primary tab is Overview/Summary: synthesized view with risk score, top forecast, recent events, calibration snapshot
- Tabs: Overview | Active Forecasts | GDELT Events | Forecast History | Entity Relations | Calibration
- Forecast History tab: line chart showing probability evolution over time + scrollable list of past forecasts below
- Entity Relations tab: toggle between force-directed graph view and structured table view
- Calibration tab: reliability diagram + Brier score decomposition (reliability, resolution, uncertainty)

### Visual identity & layout
- Minimal WM aesthetic reuse — take architectural patterns (Panel class, AppContext, h() helper, Vite build system, CSS variable structure) but build a completely new visual identity
- Intelligence/analyst aesthetic: Bloomberg Terminal meets Palantir direction. Dark default, high contrast, data-dense, monospace accents. Professional, serious.
- Panel grid with resizable panels (drag borders to adjust proportions), fixed positions (no drag-and-drop reordering)
- Dark theme default with light theme toggle available
- Panels resize with viewport

### Claude's Discretion
- Specific color palette values within the blue-to-red diverging and intelligence aesthetic direction
- Typography choices (specific monospace font, heading font)
- Panel grid default proportions and breakpoints
- Loading skeleton and error state designs
- Force-directed graph rendering library choice
- Timeline chart library choice
- Layer toggle UI component design (sidebar vs floating panel vs toolbar)
- Exact CSS variable naming and theme structure

</decisions>

<specifics>
## Specific Ideas

- "Bloomberg Terminal meets Palantir" — dark, data-dense, high contrast, monospace accents. This is NOT a consumer web app; it's an analyst tool.
- WM is a code quarry for architecture, not aesthetics. Panel class, AppContext singleton, h() helper, Vite build system, DataLoaderManager circuit breaker pattern — these carry. Every visual element is replaced.
- ScenarioExplorer evidence sidebar: branch node click populates sidebar, tree and evidence visible simultaneously (no navigation away from tree context)
- Entity Relations dual-view serves both visual thinkers (graph) and tabular thinkers (table)
- Country Brief Overview tab is a dashboard-within-a-dashboard: risk score + top forecast + recent events + calibration snapshot in one view before drilling into tabs

</specifics>

<deferred>
## Deferred Ideas

- Tauri desktop wrap (src-tauri/ scaffold, keychain secrets, persistent cache, auto-update) — Phase 13 or future
- PWA offline support (cached forecasts viewable without network) — future
- Command palette (Cmd+K search for forecasts, countries, entities) — future
- Variant system (regional forecast perspectives: MENA, Europe, Asia-Pacific) — v3.0+
- URL state binding (shareable deep links like ?country=SY&forecast=abc123) — future
- Export functionality (JSON, CSV, PNG forecast reports) — future
- Playback mode (time travel through forecast evolution) — v3.0+
- Per-CAMEO calibration breakdown in country brief — can add to Calibration tab later if needed

</deferred>

---

*Phase: 12-wm-derived-frontend*
*Context gathered: 2026-03-02*
