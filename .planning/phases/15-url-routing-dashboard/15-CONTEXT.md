# Phase 15: URL Routing & Dashboard Screen - Context

**Gathered:** 2026-03-03
**Status:** Ready for planning

<domain>
## Phase Boundary

Replace the single-screen Phase 12 layout with three URL-routed screens (/dashboard, /globe, /forecasts). Build out the Dashboard screen as an information-dense 4-column view with progressive disclosure on forecast cards, full-text search with filtering, event feed, data source health, and user submission tracking. The Globe and Forecasts screens are scaffolded (empty/placeholder) — their content is Phase 16.

</domain>

<decisions>
## Implementation Decisions

### Navigation Chrome
- Top nav bar: "Geopol" branding left, Dashboard / Globe / Forecasts links right
- Minimal content — app name + screen links only. No status badge, no search in nav bar.
- Screen transitions: fade crossfade (~150ms)
- Dark theme only — kill the Phase 12 light/dark toggle entirely. Single theme = less code, fewer edge cases.
- URL routing: /dashboard, /globe, /forecasts. Browser back/forward must work. Direct URL entry loads correct screen.

### Dashboard Column Layout
- 4-column layout (not 3): ~15% / ~35% / ~30% / ~20% split
- Rationale: removing the globe frees enough horizontal space for 4 columns. Bloomberg terminal density — analyst sees countries, forecasts, submissions, and events simultaneously.
- Column assignments:
  - **Col 1 (~15%):** Country Risk Index — country list with risk scores, trend indicators. Click a country → filters Col 2 forecast list.
  - **Col 2 (~35%):** Search bar (with inline dropdowns) → Active Forecasts (expandable cards). Primary content column.
  - **Col 3 (~30%):** My Forecasts (user submissions + status) → Sources panel (data source health/staleness).
  - **Col 4 (~20%):** Event Feed (GDELT events + RSS headlines) → Ensemble Breakdown → System Health → Calibration (including Polymarket comparisons).
- All sections expanded by default. Each column scrolls independently.
- Cross-column interaction: Col 1 country click ↔ Col 2 country dropdown are bidirectionally synced. My Forecasts (Col 3) is NOT filtered by country selection.

### Forecast Card Expansion (Progressive Disclosure)
- Multiple cards can be expanded simultaneously (not accordion). Supports comparison.
- Collapsed state (minimal): question text + colored probability bar with percentage + country code + age (time since creation).
- Expanded state: two-column layout within the card.
  - Left side: probability bar, ensemble weights (α/β split), calibration metadata
  - Right side: mini scenario tree preview (~150px d3 visualization), top 2-3 evidence summaries, "View Full Analysis" button
- "View Full Analysis" opens existing ScenarioExplorer modal (overlay on dashboard, preserves scroll position).
- Mini scenario tree reuses Phase 12 d3-hierarchy code at reduced viewport.
- Default sort: by recency (newest first). Probability visible on each card for triage.

### Search & Filtering
- Search bar at top of Col 2 with country dropdown + category dropdown inline on the same row.
- Hybrid filtering: client-side for loaded forecasts (fast), server-side fallback via BAPI-04 endpoint when >100 forecasts or complex queries.
- Debounced input (standard ~300ms).
- Empty results: message + suggested search terms as clickable links. Use static popular terms initially (conflict, sanctions, election, nuclear, trade). LLM-powered suggestions deferred (SearchResponse.suggestions field reserved).
- Col 1 country selection and Col 2 country dropdown are bidirectionally synced — single source of truth for active country filter.

### Scenario Tree Node Text
- Short label (~40 chars) visible by default on graph nodes
- Tooltip on hover showing full scenario description
- No multiline text boxes on nodes — would clutter the graph with 4-6 scenarios

### Claude's Discretion
- Expanded card internal two-column exact proportions (optimize for ~672px available width)
- Fade crossfade implementation details (CSS transitions vs JS)
- Column responsive breakpoints (if any — mobile explicitly deferred)
- Event Feed item density and truncation
- Sources panel staleness threshold visualization
- Exact panel heights and scroll behavior within columns

</decisions>

<specifics>
## Specific Ideas

- "Bloomberg terminal density" — the 4-column layout emerged from recognizing that the globe's removal frees enough space for a richer information architecture than a typical 2 or 3 column dashboard.
- Col 1 → Col 2 cross-column filtering is the key interaction that justifies the risk index having its own column. Without it, the country list could live in a sidebar.
- Probability display is event probability (calibrated ensemble output), not system confidence. The expanded card shows calibration quality as supporting metadata alongside it.
- The existing Phase 12 panels (ForecastPanel, RiskIndexPanel, EventTimelinePanel, EnsembleBreakdownPanel, SystemHealthPanel, CalibrationPanel) are being reorganized into the 4-column structure, not rebuilt from scratch. Refactor and redistribute.

</specifics>

<deferred>
## Deferred Ideas

- **Polymarket orchestration fix (pre-Phase 15):** Phase 13 built the Polymarket backend (client, matcher, comparison service, API endpoint) and frontend CalibrationPanel rendering, but no scheduler ever calls `run_matching_cycle()` and the frontend never calls the API endpoint. The `polymarket_comparisons` table is permanently empty. Fix before Phase 15: add background task to FastAPI lifespan + wire `getPolymarket()` in forecast-client.ts + call `updatePolymarket()` from refresh scheduler. Target: `/gsd:quick` fix.
- LLM-powered search suggestions (populate SearchResponse.suggestions from Gemini) — future phase
- Mobile/responsive layout — three screens + 4 columns = poor mobile experience; explicitly deferred
- Sort controls on forecast list (user-selectable sort beyond default recency) — future enhancement

</deferred>

---

*Phase: 15-url-routing-dashboard*
*Context gathered: 2026-03-03*
