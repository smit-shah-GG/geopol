# Phase 24: Global Seeding & Globe Layers - Context

**Gathered:** 2026-03-08
**Status:** Ready for planning

<domain>
## Phase Boundary

The globe choropleth renders meaningful risk data for all ~195 countries (not just those with active forecasts), and the three currently-empty globe layers (heatmap, arcs, scenarios) display real data from the event store and knowledge graph. Two distinct deliverables: (1) universal baseline risk computation, (2) globe layer data wiring.

</domain>

<decisions>
## Implementation Decisions

### Risk Score Formula — Dual-Score Model
- **Two independent scores**: baseline_risk (universal, all ~195 countries) and forecast_risk (where active predictions exist)
- **Blended score** when both exist: 70% forecast / 30% baseline
- API exposes all three fields: `baseline_risk`, `forecast_risk` (nullable), `blended_risk`
- Countries with only baseline show baseline as their choropleth color
- Countries with both show the 70/30 blend

### Baseline Risk Inputs & Weights
- **4 inputs**: GDELT event density, ACLED conflict intensity, travel advisory level, Goldstein severity
- **Component weights**: Advisory 35%, ACLED 25%, GDELT 25%, Goldstein 15%
- **GDELT normalization**: per-capita (events / population) for fair cross-country comparison
- **Decay window**: 90 days, exponential decay
- **Advisory hard floors**: Level 4 (Do Not Travel) = minimum baseline score 70; Level 3 = minimum 45. Other levels contribute via normal weighted blending.

### Heatmap Layer
- **H3 hex binning** (resolution ~4) for uniform hexagonal visualization
- Server-side aggregation — pre-computed, not raw event points
- deck.gl H3HexagonLayer renders the hexagons

### Arcs Layer
- **Contextual on zoom**: global view shows top ~20 strongest bilateral relationships; zoomed into a region shows all arcs involving visible countries
- **Arc encoding**: color = sentiment (red for conflictual/negative avg Goldstein, blue for cooperative/positive), width = event volume between the pair
- Bilateral relationships derived from knowledge graph edges

### Scenarios Layer — Risk Change Zones
- Repurposed from vague "scenario zones" to **risk delta visualization**
- Colors regions where risk score changed significantly (>10 points) in the last 7 days
- Shows where things are getting worse (red shift) or better (green shift)

### Refresh Cadence & Scheduling
- **Hourly** APScheduler job for baseline risk + all layer data (heatmap hex bins, arcs, risk deltas)
- **Heavy job** dispatched to ProcessPoolExecutor (full process isolation)
- **Skip-if-locked**: if the heavy job lock is held (daily pipeline, backtest, etc.), skip this cycle. Next hour tries again. Prevents queue stacking.
- **Run on startup**: compute baseline immediately when FastAPI starts so globe has data from first page load
- **Staleness handling**: always serve last computed data with a `computed_at` timestamp. No degraded state — frontend shows "Updated Xh ago" if stale.

### Layer Data Storage
- **PostgreSQL tables** for all pre-computed data: `baseline_country_risk`, `heatmap_hexbins`, `country_arcs`, `risk_deltas`
- Same DB as baseline risk. Consistent, queryable, transactional.
- Alembic migration for new tables.

### Country Edge Cases
- **Disputed territories**: include with ISO codes (XK, TW, PS, EH) but flag as `disputed: true` in API response. Scored independently like any country.
- **Microstates with zero signal**: excluded from map (render as neutral/gray). Only seed countries that have actual event data or advisory signal.
- **Canonical country list**: static ISO 3166-1 alpha-2 dict (~195 sovereign states + selected disputed territories), shipped as Python dict
- **FIPS-to-ISO mapping**: **dynamically generated** using `pycountry` library (or equivalent) at import time rather than a hand-coded static dict. The static ~250-entry dict approach triggers AI content filters during code generation. Dynamic generation from a standards library is more maintainable and avoids this blocker. Unmapped codes logged and skipped.

### Content Filter Constraint
- **Large country-name dicts and ISO code tables trigger AI content filters** during executor agent code generation. Any plan that requires generating large static dicts of country names, ISO codes, or geopolitical territory lists must use dynamic generation from a library (e.g., `pycountry`) or load from a data file, NOT inline static Python dicts. This is a hard operational constraint on the planning/execution workflow.

### Claude's Discretion
- Exact H3 resolution level (suggested ~4 but Claude can adjust based on visual density)
- Zoom level thresholds for contextual arc filtering
- Risk delta significance threshold (suggested >10 points, but Claude can tune)
- Population data source for per-capita normalization
- Exact exponential decay half-life within the 90-day window

</decisions>

<specifics>
## Specific Ideas

- The dual-score model means the country drill-down panel (GlobeDrillDown) can show a breakdown tooltip: "Baseline: 45, Forecast: 72, Blended: 64"
- Risk change zones (scenarios layer) should feel like a "what changed" overlay — the analytical value is showing deterioration/improvement trends
- Arcs with sentiment coloring (red/blue) against the choropleth should use distinct enough hues to avoid visual confusion with choropleth risk colors

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 24-global-seeding-globe-layers*
*Context gathered: 2026-03-08*
