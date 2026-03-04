# Frontend Redesign: Three-Screen Architecture

**Status:** Design document — not yet planned or executed
**Date:** 2026-03-02
**Context:** v2.0 shipped with Phase 12 single-screen dashboard. Live UAT revealed UX gaps: empty country views, zero evidence display, fixture bleed-through, globe competing for space with informational panels. This document captures the agreed redesign.

## Architecture: Three Screens with URL Routing

Routes: `/dashboard`, `/globe`, `/forecasts`
URL-based (not tab state) for bookmarkability and shareability.

### Screen 1: Dashboard (`/dashboard`)

**Purpose:** Primary information-dense view. Replaces the current single-screen layout.

**Layout:** WM-style scrollable columns with collapsible sections. No globe — freed space goes to feeds, sources, search, and expanded forecast cards.

**Panels:**
- **Active Forecasts** (primary) — scrollable list with search/filter. Click-to-expand inline (see Progressive Disclosure below). More space than current layout since globe is gone.
- **My Forecasts** — user-submitted questions and their status (pending → processing → complete). Links to Screen 3 for submission.
- **Event Feed** — live GDELT event stream, RSS article headlines. Compact timeline format.
- **Sources** — active data sources with health/staleness indicators (GDELT, RSS tiers, Polymarket).
- **System Health** — condensed health bar from `/api/v1/health`.
- **Ensemble Breakdown** — quick view of current ensemble weights (α/β), calibration status.

**Search:** Full-text search over active forecasts (question text, country, category). Essential at 30+ forecasts. Can use client-side filtering for <100 items, upgrade to server-side if needed.

### Screen 2: Globe (`/globe`)

**Purpose:** Geospatial exploration optimized for the deck.gl globe.

**Layout:** Full-viewport globe with overlay panels that appear contextually.

**Interactions:**
- Country click → slide-in panel with all forecasts for that country, risk timeline, GDELT event sparkline
- Choropleth coloring by aggregate risk score (real data, not mock)
- Layer toggles (forecast markers, conflict arcs, heatmap, scenario zones)
- The globe must *do* something the dashboard can't — drill-down exploration, not just a sphere with dots

**Not just a viewport:** Without contextual drill-down, this screen is a vanity page. The country click → panel flow is the minimum viable interaction.

### Screen 3: Forecast Submission (`/forecasts`)

**Purpose:** User submits forecast questions and views their history.

**NOT a chatbot.** Each forecast takes 2-3 minutes (multiple Gemini calls for scenario gen, graph validation, ensemble prediction). Chatbot UX implies conversational latency — users will stare at a spinner thinking it's broken.

**Model: Question submission queue.**
- Input form: natural language question text ("Will Iran retaliate against Israel within 30 days?")
- LLM parses into structured form (country_iso, horizon_days, category) — shown to user for confirmation
- Submit → enters queue with status `pending`
- Backend processes async (batch with daily pipeline, or on-demand if budget allows)
- Queue display: pending / processing / complete
- Complete forecasts expand inline with same UX as Screen 1 active forecasts

**Optional future enhancement:** SSE/WebSocket streaming for partial results as pipeline progresses (scenarios generated → validating → ensemble complete). Significant plumbing — defer unless UX testing shows the wait is intolerable.

## UX Improvements

### Progressive Disclosure on Forecast Cards

**Current:** Click forecast → immediately opens full ScenarioExplorer modal. Jarring.

**New (two-level):**
1. **Click once → inline expand** revealing:
   - Probability bar (already visible)
   - Ensemble weights (α LLM / β TKG split)
   - Evidence count + top 2-3 evidence summaries
   - Horizon and expiry date
   - Calibration metadata (per-category weight if available)
   - "View Full Analysis" button
2. **Click "View Full Analysis" → opens ScenarioExplorer modal** with full scenario tree, node graph, evidence sidebar

### Scenario Tree Node Text Rendering

**Current bug:** Straight-line labels on graph nodes truncate with `...` before showing anything useful. Scenario descriptions are 1-2 sentences — impossible to read.

**Fix:** Short label (first ~40 chars) visible by default + **tooltip on hover** showing full text. Multiline text boxes on every node would visually clutter the graph with 4-6 scenarios.

### Country Risk from Real Data

**Current:** `/api/v1/countries` returns hardcoded mock list (SY, UA, MM, IR, TW, SD, KP, VE with fake risk scores).

**New:** Aggregate from `predictions` table:
```sql
SELECT country_iso,
       COUNT(*) as forecast_count,
       AVG(probability) as avg_probability,
       MAX(probability) as max_risk
FROM predictions
WHERE expires_at > NOW()
GROUP BY country_iso
ORDER BY max_risk DESC
```

Risk score derived from actual forecast probabilities, not static numbers.

## Backend Changes Required

### 1. Kill Mock Fixture Fallback

The fixture fallback in `forecasts.py` (lines 213-236) is actively harmful now that real data exists. Mock fixtures bleed across countries (Myanmar showing under Syria — the `_guess_country_iso` heuristic on UUID forecast IDs is broken). Remove the fallback chain entirely; return empty results when PostgreSQL has no data for a country.

### 2. Question Submission Queue

New DB table: `forecast_requests`
- `id` (UUID)
- `question` (text, user-submitted)
- `country_iso` (parsed by LLM or user-specified)
- `horizon_days` (parsed or default 30)
- `category` (parsed or default "conflict")
- `status` (pending | processing | complete | failed)
- `submitted_by` (API key client_name)
- `submitted_at` (timestamp)
- `prediction_id` (FK to predictions, NULL until complete)

New endpoints:
- `POST /api/v1/forecasts/submit` — submit question, returns request ID
- `GET /api/v1/forecasts/requests` — list user's submitted requests with status
- Processing: either batch with daily pipeline or on-demand worker

### 3. Real Country Risk Aggregation

Replace mock `/countries` endpoint with PostgreSQL aggregation query. Include:
- `forecast_count`: number of active forecasts
- `risk_score`: derived from max/avg probability of active forecasts
- `trend`: rising/stable/falling based on recent vs historical probability
- `top_forecast`: highest-probability active forecast question

### 4. Search Endpoint

`GET /api/v1/forecasts/search?q=iran&category=conflict&country=IR`
- Full-text search on `predictions.question`
- Filterable by country, category, horizon range
- PostgreSQL `ts_vector` + GIN index for performance at scale

## Bugs to Fix (Pre-Redesign)

1. **Myanmar under Syria**: Mock fixture bleed-through. `_guess_country_iso` parses UUID as `fc-{iso}-{hash}` format — UUIDs don't follow this pattern. Fix: remove fixture fallback.
2. **Evidence count = 0**: Fixed in c3158da (per-scenario evidence now serialized from reasoning_path). New forecasts have evidence; old ones need re-generation.
3. **`_get_fixture_cache` leaks across requests**: Module-level dict accumulates forever within one process. Will be irrelevant once fixtures are removed.

## Out of Scope

- Mobile/responsive layout (three screens + globe = poor mobile experience; defer)
- Real-time collaboration (multiple users viewing same forecast)
- Forecast comparison view (side-by-side two forecasts)
- Historical forecast replay / time travel
