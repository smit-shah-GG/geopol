# Phase 26: Operational Fixes & UX Polish - Research

**Researched:** 2026-03-08
**Domain:** Polymarket filtering, d3 tree rendering, SPA route refresh, expandable card extraction
**Confidence:** HIGH

## Summary

Phase 26 is a polish/fix phase touching 6 distinct areas: (1) Polymarket binary-only filtering, (2) scenario tree root node content, (3) scenario tree text rendering overhaul, (4) clickable forecast entries across all panels, (5) route navigation data refresh, and (6) poller enablement. All 6 areas modify existing code with well-understood interfaces -- no new external dependencies required.

The codebase is in a mature state (90 plans across 25 phases). Every file that needs modification exists and has been thoroughly examined. The primary risk is the d3 tree rendering overhaul, which requires significant layout logic changes in `ScenarioExplorer.ts`. The remaining items are straightforward wiring/filtering changes.

**Primary recommendation:** Plan as 6 independent work units. The tree rendering overhaul is the only item with meaningful implementation complexity -- all others are mechanical.

## Standard Stack

No new libraries needed. All work uses existing dependencies.

### Core (already installed)
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| d3 | ^7.9.0 | Tree layout, bezier links, zoom/pan | Already in `frontend/package.json`, used by `ScenarioExplorer.ts` and `expandable-card.ts` |
| APScheduler | 3.11.2 | Job scheduling (poller enablement) | Already in `pyproject.toml`, pinned per prior decision |
| SQLAlchemy | 2.x | ORM for Prediction/PolymarketComparison models | Already wired, async session pattern |

### Supporting (already installed)
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| d3-zoom | bundled with d3 v7 | SVG pan/zoom on dense scenario trees | When tree has 5+ scenario nodes |
| d3-tree | bundled with d3 v7 | Tree layout with dynamic node spacing | Core tree rendering |

### No New Dependencies
This phase MUST NOT introduce any new npm or Python packages. Everything needed exists in the current dependency tree.

## Architecture Patterns

### Pattern 1: Polymarket Binary Market Filter

**What:** Filter at the `auto_forecaster.run()` candidate loop level using the Gamma API `outcomes` field on each market within the event.

**Current state:** `auto_forecaster.py` lines 316-337 iterate `geo_events`, checking `event_id`, `tracked_ids`, `existing_event_ids`, and `horizon`. No outcomes check exists.

**Gamma API market structure (verified via live API query):**
```json
{
  "outcomes": ["Yes", "No"],
  "outcomePrices": ["0.121", "0.879"]
}
```

Multi-option markets have different outcomes arrays, e.g.:
```json
{
  "outcomes": ["Khamenei", "Mojtaba Khamenei", "Other"]
}
```

**Implementation pattern:**
```python
def is_binary_market(event: dict) -> bool:
    """Check if ALL markets in the event are binary Yes/No."""
    markets = event.get("markets", [])
    if not markets:
        return False
    for market in markets:
        outcomes = market.get("outcomes")
        if not isinstance(outcomes, list):
            return False
        # Strict check: exactly ["Yes", "No"] (case-sensitive, Gamma API convention)
        if outcomes != ["Yes", "No"]:
            return False
    return True
```

**Where to filter:** In the candidate loop (line ~330 in `auto_forecaster.py`), add the check right after the `event_id` dedup checks. Non-binary events get logged and skipped:
```python
if not is_binary_market(event):
    logger.info("Skipping non-binary market: %s (event=%s)", title[:60], event_id)
    continue
```

**Existing non-binary comparisons:** The `PolymarketComparison.status` column supports `active | resolved | voided`. The context says "marked as excluded." Two options:
1. Add `excluded` to the status enum via Alembic migration -- creates schema change.
2. Use existing `voided` status with a distinguishing `match_confidence = 0.0` sentinel -- zero schema change.

**Recommendation:** Use `excluded` as a new status value. No Alembic migration needed -- the column is `String(20)`, not a DB-level enum. SQLAlchemy will accept any string value. Just need a one-time data fixup query:
```sql
UPDATE polymarket_comparisons
SET status = 'excluded'
WHERE polymarket_event_id IN (
  SELECT pc.polymarket_event_id FROM polymarket_comparisons pc
  -- join to identify non-binary events; or more practically,
  -- run a script that fetches event details and checks outcomes
);
```

In practice: write a one-time async function that queries all `active` comparisons, fetches their event details from Gamma API, checks `outcomes`, and sets `status = 'excluded'` for non-binary ones.

**Impact on accuracy metrics:** The `_compute_accuracy_snapshot()` in `comparison.py` already filters on `status == 'resolved'`. An `excluded` status is automatically excluded from Brier calculations. No changes needed there.

### Pattern 2: Scenario Tree Root Node Content

**What:** Root node in `ScenarioExplorer` currently shows "Root forecast node. Select a scenario branch to view evidence." (line 463). Replace with relevant news articles + LLM-generated narrative summary.

**Current state:** `ScenarioExplorer.populateSidebar()` line 460-468 has a special case for `!scenario` (root node) that shows a placeholder.

**Data flow:** The narrative and articles need to come from the `ForecastResponse` DTO. Two new fields are needed:

1. **`narrative_summary`**: LLM-generated at forecast time, stored on the `Prediction` row as a new column (or within the existing `scenarios_json` blob under a reserved key).
2. **`related_articles`**: ChromaDB semantic search results against the forecast question text, fetched at display time via the existing `/api/v1/articles?semantic=true&text=<question>` endpoint.

**Decision from context:** "Narrative generated at forecast time, stored with the Prediction row (zero latency on open)."

**Storage approach -- two options:**
1. New `narrative_summary` column on `Prediction` (requires Alembic migration, clean separation).
2. Store in `scenarios_json` under a `"_root_narrative"` key (zero migration, slightly pollutes the scenarios blob).

**Recommendation:** New nullable `Text` column `narrative_summary` on `Prediction`. This is a clean column addition -- single Alembic migration, no data loss risk. The `ForecastResponse` Pydantic schema and TypeScript `ForecastResponse` interface both need the new field.

**Narrative generation:** During `EnsemblePredictor.predict()` or in `ForecastService.persist_forecast()`, call Gemini with a summary prompt:
```python
NARRATIVE_PROMPT = """Given this geopolitical forecast question and prediction, write a 2-3 sentence narrative summary explaining the current situation and key factors:

Question: {question}
Prediction: {prediction}
Probability: {probability}
Key scenarios: {top_scenarios}

Write a concise, analytical narrative (not a list). Focus on WHY this probability, not what the question asks."""
```

This costs 1 additional Gemini call per forecast. Given the existing budget (25/day, 3 new + 5 reforecast), this is acceptable -- the narrative call is cheap (small input, small output).

**Related articles:** Fetched client-side in `ScenarioExplorer.populateSidebar()` when the root node is selected. Use the existing `forecastClient` to call `/api/v1/articles?semantic=true&text=<question>&limit=5`. Show 2-3 by default with "Show more" expanding to all 5.

### Pattern 3: Scenario Tree Text Rendering Overhaul

**What:** Replace the current ~40-char truncated single-line labels with multi-line text blocks, alternating-sides layout, dynamic spacing, and bezier curves.

**Current implementation in `ScenarioExplorer.renderTree()` (lines 242-378):**
- `d3.tree<TreeDatum>().nodeSize([200, 100])` -- fixed 200px horizontal, 100px vertical spacing
- `d3.linkVertical()` for straight connector lines
- SVG `<text>` with `truncate(node.data.name, 40)` -- single line, 40 chars max
- No pan/zoom

**d3 API for the overhaul (verified against d3 v7, HIGH confidence):**

1. **Multi-line text:** SVG `<text>` does not support line wrapping. Use `<foreignObject>` with an HTML `<div>` inside for proper word-wrapping, or manually split text into `<tspan>` elements. `<foreignObject>` is simpler and supports CSS styling.

2. **Alternating sides:** After `treeLayout(root)`, post-process nodes to determine which subtree they belong to (left child vs right child of root). Text blocks for left-subtree nodes extend left (`text-anchor: end`, negative x offset), right-subtree nodes extend right (`text-anchor: start`, positive x offset).

3. **Dynamic node separation:** Use `d3.tree().separation((a, b) => ...)` with a custom function that estimates text block height (lines * lineHeight) and returns proportional separation.

4. **Bezier curves:** Already partially there -- `d3.linkVertical()` generates cubic bezier curves by default. The current implementation uses this but calls them "straight connectors" in the context. Actually, `linkVertical` produces bezier curves already. The context may be referring to visual perception. To get more pronounced curves, use a custom link generator with explicit control points:
```typescript
const linkGen = d3.linkVertical<...>()
  .x(d => d.x)
  .y(d => d.y);
// This already produces cubic beziers in d3 v7
```

5. **Pan/zoom:** `d3.zoom()` applied to the SVG element:
```typescript
import { zoom as d3zoom, type ZoomBehavior } from 'd3';

const zoomBehavior: ZoomBehavior<SVGSVGElement, unknown> = d3zoom<SVGSVGElement, unknown>()
  .scaleExtent([0.3, 3])
  .on('zoom', (event) => {
    svgGroup.attr('transform', event.transform);
  });

d3.select(svg).call(zoomBehavior);
```

Wrap all tree content in a `<g>` group element, apply zoom transform to that group, not the SVG itself.

### Pattern 4: Shared ForecastExpandableCard Extraction

**What:** Extract `ForecastExpandableCard` component used by all three panels (Active Forecasts, My Forecasts, Polymarket Comparisons).

**Current state:**
- `expandable-card.ts` already exports `buildExpandableCard()`, `buildExpandedContent()`, `updateCardInPlace()` used by `ForecastPanel.ts` and `MyForecastsPanel.ts`.
- `MyForecastsPanel.ts` uses `buildExpandableCard()` directly for completed forecasts (lines 197-214).
- `ComparisonPanel.ts` does NOT use the expandable card -- it renders custom dual-bar entries (lines 90-160) with NO expansion capability.

**The existing `expandable-card.ts` IS the shared component.** It already handles:
- Collapsed card: question + probability bar + country + age
- Expanded card: ensemble weights, calibration, mini tree, evidence, "View Full Analysis"
- Polymarket badge (P indicator) when `f.polymarket_comparison` exists
- Inline comparison section with sparkline when expanded

**What's missing for full parity:**
1. `ComparisonPanel` entries need to become expandable cards with the same expand/collapse behavior.
2. `ComparisonPanel` entries need enrichment: the `ComparisonPanelItem` currently does NOT include the full `ForecastResponse` data. The panel would need to fetch the full forecast by `geopol_prediction_id` to render the expanded content (similar to how `MyForecastsPanel.loadAndRenderForecast()` works).
3. Polymarket-specific additions for comparison cards: inline market price, divergence (pp), and dual-line sparkline should appear in the collapsed card header, not just in the expanded section.

**Implementation approach:**
- `ComparisonPanel` entries: keep the dual-bar collapsed view (GP vs PM bars) but make the entire entry clickable to expand.
- On expansion, lazy-fetch the full `ForecastResponse` via `forecastClient.getForecastById(comp.geopol_prediction_id)`.
- Render the same `buildExpandedContent(forecast)` plus an extra "vs Polymarket" section showing market price, divergence, and sparkline.
- This matches the `MyForecastsPanel.buildCompleteRow()` pattern exactly.

### Pattern 5: Route Navigation Data Refresh

**What:** Navigating to /dashboard, /globe, or /forecasts triggers a full data refresh equivalent to ctrl+r.

**Current behavior (the bug):**
- `Router.navigate()` (line 44) has `if (window.location.pathname === path) return;` -- clicking the same nav link is a no-op.
- Even navigating to a different route and back doesn't bust the circuit breaker cache -- stale data from the cache TTL (10 minutes, per `DEFAULT_CACHE_TTL_MS`) is served.

**Two changes needed:**

1. **Router: allow same-route navigation.**
   Remove the early return on same-path or make it configurable. The context says "even clicking the same nav link" triggers refresh:
   ```typescript
   navigate(path: string): void {
     // Removed: if (window.location.pathname === path) return;
     window.history.pushState(null, '', path);
     void this.resolve();
   }
   ```

   But `resolve()` has `if (route === this.currentRoute) return;` which also short-circuits. For same-route "refresh", we need to:
   - Force unmount + remount of the current screen (true ctrl+r equivalent), OR
   - Skip the `route === this.currentRoute` guard and force a remount

   **Recommendation:** Force full unmount/remount. This is the cleanest ctrl+r equivalent. The unmount destroys the RefreshScheduler and all panels. The mount recreates everything from scratch, triggering fresh `loadInitial()`.

2. **Circuit breaker cache invalidation.**
   The `CircuitBreaker` class has no `invalidate()` or `clearCache()` method. Add one:
   ```typescript
   invalidateCache(): void {
     this.cache = null;
     this.lastDataState = { mode: 'unavailable', timestamp: null };
   }
   ```

   The `forecastClient` wraps multiple circuit breakers. Add a `bustAllCaches()` method:
   ```typescript
   bustAllCaches(): void {
     this.forecastBreaker.invalidateCache();
     this.countryBreaker.invalidateCache();
     this.healthBreaker.invalidateCache();
     // ... all breakers
   }
   ```

   Call `forecastClient.bustAllCaches()` in the router before mounting the new screen.

3. **Skeleton states during refresh.**
   Each panel's `hasData` flag should be reset on remount. Since we're doing full unmount/remount, `hasData` starts as `false` in every new panel constructor -- skeletons will show automatically. No extra work needed.

4. **Globe reset to default position/zoom.**
   `DeckGLMap` constructor sets the initial view state. Since we destroy and recreate the entire map on remount, the globe naturally resets. No extra work.

5. **Exception: /forecasts form preservation.**
   The submission form input text should NOT be cleared on same-route refresh. The `SubmissionForm` component will be destroyed and recreated, losing input. Two approaches:
   - Store input text in `sessionStorage` and restore on mount.
   - Or: only bust data caches (queue refresh), don't unmount the form on same-route `/forecasts` navigation.

   **Recommendation:** `sessionStorage` persistence for form input. Simple, survives refresh, and the `unmount` function can clear it on route change to a different screen.

### Pattern 6: Poller Enablement

**What:** Enable Polymarket poller and baseline risk poller.

**Current state:**
- `registry.py` line 151: `if settings.polymarket_enabled:` gates the polymarket job registration. `settings.polymarket_enabled` defaults to `True` in `settings.py` line 93. So the job IS already registered unless the setting was overridden via env var or DB config.
- `heavy_baseline_risk` is registered unconditionally (lines 179-188) with 3600s interval. It's already wired.

**The "enablement" is likely about ensuring the settings are correct and the jobs actually fire:**
1. Verify `POLYMARKET_ENABLED=true` in `.env` or equivalent config.
2. Verify `polymarket_enabled` in `system_config` DB table is not overridden to `false`.
3. Verify the scheduler starts correctly on app boot (no silent failures).
4. Verify the heavy_runner `run_polymarket_cycle()` completes without error (test manually or check logs).

**This is entirely mechanical -- no code changes needed.** It's an ops/config task. The planner should create a verification checklist, not a coding task.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| SVG text wrapping | Manual `<tspan>` line splitting | `<foreignObject>` with HTML `<div>` | CSS word-wrap handles arbitrary text; `<tspan>` requires manual measurement |
| Pan/zoom on SVG | Custom mousewheel/drag handlers | `d3.zoom()` | Built-in to d3 v7, handles touch, mouse, and keyboard; provides smooth transforms |
| Same-route refresh | Custom refresh signals | Full unmount/remount via Router | Guarantees clean state, no stale references, identical to page reload |
| Circuit breaker cache bust | Re-fetching with cache-control headers | `invalidateCache()` method on `CircuitBreaker` | Controls the client-side TTL cache directly; HTTP headers won't help since the cache is in-memory |

## Common Pitfalls

### Pitfall 1: Non-binary market detection at event vs market level

**What goes wrong:** Polymarket events contain multiple markets. Some events have a mix of binary and non-binary markets. Checking only the first market misses multi-market events.
**Why it happens:** The Gamma API event structure nests markets as an array. An event like "Who will be the next X?" has a single market with `outcomes: ["Person A", "Person B", ...]`. But some events have multiple binary markets (each with `["Yes", "No"]`).
**How to avoid:** Check ALL markets in the event. The filter function must return `True` only when ALL markets have exactly `["Yes", "No"]` outcomes.
**Warning signs:** Forecasts appearing for multi-outcome questions.

### Pitfall 2: Router same-route navigation breaks back button

**What goes wrong:** Removing the same-path guard in `navigate()` causes `history.pushState()` on every click, filling the browser history with duplicate entries.
**Why it happens:** Each click pushes a new history entry for the same URL.
**How to avoid:** Use `replaceState` instead of `pushState` when the path hasn't changed:
```typescript
navigate(path: string): void {
  if (window.location.pathname === path) {
    // Same route: replace state (no new history entry) but force remount
    window.history.replaceState(null, '', path);
  } else {
    window.history.pushState(null, '', path);
  }
  void this.resolve();
}
```
**Warning signs:** Hitting "back" repeatedly stays on the same page.

### Pitfall 3: Fetch triggers during view transition cause layout shift

**What goes wrong:** The context says "Fetch triggers immediately on route change, not after view transition." If fetches start before the View Transition API's `updateCallbackDone` resolves, DOM mutations during the transition cause visual glitches.
**Why it happens:** View Transition API captures a screenshot-based animation. DOM changes during the animation phase break the snapshot.
**How to avoid:** The fetch should trigger immediately on route change, but DOM updates (panel rendering) happen during the `doSwap()` callback, which the View Transition API coordinates. The existing pattern already handles this -- `doSwap()` unmounts old and mounts new, and mount triggers `loadInitial()`. The View Transition wraps `doSwap()`, so DOM mutations are within the transition callback. This is correct as-is.
**Warning signs:** Flash of empty content during route transitions.

### Pitfall 4: foreignObject in SVG breaks print/export

**What goes wrong:** `<foreignObject>` with HTML content doesn't render in SVG-to-canvas exports or print stylesheets.
**Why it happens:** `<foreignObject>` is a DOM embedding mechanism, not a pure SVG construct.
**How to avoid:** This is acceptable for an interactive web app. If SVG export is ever needed, use `<tspan>` fallback. For Phase 26's scope, `<foreignObject>` is the correct choice.
**Warning signs:** Blank text areas in screenshots/exports.

### Pitfall 5: ComparisonPanel forecast fetch creates N+1 query pattern

**What goes wrong:** Each comparison entry lazy-fetches a full `ForecastResponse` when clicked. If a user expands 10 entries, that's 10 individual API calls.
**Why it happens:** `ComparisonPanelItem` doesn't include the full forecast data.
**How to avoid:** This is acceptable -- it's the same pattern as `MyForecastsPanel.loadAndRenderForecast()` (line 217-239). The forecast is cached in a Map after first fetch. Only the first expand triggers a network call. Subsequent expands use the cached data.
**Warning signs:** Slow expansion on first click.

### Pitfall 6: d3-zoom interferes with modal scrolling

**What goes wrong:** `d3.zoom()` captures wheel events on the SVG, preventing the modal from scrolling when the cursor is over the tree.
**Why it happens:** `d3.zoom()` calls `event.preventDefault()` on wheel events by default.
**How to avoid:** Use `zoomBehavior.filter()` to disable zoom on wheel when the tree fits within the viewport (no need to zoom). Or use `zoomBehavior.wheelDelta()` to reduce sensitivity. Or restrict zoom to only activate when the tree exceeds a size threshold (5+ scenarios per context decision).
**Warning signs:** User can't scroll the modal when hovering over the tree area.

### Pitfall 7: narrative_summary column migration with existing data

**What goes wrong:** Adding a new `narrative_summary` column as NOT NULL breaks existing rows.
**Why it happens:** Existing Prediction rows have no narrative data.
**How to avoid:** Make the column `nullable=True` with `server_default=None`. Existing forecasts will have `null` narrative -- the frontend handles this gracefully by not rendering the narrative section.
**Warning signs:** Alembic migration failure on `ALTER TABLE`.

## Code Examples

### Binary Market Filter
```python
# Source: Polymarket Gamma API (verified via live query 2026-03-08)
# Market dict has: outcomes: ["Yes", "No"] for binary markets

def is_binary_market(event: dict) -> bool:
    """Check if ALL markets in the event have exactly Yes/No outcomes."""
    markets = event.get("markets", [])
    if not markets:
        return False
    for market in markets:
        outcomes = market.get("outcomes")
        if not isinstance(outcomes, list) or outcomes != ["Yes", "No"]:
            return False
    return True
```

### d3 Pan/Zoom on SVG Group
```typescript
// Source: d3 v7 API (d3-zoom), bundled with d3 ^7.9.0
import { zoom as d3zoom, select } from 'd3';

// Wrap all tree content in a <g> group
const svgGroup = document.createElementNS('http://www.w3.org/2000/svg', 'g');
// ... append all nodes and links to svgGroup ...
svg.appendChild(svgGroup);

// Apply zoom behavior to the SVG, transform the group
const zoomBehavior = d3zoom<SVGSVGElement, unknown>()
  .scaleExtent([0.3, 3])
  .on('zoom', (event) => {
    svgGroup.setAttribute('transform', event.transform.toString());
  });

select(svg).call(zoomBehavior as any);
```

### foreignObject Multi-line Text in SVG
```typescript
// SVG <foreignObject> for word-wrapped text blocks
const fo = document.createElementNS('http://www.w3.org/2000/svg', 'foreignObject');
const textWidth = 160; // px
const textHeight = 60; // px (enough for 2-3 lines at ~14px font)

// Position: offset from node center, direction depends on subtree side
const xOffset = isLeftSubtree ? -(textWidth + radius + 8) : (radius + 8);
fo.setAttribute('x', String(xOffset));
fo.setAttribute('y', String(-textHeight / 2));
fo.setAttribute('width', String(textWidth));
fo.setAttribute('height', String(textHeight));

const div = document.createElement('div');
div.className = 'scenario-node-text';
div.textContent = node.data.name.slice(0, 120);
fo.appendChild(div);
g.appendChild(fo);
```

### Circuit Breaker Cache Invalidation
```typescript
// Add to CircuitBreaker class
invalidateCache(): void {
  this.cache = null;
  this.lastDataState = { mode: 'unavailable', timestamp: null };
}

// Add to ForecastServiceClient
bustAllCaches(): void {
  for (const breaker of [this.forecastBreaker, this.countryBreaker, this.healthBreaker]) {
    breaker.invalidateCache();
  }
}
```

### Router Same-Route Remount
```typescript
navigate(path: string): void {
  if (window.location.pathname === path) {
    window.history.replaceState(null, '', path);
  } else {
    window.history.pushState(null, '', path);
  }
  void this.resolve();
}

// In resolve(): remove the `route === this.currentRoute` early return.
// Instead, force remount on same-route:
async resolve(): Promise<void> {
  const path = window.location.pathname;
  const route = this.routes.find(r => r.path === path)
    ?? this.routes.find(r => r.path === '/dashboard')
    ?? this.routes[0];
  if (!route) return;

  const prevRoute = this.currentRoute;
  this.currentRoute = route;

  // Cache bust before remount
  // (forecastClient.bustAllCaches() called here or in screen mount)

  const doSwap = async (): Promise<void> => {
    if (prevRoute) prevRoute.unmount();
    this.container.innerHTML = '';
    await route.mount(this.container);
  };
  // ... view transition wrapper unchanged ...
}
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| 40-char truncated SVG `<text>` | `<foreignObject>` HTML-in-SVG with CSS word-wrap | Phase 26 | 2-3 line readable text per node |
| Fixed node spacing (200x100px) | Dynamic `d3.tree().separation()` based on text height | Phase 26 | No text overlap on dense trees |
| Static SVG canvas | `d3.zoom()` pan/zoom for 5+ scenario trees | Phase 26 | Navigable dense trees |
| Same-route click ignored | Same-route triggers full remount with cache bust | Phase 26 | No stale data after navigation |
| ComparisonPanel: dual-bar only | Expandable card with lazy-fetched forecast data | Phase 26 | Full progressive disclosure parity |

## Open Questions

1. **Narrative generation budget impact**
   - What we know: Each new forecast adds 1 Gemini call for narrative generation. Current budget is 25/day split 3 new + 5 reforecast.
   - What's unclear: Whether the narrative call should count against the Gemini daily budget or be considered "free" (like CAMEO extraction).
   - Recommendation: Count it against the budget. It's an LLM call. The per-call cost is trivial (small prompt, short response) but the accounting should be honest.

2. **Existing predictions: backfill narratives?**
   - What we know: New `narrative_summary` column will be NULL for all existing predictions.
   - What's unclear: Should we backfill narratives for existing predictions or leave them NULL?
   - Recommendation: Leave NULL. The frontend should gracefully handle NULL by not rendering the narrative section. Backfilling would consume Gemini budget for historical data with no user-facing benefit.

3. **Non-binary comparison exclusion scope**
   - What we know: Need to mark existing non-binary comparisons as `excluded`.
   - What's unclear: How many existing comparisons are non-binary? Could be zero if all matched markets happened to be binary.
   - Recommendation: Write the exclusion script, run it, and log the count. If zero, the script is still valuable as ongoing protection via the filter.

## Sources

### Primary (HIGH confidence)
- Polymarket Gamma API -- live query to `gamma-api.polymarket.com/markets?limit=1` confirming `outcomes: ["Yes", "No"]` field structure
- d3 v7 -- `frontend/package.json` confirms `"d3": "^7.9.0"`, `d3.zoom()`, `d3.tree()`, `d3.linkVertical()` all verified against existing codebase usage
- Codebase inspection -- all 15+ source files examined directly

### Secondary (MEDIUM confidence)
- [Polymarket Gamma Structure Documentation](https://docs.polymarket.com/developers/gamma-markets-api/gamma-structure)
- [Polymarket API Overview (Medium)](https://medium.com/@gwrx2005/the-polymarket-api-architecture-endpoints-and-use-cases-f1d88fa6c1bf)

### Tertiary (LOW confidence)
- None. All findings verified against codebase or live API.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- no new dependencies, all existing libraries
- Architecture (binary filter): HIGH -- verified Gamma API structure via live query
- Architecture (tree overhaul): HIGH -- d3 v7 APIs verified against existing usage in codebase
- Architecture (route refresh): HIGH -- Router, CircuitBreaker, screen lifecycle all examined
- Architecture (root node content): MEDIUM -- narrative generation approach is sound but involves new Alembic migration + Gemini prompt design
- Architecture (poller enablement): HIGH -- registry.py and settings.py confirm wiring exists
- Pitfalls: HIGH -- all based on direct codebase inspection

**Research date:** 2026-03-08
**Valid until:** 2026-04-08 (stable codebase, no external dependency changes expected)
