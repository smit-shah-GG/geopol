# Phase 18: Polymarket-Driven Forecasting - Research

**Researched:** 2026-03-04
**Domain:** Backend auto-forecast orchestration + frontend comparison UI
**Confidence:** HIGH

## Summary

Phase 18 transforms Polymarket from a passive comparison tool into an active forecast driver. The existing Phase 13 infrastructure (PolymarketClient, PolymarketMatcher, PolymarketComparisonService, DB tables, API routes, PolymarketPanel) provides a solid foundation. Phase 18 adds: (1) an auto-forecast trigger that generates Geopol predictions for unmatched high-volume Polymarket questions, (2) daily re-forecasting of active comparisons, (3) a badge system on forecast cards, (4) inline comparison data in expanded cards with sparklines, and (5) a dedicated ComparisonPanel in Col 2.

The codebase already has all the building blocks: `EnsemblePredictor.predict()` for forecast generation, `ForecastService.persist_forecast()` for persistence, `question_parser.py` patterns for LLM-based extraction, `COUNTRY_NAME_TO_ISO` dict in `advisory_poller.py` for heuristic country extraction, `polymarket_snapshots` table for time-series data, and the `expandable-card.ts` shared utility for card modification. The primary engineering challenge is orchestration -- wiring the existing components into a new background loop with volume filtering, daily caps, and deduplication.

**Primary recommendation:** Extend the existing `_polymarket_loop` in `app.py` to run auto-forecast generation after the matching cycle, reusing `EnsemblePredictor` + `ForecastService` for forecast production. Add a `provenance` column to the `Prediction` table to distinguish Polymarket-driven forecasts. Build the ComparisonPanel as a new Panel subclass in Col 2 below ForecastPanel.

## Standard Stack

### Core (already in the project)

| Library | Purpose | Phase 18 Role |
|---------|---------|---------------|
| SQLAlchemy 2.x (async) | ORM + async queries | New columns on Prediction, new queries for comparison panel |
| Alembic | Schema migrations | Migration for `provenance` column + `polymarket_event_id` on Prediction |
| aiohttp | HTTP client | Already used by PolymarketClient -- no changes needed |
| google-generativeai | Gemini LLM | CAMEO extraction + ambiguous country fallback |
| pydantic-settings | Settings | New settings: volume threshold, daily cap |
| d3 | Frontend charts | Already imported; used for mini scenario tree |

### Supporting (new for Phase 18)

| Library | Purpose | When to Use |
|---------|---------|-------------|
| N/A | SVG sparkline | Hand-roll a 30-line function -- d3 is already imported and can generate sparkline paths trivially |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Hand-rolled SVG sparkline | fnando/sparkline npm package | Extra dependency for ~30 lines of code; d3 already in bundle handles path generation |
| Extending `_polymarket_loop` | Separate background task | Would duplicate session/client setup; loop already runs hourly, just add a phase |
| `provenance` column on Prediction | Separate `polymarket_predictions` table | Separate table fragments the forecast list; column keeps them first-class |

**Installation:** No new packages needed. All dependencies already present.

## Architecture Patterns

### Recommended Project Structure (new files only)

```
src/
  polymarket/
    client.py          # Existing (no changes)
    matcher.py         # Existing (no changes)
    comparison.py      # Existing (extend with re-forecast and snapshot queries)
    auto_forecaster.py # NEW: volume filter, extraction, pipeline trigger, cap tracking
  api/
    routes/v1/
      calibration.py   # Extend: new endpoint for comparison panel + snapshot data
    schemas/
      forecast.py      # Extend: ForecastResponse gains polymarket_comparison field
  db/
    models.py          # Extend: Prediction gets provenance + polymarket_event_id columns

frontend/src/
  components/
    ComparisonPanel.ts        # NEW: dual-bar comparison panel for Col 2
    expandable-card.ts        # Extend: badge rendering + inline comparison row
  types/
    api.ts                    # Extend: new interfaces for comparison data + snapshots
  services/
    forecast-client.ts        # Extend: new API methods for comparison panel + snapshots

alembic/versions/
  YYYYMMDD_005_polymarket_provenance.py  # NEW migration
```

### Pattern 1: Auto-Forecast Orchestration (Backend)

**What:** A new `PolymarketAutoForecaster` class that runs inside the existing `_polymarket_loop`. After the matching cycle identifies unmatched geo events, the auto-forecaster filters by volume threshold, extracts pipeline parameters (country, horizon, CAMEO), runs `EnsemblePredictor.predict()`, persists via `ForecastService`, and creates the comparison row.

**When to use:** Every polling cycle (hourly), after `run_matching_cycle()`.

**Example:**
```python
# src/polymarket/auto_forecaster.py
class PolymarketAutoForecaster:
    """Generate Geopol forecasts for unmatched high-volume Polymarket questions."""

    def __init__(
        self,
        session_factory: Callable,
        gemini_client: GeminiClient,
        settings: Settings,
    ) -> None:
        self._session_factory = session_factory
        self._gemini = gemini_client
        self._volume_threshold = settings.polymarket_volume_threshold
        self._daily_cap = settings.polymarket_daily_forecast_cap

    async def run(self, geo_events: list[dict]) -> dict[str, int]:
        """Filter unmatched events by volume, extract params, generate forecasts.

        Returns: {candidates: int, generated: int, skipped_cap: int}
        """
        # 1. Filter to unmatched events above volume threshold
        # 2. Sort by volume DESC, apply daily cap
        # 3. For each candidate:
        #    a. Extract country (heuristic -> LLM fallback)
        #    b. Extract CAMEO category (always LLM)
        #    c. Compute horizon from endDate
        #    d. Run EnsemblePredictor.predict() via asyncio.to_thread
        #    e. Persist via ForecastService with provenance="polymarket_driven"
        #    f. Create PolymarketComparison row
        ...
```

### Pattern 2: Tiered Country Extraction

**What:** Two-phase extraction reusing existing infrastructure.

**When to use:** For every Polymarket question before pipeline execution.

**Example:**
```python
# Heuristic phase: reuse advisory_poller's COUNTRY_NAME_TO_ISO dict
from src.ingest.advisory_poller import COUNTRY_NAME_TO_ISO

def extract_country_heuristic(title: str, tags: list[dict]) -> str | None:
    """Check title and tag labels against ~200 country names."""
    text = title.lower()
    for tag in tags:
        label = tag.get("label", "")
        if isinstance(label, str):
            text += " " + label.lower()

    for name, iso in COUNTRY_NAME_TO_ISO.items():
        if name in text:
            return iso
    return None  # Falls back to LLM

# LLM fallback: reuse question_parser.py pattern
async def extract_country_llm(title: str, description: str) -> str:
    """Ask Gemini to infer the most relevant country ISO code."""
    # Similar to question_parser._call_gemini_parser but simpler prompt
    ...
```

### Pattern 3: Badge System on Expandable Cards

**What:** Add a small Polymarket icon to the collapsed card header when the forecast has a linked comparison. The badge is data-driven: the backend includes `polymarket_comparison` data in the ForecastResponse, and the frontend conditionally renders the badge.

**When to use:** On every card render in `buildExpandableCard()`.

**Example:**
```typescript
// In expandable-card.ts buildExpandableCard():
const pmData = f.polymarket_comparison;
const badge = pmData
  ? h('span', {
      className: 'pm-badge',
      title: pmData.provenance === 'polymarket_driven'
        ? 'Polymarket-driven forecast'
        : 'Polymarket-tracked forecast',
    })
  : null;

// Insert badge into forecast-meta row
```

### Pattern 4: Inline Comparison Row in Expanded Card

**What:** When a badged card is expanded, inject a comparison row showing: Polymarket price, divergence, and SVG sparkline of both probabilities over time.

**When to use:** In `buildExpandedContent()` when `f.polymarket_comparison` is present.

**Example:**
```typescript
// In expandable-card.ts buildExpandedContent():
if (f.polymarket_comparison) {
  const compRow = buildComparisonRow(f.polymarket_comparison);
  leftCol.appendChild(compRow);
}

function buildComparisonRow(comp: PolymarketComparisonData): HTMLElement {
  const divergence = comp.geopol_probability - comp.polymarket_price;
  const divergenceClass = Math.abs(divergence) > 0.15 ? 'high-divergence' : 'low-divergence';

  return h('div', { className: 'expanded-comparison-section' },
    h('div', { className: 'expanded-section-label' }, 'vs Polymarket'),
    h('div', { className: 'comparison-values' },
      h('span', {}, `Market: ${(comp.polymarket_price * 100).toFixed(1)}%`),
      h('span', { className: divergenceClass },
        `${divergence > 0 ? '+' : ''}${(divergence * 100).toFixed(1)}pp`),
    ),
    renderSparkline(comp.snapshots),  // SVG sparkline
  );
}
```

### Pattern 5: ComparisonPanel (Col 2, below Active Forecasts)

**What:** A new Panel subclass showing all active + resolved comparisons as dual probability bars with 5-minute auto-refresh.

**When to use:** Mounted in dashboard-screen.ts Col 2, after ForecastPanel.

**Example:**
```typescript
export class ComparisonPanel extends Panel {
  constructor() {
    super({ id: 'comparisons', title: 'POLYMARKET COMPARISONS', showCount: true });
  }

  public update(data: ComparisonPanelResponse): void {
    // Render list of comparison entries with dual bars
    // Each entry: title, Geopol bar, Polymarket bar, divergence indicator
    // Resolved entries: inline with "Resolved" badge + correct/wrong indicator
  }
}
```

### Anti-Patterns to Avoid

- **Separate forecast list for Polymarket forecasts:** Fragments the UI. Polymarket-driven forecasts MUST appear in the main Active Forecasts list (with badge).
- **Running EnsemblePredictor synchronously in the event loop:** `predict()` is CPU-bound and synchronous. ALWAYS wrap in `asyncio.to_thread()`.
- **Creating a new GeminiClient per forecast:** The client has rate limiting built in. Reuse the same instance across the entire cycle.
- **Polling Polymarket per-market for prices in the auto-forecast loop:** Use the bulk event data from `fetch_geopolitical_markets()` which already includes market prices. Only call `fetch_event_prices()` for snapshot capture.
- **Storing sparkline data client-side:** The `polymarket_snapshots` table already captures time-series data. Serve it from the API, don't accumulate it in frontend state.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Country name -> ISO mapping | Custom dict | `COUNTRY_NAME_TO_ISO` from `advisory_poller.py` | Already ~200 entries, tested, maintained |
| CAMEO category extraction | Regex pattern matching | Gemini LLM via `GeminiClient.generate_content()` | Per-CAMEO calibration weights require accurate categorization; LLM handles nuance |
| Forecast pipeline execution | Stripped-down binary predictor | Full `EnsemblePredictor.predict()` + `ForecastService.persist_forecast()` | Standard scenario-tree output is decided; reuse existing infrastructure |
| Budget tracking | Custom counter | Redis-backed `gemini_budget_remaining()` + `increment_gemini_usage()` | Already exists in `submission_worker.py`, handles rate limiting |
| Question parsing/validation | Custom validator | Pattern from `question_parser._validate_parsed_result()` | Already handles malformed LLM output gracefully |
| Sparkline SVG generation | npm sparkline library | `d3.line()` generator (already in bundle) | d3 is already imported for mini scenario trees; `d3.line()` produces SVG path strings in ~10 lines |

**Key insight:** Phase 18 is an orchestration phase, not an infrastructure phase. Every component needed already exists -- the work is wiring them together with new trigger logic and UI rendering.

## Common Pitfalls

### Pitfall 1: Double-forecasting the same Polymarket question across cycles
**What goes wrong:** The hourly polling loop fetches geo events, finds an unmatched one above volume threshold, generates a forecast, but the next cycle doesn't know a forecast was already generated (race condition or missing dedup check).
**Why it happens:** `run_matching_cycle()` checks `tracked_ids` (polymarket_event_id in comparisons table), but if the auto-forecaster creates the comparison row and the Prediction row in the same transaction, a failure between forecast generation and comparison creation leaves orphans.
**How to avoid:** Check for existing `PolymarketComparison` rows by `polymarket_event_id` BEFORE triggering the pipeline. Use `SELECT FOR UPDATE SKIP LOCKED` pattern (already used in `submission_worker.py`) if concurrency is a concern. Additionally, add a `polymarket_event_id` column to Prediction for direct dedup lookup.
**Warning signs:** Duplicate predictions for the same question in the forecasts list.

### Pitfall 2: Exhausting Gemini budget with Polymarket-triggered forecasts
**What goes wrong:** The daily cap (5) seems safe, but each forecast requires ~2-3 Gemini calls (CAMEO extraction + scenario generation + optional country fallback). Combined with the daily pipeline's own question generation and user submissions, total daily Gemini usage exceeds the 25-call budget.
**Why it happens:** Three independent consumers share the same Gemini budget: daily pipeline, user submissions, and now Polymarket auto-forecaster.
**How to avoid:** Check `gemini_budget_remaining()` before each auto-forecast, same as `submission_worker.py` does. Count auto-forecasts against the same daily budget. Consider a sub-budget (e.g., 5 reserved for Polymarket out of 25 total).
**Warning signs:** `BudgetExhaustedError` in daily pipeline logs, user submissions failing.

### Pitfall 3: Stale sparkline data due to infrequent snapshots
**What goes wrong:** The sparkline shows only 2-3 data points because the existing snapshot capture runs hourly (inside `_polymarket_loop`). A 30-day forecast horizon with hourly snapshots produces ~720 points, but the initial implementation might capture fewer.
**Why it happens:** The existing `capture_snapshots()` only runs inside the polling loop. If re-forecasting only happens daily, the Geopol probability in snapshots only changes daily while Polymarket price changes hourly.
**How to avoid:** Keep the existing hourly snapshot capture (Polymarket price updates hourly) but understand that the Geopol probability line in the sparkline will be step-shaped (changes daily when re-forecast runs). This is fine -- it accurately reflects how often Geopol updates its estimate.
**Warning signs:** Sparklines with flat Geopol lines and jumpy Polymarket lines (this is expected, not a bug).

### Pitfall 4: ForecastResponse bloat from embedding comparison data
**What goes wrong:** Adding `polymarket_comparison` with snapshot arrays to every ForecastResponse makes the `/forecasts/top` response significantly larger, slowing the 60-second refresh cycle.
**Why it happens:** Snapshots accumulate over time (720+ per comparison over 30 days).
**How to avoid:** Two strategies: (1) For the top forecasts list, include only current comparison metadata (price, divergence, badge type) -- no snapshots. (2) Load snapshots lazily when a card is expanded, via a dedicated API endpoint (e.g., `GET /calibration/polymarket/comparisons/{id}/snapshots`). This keeps the list response small and loads sparkline data on demand.
**Warning signs:** Increasing payload size on `/forecasts/top` response.

### Pitfall 5: Volume threshold too low catches low-quality questions
**What goes wrong:** Setting the volume threshold too low triggers forecasts on illiquid, poorly-defined Polymarket questions that the pipeline can't meaningfully answer.
**Why it happens:** Polymarket geopolitical markets span from $100K to $500M+ volume. The long tail has many low-quality or hyper-specific questions.
**How to avoid:** Based on current Polymarket data (March 2026), geopolitical events above ~$100K volume are typically well-defined and significant. Recommend a default threshold of $100,000. The setting should be configurable via `Settings.polymarket_volume_threshold`.
**Warning signs:** Auto-forecasts for obscure questions nobody cares about; wasted API budget.

### Pitfall 6: endDate parsing edge cases
**What goes wrong:** Polymarket `endDate` is ISO 8601 (`"2024-11-05T12:00:00Z"`), but some events have distant or missing end dates (e.g., "Will X happen before 2030?").
**Why it happens:** Not all Polymarket events have short horizons relevant to Geopol's 7-365 day range.
**How to avoid:** Clamp parsed horizon to 7-365 days (matching `question_parser._validate_parsed_result()`). If `endDate` is >365 days out, skip the event -- Geopol's pipeline isn't calibrated for multi-year forecasting.
**Warning signs:** Forecasts with 1000+ day horizons appearing in the dashboard.

## Code Examples

### Horizon Computation from Polymarket endDate

```python
from datetime import datetime, timezone

def compute_horizon_days(end_date_str: str | None) -> int | None:
    """Compute forecast horizon from Polymarket event endDate.

    Returns None if endDate is missing, unparseable, or outside valid range.
    Valid range: 7-365 days from now.
    """
    if not end_date_str:
        return None

    try:
        end_date = datetime.fromisoformat(end_date_str.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None

    now = datetime.now(timezone.utc)
    delta_days = (end_date - now).days

    if delta_days < 7 or delta_days > 365:
        return None  # Outside Geopol's calibrated range

    return delta_days
```

### CAMEO Extraction Prompt

```python
_CAMEO_EXTRACTION_PROMPT = """Classify this prediction market question into a CAMEO event category.

QUESTION: {question}

Return ONLY a JSON object:
{{"cameo_root_code": "<2-digit CAMEO code>", "category": "<one of: conflict, diplomatic, economic, security, political>"}}

CAMEO codes:
- 01-05: Verbal/material cooperation (diplomatic)
- 06-09: Verbal/material conflict (security)
- 10-14: Demands, protests, sanctions (political/economic)
- 15-17: Military action (conflict)
- 18-20: Physical assault, mass violence (conflict)

If ambiguous, choose the most likely category. Return ONLY JSON."""
```

### SVG Sparkline Renderer (d3-based, zero new dependencies)

```typescript
import * as d3 from 'd3';
import { h } from '@/utils/dom-utils';

interface SparklinePoint { polymarket_price: number; geopol_probability: number; captured_at: string; }

function renderSparkline(snapshots: SparklinePoint[]): HTMLElement {
  if (snapshots.length < 2) {
    return h('div', { className: 'sparkline-empty' }, 'Insufficient data');
  }

  const width = 180;
  const height = 40;
  const pad = 2;

  const xScale = d3.scaleLinear()
    .domain([0, snapshots.length - 1])
    .range([pad, width - pad]);

  const yScale = d3.scaleLinear()
    .domain([0, 1])
    .range([height - pad, pad]);

  const pmLine = d3.line<SparklinePoint>()
    .x((_, i) => xScale(i))
    .y(d => yScale(d.polymarket_price));

  const gpLine = d3.line<SparklinePoint>()
    .x((_, i) => xScale(i))
    .y(d => yScale(d.geopol_probability));

  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
  svg.setAttribute('width', '100%');
  svg.setAttribute('height', String(height));
  svg.classList.add('sparkline-svg');

  // Polymarket line (blue)
  const pmPath = document.createElementNS('http://www.w3.org/2000/svg', 'path');
  pmPath.setAttribute('d', pmLine(snapshots) ?? '');
  pmPath.setAttribute('fill', 'none');
  pmPath.setAttribute('stroke', 'var(--accent)');
  pmPath.setAttribute('stroke-width', '1.5');
  svg.appendChild(pmPath);

  // Geopol line (orange/critical)
  const gpPath = document.createElementNS('http://www.w3.org/2000/svg', 'path');
  gpPath.setAttribute('d', gpLine(snapshots) ?? '');
  gpPath.setAttribute('fill', 'none');
  gpPath.setAttribute('stroke', 'var(--semantic-critical)');
  gpPath.setAttribute('stroke-width', '1.5');
  svg.appendChild(gpPath);

  return h('div', { className: 'sparkline-container' }, svg);
}
```

### Daily Cap Tracking

```python
from datetime import datetime, timezone
from sqlalchemy import func, select

async def count_today_auto_forecasts(session: AsyncSession) -> int:
    """Count Polymarket-driven forecasts generated today (UTC)."""
    today_start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    stmt = select(func.count()).where(
        Prediction.provenance == "polymarket_driven",
        Prediction.created_at >= today_start,
    ).select_from(Prediction)
    result = await session.execute(stmt)
    return result.scalar() or 0
```

### Dual Probability Bars (ComparisonPanel entry)

```typescript
function buildComparisonEntry(comp: ComparisonItem): HTMLElement {
  const divergence = Math.abs(comp.geopol_probability - comp.polymarket_price);
  const divClass = divergence > 0.2 ? 'div-high' : divergence > 0.1 ? 'div-medium' : 'div-low';

  const isResolved = comp.status === 'resolved';

  return h('div', { className: `comparison-entry ${isResolved ? 'resolved' : ''}` },
    h('div', { className: 'comparison-title' },
      truncate(comp.polymarket_title, 50),
      isResolved ? h('span', { className: 'resolved-badge' }, 'RESOLVED') : null,
    ),
    h('div', { className: 'comparison-bars' },
      h('div', { className: 'bar-row' },
        h('span', { className: 'bar-label' }, 'GP'),
        h('div', { className: 'bar-track' },
          h('div', { className: `bar-fill bar-geopol ${divClass}`, style: `width:${comp.geopol_probability * 100}%` }),
        ),
        h('span', { className: 'bar-value mono' }, `${(comp.geopol_probability * 100).toFixed(0)}%`),
      ),
      h('div', { className: 'bar-row' },
        h('span', { className: 'bar-label' }, 'PM'),
        h('div', { className: 'bar-track' },
          h('div', { className: `bar-fill bar-polymarket ${divClass}`, style: `width:${comp.polymarket_price * 100}%` }),
        ),
        h('span', { className: 'bar-value mono' }, `${(comp.polymarket_price * 100).toFixed(0)}%`),
      ),
    ),
  );
}
```

## State of the Art

| Old Approach (Phase 13) | Current Approach (Phase 18) | What Changed | Impact |
|---|---|---|---|
| Passive matching: wait for organic Geopol predictions to match markets | Active forecasting: detect unmatched markets and auto-generate predictions | Phase 18 scope | Geopol produces forecasts for questions the market cares about |
| No visual indication on forecast cards | Badge system (driven/tracked) | Phase 18 scope | Users see which forecasts have market counterparts |
| Comparison data only in Cal 4 (top events table) | Inline in expanded cards + dedicated Col 2 panel | Phase 18 scope | Head-to-head data is immediately accessible |
| Static comparison snapshot | Daily re-forecasting + sparkline time-series | Phase 18 scope | Divergence tracking becomes meaningful over time |

**Deprecated/outdated:**
- The existing `PolymarketPanel` in Col 4 remains as-is (top events by volume). Phase 18 adds a ComparisonPanel in Col 2 focused on head-to-head comparisons. These are complementary, not redundant.

## Open Questions

1. **Re-forecast update strategy: append vs overwrite**
   - What we know: The `Prediction` table stores one row per forecast. Re-forecasting the same question creates a new probability value.
   - What's unclear: Should the re-forecast UPDATE the existing Prediction row's probability (overwrite) or CREATE a new Prediction row (append)?
   - Recommendation: **Overwrite** the existing row (update `probability`, `scenarios_json`, `ensemble_info_json`, `calibration_json`, `created_at`). Rationale: the card in Active Forecasts should show the latest estimate. Historical probability values are already captured in `polymarket_snapshots` (the `geopol_probability` column captures each snapshot). Creating new rows would flood the forecast list with duplicate questions.

2. **Daily cap: per-cycle or rolling 24-hour**
   - What we know: Settings defines `polymarket_daily_forecast_cap = 5`.
   - What's unclear: "Daily" could mean midnight-to-midnight UTC or rolling 24h.
   - Recommendation: **Midnight-to-midnight UTC** (per-day). Simpler query (`WHERE created_at >= today_start`), matches the daily pipeline's concept of a "day", and avoids edge cases where the rolling window straddles two calendar days.

3. **Whether re-forecasts count against the daily cap**
   - What we know: CONTEXT.md states "Re-forecasts count against the daily cap."
   - What's unclear: With 5 active comparisons needing daily re-forecast + allowing new forecasts, 5 cap may be too low.
   - Recommendation: Use a split cap: `daily_new_forecast_cap = 3` and `daily_reforecast_cap = 5`. This ensures active comparisons always get re-forecasted while still allowing new discoveries. Total Gemini usage is bounded at ~8 forecast calls/day (well within budget).

4. **Snapshot data volume for sparklines**
   - What we know: Hourly snapshots over 30 days = ~720 points per comparison.
   - What's unclear: How many points should the sparkline API return?
   - Recommendation: Return last 30 data points (sampled daily from hourly snapshots, or thin to every 24th point). 30 points render well in a 180px sparkline.

## Sources

### Primary (HIGH confidence)
- Polymarket Gamma API (live): `https://gamma-api.polymarket.com/events` -- verified event schema (endDate, volume, tags, markets array with outcomePrices)
- Existing codebase: `src/polymarket/client.py`, `matcher.py`, `comparison.py` -- verified all existing infrastructure
- Existing codebase: `src/api/services/submission_worker.py` -- verified forecast execution pattern
- Existing codebase: `src/pipeline/daily_forecast.py` -- verified daily pipeline budget management
- Existing codebase: `src/ingest/advisory_poller.py` -- verified COUNTRY_NAME_TO_ISO dict (~200 entries)
- Existing codebase: `src/api/services/question_parser.py` -- verified LLM extraction pattern
- Existing codebase: `frontend/src/components/expandable-card.ts` -- verified card rendering architecture
- [Polymarket Gamma API Events docs](https://docs.polymarket.com/developers/gamma-markets-api/get-events) -- confirmed endDate ISO 8601 format

### Secondary (MEDIUM confidence)
- [Polymarket geopolitical volume data](https://www.crowdfundinsider.com/2026/03/264714-polymarket-achieves-new-trading-volume-milestones-amid-geopolitical-tensions/) -- informed volume threshold recommendation
- [Polymarket $529M Iran crisis volume](https://themarketperiodical.com/2026/03/03/polymarket-hits-529m-in-betting-on-iran-crisis-after-u-s-israel-strikes/) -- validated that high-volume geopolitical markets exist
- [SVG sparkline approaches](https://alexplescan.com/posts/2023/07/08/easy-svg-sparklines/) -- confirmed hand-rolled SVG sparklines are standard practice

### Tertiary (LOW confidence)
- Volume threshold value ($100K) is a heuristic based on general Polymarket volume distribution. Actual geopolitical event volume ranges should be validated against live API data before finalizing.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- all components already exist in the codebase
- Architecture: HIGH -- patterns derived directly from existing code (submission_worker, daily_pipeline, expandable-card)
- Pitfalls: HIGH -- identified through codebase analysis of actual integration points
- Volume threshold: LOW -- heuristic, needs empirical validation against live Polymarket data
- Sparkline rendering: HIGH -- d3 already in bundle, SVG path generation is well-understood

**Research date:** 2026-03-04
**Valid until:** 2026-04-04 (stable domain, no fast-moving dependencies)
