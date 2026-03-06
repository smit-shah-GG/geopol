# Phase 22: Polymarket Hardening - Research

**Researched:** 2026-03-06
**Domain:** Polymarket integration hardening, accuracy tracking, admin panel
**Confidence:** HIGH

## Summary

This phase targets five areas: (1) fixing the `reforecast_active()` bug that overwrites `Prediction.created_at` and corrupts daily cap tracking, (2) implementing cumulative Brier score tracking with a `polymarket_accuracy` table, (3) building an admin head-to-head accuracy panel, (4) handling voided/cancelled Polymarket market resolutions, and (5) hardening the Polymarket poller with exponential backoff and graceful degradation.

The existing codebase is well-structured for these changes. The auto_forecaster.py bug is a single line (line 601: `existing.created_at = datetime.now(timezone.utc)`). The fix is straightforward: add a `reforecasted_at` column to `Prediction` and stop overwriting `created_at`. The `count_today_reforecasts()` function (line 190-212) is also broken -- it counts predictions with `created_at >= today`, which after the bug fix will no longer track reforecasts at all. It must be rewritten to use the new `reforecasted_at` column.

The admin panel framework uses vanilla TypeScript with a `h()` DOM builder utility, class-based panels implementing `AdminPanel { mount(), destroy() }`, and an `AdminClient` for typed API calls. The sidebar navigation currently has 4 sections; adding "Accuracy" requires touching `admin-types.ts` (AdminSection union), `AdminSidebar.ts` (NAV_ITEMS array), and `admin-layout.ts` (createPanel switch). The existing SourceManager panel provides the closest template for a data-rich panel with auto-refresh.

**Primary recommendation:** Fix the created_at bug first (POLY-01), then layer accuracy tracking (POLY-02/04) and the admin panel (POLY-03) on top, with poller hardening (POLY-05) as the final independent track.

## Standard Stack

No new libraries needed. All work uses existing project dependencies.

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| SQLAlchemy 2.0 | (existing) | ORM models, async queries | Already used throughout |
| Alembic | (existing) | Schema migrations | Already used for all DB changes |
| FastAPI | (existing) | Admin API endpoints | Already used for all routes |
| aiohttp | (existing) | Polymarket Gamma API client | Already used in client.py |
| tenacity | (existing) | Retry with exponential backoff | Already imported in client.py |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| Pydantic | (existing) | Admin DTOs, route schemas | API response models |
| APScheduler 3.x | (existing) | Polymarket poller scheduling | Job wrappers |

## Architecture Patterns

### Existing File Layout
```
src/polymarket/
  auto_forecaster.py    # PolymarketAutoForecaster (run + reforecast_active)
  client.py             # PolymarketClient (Gamma API, circuit breaker)
  comparison.py         # PolymarketComparisonService (matching, snapshots, resolution)
  matcher.py            # PolymarketMatcher (keyword + LLM matching)

src/api/routes/v1/
  calibration.py        # Existing polymarket comparison endpoints
  admin.py              # Admin endpoints (processes, config, logs, sources, feeds)

src/api/schemas/
  admin.py              # Admin Pydantic DTOs

src/api/services/
  admin_service.py      # Admin business logic

frontend/src/admin/
  admin-layout.ts       # Section routing (4 sections currently)
  admin-client.ts       # Typed fetch wrapper
  admin-types.ts        # AdminSection type + interfaces
  components/AdminSidebar.ts  # Sidebar nav
  panels/ProcessTable.ts      # Existing panel example
  panels/SourceManager.ts     # Complex panel with cards
  panels/ConfigEditor.ts      # Settings panel
  panels/LogViewer.ts         # Logs panel
```

### Pattern 1: Admin Panel Registration
**What:** Adding a new admin panel requires changes in 4 files (backend + frontend).
**When to use:** POLY-03 (head-to-head accuracy panel).

Backend:
1. Add new admin endpoint(s) to `src/api/routes/v1/admin.py` (or dedicated route file)
2. Add Pydantic DTOs to `src/api/schemas/admin.py`
3. Add business logic to `src/api/services/admin_service.py`

Frontend:
1. Add section to `AdminSection` union in `admin-types.ts` (`'accuracy'`)
2. Add nav item to `NAV_ITEMS` array in `AdminSidebar.ts`
3. Add case to `createPanel()` switch in `admin-layout.ts`
4. Create panel class implementing `AdminPanel` in `admin/panels/AccuracyPanel.ts`
5. Add client method(s) to `AdminClient`

### Pattern 2: DB Schema Extension via Alembic
**What:** All schema changes go through Alembic migrations.
**When to use:** Adding `reforecasted_at` to predictions, adding `polymarket_accuracy` table, adding `voided` status.

Migration naming convention: `{date}_{sequence}_{description}.py`
Latest migration: `20260305_007_rss_feeds.py` (revision "007")
Next migration: revision "008", down_revision "007"

### Pattern 3: Polymarket Cycle Execution
**What:** The polymarket cycle runs in a ProcessPoolExecutor worker via `heavy_runner.py:run_polymarket_cycle()`.
**When to use:** Understanding where new accuracy/resolution logic hooks in.

Execution order in `run_polymarket_cycle()`:
1. `service.run_matching_cycle()` -- match events to predictions
2. `service.capture_snapshots()` -- price/probability snapshots
3. `auto_forecaster.run()` -- new forecasts for unmatched questions
4. `auto_forecaster.reforecast_active()` -- re-forecast existing (once/day)

New hooks needed:
- After step 2: `service.resolve_completed()` already exists but needs enhancement for voided markets
- After resolution: compute and persist cumulative accuracy scores

### Pattern 4: ComparisonService Session Factory Wrapping
**What:** API route handlers wrap the injected DB session in a fake session factory for PolymarketComparisonService.
**When to use:** Existing pattern in calibration.py routes.

```python
from contextlib import asynccontextmanager

@asynccontextmanager
async def _session_wrapper():
    yield db

service = PolymarketComparisonService.__new__(PolymarketComparisonService)
service._session_factory = _session_wrapper
```

This pattern is used because `PolymarketComparisonService` expects a session factory (for background jobs) but API routes get scoped sessions from FastAPI DI. The planner should note this for new accuracy query methods.

### Anti-Patterns to Avoid
- **Overwriting immutable audit fields:** The core bug. `created_at` must be treated as immutable once set. Reforecast timestamps go in a separate column.
- **Using `created_at` for activity tracking:** The `count_today_reforecasts()` function proxies reforecast activity via `created_at >= today`, which is fragile. Use a dedicated `reforecasted_at` column.
- **Resolving from price convergence:** The existing `_extract_outcome()` uses price thresholds (>0.95, <0.05) which can produce false resolutions. Gamma API has explicit `closed`, `resolutionSource`, and `automaticallyResolved` fields.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Retry with backoff | Custom retry loop | `tenacity` decorators | Already in client.py, handles jitter, max attempts |
| Circuit breaker | Custom state machine | Existing `PolymarketClient` circuit breaker | Already implemented with half-open probes |
| Brier score formula | Complex statistical library | `(forecast - outcome) ** 2` | Already correct in comparison.py |
| Rolling window aggregation | Materialized view | SQL `WHERE resolved_at >= now - 30d` | Simple enough for on-read compute at this scale |

**Key insight:** The existing codebase already has circuit breaker + retry infrastructure in `client.py`. POLY-05 is about tuning and extending what exists, not building new retry mechanisms.

## Common Pitfalls

### Pitfall 1: count_today_reforecasts() Will Break After Bug Fix
**What goes wrong:** After fixing `created_at` overwrite, `count_today_reforecasts()` returns 0 forever because it relies on `Prediction.created_at >= today` to detect reforecasts.
**Why it happens:** The function was written around the bug as a feature -- it counted predictions whose `created_at` was today because the bug was setting `created_at` to now on reforecast.
**How to avoid:** Rewrite `count_today_reforecasts()` to query `Prediction.reforecasted_at >= today` for predictions with polymarket provenance. This must be done in the same task as the bug fix.
**Warning signs:** After deploying the fix, reforecast cap never triggers (all 5 daily reforecasts allowed).

### Pitfall 2: Brier Score Denominator -- Cumulative vs Average
**What goes wrong:** Mixing up cumulative Brier score (mean of per-question scores) with sum-of-squares.
**Why it happens:** "Cumulative" can mean running sum or running average.
**How to avoid:** CONTEXT.md specifies global cumulative Brier = mean of all per-question Brier scores. Store individual per-comparison Brier scores (already on `PolymarketComparison.geopol_brier` and `.polymarket_brier`) and compute the mean on-read via SQL `AVG()`. The `polymarket_accuracy` table stores the precomputed aggregate updated after each resolution.
**Warning signs:** Brier score > 1.0 (means you summed instead of averaged).

### Pitfall 3: Race Between Resolution Detection and Reforecast
**What goes wrong:** If `reforecast_active()` runs after a market resolves but before `resolve_completed()`, it overwrites the last probability on a now-resolved comparison.
**Why it happens:** The `run_polymarket_cycle()` runs reforecast (step 4) after snapshots (step 2) but resolution checking also happens in step 2 flow.
**How to avoid:** Run `resolve_completed()` BEFORE `reforecast_active()` in the cycle. Resolution should transition comparisons to "resolved" status, and `reforecast_active()` only queries `status == "active"` comparisons.
**Warning signs:** Geopol Brier score computed from a post-resolution reforecast (probability ~1.0 or ~0.0 after market resolves).

### Pitfall 4: Voided Market Detection False Positives
**What goes wrong:** Marking active markets as voided because their price is exactly 0.5 or because `resolutionSource` is empty.
**Why it happens:** The Gamma API field `resolutionSource` may be empty string on active markets.
**How to avoid:** Only check resolution fields when `closed == true`. A voided market is: `closed=true AND (outcomePrices near [0.5, 0.5] OR resolutionSource contains "void")`. The conservative approach: require `closed=true` before any resolution logic.
**Warning signs:** Active markets suddenly appearing as "voided" in the admin panel.

### Pitfall 5: Admin Panel State Loss on Auto-Refresh
**What goes wrong:** Table sort order or scroll position resets every refresh cycle.
**Why it happens:** `innerHTML = ''` destroys DOM state.
**How to avoid:** Use the SourceManager pattern: maintain refs to DOM nodes and update text content in-place. Only rebuild the full DOM when the row set changes (new/removed entries).
**Warning signs:** User sorts table by Brier score, table jumps back to default sort after 15s.

### Pitfall 6: ProcessPoolExecutor Context Loss
**What goes wrong:** New code imported inside `run_polymarket_cycle()` fails because the subprocess doesn't have the same module-level state.
**Why it happens:** `ProcessPoolExecutor` forks a new process -- module-level singletons (settings, DB engine) are not shared.
**How to avoid:** Follow the existing pattern in `heavy_runner.py`: call `init_db()` inside the async function, re-import `get_settings()`, create fresh client instances. Any new accuracy persistence logic must work within this subprocess context.
**Warning signs:** `async_session_factory is None` errors in production logs.

## Code Examples

### Bug Fix: reforecast_active() created_at Overwrite
```python
# BEFORE (line 601 of auto_forecaster.py) -- THE BUG:
existing.created_at = datetime.now(timezone.utc)

# AFTER:
existing.reforecasted_at = datetime.now(timezone.utc)
# Do NOT touch existing.created_at
```

### New Column: reforecasted_at on Prediction
```python
# In src/db/models.py, add to Prediction class:
reforecasted_at: Mapped[Optional[datetime]] = mapped_column(
    DateTime(timezone=True), nullable=True, index=True
)
```

### Fixed count_today_reforecasts()
```python
async def count_today_reforecasts(session: AsyncSession) -> int:
    """Count reforecasts done today using the reforecasted_at column."""
    today_start = datetime.now(timezone.utc).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    stmt = (
        select(func.count())
        .where(
            Prediction.provenance.in_(["polymarket_driven", "polymarket_tracked"]),
            Prediction.reforecasted_at >= today_start,
        )
        .select_from(Prediction)
    )
    result = await session.execute(stmt)
    return result.scalar() or 0
```

### polymarket_accuracy Table Schema
```python
class PolymarketAccuracy(Base):
    """Cumulative accuracy metrics: Geopol vs Polymarket."""

    __tablename__ = "polymarket_accuracy"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    # Snapshot of cumulative metrics at this point in time
    total_resolved: Mapped[int] = mapped_column(Integer, nullable=False)
    geopol_cumulative_brier: Mapped[float] = mapped_column(Float, nullable=False)
    polymarket_cumulative_brier: Mapped[float] = mapped_column(Float, nullable=False)
    geopol_wins: Mapped[int] = mapped_column(Integer, nullable=False)
    polymarket_wins: Mapped[int] = mapped_column(Integer, nullable=False)
    draws: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    # Rolling 30-day window
    rolling_30d_geopol_brier: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    rolling_30d_polymarket_brier: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    rolling_30d_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    # Metadata
    computed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )
    # FK to the comparison that triggered this recompute
    triggered_by_comparison_id: Mapped[Optional[int]] = mapped_column(
        Integer, nullable=True
    )
```

### Voided Status on PolymarketComparison
```python
# Extend status values: active | resolved | voided
# In PolymarketComparison model, status column already allows any string(20)
# No model change needed, just new status value "voided"
```

### Enhanced Resolution Detection (Gamma API Fields)
```python
# In comparison.py resolve_completed(), use Gamma API fields:
async def resolve_completed(self) -> int:
    for comp in comparisons:
        # Fetch full event data (not just prices)
        event_data = await self._client._get_json(
            f"{self._client.GAMMA_API_BASE}/events/{comp.polymarket_event_id}"
        )
        markets = event_data.get("markets", [])
        for market in markets:
            if not market.get("closed"):
                continue  # Only process closed markets

            # Check for voided/ambiguous resolution
            resolution_source = market.get("resolutionSource", "")
            if "void" in resolution_source.lower() or _is_ambiguous_resolution(market):
                comp.status = "voided"
                comp.resolved_at = _utcnow()
                break

            # Normal resolution
            outcome = self._extract_outcome(market)
            if outcome is not None:
                # ... existing Brier score logic ...
                break
```

### Admin Panel Class Pattern
```typescript
// In frontend/src/admin/panels/AccuracyPanel.ts
import type { AdminPanel } from '@/admin/panels/ProcessTable';
import type { AdminClient } from '@/admin/admin-client';

export class AccuracyPanel implements AdminPanel {
  private el: HTMLElement | null = null;
  private intervalId: ReturnType<typeof setInterval> | null = null;

  constructor(private readonly client: AdminClient) {}

  async mount(container: HTMLElement): Promise<void> {
    this.el = h('div', { className: 'accuracy-panel' });
    container.appendChild(this.el);
    await this.refresh();
    this.intervalId = setInterval(() => { void this.refresh(); }, 30_000);
  }

  destroy(): void {
    if (this.intervalId !== null) clearInterval(this.intervalId);
    this.el?.remove();
    this.el = null;
  }

  private async refresh(): Promise<void> {
    const data = await this.client.getAccuracy();
    this.render(data);
  }
}
```

### Admin Sidebar Registration
```typescript
// In admin-types.ts, extend the union:
export type AdminSection = 'processes' | 'config' | 'logs' | 'sources' | 'accuracy';

// In AdminSidebar.ts, add to NAV_ITEMS:
{ section: 'accuracy', label: 'Accuracy', icon: '\u2630' }  // trigram

// In admin-layout.ts, add to SECTION_TITLES and createPanel():
accuracy: 'POLYMARKET ACCURACY',
// ...
case 'accuracy': return new AccuracyPanel(client);
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Overwrite `created_at` on reforecast | Add `reforecasted_at` column | Phase 22 | Preserves audit trail, fixes cap tracking |
| Price convergence for resolution | Gamma API `closed` + `resolutionSource` | Phase 22 | Eliminates false resolutions |
| No accuracy tracking | `polymarket_accuracy` table + rolling window | Phase 22 | Rigorous head-to-head comparison |
| Silent API failures | Circuit breaker + degraded status | Phase 22 (enhance) | Operator visibility into poller health |

## Gamma API Resolution Fields (Verified from Go SDK)

The Polymarket Gamma API Market schema includes these resolution-relevant fields:

| Field | Type | Purpose |
|-------|------|---------|
| `closed` | bool | Market is closed for trading |
| `active` | bool | Market is active |
| `archived` | bool | Market is archived |
| `resolutionSource` | string | Source/oracle for resolution |
| `automaticallyResolved` | bool | Resolved without human intervention |
| `resolvedBy` | string | Entity that resolved |
| `umaResolutionStatus` | string | UMA oracle resolution status |
| `closedTime` | datetime | When market closed |
| `outcomePrices` | array | Final outcome prices ([1.0, 0.0] for resolved Yes) |

**No explicit `voided` or `winner` field exists.** Voided markets must be detected by:
1. `closed=true` AND outcome prices near `[0.5, 0.5]` (refund state)
2. OR `resolutionSource` containing void/cancel indicators
3. OR `umaResolutionStatus` indicating dispute/void

The existing `_extract_outcome()` in comparison.py handles normal resolution well (prices near 0 or 1) but has no voided detection path. A new `_is_voided_market()` helper is needed.

## Detailed Codebase Findings

### POLY-01: The Bug and Its Full Impact

**Location:** `src/polymarket/auto_forecaster.py:601`
```python
existing.created_at = datetime.now(timezone.utc)
```

**Impact chain:**
1. `reforecast_active()` overwrites `created_at` on every reforecast cycle
2. `count_today_new_forecasts()` (line 173) counts `Prediction.created_at >= today` with provenance `polymarket_driven` -- this now catches reforecasted predictions too
3. `count_today_reforecasts()` (line 190) subtracts new forecasts from total predictions updated today -- but since reforecast overwrites `created_at`, the total includes them, making the subtraction meaningless
4. Net effect: reforecasts eat into the new forecast cap (3/day) and reforecast cap tracking is unreliable

**Fix requires:**
- Add `reforecasted_at` column to `Prediction` model
- Alembic migration for the new column
- Stop overwriting `created_at` in `reforecast_active()`
- Set `reforecasted_at = datetime.now(timezone.utc)` instead
- Rewrite `count_today_reforecasts()` to use `reforecasted_at`

### POLY-02: Accuracy Table Design

The CONTEXT.md specifies:
- Global cumulative Brier score (both sides), updated after each resolution
- Rolling 30-day window Brier score
- No per-category breakdown

Per-comparison Brier scores already exist on `PolymarketComparison.geopol_brier` and `.polymarket_brier`. The `polymarket_accuracy` table is an append-only ledger of snapshots -- each resolution appends a row with the new cumulative totals.

Rolling 30-day: compute on-write via `SELECT AVG(geopol_brier) FROM polymarket_comparisons WHERE status='resolved' AND resolved_at >= now - 30d`.

### POLY-04: Resolution Enhancement

The existing `resolve_completed()` in comparison.py:
1. Queries active comparisons
2. Fetches event prices via `client.fetch_event_prices()`
3. Checks `market.get("resolved") or market.get("closed")`
4. Extracts outcome from price convergence

**Problems:**
- `fetch_event_prices()` only returns markets with `outcomePrices` or `bestBid` -- does not expose `resolutionSource`, `automaticallyResolved`, or other resolution metadata
- Uses price convergence (>0.95, <0.05) which misses voided markets
- No voided/cancelled handling

**Fix requires:**
- Enhance `fetch_event_prices()` (or add `fetch_event_details()`) to return full market data including resolution fields
- Add `_is_voided_market()` helper
- Add `voided` status handling in `resolve_completed()`
- Exclude voided comparisons from accuracy calculations

### POLY-05: Existing Retry Infrastructure

The client.py already has:
- `tenacity` retry decorator on `_get_json()`: 3 attempts, exponential backoff (2s min, 16s max)
- Circuit breaker: 5 consecutive failures opens circuit, 5-minute recovery window
- Per-request timeout: 15 seconds

**What needs enhancement:**
- 429 (rate limit) handling: currently treated as generic failure, should have specific backoff
- Degraded status exposure to admin: circuit breaker state is internal to `PolymarketClient` instance
- Extended outage behavior: when circuit is open, active comparisons should show last known data (already happens -- circuit returns empty lists, no data modification)
- The ProcessTable already shows polymarket daemon status; circuit breaker state could be surfaced as an additional signal

### Admin Panel Integration Points

The admin API is at `/api/v1/admin/` with `X-Admin-Key` auth. New accuracy endpoints should follow this pattern:

```
GET /api/v1/admin/accuracy          -- summary stats + recent resolved
GET /api/v1/admin/accuracy/resolved -- full resolved comparisons table
```

Alternatively, accuracy data could go on the existing calibration routes since it's related:
```
GET /api/v1/calibration/polymarket/accuracy
```

**Recommendation:** Use admin routes (`/api/v1/admin/accuracy`) because:
1. The panel is admin-only
2. Uses X-Admin-Key auth (simpler than API key auth)
3. Follows the existing admin panel pattern exactly
4. No reason for public API consumers to see this data

## Open Questions

1. **Voided market detection heuristics**
   - What we know: Gamma API has `resolutionSource`, `automaticallyResolved`, `umaResolutionStatus` fields but no explicit `voided` boolean
   - What's unclear: Exact string values that indicate void/cancel in `resolutionSource` and `umaResolutionStatus`
   - Recommendation: Start conservative -- treat as voided when `closed=true` AND outcome prices are near [0.5, 0.5] (within 0.05 of each other). Log all resolutions so we can refine heuristics from real data.

2. **Top-10 active set enforcement**
   - What we know: CONTEXT.md says "forecast exactly the top 10 geo markets by volume"
   - What's unclear: Whether this replaces the current volume threshold model or layers on top
   - Recommendation: Replace -- `fetch_top_geopolitical(limit=10)` already returns the right set. Remove volume threshold gating from `auto_forecaster.run()` and instead feed it exactly the top-10 set.

3. **Admin CSS for new panel**
   - What we know: Admin CSS is in `frontend/src/admin/admin-styles.css`, dynamically imported
   - What's unclear: Whether new table styles need custom CSS or can reuse ProcessTable styles
   - Recommendation: Reuse `.process-table` class for the sortable table. The summary stats row can use `.feed-stats-footer` pattern.

## Sources

### Primary (HIGH confidence)
- `src/polymarket/auto_forecaster.py` -- Full read, bug confirmed at line 601
- `src/polymarket/client.py` -- Circuit breaker and retry implementation verified
- `src/polymarket/comparison.py` -- Resolution logic, Brier score computation
- `src/polymarket/matcher.py` -- Matching pipeline (not changed in this phase)
- `src/db/models.py` -- Full schema: Prediction, PolymarketComparison, PolymarketSnapshot
- `src/settings.py` -- All polymarket_* settings
- `src/scheduler/job_wrappers.py` -- heavy_polymarket_cycle dispatch
- `src/scheduler/heavy_runner.py` -- run_polymarket_cycle execution order
- `src/api/routes/v1/admin.py` -- Admin endpoint patterns
- `src/api/routes/v1/calibration.py` -- Existing polymarket API endpoints
- `src/api/services/admin_service.py` -- Admin business logic patterns
- `src/api/schemas/admin.py` -- Admin Pydantic DTO patterns
- `frontend/src/admin/` -- All admin panel registration patterns

### Secondary (MEDIUM confidence)
- [Polymarket Gamma Structure docs](https://docs.polymarket.com/developers/gamma-markets-api/gamma-structure) -- Market schema
- [Go SDK market schema](https://pkg.go.dev/github.com/ivanzzeth/polymarket-go-gamma-client) -- Resolution fields (resolved, resolutionSource, closed, automaticallyResolved)

### Tertiary (LOW confidence)
- Voided market detection heuristics -- no official documentation on void indicators in Gamma API response fields

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- all existing dependencies, no new libraries
- Architecture: HIGH -- all patterns derived from codebase read, not inference
- Bug analysis: HIGH -- exact line number confirmed, impact chain traced through code
- Admin panel: HIGH -- full pattern derived from 4 existing panels + registration code
- Resolution/voided handling: MEDIUM -- Gamma API fields verified via Go SDK but void detection heuristics are unverified
- Pitfalls: HIGH -- all derived from actual code analysis, not hypothetical

**Research date:** 2026-03-06
**Valid until:** 2026-04-06 (stable -- internal codebase patterns, minimal external dependency)
