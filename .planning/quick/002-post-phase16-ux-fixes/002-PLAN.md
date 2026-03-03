---
phase: quick
plan: 002
type: execute
wave: 1
depends_on: []
autonomous: true
files_modified:
  # Task 1 (Backend: Polymarket top-10)
  - src/polymarket/client.py
  - src/api/routes/v1/calibration.py
  # Task 2 (Frontend: Polymarket + map + ensemble removal + scheduler)
  - frontend/src/types/api.ts
  - frontend/src/services/forecast-client.ts
  - frontend/src/components/CalibrationPanel.ts
  - frontend/src/components/DeckGLMap.ts
  - frontend/src/screens/dashboard-screen.ts
  - frontend/src/app/refresh-scheduler.ts
  - frontend/src/styles/panels.css
  # Task 3 (Frontend: MyForecastsPanel + SubmissionQueue status merge)
  - frontend/src/components/MyForecastsPanel.ts
  - frontend/src/components/SubmissionQueue.ts
  # Deleted
  - frontend/src/components/EnsembleBreakdownPanel.ts

must_haves:
  truths:
    - "CalibrationPanel shows 10 Polymarket geo events sorted by volume, with GEOPOL column showing probability when matched and '--' when not"
    - "Globe map fills the full viewport on first render without requiring window resize"
    - "Dashboard panels repopulate immediately on re-navigation (scheduler fires first tick at t=0)"
    - "EnsembleBreakdownPanel is gone from dashboard; ensemble info only lives in expanded cards"
    - "Completed forecasts in MyForecastsPanel use expandable cards identical to SubmissionQueue"
    - "Pending and confirmed statuses both display as 'QUEUED' in MyForecastsPanel and SubmissionQueue"
  artifacts:
    - path: "src/polymarket/client.py"
      provides: "fetch_top_geopolitical(limit=10) method"
      contains: "fetch_top_geopolitical"
    - path: "src/api/routes/v1/calibration.py"
      provides: "GET /calibration/polymarket/top endpoint"
      contains: "/polymarket/top"
    - path: "frontend/src/components/CalibrationPanel.ts"
      provides: "Renders top-10 Polymarket events with optional Geopol match"
    - path: "frontend/src/components/MyForecastsPanel.ts"
      provides: "Expandable cards for completed forecasts, QUEUED status"
      contains: "buildExpandableCard"
  key_links:
    - from: "src/api/routes/v1/calibration.py"
      to: "src/polymarket/client.py"
      via: "fetch_top_geopolitical() call"
      pattern: "fetch_top_geopolitical"
    - from: "frontend/src/services/forecast-client.ts"
      to: "/calibration/polymarket/top"
      via: "getPolymarketTop() method"
      pattern: "polymarket/top"
---

<objective>
Post-Phase 16 UX fixes: 6 issues covering Polymarket revamp (top-10 by volume with optional Geopol match), map sizing (MapLibre resize on load), dashboard re-navigation (immediate scheduler tick), EnsembleBreakdownPanel removal (redundant with expandable cards), MyForecastsPanel expandable cards, and pending/confirmed state merge to QUEUED.

Purpose: Fix the 6 highest-priority UX regressions/improvements identified during Phase 16 review.
Output: Backend endpoint + frontend changes across dashboard, globe, and forecasts screens.
</objective>

<execution_context>
@~/.claude/get-shit-done/workflows/execute-plan.md
@~/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/PROJECT.md
@.planning/ROADMAP.md
@.planning/STATE.md
@src/polymarket/client.py
@src/polymarket/comparison.py
@src/api/routes/v1/calibration.py
@frontend/src/types/api.ts
@frontend/src/services/forecast-client.ts
@frontend/src/components/CalibrationPanel.ts
@frontend/src/components/DeckGLMap.ts
@frontend/src/components/EnsembleBreakdownPanel.ts
@frontend/src/components/MyForecastsPanel.ts
@frontend/src/components/SubmissionQueue.ts
@frontend/src/components/expandable-card.ts
@frontend/src/screens/dashboard-screen.ts
@frontend/src/screens/globe-screen.ts
@frontend/src/app/refresh-scheduler.ts
@frontend/src/styles/panels.css
</context>

<tasks>

<task type="auto">
  <name>Task 1: Polymarket top-10 backend + map resize + scheduler immediate tick</name>
  <files>
    src/polymarket/client.py
    src/api/routes/v1/calibration.py
    frontend/src/components/DeckGLMap.ts
    frontend/src/app/refresh-scheduler.ts
  </files>
  <action>
**A. Polymarket top-10 backend (`src/polymarket/client.py`)**

Add method `fetch_top_geopolitical(self, limit: int = 10) -> list[dict]` to `PolymarketClient`:
1. Call existing `self.fetch_geopolitical_markets(limit=200)` to get the full geo-filtered event list (increase limit to cast wider net).
2. Sort the returned events by volume descending. The Gamma API returns `volume` field (string) on event dicts. Parse to float, default 0.0 for missing/malformed values. Secondary sort by `liquidity` (also string, float parse) if volumes are equal.
3. Return the top `limit` events.
4. If `fetch_geopolitical_markets` returns empty (circuit breaker open, API down), return empty list.
5. Same error handling pattern: never raise from this method.

**B. New endpoint (`src/api/routes/v1/calibration.py`)**

Add `GET /polymarket/top` endpoint:

1. Define response models:
   ```python
   class PolymarketTopEventItem(BaseModel):
       event_id: str
       title: str
       slug: str
       volume: float
       liquidity: float
       # Optional Geopol match data (None if no match)
       geopol_prediction_id: str | None = None
       geopol_probability: float | None = None
       geopol_question: str | None = None
       match_confidence: float | None = None

   class PolymarketTopResponse(BaseModel):
       events: list[PolymarketTopEventItem]
       total_geo_markets: int  # How many geo events exist before top-N cut
   ```

2. Implementation:
   - Instantiate `PolymarketClient()` (session-less, it creates its own).
   - Call `await client.fetch_top_geopolitical(limit=10)`.
   - Query `PolymarketComparison` table for all active comparisons (join on polymarket_event_id).
   - Also query `Prediction` table to get the geopol question text for matched predictions.
   - For each of the top-10 events, check if a comparison row exists. If so, populate geopol fields. If not, leave them None.
   - Close the client session after use (`await client.close()`).
   - Return `PolymarketTopResponse`.

3. Wire the endpoint: `@router.get("/polymarket/top", response_model=PolymarketTopResponse)` with same auth dependency as existing endpoints.

**C. Map resize fix (`frontend/src/components/DeckGLMap.ts`)**

In `initMap()` method (line ~384), inside the `this.map.on('load', () => { ... })` callback, add `this.map!.resize();` AFTER `this.initDeckOverlay()`:

```typescript
this.map.on('load', () => {
  this.initDeckOverlay();
  // Force MapLibre to recalculate viewport dimensions.
  // Container may not have had computed dimensions at construction time
  // (e.g., absolutely-positioned parent not yet laid out).
  this.map!.resize();
});
```

**D. Scheduler immediate first tick (`frontend/src/app/refresh-scheduler.ts`)**

In `scheduleRefresh()`, the first call to `scheduleNext()` at the end uses `computeDelay(intervalMs, ...)` which starts with a full interval delay. Change the initial scheduling to fire immediately (delay=0) for the first tick, then normal interval thereafter:

Add a `firstTick: boolean` flag. On the first call in `scheduleNext`, use delay=0 instead of the computed delay. After the first `run()` completes, subsequent `scheduleNext` calls use normal `computeDelay`. Concrete implementation:

Replace the final line `scheduleNext(computeDelay(intervalMs, document.visibilityState === 'hidden'));` with `scheduleNext(0);` -- this fires the first tick immediately. The `run()` function already calls `scheduleNext(computeDelay(...))` after completion, so subsequent ticks use normal intervals.

This is safe because `loadInitial()` in dashboard-screen already runs the same data fetches fire-and-forget. The worst case is a double-fetch on first mount, which dedup handles (same key = same promise). But it makes re-navigation always trigger an immediate data fetch via the scheduler.

IMPORTANT: Since `loadInitial()` in dashboard-screen.ts and the scheduler now both fire immediately, they will race. The dedup layer in `forecastClient` means the second call for the same endpoint returns the same promise. No wasted requests. However, to be cleaner: remove the `loadInitial()` function from `dashboard-screen.ts` entirely. The scheduler's immediate first tick handles the same work. Keep the initial `calibrationPanel.updatePolymarket()` and `eventTimelinePanel.refresh()` calls since those aren't registered in the scheduler OR register them in the scheduler too.

Actually, simpler approach: keep `loadInitial()` as-is (it's a good pattern for coordinated initial load with Promise.all). Just change the scheduler's first tick from `computeDelay(intervalMs, ...)` to `computeDelay(intervalMs, ...)` but with a floor of `intervalMs` for the first call. Wait -- the actual problem is: on re-navigation, panels are empty because loadInitial might fail. The scheduler should fire at t=0. Change the bottom of `scheduleRefresh`:

```typescript
// Fire first tick immediately (delay=0). Subsequent ticks use normal interval.
scheduleNext(0);
```

This replaces `scheduleNext(computeDelay(intervalMs, document.visibilityState === 'hidden'));`.
  </action>
  <verify>
    - `uv run python -c "from src.polymarket.client import PolymarketClient; print('OK')"` -- import succeeds
    - `uv run python -c "from src.api.routes.v1.calibration import router; routes = [r.path for r in router.routes]; print(routes); assert '/polymarket/top' in routes"` -- new route exists
    - `cd frontend && npx tsc --noEmit` -- TypeScript compilation passes with DeckGLMap and scheduler changes
  </verify>
  <done>
    - `PolymarketClient.fetch_top_geopolitical()` returns up to N events sorted by volume from geo-filtered set.
    - `GET /calibration/polymarket/top` returns 10 events with optional Geopol match data.
    - MapLibre calls `resize()` in the `load` callback so the globe fills the viewport on first render.
    - `RefreshScheduler.scheduleRefresh()` fires the first tick at delay=0, making dashboard repopulate immediately on re-navigation.
  </done>
</task>

<task type="auto">
  <name>Task 2: Frontend Polymarket top-10 + EnsembleBreakdownPanel removal + dashboard cleanup</name>
  <files>
    frontend/src/types/api.ts
    frontend/src/services/forecast-client.ts
    frontend/src/components/CalibrationPanel.ts
    frontend/src/screens/dashboard-screen.ts
    frontend/src/components/EnsembleBreakdownPanel.ts
    frontend/src/styles/panels.css
  </files>
  <action>
**A. New types (`frontend/src/types/api.ts`)**

Add after the `PolymarketComparisonResponse` interface:

```typescript
/** Single event from GET /calibration/polymarket/top. */
export interface PolymarketTopEvent {
  event_id: string;
  title: string;
  slug: string;
  volume: number;
  liquidity: number;
  // Optional Geopol match data
  geopol_prediction_id: string | null;
  geopol_probability: number | null;
  geopol_question: string | null;
  match_confidence: number | null;
}

/** GET /calibration/polymarket/top response. */
export interface PolymarketTopResponse {
  events: PolymarketTopEvent[];
  total_geo_markets: number;
}
```

**B. New client method (`frontend/src/services/forecast-client.ts`)**

Add `getPolymarketTop()` method to `ForecastServiceClient`:

```typescript
/** GET /calibration/polymarket/top */
async getPolymarketTop(): Promise<PolymarketTopResponse> {
  const key = '/calibration/polymarket/top';
  return this.dedup(key, () =>
    this.forecastBreaker.execute(
      () => this.fetchJson<PolymarketTopResponse>(key),
      FALLBACK_POLYMARKET_TOP,
    ),
  ) as Promise<PolymarketTopResponse>;
}
```

Add fallback constant:
```typescript
const FALLBACK_POLYMARKET_TOP: PolymarketTopResponse = {
  events: [],
  total_geo_markets: 0,
};
```

Import the new types. Keep `getPolymarket()` for now (existing comparison service still uses it).

**C. CalibrationPanel rewrite (`frontend/src/components/CalibrationPanel.ts`)**

Replace the Polymarket section with a top-10 rendering:

1. Replace `updatePolymarket(data: PolymarketComparisonResponse)` with `updatePolymarketTop(data: PolymarketTopResponse)`.

2. Rewrite `buildPolymarketSection()` to accept `PolymarketTopResponse`. Build a table with columns:
   - QUESTION (title, truncated to 60 chars, linked to `https://polymarket.com/event/${slug}` via `<a>` tag)
   - VOLUME (formatted with K/M suffix for readability)
   - GEOPOL (probability from match, or "--" if no match)
   - MARKET (market price from Polymarket, or "--" if not available -- note: the top endpoint doesn't return market price directly, so omit this column for now OR derive it from event data if available)

   Actually, simplify. The `PolymarketTopEvent` doesn't carry a market price (the price is per-outcome, not per-event). The key display is:
   - QUESTION (title)
   - VOL (formatted volume)
   - GEOPOL (matched probability or "--")
   - MATCH (confidence badge if matched, empty if not)

3. Rewrite `buildPolymarketTable()` accordingly. Each row shows a Polymarket event; rows with a Geopol match get a highlighted GEOPOL probability value.

4. Keep existing calibration rendering (reliability diagram, Brier table, track record sparkline) unchanged.

5. At bottom of section, show `total_geo_markets` count: "Showing top 10 of {N} geopolitical markets".

**D. Dashboard EnsembleBreakdownPanel removal (`frontend/src/screens/dashboard-screen.ts`)**

1. Remove `import { EnsembleBreakdownPanel } from '@/components/EnsembleBreakdownPanel';` (line 23).
2. Remove `const ensemblePanel = new EnsembleBreakdownPanel();` (line 87).
3. Remove `columns.col4.appendChild(ensemblePanel.getElement());` (line 103).
4. Remove `ctx.panels['ensemble'] = ensemblePanel;` (line 111).
5. In the `forecastSelectedHandler` (line 146-150), remove `ensemblePanel.update(forecast);`. Keep only `calibrationPanel.update([forecast.calibration]);`.

6. Update the Polymarket scheduling: change `forecastClient.getPolymarket()` to `forecastClient.getPolymarketTop()` in both `loadInitial()` and the scheduler registration. Change `calibrationPanel.updatePolymarket(polymarket)` to `calibrationPanel.updatePolymarketTop(polymarket)`.

**E. Delete EnsembleBreakdownPanel file**

Delete `frontend/src/components/EnsembleBreakdownPanel.ts`.

**F. Remove ensemble CSS from panels.css**

Remove lines 772-845 (the entire "Ensemble Breakdown" CSS section from `/* ======== Ensemble Breakdown */` through `.ensemble-temp { ... }`).

**G. Column layout adjustment**

With EnsembleBreakdownPanel gone from Col 4, the remaining panels (EventTimeline, SystemHealth, Calibration) have more room. No CSS changes needed -- flex column just uses less space.
  </action>
  <verify>
    - `cd frontend && npx tsc --noEmit` -- no type errors
    - Verify `EnsembleBreakdownPanel.ts` is deleted: `! test -f frontend/src/components/EnsembleBreakdownPanel.ts`
    - Grep for dead references: `grep -r "EnsembleBreakdownPanel\|ensemble-section\|ensemble-bar\|ensemble-segment\|ensemble-label" frontend/src/ --include="*.ts" --include="*.css"` should return nothing (or only the expanded-card.ts which uses its own classnames prefixed with `expanded-ensemble-`)
  </verify>
  <done>
    - CalibrationPanel renders top-10 Polymarket geo events with optional Geopol match indicator.
    - EnsembleBreakdownPanel completely removed (file, imports, CSS, ctx registration).
    - Dashboard loads Polymarket data via new top-10 endpoint.
    - No dead imports or CSS rules remain.
  </done>
</task>

<task type="auto">
  <name>Task 3: MyForecastsPanel expandable cards + QUEUED status merge</name>
  <files>
    frontend/src/components/MyForecastsPanel.ts
    frontend/src/components/SubmissionQueue.ts
  </files>
  <action>
**A. MyForecastsPanel expandable cards (`frontend/src/components/MyForecastsPanel.ts`)**

Refactor to use expandable-card pattern for completed forecasts, following SubmissionQueue as reference:

1. Add imports:
   ```typescript
   import {
     buildExpandableCard,
     buildExpandedContent,
     isoToFlag,
     relativeTime as ecRelativeTime,
     truncate as ecTruncate,
     type ExpandableCardOptions,
   } from '@/components/expandable-card';
   import type { ForecastResponse } from '@/types/api';
   ```
   Remove the local `relativeTime` and `truncate` functions -- use the shared versions from expandable-card.ts.

2. Add instance state:
   ```typescript
   private readonly expandedIds = new Set<string>();
   private readonly forecastCache = new Map<string, ForecastResponse>();
   ```

3. Rewrite `buildRequestRow(r)` for completed status:
   - If `r.status === 'complete' && r.prediction_ids.length > 0`:
     - Build a wrapper div with status badge header (COMPLETE + time).
     - Check `forecastCache` for the prediction. If cached, render via `buildExpandableCard()` with same toggle wiring as SubmissionQueue (lines 299-317 in SubmissionQueue.ts).
     - If not cached, show question text as loading placeholder, fire background `forecastClient.getForecastById()` fetch, cache result, replace placeholder with expandable card.
   - Other statuses: keep existing simple row rendering (pending/confirmed/processing/failed).

4. Remove the `openCompletedForecast()` method that dispatches `forecast-selected` directly. The expandable card's "View Full Analysis" button handles that via `buildExpandedContent`.

5. Add `destroy()` override to clear `forecastCache` and `expandedIds`.

**B. QUEUED status merge -- both components**

In `MyForecastsPanel.ts`, update `statusCssClass()`:
```typescript
function statusCssClass(status: ForecastRequestStatus['status']): string {
  switch (status) {
    case 'pending':
    case 'confirmed':
      return 'status-queued';
    case 'processing': return 'status-processing';
    case 'complete': return 'status-complete';
    case 'failed': return 'status-failed';
  }
}
```

Update the `statusLabel` used in `buildRequestRow`:
```typescript
const statusLabel = (status === 'pending' || status === 'confirmed') ? 'QUEUED' : status.toUpperCase();
```

In `SubmissionQueue.ts`, update `statusCssClass()`:
```typescript
function statusCssClass(status: ForecastRequestStatus['status']): string {
  switch (status) {
    case 'pending':
    case 'confirmed':
      return 'sq-status-queued';
    case 'processing': return 'sq-status-processing';
    case 'complete': return 'sq-status-complete';
    case 'failed': return 'sq-status-failed';
  }
}
```

Update `statusLabel()`:
```typescript
function statusLabel(status: ForecastRequestStatus['status']): string {
  switch (status) {
    case 'pending':
    case 'confirmed':
      return 'QUEUED';
    case 'processing': return 'PROCESSING';
    case 'complete': return 'COMPLETE';
    case 'failed': return 'FAILED';
  }
}
```

In `panels.css`, add the new QUEUED status badge classes. The `status-queued` class should use the same color as the existing `status-pending` (neutral/muted). The `sq-status-queued` should use the same as `sq-status-pending`. Map both `status-pending` and `status-confirmed` to the same queued style, OR just add the new classes and leave old ones as dead CSS (they won't be referenced).

Specifically, add:
```css
.status-queued {
  /* Same as status-pending -- neutral */
  color: var(--text-muted);
  background: var(--surface-elevated);
}

.sq-status-queued {
  /* Same as sq-status-pending -- neutral */
  color: var(--text-muted);
  background: var(--surface-elevated);
}
```

Find the existing `.status-pending`, `.status-confirmed`, `.sq-status-pending`, `.sq-status-confirmed` rules and add the queued equivalents next to them. Remove the old `pending` and `confirmed` classes if they are no longer referenced anywhere.
  </action>
  <verify>
    - `cd frontend && npx tsc --noEmit` -- TypeScript compiles cleanly
    - `grep -r "status-pending\|status-confirmed\|sq-status-pending\|sq-status-confirmed" frontend/src/ --include="*.ts"` should return nothing (old classes no longer referenced in TS)
    - `grep "QUEUED" frontend/src/components/MyForecastsPanel.ts frontend/src/components/SubmissionQueue.ts` -- both files show QUEUED label
    - `grep "buildExpandableCard" frontend/src/components/MyForecastsPanel.ts` -- expandable card imported and used
  </verify>
  <done>
    - Completed forecasts in MyForecastsPanel use expandable cards with progressive disclosure (click-expand with ensemble/evidence/tree, "View Full Analysis" for ScenarioExplorer).
    - Pending and confirmed statuses both display as "QUEUED" with identical badge styling in both MyForecastsPanel and SubmissionQueue.
    - Forecast data is cached per prediction ID to avoid re-fetching on refresh.
    - No more direct `forecast-selected` dispatch from row click -- expandable card handles it.
  </done>
</task>

</tasks>

<verification>
1. `cd /home/kondraki/personal/geopol && uv run python -c "from src.polymarket.client import PolymarketClient; c = PolymarketClient(); print(hasattr(c, 'fetch_top_geopolitical'))"` -- True
2. `cd /home/kondraki/personal/geopol && uv run python -c "from src.api.routes.v1.calibration import router; print([r.path for r in router.routes])"` -- includes `/polymarket/top`
3. `cd /home/kondraki/personal/geopol/frontend && npx tsc --noEmit` -- zero errors
4. `! test -f /home/kondraki/personal/geopol/frontend/src/components/EnsembleBreakdownPanel.ts` -- file deleted
5. `grep -c "QUEUED" /home/kondraki/personal/geopol/frontend/src/components/MyForecastsPanel.ts /home/kondraki/personal/geopol/frontend/src/components/SubmissionQueue.ts` -- both > 0
6. `grep "resize()" /home/kondraki/personal/geopol/frontend/src/components/DeckGLMap.ts` -- resize call exists in load handler
7. `grep "scheduleNext(0)" /home/kondraki/personal/geopol/frontend/src/app/refresh-scheduler.ts` -- immediate first tick
</verification>

<success_criteria>
- All 6 issues resolved in a single execution pass
- Backend: new `fetch_top_geopolitical()` method + `GET /polymarket/top` endpoint
- Frontend: CalibrationPanel shows top-10 events with optional match, map fills viewport, EnsembleBreakdownPanel deleted, MyForecastsPanel has expandable cards, pending/confirmed merged to QUEUED, scheduler fires immediately on registration
- TypeScript compiles without errors
- No dead imports, no dead CSS rules for ensemble panel
</success_criteria>

<output>
After completion, create `.planning/quick/002-post-phase16-ux-fixes/002-SUMMARY.md`
</output>
