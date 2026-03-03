# Phase 15: URL Routing & Dashboard Screen - Research

**Researched:** 2026-03-03
**Domain:** Vanilla TypeScript SPA routing, CSS multi-column dashboard layout, progressive disclosure UI patterns
**Confidence:** HIGH

## Summary

Phase 15 transforms the Geopol single-screen Phase 12 layout into a three-route SPA (`/dashboard`, `/globe`, `/forecasts`) and implements the Dashboard screen as an information-dense 4-column Bloomberg-terminal-style view. The codebase is vanilla TypeScript (no React, no framework) using a custom `h()` DOM helper, a `Panel` base class hierarchy, and `CustomEvent`-based inter-component communication.

The technical domain is well-understood: the History API (`pushState`/`popstate`) for URL routing is browser-standard and has no library dependency. The 4-column layout uses CSS flexbox with independent `overflow-y` scrolling per column. Progressive disclosure on forecast cards requires expanding the existing `ForecastPanel` card rendering with toggle state management. The d3-hierarchy mini scenario tree is a viewport-reduced version of the existing `ScenarioExplorer` tree code. The View Transitions API (same-document) is Baseline as of October 2025 and can provide the ~150ms fade crossfade transition, with a CSS-only fallback.

**Primary recommendation:** Build a minimal ~80-line `Router` class using `pushState`/`popstate` (no library). Refactor `main.ts` from a flat boot sequence into screen-aware lifecycle management. Redistribute existing Phase 12 panels into 4 columns without rebuilding them. The `top_question` to `top_forecast` rename is a 3-file change.

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| History API (browser) | N/A | URL routing via `pushState`/`popstate` | Built into all browsers. Zero-dependency. Vanilla TS codebase has no router framework. |
| d3 | ^7.9.0 | Scenario tree hierarchy rendering | Already in `package.json`. Mini tree reuses `ScenarioExplorer` d3-hierarchy code at reduced viewport. |
| CSS Flexbox | N/A | 4-column independent-scroll layout | No Grid needed. Flexbox with `overflow-y: auto` per column achieves independent scrolling trivially. |
| View Transitions API | N/A | ~150ms fade crossfade on screen change | Baseline Newly Available (Oct 2025). `document.startViewTransition()` wraps DOM swap. |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| deck.gl | ^9.2.6 | Globe screen (Phase 16) | Already installed. Globe screen is placeholder in Phase 15 -- DeckGLMap conditionally mounted only on `/globe`. |
| maplibre-gl | ^5.16.0 | Globe basemap | Same as deck.gl -- conditionally loaded on `/globe` route only. |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Custom Router | navigo / page.js | Adds dependency for ~80 lines of code. Not justified for 3 routes. |
| View Transitions API | CSS opacity transition on container | View Transitions is cleaner (snapshot-based) but has older Firefox gap. CSS fallback is trivial. |
| Flexbox columns | CSS Grid columns | Grid adds complexity for this use case. Flexbox with fixed widths is simpler and matches WM reference. |

**Installation:**
```bash
# No new packages required. All dependencies already in package.json.
```

## Architecture Patterns

### Recommended Project Structure
```
frontend/src/
├── app/
│   ├── app-context.ts      # MODIFY: add currentRoute, selectedCountryIso state
│   ├── panel-layout.ts      # REPLACE: new 4-column dashboard layout function
│   ├── refresh-scheduler.ts # MODIFY: screen-aware refresh (only refresh visible panels)
│   └── router.ts            # NEW: Router class (~80 lines)
├── screens/                  # NEW: screen mount/unmount lifecycle
│   ├── dashboard-screen.ts   # Dashboard 4-column: creates panels, mounts into columns
│   ├── globe-screen.ts       # Placeholder: "Coming in Phase 16" + DeckGLMap shell
│   └── forecasts-screen.ts   # Placeholder: "Coming in Phase 16" + submission form shell
├── components/
│   ├── Panel.ts              # MODIFY: remove resize handle (columns own height now)
│   ├── ForecastPanel.ts      # MAJOR REFACTOR: expandable cards, search bar, country filter
│   ├── RiskIndexPanel.ts     # MODIFY: top_forecast rename, click dispatches filter event
│   ├── EventTimelinePanel.ts # MINOR: no structural change, move to Col 4
│   ├── EnsembleBreakdownPanel.ts # MINOR: move to Col 4
│   ├── SystemHealthPanel.ts  # MINOR: move to Col 4
│   ├── CalibrationPanel.ts   # MINOR: move to Col 4
│   ├── ScenarioExplorer.ts   # MODIFY: decouple from forecast-selected global event
│   ├── MyForecastsPanel.ts   # NEW: user submission tracking with status badges
│   ├── SourcesPanel.ts       # NEW: data source health/staleness indicators
│   ├── NavBar.ts             # NEW: replaces createHeader(), adds route links
│   └── SearchBar.ts          # NEW: search input + country/category dropdowns
├── services/
│   ├── forecast-client.ts    # MODIFY: add search(), getRequests(), submitQuestion(), confirmSubmission()
│   └── country-geometry.ts   # NO CHANGE
├── types/
│   ├── api.ts                # MODIFY: top_question -> top_forecast, add submission + search types
│   └── index.ts              # MODIFY: re-export new types
├── utils/
│   ├── dom-utils.ts          # NO CHANGE
│   ├── theme-manager.ts      # SIMPLIFY: remove light theme, remove toggle
│   ├── theme-colors.ts       # NO CHANGE
│   ├── circuit-breaker.ts    # NO CHANGE
│   └── sanitize.ts           # NO CHANGE
├── styles/
│   ├── main.css              # MAJOR: new 4-column layout, nav bar, remove light theme
│   └── panels.css            # MODIFY: expandable cards, search bar, status badges
└── main.ts                   # MAJOR REFACTOR: Router-driven boot, screen lifecycle
```

### Pattern 1: Minimal Hash-Free Router
**What:** A Router class that maps URL pathnames to screen mount/unmount functions using `pushState` and `popstate`.
**When to use:** SPA with 3 fixed routes, no dynamic segments, no nested routes.
**Example:**
```typescript
// Source: History API standard (MDN)
type Route = {
  path: string;
  mount: (container: HTMLElement) => void;
  unmount: () => void;
};

class Router {
  private routes: Route[] = [];
  private currentRoute: Route | null = null;
  private container: HTMLElement;

  constructor(container: HTMLElement) {
    this.container = container;
    window.addEventListener('popstate', () => this.resolve());
  }

  addRoute(route: Route): void {
    this.routes.push(route);
  }

  navigate(path: string): void {
    if (window.location.pathname === path) return;
    history.pushState(null, '', path);
    this.resolve();
  }

  resolve(): void {
    const path = window.location.pathname;
    const match = this.routes.find(r => r.path === path)
      ?? this.routes.find(r => r.path === '/dashboard'); // default

    if (this.currentRoute === match) return;

    // Unmount previous screen
    if (this.currentRoute) {
      this.currentRoute.unmount();
    }

    // Mount new screen with optional View Transition
    const doSwap = () => {
      this.container.innerHTML = '';
      match!.mount(this.container);
    };

    if (document.startViewTransition) {
      document.startViewTransition(doSwap);
    } else {
      doSwap();
    }

    this.currentRoute = match!;
  }
}
```

### Pattern 2: Screen Lifecycle (Mount/Unmount)
**What:** Each screen is a module that exports `mount(container)` and `unmount()`. Mount creates panels and wires events. Unmount destroys panels and clears refresh schedules.
**When to use:** When different screens have different panel compositions and data refresh needs.
**Example:**
```typescript
// dashboard-screen.ts
export function mountDashboard(container: HTMLElement, ctx: AppContext): void {
  // Create 4-column layout
  const layout = createDashboardLayout();
  container.appendChild(layout.element);

  // Create and mount panels into columns
  const riskIndex = new RiskIndexPanel();
  layout.col1.appendChild(riskIndex.getElement());
  // ... etc

  // Register panels in context for lifecycle
  ctx.panels['risk-index'] = riskIndex;
  // ... wire events, start refresh
}

export function unmountDashboard(ctx: AppContext): void {
  // Destroy all panels, clear refresh schedules
  for (const panel of Object.values(ctx.panels)) {
    panel.destroy();
  }
  ctx.panels = {};
}
```

### Pattern 3: Cross-Column State Synchronization
**What:** A single source of truth for the active country filter, with bidirectional sync between Col 1 (country click) and Col 2 (country dropdown).
**When to use:** When multiple UI elements control the same filter state.
**Example:**
```typescript
// Use a simple observable pattern or CustomEvent
// Col 1 click -> dispatch 'country-filter-changed' with iso
// Col 2 dropdown change -> dispatch 'country-filter-changed' with iso
// Both listen to the same event and update their UI accordingly

// Single source of truth in AppContext:
interface AppContext {
  activeCountryFilter: string | null;
  setCountryFilter(iso: string | null): void;
}

// setCountryFilter dispatches CustomEvent, both components listen
```

### Pattern 4: Expandable Forecast Cards (Non-Accordion)
**What:** Cards toggle expanded state individually. Multiple can be open simultaneously. Expanded state is per-card, not per-list.
**When to use:** When users need to compare multiple items side-by-side.
**Example:**
```typescript
// Track expanded state per forecast_id
private expandedIds = new Set<string>();

private toggleCard(id: string): void {
  if (this.expandedIds.has(id)) {
    this.expandedIds.delete(id);
  } else {
    this.expandedIds.add(id);
  }
  // Re-render only the affected card, not the entire list
  this.updateCardExpansion(id);
}
```

### Anti-Patterns to Avoid
- **Full re-render on card toggle:** Do NOT call `replaceChildren(this.content, ...allCards)` when a single card toggles. Mutate the specific card's DOM. Full list re-render destroys scroll position and kills expanded state on other cards.
- **Global event listeners for search debounce:** Do NOT attach debounced input listeners at the window level. Scope them to the SearchBar component. Clean up on unmount.
- **Screen-switching without cleanup:** ALWAYS call `unmount()` on the previous screen before mounting the new one. Failure to destroy panels leaks event listeners, refresh timers, and abort controllers.
- **Lazy-loading DeckGLMap on dashboard:** Do NOT import deck.gl/maplibre on the dashboard screen. These are ~2MB combined. Only import on globe screen. Use dynamic `import()` or Vite code splitting.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Search debounce | Custom timer management | Standard debounce utility (~8 lines) | Edge cases with rapid input, component unmount, pending requests |
| URL routing | Hash-based routing or manual URL parsing | History API `pushState`/`popstate` (built-in) | Standard, works with server-side routing, supports direct URL entry |
| Fade transitions | Manual opacity animation with JS timers | View Transitions API with CSS fallback | Browser-native, handles snapshot/crossfade automatically |
| SVG tooltips for scenario nodes | Custom tooltip positioning math | Absolute-positioned HTML div at `event.pageX/Y` | SVG `<title>` is ugly/unstyled; HTML tooltip positioned via mouse events is standard d3 pattern |
| Status badge color mapping | If/else chains | CSS class lookup map (`status-pending`, `status-processing`, `status-complete`) | Cleaner, more maintainable, follows existing codebase pattern (severity classes) |

**Key insight:** The entire routing system is ~80 lines of code. The codebase already has every pattern needed (Panel lifecycle, CustomEvent communication, h() DOM helper, circuit-breaker API client). This phase is primarily refactoring and redistribution, not new architecture.

## Common Pitfalls

### Pitfall 1: Vite Dev Server Returns 404 on Direct URL Entry
**What goes wrong:** User navigates directly to `/globe` in browser. Vite dev server returns 404 because no `globe.html` exists.
**Why it happens:** SPA routing requires ALL routes to serve `index.html`. Vite needs explicit `historyApiFallback` configuration in the dev server.
**How to avoid:** Add to `vite.config.ts`:
```typescript
server: {
  proxy: { '/api': { target: 'http://localhost:8000', changeOrigin: true } },
  // SPA fallback -- already default in Vite, but verify:
  // Vite automatically falls back to index.html for non-file URLs
}
```
Vite's dev server handles this by default (returns `index.html` for any non-file request). For production deployment, the reverse proxy (nginx/caddy) must be configured with `try_files $uri /index.html`.
**Warning signs:** Direct URL entry works in dev but not in production deploy.

### Pitfall 2: popstate Does Not Fire on pushState
**What goes wrong:** Calling `history.pushState()` does NOT fire the `popstate` event. Only browser back/forward buttons and `history.back()`/`history.forward()` fire it.
**Why it happens:** This is by design in the History API spec.
**How to avoid:** The `navigate()` method must call `pushState()` AND manually call `resolve()`. The `popstate` listener only handles browser back/forward.
**Warning signs:** Clicking nav links doesn't change screen content, but back button works.

### Pitfall 3: Scroll Position Leak Between Screens
**What goes wrong:** Navigating from dashboard (scrolled down in Col 2) to globe and back shows dashboard with scroll position reset to top, OR globe inherits dashboard's scroll state.
**Why it happens:** Screen mount creates fresh DOM, losing scroll. Or, if container overflow isn't reset, inherited scroll offsets persist.
**How to avoid:** Optionally save/restore scroll positions per route. Or accept fresh scroll on mount (simpler, standard SPA behavior). The key is that each column's `overflow-y: auto` container is freshly created on mount, so this is naturally handled.
**Warning signs:** Users complain about losing their place when switching screens.

### Pitfall 4: DeckGLMap Memory Leak on Screen Switch
**What goes wrong:** Navigating away from `/globe` without properly destroying the DeckGLMap leaks WebGL contexts, maplibre-gl instances, and deck.gl layers.
**Why it happens:** DeckGLMap allocates GPU resources (WebGL context, tile cache, textures). `container.innerHTML = ''` removes DOM but not GPU state.
**How to avoid:** The globe screen's `unmount()` must call `deckMap.destroy()` (which must call `map.remove()` on the maplibre instance and `overlay.finalize()` on the deck.gl overlay). Verify DeckGLMap has a `destroy()` method; if not, add one.
**Warning signs:** Performance degrades after repeated globe visits. Browser tab memory grows monotonically.

### Pitfall 5: Expanded Card State Lost on Data Refresh
**What goes wrong:** RefreshScheduler fetches new forecast data every 60s. ForecastPanel.update() re-renders all cards, collapsing any expanded ones.
**Why it happens:** Current `renderForecasts()` calls `replaceChildren(this.content, ...cards)`, which destroys all existing DOM.
**How to avoid:** Preserve `expandedIds` set across updates. After re-render, re-expand cards whose IDs are still in the set. Better: do a diff-based update that only modifies changed cards.
**Warning signs:** Cards randomly collapse every 60 seconds. Users cannot read expanded details.

### Pitfall 6: top_question/top_forecast Type Mismatch
**What goes wrong:** Backend returns `top_forecast` (Phase 14 rename). Frontend TypeScript type says `top_question`. JSON parsing silently drops the field (becomes `undefined`), and `top_question` reference throws at runtime.
**Why it happens:** Backend schema and frontend type diverged. Breaking change was deferred to Phase 15.
**How to avoid:** Rename in lockstep: `api.ts` type, `RiskIndexPanel.ts` references, `types/index.ts` re-export. Three files, three changes. Do this FIRST before any other work.
**Warning signs:** Country risk panel shows "undefined" for top forecast question. TypeScript compiler does NOT catch this (JSON comes in at runtime).

### Pitfall 7: Search Debounce Race Condition
**What goes wrong:** User types "iran", pauses (server query fires), then types "iran nuclear". First response returns (for "iran") and overwrites the results that should show "iran nuclear" matches.
**Why it happens:** Debounced requests fire in order but responses return out of order.
**How to avoid:** Use an AbortController. Each new search request aborts the previous in-flight request. The pattern:
```typescript
private searchController: AbortController | null = null;

private async doSearch(query: string): Promise<void> {
  this.searchController?.abort();
  this.searchController = new AbortController();
  try {
    const results = await forecastClient.search(query, { signal: this.searchController.signal });
    this.renderResults(results);
  } catch (e) {
    if (e instanceof DOMException && e.name === 'AbortError') return; // expected
    throw e;
  }
}
```
**Warning signs:** Search results flash between two different result sets.

## Code Examples

### Debounce Utility
```typescript
// Source: standard pattern, no library needed
function debounce<T extends (...args: unknown[]) => void>(
  fn: T,
  delayMs: number,
): (...args: Parameters<T>) => void {
  let timer: ReturnType<typeof setTimeout> | null = null;
  return (...args: Parameters<T>) => {
    if (timer) clearTimeout(timer);
    timer = setTimeout(() => fn(...args), delayMs);
  };
}
```

### 4-Column CSS Layout
```css
/* Source: standard flexbox pattern */
.dashboard-columns {
  display: flex;
  flex: 1;
  min-height: 0; /* critical: allows flex children to shrink below content height */
  overflow: hidden;
}

.dashboard-col {
  display: flex;
  flex-direction: column;
  overflow-y: auto;
  overflow-x: hidden;
  min-height: 0;
}

.dashboard-col--1 { width: 15%; }
.dashboard-col--2 { width: 35%; }
.dashboard-col--3 { width: 30%; }
.dashboard-col--4 { width: 20%; }

/* Column borders */
.dashboard-col + .dashboard-col {
  border-left: 1px solid var(--border);
}
```

### View Transition with Fallback
```typescript
// Source: MDN View Transition API
function transitionScreens(doSwap: () => void): void {
  if ('startViewTransition' in document) {
    (document as any).startViewTransition(doSwap);
  } else {
    doSwap();
  }
}
```

### NavBar Component
```typescript
// Source: project codebase pattern (h() helper)
function createNavBar(router: Router): HTMLElement {
  const links: Array<{ label: string; path: string }> = [
    { label: 'Dashboard', path: '/dashboard' },
    { label: 'Globe', path: '/globe' },
    { label: 'Forecasts', path: '/forecasts' },
  ];

  const nav = h('nav', { className: 'nav-bar' },
    h('div', { className: 'nav-left' },
      h('span', { className: 'nav-logo' }, 'GEOPOL'),
    ),
    h('div', { className: 'nav-links' },
      ...links.map(({ label, path }) => {
        const link = h('a', {
          className: 'nav-link',
          href: path,
          dataset: { path },
          onClick: (e: Event) => {
            e.preventDefault();
            router.navigate(path);
          },
        }, label);
        return link;
      }),
    ),
  );

  return nav;
}

// Active link highlighting: listen to route changes, toggle .nav-link--active
```

### Forecast Client Extensions (New Endpoints)
```typescript
// Source: existing forecastClient pattern in codebase

/** GET /forecasts/search?q=...&country=...&category=... */
async search(
  q: string,
  options?: { country?: string; category?: string; limit?: number; signal?: AbortSignal },
): Promise<SearchResponse> {
  const params = new URLSearchParams({ q });
  if (options?.country) params.set('country', options.country);
  if (options?.category) params.set('category', options.category);
  if (options?.limit) params.set('limit', String(options.limit));
  const path = `/forecasts/search?${params}`;
  return this.fetchJson<SearchResponse>(path, { signal: options?.signal });
}

/** POST /forecasts/submit */
async submitQuestion(question: string): Promise<ParsedQuestionResponse> {
  return this.fetchJson<ParsedQuestionResponse>('/forecasts/submit', {
    method: 'POST',
    body: JSON.stringify({ question }),
  });
}

/** POST /forecasts/submit/{id}/confirm */
async confirmSubmission(requestId: string): Promise<ConfirmSubmissionResponse> {
  return this.fetchJson<ConfirmSubmissionResponse>(
    `/forecasts/submit/${encodeURIComponent(requestId)}/confirm`,
    { method: 'POST' },
  );
}

/** GET /forecasts/requests */
async getRequests(statusFilter?: string): Promise<ForecastRequestStatus[]> {
  const params = new URLSearchParams();
  if (statusFilter) params.set('status_filter', statusFilter);
  const qs = params.toString();
  return this.fetchJson<ForecastRequestStatus[]>(
    `/forecasts/requests${qs ? `?${qs}` : ''}`,
  );
}
```

### TypeScript Types for New Backend Endpoints
```typescript
// Source: backend schemas (submission.py, search.py)

/** POST /forecasts/submit response */
export interface ParsedQuestionResponse {
  request_id: string;
  question: string;
  country_iso_list: string[];
  horizon_days: number;
  category: string;
  status: string;
  parsed_at: string;
}

/** GET /forecasts/requests item */
export interface ForecastRequestStatus {
  request_id: string;
  question: string;
  country_iso_list: string[];
  horizon_days: number;
  category: string;
  status: 'pending' | 'confirmed' | 'processing' | 'complete' | 'failed';
  submitted_at: string;
  completed_at: string | null;
  prediction_ids: string[];
  error_message: string | null;
}

/** POST /forecasts/submit/{id}/confirm response */
export interface ConfirmSubmissionResponse {
  request_id: string;
  status: string;
  message: string;
}

/** GET /forecasts/search result item */
export interface SearchResult {
  forecast: ForecastResponse;
  relevance: number;
}

/** GET /forecasts/search response */
export interface SearchResponse {
  results: SearchResult[];
  total: number;
  query: string;
  suggestions: string[] | null;
}
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Hash routing (`#/path`) | History API `pushState` | Browser standard since IE10 | Clean URLs, server-compatible, bookmarkable |
| JS animation for page transitions | View Transitions API `startViewTransition()` | Baseline Oct 2025 | Browser-native snapshot-based crossfade, no JS timer management |
| Manual CSS class toggle for themes | Single theme (dark only per CONTEXT.md) | Phase 15 decision | Delete light theme code entirely -- less CSS, fewer variables, simpler |
| 3-column grid with globe center | 4-column flexbox, no globe | Phase 15 decision | Globe moves to dedicated `/globe` route. Dashboard fills freed space. |

**Deprecated/outdated:**
- `[data-theme="light"]` CSS rules: Kill entirely. Single dark theme per user decision.
- `toggleTheme()` / `applyStoredTheme()` with light support: Simplify to dark-only.
- `panel-grid` CSS Grid layout: Replace with 4-column flexbox. Grid template areas are wrong for new layout.
- `createHeader()` in `main.ts`: Replace with NavBar component.

## Key Codebase Analysis

### Files That Change
| File | Change Type | Reason |
|------|-------------|--------|
| `main.ts` | **MAJOR rewrite** | Router-driven boot replacing flat bootstrap. Screen lifecycle management. |
| `panel-layout.ts` | **REPLACE** | New `createDashboardLayout()` for 4-column structure. Old 3-column grid deleted. |
| `app-context.ts` | **MODIFY** | Add `currentRoute`, `activeCountryFilter`, `setCountryFilter()` |
| `forecast-client.ts` | **MODIFY** | Add `search()`, `submitQuestion()`, `confirmSubmission()`, `getRequests()` methods |
| `api.ts` | **MODIFY** | `top_question` -> `top_forecast` rename. Add submission/search types. |
| `ForecastPanel.ts` | **MAJOR refactor** | Expandable cards with two-column layout, mini scenario tree, "View Full Analysis" |
| `RiskIndexPanel.ts` | **MODIFY** | `top_question` -> `top_forecast`. Click dispatches country filter event. |
| `ScenarioExplorer.ts` | **MODIFY** | Add tooltip on node hover. Decouple from global `forecast-selected` event. |
| `main.css` | **MAJOR** | New nav bar, 4-column layout, remove light theme, View Transitions CSS. |
| `panels.css` | **MODIFY** | Expandable card styles, search bar, status badges, sources panel. |
| `theme-manager.ts` | **SIMPLIFY** | Remove light theme toggle. Dark-only. |

### Files That Stay Unchanged
| File | Why |
|------|-----|
| `dom-utils.ts` | Core helper, no changes needed |
| `circuit-breaker.ts` | Works as-is for new endpoints |
| `sanitize.ts` | No new sanitization requirements |
| `theme-colors.ts` | Cache mechanism still valid with single theme |
| `country-geometry.ts` | GeoJSON loading unchanged |

### New Files
| File | Purpose |
|------|---------|
| `app/router.ts` | Router class (~80 lines) |
| `screens/dashboard-screen.ts` | Dashboard mount/unmount lifecycle |
| `screens/globe-screen.ts` | Globe placeholder (Phase 16 content) |
| `screens/forecasts-screen.ts` | Forecasts placeholder (Phase 16 content) |
| `components/NavBar.ts` | Top navigation bar with route links |
| `components/SearchBar.ts` | Search input + dropdowns for Col 2 |
| `components/MyForecastsPanel.ts` | User submission tracking (Col 3) |
| `components/SourcesPanel.ts` | Data source health indicators (Col 3) |

### Breaking Changes
| Change | Scope | Migration |
|--------|-------|-----------|
| `top_question` -> `top_forecast` in `CountryRiskSummary` | `api.ts`, `RiskIndexPanel.ts` | Rename field and all references. Backend already returns `top_forecast`. |
| `createPanelLayout()` signature | `main.ts` | Delete old, replace with `createDashboardLayout()` |
| `createHeader()` removal | `main.ts` | Replace with `NavBar` component |
| Light theme CSS deletion | `main.css`, `panels.css` | Remove all `[data-theme="light"]` blocks |

## Open Questions

1. **DeckGLMap `destroy()` method existence**
   - What we know: DeckGLMap creates maplibre-gl Map and deck.gl MapboxOverlay. The file is ~600 lines. It has `updateForecasts()`, `updateRiskScores()`, etc. but I did not see a `destroy()` method in the first 50 lines.
   - What's unclear: Whether the class has a `destroy()` or `dispose()` method for cleanup.
   - Recommendation: Verify during planning. If missing, add one that calls `map.remove()` and `overlay.finalize()`. Critical for screen switching.

2. **Vite SPA fallback in production**
   - What we know: Vite dev server handles SPA fallback by default. Production build needs nginx/caddy config.
   - What's unclear: Current deployment method (nginx, caddy, static host). No deploy config visible in repo.
   - Recommendation: Document the required production config (`try_files $uri /index.html`) but don't block Phase 15 on it. Dev server works correctly by default.

3. **GDELT event feed data source**
   - What we know: `EventTimelinePanel` currently renders hardcoded `MOCK_EVENTS`. No real event endpoint exists.
   - What's unclear: Whether a real GDELT event endpoint exists or if the feed stays mock for Phase 15.
   - Recommendation: Keep mock events for now. The panel structure is what matters for Phase 15 layout. Real data integration is orthogonal.

4. **Sources panel data source**
   - What we know: The `/api/v1/health` endpoint returns subsystem status. No dedicated "data source staleness" endpoint exists.
   - What's unclear: Whether subsystem health data is sufficient for the Sources panel or if a new endpoint is needed.
   - Recommendation: Derive Sources panel from HealthResponse subsystems (GDELT, RSS, Polymarket show as subsystems with `checked_at` timestamps). No new endpoint needed.

## Sources

### Primary (HIGH confidence)
- Codebase analysis: All 25 TypeScript source files in `frontend/src/` read and analyzed
- Backend API schemas: `submission.py`, `search.py`, `country.py` -- wire format verified
- Backend routes: `submissions.py`, `forecasts.py` (search endpoint) -- endpoint contracts verified
- MDN View Transition API: [MDN docs](https://developer.mozilla.org/en-US/docs/Web/API/View_Transition_API) -- Baseline Oct 2025
- MDN History API: Standard browser API, verified pushState/popstate behavior

### Secondary (MEDIUM confidence)
- [View Transitions 2025 update](https://developer.chrome.com/blog/view-transitions-in-2025) -- Chrome blog, confirmed Baseline status
- [Same-document view transitions Baseline](https://web.dev/blog/same-document-view-transitions-are-now-baseline-newly-available) -- web.dev confirmation
- [Can I Use View Transitions](https://caniuse.com/view-transitions) -- browser support table
- [Independent scrolling panels CSS](https://benfrain.com/independent-scrolling-panels-body-scroll-using-just-css/) -- flexbox overflow pattern
- [D3 tooltip pattern](https://d3-graph-gallery.com/graph/interactivity_tooltip.html) -- HTML div positioned via mouse events

### Tertiary (LOW confidence)
- [Vanilla JS SPA router patterns](https://www.willtaylor.blog/client-side-routing-in-vanilla-js/) -- community blog, basic pattern confirmed via MDN
- [SPA Router implementation](https://jsdev.space/spa-vanilla-js/) -- community reference

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- No new dependencies. All patterns are browser-native or already in codebase.
- Architecture: HIGH -- Codebase fully analyzed. Panel class hierarchy, event wiring, refresh lifecycle all understood. Refactoring paths are clear.
- Pitfalls: HIGH -- All pitfalls identified from actual codebase analysis (e.g., the specific `top_question`/`top_forecast` mismatch, the specific `replaceChildren` pattern in ForecastPanel that will break expanded state).
- API contract: HIGH -- Backend schemas and routes read directly. Wire format matches TypeScript type additions.

**Research date:** 2026-03-03
**Valid until:** 2026-04-03 (stable -- no external dependency changes expected)
