# Phase 12: WM-Derived Frontend - Research

**Researched:** 2026-03-02
**Domain:** TypeScript frontend dashboard (vanilla TS + deck.gl + maplibre-gl, no framework)
**Confidence:** HIGH (primary source is the actual WM codebase + existing Geopol API code)

## Summary

Phase 12 builds a geopolitical forecasting dashboard from World Monitor's architectural patterns. The research centered on reading WM's actual source code to extract the salvageable patterns, mapping the existing Geopol API contract to TypeScript types, and identifying the right libraries for domain-specific visualizations (scenario trees, reliability diagrams, force-directed graphs).

The standard approach is: fork WM's build system (Vite 6 + TypeScript strict), carry the Panel base class (~460 lines), AppContext singleton, h() hyperscript helper, CircuitBreaker, RefreshScheduler, theme-manager, and CSS variable system wholesale. Strip all 45+ WM-specific panels, all 50+ API integrations, and all WM types. Build ~8 new panels that consume exactly one API: Geopol's FastAPI server. For visualization-heavy components (scenario trees, calibration charts, entity graphs), use D3 sub-modules (d3-hierarchy, d3-force, d3-shape) rendered to SVG -- D3 is already a WM dependency and works natively with vanilla TypeScript.

**Primary recommendation:** Start from WM's scaffolding (build system + Panel + AppContext + DeckGLMap + theme system), define TypeScript interfaces mirroring the Pydantic DTOs, build a ForecastServiceClient with the WM CircuitBreaker pattern, then implement panels one at a time against mock fixtures.

## Standard Stack

The stack is locked by the CONTEXT.md decisions: WM-derived vanilla TypeScript, no framework.

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| TypeScript | 5.7+ | Type safety, strict mode | WM uses 5.7, tsconfig strict + noUncheckedIndexedAccess |
| Vite | 6.x | Build system, HMR, Brotli precompression | WM's exact config; fast dev server, production-optimized builds |
| deck.gl | 9.2.x | WebGL map layers (choropleth, scatter, heatmap, arcs) | WM uses 9.2.6; GeoJsonLayer, ScatterplotLayer, HeatmapLayer, ArcLayer all proven |
| maplibre-gl | 5.16.x | Basemap tiles (CARTO dark-matter/voyager) | WM uses 5.16.0; free vector tiles, no Mapbox token required |
| @deck.gl/mapbox | 9.2.x | MapboxOverlay integration (deck.gl layers on maplibre) | WM pattern: `new MapboxOverlay({ interleaved: true, layers: [...] })` |
| d3 | 7.9.x | d3-hierarchy (scenario trees), d3-force (entity graphs), d3-shape (calibration charts) | WM already depends on d3 ^7.9.0; framework-agnostic, SVG-native |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| @deck.gl/aggregation-layers | 9.2.x | HeatmapLayer for GDELT events | GDELTEventHeatmap layer |
| @deck.gl/layers | 9.2.x | GeoJsonLayer, ScatterplotLayer, ArcLayer, PathLayer | All 5 map layers |
| @deck.gl/core | 9.2.x | Layer base, PickingInfo types | TypeScript types for deck.gl layer construction |
| supercluster | latest | Point clustering at low zoom | If ActiveForecastMarkers cluster at low zoom (optional) |

### Not Needed

| Library | Why Not |
|---------|---------|
| React / Vue / Svelte / Solid | CONTEXT.md locks vanilla TypeScript with h() hyperscript |
| Chart.js / ECharts / Highcharts | Overkill; D3 sub-modules + hand-rolled SVG for 3 chart types |
| Treant.js / JointJS | Heavyweight tree libs; d3-hierarchy + SVG rendering is lighter, more controllable |
| Sigma.js | d3-force already in the dep tree and sufficient for entity graphs |
| i18next | Out of scope (English only, no i18n in Phase 12) |

**Installation:**
```bash
# In frontend/ directory
npm init -y
npm install typescript vite deck.gl @deck.gl/core @deck.gl/layers @deck.gl/aggregation-layers @deck.gl/mapbox maplibre-gl d3
npm install -D @types/d3 @types/maplibre-gl
```

## Architecture Patterns

### Recommended Project Structure
```
frontend/
├── index.html                      # Single entry, FOUC-free theme script
├── vite.config.ts                  # Simplified WM config (no PWA, no variants, no Tauri)
├── tsconfig.json                   # strict, noUncheckedIndexedAccess, @/ path alias
├── package.json
├── public/
│   └── data/
│       └── countries.geojson       # Country boundaries for choropleth
├── src/
│   ├── main.ts                     # Bootstrap: theme, AppContext, init panels, init map
│   ├── types/
│   │   ├── api.ts                  # TypeScript mirrors of Pydantic DTOs
│   │   └── index.ts
│   ├── app/
│   │   ├── app-context.ts          # AppContext singleton (stripped WM version)
│   │   ├── panel-layout.ts         # Panel grid creation, resize handling
│   │   └── refresh-scheduler.ts    # Periodic data refresh (carried from WM)
│   ├── services/
│   │   ├── forecast-client.ts      # ForecastServiceClient with CircuitBreaker
│   │   └── country-geometry.ts     # GeoJSON loader + point-in-polygon (from WM)
│   ├── components/
│   │   ├── Panel.ts                # Base class (from WM, stripped of i18n/Tauri)
│   │   ├── DeckGLMap.ts            # Globe with 5 layers (from WM, stripped to core)
│   │   ├── ForecastPanel.ts        # Top N forecasts list
│   │   ├── RiskIndexPanel.ts       # Per-country aggregate risk (CII pattern)
│   │   ├── EventTimelinePanel.ts   # Recent GDELT events
│   │   ├── EnsembleBreakdownPanel.ts # LLM vs TKG weights
│   │   ├── SystemHealthPanel.ts    # Subsystem health from /api/v1/health
│   │   ├── CalibrationPanel.ts     # Reliability diagram + Brier decomposition
│   │   ├── ScenarioExplorer.ts     # Full-screen modal with d3-hierarchy tree
│   │   └── CountryBriefPage.ts     # Full-screen modal with tabs
│   ├── utils/
│   │   ├── dom-utils.ts            # h(), fragment(), replaceChildren(), safeHtml() (from WM)
│   │   ├── circuit-breaker.ts      # CircuitBreaker class (from WM, stripped of Tauri/IndexedDB)
│   │   ├── theme-manager.ts        # getStoredTheme(), setTheme(), applyStoredTheme() (from WM)
│   │   ├── theme-colors.ts         # getCSSColor() with cache invalidation (from WM)
│   │   └── sanitize.ts             # escapeHtml, sanitizeUrl
│   └── styles/
│       ├── main.css                # CSS variables (new palette), base layout, theme overrides
│       └── panels.css              # Panel-specific styles
```

### Pattern 1: Panel Base Class (from WM)

**What:** Every dashboard panel extends `Panel`. Provides header, content container, resize handle, loading/error states, data badge (live/cached/unavailable), content debouncing.

**When to use:** Every panel on the dashboard.

**Key WM source:** `/home/kondraki/personal/worldmonitor/src/components/Panel.ts` (~460 lines)

**Adaptations needed:**
- Remove `i18n` calls (replace `t('...')` with string literals)
- Remove `isDesktopRuntime()` / `invokeTauri()` calls
- Remove `trackPanelResized()` analytics call
- Keep: constructor with PanelOptions, showLoading(), showError(), setDataBadge(), setContent(), resize handle with span persistence, destroy() cleanup, AbortController signal

**Example panel implementation:**
```typescript
// Source: WM CIIPanel pattern adapted for Geopol
import { Panel } from './Panel';
import { h, replaceChildren } from '../utils/dom-utils';
import { getCSSColor } from '../utils/theme-colors';
import type { CountryRiskSummary } from '../types/api';

export class RiskIndexPanel extends Panel {
  private countries: CountryRiskSummary[] = [];

  constructor() {
    super({ id: 'risk-index', title: 'RISK INDEX' });
  }

  public update(countries: CountryRiskSummary[]): void {
    this.countries = countries;
    this.setCount(countries.length);

    const list = h('div', { className: 'risk-list' },
      ...countries.map(c => this.buildCountryRow(c))
    );
    replaceChildren(this.content, list);
    this.setDataBadge('live');
  }

  private buildCountryRow(c: CountryRiskSummary): HTMLElement {
    const color = this.riskColor(c.risk_score);
    const arrow = c.trend === 'rising' ? '↑' : c.trend === 'falling' ? '↓' : '→';
    return h('div', { className: 'risk-row', dataset: { iso: c.iso_code } },
      h('span', { className: 'risk-flag' }, this.countryFlag(c.iso_code)),
      h('span', { className: 'risk-score', style: `color: ${color}` },
        Math.round(c.risk_score * 100).toString()),
      h('span', { className: `risk-trend trend-${c.trend}` }, arrow),
      h('span', { className: 'risk-question' }, c.top_question),
    );
  }

  private riskColor(score: number): string {
    if (score >= 0.8) return getCSSColor('--semantic-critical');
    if (score >= 0.6) return getCSSColor('--semantic-high');
    if (score >= 0.4) return getCSSColor('--semantic-elevated');
    return getCSSColor('--semantic-normal');
  }

  private countryFlag(code: string): string {
    return code.toUpperCase().split('')
      .map(c => String.fromCodePoint(0x1f1e6 + c.charCodeAt(0) - 65)).join('');
  }
}
```

### Pattern 2: AppContext Singleton (from WM)

**What:** A single typed object holding all shared state: map reference, panels record, cached data, inFlight dedup set, modal references.

**When to use:** Central state coordination. Passed to RefreshScheduler, PanelLayoutManager, data loader callbacks.

**Key WM source:** `/home/kondraki/personal/worldmonitor/src/app/app-context.ts`

**Geopol adaptation — stripped interface:**
```typescript
export interface GeoPolAppContext {
  map: DeckGLMap | null;
  readonly container: HTMLElement;

  panels: Record<string, Panel>;

  // Cached API data
  forecasts: ForecastResponse[];
  countries: CountryRiskSummary[];
  healthStatus: HealthResponse | null;

  // UI state
  inFlight: Set<string>;
  selectedCountry: string | null;

  // Modals
  scenarioExplorer: ScenarioExplorer | null;
  countryBriefPage: CountryBriefPage | null;

  isDestroyed: boolean;
  initialLoadComplete: boolean;
}
```

### Pattern 3: ForecastServiceClient with CircuitBreaker

**What:** A single service client consuming the Geopol FastAPI backend. Uses the WM CircuitBreaker pattern for resilience — API failures serve stale cached data with "unavailable" indicator, not a broken UI.

**Key WM source:** `/home/kondraki/personal/worldmonitor/src/utils/circuit-breaker.ts` (~280 lines)

**Adaptations:**
- Strip IndexedDB persistent cache (unnecessary for single-API client)
- Strip Tauri offline detection
- Keep: failure counting, cooldown, TTL-based cache, stale-while-revalidate, data state tracking (live/cached/unavailable)
- Add: inFlight request deduplication (WM does this in DataLoaderManager, we move it into the client)
- Add: freshness tracking per endpoint

**Example:**
```typescript
import { CircuitBreaker, createCircuitBreaker } from '../utils/circuit-breaker';
import type { ForecastResponse, CountryRiskSummary, HealthResponse } from '../types/api';

const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000/api/v1';
const API_KEY = import.meta.env.VITE_API_KEY || '';

export class ForecastServiceClient {
  private forecastBreaker = createCircuitBreaker<ForecastResponse[]>({
    name: 'forecasts-top', maxFailures: 2, cooldownMs: 30_000, cacheTtlMs: 60_000,
  });
  private countryBreaker = createCircuitBreaker<CountryRiskSummary[]>({
    name: 'countries', maxFailures: 2, cooldownMs: 30_000, cacheTtlMs: 120_000,
  });
  private inFlight = new Map<string, Promise<unknown>>();

  async getTopForecasts(limit = 5): Promise<ForecastResponse[]> {
    return this.forecastBreaker.execute(
      () => this.dedup(`top-${limit}`, () => this.fetch<ForecastResponse[]>(
        `/forecasts/top?limit=${limit}`
      )),
      [],
    );
  }

  async getCountries(): Promise<CountryRiskSummary[]> {
    return this.countryBreaker.execute(
      () => this.dedup('countries', () => this.fetch<CountryRiskSummary[]>('/countries')),
      [],
    );
  }

  private async dedup<T>(key: string, fn: () => Promise<T>): Promise<T> {
    const existing = this.inFlight.get(key);
    if (existing) return existing as Promise<T>;
    const promise = fn().finally(() => this.inFlight.delete(key));
    this.inFlight.set(key, promise);
    return promise;
  }

  private async fetch<T>(path: string): Promise<T> {
    const res = await fetch(`${API_BASE}${path}`, {
      headers: { 'X-API-Key': API_KEY },
    });
    if (!res.ok) throw new Error(`API ${res.status}: ${res.statusText}`);
    return res.json() as Promise<T>;
  }
}
```

### Pattern 4: DeckGLMap with MapboxOverlay (from WM)

**What:** maplibre-gl map with deck.gl layers rendered via MapboxOverlay in interleaved mode. CARTO dark-matter basemap (dark) / voyager (light). Country boundaries loaded from GeoJSON for choropleth.

**Key WM source:** `/home/kondraki/personal/worldmonitor/src/components/DeckGLMap.ts`

**Critical init sequence (from WM):**
```typescript
// 1. Create maplibre-gl map
this.maplibreMap = new maplibregl.Map({
  container: 'deckgl-basemap',
  style: theme === 'light' ? LIGHT_STYLE : DARK_STYLE,
  center: [0, 20], zoom: 1.5,
  renderWorldCopies: false,
  attributionControl: false,
});

// 2. On map load, create deck.gl overlay
this.maplibreMap.on('load', () => {
  this.deckOverlay = new MapboxOverlay({
    interleaved: true,
    layers: this.buildLayers(),
    getTooltip: (info) => this.getTooltip(info),
    onClick: (info) => this.handleClick(info),
    pickingRadius: 10,
  });
  this.maplibreMap.addControl(this.deckOverlay as unknown as maplibregl.IControl);
});

// 3. Theme switch: swap basemap style + rebuild layers
window.addEventListener('theme-changed', (e) => {
  const theme = (e as CustomEvent).detail?.theme;
  this.maplibreMap.setStyle(theme === 'light' ? LIGHT_STYLE : DARK_STYLE);
  this.deckOverlay?.setProps({ layers: this.buildLayers() });
});
```

**5 layers for Geopol (from CONTEXT.md decisions):**

| Layer | deck.gl Class | Data Source | Notes |
|-------|--------------|-------------|-------|
| ForecastRiskChoropleth | GeoJsonLayer | countries.geojson + GET /countries | Blue-to-red diverging fill per country |
| ActiveForecastMarkers | ScatterplotLayer | GET /forecasts/top | Sized by probability |
| KnowledgeGraphArcs | ArcLayer | GET /forecasts/country/{iso} (on hover/select) | Actor-to-actor arcs, on-demand only |
| GDELTEventHeatmap | HeatmapLayer | Future GDELT endpoint / mock | Time-range selectable |
| ScenarioZones | GeoJsonLayer (circles/polygons) | Derived from scenario entities | Highlight countries relevant to selected forecast |

**Choropleth pattern (adapted from WM's happiness choropleth):**
```typescript
private createRiskChoroplethLayer(): GeoJsonLayer | null {
  if (!this.countriesGeoJson || this.riskScores.size === 0) return null;
  const scores = this.riskScores; // Map<iso2, number>
  return new GeoJsonLayer({
    id: 'risk-choropleth',
    data: this.countriesGeoJson,
    filled: true,
    stroked: true,
    getFillColor: (feature: { properties?: Record<string, unknown> }) => {
      const code = feature.properties?.['ISO3166-1-Alpha-2'] as string | undefined;
      const score = code ? scores.get(code) : undefined;
      if (score == null) return [30, 30, 30, 40]; // Transparent gray for no-data
      // Blue-to-red diverging: blue(low) -> gray(mid) -> red(high)
      return this.divergingColor(score);
    },
    getLineColor: [100, 100, 100, 60],
    getLineWidth: 1,
    lineWidthMinPixels: 0.5,
    pickable: true,
    updateTriggers: { getFillColor: [scores.size, this.lastScoreUpdate] },
  });
}
```

### Pattern 5: h() Hyperscript Helper (from WM)

**What:** Lightweight DOM element factory. Creates elements, applies props (className, style, dataset, event listeners), appends children. Zero dependencies.

**Key WM source:** `/home/kondraki/personal/worldmonitor/src/utils/dom-utils.ts` (~148 lines)

**Carry wholesale.** Also includes: `fragment()`, `replaceChildren()`, `rawHtml()`, `safeHtml()`, `text()`.

No adaptations needed. This is framework-agnostic utility code.

### Pattern 6: CSS Variable Theme System (from WM)

**What:** CSS custom properties on `:root` for dark theme, overridden by `[data-theme="light"]`. Semantic severity colors (critical/high/elevated/normal/low), text hierarchy (text/text-secondary/text-dim/text-muted), surface hierarchy (bg/surface/surface-hover).

**Key WM sources:**
- `/home/kondraki/personal/worldmonitor/src/styles/main.css` (CSS variables, 200+ lines)
- `/home/kondraki/personal/worldmonitor/src/utils/theme-manager.ts` (~86 lines)
- `/home/kondraki/personal/worldmonitor/src/utils/theme-colors.ts` (~32 lines)

**Carry the structure, replace the palette.** WM's variables define:
- `--bg`, `--bg-secondary`, `--surface`, `--surface-hover`, `--surface-active`
- `--border`, `--border-strong`, `--border-subtle`
- `--text`, `--text-secondary`, `--text-dim`, `--text-muted`, `--text-faint`, `--text-ghost`
- `--semantic-critical` (#ff4444), `--semantic-high` (#ff8800), `--semantic-elevated` (#ffaa00), `--semantic-normal` (#44aa44), `--semantic-low` (#3388ff)
- `--font-mono` (SF Mono, Monaco, Cascadia Code, Fira Code, DejaVu Sans Mono)

**For the intelligence/analyst aesthetic (Bloomberg Terminal meets Palantir), adjust:**
- Darker bg: `--bg: #0a0e14` (near-black with blue tint)
- `--surface: #111922` (dark slate)
- Accent blue: `--accent-primary: #00d4ff` (cyan accent for active elements)
- Keep WM's severity colors as-is (already well-calibrated for dark backgrounds)
- Monospace body font with tabular-nums for data-dense readability

### Anti-Patterns to Avoid

- **innerHTML for panel content:** WM's `setContent(html: string)` method uses innerHTML. This is acceptable for WM's RSS feed rendering but dangerous for Geopol. Prefer `h()` + `replaceChildren()` for DOM construction. Use `safeHtml()` only for server-provided markdown-like content.
- **Global singleton mutation from anywhere:** WM's AppContext is mutated freely by DataLoaderManager callbacks. Geopol should use the AppContext for shared state but funnel mutations through the ForecastServiceClient's update callbacks, not scattered `ctx.forecasts = ...` assignments.
- **Unbounded layer rebuilds:** WM's `buildLayers()` can produce 36+ layers. Geopol has only 5. Don't over-architect the layer system with caching, superclustering, and ghost layers unless profiling shows a need. Start simple.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Circuit breaker with fallback | Custom retry/fallback logic | WM's `CircuitBreaker` class | Handles failure counting, cooldown, TTL cache, stale-while-revalidate, data state tracking. 280 lines of battle-tested code. |
| DOM element construction | Template literals + innerHTML | WM's `h()` helper | Type-safe props, event listener attachment, style objects, dataset — avoids XSS and enables proper cleanup |
| Theme switching | Manual class toggling | WM's `setTheme()` + CSS variable system | Handles localStorage, meta theme-color, color cache invalidation, CustomEvent dispatch |
| Country boundary hit-testing | Custom geospatial code | WM's `country-geometry.ts` service | Point-in-polygon with bbox acceleration, ISO code normalization, GeoJSON loading. 304 lines. |
| Panel resize persistence | Custom localStorage code | WM Panel base class | Span-based grid sizing with localStorage persistence, double-click reset, touch support |
| Scenario tree layout | Custom tree positioning algorithm | `d3.tree()` from d3-hierarchy | Tidy tree layout algorithm. Feed it the recursive ScenarioDTO, get x/y coordinates. |
| Force-directed graph layout | Custom physics simulation | `d3.forceSimulation()` from d3-force | Velocity Verlet integration. Works with SVG or Canvas. |
| Color interpolation for choropleth | Manual RGB math | `d3.interpolateRdYlBu` or custom d3 scale | Perceptually uniform, colorblind-accessible diverging scales |

**Key insight:** WM has already solved every "infrastructure" problem Geopol's frontend faces. The panel system, circuit breaker, theme system, globe rendering, and DOM helpers are all domain-agnostic. The only novel code is the Geopol-specific panel content and the visualization-specific rendering (scenario trees, calibration charts, entity graphs).

## Common Pitfalls

### Pitfall 1: GeoJSON Choropleth ISO Code Mismatch

**What goes wrong:** Country GeoJSON files use inconsistent property names for ISO codes. Some use `ISO3166-1-Alpha-2`, others `ISO_A2` or `iso_a2`. The Geopol API returns `iso_code` (2 or 3 letter). Mismatches cause invisible choropleth.
**Why it happens:** Different GeoJSON sources use different schemas.
**How to avoid:** Use WM's `normalizeCode()` function from `country-geometry.ts` which handles all variants. The function checks `properties['ISO3166-1-Alpha-2'] ?? properties.ISO_A2 ?? properties.iso_a2`.
**Warning signs:** Countries appear gray/transparent on the choropleth despite having risk data.

### Pitfall 2: deck.gl Layer Update Triggers

**What goes wrong:** deck.gl aggressively caches layer props for performance. If you update data in-place (mutating an array), deck.gl won't notice and won't re-render.
**Why it happens:** deck.gl uses shallow comparison on layer props. Same array reference = no update.
**How to avoid:** Always create new array/object references when data changes. Use `updateTriggers` prop to force recalculation of accessor functions:
```typescript
updateTriggers: { getFillColor: [scores.size, lastUpdateTimestamp] }
```
**Warning signs:** Map layers show stale data after API refresh. New data visible in panels but not on globe.

### Pitfall 3: MapboxOverlay + maplibre-gl Version Compatibility

**What goes wrong:** `@deck.gl/mapbox` MapboxOverlay requires specific maplibre-gl versions. Version mismatches cause silent rendering failures or WebGL context conflicts.
**Why it happens:** deck.gl 9.x was built for maplibre-gl 4.x/5.x. The API surface changed between major versions.
**How to avoid:** Pin to the exact versions WM uses: `deck.gl@9.2.6` + `maplibre-gl@5.16.0`. These are proven compatible.
**Warning signs:** Globe renders but layers don't appear. WebGL context lost warnings in console.

### Pitfall 4: CARTO Basemap Rate Limits

**What goes wrong:** CARTO's free vector tile CDN has undocumented rate limits. During development with HMR, rapid page reloads can temporarily block tile fetches.
**Why it happens:** CARTO's CDN serves free tiles but isn't unlimited.
**How to avoid:** Use `renderWorldCopies: false` (WM does this). Consider caching the basemap style JSON locally. For production, consider self-hosting tiles or using a MapTiler free tier as fallback.
**Warning signs:** Map shows gray with no tiles after many rapid dev reloads.

### Pitfall 5: TypeScript Types Drifting from Pydantic DTOs

**What goes wrong:** The Pydantic V2 DTOs define `snake_case` fields. TypeScript code wants `camelCase`. Manual type mirroring leads to drift over time -- a field added to the Pydantic model is forgotten in the TypeScript interface.
**Why it happens:** No automated schema generation between Python and TypeScript.
**How to avoid:** Define TypeScript interfaces that exactly mirror the Pydantic models' JSON output (which uses `snake_case` per Pydantic V2 default). Keep types in a single `api.ts` file with comments referencing the Pydantic source. Accept `snake_case` in TypeScript interfaces -- it's ugly but prevents translation bugs. Optionally, add a `pydantic2ts` build step later.
**Warning signs:** API responses fail to render because the TypeScript code accesses `forecastId` but the JSON has `forecast_id`.

### Pitfall 6: ScenarioDTO Recursive Type Rendering

**What goes wrong:** ScenarioDTO is self-referential (`child_scenarios: list[ScenarioDTO]`). Naive recursive rendering without depth limits causes stack overflow or renders an unreadably deep tree.
**Why it happens:** The data model supports arbitrary nesting depth. In practice, forecasts have 2-3 levels.
**How to avoid:** Set a maximum render depth (4 levels). Use d3-hierarchy's `hierarchy()` which naturally handles recursive structures. Add expand/collapse on click rather than rendering the full tree on load.
**Warning signs:** ScenarioExplorer freezes on forecasts with deeply nested scenario trees.

### Pitfall 7: Theme Change Without Layer Rebuild

**What goes wrong:** Switching dark/light theme updates CSS variables and basemap style, but deck.gl layers keep their old hardcoded RGBA colors.
**Why it happens:** deck.gl layers use RGBA tuples, not CSS variables. WM solves this by rebuilding all layers on theme change.
**How to avoid:** Listen for `theme-changed` CustomEvent. Call `this.deckOverlay?.setProps({ layers: this.buildLayers() })` on theme change (WM pattern). Recompute all color accessor functions.
**Warning signs:** Switching to light mode makes the basemap light but overlay dots/fills remain dark-mode colors.

### Pitfall 8: API Key Exposure in Frontend

**What goes wrong:** `VITE_API_KEY` is baked into the frontend bundle at build time and visible in browser devtools.
**Why it happens:** Vite env variables prefixed with `VITE_` are included in client bundles.
**How to avoid:** For development against local FastAPI, this is acceptable. For production, use a backend proxy or cookie-based auth. The API key in the frontend is a development convenience, not a security mechanism. Phase 12 is internal/analyst-facing, so this is acceptable scope.
**Warning signs:** API key visible in Network tab request headers.

## Code Examples

### TypeScript API Types (mirror Pydantic DTOs)

```typescript
// src/types/api.ts — mirrors src/api/schemas/ in Python backend
// Keep snake_case to match JSON wire format exactly

export interface EvidenceDTO {
  source: string;       // "GDELT" | "TKG pattern" | "RAG match"
  description: string;
  confidence: number;   // 0.0-1.0
  timestamp: string | null;  // ISO 8601
  gdelt_event_id: string | null;
}

export interface EnsembleInfoDTO {
  llm_probability: number;
  tkg_probability: number | null;
  weights: Record<string, number>;  // e.g. {"llm": 0.6, "tkg": 0.4}
  temperature_applied: number;
}

export interface CalibrationDTO {
  category: string;
  temperature: number;
  historical_accuracy: number;
  brier_score: number | null;
  sample_size: number;
}

export interface ScenarioDTO {
  scenario_id: string;
  description: string;
  probability: number;
  answers_affirmative: boolean;
  entities: string[];
  timeline: string[];
  evidence_sources: EvidenceDTO[];
  child_scenarios: ScenarioDTO[];  // Recursive
}

export interface ForecastResponse {
  forecast_id: string;
  question: string;
  prediction: string;
  probability: number;
  confidence: number;
  horizon_days: number;
  scenarios: ScenarioDTO[];
  reasoning_summary: string;
  evidence_count: number;
  ensemble_info: EnsembleInfoDTO;
  calibration: CalibrationDTO;
  created_at: string;   // ISO 8601
  expires_at: string;
}

export interface CountryRiskSummary {
  iso_code: string;
  risk_score: number;
  forecast_count: number;
  top_question: string;
  top_probability: number;
  trend: 'rising' | 'stable' | 'falling';
  last_updated: string;
}

export interface SubsystemStatus {
  name: string;
  healthy: boolean;
  detail: string | null;
  checked_at: string;
}

export interface HealthResponse {
  status: 'healthy' | 'degraded' | 'unhealthy';
  subsystems: SubsystemStatus[];
  timestamp: string;
  version: string;
}

export interface PaginatedResponse<T> {
  items: T[];
  next_cursor: string | null;
  has_more: boolean;
}
```

### Scenario Tree Rendering with d3-hierarchy

```typescript
// Source: d3-hierarchy documentation + d3js.org/d3-hierarchy/tree
import * as d3 from 'd3';
import type { ScenarioDTO } from '../types/api';
import { h, replaceChildren } from '../utils/dom-utils';

interface TreeNode {
  id: string;
  label: string;
  probability: number;
  affirmative: boolean;
  evidence: EvidenceDTO[];
  children: TreeNode[];
}

function scenarioToTree(scenarios: ScenarioDTO[], rootLabel: string): TreeNode {
  return {
    id: 'root',
    label: rootLabel,
    probability: 1.0,
    affirmative: false,
    evidence: [],
    children: scenarios.map(s => mapScenario(s)),
  };
}

function mapScenario(s: ScenarioDTO): TreeNode {
  return {
    id: s.scenario_id,
    label: s.description,
    probability: s.probability,
    affirmative: s.answers_affirmative,
    evidence: s.evidence_sources,
    children: s.child_scenarios.map(c => mapScenario(c)),
  };
}

function renderTree(container: HTMLElement, root: TreeNode, onNodeClick: (node: TreeNode) => void): void {
  const width = container.clientWidth;
  const nodeHeight = 80;
  const hierarchy = d3.hierarchy(root);
  const treeLayout = d3.tree<TreeNode>().nodeSize([200, nodeHeight]);
  const treeData = treeLayout(hierarchy);
  const nodes = treeData.descendants();
  const links = treeData.links();

  // Compute bounds for viewBox
  let minX = Infinity, maxX = -Infinity, minY = 0, maxY = 0;
  nodes.forEach(n => {
    if (n.x < minX) minX = n.x;
    if (n.x > maxX) maxX = n.x;
    if (n.y > maxY) maxY = n.y;
  });

  const svgWidth = maxX - minX + 400;
  const svgHeight = maxY + 200;

  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  svg.setAttribute('viewBox', `${minX - 200} -50 ${svgWidth} ${svgHeight}`);
  svg.setAttribute('width', '100%');
  svg.setAttribute('height', `${svgHeight}px`);

  // Draw links
  const linkGen = d3.linkVertical<unknown, { x: number; y: number }>()
    .x(d => d.x).y(d => d.y);
  links.forEach(link => {
    const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
    path.setAttribute('d', linkGen(link as any) || '');
    path.setAttribute('fill', 'none');
    path.setAttribute('stroke', 'var(--border-strong)');
    path.setAttribute('stroke-width', '2');
    svg.appendChild(path);
  });

  // Draw nodes
  nodes.forEach(node => {
    const g = document.createElementNS('http://www.w3.org/2000/svg', 'g');
    g.setAttribute('transform', `translate(${node.x}, ${node.y})`);
    g.style.cursor = 'pointer';
    g.addEventListener('click', () => onNodeClick(node.data));

    // Node circle sized by probability
    const radius = 8 + node.data.probability * 20;
    const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
    circle.setAttribute('r', String(radius));
    circle.setAttribute('fill', node.data.affirmative
      ? 'var(--semantic-critical)' : 'var(--semantic-low)');
    circle.setAttribute('opacity', '0.8');
    g.appendChild(circle);

    // Label
    const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
    text.setAttribute('dy', String(radius + 16));
    text.setAttribute('text-anchor', 'middle');
    text.setAttribute('fill', 'var(--text-secondary)');
    text.setAttribute('font-size', '11');
    text.textContent = node.data.label.slice(0, 40) + (node.data.label.length > 40 ? '...' : '');
    g.appendChild(text);

    svg.appendChild(g);
  });

  replaceChildren(container, svg);
}
```

### Reliability Diagram (CalibrationPanel)

```typescript
// Reliability diagram: predicted probability vs observed frequency
// Source: d3-shape for line generator, hand-rolled SVG
function renderReliabilityDiagram(
  container: HTMLElement,
  bins: Array<{ predicted: number; observed: number; count: number }>,
): void {
  const width = 300, height = 300;
  const margin = { top: 20, right: 20, bottom: 40, left: 50 };
  const w = width - margin.left - margin.right;
  const h = height - margin.top - margin.bottom;

  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
  svg.setAttribute('width', '100%');

  const g = document.createElementNS('http://www.w3.org/2000/svg', 'g');
  g.setAttribute('transform', `translate(${margin.left}, ${margin.top})`);

  // Perfect calibration diagonal
  const diag = document.createElementNS('http://www.w3.org/2000/svg', 'line');
  diag.setAttribute('x1', '0'); diag.setAttribute('y1', String(h));
  diag.setAttribute('x2', String(w)); diag.setAttribute('y2', '0');
  diag.setAttribute('stroke', 'var(--text-ghost)');
  diag.setAttribute('stroke-dasharray', '4');
  g.appendChild(diag);

  // Calibration curve
  const xScale = (v: number) => v * w;
  const yScale = (v: number) => h - v * h;

  bins.forEach(bin => {
    const cx = xScale(bin.predicted);
    const cy = yScale(bin.observed);
    const r = Math.max(3, Math.min(12, Math.sqrt(bin.count)));
    const dot = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
    dot.setAttribute('cx', String(cx));
    dot.setAttribute('cy', String(cy));
    dot.setAttribute('r', String(r));
    dot.setAttribute('fill', 'var(--accent-primary)');
    dot.setAttribute('opacity', '0.8');
    g.appendChild(dot);
  });

  // Axes labels
  const xLabel = document.createElementNS('http://www.w3.org/2000/svg', 'text');
  xLabel.setAttribute('x', String(w / 2)); xLabel.setAttribute('y', String(h + 35));
  xLabel.setAttribute('text-anchor', 'middle');
  xLabel.setAttribute('fill', 'var(--text-dim)'); xLabel.setAttribute('font-size', '11');
  xLabel.textContent = 'Predicted Probability';
  g.appendChild(xLabel);

  const yLabel = document.createElementNS('http://www.w3.org/2000/svg', 'text');
  yLabel.setAttribute('transform', `translate(-35, ${h / 2}) rotate(-90)`);
  yLabel.setAttribute('text-anchor', 'middle');
  yLabel.setAttribute('fill', 'var(--text-dim)'); yLabel.setAttribute('font-size', '11');
  yLabel.textContent = 'Observed Frequency';
  g.appendChild(yLabel);

  svg.appendChild(g);
  replaceChildren(container, svg);
}
```

### Vite Config (stripped from WM)

```typescript
// vite.config.ts — minimal, derived from WM
import { defineConfig } from 'vite';
import { resolve } from 'path';

export default defineConfig({
  resolve: {
    alias: { '@': resolve(__dirname, 'src') },
  },
  build: {
    target: 'es2020',
    sourcemap: true,
    rollupOptions: {
      output: {
        manualChunks: {
          'deckgl': ['deck.gl', '@deck.gl/core', '@deck.gl/layers',
                     '@deck.gl/aggregation-layers', '@deck.gl/mapbox'],
          'maplibre': ['maplibre-gl'],
          'd3': ['d3'],
        },
      },
    },
  },
  server: {
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },
});
```

### tsconfig.json (from WM)

```json
{
  "compilerOptions": {
    "target": "ES2020",
    "useDefineForClassFields": true,
    "module": "ESNext",
    "lib": ["ES2020", "DOM", "DOM.Iterable"],
    "types": ["vite/client"],
    "skipLibCheck": true,
    "moduleResolution": "bundler",
    "allowImportingTsExtensions": true,
    "isolatedModules": true,
    "moduleDetection": "force",
    "noEmit": true,
    "strict": true,
    "noUnusedLocals": true,
    "noUnusedParameters": true,
    "noFallthroughCasesInSwitch": true,
    "noUncheckedIndexedAccess": true,
    "resolveJsonModule": true,
    "esModuleInterop": true,
    "baseUrl": ".",
    "paths": {
      "@/*": ["src/*"]
    }
  },
  "include": ["src"]
}
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Streamlit dashboard (v1.0) | Vanilla TypeScript + deck.gl (v2.0) | 2026-02-27 decision | Full frontend rewrite; Streamlit removed entirely |
| React/Vue framework | WM's vanilla TypeScript + h() | Architecture decision | No framework dep, no virtual DOM overhead, direct DOM control |
| Mapbox GL | maplibre-gl (open source fork) | WM architecture | No API token required; free CARTO vector tiles |
| deck.gl 8.x | deck.gl 9.2.x | WM uses 9.2.6 | MapboxOverlay API, improved TypeScript types |

**Key version facts from WM's package.json:**
- deck.gl: `^9.2.6`
- maplibre-gl: `^5.16.0`
- d3: `^7.9.0`
- TypeScript: `^5.7.2`
- Vite: `^6.0.7`

## Open Questions

1. **GeoJSON source for country boundaries**
   - What we know: WM loads `/data/countries.geojson` from the public directory. This is a static file containing all country polygons with ISO codes.
   - What's unclear: Where did WM get this file? Natural Earth? Need to source a similar file for Geopol.
   - Recommendation: Use Natural Earth's `ne_110m_admin_0_countries.geojson` (110m resolution is sufficient for country-level choropleth, smaller file). Ensure ISO 3166-1 Alpha-2 codes are present.

2. **GDELT Event Heatmap data format**
   - What we know: The heatmap needs lat/lon points with weights. GDELT events have GeoLocations.
   - What's unclear: There's no dedicated GDELT events API endpoint yet (Phase 10 territory). The mock fixtures don't include raw GDELT event positions.
   - Recommendation: Define the frontend interface now (array of `{lat, lon, weight}`), build with empty data and a mock generator. The endpoint can be added in Phase 10.

3. **KnowledgeGraphArcs data source**
   - What we know: CONTEXT.md says arcs appear on country hover/select, showing actor-to-actor relations.
   - What's unclear: There's no API endpoint for knowledge graph subgraph queries. The EvidenceDTO has `source` and `gdelt_event_id` but no entity relationship data.
   - Recommendation: This layer can be populated from `ScenarioDTO.entities` pairs when a country is selected. Each scenario's entities list implies relationships. Full KG subgraph query is Phase 10+ scope.

4. **ScenarioZones layer semantics**
   - What we know: CONTEXT.md lists this as one of 5 initial layers.
   - What's unclear: How are "scenario zones" visualized? Highlight countries mentioned in active scenarios? Draw circles around scenario-relevant regions?
   - Recommendation: Implement as a GeoJsonLayer that highlights countries mentioned in `ScenarioDTO.entities` when a forecast is selected/hovered. Uses the same choropleth GeoJSON, different fill logic.

5. **Force-directed graph for Entity Relations**
   - What we know: CONTEXT.md says Entity Relations tab has toggle between force-directed graph and structured table.
   - What's unclear: What entities and relationships are available? The API serves ScenarioDTO with `entities: string[]` but no explicit relationship edges.
   - Recommendation: Build entity pairs from evidence co-occurrence within scenarios. Two entities in the same scenario = one edge, weighted by scenario probability. d3-force is the right tool.

## Sources

### Primary (HIGH confidence)
- WM Panel.ts: `/home/kondraki/personal/worldmonitor/src/components/Panel.ts` — Full Panel base class source
- WM AppContext: `/home/kondraki/personal/worldmonitor/src/app/app-context.ts` — Singleton interface
- WM dom-utils.ts: `/home/kondraki/personal/worldmonitor/src/utils/dom-utils.ts` — h() helper, 148 lines
- WM CircuitBreaker: `/home/kondraki/personal/worldmonitor/src/utils/circuit-breaker.ts` — 280 lines
- WM RefreshScheduler: `/home/kondraki/personal/worldmonitor/src/app/refresh-scheduler.ts` — 108 lines
- WM DeckGLMap: `/home/kondraki/personal/worldmonitor/src/components/DeckGLMap.ts` — Globe + 36 layers
- WM CountryBriefPage: `/home/kondraki/personal/worldmonitor/src/components/CountryBriefPage.ts` — Tab-based modal
- WM CIIPanel: `/home/kondraki/personal/worldmonitor/src/components/CIIPanel.ts` — Risk index pattern
- WM theme-manager: `/home/kondraki/personal/worldmonitor/src/utils/theme-manager.ts` — 86 lines
- WM theme-colors: `/home/kondraki/personal/worldmonitor/src/utils/theme-colors.ts` — CSS color cache
- WM main.css: `/home/kondraki/personal/worldmonitor/src/styles/main.css` — CSS variable system
- WM vite.config.ts: `/home/kondraki/personal/worldmonitor/vite.config.ts` — Build config
- WM tsconfig.json: `/home/kondraki/personal/worldmonitor/tsconfig.json` — TypeScript config
- WM package.json: `/home/kondraki/personal/worldmonitor/package.json` — Dependency versions
- WM panel-layout.ts: `/home/kondraki/personal/worldmonitor/src/app/panel-layout.ts` — Panel grid management
- WM country-geometry.ts: `/home/kondraki/personal/worldmonitor/src/services/country-geometry.ts` — GeoJSON + hit testing
- Geopol forecast schemas: `/home/kondraki/personal/geopol/src/api/schemas/forecast.py` — ForecastResponse DTO
- Geopol country schemas: `/home/kondraki/personal/geopol/src/api/schemas/country.py` — CountryRiskSummary DTO
- Geopol health schemas: `/home/kondraki/personal/geopol/src/api/schemas/health.py` — HealthResponse DTO
- Geopol common schemas: `/home/kondraki/personal/geopol/src/api/schemas/common.py` — PaginatedResponse
- Geopol forecast routes: `/home/kondraki/personal/geopol/src/api/routes/v1/forecasts.py` — All forecast endpoints
- Geopol country routes: `/home/kondraki/personal/geopol/src/api/routes/v1/countries.py` — Country risk endpoints
- Geopol health routes: `/home/kondraki/personal/geopol/src/api/routes/v1/health.py` — Health endpoint

### Secondary (MEDIUM confidence)
- [D3 Hierarchy Tree](https://d3js.org/d3-hierarchy/tree) — d3.tree() layout algorithm
- [D3 Force](https://d3js.org/d3-force) — forceSimulation() for entity graphs
- [deck.gl GeoJsonLayer docs](https://deck.gl/docs/api-reference/layers/geojson-layer) — Choropleth pattern

### Tertiary (LOW confidence)
- [Treant.js](https://fperucic.github.io/treant-js/) — Considered but rejected (heavyweight, jQuery-era)
- [Chartist](https://gionkunz.github.io/chartist-js/) — Considered but D3 sub-modules are already in deps

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — locked by CONTEXT.md decisions and WM's proven versions
- Architecture: HIGH — patterns read directly from WM source code, not documentation
- Pitfalls: HIGH — observed patterns from WM's codebase, not theoretical
- Visualizations (scenario tree, calibration, entity graph): MEDIUM — d3 approach is standard but exact rendering code needs iteration
- Open questions (KG arcs, ScenarioZones): MEDIUM — data availability depends on API endpoints not yet built

**Research date:** 2026-03-02
**Valid until:** 2026-04-02 (stable — WM codebase and Geopol API are under our control)
