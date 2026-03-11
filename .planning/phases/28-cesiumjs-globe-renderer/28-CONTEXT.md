# Phase 28: CesiumJS Globe Renderer - Context

**Gathered:** 2026-03-12
**Status:** Ready for planning

<domain>
## Phase Boundary

Replace the dual-renderer globe architecture (globe.gl/Three.js 3D + deck.gl/MapboxOverlay/MapLibre 2D) with a single CesiumJS Viewer. This is a renderer migration: same data contracts, same overlay components, same event wiring, different rendering engine.

**What changes:** The rendering stack — 3 component files deleted, 1 created, 6+ npm dependencies removed, 1-2 added. NavBar toggle becomes a 3-mode segmented control. Vite build config updated.

**What does NOT change:** Backend API, forecast-client.ts, data types (HexBinDatum, BilateralArcDatum, RiskDeltaDatum, CountryRiskSummary, ForecastResponse), overlay components (GlobeHud, GlobeDrillDown, LayerPillBar), country-geometry.ts, globe-screen.ts data loading logic, CustomEvent contracts, or any non-globe screen.

</domain>

<decisions>
## Implementation Decisions

### Motivation / Problems Being Solved

The dual-renderer architecture has 4 unfixable defects that CesiumJS eliminates:

1. **Three.js z-fighting / polygon tearing** — globe.gl renders extruded Three.js geometries on a sphere with a linear depth buffer. Countries with complex multi-polygon boundaries (Somalia, Russia, US) get depth buffer precision artifacts: chunks cut out, patches flickering. CesiumJS uses a logarithmic depth buffer — no z-fighting even at sub-meter altitude differences.

2. **MapboxOverlay interleaved-mode init race** — `@deck.gl/mapbox` MapboxOverlay in `interleaved: true` mode crashes with `Cannot read properties of null (reading 'id')` in DrawLayersPass. The Deck constructor's animation loop races against MapLibre's synchronous `addLayer` render cycle. Currently mitigated by `interleaved: false` (separate overlay canvas), but this is a workaround, not a fix.

3. **Dual WebGL contexts** — Two simultaneous WebGL contexts (MapLibre + globe.gl) consume GPU memory and risk context loss on resource-constrained devices. Both contexts remain alive during CSS display toggle (by design, for instant switching). CesiumJS provides a single WebGL context for all scene modes.

4. **Three.js deprecation warnings** — `THREE.Clock: .getElapsedTime() now supports stopped clocks` and NaN bounding sphere warnings in console from globe.gl internals. Cannot be suppressed without patching globe.gl.

### File Deletion Plan

**Files to DELETE (total ~2,239 lines):**
- `frontend/src/components/GlobeMap.ts` (~1,112 lines) — globe.gl 3D renderer
- `frontend/src/components/DeckGLMap.ts` (~800 lines) — deck.gl/MapLibre 2D renderer
- `frontend/src/components/MapContainer.ts` (~327 lines) — dual-renderer dispatch wrapper

**File to CREATE:**
- `frontend/src/components/CesiumMap.ts` — single CesiumJS renderer (~800-1,000 lines estimated)

### Dependency Changes

**npm packages to REMOVE from package.json:**
- `globe.gl` (^2.45.0)
- `three` (^0.183.2)
- `@types/three` (^0.183.1, devDependency)
- `@deck.gl/mapbox` (^9.2.6)
- `@deck.gl/core` (^9.2.6)
- `@deck.gl/layers` (^9.2.6)
- `@deck.gl/geo-layers` (^9.2.11)
- `@deck.gl/aggregation-layers` (^9.2.6)
- `deck.gl` (^9.2.6)
- `maplibre-gl` (^5.16.0)

All 10 packages are consumed ONLY by the 3 files being deleted. Verified: no other file in `frontend/src/` imports from these packages (only type re-exports through the deletion targets).

**npm packages to ADD:**
- `cesium` — CesiumJS core library (MIT licensed)
- `vite-plugin-cesium` (or equivalent) — handles CesiumJS static asset copying (Workers/, Assets/) and `CESIUM_BASE_URL` injection for both dev server and production build

**npm packages RETAINED (not affected):**
- `h3-js` (^4.4.0) — still needed for heatmap H3 hex center coordinate conversion (`cellToLatLng`)
- `d3` (^7.9.0) — used by ScenarioExplorer (d3-hierarchy, d3-zoom) and BacktestingPanel (d3 SVG charts); NOT used by any map code being deleted

### CesiumJS Viewer Configuration

Self-hosted CesiumJS — NO Cesium Ion token required. No terrain, no 3D tiles, no geocoder, no cloud services.

Viewer constructor stripped down:
```
timeline: false
animation: false
baseLayerPicker: false
geocoder: false
homeButton: false
sceneModePicker: false    // we control this ourselves via segmented control
navigationHelpButton: false
infoBox: false            // we use custom tooltip, not Cesium InfoBox
selectionIndicator: false // we use custom selection highlight
creditContainer: <custom-hidden-div>  // or creditDisplay: false equivalent
```

### Basemap

CartoDB dark-matter tiles via `UrlTemplateImageryProvider`:
- URL template: `https://basemaps.cartocdn.com/dark_all/{z}/{x}/{y}@2x.png` (or equivalent)
- Same tiles currently used by MapLibre in the 2D view
- No API key required, free tier, global coverage
- `earth-topo-bathy.jpg` static texture (currently used by globe.gl) is DROPPED — CesiumJS tile-based imagery is higher quality

### Atmosphere

CesiumJS built-in `scene.skyAtmosphere` — enabled by default. Realistic atmospheric scattering effect (blue haze at horizon, natural edge glow). Replaces the custom Three.js dual-BackSide SphereGeometry hack in `#4080dd`. No custom shader code needed.

### Auto-Rotate

REMOVED. Globe stays static when not interacted with. No idle timeout, no rotation animation. Simpler and less distracting for analysis work. The current 120s idle auto-rotate (GlobeMap.ts lines 753-790) is not reimplemented.

### 3D/2D Toggle — Three Scene Modes

CesiumJS exposes THREE scene modes. ALL THREE are exposed to the user:

1. **3D** (`Cesium.SceneMode.SCENE3D`) — spherical globe, default
2. **Columbus** (`Cesium.SceneMode.COLUMBUS_VIEW`) — 2.5D flat surface with perspective/tilt
3. **2D** (`Cesium.SceneMode.SCENE2D`) — flat Mercator projection

**Transition animation:** Animated morph via `scene.morphTo3D()`, `scene.morphToColumbusView()`, `scene.morphTo2D()`. CesiumJS handles the smooth ~2s transition where the globe flattens/inflates/tilts. No instant swap, no CSS display toggle.

**Default mode:** 3D (sphere) for first-time visitors. Preference persisted in localStorage key `geopol-globe-mode`. Stored values: `'3d'`, `'columbus'`, `'2d'`.

**No dual-renderer, no dual WebGL contexts.** A single CesiumJS Viewer handles all three modes internally. MapContainer's CSS display swap architecture is entirely eliminated.

### NavBar Segmented Control

The current NavBar has a single toggle button (`<button class="nav-view-toggle">`) that cycles between "3D" and "2D" text. This is REPLACED by a **segmented control** (3 small pills side-by-side):

```
[3D] [CV] [2D]
```

- Active pill is highlighted (same accent color `#4080dd`)
- Clicking any pill directly jumps to that scene mode (no cycling)
- Visible ONLY on `/globe` route (same as current toggle)
- Dispatches `globe-view-toggle` CustomEvent with `{ mode: '3d' | 'columbus' | '2d' }` (changed from no-payload toggle)
- Listens for `globe-mode-changed` CustomEvent to confirm and sync pill state

**Event contract change:** `globe-view-toggle` CustomEvent gains a `detail.mode` payload. `globe-mode-changed` CustomEvent `detail.mode` gains `'columbus'` as a third possible value.

### CesiumMap Public API

CesiumMap.ts must implement the following public methods (same contract as MapContainer + DeckGLMap + GlobeMap combined, minus the dual-dispatch):

**Data push methods (called by globe-screen.ts push functions):**
```typescript
updateRiskScores(summaries: CountryRiskSummary[]): void
updateForecasts(forecasts: ForecastResponse[]): void
updateHeatmapData(data: HexBinDatum[]): void
updateArcData(data: BilateralArcDatum[]): void
updateRiskDeltas(deltas: RiskDeltaDatum[]): void
```

**Selection methods (called by globe-screen.ts event handlers):**
```typescript
setSelectedCountry(iso: string | null): void
setSelectedForecast(forecast: ForecastResponse | null): void
```

**Camera methods:**
```typescript
flyToCountry(iso: string): void
flyToRegion(region: string): void
```

**Layer visibility methods (satisfies LayerController interface for LayerPillBar):**
```typescript
setLayerVisible(layerId: LayerId, visible: boolean): void
getLayerVisible(layerId: LayerId): boolean
setLayerDefaults(defaults: Partial<Record<LayerId, boolean>>): void
```

**Scene mode methods (new — replaces MapContainer.toggleMode/getActiveMode):**
```typescript
setSceneMode(mode: '3d' | 'columbus' | '2d'): void
getSceneMode(): '3d' | 'columbus' | '2d'
```

**Lifecycle:**
```typescript
constructor(container: HTMLElement)
destroy(): void
```

**Removed methods (no longer needed):**
- `getMap()` — returned MapLibre Map instance. No MapLibre, no need.
- `pauseAnimation()` / `resumeAnimation()` — were for GPU savings when 3D view hidden via CSS. CesiumJS has one context, always visible.

**Layer state:** Single layer visibility record (no more independent `layerState3d`/`layerState2d`). CesiumJS renders the same entities in all scene modes — toggle the entity's `show` property.

### Exported Types

The following types are currently exported from `DeckGLMap.ts` and re-exported through `MapContainer.ts`:

```typescript
export const LAYER_IDS = [
  'ForecastRiskChoropleth',
  'ActiveForecastMarkers',
  'KnowledgeGraphArcs',
  'GDELTEventHeatmap',
  'ScenarioZones',
] as const;

export type LayerId = (typeof LAYER_IDS)[number];

export interface HexBinDatum {
  h3_index: string;
  weight: number;
  event_count: number;
}

export interface BilateralArcDatum {
  sourceIso: string;
  targetIso: string;
  source: [number, number];  // [lng, lat]
  target: [number, number];  // [lng, lat]
  eventCount: number;
  avgGoldstein: number;
}

export interface RiskDeltaDatum {
  iso: string;
  delta: number;  // positive = worsening, negative = improving
}
```

These types MOVE to `CesiumMap.ts` (or a shared types file). `LayerPillBar.ts` and `globe-screen.ts` re-pointed to import from the new location.

### Layer Stacking via Height Offset

Polygon layers use explicit height offset to prevent z-fighting:

- **Layer 1 (ForecastRiskChoropleth):** `height: 0` (ground-clamped)
- **Layer 5 (ScenarioZones):** `height: 200` (~200 meters above surface)
- **Layers 2/3/4 (markers, arcs, heatmap):** Rendered as billboards/polylines/points which naturally sit above polygon surfaces

CesiumJS logarithmic depth buffer resolves even 50-meter offsets artifact-free. This is conceptually the same as the current globe.gl altitude discrimination (0.002 vs 0.004 globe radii) but works correctly due to log depth precision.

### Layer 1: Forecast Risk Choropleth

- Implementation: `GeoJsonDataSource` or individual `Entity` polygons
- Data source: `country-geometry.ts` GeoJSON features, colored by `riskScores` map
- Fill: Semi-transparent over basemap tiles (~0.5-0.7 alpha). Basemap labels/coastlines visible through polygons. Blue-to-red diverging color scale for risk scores [0, 1].
- Height: 0 (ground-clamped)
- Click: `scene.pick()` identifies country Entity, dispatches `country-selected` CustomEvent with `{ iso: code }`
- Hover: Outline-only highlight (no fill change). Bright polyline border around hovered country.
- Selection: Outline-only highlight. Bright polyline border around selected country.

### Layer 2: Active Forecast Markers

- Implementation: `BillboardCollection` with canvas-rendered pin/flag sprite
- Data: Marker at each forecast's country centroid (from `country-geometry.getCentroid()`)
- Visual: Pin/flag icons, NOT plain dots. Color-coded by probability (same diverging scale).
- Click: Dispatches `country-selected` CustomEvent with `{ iso: marker.iso }`
- Hover: Tooltip with country name, truncated question (80 chars), probability %

Internal `MarkerDatum` structure (private, not exported):
```typescript
interface MarkerDatum {
  iso: string;
  position: [number, number];  // [lng, lat]
  probability: number;
  question: string;
  lat: number;
  lng: number;
}
```

### Layer 3: Knowledge Graph Arcs

- Implementation: `PolylineCollection` or `Entity` polylines
- Visual: **Parabolic arcs above surface** — arcs curve upward between countries like flight-tracker visualizations. NOT flat great-circle lines on the surface. Visually distinct from ground polygons, easier to follow crossing paths.
- Height: Peak altitude proportional to arc distance (longer arcs = higher peak)
- Color: Sentiment-based (cooperative = blue/green, conflictual = red/orange) using `avgGoldstein`
- Width: Proportional to `eventCount`
- Modes: Global bilateral arcs (default) vs per-country filtered arcs (when country selected, via `buildArcsForCountry()` logic)
- Hover: Tooltip with "Source <-> Target" country names, sentiment label, Goldstein score, event count

### Layer 4: GDELT Event Heatmap

- Implementation: `PointPrimitiveCollection` at H3 hex center coordinates
- Data: `h3-js` `cellToLatLng()` converts H3 index to lat/lng (same as current, async import retained)
- Visual: Colored dots/circles at H3 hex centers. Heat-colored by weight (cool-to-hot gradient).
- NaN guard: `Number.isFinite()` check on lat/lng from h3-js (same as current GlobeMap fix)
- Hover: Tooltip with event count and weight
- Note: User is rethinking heatmap visualization approach after this phase. Current implementation is acceptable as baseline — dots at hex centers, not hex outlines.

**Performance consideration:** Use `PointPrimitiveCollection` (batch primitive), NOT individual `Entity` objects. Datasets can have thousands of H3 hexes. Entity API is too slow for this volume.

### Layer 5: Scenario Zones (Risk Deltas)

- Implementation: `Entity` polygons (same GeoJSON source as choropleth, filtered to countries with delta data)
- Height: 200m above surface (height offset from choropleth at 0m)
- Color: Red/orange for worsening (positive delta), green/teal for improving (negative delta). Semi-transparent fill.
- Hover: Tooltip with country name, direction (Worsening/Improving), delta magnitude

### Hover / Tooltip System

Full hover parity with current 2D DeckGLMap implementation. CesiumJS `ScreenSpaceEventHandler` replaces deck.gl `getTooltip` callback.

**Architecture:**
- `ScreenSpaceEventHandler` on `viewer.canvas` for `MOUSE_MOVE` events
- `scene.pick(movement.endPosition)` returns picked Entity/Primitive
- Each entity/primitive tagged with metadata (layer ID, ISO code, data reference) via `entity.properties` or primitive `id` field
- Custom positioned `<div class="map-tooltip">` identical to current DeckGLMap tooltip
- Tooltip positioned at `(screenX + 12, screenY - 12)` — same offset as current

**Tooltip content per layer (matching DeckGLMap.handleTooltip lines 837-870):**

1. **ForecastRiskChoropleth:** `<strong>Country Name</strong><br/>Risk: XX.X%`
2. **ActiveForecastMarkers:** `<strong>Country Name</strong><br/>Question text (truncated 80ch)...<br/>P: XX.X%`
3. **GDELTEventHeatmap:** `Events: N<br/>Weight: X.X`
4. **KnowledgeGraphArcs** (global mode only): `<strong>Source <-> Target</strong><br/>Sentiment (Goldstein)<br/>Events: N`
5. **ScenarioZones** (when delta data present, no forecast selected): `<strong>Country Name</strong><br/>Direction: +/-X.X pts`

**Click handler:**
- `ScreenSpaceEventHandler` for `LEFT_CLICK`
- Choropleth polygon click → dispatches `country-selected` CustomEvent `{ iso: code }`
- Marker click → dispatches `country-selected` CustomEvent `{ iso: marker.iso }`
- Choropleth click also triggers per-country arc filtering (same `buildArcsForCountry()` logic)

### VIEW_POVS Region Presets

8 region presets reimplemented as CesiumJS camera positions:

```typescript
const VIEW_POVS: Record<string, { lat: number; lng: number; altitude: number }> = {
  global:  { lat: 20,  lng:   0,  altitude: 1.8 },
  america: { lat: 20,  lng: -90,  altitude: 1.5 },
  mena:    { lat: 25,  lng:  40,  altitude: 1.2 },
  eu:      { lat: 50,  lng:  10,  altitude: 1.2 },
  asia:    { lat: 35,  lng: 105,  altitude: 1.5 },
  latam:   { lat: -15, lng: -60,  altitude: 1.5 },
  africa:  { lat:  5,  lng:  20,  altitude: 1.5 },
  oceania: { lat: -25, lng: 140,  altitude: 1.5 },
};
```

Altitude values are globe.gl-relative (globe radii). Need conversion to CesiumJS camera height (meters above ellipsoid). Approximate: `altitude * EARTH_RADIUS` where EARTH_RADIUS ≈ 6,371,000m. So altitude 1.8 ≈ 11,467,800m camera height.

Camera fly-to via `viewer.camera.flyTo({ destination: Cartesian3.fromDegrees(lng, lat, heightMeters) })`.

### globe-screen.ts Rewiring

The globe screen orchestrator (`frontend/src/screens/globe-screen.ts`) is SIMPLIFIED:

**Current (6 dynamic imports, dual sub-containers):**
```typescript
const [MapContainerClass, DeckGLMap, GlobeMap, ...] = await Promise.all([
  import('@/components/MapContainer'),
  import('@/components/DeckGLMap'),
  import('@/components/GlobeMap'),
  ...
]);
const deckEl = h('div', ...);
const globeEl = h('div', ...);
const deckMap = new DeckGLMap(deckEl);
const globeMap = new GlobeMap(globeEl);
mapContainer = new MapContainerClass(deckEl, globeEl, deckMap, globeMap);
```

**New (single import, single container):**
```typescript
const [{ CesiumMap }, ...] = await Promise.all([
  import('@/components/CesiumMap'),
  ...
]);
cesiumMap = new CesiumMap(mapEl);
```

- No dual sub-containers (`deckEl`, `globeEl` eliminated)
- No MapContainer wrapper
- `maplibre-gl/dist/maplibre-gl.css` import removed
- `CesiumMap` replaces `mapContainer` as the receiver for all data push and event methods
- `LayerPillBar` constructor takes `CesiumMap` directly (already duck-typed via `LayerController` interface)
- `setLayerDefaults()` called on CesiumMap directly

**Module-scoped state change:**
```typescript
// Old
let mapContainer: MapContainer | null = null;
// New
let cesiumMap: CesiumMap | null = null;
```

All `mapContainer.xxx()` calls in push functions, event handlers, and unmount become `cesiumMap.xxx()`.

### NavBar.ts Changes

The toggle button is replaced by a segmented control:

**Current:**
```typescript
const viewToggle = h('button', { className: 'nav-view-toggle' }, is3d ? '3D' : '2D');
viewToggle.addEventListener('click', () => {
  window.dispatchEvent(new CustomEvent('globe-view-toggle'));
});
```

**New:** Three-pill segmented control `[3D] [CV] [2D]`:
- Each pill is a `<button>` inside a container `<div class="nav-scene-mode">`
- Clicking a pill dispatches `globe-view-toggle` with `{ mode: '3d' | 'columbus' | '2d' }`
- Listens for `globe-mode-changed` to highlight the active pill
- localStorage key `geopol-globe-mode` stores `'3d'` | `'columbus'` | `'2d'`

### CesiumMap Event Listeners

CesiumMap listens for (replaces MapContainer's listeners):
- `globe-view-toggle` CustomEvent with `{ mode: '3d' | 'columbus' | '2d' }` — calls `setSceneMode()`
- `globe-region-change` CustomEvent with `{ region: string }` — calls `flyToRegion()`

CesiumMap dispatches:
- `globe-mode-changed` CustomEvent with `{ mode: '3d' | 'columbus' | '2d' }` — after scene mode transition completes
- `country-selected` CustomEvent with `{ iso: string }` — on polygon/marker click

### LayerPillBar — No Changes Needed

LayerPillBar is duck-typed via the `LayerController` interface:
```typescript
export interface LayerController {
  getLayerVisible(layerId: LayerId): boolean;
  setLayerVisible(layerId: LayerId, visible: boolean): void;
}
```

CesiumMap implements this interface. LayerPillBar's only import from `DeckGLMap` is the `LayerId` type — repoint to `CesiumMap.ts`.

LayerPillBar also listens for `globe-mode-changed` to resync pill states via `syncFromController()`. This still works — CesiumMap dispatches the same event.

### GlobeHud — No Changes Needed

GlobeHud dispatches `globe-region-change` CustomEvent. CesiumMap listens for it. No code changes.

### GlobeDrillDown — No Changes Needed

GlobeDrillDown opens on `country-selected` CustomEvent (handled by globe-screen.ts event wiring). CesiumMap dispatches the same event. No code changes.

### Vite Config Changes

**Current `vite.config.ts` manual chunks:**
```typescript
manualChunks: {
  deckgl: ['deck.gl', '@deck.gl/core', '@deck.gl/layers', '@deck.gl/aggregation-layers', '@deck.gl/mapbox'],
  maplibre: ['maplibre-gl'],
  d3: ['d3'],
  globe: ['globe.gl', 'three'],
}
```

**New:**
```typescript
manualChunks: {
  cesium: ['cesium'],  // CesiumJS in its own chunk, only loaded on /globe route
  d3: ['d3'],
}
```

`deckgl`, `maplibre`, and `globe` chunks all deleted. CesiumJS gets its own manual chunk.

Additionally, `vite-plugin-cesium` (or equivalent) must be added to handle:
- Copying `node_modules/cesium/Build/Cesium/Workers/` to build output
- Copying `node_modules/cesium/Build/Cesium/Assets/` to build output
- Setting `CESIUM_BASE_URL` for both `vite dev` and `vite build`
- CesiumJS CSS import (`cesium/Build/Cesium/Widgets/widgets.css` or equivalent, though most widgets are disabled)

### Loading UX

- **Init loading:** Same skeleton shimmer as current (Phase 25 pattern). CesiumJS Viewer container hidden until ready, then revealed.
- **Tile loading:** Globe shown IMMEDIATELY when Viewer is ready. CartoDB tiles load progressively (blurry → sharp). Natural CesiumJS behavior — no waiting for specific zoom level.
- **Code splitting:** CesiumJS module dynamically imported at route level. Only loaded when `/globe` mounts. Dashboard and Forecasts screens load zero CesiumJS code.
- **Error state:** Same `'Globe failed to load'` placeholder in globe-screen.ts catch block. No fallback image.

### Performance Notes for Researcher/Planner

- **PointPrimitiveCollection** (not Entity) for heatmap layer — Entity API is O(n) per frame for large counts
- **PolylineCollection** (not Entity) for arcs — same performance reasoning
- **GeoJsonDataSource** is acceptable for choropleth (~195 polygons) — not performance-critical at this count
- **BillboardCollection** for markers — efficient for up to thousands of billboards
- CesiumJS bundle is large (~40MB uncompressed with workers/assets). Tree-shaking limited. Static worker/asset files served separately. Gzipped JS is ~3-4MB — comparable to current deck.gl + globe.gl + three + maplibre total.

### Claude's Discretion

- Exact pin/flag marker icon design (canvas-rendered sprite dimensions, shape, shadow)
- Parabolic arc height interpolation formula (peak altitude as function of surface distance)
- CesiumJS camera animation duration/easing for flyTo operations
- Tooltip hide/show timing and debouncing
- Whether to use `GeoJsonDataSource.load()` or manually construct `Entity` polygons for choropleth
- Columbus View camera orientation defaults
- Segmented control pill sizing and spacing within NavBar
- Whether to import CesiumJS CSS or strip it entirely (most widgets disabled)
- H3 hex rendering as circles vs squares vs other point shapes

</decisions>

<specifics>
## Specific Ideas

- Layer stacking uses explicit height offset (choropleth at 0m, scenario zones at 200m) — CesiumJS logarithmic depth buffer makes this work cleanly, unlike globe.gl's linear depth buffer which caused the tearing artifacts
- Hover/selection uses outline-only highlight (bright polyline border around country), NOT fill color change. Cleaner, less flickery on complex multi-polygon countries
- Parabolic arcs should feel like flight-tracker visualizations — curves above the surface, visually distinct from ground layers
- Markers are pin/flag icons, not plain colored dots — more visually distinct on the globe
- No auto-rotate — removed entirely. Globe is an analysis tool, not a screensaver.
- Animated morph transitions between scene modes (3D unfolds to 2D and back) — CesiumJS does this natively, visually impressive
- The current `earth-topo-bathy.jpg` static globe texture is dropped. CesiumJS tile-based imagery from CartoDB is higher quality and supports progressive loading at all zoom levels.

</specifics>

<deferred>
## Deferred Ideas

- **Heatmap rethink:** User plans to rethink the entire heatmap visualization approach after this phase. Current hex-center dot implementation is acceptable as a baseline for Phase 28, but a future phase may replace it with something fundamentally different (hex outlines, kernel density, animated particles, etc.)
- **Cesium Ion integration:** Self-hosted for now. If terrain, 3D tiles, or higher-quality basemaps are wanted later, a Cesium Ion token can be added without code changes (just config).
- **Columbus View layer adjustments:** Some layers may benefit from different visual treatment in Columbus View vs 3D/2D. Not in scope — same rendering across all three modes for now.

</deferred>

---

*Phase: 28-cesiumjs-globe-renderer*
*Context gathered: 2026-03-12*
