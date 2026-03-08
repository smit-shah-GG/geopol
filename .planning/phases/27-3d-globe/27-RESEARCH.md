# Phase 27: 3D Globe - Research

**Researched:** 2026-03-09
**Domain:** globe.gl (Three.js-based 3D globe), dual-renderer wrapper, data layer porting
**Confidence:** HIGH

## Summary

Phase 27 adds a globe.gl-based 3D spherical globe as the default `/globe` view, retaining the existing deck.gl 2D flat map as a toggleable alternative. The research investigated Geopol's current DeckGLMap public API (886 lines, 13 public methods), the globe-screen orchestration (340 lines), and World Monitor's GlobeMap.ts (2,188 lines) and MapContainer.ts (921 lines) as code quarries.

The standard approach is: (1) new `GlobeMap` class wrapping globe.gl with Geopol's 5-layer data API, (2) new `MapContainer` wrapper that holds both `GlobeMap` and `DeckGLMap` instances and dispatches all calls based on active mode, (3) modified `globe-screen.ts` that instantiates `MapContainer` instead of `DeckGLMap` directly, (4) view toggle button added to the NavBar, (5) region preset selector added to GlobeHud.

**Primary recommendation:** Follow WM's delegation pattern faithfully. The MapContainer caches all data pushes and callback registrations, surviving view mode switches. Both WebGL contexts stay alive -- toggle swaps CSS visibility, never destroys/recreates.

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| globe.gl | ^2.45.0 | 3D globe rendering (Three.js wrapper) | WM uses this exact version; stable, well-maintained by vasturiano |
| three | >=0.154 <1 | WebGL rendering engine (peer dep of globe.gl) | Required peer dependency -- globe.gl delegates all 3D to Three.js |
| @types/three | ^0.183.1 | TypeScript definitions for three.js | WM uses this version for type safety |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| (existing) deck.gl | ^9.2.6 | Flat 2D map -- already installed | Retained as toggle alternative |
| (existing) maplibre-gl | ^5.16.0 | Basemap tiles for 2D mode | Already installed, no change |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| globe.gl | cesiumjs | Overkill -- Cesium is a full geospatial platform; globe.gl is lightweight and WM-proven |
| globe.gl | react-globe.gl | Geopol is NOT React -- vanilla TS only |
| htmlElementsData | pointsData | pointsData uses WebGL sprites (can't do custom DOM); htmlElementsData allows rich HTML markers |

**Installation:**
```bash
cd frontend && npm install globe.gl three @types/three
```

**Vite config addition** (new manual chunk):
```typescript
manualChunks: {
  // existing chunks...
  globe: ['globe.gl', 'three'],
}
```

## Architecture Patterns

### Current DeckGLMap Public API (13 methods -- MapContainer MUST delegate ALL)

From reading `DeckGLMap.ts` (886 lines):

| Method | Signature | Description |
|--------|-----------|-------------|
| `updateRiskScores` | `(summaries: CountryRiskSummary[]) => void` | Push country risk data for choropleth |
| `updateForecasts` | `(forecasts: ForecastResponse[]) => void` | Push forecast data for markers + scenarios |
| `setSelectedCountry` | `(iso: string \| null) => void` | Select country, build per-country arcs |
| `setSelectedForecast` | `(forecast: ForecastResponse \| null) => void` | Highlight scenario-relevant countries |
| `updateHeatmapData` | `(data: HexBinDatum[]) => void` | Push H3 hexbin data for heatmap layer |
| `updateArcData` | `(data: BilateralArcDatum[]) => void` | Push bilateral arc data (global view) |
| `updateRiskDeltas` | `(deltas: RiskDeltaDatum[]) => void` | Push risk delta data for scenario zones |
| `flyToCountry` | `(iso: string) => void` | Animate camera to country centroid |
| `setLayerVisible` | `(layerId: LayerId, visible: boolean) => void` | Toggle individual layer visibility |
| `setLayerDefaults` | `(defaults: Partial<Record<LayerId, boolean>>) => void` | Batch-set layer visibility |
| `getLayerVisible` | `(layerId: LayerId) => boolean` | Query layer visibility state |
| `getMap` | `() => maplibregl.Map \| null` | Access underlying map (2D only) |
| `destroy` | `() => void` | Clean up all resources |

### GlobeMap Must Implement (same data API, different rendering)

The `GlobeMap` class MUST expose the exact same 13 public methods as `DeckGLMap`. Internally, it maps them to globe.gl channels:

| DeckGLMap Method | GlobeMap Channel | Notes |
|------------------|------------------|-------|
| `updateRiskScores` | `polygonsData` | GeoJSON country fills with risk-colored caps |
| `updateForecasts` | `htmlElementsData` | DOM markers at lat/lon |
| `setSelectedCountry` | Rebuild arcs + fly-to | `pointOfView()` with 1.2s animation |
| `setSelectedForecast` | Rebuild polygon overlays | Scenario country highlights |
| `updateHeatmapData` | `htmlElementsData` or `pointsData` | Colored dots at H3 hex centers (no hex tessellation) |
| `updateArcData` | `arcsData` | Great-circle arcs with sentiment coloring |
| `updateRiskDeltas` | `polygonsData` | Red/green country overlays |
| `flyToCountry` | `pointOfView()` | 1-1.5s animation to country centroid |
| `setLayerVisible` | Internal toggle + rebuild | Same LayerId enum |
| `setLayerDefaults` | Internal toggle + rebuild | Same LayerId enum |
| `getLayerVisible` | Read internal state | Same LayerId enum |
| `getMap` | Returns `null` | 3D mode has no maplibre Map |
| `destroy` | `globe._destructor()` + cleanup | Dispose Three.js objects |

### Recommended File Structure
```
frontend/src/
├── components/
│   ├── DeckGLMap.ts          # EXISTING (886 lines) -- unchanged
│   ├── GlobeMap.ts           # NEW -- globe.gl 3D renderer, same API as DeckGLMap
│   ├── MapContainer.ts       # NEW -- wrapper dispatching to GlobeMap or DeckGLMap
│   ├── GlobeHud.ts           # MODIFIED -- add region preset selector
│   ├── LayerPillBar.ts       # MODIFIED -- accept MapContainer instead of DeckGLMap
│   ├── NavBar.ts             # MODIFIED -- add 3D/2D view toggle button
│   ├── GlobeDrillDown.ts     # UNCHANGED
│   └── ...
├── screens/
│   └── globe-screen.ts       # MODIFIED -- instantiate MapContainer, not DeckGLMap
└── ...
```

### Pattern 1: MapContainer Delegation (adapted from WM)

**What:** Wrapper class that holds both renderers, dispatches all calls conditionally, caches data for mode switches.

**When to use:** Always -- this is the integration layer between globe-screen and both renderers.

**Key differences from WM's MapContainer:**
- WM has 3 renderers (DeckGLMap, SVGMap, GlobeMap) -- Geopol has 2 (DeckGLMap, GlobeMap)
- WM has 30+ data setters -- Geopol has 7 data setters
- WM has complex callback caching for 6+ callbacks -- Geopol needs 0 callback caching (events are window CustomEvents, not callbacks)
- WM does NOT cache event callbacks on its DeckGLMap (uses `.setOnCountryClick()`) -- Geopol dispatches `country-selected` as a window event, so the MapContainer needs to wire globe.gl's `onPolygonClick` to fire the same CustomEvent

**Example (adapted from WM MapContainer.ts lines 76-250):**
```typescript
export class MapContainer {
  private readonly container: HTMLElement;
  private deckMap: DeckGLMap | null = null;
  private globeMap: GlobeMap | null = null;
  private activeMode: '3d' | '2d';

  // Data cache -- survives mode switches (WM pattern)
  private cachedRiskScores: CountryRiskSummary[] | null = null;
  private cachedForecasts: ForecastResponse[] | null = null;
  private cachedHexBins: HexBinDatum[] | null = null;
  private cachedArcs: BilateralArcDatum[] | null = null;
  private cachedDeltas: RiskDeltaDatum[] | null = null;
  private cachedSelectedCountry: string | null = null;
  private cachedSelectedForecast: ForecastResponse | null = null;

  // Layer state -- independent per view (CONTEXT.md decision)
  private layerState3d: Record<LayerId, boolean>;
  private layerState2d: Record<LayerId, boolean>;

  constructor(container: HTMLElement, deckMapContainer: HTMLElement, globeMapContainer: HTMLElement) {
    this.container = container;
    const pref = localStorage.getItem('geopol-globe-mode') ?? '3d';
    this.activeMode = pref === '2d' ? '2d' : '3d';
    // Construct BOTH renderers immediately -- both WebGL contexts alive
    this.deckMap = new DeckGLMap(deckMapContainer);
    this.globeMap = new GlobeMap(globeMapContainer);
    this.applyVisibility();
  }

  // Dispatch pattern -- EVERY public method follows this
  updateRiskScores(summaries: CountryRiskSummary[]): void {
    this.cachedRiskScores = summaries;
    this.deckMap?.updateRiskScores(summaries);
    this.globeMap?.updateRiskScores(summaries);
    // Push to BOTH so toggle is instant (no re-fetch)
  }
}
```

### Pattern 2: Debounced Flush (from WM GlobeMap.ts lines 1183-1200)

**What:** Dual-timer debounce for globe.gl data updates. 100ms trailing debounce + 300ms max wait to prevent burst updates from hammering the Three.js render loop.

**When to use:** Every time data changes on the GlobeMap (markers, arcs, polygons). Globe.gl rebuilds internal geometries on every data push -- without debouncing, rapid sequential calls (risk scores + forecasts + layers all arriving within 50ms) would trigger 3 expensive rebuilds instead of 1.

**Example (from WM GlobeMap.ts):**
```typescript
private flushTimer: ReturnType<typeof setTimeout> | null = null;
private flushMaxTimer: ReturnType<typeof setTimeout> | null = null;

private scheduleFlush(): void {
  if (!this.globe || !this.initialized || this.destroyed) return;
  // Max wait: ensure we flush within 300ms even if updates keep arriving
  if (!this.flushMaxTimer) {
    this.flushMaxTimer = setTimeout(() => {
      this.flushMaxTimer = null;
      if (this.flushTimer) { clearTimeout(this.flushTimer); this.flushTimer = null; }
      this.flushImmediate();
    }, 300);
  }
  // Trailing debounce: wait 100ms after last update
  if (this.flushTimer) clearTimeout(this.flushTimer);
  this.flushTimer = setTimeout(() => {
    this.flushTimer = null;
    if (this.flushMaxTimer) { clearTimeout(this.flushMaxTimer); this.flushMaxTimer = null; }
    this.flushImmediate();
  }, 100);
}
```

### Pattern 3: Atmosphere Glow (from WM GlobeMap.ts lines 1992-2056)

**What:** Custom Three.js scene objects for atmosphere glow effect, adapted with Geopol blue (#4080dd) instead of WM cyan (#00d4ff).

**Example (adapted from WM -- NO starfield per CONTEXT.md):**
```typescript
private async applyAtmosphereGlow(): Promise<void> {
  const THREE = await import('three');
  const scene = this.globe.scene();

  // Upgrade material: matte analytical look
  const oldMat = this.globe.globeMaterial();
  const stdMat = new THREE.MeshStandardMaterial({
    color: 0xffffff, roughness: 0.8, metalness: 0.1,
    emissive: new THREE.Color(0x0a1f2e), emissiveIntensity: 0.3,
  });
  if ((oldMat as any).map) stdMat.map = (oldMat as any).map;
  (this.globe as any).globeMaterial(stdMat);

  // Accent light (Geopol blue instead of WM cyan)
  const light = new THREE.PointLight(0x4080dd, 0.3);
  light.position.set(-10, -10, -10);
  scene.add(light);

  // Outer glow sphere (BackSide rendering)
  const outerGeo = new THREE.SphereGeometry(2.15, 64, 64);
  const outerMat = new THREE.MeshBasicMaterial({
    color: 0x4080dd, side: THREE.BackSide, transparent: true, opacity: 0.15,
  });
  this.outerGlow = new THREE.Mesh(outerGeo, outerMat);
  scene.add(this.outerGlow);

  // Inner glow sphere (subtler)
  const innerGeo = new THREE.SphereGeometry(2.08, 64, 64);
  const innerMat = new THREE.MeshBasicMaterial({
    color: 0x3060aa, side: THREE.BackSide, transparent: true, opacity: 0.1,
  });
  this.innerGlow = new THREE.Mesh(innerGeo, innerMat);
  scene.add(this.innerGlow);

  // NO starfield (CONTEXT.md: atmosphere glow only, no starfield)
}
```

### Pattern 4: VIEW_POVS Region Presets (from WM GlobeMap.ts lines 1590-1606)

**What:** 8 pre-defined camera positions for regional views. Port directly from WM.

```typescript
const VIEW_POVS: Record<string, { lat: number; lng: number; altitude: number }> = {
  global:   { lat: 20,  lng:  0,   altitude: 1.8 },
  america:  { lat: 20,  lng: -90,  altitude: 1.5 },
  mena:     { lat: 25,  lng:  40,  altitude: 1.2 },
  eu:       { lat: 50,  lng:  10,  altitude: 1.2 },
  asia:     { lat: 35,  lng: 105,  altitude: 1.5 },
  latam:    { lat: -15, lng: -60,  altitude: 1.5 },
  africa:   { lat:  5,  lng:  20,  altitude: 1.5 },
  oceania:  { lat: -25, lng: 140,  altitude: 1.5 },
};
```

### Pattern 5: Country Click via onPolygonClick

**What:** globe.gl has a built-in `onPolygonClick` callback. Wire it to dispatch the same `country-selected` CustomEvent that DeckGLMap uses.

**Critical note:** WM's GlobeMap.ts line 1664 has `setOnCountryClick` as a NO-OP ("Globe country click not yet implemented"). Geopol MUST implement this -- it is a success criterion. The implementation requires:
1. Setting `polygonsData` with the GeoJSON feature collection
2. Configuring `polygonGeoJsonGeometry` accessor
3. Setting `onPolygonClick` callback
4. Extracting ISO code from the clicked polygon's properties
5. Dispatching `window.dispatchEvent(new CustomEvent('country-selected', { detail: { iso } }))`

### Anti-Patterns to Avoid
- **Destroy/recreate on toggle:** Both WebGL contexts must stay alive. Destroying globe.gl's Three.js scene and recreating it takes 800ms+ (texture reload, scene graph rebuild). CSS `display:none` on the inactive container is the correct approach.
- **Push data only to active renderer:** Both renderers must receive all data pushes. Toggle must be instant with no visible flash of empty state.
- **Shared layer state between views:** CONTEXT.md explicitly requires independent layer visibility per view. Do NOT use a single `layerVisible` record.
- **Camera position transfer:** CONTEXT.md explicitly states independent camera positions. Do NOT sync camera between 2D and 3D.
- **Calling `rebuildLayers()` per data push on GlobeMap:** Use the debounced flush pattern. Globe.gl's internal data processing is heavier than deck.gl's.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| 3D globe rendering | Custom Three.js globe | `globe.gl` | Handles sphere mesh, texture mapping, orbit controls, object placement, raycasting, CSS2DRenderer for HTML overlays -- 10K+ lines of code |
| Atmosphere glow | Fresnel shader from scratch | `globe.atmosphereColor()` + custom BackSide SphereGeometry | globe.gl has built-in atmosphere; supplement with custom glow spheres per WM pattern |
| Great-circle arcs | Spherical interpolation math | `globe.arcsData()` | globe.gl handles geodesic arc computation, altitude, animation |
| Orbit camera controls | Custom Three.js OrbitControls | `globe.controls()` | Returns Three.js OrbitControls instance -- configure autoRotate, enablePan, zoom limits directly |
| GeoJSON polygon rendering | Custom spherical tessellation | `globe.polygonsData()` + `polygonGeoJsonGeometry()` | globe.gl handles spherical projection of GeoJSON polygons including MultiPolygon |
| HTML marker positioning | CSS3DRenderer from scratch | `globe.htmlElementsData()` | globe.gl handles geographic-to-screen coordinate conversion for DOM elements |
| Earth texture darkening | Image processing pipeline | `MeshStandardMaterial.emissive` + low `emissiveIntensity` | Darken at material level, not texture level. WM pattern: emissive 0x0a1f2e at 0.3 intensity |

**Key insight:** globe.gl provides 11 data channels covering all of Geopol's 5 layer types. The work is mapping Geopol's data model to globe.gl's accessor pattern, not building 3D rendering from scratch.

## Common Pitfalls

### Pitfall 1: Dual WebGL Context Limits
**What goes wrong:** Browsers limit WebGL contexts to ~8-16 per page. Two full WebGL renderers (deck.gl + globe.gl/Three.js) consume 2 contexts permanently.
**Why it happens:** Both renderers stay alive simultaneously for instant toggle.
**How to avoid:** This is acceptable for desktop-only target. Monitor via `performance.now()` frame timing. If needed, can call `globe.pauseAnimation()` on the inactive renderer to reduce GPU load.
**Warning signs:** Black/blank canvas after toggle, console warning "Too many active WebGL contexts."

### Pitfall 2: GeoJSON Ring Winding Order
**What goes wrong:** globe.gl requires counterclockwise winding for polygon outer rings (Three.js convention). Natural Earth GeoJSON uses clockwise (RFC 7946). Polygons render inside-out or invisible.
**Why it happens:** Three.js BackSide/FrontSide culling depends on winding order.
**How to avoid:** WM's GlobeMap has a `getReversedRing()` method with a cache (line 1277). Geopol needs the same: reverse the coordinates array of each polygon ring before passing to `polygonGeoJsonGeometry`.
**Warning signs:** Countries appear as dark voids or don't render at all.

### Pitfall 3: globe.gl Data Push Frequency
**What goes wrong:** Every call to `polygonsData()`, `arcsData()`, or `htmlElementsData()` triggers a full internal rebuild (geometry disposal + recreation). Calling these 5 times in rapid succession causes 5 rebuilds.
**Why it happens:** globe.gl is designed for batch updates, not incremental changes.
**How to avoid:** Debounced flush pattern (Pattern 2 above). Coalesce all data changes within a 100ms window into a single flush.
**Warning signs:** Frame drops during data refresh cycles, visible flickering.

### Pitfall 4: Dynamic Import Race with Three.js
**What goes wrong:** `import('three')` inside `applyAtmosphereGlow()` must resolve AFTER globe.gl has finished initializing its Three.js scene. If called too early, `globe.scene()` returns undefined.
**Why it happens:** Globe initialization is async (texture loading), but the constructor returns synchronously.
**How to avoid:** Wait for globe.gl's `onGlobeReady` callback or use a delay (WM uses `setTimeout(() => this.applyEnhancedVisuals(), 800)`). Better: chain off globe.gl's internal ready state.
**Warning signs:** "Cannot read properties of undefined (reading 'add')" errors.

### Pitfall 5: Country Click Not Working on 3D Globe
**What goes wrong:** WM's GlobeMap has `setOnCountryClick` as a NO-OP (line 1664). If you copy WM's pattern without implementing this, the GlobeDrillDown panel never opens.
**Why it happens:** WM never implemented polygon click detection on its globe. Geopol MUST.
**How to avoid:** Use globe.gl's `onPolygonClick` callback. Requires: (a) polygon data loaded via `polygonsData()`, (b) `polygonGeoJsonGeometry()` accessor configured, (c) click handler extracting ISO from feature properties.
**Warning signs:** Country clicks do nothing in 3D mode.

### Pitfall 6: Earth Texture Loading
**What goes wrong:** The `earth-topo-bathy.jpg` texture (699KB from WM) must be placed in Geopol's `frontend/public/textures/` and loaded via `globeImageUrl('/textures/earth-topo-bathy.jpg')`. If the path is wrong, the globe renders as a white/blank sphere.
**Why it happens:** globe.gl loads the texture asynchronously via fetch; 404 fails silently.
**How to avoid:** Copy the texture file from WM, verify path, add a console.warn on globe load failure.
**Warning signs:** White/gray sphere with no land features visible.

### Pitfall 7: LayerPillBar Type Coupling
**What goes wrong:** LayerPillBar currently takes `DeckGLMap` as its constructor argument (line 32). After Phase 27, it must accept `MapContainer` instead, or a shared interface.
**Why it happens:** Tight coupling to the concrete class.
**How to avoid:** Define a `MapRenderer` interface with `setLayerVisible()`, `getLayerVisible()`, then have both `DeckGLMap` and `MapContainer` implement it. Or simpler: have LayerPillBar accept MapContainer directly.
**Warning signs:** TypeScript compilation errors when wiring LayerPillBar to MapContainer.

## Code Examples

### globe.gl Initialization (adapted from WM GlobeMap.ts lines 414-500)
```typescript
import Globe from 'globe.gl';
import type { GlobeInstance, ConfigOptions } from 'globe.gl';

const config: ConfigOptions = {
  animateIn: false,
  rendererConfig: {
    powerPreference: 'high-performance',
    antialias: window.devicePixelRatio > 1,
  },
};

const globe = new Globe(container, config) as GlobeInstance;

globe
  .globeImageUrl('/textures/earth-topo-bathy.jpg')
  .backgroundImageUrl('')          // Black background, no starfield image
  .atmosphereColor('#4080dd')      // Geopol accent blue
  .atmosphereAltitude(0.18)
  .width(container.clientWidth)
  .height(container.clientHeight)
  .pathTransitionDuration(0);      // No animation delay on data updates

// Orbit controls
const controls = globe.controls();
controls.autoRotate = true;
controls.autoRotateSpeed = 0.3;
controls.enablePan = false;        // Rotate + zoom only (CONTEXT.md)
controls.enableZoom = true;
controls.zoomSpeed = 1.4;
controls.minDistance = 101;
controls.maxDistance = 600;
controls.enableDamping = true;
```

### Polygon Country Click Handler
```typescript
globe
  .polygonsData(geoJsonFeatures)
  .polygonGeoJsonGeometry((d: Feature) => d.geometry)
  .polygonCapColor((d: Feature) => {
    const code = normalizeCode(d.properties);
    if (!code) return 'rgba(40, 44, 52, 0.5)';
    const score = riskScores.get(code);
    if (score === undefined) return 'rgba(40, 44, 52, 0.5)';
    return riskColorCSS(score / 100);  // Must return CSS color string for globe.gl
  })
  .polygonStrokeColor('rgba(80, 85, 95, 0.6)')
  .polygonSideColor('rgba(0, 0, 0, 0)')
  .polygonAltitude(0.002)
  .onPolygonClick((polygon: Feature, event: MouseEvent, coords: {lat: number, lng: number}) => {
    const code = normalizeCode(polygon.properties);
    if (!code) return;
    window.dispatchEvent(
      new CustomEvent('country-selected', { detail: { iso: code }, bubbles: true })
    );
  });
```

### Fly-To Country (adapted from WM GlobeMap.ts lines 1751-1761)
```typescript
flyToCountry(iso: string): void {
  if (!this.globe) return;
  const centroid = countryGeometry.getCentroid(iso.toUpperCase());
  if (!centroid) return;
  // centroid is [lon, lat] -- globe.gl wants { lat, lng, altitude }
  this.globe.pointOfView(
    { lat: centroid[1], lng: centroid[0], altitude: 0.5 },
    1200  // 1.2s animation duration
  );
}
```

### Auto-Rotate with Idle Timer (adapted from WM lines 506-526)
```typescript
private autoRotateTimer: ReturnType<typeof setTimeout> | null = null;

private setupAutoRotate(): void {
  const canvas = this.container.querySelector('canvas');
  if (!canvas || !this.controls) return;

  const pause = () => {
    if (this.controls) this.controls.autoRotate = false;
    if (this.autoRotateTimer) clearTimeout(this.autoRotateTimer);
  };
  const scheduleResume = () => {
    if (this.autoRotateTimer) clearTimeout(this.autoRotateTimer);
    this.autoRotateTimer = setTimeout(() => {
      if (this.controls) this.controls.autoRotate = true;
    }, 120_000);  // 120s idle timeout (CONTEXT.md -- longer than WM's 60s)
  };

  canvas.addEventListener('mousedown', pause);
  canvas.addEventListener('mouseup', scheduleResume);
  canvas.addEventListener('wheel', () => { pause(); scheduleResume(); }, { passive: true });
}
```

### Color Functions for Globe.gl (CSS strings, not RGBA tuples)

globe.gl expects CSS color strings, NOT `[r,g,b,a]` RGBA tuples like deck.gl:
```typescript
// DeckGLMap uses:  [220, 50, 50, 180] (RGBA tuple)
// GlobeMap uses:   'rgba(220, 50, 50, 0.7)' (CSS string)

function riskColorCSS(score: number, alpha = 0.7): string {
  const t = Math.max(0, Math.min(1, score));
  if (t <= 0.5) {
    const u = t * 2;
    const r = Math.round(70 + (128 - 70) * u);
    const g = Math.round(130 + (128 - 130) * u);
    const b = Math.round(180 + (128 - 180) * u);
    return `rgba(${r},${g},${b},${alpha})`;
  }
  const u = (t - 0.5) * 2;
  const r = Math.round(128 + (220 - 128) * u);
  const g = Math.round(128 + (50 - 128) * u);
  const b = Math.round(128 + (50 - 128) * u);
  // ~10-15% brightness boost for 3D (CONTEXT.md decision)
  return `rgba(${Math.min(255, r + 20)},${Math.min(255, g + 15)},${Math.min(255, b + 15)},${alpha})`;
}
```

## Current Codebase Integration Points

### globe-screen.ts (340 lines) -- What Changes

**Current flow:**
1. Dynamic `import('@/components/DeckGLMap')` + `import('@/components/LayerPillBar')` + modals
2. `new DeckGLMap(mapContainer)` -- direct construction
3. `new LayerPillBar(deckMap)` -- direct coupling
4. `deckMap.setLayerDefaults({...})` -- on DeckGLMap
5. `wireEvents()` -- reads `deckMap` from module scope
6. `pushCountries/pushForecasts/pushHeatmap/pushArcs/pushDeltas` -- push to `deckMap`
7. `unmountGlobe()` -- calls `deckMap.destroy()`

**After Phase 27:**
1. Dynamic `import('@/components/MapContainer')` (which internally imports DeckGLMap + GlobeMap)
2. `new MapContainer(mapContainer)` -- wrapper construction (builds both renderers)
3. `new LayerPillBar(mapContainer)` -- takes wrapper
4. `mapContainer.setLayerDefaults({...})` -- dispatches to active renderer
5. `wireEvents()` -- reads `mapContainer` from module scope (country-selected events work for both)
6. All `push*` functions call `mapContainer.updateRiskScores()` etc. (dispatches to both)
7. `unmountGlobe()` -- calls `mapContainer.destroy()` (destroys both)

### NavBar.ts (78 lines) -- What Changes

**Current:** 3 nav links (Dashboard, Globe, Forecasts) with active-state highlighting.

**After Phase 27:** Add a view toggle button AFTER the nav links, visible ONLY on `/globe` route. Toggle dispatches a `globe-view-toggle` CustomEvent that `globe-screen.ts` listens for.

```typescript
// In nav-links container, add conditionally visible toggle
const viewToggle = h('button', {
  className: 'nav-view-toggle',
  'aria-label': 'Toggle 3D/2D view',
  'data-active-route': '/globe',  // Only show on globe route
}, '3D');

viewToggle.addEventListener('click', () => {
  window.dispatchEvent(new CustomEvent('globe-view-toggle'));
});
```

### GlobeHud.ts (75 lines) -- What Changes

**Current:** 3 stats (FORECASTS, COUNTRIES, UPDATED).

**After Phase 27:** Add region preset selector below the stats. 8 buttons (Global, Americas, MENA, Europe, Asia, LatAm, Africa, Oceania) that dispatch `globe-view-change` CustomEvents.

### LayerPillBar.ts (89 lines) -- What Changes

**Current:** Constructor takes `DeckGLMap`, calls `deckMap.setLayerVisible()` / `deckMap.getLayerVisible()`.

**After Phase 27:** Constructor takes `MapContainer` (or a `MapRenderer` interface). Dispatches to whichever renderer is active. Per CONTEXT.md, each view has independent layer state -- the pill bar must read/write from the active view's state.

### Asset Requirements

| Asset | Source | Size | Destination |
|-------|--------|------|-------------|
| `earth-topo-bathy.jpg` | `/home/kondraki/personal/worldmonitor/public/textures/` | 699KB | `frontend/public/textures/earth-topo-bathy.jpg` |

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| globe.gl v1.x (Globe constructor, no types) | globe.gl v2.45.0 (typed GlobeInstance, ConfigOptions) | 2024 | Better TypeScript support, configurable renderer |
| deck.gl only (2D projection) | globe.gl + deck.gl dual renderer | Phase 27 | True 3D spherical view with atmosphere |

**Deprecated/outdated:**
- WM's `GlobeMap.setOnCountryClick` is a no-op -- Geopol MUST implement this via `onPolygonClick`
- WM's starfield pattern -- explicitly deferred in CONTEXT.md (atmosphere glow only)
- WM's performance preset system (Eco/Sharp toggle) -- CONTEXT.md says one fixed config

## Open Questions

1. **globe.gl polygon click raycasting reliability at low polygon altitude**
   - What we know: globe.gl `onPolygonClick` works. WM doesn't use it (no-op stub).
   - What's unclear: At `polygonAltitude(0.002)`, the polygon geometry is nearly flush with the globe surface. Raycasting may have precision issues.
   - Recommendation: Test with `polygonAltitude(0.005)` if click detection is flaky. WM uses 0.002-0.006 range.

2. **Exact three.js version to pin**
   - What we know: globe.gl 2.45.0 requires `three >=0.154 <1`. WM uses `@types/three ^0.183.1`.
   - What's unclear: Whether the latest three.js (r174+) has breaking changes with globe.gl 2.45.0.
   - Recommendation: Pin to the same three.js version WM uses. Check `node_modules/three/package.json` in WM's lockfile, or install and test.

3. **Bundle size impact**
   - What we know: three.js is ~600KB minified. Globe.gl adds ~50KB. Total new JS: ~650KB.
   - What's unclear: Exact tree-shaking behavior with Vite/Rollup for three.js.
   - Recommendation: The `manualChunks: { globe: ['globe.gl', 'three'] }` config ensures this only loads on `/globe` route. Acceptable for desktop-only target.

## Sources

### Primary (HIGH confidence)
- **Geopol DeckGLMap.ts** (886 lines) -- read in full; all 13 public methods documented
- **Geopol globe-screen.ts** (340 lines) -- read in full; mount/unmount/event wiring/data flow mapped
- **WM GlobeMap.ts** (2,188 lines) -- read lines 1-1000 + targeted sections (flush, VIEW_POVS, atmosphere, destroy)
- **WM MapContainer.ts** (921 lines) -- read in full; delegation pattern, data caching, mode switching
- **Geopol LayerPillBar.ts** (89 lines), **GlobeHud.ts** (75 lines), **NavBar.ts** (78 lines) -- read in full
- **Geopol country-geometry.ts** (328 lines) -- read in full; GeoJSON loading, normalizeCode, centroid lookup

### Secondary (MEDIUM confidence)
- [globe.gl GitHub README](https://github.com/vasturiano/globe.gl) -- API documentation (11 data channels, interaction callbacks, camera control)
- [three-globe npm](https://www.npmjs.com/package/three-globe) -- three.js peer dependency constraint `>=0.154 <1`
- [globe.gl npm](https://www.npmjs.com/package/globe.gl) -- Latest version 2.45.0

### Tertiary (LOW confidence)
- WebSearch for three.js version compatibility -- needs validation by installing and testing

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- globe.gl 2.45.0 is proven in WM, exact same pattern
- Architecture: HIGH -- MapContainer delegation is a direct adaptation of WM's 921-line implementation, mapped to Geopol's simpler 7-method data API
- Pitfalls: HIGH -- identified from reading actual WM code (winding order cache, debounced flush, no-op country click)
- Data layer mapping: MEDIUM -- globe.gl's polygon/arc/HTML channels map well to 4 of 5 layers; heatmap (H3 hex tessellation) degrades to point markers on 3D

**Research date:** 2026-03-09
**Valid until:** 2026-04-09 (globe.gl 2.45.0 is stable; no breaking changes expected)
