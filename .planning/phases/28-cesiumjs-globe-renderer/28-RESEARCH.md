# Phase 28: CesiumJS Globe Renderer - Research

**Researched:** 2026-03-12
**Domain:** CesiumJS WebGL globe rendering, Vite integration, GeoJSON choropleth, parabolic arc primitives, scene mode morphing
**Confidence:** HIGH

## Summary

CesiumJS v1.139.1 is the current release on npm (MIT licensed). The library provides a single WebGL context that handles 3D globe, 2D flat map, and Columbus View modes with animated morph transitions. The Viewer constructor accepts a `baseLayer` option (replacing the deprecated `imageryProvider` since v1.104) for custom tile sources, and all Ion-dependent defaults can be disabled for fully self-hosted operation.

Vite integration uses the official pattern from `CesiumGS/cesium-vite-example`: `vite-plugin-static-copy` (v3.2.0, supports Vite 5/6/7) copies Workers/, Assets/, ThirdParty/, and Widgets/ directories to the build output, and a `define` block injects `CESIUM_BASE_URL`. The community `vite-plugin-cesium` (v1.2.23) is effectively abandoned -- do not use it.

For the 5 analytic layers: GeoJsonDataSource handles ~195 country polygons (acceptable Entity count for choropleth); BillboardCollection for markers; PolylineCollection with computed parabolic vertex arrays for arcs; PointPrimitiveCollection for heatmap hex centers; and Entity polygons with height offset for scenario zones. ScreenSpaceEventHandler provides click/hover picking. Scene mode morphing uses `scene.morphTo3D()` / `morphTo2D()` / `morphToColumbusView()` with a `morphComplete` event.

**Primary recommendation:** Use the `cesium` npm package (not `@cesium/engine`) at v1.139.x with `vite-plugin-static-copy` for asset handling. Self-host with CartoDB dark-matter tiles, no Ion token.

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `cesium` | ^1.139.0 | 3D globe rendering, scene modes, entity API, primitives | Official CesiumJS package, MIT license, single WebGL context for all modes |
| `vite-plugin-static-copy` | ^3.2.0 | Copies CesiumJS Workers/Assets/ThirdParty/Widgets to build output | Official Cesium Vite example uses this; supports Vite 5/6/7 via peerDependency |
| `h3-js` | ^4.4.0 (retained) | H3 hex center coordinate conversion for heatmap layer | Already in use, still needed for `cellToLatLng()` |

### Packages Removed
| Library | Version | Reason for Removal |
|---------|---------|-------------------|
| `globe.gl` | ^2.45.0 | Replaced by CesiumJS 3D globe |
| `three` | ^0.183.2 | Was globe.gl dependency; CesiumJS has own renderer |
| `@types/three` | ^0.183.1 | No longer needed |
| `@deck.gl/mapbox` | ^9.2.6 | Replaced by CesiumJS |
| `@deck.gl/core` | ^9.2.6 | Replaced by CesiumJS |
| `@deck.gl/layers` | ^9.2.6 | Replaced by CesiumJS |
| `@deck.gl/geo-layers` | ^9.2.11 | Replaced by CesiumJS |
| `@deck.gl/aggregation-layers` | ^9.2.6 | Replaced by CesiumJS |
| `deck.gl` | ^9.2.6 | Replaced by CesiumJS |
| `maplibre-gl` | ^5.16.0 | Replaced by CesiumJS |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `cesium` (full) | `@cesium/engine` + `@cesium/widgets` | Scoped packages offer finer control but Viewer is in widgets; full package is simpler and what the Vite example uses |
| `vite-plugin-static-copy` | `vite-plugin-cesium` (v1.2.23) | Abandoned, not maintained since ~2025, no Vite 6 support. Do not use. |
| `vite-plugin-static-copy` | `vite-plugin-cesium-build` | Community alternative, less established. Stick with official pattern. |

**Installation:**
```bash
cd frontend
npm install cesium@^1.139.0
npm install -D vite-plugin-static-copy@^3.2.0
npm uninstall globe.gl three @types/three @deck.gl/mapbox @deck.gl/core @deck.gl/layers @deck.gl/geo-layers @deck.gl/aggregation-layers deck.gl maplibre-gl
```

## Architecture Patterns

### Recommended File Structure
```
frontend/src/
├── components/
│   ├── CesiumMap.ts          # NEW: single CesiumJS renderer (~800-1000 lines)
│   ├── GlobeMap.ts           # DELETED
│   ├── DeckGLMap.ts          # DELETED
│   ├── MapContainer.ts       # DELETED
│   ├── NavBar.ts             # MODIFIED: segmented control replaces toggle
│   ├── LayerPillBar.ts       # MODIFIED: import repointed
│   ├── GlobeHud.ts           # UNCHANGED
│   ├── GlobeDrillDown.ts     # UNCHANGED
│   └── ...
├── screens/
│   └── globe-screen.ts       # MODIFIED: CesiumMap replaces MapContainer
└── ...
```

### Pattern 1: Self-Hosted CesiumJS Viewer (No Ion Token)

**What:** Initialize CesiumJS Viewer without Cesium Ion cloud services.
**When to use:** Always -- this project uses CartoDB tiles, no terrain, no 3D tiles.

```typescript
// Source: https://gist.github.com/banesullivan/e3cc15a3e2e865d5ab8bae6719733752
import {
  Ion, Viewer, ImageryLayer, UrlTemplateImageryProvider,
  SceneMode, EllipsoidTerrainProvider,
} from 'cesium';

// Suppress Ion token warning
Ion.defaultAccessToken = undefined;

// Hidden credit container
const creditDiv = document.createElement('div');
creditDiv.style.display = 'none';
container.appendChild(creditDiv);

const viewer = new Viewer(container, {
  // Custom CartoDB dark-matter tiles
  baseLayer: new ImageryLayer(
    new UrlTemplateImageryProvider({
      url: 'https://basemaps.cartocdn.com/dark_all/{z}/{x}/{y}@2x.png',
      maximumLevel: 18,
      credit: 'Map tiles by CARTO. Data by OpenStreetMap',
    }),
  ),

  // No terrain (flat ellipsoid)
  terrainProvider: new EllipsoidTerrainProvider(),

  // Disable all Ion-dependent widgets
  baseLayerPicker: false,
  geocoder: false,
  homeButton: false,
  sceneModePicker: false,
  navigationHelpButton: false,
  animation: false,
  timeline: false,
  infoBox: false,
  selectionIndicator: false,
  creditContainer: creditDiv,

  // Start in 3D mode
  sceneMode: SceneMode.SCENE3D,

  // Performance: only render when scene changes
  requestRenderMode: true,
  maximumRenderTimeChange: Infinity,
});
```

**CRITICAL: `baseLayer` not `imageryProvider`.** The `imageryProvider` option was deprecated in CesiumJS 1.104 and removed in 1.107. Use `baseLayer: new ImageryLayer(provider)` instead.

**CRITICAL: `requestRenderMode: true`.** CesiumJS renders at 60fps by default even when nothing changes. With `requestRenderMode: true`, it only renders when the scene is dirty (camera move, data change, morph animation). Call `viewer.scene.requestRender()` after data updates to trigger a frame.

### Pattern 2: Parabolic Arc Generation via Interpolated Positions

**What:** Generate curved polylines that rise above the globe surface between two geographic points.
**When to use:** Layer 3 (Knowledge Graph Arcs) -- flight-tracker-style visualization.

```typescript
// Source: Cesium community pattern + Cartesian3.fromDegreesArrayHeights API
import { Cartesian3 } from 'cesium';

function generateArcPositions(
  startLng: number, startLat: number,
  endLng: number, endLat: number,
  numSegments: number = 40,
): Cartesian3[] {
  // Surface distance approximation for peak height scaling
  const dLat = endLat - startLat;
  const dLng = endLng - startLng;
  const surfaceDist = Math.sqrt(dLat * dLat + dLng * dLng);

  // Peak height proportional to distance: min 100km, max 2000km
  const peakHeight = Math.max(100_000, Math.min(2_000_000, surfaceDist * 50_000));

  const coords: number[] = [];
  for (let i = 0; i <= numSegments; i++) {
    const t = i / numSegments;
    const lng = startLng + t * (endLng - startLng);
    const lat = startLat + t * (endLat - startLat);
    // Parabolic height: h(t) = peakHeight * 4 * t * (1 - t)
    const height = peakHeight * 4 * t * (1 - t);
    coords.push(lng, lat, height);
  }

  return Cartesian3.fromDegreesArrayHeights(coords);
}
```

**Key insight:** `Cartesian3.fromDegreesArrayHeights` takes a flat array of `[lng, lat, height, lng, lat, height, ...]` and returns `Cartesian3[]` suitable for polyline positions. The parabolic formula `4t(1-t)` peaks at 0.5 with value 1.0. This is simpler and more performant than using CesiumJS SampledPositionProperty + LagrangePolynomialApproximation (which requires JulianDate time-based interpolation).

### Pattern 3: Scene Mode Morphing with Event

**What:** Animate transitions between 3D, Columbus View, and 2D.
**When to use:** Scene mode toggle (NavBar segmented control).

```typescript
// Source: Cesium Scene API docs
import { SceneMode } from 'cesium';

// Morph with 2-second animation (default)
viewer.scene.morphTo3D(2.0);
viewer.scene.morphToColumbusView(2.0);
viewer.scene.morphTo2D(2.0);

// Listen for morph completion
viewer.scene.morphComplete.addEventListener(() => {
  const mode = viewer.scene.mode;
  // mode is SceneMode.SCENE3D, SceneMode.COLUMBUS_VIEW, or SceneMode.SCENE2D
  window.dispatchEvent(new CustomEvent('globe-mode-changed', {
    detail: { mode: sceneModeToString(mode) },
  }));
});

// IMPORTANT: Camera flyTo during morph causes race conditions.
// Always wait for morphComplete before flying to a position.
```

### Pattern 4: ScreenSpaceEventHandler for Pick/Hover

**What:** Click and hover detection on entities and primitives.
**When to use:** Country polygon click, marker click, tooltip on hover.

```typescript
// Source: Cesium ScreenSpaceEventHandler API docs
import { ScreenSpaceEventHandler, ScreenSpaceEventType, defined } from 'cesium';

const handler = new ScreenSpaceEventHandler(viewer.canvas);

// Click handler
handler.setInputAction((event: { position: Cartesian2 }) => {
  const picked = viewer.scene.pick(event.position);
  if (defined(picked) && picked.id instanceof Entity) {
    const entity = picked.id;
    // entity.properties contains custom metadata
    const iso = entity.properties?.iso?.getValue();
  }
}, ScreenSpaceEventType.LEFT_CLICK);

// Hover handler
handler.setInputAction((event: { endPosition: Cartesian2 }) => {
  const picked = viewer.scene.pick(event.endPosition);
  if (defined(picked)) {
    // For Entity: picked.id is the Entity
    // For Primitive (Billboard/Point): picked.primitive is the collection,
    //   picked.id is the value passed to billboard.id / point.id
    showTooltip(event.endPosition, picked);
  } else {
    hideTooltip();
  }
}, ScreenSpaceEventType.MOUSE_MOVE);

// Cleanup
handler.destroy();
```

**CRITICAL: Pick result structure differs between Entity API and Primitive API.**
- Entity pick: `picked.id` is the Entity instance. Access `entity.properties` for custom data.
- Primitive pick (BillboardCollection, PointPrimitiveCollection, PolylineCollection): `picked.primitive` is the collection, `picked.id` is the value set on the individual billboard/point/polyline's `id` property.

### Pattern 5: GeoJsonDataSource for Choropleth

**What:** Load country GeoJSON and style polygon entities by risk score.
**When to use:** Layer 1 (Forecast Risk Choropleth).

```typescript
// Source: Cesium GeoJsonDataSource API docs
import {
  GeoJsonDataSource, Color, ColorMaterialProperty, ConstantProperty,
} from 'cesium';

// Load GeoJSON with default styling
const ds = await GeoJsonDataSource.load(geoJsonData, {
  stroke: Color.fromCssColorString('rgba(80,85,95,0.6)'),
  strokeWidth: 1,
  fill: Color.fromCssColorString('rgba(40,44,52,0.47)'),
  clampToGround: true,
});

viewer.dataSources.add(ds);

// Post-load: style each entity by risk score
for (const entity of ds.entities.values) {
  const iso = normalizeCode(entity.properties?.getValue(viewer.clock.currentTime));
  if (!iso) continue;

  const score = riskScores.get(iso);
  if (score !== undefined) {
    entity.polygon!.material = new ColorMaterialProperty(
      riskColor(score / 100),
    );
  }

  // Tag entity with ISO code for picking
  entity.properties!.addProperty('iso', iso);
  entity.properties!.addProperty('layerId', 'ForecastRiskChoropleth');
}
```

**Post-load iteration for styling is the standard pattern.** GeoJsonDataSource.load() creates Entity objects from features, then you iterate and set materials. For ~195 entities this is negligible.

**Performance note:** GeoJsonDataSource creates Entity objects, not Primitives. For ~195 countries this is fine. Do NOT use GeoJsonDataSource for thousands of features -- use Primitive API instead. This is why Layer 4 (heatmap, potentially thousands of points) uses PointPrimitiveCollection.

### Pattern 6: requestRender After Data Updates

**What:** Trigger a render frame after pushing data changes.
**When to use:** After every data update method (updateRiskScores, updateForecasts, etc.).

```typescript
// With requestRenderMode: true, CesiumJS won't render unless told to
function updateRiskScores(summaries: CountryRiskSummary[]): void {
  // ... update entity materials ...
  viewer.scene.requestRender();
}
```

**Without `requestRender()`, data changes will not appear on screen** until the user interacts with the globe (which triggers a render). This is the #1 gotcha with `requestRenderMode: true`.

### Anti-Patterns to Avoid

- **Do NOT use `imageryProvider` constructor option.** Deprecated in CesiumJS 1.104, removed in 1.107. Use `baseLayer: new ImageryLayer(provider)`.
- **Do NOT use Entity API for high-volume data (>500 items).** Entity API is ~O(n) per frame. Use PointPrimitiveCollection, BillboardCollection, or PolylineCollection for layers with many features.
- **Do NOT call `scene.pick()` on every `MOUSE_MOVE` without throttling.** Pick is a GPU readback operation. Throttle to 16ms (60fps) or 33ms (30fps) to avoid janking the render loop.
- **Do NOT flyTo during an active morph.** Wait for `scene.morphComplete` event, then fly. Otherwise camera position is undefined.
- **Do NOT use `vite-plugin-cesium`.** Abandoned, no Vite 6 support.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Atmosphere glow | Custom Three.js BackSide SphereGeometry shader | CesiumJS `scene.skyAtmosphere` (built-in) | Realistic atmospheric scattering, zero custom code |
| Scene mode transitions | CSS display swap + dual WebGL contexts | `scene.morphTo3D()` / `morphTo2D()` / `morphToColumbusView()` | Animated morph, single context, camera state preserved |
| Logarithmic depth buffer | N/A (linear depth in Three.js caused z-fighting) | CesiumJS `scene.logarithmicDepthBuffer` (enabled by default) | Eliminates polygon z-fighting at any altitude offset |
| Tile-based basemap | Static texture image (`earth-topo-bathy.jpg`) | `UrlTemplateImageryProvider` with CartoDB tiles | Progressive loading, sharp at all zoom levels |
| World coordinates | Manual lat/lng to 3D conversion | `Cartesian3.fromDegrees(lng, lat, height)` | Handles ellipsoid, geodesic, all projections |
| Color from CSS strings | Custom RGBA parsing | `Color.fromCssColorString('rgba(r,g,b,a)')` | Handles all CSS color formats |
| Parabolic arcs | SampledPositionProperty + LagrangePolynomial | `Cartesian3.fromDegreesArrayHeights()` with manual `4t(1-t)` formula | Simpler, no time-based interpolation needed |
| Static asset copying for build | Manual copy scripts | `vite-plugin-static-copy` with Cesium's official targets | Battle-tested, handles both dev and build |

## Common Pitfalls

### Pitfall 1: Ion Token Warning in Console
**What goes wrong:** Console shows "Please assign Cesium.Ion.defaultAccessToken" even without using Ion services.
**Why it happens:** CesiumJS default baseLayer and terrain providers are Ion-based.
**How to avoid:** Set `Ion.defaultAccessToken = undefined` before creating Viewer. Set `baseLayer` to a non-Ion ImageryLayer. Set `terrainProvider` to `EllipsoidTerrainProvider`.
**Warning signs:** Console warning containing "Ion.defaultAccessToken".

### Pitfall 2: requestRenderMode Causes Blank Screen After Data Update
**What goes wrong:** Data is pushed but globe doesn't re-render, appears frozen.
**Why it happens:** `requestRenderMode: true` means CesiumJS only renders when the scene is dirty. Programmatic data changes don't automatically mark the scene as dirty.
**How to avoid:** Call `viewer.scene.requestRender()` after every data update. For continuous animations, set `shouldAnimate: true` or disable `requestRenderMode`.
**Warning signs:** Data updates that only appear after mouse interaction.

### Pitfall 3: Camera FlyTo During Active Morph
**What goes wrong:** Camera position becomes undefined, globe jumps to wrong location.
**Why it happens:** Morph animation interpolates camera position. FlyTo fights with the morph interpolation.
**How to avoid:** Listen for `scene.morphComplete` event before calling `camera.flyTo()`. Queue flyTo requests during morph.
**Warning signs:** Camera snapping to random positions after mode switch.

### Pitfall 4: Missing CESIUM_BASE_URL
**What goes wrong:** CesiumJS throws errors about missing Workers, blank globe, web workers fail to load.
**Why it happens:** CesiumJS uses web workers for terrain/imagery processing. It needs to know where Worker JS files are served from.
**How to avoid:** Add `define: { CESIUM_BASE_URL: JSON.stringify('/cesiumStatic') }` to vite.config.ts. Use `vite-plugin-static-copy` to copy Worker/Asset directories.
**Warning signs:** 404 errors for `.js` files under `/cesiumStatic/Workers/`, blank globe.

### Pitfall 5: Entity Properties Access Requires `.getValue()` / JulianDate
**What goes wrong:** `entity.properties.iso` returns a Property object, not a string.
**Why it happens:** CesiumJS Entity properties are time-dynamic by default. Even static values are wrapped in Property objects.
**How to avoid:** Use `entity.properties?.iso?.getValue(viewer.clock.currentTime)` or access raw GeoJSON properties via `entity.properties?.getValue(Cesium.JulianDate.now())` which returns a plain object.
**Warning signs:** Getting `[object Object]` instead of expected string values.

### Pitfall 6: PolylineCollection Width is in Pixels, Not Meters
**What goes wrong:** Arc widths don't scale with zoom level.
**Why it happens:** PolylineCollection width is specified in screen pixels, not world units.
**How to avoid:** This is actually the desired behavior for this use case (arcs should be visible at all zoom levels). Just be aware that width=2 means 2 pixels regardless of altitude.
**Warning signs:** Arcs appearing too thin at high altitude or too thick when zoomed in.

### Pitfall 7: Pick Returns Different Structures for Entity vs Primitive
**What goes wrong:** Click handler works for choropleth (Entity) but crashes for markers (BillboardCollection).
**Why it happens:** Entity pick returns `{ id: Entity }`. Primitive pick returns `{ primitive: Collection, id: userDefinedId }`.
**How to avoid:** Check both `picked.id instanceof Entity` and `picked.id` as a plain object. Use a discriminated union or layerId tag.
**Warning signs:** TypeError when accessing `.properties` on a non-Entity pick result.

### Pitfall 8: CesiumJS Widget CSS Overrides Application Styles
**What goes wrong:** CesiumJS imports CSS that conflicts with application styles (cursor, font, layout).
**Why it happens:** `cesium/Build/Cesium/Widgets/widgets.css` contains global styles for Cesium's built-in widgets.
**How to avoid:** Since all widgets are disabled (timeline, animation, etc.), you likely don't need widgets.css at all. If imported, scope it or override conflicting rules. Test whether the Viewer initializes correctly without the CSS import.
**Warning signs:** Cursor changes, unexpected font changes, layout shifts on the globe screen.

## Code Examples

### Viewer Initialization (Self-Hosted, No Ion)
```typescript
// Source: Cesium Vite example + Bane Sullivan gist
import {
  Ion, Viewer, ImageryLayer, UrlTemplateImageryProvider,
  SceneMode, EllipsoidTerrainProvider,
} from 'cesium';

Ion.defaultAccessToken = undefined;

const creditContainer = document.createElement('div');
creditContainer.style.display = 'none';
container.appendChild(creditContainer);

const viewer = new Viewer(container, {
  baseLayer: new ImageryLayer(
    new UrlTemplateImageryProvider({
      url: 'https://basemaps.cartocdn.com/dark_all/{z}/{x}/{y}@2x.png',
      maximumLevel: 18,
      credit: 'CARTO, OpenStreetMap contributors',
    }),
  ),
  terrainProvider: new EllipsoidTerrainProvider(),
  baseLayerPicker: false,
  geocoder: false,
  homeButton: false,
  sceneModePicker: false,
  navigationHelpButton: false,
  animation: false,
  timeline: false,
  infoBox: false,
  selectionIndicator: false,
  creditContainer,
  sceneMode: SceneMode.SCENE3D,
  requestRenderMode: true,
  maximumRenderTimeChange: Infinity,
});

// Atmosphere is enabled by default via scene.skyAtmosphere
// Logarithmic depth buffer is enabled by default
```

### Vite Config (Official Cesium Pattern)
```typescript
// Source: https://github.com/CesiumGS/cesium-vite-example
import { defineConfig } from 'vite';
import { resolve } from 'path';
import { viteStaticCopy } from 'vite-plugin-static-copy';

const cesiumSource = 'node_modules/cesium/Build/Cesium';
const cesiumBaseUrl = 'cesiumStatic';

export default defineConfig({
  define: {
    CESIUM_BASE_URL: JSON.stringify(`/${cesiumBaseUrl}`),
  },
  resolve: {
    alias: {
      '@': resolve(__dirname, 'src'),
    },
  },
  plugins: [
    viteStaticCopy({
      targets: [
        { src: `${cesiumSource}/ThirdParty`, dest: cesiumBaseUrl },
        { src: `${cesiumSource}/Workers`, dest: cesiumBaseUrl },
        { src: `${cesiumSource}/Assets`, dest: cesiumBaseUrl },
        { src: `${cesiumSource}/Widgets`, dest: cesiumBaseUrl },
      ],
    }),
  ],
  build: {
    target: 'es2020',
    sourcemap: true,
    rollupOptions: {
      output: {
        manualChunks: {
          cesium: ['cesium'],
          d3: ['d3'],
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

### BillboardCollection for Markers (Layer 2)
```typescript
// Source: Cesium BillboardCollection API docs
import { BillboardCollection, Cartesian3, Color, HorizontalOrigin, VerticalOrigin } from 'cesium';

const billboards = new BillboardCollection();
viewer.scene.primitives.add(billboards);

// Canvas-rendered marker sprite
function createMarkerCanvas(probability: number): HTMLCanvasElement {
  const canvas = document.createElement('canvas');
  canvas.width = 32;
  canvas.height = 32;
  const ctx = canvas.getContext('2d')!;
  const intensity = Math.round(100 + probability * 155);
  ctx.fillStyle = `rgb(${intensity},40,40)`;
  ctx.beginPath();
  ctx.arc(16, 16, 12, 0, Math.PI * 2);
  ctx.fill();
  ctx.strokeStyle = 'rgba(255,255,255,0.3)';
  ctx.lineWidth = 1;
  ctx.stroke();
  return canvas;
}

// Add markers
for (const marker of markers) {
  billboards.add({
    position: Cartesian3.fromDegrees(marker.lng, marker.lat),
    image: createMarkerCanvas(marker.probability),
    id: { layerId: 'ActiveForecastMarkers', iso: marker.iso, data: marker },
    scale: 1.0,
    horizontalOrigin: HorizontalOrigin.CENTER,
    verticalOrigin: VerticalOrigin.CENTER,
  });
}
```

### PointPrimitiveCollection for Heatmap (Layer 4)
```typescript
// Source: Cesium PointPrimitiveCollection API docs
import { PointPrimitiveCollection, Cartesian3, Color } from 'cesium';

const points = new PointPrimitiveCollection();
viewer.scene.primitives.add(points);

for (const hex of hexBinData) {
  const [lat, lng] = h3Module.cellToLatLng(hex.h3_index);
  if (!Number.isFinite(lat) || !Number.isFinite(lng)) continue;

  const t = Math.min(1, hex.weight / maxWeight);
  points.add({
    position: Cartesian3.fromDegrees(lng, lat, 0),
    color: Color.fromCssColorString(
      `rgba(255,${Math.round(255 * (1 - t))},0,${(0.4 + 0.5 * t).toFixed(2)})`
    ),
    pixelSize: 6 + t * 6,
    id: { layerId: 'GDELTEventHeatmap', data: hex },
  });
}
```

### Parabolic Arc Polylines (Layer 3)
```typescript
// Source: Cesium PolylineCollection + community arc pattern
import {
  PolylineCollection, Cartesian3, Color, Material,
} from 'cesium';

const polylines = new PolylineCollection();
viewer.scene.primitives.add(polylines);

for (const arc of bilateralArcs) {
  const positions = generateArcPositions(
    arc.source[0], arc.source[1],  // [lng, lat]
    arc.target[0], arc.target[1],
    40,
  );

  const color = arc.avgGoldstein < 0
    ? Color.fromCssColorString('rgba(255,80,80,0.7)')   // conflictual
    : Color.fromCssColorString('rgba(80,180,255,0.7)');  // cooperative

  polylines.add({
    positions,
    width: Math.max(1, Math.min(5, arc.eventCount / 20)),
    material: Material.fromType('Color', { color }),
    id: { layerId: 'KnowledgeGraphArcs', data: arc },
  });
}
```

### Scene Mode Morph + Camera FlyTo (Safe Pattern)
```typescript
// Source: Cesium Scene API + community discussion on morph+flyTo race
function setSceneMode(mode: '3d' | 'columbus' | '2d'): void {
  const scene = viewer.scene;

  switch (mode) {
    case '3d': scene.morphTo3D(2.0); break;
    case 'columbus': scene.morphToColumbusView(2.0); break;
    case '2d': scene.morphTo2D(2.0); break;
  }

  localStorage.setItem('geopol-globe-mode', mode);
}

// Camera flyTo queued after morph
let pendingFlyTo: (() => void) | null = null;

viewer.scene.morphComplete.addEventListener(() => {
  viewer.scene.requestRender();
  if (pendingFlyTo) {
    pendingFlyTo();
    pendingFlyTo = null;
  }
});

function flyToCountry(iso: string): void {
  const centroid = countryGeometry.getCentroid(iso);
  if (!centroid) return;

  const fly = () => viewer.camera.flyTo({
    destination: Cartesian3.fromDegrees(centroid[0], centroid[1], 3_000_000),
    duration: 1.2,
  });

  // If morphing, queue the flyTo
  if (viewer.scene.mode !== viewer.scene.mode) { // simplified check
    pendingFlyTo = fly;
  } else {
    fly();
  }
}
```

### VIEW_POVS Altitude Conversion
```typescript
// Globe radii -> meters above ellipsoid
// EARTH_RADIUS ~= 6_371_000m
// globe.gl altitude 1.8 -> 1.8 * 6_371_000 = 11_467_800m camera height
const EARTH_RADIUS = 6_371_000;

const VIEW_POVS: Record<string, { lat: number; lng: number; height: number }> = {
  global:  { lat: 20,  lng:   0,  height: 1.8 * EARTH_RADIUS },
  america: { lat: 20,  lng: -90,  height: 1.5 * EARTH_RADIUS },
  mena:    { lat: 25,  lng:  40,  height: 1.2 * EARTH_RADIUS },
  eu:      { lat: 50,  lng:  10,  height: 1.2 * EARTH_RADIUS },
  asia:    { lat: 35,  lng: 105,  height: 1.5 * EARTH_RADIUS },
  latam:   { lat: -15, lng: -60,  height: 1.5 * EARTH_RADIUS },
  africa:  { lat:  5,  lng:  20,  height: 1.5 * EARTH_RADIUS },
  oceania: { lat: -25, lng: 140,  height: 1.5 * EARTH_RADIUS },
};

function flyToRegion(region: string): void {
  const pov = VIEW_POVS[region];
  if (!pov) return;
  viewer.camera.flyTo({
    destination: Cartesian3.fromDegrees(pov.lng, pov.lat, pov.height),
    duration: 1.2,
  });
}
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `imageryProvider` Viewer option | `baseLayer: new ImageryLayer(provider)` | CesiumJS 1.104 (mid-2023) | Must use `baseLayer`, `imageryProvider` throws in v1.107+ |
| `createWorldTerrain()` | `Terrain.fromWorldTerrain()` | CesiumJS 1.104 | Not relevant (we use EllipsoidTerrainProvider) |
| Synchronous imagery provider constructors | `fromAssetId()` / `fromUrl()` async factories | CesiumJS 1.104 | `UrlTemplateImageryProvider` constructor still works (sync) |
| `vite-plugin-cesium` (community) | `vite-plugin-static-copy` (official pattern) | ~2024 | `vite-plugin-cesium` abandoned; official example uses static-copy |
| Full `cesium` package only | `@cesium/engine` + `@cesium/widgets` scoped packages | CesiumJS 1.113 | Scoped packages available but full package still recommended for Viewer usage |

**Deprecated/outdated:**
- `Viewer({ imageryProvider: ... })` -- use `baseLayer` instead
- `vite-plugin-cesium` npm package -- abandoned, use `vite-plugin-static-copy`
- `ImageryProvider.readyPromise` -- removed, providers are ready on construction

## Open Questions

1. **CesiumJS CSS import necessity**
   - What we know: All built-in widgets (timeline, animation, info box, etc.) are disabled. The CSS is mainly for widget styling.
   - What's unclear: Whether the Viewer container div itself requires any styles from `widgets.css` to render correctly (e.g., cursor, overflow, positioning).
   - Recommendation: Try without importing the CSS first. If the viewer canvas doesn't fill its container, import it and scope-override conflicting rules. This is Claude's discretion per CONTEXT.md.

2. **requestRenderMode interaction with morph animations**
   - What we know: Morph animations are internal to CesiumJS and should drive their own render loop.
   - What's unclear: Whether `requestRenderMode: true` pauses morph animations (they would need to call `requestRender()` internally).
   - Recommendation: Test with `requestRenderMode: true`. If morph animations don't play, temporarily set `viewer.scene.requestRenderMode = false` during morph and re-enable on `morphComplete`. Morph only lasts 2 seconds.

3. **GeoJsonDataSource entity update performance for choropleth re-coloring**
   - What we know: Iterating ~195 entities and setting `polygon.material` works. Entity API tracks property changes and marks scene dirty.
   - What's unclear: Whether re-coloring all 195 entities on every 120s risk score refresh causes a visible frame drop.
   - Recommendation: Profile in practice. If slow, consider destroying and recreating the DataSource (clean approach) or caching entity references by ISO in a Map for O(1) lookup.

4. **PolylineCollection material type for colored polylines**
   - What we know: `Material.fromType('Color', { color })` creates a solid-color material. PolylineCollection `add()` accepts a `material` property.
   - What's unclear: Whether per-polyline material creates separate draw calls (performance concern with 50+ arcs) or batches correctly.
   - Recommendation: Use a single PolylineCollection for all arcs. If draw call overhead is visible, consider using only 2-3 colors (cooperative, conflictual, neutral) and batching by color into separate collections.

## Sources

### Primary (HIGH confidence)
- [Cesium Vite Example](https://github.com/CesiumGS/cesium-vite-example) -- Official Vite integration pattern, vite.config.js reference
- [Cesium Viewer API](https://cesium.com/learn/cesiumjs/ref-doc/Viewer.html) -- Constructor options, baseLayer, sceneMode, requestRenderMode
- [Cesium Scene API](https://cesium.com/learn/cesiumjs/ref-doc/Scene.html) -- morphTo3D/2D/ColumbusView, morphComplete, pick(), skyAtmosphere, logarithmicDepthBuffer
- [Cesium Camera API](https://cesium.com/learn/cesiumjs/ref-doc/Camera.html) -- flyTo(), setView(), Cartesian3.fromDegrees
- [Cesium UrlTemplateImageryProvider](https://cesium.com/learn/cesiumjs/ref-doc/UrlTemplateImageryProvider.html) -- URL template parameters, maximumLevel, credit
- [Cesium ScreenSpaceEventHandler](https://cesium.com/learn/cesiumjs/ref-doc/ScreenSpaceEventHandler.html) -- setInputAction, event types, movement objects
- [Cesium BillboardCollection](https://cesium.com/learn/cesiumjs/ref-doc/BillboardCollection.html) -- add(), Billboard properties, performance notes
- [Cesium PointPrimitiveCollection](https://cesium.com/learn/cesiumjs/ref-doc/PointPrimitiveCollection.html) -- add(), PointPrimitive properties, id field
- [Cesium PolylineCollection](https://cesium.com/learn/cesiumjs/ref-doc/PolylineCollection.html) -- add(), positions, width, material
- [Cesium GeoJsonDataSource](https://cesium.com/learn/cesiumjs/ref-doc/GeoJsonDataSource.html) -- load() options, entity styling, post-load iteration
- [Cesium Entity API Tutorial](https://cesium.com/learn/cesiumjs-learn/cesiumjs-creating-entities/) -- Entity properties, polygon materials, pick patterns
- [vite-plugin-static-copy npm](https://www.npmjs.com/package/vite-plugin-static-copy) -- v3.2.0, Vite 5/6/7 peerDependency
- [cesium npm](https://www.npmjs.com/package/cesium) -- v1.139.1, latest release

### Secondary (MEDIUM confidence)
- [Bane Sullivan: CesiumJS without Ion Token](https://gist.github.com/banesullivan/e3cc15a3e2e865d5ab8bae6719733752) -- Self-hosted Viewer pattern, CartoDB tile providers
- [Cesium Community: Lift polyline above surface](https://community.cesium.com/t/how-to-lift-the-polyline-above-earth-surface/3382) -- EllipsoidGeodesic + interpolation for arcs
- [Cesium Community: Arc curve with coordinates](https://community.cesium.com/t/make-arc-curve-with-coordinates/22599) -- Spline interpolation alternatives
- [Cesium Community: Scene mode switching](https://community.cesium.com/t/how-to-change-from-3d-to-2d-columbus-view/102) -- morphTo3D/2D/ColumbusView usage
- [Cesium Community: flyTo after morph](https://community.cesium.com/t/flyto-after-morphing-scene-mode/4902) -- morphComplete + setTimeout workaround
- [Cesium Community: imageryProvider deprecated](https://community.cesium.com/t/cesium-viewer-constructoroptions-imageryprovider-deprecated/23640) -- baseLayer migration
- [Cesium Community: hide credits](https://community.cesium.com/t/how-to-remove-the-creditsdisplay/6961) -- creditContainer approach

### Tertiary (LOW confidence)
- Parabolic arc `4t(1-t)` formula height scaling factor (50,000 * surface distance degrees) -- derived from community patterns, needs visual tuning in practice

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- Official Cesium npm package, official Vite integration pattern, version numbers verified on npm
- Architecture: HIGH -- All APIs verified against official CesiumJS documentation
- Pitfalls: HIGH -- Ion token, requestRenderMode, morph+flyTo race all documented in official docs and community forums
- Parabolic arc formula: MEDIUM -- Community pattern, specific height constants need visual tuning
- Widget CSS necessity: LOW -- Untested whether Viewer works without widgets.css import

**Research date:** 2026-03-12
**Valid until:** 2026-04-12 (CesiumJS releases monthly, but core API is stable)
