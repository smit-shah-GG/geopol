/**
 * CesiumMap -- Single CesiumJS globe renderer replacing GlobeMap.ts, DeckGLMap.ts,
 * and MapContainer.ts with a unified 3D/2D/Columbus View renderer.
 *
 * Layers:
 *   1. ForecastRiskChoropleth  (GeoJsonDataSource)         -- country fill by risk_score
 *   2. ActiveForecastMarkers   (BillboardCollection)       -- centroids of forecast targets
 *   3. KnowledgeGraphArcs      (PolylineCollection)        -- parabolic bilateral arcs
 *   4. GDELTEventHeatmap       (PointPrimitiveCollection)  -- H3 hex event density
 *   5. ScenarioZones           (CustomDataSource)          -- risk delta regions
 *
 * Public API matches the combined contract from MapContainer + DeckGLMap + GlobeMap,
 * minus dual-dispatch (single renderer handles all scene modes).
 *
 * Handles: click/hover events via ScreenSpaceEventHandler, scene mode morphing
 * (3D/Columbus/2D), camera flyTo with morph queueing, layer visibility toggling,
 * per-country arc filtering, and requestRenderMode for GPU efficiency.
 */

import 'cesium/Build/Cesium/Widgets/widgets.css';

import {
  Ion,
  Viewer,
  ImageryLayer,
  UrlTemplateImageryProvider,
  SceneMode,
  EllipsoidTerrainProvider,
  GeoJsonDataSource,
  BillboardCollection,
  PolylineCollection,
  PointPrimitiveCollection,
  CustomDataSource,
  ScreenSpaceEventHandler,
  ScreenSpaceEventType,
  Cartesian3,
  Color,
  ColorMaterialProperty,
  ConstantProperty,
  ArcType,
  Material,
  Entity,
  PolygonHierarchy,
  defined,
  HorizontalOrigin,
  VerticalOrigin,
} from 'cesium';
import type { Cartesian2 } from 'cesium';

import { countryGeometry, normalizeCode } from '@/services/country-geometry.ts';
import type { ForecastResponse, CountryRiskSummary } from '@/types/api.ts';

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

export const LAYER_IDS = [
  'ForecastRiskChoropleth',
  'ActiveForecastMarkers',
  'KnowledgeGraphArcs',
  'GDELTEventHeatmap',
  'ScenarioZones',
] as const;

export type LayerId = (typeof LAYER_IDS)[number];

const STORAGE_KEY = 'geopol-globe-mode';

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

// ---------------------------------------------------------------------------
// Data interfaces (exported for external consumers)
// ---------------------------------------------------------------------------

/** H3 hexagonal bin datum for the event heatmap layer. */
export interface HexBinDatum {
  h3_index: string;
  weight: number;
  event_count: number;
}

/** Bilateral arc datum with sentiment (Goldstein scale) and volume. */
export interface BilateralArcDatum {
  sourceIso: string;
  targetIso: string;
  source: [number, number]; // [lng, lat]
  target: [number, number]; // [lng, lat]
  eventCount: number;
  avgGoldstein: number;
}

/** Risk delta datum for the scenarios/risk-change layer. */
export interface RiskDeltaDatum {
  iso: string;
  delta: number; // positive = worsening, negative = improving
}

// ---------------------------------------------------------------------------
// Internal marker datum
// ---------------------------------------------------------------------------

interface MarkerDatum {
  iso: string;
  position: [number, number]; // [lng, lat]
  probability: number;
  question: string;
}

// ---------------------------------------------------------------------------
// Color utilities
// ---------------------------------------------------------------------------

/**
 * Blue-to-gray-to-red diverging color scale for risk scores [0, 1].
 * Returns a CesiumJS Color with the specified alpha.
 */
function riskColor(score: number, alpha = 0.5): Color {
  const t = Math.max(0, Math.min(1, score));
  let r: number, g: number, b: number;
  if (t <= 0.5) {
    const u = t * 2;
    r = (70 + (128 - 70) * u) / 255;
    g = (130 + (128 - 130) * u) / 255;
    b = (180 + (128 - 180) * u) / 255;
  } else {
    const u = (t - 0.5) * 2;
    r = (128 + (220 - 128) * u) / 255;
    g = (128 + (50 - 128) * u) / 255;
    b = (128 + (50 - 128) * u) / 255;
  }
  return new Color(r, g, b, alpha);
}

/**
 * Generate parabolic arc positions between two geographic points.
 * Returns an array of Cartesian3 positions for a PolylineCollection polyline.
 */
function generateArcPositions(
  startLng: number, startLat: number,
  endLng: number, endLat: number,
  numSegments = 40,
): Cartesian3[] {
  const dLat = endLat - startLat;
  const dLng = endLng - startLng;
  const surfaceDist = Math.sqrt(dLat * dLat + dLng * dLng);
  const peakHeight = Math.max(100_000, Math.min(2_000_000, surfaceDist * 50_000));

  const coords: number[] = [];
  for (let i = 0; i <= numSegments; i++) {
    const t = i / numSegments;
    const lng = startLng + t * dLng;
    const lat = startLat + t * dLat;
    const height = peakHeight * 4 * t * (1 - t);
    coords.push(lng, lat, height);
  }

  return Cartesian3.fromDegreesArrayHeights(coords);
}

/**
 * Create a canvas-rendered pin marker for BillboardCollection.
 * Color intensity scales with probability.
 */
function createMarkerCanvas(probability: number): HTMLCanvasElement {
  const canvas = document.createElement('canvas');
  canvas.width = 32;
  canvas.height = 32;
  const ctx = canvas.getContext('2d')!;
  const intensity = Math.round(100 + probability * 155);

  // Outer glow
  ctx.fillStyle = `rgba(${intensity},40,40,0.3)`;
  ctx.beginPath();
  ctx.arc(16, 16, 14, 0, Math.PI * 2);
  ctx.fill();

  // Core circle
  ctx.fillStyle = `rgb(${intensity},40,40)`;
  ctx.beginPath();
  ctx.arc(16, 16, 10, 0, Math.PI * 2);
  ctx.fill();

  // Rim
  ctx.strokeStyle = 'rgba(255,255,255,0.35)';
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.arc(16, 16, 10, 0, Math.PI * 2);
  ctx.stroke();

  return canvas;
}

/**
 * Convert GeoJSON Polygon/MultiPolygon coordinates to a CesiumJS PolygonHierarchy.
 * Handles both simple Polygon and MultiPolygon geometries (uses first polygon's
 * outer ring with holes for simple cases; for MultiPolygon, flattens all outer rings).
 */
/**
 * Flatten a ring (Position[]) into a flat number[] for Cartesian3.fromDegreesArray.
 * GeoJSON Position is [lng, lat] or [lng, lat, alt] -- we only need lng/lat pairs.
 */
function flattenRing(ring: number[][]): number[] {
  const flat: number[] = [];
  for (const pos of ring) {
    // GeoJSON Position always has [lng, lat] as first two elements
    flat.push(pos[0]!, pos[1]!);
  }
  return flat;
}

function geoJsonToHierarchy(
  geometry: GeoJSON.Geometry,
): PolygonHierarchy | null {
  if (geometry.type === 'Polygon') {
    const rings = geometry.coordinates;
    if (!rings || rings.length === 0) return null;
    const outerRing = rings[0];
    if (!outerRing || outerRing.length === 0) return null;
    const positions = Cartesian3.fromDegreesArray(flattenRing(outerRing));
    const holes: PolygonHierarchy[] = [];
    for (let i = 1; i < rings.length; i++) {
      const holeRing = rings[i];
      if (holeRing && holeRing.length > 0) {
        holes.push(new PolygonHierarchy(Cartesian3.fromDegreesArray(flattenRing(holeRing))));
      }
    }
    return new PolygonHierarchy(positions, holes);
  }
  if (geometry.type === 'MultiPolygon') {
    // For MultiPolygon, use the first polygon and its holes.
    // CesiumJS PolygonHierarchy supports one outer ring with holes.
    const polygons = geometry.coordinates;
    if (!polygons || polygons.length === 0) return null;
    const firstPoly = polygons[0];
    if (!firstPoly || firstPoly.length === 0) return null;
    const outerRing = firstPoly[0];
    if (!outerRing || outerRing.length === 0) return null;
    const positions = Cartesian3.fromDegreesArray(flattenRing(outerRing));
    const holes: PolygonHierarchy[] = [];
    for (let i = 1; i < firstPoly.length; i++) {
      const holeRing = firstPoly[i];
      if (holeRing && holeRing.length > 0) {
        holes.push(new PolygonHierarchy(Cartesian3.fromDegreesArray(flattenRing(holeRing))));
      }
    }
    return new PolygonHierarchy(positions, holes);
  }
  return null;
}

// ---------------------------------------------------------------------------
// CesiumMap
// ---------------------------------------------------------------------------

export class CesiumMap {
  private viewer!: Viewer;
  private eventHandler!: ScreenSpaceEventHandler;
  private choroplethDS: GeoJsonDataSource | null = null;
  private billboards!: BillboardCollection;
  private polylines!: PolylineCollection;
  private points!: PointPrimitiveCollection;
  private scenarioDS!: CustomDataSource;
  private layerVisibility: Record<LayerId, boolean> = {
    ForecastRiskChoropleth: true,
    ActiveForecastMarkers: true,
    KnowledgeGraphArcs: true,
    GDELTEventHeatmap: true,
    ScenarioZones: true,
  };

  private selectedCountryIso: string | null = null;
  private selectedForecast: ForecastResponse | null = null;
  private scenarioIsos = new Set<string>();
  private allArcData: BilateralArcDatum[] = [];
  private tooltipEl!: HTMLDivElement;
  private pendingFlyTo: (() => void) | null = null;
  private h3Module: typeof import('h3-js') | null = null;
  private h3Loading = false;
  private pickThrottleTimer: number | null = null;
  private viewToggleHandler!: EventListener;
  private regionChangeHandler!: EventListener;
  private riskScoreMap = new Map<string, number>(); // ISO -> risk score 0-100
  private entityIsoMap = new Map<string, Entity>(); // ISO -> choropleth Entity
  private markerData: MarkerDatum[] = [];
  private riskDeltaMap = new Map<string, number>(); // ISO -> delta
  private morphing = false;
  private destroyed = false;
  private choroplethLoaded = false;
  private prevSelectedEntity: Entity | null = null;

  constructor(container: HTMLElement) {
    this._initViewer(container);
    this._initTooltip(container);
    this._initPrimitives();
    this._initScenarioDataSource();
    this._initEventHandler();
    this._initMorphListener();
    this._initCustomEventListeners();
  }

  // =========================================================================
  // Initialization
  // =========================================================================

  private _initViewer(container: HTMLElement): void {
    // Suppress Ion token warning
    Ion.defaultAccessToken = undefined as any;

    // Hidden credit container
    const creditDiv = document.createElement('div');
    creditDiv.style.display = 'none';
    container.appendChild(creditDiv);

    // Read persisted scene mode (default 3D)
    const pref = localStorage.getItem(STORAGE_KEY) ?? '3d';
    let initialMode = SceneMode.SCENE3D;
    if (pref === '2d') initialMode = SceneMode.SCENE2D;
    else if (pref === 'columbus') initialMode = SceneMode.COLUMBUS_VIEW;

    this.viewer = new Viewer(container, {
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
      fullscreenButton: false,
      creditContainer: creditDiv,
      sceneMode: initialMode,
      requestRenderMode: true,
      maximumRenderTimeChange: Infinity,
    });

    // Ensure atmosphere is visible in 3D mode
    if (this.viewer.scene.skyAtmosphere) {
      this.viewer.scene.skyAtmosphere.show = true;
    }
  }

  private _initTooltip(container: HTMLElement): void {
    this.tooltipEl = document.createElement('div');
    this.tooltipEl.className = 'map-tooltip';
    this.tooltipEl.style.cssText =
      'position: absolute; pointer-events: none; background: rgba(20,22,28,0.92); ' +
      'color: #e2e8f0; padding: 6px 10px; border-radius: 4px; font-size: 12px; ' +
      'z-index: 10; display: none; max-width: 280px; white-space: pre-line;';
    container.appendChild(this.tooltipEl);
  }

  private _initPrimitives(): void {
    this.billboards = new BillboardCollection({ scene: this.viewer.scene });
    this.viewer.scene.primitives.add(this.billboards);

    this.polylines = new PolylineCollection();
    this.viewer.scene.primitives.add(this.polylines);

    this.points = new PointPrimitiveCollection();
    this.viewer.scene.primitives.add(this.points);
  }

  private _initScenarioDataSource(): void {
    this.scenarioDS = new CustomDataSource('ScenarioZones');
    this.viewer.dataSources.add(this.scenarioDS);
  }

  private _initEventHandler(): void {
    this.eventHandler = new ScreenSpaceEventHandler(this.viewer.canvas);

    // LEFT_CLICK: dispatch country-selected or deselect on empty space
    this.eventHandler.setInputAction((event: { position: Cartesian2 }) => {
      const picked = this.viewer.scene.pick(event.position);

      if (!defined(picked)) {
        // Clicked empty space -- clear selection
        if (this.selectedCountryIso) {
          this.setSelectedCountry(null);
          window.dispatchEvent(
            new CustomEvent('country-deselected', { bubbles: true }),
          );
        }
        return;
      }

      let iso: string | null = null;

      // Entity pick (choropleth polygon or scenario zone)
      if (picked.id instanceof Entity) {
        const entity = picked.id;
        iso = this._getEntityIso(entity);
      }
      // Primitive pick (billboard/point)
      else if (picked.id && typeof picked.id === 'object' && 'iso' in picked.id) {
        iso = (picked.id as { iso: string }).iso;
      }

      if (iso) {
        window.dispatchEvent(
          new CustomEvent('country-selected', {
            detail: { iso },
            bubbles: true,
          }),
        );
      }
    }, ScreenSpaceEventType.LEFT_CLICK);

    // MOUSE_MOVE: throttled hover tooltip
    let lastMoveTime = 0;
    this.eventHandler.setInputAction((event: { endPosition: Cartesian2 }) => {
      const now = performance.now();
      if (now - lastMoveTime < 16) return; // 16ms throttle (~60fps)
      lastMoveTime = now;

      const picked = this.viewer.scene.pick(event.endPosition);
      if (!defined(picked)) {
        this.tooltipEl.style.display = 'none';
        return;
      }

      let content = '';
      let layerId: string | null = null;

      // Entity pick
      if (picked.id instanceof Entity) {
        const entity = picked.id;
        layerId = this._getEntityLayerId(entity);
        const iso = this._getEntityIso(entity);

        if (layerId === 'ForecastRiskChoropleth' && iso) {
          const name = countryGeometry.getNameByIso(iso) ?? iso;
          const score = this.riskScoreMap.get(iso);
          const scoreStr = score !== undefined ? score.toFixed(1) + '%' : 'N/A';
          content = `<strong>${name}</strong><br/>Risk: ${scoreStr}`;
        } else if (layerId === 'ScenarioZones' && iso) {
          const name = countryGeometry.getNameByIso(iso) ?? iso;
          const delta = this.riskDeltaMap.get(iso) ?? 0;
          const direction = delta > 0 ? 'Worsening' : 'Improving';
          content = `<strong>${name}</strong><br/>${direction}: ${delta > 0 ? '+' : ''}${delta.toFixed(1)} pts`;
        }
      }
      // Primitive pick (billboard, point, polyline)
      else if (picked.id && typeof picked.id === 'object') {
        const idObj = picked.id as Record<string, any>;
        layerId = idObj.layerId ?? null;

        if (layerId === 'ActiveForecastMarkers' && idObj.data) {
          const marker = idObj.data as MarkerDatum;
          const name = countryGeometry.getNameByIso(marker.iso) ?? marker.iso;
          const pct = (marker.probability * 100).toFixed(1);
          const q = marker.question.length > 80
            ? marker.question.slice(0, 77) + '...'
            : marker.question;
          content = `<strong>${name}</strong><br/>${q}<br/>P: ${pct}%`;
        } else if (layerId === 'GDELTEventHeatmap' && idObj.data) {
          const hex = idObj.data as HexBinDatum;
          content = `Events: ${hex.event_count}<br/>Weight: ${hex.weight.toFixed(1)}`;
        } else if (layerId === 'KnowledgeGraphArcs' && idObj.data) {
          const arc = idObj.data as BilateralArcDatum;
          const srcName = countryGeometry.getNameByIso(arc.sourceIso) ?? arc.sourceIso;
          const tgtName = countryGeometry.getNameByIso(arc.targetIso) ?? arc.targetIso;
          const sentiment = arc.avgGoldstein < 0 ? 'Conflictual' : 'Cooperative';
          content = `<strong>${srcName} &harr; ${tgtName}</strong><br/>${sentiment} (${arc.avgGoldstein.toFixed(1)})<br/>Events: ${arc.eventCount}`;
        }
      }

      if (content) {
        this.tooltipEl.innerHTML = content;
        this.tooltipEl.style.display = 'block';
        const screenX = event.endPosition.x;
        const screenY = event.endPosition.y;
        this.tooltipEl.style.left = `${screenX + 12}px`;
        this.tooltipEl.style.top = `${screenY - 12}px`;
      } else {
        this.tooltipEl.style.display = 'none';
      }
    }, ScreenSpaceEventType.MOUSE_MOVE);
  }

  private _initMorphListener(): void {
    this.viewer.scene.morphStart.addEventListener(() => {
      this.morphing = true;
      // Temporarily disable requestRenderMode during morph to ensure animation plays
      this.viewer.scene.requestRenderMode = false;
    });

    this.viewer.scene.morphComplete.addEventListener(() => {
      this.morphing = false;
      this.viewer.scene.requestRenderMode = true;

      // Dispatch mode-changed event
      const mode = this.getSceneMode();
      window.dispatchEvent(new CustomEvent('globe-mode-changed', {
        detail: { mode },
      }));

      this.viewer.scene.requestRender();

      // Execute queued flyTo
      if (this.pendingFlyTo) {
        const fly = this.pendingFlyTo;
        this.pendingFlyTo = null;
        fly();
      }
    });
  }

  private _initCustomEventListeners(): void {
    this.viewToggleHandler = ((e: Event) => {
      const detail = (e as CustomEvent<{ mode: '3d' | 'columbus' | '2d' }>).detail;
      if (detail?.mode) {
        this.setSceneMode(detail.mode);
      }
    }) as EventListener;
    window.addEventListener('globe-view-toggle', this.viewToggleHandler);

    this.regionChangeHandler = ((e: Event) => {
      const detail = (e as CustomEvent<{ region: string }>).detail;
      if (detail?.region) {
        this.flyToRegion(detail.region);
      }
    }) as EventListener;
    window.addEventListener('globe-region-change', this.regionChangeHandler);
  }

  // =========================================================================
  // Entity helpers
  // =========================================================================

  private _getEntityIso(entity: Entity): string | null {
    try {
      if (entity.properties && entity.properties.hasProperty('_cesiumIso')) {
        return entity.properties._cesiumIso?.getValue() ?? null;
      }
      // Fall back to standard GeoJSON properties
      const props = entity.properties?.getValue(this.viewer.clock.currentTime);
      if (props) {
        return normalizeCode(props);
      }
    } catch { /* Property access may fail during destruction */ }
    return null;
  }

  private _getEntityLayerId(entity: Entity): string | null {
    try {
      if (entity.properties && entity.properties.hasProperty('_cesiumLayerId')) {
        return entity.properties._cesiumLayerId?.getValue() ?? null;
      }
    } catch { /* ignore */ }
    return null;
  }

  // =========================================================================
  // Public API -- Data push
  // =========================================================================

  /**
   * Push country risk scores from CountryRiskSummary[].
   * First call loads the GeoJSON choropleth; subsequent calls only re-color.
   */
  updateRiskScores(summaries: CountryRiskSummary[]): void {
    this.riskScoreMap.clear();
    for (const s of summaries) {
      this.riskScoreMap.set(s.iso_code.toUpperCase(), s.risk_score);
    }

    if (!this.choroplethLoaded) {
      this._loadChoropleth();
    } else {
      this._recolorChoropleth();
    }
  }

  /**
   * Push forecast data. Updates billboard markers.
   */
  updateForecasts(forecasts: ForecastResponse[]): void {
    this.markerData = [];
    for (const f of forecasts) {
      const isos = new Set<string>();
      const collectIsos = (scenarios: ForecastResponse['scenarios']): void => {
        for (const s of scenarios) {
          for (const entity of s.entities) {
            const upper = entity.toUpperCase();
            if (/^[A-Z]{2}$/.test(upper)) {
              isos.add(upper);
            }
          }
          if (s.child_scenarios.length > 0) {
            collectIsos(s.child_scenarios);
          }
        }
      };
      collectIsos(f.scenarios);

      for (const iso of isos) {
        const centroid = countryGeometry.getCentroid(iso);
        if (!centroid) continue;
        this.markerData.push({
          iso,
          position: centroid,
          probability: f.probability,
          question: f.question,
        });
      }
    }

    this._rebuildMarkers();
    this.viewer.scene.requestRender();
  }

  /**
   * Push H3 hexbin data for the heatmap layer.
   * Async: lazy-loads h3-js on first call.
   */
  updateHeatmapData(data: HexBinDatum[]): void {
    if (!this.h3Module) {
      if (!this.h3Loading) {
        this.h3Loading = true;
        import('h3-js').then((mod) => {
          this.h3Module = mod;
          this.h3Loading = false;
          this._rebuildHeatmap(data);
        }).catch(() => {
          this.h3Loading = false;
          console.warn('[CesiumMap] Failed to load h3-js for heatmap');
        });
      }
      return;
    }
    this._rebuildHeatmap(data);
  }

  /**
   * Push bilateral arc data for the arcs layer.
   */
  updateArcData(data: BilateralArcDatum[]): void {
    this.allArcData = data;
    this._rebuildArcs();
    this.viewer.scene.requestRender();
  }

  /**
   * Push risk delta data for the ScenarioZones layer.
   */
  updateRiskDeltas(deltas: RiskDeltaDatum[]): void {
    this.riskDeltaMap.clear();
    for (const d of deltas) {
      this.riskDeltaMap.set(d.iso.toUpperCase(), d.delta);
    }
    this._rebuildScenarioZones();
    this.viewer.scene.requestRender();
  }

  // =========================================================================
  // Public API -- Selection
  // =========================================================================

  /**
   * Select a country. Highlights the choropleth polygon and filters arcs
   * to only those incident on the selected country.
   * Pass null to deselect and restore global arc display.
   */
  setSelectedCountry(iso: string | null): void {
    this.selectedCountryIso = iso ? iso.toUpperCase() : null;
    this._updateChoroplethHighlight();
    this._rebuildArcs();
    this.viewer.scene.requestRender();
  }

  /**
   * Select a forecast. Collects scenario entity ISOs and triggers arc rebuild.
   * Pass null to deselect.
   */
  setSelectedForecast(forecast: ForecastResponse | null): void {
    this.scenarioIsos.clear();
    this.selectedForecast = forecast;

    if (forecast) {
      const collectEntities = (scenarios: ForecastResponse['scenarios']): void => {
        for (const s of scenarios) {
          for (const entity of s.entities) {
            const upper = entity.toUpperCase();
            if (/^[A-Z]{2}$/.test(upper)) {
              this.scenarioIsos.add(upper);
            }
          }
          if (s.child_scenarios.length > 0) {
            collectEntities(s.child_scenarios);
          }
        }
      };
      collectEntities(forecast.scenarios);
    }

    this._rebuildArcs();
    this._rebuildScenarioZones();
    this.viewer.scene.requestRender();
  }

  // =========================================================================
  // Public API -- Camera
  // =========================================================================

  /**
   * Animate camera to the centroid of the given country.
   * Queues the flyTo if a morph transition is active.
   */
  flyToCountry(iso: string): void {
    const centroid = countryGeometry.getCentroid(iso.toUpperCase());
    if (!centroid) return;

    const fly = () => {
      this.viewer.camera.flyTo({
        destination: Cartesian3.fromDegrees(centroid[0], centroid[1], 3_000_000),
        duration: 1.2,
      });
    };

    if (this.morphing) {
      this.pendingFlyTo = fly;
    } else {
      fly();
    }
  }

  /**
   * Fly to a named region preset (8 presets matching GlobeHud buttons).
   * Queues if a morph transition is active.
   */
  flyToRegion(region: string): void {
    const pov = VIEW_POVS[region];
    if (!pov) return;

    const fly = () => {
      this.viewer.camera.flyTo({
        destination: Cartesian3.fromDegrees(pov.lng, pov.lat, pov.height),
        duration: 1.2,
      });
    };

    if (this.morphing) {
      this.pendingFlyTo = fly;
    } else {
      fly();
    }
  }

  // =========================================================================
  // Public API -- Layer visibility (satisfies LayerController)
  // =========================================================================

  setLayerVisible(layerId: LayerId, visible: boolean): void {
    this.layerVisibility[layerId] = visible;
    this._applyLayerVisibility(layerId);
    this.viewer.scene.requestRender();
  }

  getLayerVisible(layerId: LayerId): boolean {
    return this.layerVisibility[layerId];
  }

  setLayerDefaults(defaults: Partial<Record<LayerId, boolean>>): void {
    for (const [id, visible] of Object.entries(defaults)) {
      if (id in this.layerVisibility) {
        this.layerVisibility[id as LayerId] = visible as boolean;
      }
    }
    // Apply all layer visibility
    for (const id of LAYER_IDS) {
      this._applyLayerVisibility(id);
    }
    this.viewer.scene.requestRender();
  }

  // =========================================================================
  // Public API -- Scene mode
  // =========================================================================

  setSceneMode(mode: '3d' | 'columbus' | '2d'): void {
    switch (mode) {
      case '3d': this.viewer.scene.morphTo3D(2.0); break;
      case 'columbus': this.viewer.scene.morphToColumbusView(2.0); break;
      case '2d': this.viewer.scene.morphTo2D(2.0); break;
    }
    localStorage.setItem(STORAGE_KEY, mode);
  }

  getSceneMode(): '3d' | 'columbus' | '2d' {
    switch (this.viewer.scene.mode) {
      case SceneMode.SCENE3D: return '3d';
      case SceneMode.COLUMBUS_VIEW: return 'columbus';
      case SceneMode.SCENE2D: return '2d';
      default: return '3d';
    }
  }

  // =========================================================================
  // Public API -- Lifecycle
  // =========================================================================

  destroy(): void {
    this.destroyed = true;

    window.removeEventListener('globe-view-toggle', this.viewToggleHandler);
    window.removeEventListener('globe-region-change', this.regionChangeHandler);

    if (this.pickThrottleTimer != null) {
      clearTimeout(this.pickThrottleTimer);
      this.pickThrottleTimer = null;
    }

    this.eventHandler.destroy();

    if (this.tooltipEl.parentNode) {
      this.tooltipEl.parentNode.removeChild(this.tooltipEl);
    }

    this.viewer.destroy();
  }

  // =========================================================================
  // Private -- Layer 1: Choropleth
  // =========================================================================

  /**
   * Load the GeoJSON choropleth. Called once on first updateRiskScores call.
   */
  private async _loadChoropleth(): Promise<void> {
    const geoJson = countryGeometry.getFeatureCollection();
    if (!geoJson) return;

    try {
      const ds = await GeoJsonDataSource.load(geoJson, {
        stroke: Color.fromCssColorString('rgba(80,85,95,0.6)'),
        strokeWidth: 1,
        fill: Color.fromCssColorString('rgba(40,44,52,0.47)'),
        clampToGround: false,
      });

      if (this.destroyed) return;

      this.viewer.dataSources.add(ds);
      this.choroplethDS = ds;
      this.entityIsoMap.clear();

      // Post-load: tag entities with ISO and layerId, apply risk colors
      for (const entity of ds.entities.values) {
        const props = entity.properties?.getValue(this.viewer.clock.currentTime);
        const iso = normalizeCode(props);
        if (!iso) continue;

        this.entityIsoMap.set(iso, entity);

        // Tag entity with metadata for pick identification
        if (entity.properties) {
          entity.properties.addProperty('_cesiumIso', iso);
          entity.properties.addProperty('_cesiumLayerId', 'ForecastRiskChoropleth');
        }

        // Minimize rhumb-line subdivision to prevent worker crash on complex polygons.
        // Large granularity = fewer subdivision points along polygon edges.
        if (entity.polygon) {
          entity.polygon.arcType = new ConstantProperty(ArcType.GEODESIC);
          entity.polygon.granularity = new ConstantProperty(Math.PI);
          entity.polygon.outline = new ConstantProperty(false) as any;
        }

        // Apply risk color
        const score = this.riskScoreMap.get(iso);
        if (score !== undefined && entity.polygon) {
          entity.polygon.material = new ColorMaterialProperty(riskColor(score / 100));
        }

        // Apply visibility
        entity.show = this.layerVisibility.ForecastRiskChoropleth;
      }

      this.choroplethLoaded = true;
      this._updateChoroplethHighlight();
      this.viewer.scene.requestRender();
    } catch (err) {
      console.warn('[CesiumMap] Failed to load choropleth GeoJSON:', err);
    }
  }

  /**
   * Re-color existing choropleth entities without reloading GeoJSON.
   */
  private _recolorChoropleth(): void {
    if (!this.choroplethDS) return;

    for (const [iso, entity] of this.entityIsoMap) {
      if (!entity.polygon) continue;
      const score = this.riskScoreMap.get(iso);
      if (score !== undefined) {
        entity.polygon.material = new ColorMaterialProperty(riskColor(score / 100));
      } else {
        entity.polygon.material = new ColorMaterialProperty(
          Color.fromCssColorString('rgba(40,44,52,0.47)'),
        );
      }
    }

    this.viewer.scene.requestRender();
  }

  /**
   * Update choropleth highlight: bright outline on selected country,
   * reset previous selection.
   */
  private _updateChoroplethHighlight(): void {
    // Reset previous highlight
    if (this.prevSelectedEntity?.polygon) {
      this.prevSelectedEntity.polygon.outlineColor = new ConstantProperty(
        Color.fromCssColorString('rgba(80,85,95,0.6)'),
      ) as any;
      this.prevSelectedEntity.polygon.outlineWidth = new ConstantProperty(1) as any;
    }

    // Apply new highlight
    if (this.selectedCountryIso) {
      const entity = this.entityIsoMap.get(this.selectedCountryIso);
      if (entity?.polygon) {
        entity.polygon.outline = new ConstantProperty(true) as any;
        entity.polygon.outlineColor = new ConstantProperty(
          Color.fromCssColorString('#4080dd'),
        ) as any;
        entity.polygon.outlineWidth = new ConstantProperty(3) as any;
        this.prevSelectedEntity = entity;
      }
    } else {
      this.prevSelectedEntity = null;
    }
  }

  // =========================================================================
  // Private -- Layer 2: Markers
  // =========================================================================

  private _rebuildMarkers(): void {
    this.billboards.removeAll();

    if (!this.layerVisibility.ActiveForecastMarkers) return;

    for (const marker of this.markerData) {
      const [lng, lat] = marker.position;
      this.billboards.add({
        position: Cartesian3.fromDegrees(lng, lat, 0),
        image: createMarkerCanvas(marker.probability),
        id: { layerId: 'ActiveForecastMarkers', iso: marker.iso, data: marker },
        scale: 1.0,
        horizontalOrigin: HorizontalOrigin.CENTER,
        verticalOrigin: VerticalOrigin.CENTER,
      });
    }
  }

  // =========================================================================
  // Private -- Layer 3: Arcs (with per-country filtering)
  // =========================================================================

  /**
   * Rebuild arc polylines. Filters to incident arcs when a country is selected,
   * displays global bilateral arcs when no country is selected.
   */
  private _rebuildArcs(): void {
    this.polylines.removeAll();

    if (!this.layerVisibility.KnowledgeGraphArcs) return;

    // Determine filtered arc set
    let arcs: BilateralArcDatum[];
    if (this.selectedCountryIso) {
      arcs = this.allArcData.filter(
        (arc) =>
          arc.sourceIso === this.selectedCountryIso ||
          arc.targetIso === this.selectedCountryIso,
      );
    } else {
      arcs = this.allArcData;
    }

    for (const arc of arcs) {
      // NaN guard on coordinates
      const [sLng, sLat] = arc.source;
      const [eLng, eLat] = arc.target;
      if (
        !Number.isFinite(sLng) || !Number.isFinite(sLat) ||
        !Number.isFinite(eLng) || !Number.isFinite(eLat)
      ) continue;

      const positions = generateArcPositions(sLng, sLat, eLng, eLat, 40);
      const color = arc.avgGoldstein < 0
        ? Color.fromCssColorString('rgba(255,80,80,0.7)')
        : Color.fromCssColorString('rgba(80,180,255,0.7)');
      const width = Math.max(1, Math.min(5, arc.eventCount / 20));

      this.polylines.add({
        positions,
        width,
        material: Material.fromType('Color', { color }),
        id: { layerId: 'KnowledgeGraphArcs', data: arc },
      });
    }

    this.viewer.scene.requestRender();
  }

  // =========================================================================
  // Private -- Layer 4: Heatmap
  // =========================================================================

  private _rebuildHeatmap(data: HexBinDatum[]): void {
    this.points.removeAll();

    if (!this.layerVisibility.GDELTEventHeatmap || !this.h3Module) return;

    const maxWeight = data.reduce((max, d) => Math.max(max, d.weight), 1);

    for (const hex of data) {
      try {
        const [lat, lng] = this.h3Module.cellToLatLng(hex.h3_index);
        if (!Number.isFinite(lat) || !Number.isFinite(lng)) continue;

        const t = Math.min(1, hex.weight / maxWeight);
        const g = Math.round(255 * (1 - t));
        const a = 0.4 + 0.5 * t;
        const color = Color.fromCssColorString(
          `rgba(255,${g},0,${a.toFixed(2)})`,
        );

        this.points.add({
          position: Cartesian3.fromDegrees(lng, lat, 0),
          color,
          pixelSize: 6 + t * 6,
          id: { layerId: 'GDELTEventHeatmap', data: hex },
        });
      } catch {
        // Invalid H3 index -- skip
        continue;
      }
    }

    this.viewer.scene.requestRender();
  }

  // =========================================================================
  // Private -- Layer 5: Scenario Zones
  // =========================================================================

  /**
   * Rebuild scenario zone entities. Uses CustomDataSource for independent
   * visibility toggling from the choropleth layer.
   *
   * Two modes:
   *   a) Forecast selected + scenarioIsos: accent-colored highlights for scenario entities
   *   b) No forecast selected + riskDeltaMap: red/green risk change zones
   */
  private _rebuildScenarioZones(): void {
    this.scenarioDS.entities.removeAll();

    if (!this.layerVisibility.ScenarioZones) return;

    if (this.selectedForecast && this.scenarioIsos.size > 0) {
      // Mode A: scenario entity highlights (accent color)
      const accentFill = Color.fromCssColorString('rgba(0,212,255,0.24)');
      const accentOutline = Color.fromCssColorString('rgba(0,212,255,0.63)');

      for (const iso of this.scenarioIsos) {
        const feature = countryGeometry.getFeatureByIso(iso);
        if (!feature) continue;

        const hierarchy = geoJsonToHierarchy(feature.geometry);
        if (!hierarchy) continue;

        this.scenarioDS.entities.add({
          polygon: {
            hierarchy,
            height: 200,
            material: accentFill,
            outline: true,
            outlineColor: accentOutline,
            outlineWidth: 1,
          },
          properties: {
            _cesiumIso: iso,
            _cesiumLayerId: 'ScenarioZones',
          },
        });
      }
    } else if (this.riskDeltaMap.size > 0) {
      // Mode B: risk delta visualization
      for (const [iso, delta] of this.riskDeltaMap) {
        const feature = countryGeometry.getFeatureByIso(iso);
        if (!feature) continue;

        const hierarchy = geoJsonToHierarchy(feature.geometry);
        if (!hierarchy) continue;

        const fillColor = delta > 0
          ? Color.fromCssColorString('rgba(220,60,60,0.4)')
          : Color.fromCssColorString('rgba(40,180,120,0.4)');

        const outlineColor = delta > 0
          ? Color.fromCssColorString('rgba(220,60,60,0.6)')
          : Color.fromCssColorString('rgba(40,180,120,0.6)');

        this.scenarioDS.entities.add({
          polygon: {
            hierarchy,
            height: 200,
            material: fillColor,
            outline: true,
            outlineColor,
            outlineWidth: 1,
          },
          properties: {
            _cesiumIso: iso,
            _cesiumLayerId: 'ScenarioZones',
          },
        });
      }
    }

    this.viewer.scene.requestRender();
  }

  // =========================================================================
  // Private -- Layer visibility
  // =========================================================================

  private _applyLayerVisibility(layerId: LayerId): void {
    const visible = this.layerVisibility[layerId];

    switch (layerId) {
      case 'ForecastRiskChoropleth':
        if (this.choroplethDS) {
          for (const entity of this.choroplethDS.entities.values) {
            entity.show = visible;
          }
        }
        break;
      case 'ActiveForecastMarkers':
        this.billboards.show = visible;
        break;
      case 'KnowledgeGraphArcs':
        this.polylines.show = visible;
        break;
      case 'GDELTEventHeatmap':
        this.points.show = visible;
        break;
      case 'ScenarioZones':
        this.scenarioDS.show = visible;
        break;
    }
  }
}
