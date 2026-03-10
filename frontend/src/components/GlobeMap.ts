/**
 * GlobeMap -- 3D interactive globe using globe.gl (Three.js-based).
 *
 * Exposes the exact same 13-method public API as DeckGLMap so that
 * MapContainer (Plan 02) can delegate to either renderer transparently.
 *
 * Layers (mapped to globe.gl data channels):
 *   1. ForecastRiskChoropleth  (polygonsData)       -- country fill by risk_score
 *   2. ActiveForecastMarkers   (htmlElementsData)   -- DOM dots at forecast centroids
 *   3. KnowledgeGraphArcs      (arcsData)           -- bilateral arcs (global or per-country)
 *   4. GDELTEventHeatmap       (pointsData)         -- colored dots at H3 hex centers
 *   5. ScenarioZones           (polygonsData)       -- risk delta / scenario highlights
 *
 * Key differences from DeckGLMap:
 *   - Colors are CSS strings, not RGBA tuples (globe.gl convention)
 *   - All data mutations go through scheduleFlush() to prevent per-push rebuilds
 *   - getMap() always returns null (no maplibre in 3D mode)
 *   - Custom atmosphere glow via Three.js scene objects (dynamic import)
 *   - Auto-rotate with 120s idle timeout
 *
 * Reference: WM GlobeMap.ts (2,188 lines) adapted to Geopol's 5-layer model.
 */

import Globe from 'globe.gl';
import type { GlobeInstance, ConfigOptions } from 'globe.gl';
import { countryGeometry, normalizeCode } from '@/services/country-geometry.ts';
import { h } from '@/utils/dom-utils.ts';
import type { Feature, Geometry } from 'geojson';
import type { ForecastResponse, CountryRiskSummary } from '@/types/api.ts';
import type { LayerId, HexBinDatum, BilateralArcDatum, RiskDeltaDatum } from '@/components/DeckGLMap.ts';

// ---------------------------------------------------------------------------
// View presets -- camera positions for regional views
// ---------------------------------------------------------------------------

export const VIEW_POVS: Record<string, { lat: number; lng: number; altitude: number }> = {
  global:  { lat: 20,  lng:   0,  altitude: 1.8 },
  america: { lat: 20,  lng: -90,  altitude: 1.5 },
  mena:    { lat: 25,  lng:  40,  altitude: 1.2 },
  eu:      { lat: 50,  lng:  10,  altitude: 1.2 },
  asia:    { lat: 35,  lng: 105,  altitude: 1.5 },
  latam:   { lat: -15, lng: -60,  altitude: 1.5 },
  africa:  { lat:  5,  lng:  20,  altitude: 1.5 },
  oceania: { lat: -25, lng: 140,  altitude: 1.5 },
};

// ---------------------------------------------------------------------------
// Color utilities -- CSS string output for globe.gl
// ---------------------------------------------------------------------------

/**
 * Blue-to-gray-to-red diverging color scale for risk scores [0, 1].
 * ~10-15% brightness boost vs DeckGLMap to compensate for dark topo texture
 * and curved surface light falloff on the 3D sphere.
 */
function riskColorCSS(score: number, alpha = 0.7): string {
  const t = Math.max(0, Math.min(1, score));
  let r: number, g: number, b: number;
  if (t <= 0.5) {
    const u = t * 2;
    r = Math.round(70 + (128 - 70) * u);
    g = Math.round(130 + (128 - 130) * u);
    b = Math.round(180 + (128 - 180) * u);
  } else {
    const u = (t - 0.5) * 2;
    r = Math.round(128 + (220 - 128) * u);
    g = Math.round(128 + (50 - 128) * u);
    b = Math.round(128 + (50 - 128) * u);
  }
  // Brightness boost for 3D: +15 per channel, clamped
  r = Math.min(255, r + 15);
  g = Math.min(255, g + 15);
  b = Math.min(255, b + 15);
  return `rgba(${r},${g},${b},${alpha})`;
}

/** Default country fill when no risk data is available. */
const DEFAULT_FILL = 'rgba(40,44,52,0.47)';

/** Country polygon stroke color. */
const POLYGON_STROKE = 'rgba(80,85,95,0.6)';

/** Transparent side color (polygons are flat on the sphere). */
const POLYGON_SIDE = 'rgba(0,0,0,0)';

/** Accent color for scenario zone highlights. */
const ACCENT_FILL = 'rgba(0,212,255,0.24)';
const ACCENT_STROKE = 'rgba(0,212,255,0.63)';

// ---------------------------------------------------------------------------
// Orbit controls type (subset exposed by globe.controls())
// ---------------------------------------------------------------------------

interface GlobeControlsLike {
  autoRotate: boolean;
  autoRotateSpeed: number;
  enablePan: boolean;
  enableZoom: boolean;
  zoomSpeed: number;
  minDistance: number;
  maxDistance: number;
  enableDamping: boolean;
}

// ---------------------------------------------------------------------------
// Internal marker datum for htmlElementsData
// ---------------------------------------------------------------------------

interface MarkerDatum {
  iso: string;
  lat: number;
  lng: number;
  probability: number;
  question: string;
}

// ---------------------------------------------------------------------------
// Internal arc datum for arcsData
// ---------------------------------------------------------------------------

interface ArcDatum {
  sourceIso: string;
  targetIso: string;
  startLat: number;
  startLng: number;
  endLat: number;
  endLng: number;
}

// ---------------------------------------------------------------------------
// Polygon datum for polygonsData (discriminated union)
// ---------------------------------------------------------------------------

interface ChoroplethPolygon {
  _kind: 'choropleth';
  feature: Feature<Geometry>;
  iso: string;
  color: string;
}

interface ScenarioPolygon {
  _kind: 'scenario';
  feature: Feature<Geometry>;
  iso: string;
  fillColor: string;
  strokeColor: string;
}

type GlobePolygon = ChoroplethPolygon | ScenarioPolygon;

// ---------------------------------------------------------------------------
// GlobeMap
// ---------------------------------------------------------------------------

export class GlobeMap {
  private readonly container: HTMLElement;
  private wrapper: HTMLElement | null = null;
  private globe: GlobeInstance | null = null;
  private controls: GlobeControlsLike | null = null;

  // Three.js scene objects (created via dynamic import)
  private outerGlow: any = null;
  private innerGlow: any = null;
  private accentLight: any = null;
  private extrasAnimFrameId: number | null = null;

  // Lifecycle
  private initialized = false;
  private destroyed = false;

  // Debounced flush timers
  private flushTimer: ReturnType<typeof setTimeout> | null = null;
  private flushMaxTimer: ReturnType<typeof setTimeout> | null = null;

  // Auto-rotate idle timer
  private autoRotateTimer: ReturnType<typeof setTimeout> | null = null;

  // ResizeObserver
  private resizeObserver: ResizeObserver | null = null;

  // Layer toggle state -- defaults all true
  private readonly layerVisible: Record<LayerId, boolean> = {
    ForecastRiskChoropleth: true,
    ActiveForecastMarkers: true,
    KnowledgeGraphArcs: true,
    GDELTEventHeatmap: true,
    ScenarioZones: true,
  };

  // Data stores
  private riskScores = new Map<string, number>();
  private markers: MarkerDatum[] = [];
  private hexBinData: HexBinDatum[] = [];
  private bilateralArcs: BilateralArcDatum[] = [];
  private riskDeltaIsos = new Map<string, number>();
  private selectedCountry: string | null = null;
  private selectedForecast: ForecastResponse | null = null;
  private scenarioIsos = new Set<string>();
  private arcs: ArcDatum[] = [];

  constructor(container: HTMLElement) {
    this.container = container;
    this.setupDOM();
    this.initGlobe();

    // ResizeObserver: recalculate globe dimensions when container changes
    this.resizeObserver = new ResizeObserver(() => {
      if (!this.globe || this.destroyed) return;
      const w = this.container.clientWidth || window.innerWidth;
      const h = this.container.clientHeight || window.innerHeight;
      if (w > 0 && h > 0) this.globe.width(w).height(h);
    });
    this.resizeObserver.observe(this.container);
  }

  // =========================================================================
  // Public API (13 methods -- identical signatures to DeckGLMap)
  // =========================================================================

  /**
   * Push country risk scores from CountryRiskSummary[].
   * Rebuilds the choropleth layer with new colors.
   */
  updateRiskScores(summaries: CountryRiskSummary[]): void {
    this.riskScores.clear();
    for (const s of summaries) {
      this.riskScores.set(s.iso_code.toUpperCase(), s.risk_score);
    }
    this.scheduleFlush();
  }

  /**
   * Push forecast data. Updates scatter markers + scenario zones.
   */
  updateForecasts(forecasts: ForecastResponse[]): void {
    this.markers = [];
    for (const f of forecasts) {
      // Extract country ISOs from scenario entity lists (not calibration.category
      // which contains CAMEO strings like "conflict", not ISO codes).
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

        this.markers.push({
          iso,
          lat: centroid[1],
          lng: centroid[0],
          probability: f.probability,
          question: f.question,
        });
      }
    }
    this.scheduleFlush();
  }

  /**
   * Select a country. Shows KnowledgeGraphArcs for that country's forecasts.
   * Flies to the country centroid. Pass null to deselect.
   */
  setSelectedCountry(iso: string | null): void {
    this.selectedCountry = iso ? iso.toUpperCase() : null;
    this.buildArcsForCountry();

    if (this.selectedCountry) {
      this.flyToCountry(this.selectedCountry);
    }

    this.scheduleFlush();
  }

  /**
   * Select a forecast. Highlights scenario-relevant countries.
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

    this.scheduleFlush();
  }

  /**
   * Push H3 hexbin data for the heatmap layer.
   * Replaces any existing heatmap data and rebuilds.
   */
  updateHeatmapData(data: HexBinDatum[]): void {
    this.hexBinData = data;
    this.scheduleFlush();
  }

  /**
   * Push bilateral arc data for the arcs layer (global view).
   * When no country is selected, these arcs render as the KnowledgeGraphArcs layer.
   */
  updateArcData(data: BilateralArcDatum[]): void {
    this.bilateralArcs = data;
    this.scheduleFlush();
  }

  /**
   * Push risk delta data for the ScenarioZones layer.
   * When no forecast is selected, renders countries with significant risk changes.
   */
  updateRiskDeltas(deltas: RiskDeltaDatum[]): void {
    this.riskDeltaIsos.clear();
    for (const d of deltas) {
      this.riskDeltaIsos.set(d.iso.toUpperCase(), d.delta);
    }
    this.scheduleFlush();
  }

  /**
   * Animate camera to the centroid of the given country.
   * Uses globe.gl pointOfView with 1.2s animation.
   */
  public flyToCountry(iso: string): void {
    if (!this.globe) return;
    const centroid = countryGeometry.getCentroid(iso.toUpperCase());
    if (!centroid) return;
    // centroid is [lon, lat] -- globe.gl wants { lat, lng, altitude }
    this.globe.pointOfView(
      { lat: centroid[1], lng: centroid[0], altitude: 1.2 },
      1200,
    );
  }

  /**
   * Toggle visibility of an individual layer. Triggers flush.
   */
  public setLayerVisible(layerId: LayerId, visible: boolean): void {
    this.layerVisible[layerId] = visible;
    this.scheduleFlush();
  }

  /**
   * Batch-set layer visibility defaults. Only modifies keys present in the
   * defaults map; others remain unchanged.
   */
  public setLayerDefaults(defaults: Partial<Record<LayerId, boolean>>): void {
    for (const [id, visible] of Object.entries(defaults)) {
      if (id in this.layerVisible) {
        this.layerVisible[id as LayerId] = visible as boolean;
      }
    }
    this.scheduleFlush();
  }

  /**
   * Query the current visibility state of a layer.
   */
  public getLayerVisible(layerId: LayerId): boolean {
    return this.layerVisible[layerId];
  }

  /**
   * Access the underlying map instance.
   * Always returns null in 3D mode -- no maplibre.
   */
  public getMap(): null {
    return null;
  }

  /**
   * Pause auto-rotate and animation frame to reduce GPU usage when hidden.
   * Called by MapContainer when the 2D view becomes active.
   */
  pauseAnimation(): void {
    if (this.controls) this.controls.autoRotate = false;
    if (this.autoRotateTimer) {
      clearTimeout(this.autoRotateTimer);
      this.autoRotateTimer = null;
    }
    if (this.extrasAnimFrameId != null) {
      cancelAnimationFrame(this.extrasAnimFrameId);
      this.extrasAnimFrameId = null;
    }
  }

  /**
   * Resume auto-rotate and atmosphere glow animation when made visible again.
   * Called by MapContainer when the 3D view becomes active.
   */
  resumeAnimation(): void {
    if (this.destroyed) return;
    // Restart glow rotation animation
    if (this.outerGlow && this.extrasAnimFrameId == null) {
      const animateGlow = (): void => {
        if (this.destroyed) return;
        if (this.outerGlow) this.outerGlow.rotation.y += 0.0003;
        this.extrasAnimFrameId = requestAnimationFrame(animateGlow);
      };
      animateGlow();
    }
    // Re-enable auto-rotate (will start after idle timeout)
    if (this.controls) this.controls.autoRotate = true;
  }

  /**
   * Fly to a named region preset. Used by MapContainer for GlobeHud region dispatch.
   * @param region Key into VIEW_POVS (e.g. 'global', 'eu', 'mena')
   */
  flyToRegion(region: string): void {
    const pov = VIEW_POVS[region];
    if (!pov || !this.globe) return;
    this.globe.pointOfView(pov, 1200);
  }

  /**
   * Clean up globe, Three.js objects, timers, and observers.
   */
  destroy(): void {
    this.destroyed = true;

    // Cancel animation frame
    if (this.extrasAnimFrameId != null) {
      cancelAnimationFrame(this.extrasAnimFrameId);
      this.extrasAnimFrameId = null;
    }

    // Dispose Three.js scene objects
    const scene = this.globe?.scene();
    for (const obj of [this.outerGlow, this.innerGlow, this.accentLight]) {
      if (!obj) continue;
      if (scene) scene.remove(obj);
      if (obj.geometry) obj.geometry.dispose();
      if (obj.material) obj.material.dispose();
    }

    // Dispose globe material if upgraded to MeshStandardMaterial
    if (this.globe) {
      const mat = this.globe.globeMaterial();
      if (mat && (mat as any).isMeshStandardMaterial) mat.dispose();
    }

    this.outerGlow = null;
    this.innerGlow = null;
    this.accentLight = null;

    // Clear timers
    if (this.flushTimer) { clearTimeout(this.flushTimer); this.flushTimer = null; }
    if (this.flushMaxTimer) { clearTimeout(this.flushMaxTimer); this.flushMaxTimer = null; }
    if (this.autoRotateTimer) { clearTimeout(this.autoRotateTimer); this.autoRotateTimer = null; }

    this.controls = null;

    // ResizeObserver
    this.resizeObserver?.disconnect();
    this.resizeObserver = null;

    // Destroy globe.gl instance
    if (this.globe) {
      try { this.globe._destructor(); } catch { /* ignore */ }
      this.globe = null;
    }

    // Remove DOM
    if (this.wrapper) {
      this.wrapper.remove();
      this.wrapper = null;
    }
  }

  // =========================================================================
  // DOM setup
  // =========================================================================

  private setupDOM(): void {
    this.wrapper = h('div', {
      className: 'globe-map-wrapper',
      role: 'application',
      'aria-label': 'Geopolitical forecast 3D globe',
      style: 'position: relative; width: 100%; height: 100%; overflow: hidden; background: #000;',
    });

    // Attribution
    const attribution = h('div', { className: 'map-attribution' });
    attribution.innerHTML =
      '&copy; <a href="https://www.naturalearthdata.com" target="_blank" rel="noopener">Natural Earth</a> ' +
      '&copy; <a href="https://www.openstreetmap.org/copyright" target="_blank" rel="noopener">OpenStreetMap</a>';
    this.wrapper.appendChild(attribution);

    this.container.appendChild(this.wrapper);
  }

  // =========================================================================
  // Globe initialization
  // =========================================================================

  private initGlobe(): void {
    if (!this.wrapper || this.destroyed) return;

    const config: ConfigOptions = {
      animateIn: false,
      rendererConfig: {
        powerPreference: 'high-performance',
        antialias: window.devicePixelRatio > 1,
      },
    };

    const globe = new Globe(this.wrapper, config) as GlobeInstance;

    if (this.destroyed) {
      globe._destructor();
      return;
    }

    // Initial sizing: use container dimensions, fall back to window
    const initW = this.container.clientWidth || window.innerWidth;
    const initH = this.container.clientHeight || window.innerHeight;

    globe
      .globeImageUrl('/textures/earth-topo-bathy.jpg')
      .backgroundImageUrl('')
      .showAtmosphere(false)                   // Custom atmosphere via Three.js
      .atmosphereColor('#4080dd')
      .atmosphereAltitude(0.18)
      .width(initW)
      .height(initH)
      .pathTransitionDuration(0);

    // Force canvas to fill container
    const glCanvas = this.wrapper.querySelector('canvas');
    if (glCanvas) {
      (glCanvas as HTMLElement).style.cssText =
        'position:absolute;top:0;left:0;width:100% !important;height:100% !important;';
    }

    // Orbit controls
    const controls = globe.controls() as GlobeControlsLike;
    this.controls = controls;
    controls.autoRotate = true;
    controls.autoRotateSpeed = 0.3;
    controls.enablePan = false;                // Rotate + zoom only (CONTEXT.md)
    controls.enableZoom = true;
    controls.zoomSpeed = 1.4;
    controls.minDistance = 101;
    controls.maxDistance = 600;
    controls.enableDamping = true;

    // Configure polygon accessors (set once, data changes on flush)
    (globe as any)
      .polygonGeoJsonGeometry((d: GlobePolygon) => d.feature.geometry)
      .polygonCapColor((d: GlobePolygon) => {
        if (d._kind === 'choropleth') return d.color;
        return d.fillColor;
      })
      .polygonStrokeColor((d: GlobePolygon) => {
        if (d._kind === 'scenario') return d.strokeColor;
        return POLYGON_STROKE;
      })
      .polygonSideColor(() => POLYGON_SIDE)
      .polygonAltitude((d: GlobePolygon) => {
        // Scenario/delta polygons render above choropleth
        return d._kind === 'scenario' ? 0.004 : 0.002;
      })
      .onPolygonClick((polygon: any, _event: MouseEvent, _coords: { lat: number; lng: number }) => {
        const feat = polygon as GlobePolygon;
        const code = normalizeCode(feat.feature.properties);
        if (!code) return;
        window.dispatchEvent(
          new CustomEvent('country-selected', {
            detail: { iso: code },
            bubbles: true,
          }),
        );
      });

    // Configure HTML marker accessors (set once)
    globe
      .htmlLat((d: object) => (d as MarkerDatum).lat)
      .htmlLng((d: object) => (d as MarkerDatum).lng)
      .htmlAltitude(0.01)
      .htmlElement((d: object) => this.buildMarkerElement(d as MarkerDatum));

    // Configure arc accessors (set once)
    (globe as any)
      .arcStartLat((d: any) => d.startLat)
      .arcStartLng((d: any) => d.startLng)
      .arcEndLat((d: any) => d.endLat)
      .arcEndLng((d: any) => d.endLng)
      .arcColor((d: any) => {
        // Bilateral arcs: sentiment-based coloring
        if (d.avgGoldstein !== undefined) {
          return d.avgGoldstein < 0
            ? ['rgba(255,80,80,0.15)', 'rgba(255,80,80,0.7)']    // conflictual
            : ['rgba(80,180,255,0.15)', 'rgba(80,180,255,0.7)'];  // cooperative
        }
        // Per-country arcs: fixed gradient
        return ['rgba(0,200,255,0.15)', 'rgba(0,200,255,0.7)'];
      })
      .arcDashLength(0.4)
      .arcDashGap(0.2)
      .arcDashAnimateTime(2000)
      .arcStroke((d: any) => {
        if (d.eventCount !== undefined) {
          return Math.max(0.3, Math.min(2, d.eventCount / 20));
        }
        return 0.5;
      });

    // Configure point accessors for heatmap (set once)
    (globe as any)
      .pointLat((d: any) => d.lat)
      .pointLng((d: any) => d.lng)
      .pointColor((d: any) => d.color)
      .pointAltitude(0.005)
      .pointRadius((d: any) => d.radius)
      .pointsMerge(true);

    // Default camera position
    globe.pointOfView(VIEW_POVS.global!, 0);

    this.globe = globe;

    // Apply atmosphere glow after globe is ready.
    // globe.gl's internal mesh needs time to initialize (texture load).
    // Use setTimeout like WM (800ms) -- safe for async texture load.
    setTimeout(() => {
      if (!this.destroyed) {
        this.applyAtmosphereGlow();
      }
    }, 800);

    // Auto-rotate interaction handlers
    this.setupAutoRotate();

    // Mark initialized (enables flush)
    this.initialized = true;

    // WebGL context loss/restore
    if (glCanvas) {
      glCanvas.addEventListener('webglcontextlost', (e) => {
        e.preventDefault();
        console.warn('[GlobeMap] WebGL context lost');
      });
      glCanvas.addEventListener('webglcontextrestored', () => {
        console.info('[GlobeMap] WebGL context restored');
      });
    }
  }

  // =========================================================================
  // Atmosphere glow (custom Three.js scene objects)
  // =========================================================================

  /**
   * Apply custom atmosphere glow using Geopol blue (#4080dd).
   * Dynamically imports Three.js to avoid loading 600KB at module parse time.
   * Adapted from WM GlobeMap.ts applyEnhancedVisuals() -- NO starfield.
   */
  private async applyAtmosphereGlow(): Promise<void> {
    if (!this.globe || this.destroyed) return;

    try {
      const THREE = await import('three');
      const scene = this.globe.scene();

      // Upgrade globe material to MeshStandardMaterial for matte analytical look
      const oldMat = this.globe.globeMaterial();
      if (oldMat) {
        const stdMat = new THREE.MeshStandardMaterial({
          color: 0xffffff,
          roughness: 0.8,
          metalness: 0.1,
          emissive: new THREE.Color(0x0a1f2e),
          emissiveIntensity: 0.3,
        });
        if ((oldMat as any).map) stdMat.map = (oldMat as any).map;
        (this.globe as any).globeMaterial(stdMat);
      }

      // Accent light (Geopol blue, not WM cyan)
      this.accentLight = new THREE.PointLight(0x4080dd, 0.3);
      this.accentLight.position.set(-10, -10, -10);
      scene.add(this.accentLight);

      // Outer glow sphere (BackSide rendering -- visible from inside)
      const outerGeo = new THREE.SphereGeometry(2.15, 64, 64);
      const outerMat = new THREE.MeshBasicMaterial({
        color: 0x4080dd,
        side: THREE.BackSide,
        transparent: true,
        opacity: 0.15,
      });
      this.outerGlow = new THREE.Mesh(outerGeo, outerMat);
      scene.add(this.outerGlow);

      // Inner glow sphere (subtler, slightly different blue)
      const innerGeo = new THREE.SphereGeometry(2.08, 64, 64);
      const innerMat = new THREE.MeshBasicMaterial({
        color: 0x3060aa,
        side: THREE.BackSide,
        transparent: true,
        opacity: 0.1,
      });
      this.innerGlow = new THREE.Mesh(innerGeo, innerMat);
      scene.add(this.innerGlow);

      // Slow rotation animation for the outer glow
      const animateGlow = (): void => {
        if (this.destroyed) return;
        if (this.outerGlow) this.outerGlow.rotation.y += 0.0003;
        this.extrasAnimFrameId = requestAnimationFrame(animateGlow);
      };
      animateGlow();
    } catch {
      // Cosmetic enhancement -- failure is non-critical
      console.warn('[GlobeMap] Failed to apply atmosphere glow');
    }
  }

  // =========================================================================
  // Auto-rotate with idle timer
  // =========================================================================

  /**
   * Pause auto-rotate on user interaction, resume after 120s idle.
   * Longer timeout than WM (60s) -- analytical tool, user studies data.
   */
  private setupAutoRotate(): void {
    if (!this.wrapper || !this.controls) return;

    const canvas = this.wrapper.querySelector('canvas');
    if (!canvas) return;

    const pause = (): void => {
      if (this.controls) this.controls.autoRotate = false;
      if (this.autoRotateTimer) {
        clearTimeout(this.autoRotateTimer);
        this.autoRotateTimer = null;
      }
    };

    const scheduleResume = (): void => {
      if (this.autoRotateTimer) clearTimeout(this.autoRotateTimer);
      this.autoRotateTimer = setTimeout(() => {
        if (this.controls && !this.destroyed) {
          this.controls.autoRotate = true;
        }
      }, 120_000); // 120s idle timeout (CONTEXT.md)
    };

    canvas.addEventListener('mousedown', pause);
    canvas.addEventListener('touchstart', pause, { passive: true });
    canvas.addEventListener('mouseup', scheduleResume);
    canvas.addEventListener('touchend', scheduleResume);
    canvas.addEventListener('wheel', () => { pause(); scheduleResume(); }, { passive: true });
  }

  // =========================================================================
  // Debounced flush (WM dual-timer pattern)
  // =========================================================================

  /**
   * Schedule a debounced flush of all data to globe.gl channels.
   * 100ms trailing debounce + 300ms max wait to coalesce rapid updates.
   * CRITICAL: globe.gl triggers full geometry rebuild on every data setter.
   * Without this, 7 data pushes = 7 expensive rebuilds instead of 1.
   */
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

  /**
   * Immediately push all data to globe.gl channels based on current state
   * and layer visibility. Called by the debounce timers.
   */
  private flushImmediate(): void {
    if (!this.globe || !this.initialized || this.destroyed) return;

    try {
      this.flushPolygons();
      this.flushMarkers();
      this.flushArcs();
      this.flushPoints();
    } catch (err) {
      if (import.meta.env.DEV) console.warn('[GlobeMap] flush error', err);
    }
  }

  // =========================================================================
  // Polygon flush (Layer 1 choropleth + Layer 5 scenario/delta)
  // =========================================================================

  /**
   * Build combined polygon array for choropleth + scenario zones.
   * globe.gl has ONE polygonsData channel -- both layers share it.
   * Scenario/delta polygons use altitude 0.004 to render above choropleth (0.002).
   */
  private flushPolygons(): void {
    if (!this.globe) return;

    const polys: GlobePolygon[] = [];
    const geoJson = countryGeometry.getFeatureCollection();

    // Layer 1: ForecastRiskChoropleth
    if (this.layerVisible.ForecastRiskChoropleth && geoJson) {
      for (const feature of geoJson.features) {
        const code = normalizeCode(feature.properties);
        if (!code) continue;

        const score = this.riskScores.get(code);
        const color = score !== undefined
          ? riskColorCSS(score / 100)
          : DEFAULT_FILL;

        polys.push({
          _kind: 'choropleth',
          feature,
          iso: code,
          color,
        });
      }
    }

    // Layer 5: ScenarioZones / Risk Delta Regions
    if (this.layerVisible.ScenarioZones && geoJson) {
      if (this.selectedForecast && this.scenarioIsos.size > 0) {
        // Mode A: scenario entity highlights
        for (const feature of geoJson.features) {
          const code = normalizeCode(feature.properties);
          if (!code || !this.scenarioIsos.has(code)) continue;

          polys.push({
            _kind: 'scenario',
            feature,
            iso: code,
            fillColor: ACCENT_FILL,
            strokeColor: ACCENT_STROKE,
          });
        }
      } else if (this.riskDeltaIsos.size > 0) {
        // Mode B: risk delta visualization
        for (const feature of geoJson.features) {
          const code = normalizeCode(feature.properties);
          if (!code || !this.riskDeltaIsos.has(code)) continue;

          const delta = this.riskDeltaIsos.get(code) ?? 0;
          const alpha = Math.round(60 + Math.min(100, (Math.abs(delta) / 50) * 100)) / 255;

          if (delta > 0) {
            // Risk worsening: red fill
            polys.push({
              _kind: 'scenario',
              feature,
              iso: code,
              fillColor: `rgba(235,65,65,${alpha.toFixed(2)})`,
              strokeColor: 'rgba(220,50,50,0.47)',
            });
          } else {
            // Risk improving: green fill
            polys.push({
              _kind: 'scenario',
              feature,
              iso: code,
              fillColor: `rgba(65,195,95,${alpha.toFixed(2)})`,
              strokeColor: 'rgba(50,180,80,0.47)',
            });
          }
        }
      }
    }

    (this.globe as any).polygonsData(polys);
  }

  // =========================================================================
  // HTML marker flush (Layer 2: ActiveForecastMarkers)
  // =========================================================================

  /**
   * Push forecast markers as HTML elements on the globe surface.
   * Each marker is a small colored dot sized by probability.
   */
  private flushMarkers(): void {
    if (!this.globe) return;

    if (this.layerVisible.ActiveForecastMarkers && this.markers.length > 0) {
      this.globe.htmlElementsData(this.markers);
    } else {
      this.globe.htmlElementsData([]);
    }
  }

  /**
   * Build a DOM element for a forecast marker on the globe surface.
   */
  private buildMarkerElement(marker: MarkerDatum): HTMLElement {
    const size = Math.round(8 + marker.probability * 16);
    const intensity = Math.round(100 + marker.probability * 155);
    const color = `rgb(${intensity},40,40)`;

    const el = document.createElement('div');
    el.style.cssText = `
      width: ${size}px;
      height: ${size}px;
      border-radius: 50%;
      background: ${color};
      border: 1px solid rgba(255,255,255,0.3);
      cursor: pointer;
      pointer-events: auto;
      transform: translate(-50%, -50%);
    `;
    el.title = `${marker.iso}: ${marker.question.slice(0, 60)}... (${(marker.probability * 100).toFixed(0)}%)`;

    // Click on marker dispatches country-selected
    el.addEventListener('click', (e) => {
      e.stopPropagation();
      window.dispatchEvent(
        new CustomEvent('country-selected', {
          detail: { iso: marker.iso },
          bubbles: true,
        }),
      );
    });

    return el;
  }

  // =========================================================================
  // Arc flush (Layer 3: KnowledgeGraphArcs)
  // =========================================================================

  /**
   * Push arcs to globe.gl. Two modes:
   *   a) Country selected: per-country scenario arcs
   *   b) No country: global bilateral arcs
   */
  private flushArcs(): void {
    if (!this.globe) return;

    if (!this.layerVisible.KnowledgeGraphArcs) {
      (this.globe as any).arcsData([]);
      return;
    }

    if (this.selectedCountry && this.arcs.length > 0) {
      // Mode A: per-country arcs
      (this.globe as any).arcsData(this.arcs);
    } else if (this.bilateralArcs.length > 0) {
      // Mode B: global bilateral arcs with sentiment
      const arcData = this.bilateralArcs.map((d) => ({
        startLat: d.source[1],
        startLng: d.source[0],
        endLat: d.target[1],
        endLng: d.target[0],
        avgGoldstein: d.avgGoldstein,
        eventCount: d.eventCount,
        sourceIso: d.sourceIso,
        targetIso: d.targetIso,
      }));
      (this.globe as any).arcsData(arcData);
    } else {
      (this.globe as any).arcsData([]);
    }
  }

  // =========================================================================
  // Point flush (Layer 4: GDELTEventHeatmap as colored dots)
  // =========================================================================

  /**
   * Push H3 hex bin centers as colored point markers on the globe surface.
   * Globe.gl has no native H3 hexagon layer -- we degrade to point markers.
   * Uses pointsData channel with merged geometry for performance.
   *
   * h3-js is dynamically imported and cached to avoid top-level import.
   * The first call with hex data triggers the import; subsequent calls use cache.
   */
  // Cached h3-js module (loaded on first heatmap flush)
  private h3Module: { cellToLatLng: (h3Index: string) => [number, number] } | null = null;
  private h3Loading = false;

  private flushPoints(): void {
    if (!this.globe) return;

    if (!this.layerVisible.GDELTEventHeatmap || this.hexBinData.length === 0) {
      (this.globe as any).pointsData([]);
      return;
    }

    // If h3-js not yet loaded, trigger async load and re-flush when ready
    if (!this.h3Module) {
      if (!this.h3Loading) {
        this.h3Loading = true;
        import('h3-js').then((mod) => {
          this.h3Module = { cellToLatLng: mod.cellToLatLng };
          this.h3Loading = false;
          this.scheduleFlush(); // Re-flush via debounced path (not standalone)
        }).catch(() => {
          this.h3Loading = false;
          console.warn('[GlobeMap] Failed to load h3-js for heatmap points');
        });
      }
      return;
    }

    const maxWeight = this.hexBinData.reduce(
      (max, d) => Math.max(max, d.weight), 1,
    );

    const points: Array<{ lat: number; lng: number; color: string; radius: number }> = [];

    for (const hex of this.hexBinData) {
      try {
        const [lat, lng] = this.h3Module.cellToLatLng(hex.h3_index);

        // Weight-based color: yellow [255,255,0] -> red [255,0,0]
        const t = Math.min(1, hex.weight / maxWeight);
        const g = Math.round(255 * (1 - t));
        const a = (100 + 120 * t) / 255;
        const color = `rgba(255,${g},0,${a.toFixed(2)})`;
        const radius = 0.2 + t * 0.3;

        points.push({ lat, lng, color, radius });
      } catch {
        // Invalid H3 index -- skip
        continue;
      }
    }

    (this.globe as any).pointsData(points);
  }

  // =========================================================================
  // Arc generation (per-country scenario arcs)
  // =========================================================================

  /**
   * Build arcs from the selected country to other countries that appear
   * in active forecast scenario entities. Mirrors DeckGLMap lines 849-885.
   */
  private buildArcsForCountry(): void {
    this.arcs = [];
    if (!this.selectedCountry) return;

    const sourceCentroid = countryGeometry.getCentroid(this.selectedCountry);
    if (!sourceCentroid) return;

    const targetIsos = new Set<string>();
    for (const marker of this.markers) {
      if (marker.iso === this.selectedCountry) continue;
      targetIsos.add(marker.iso);
    }

    for (const iso of this.scenarioIsos) {
      if (iso !== this.selectedCountry) {
        targetIsos.add(iso);
      }
    }

    for (const targetIso of targetIsos) {
      const targetCentroid = countryGeometry.getCentroid(targetIso);
      if (!targetCentroid) continue;

      this.arcs.push({
        sourceIso: this.selectedCountry,
        targetIso,
        startLat: sourceCentroid[1],
        startLng: sourceCentroid[0],
        endLat: targetCentroid[1],
        endLng: targetCentroid[0],
      });
    }
  }

}
