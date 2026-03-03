/**
 * DeckGLMap -- WebGL globe with 5 analytic layers for the Geopol dashboard.
 *
 * Layers:
 *   1. ForecastRiskChoropleth  (GeoJsonLayer)   -- country fill by risk_score
 *   2. ActiveForecastMarkers   (ScatterplotLayer)-- centroids of forecast targets
 *   3. KnowledgeGraphArcs      (ArcLayer)        -- actor-to-actor on selection
 *   4. GDELTEventHeatmap       (HeatmapLayer)    -- event density
 *   5. ScenarioZones           (GeoJsonLayer)     -- scenario-relevant countries
 *
 * NOT a Panel subclass. Standalone map component that renders into a given
 * container element and exposes a public data-push API for screen wiring.
 *
 * Public API for external layer control:
 *   - flyToCountry(iso)       -- animate camera to country centroid
 *   - setLayerVisible(id, v)  -- toggle individual layer visibility
 *   - setLayerDefaults(map)   -- batch-set layer visibility (e.g. globe screen)
 *   - getLayerVisible(id)     -- query layer visibility state
 *   - getMap()                -- access underlying maplibre-gl Map instance
 *
 * Built-in toggle panel removed -- external LayerPillBar (Plan 02) manages
 * layer visibility via setLayerVisible()/getLayerVisible() public API.
 */

import { MapboxOverlay } from '@deck.gl/mapbox';
import type { PickingInfo } from '@deck.gl/core';
import { GeoJsonLayer, ScatterplotLayer, ArcLayer } from '@deck.gl/layers';
import { HeatmapLayer } from '@deck.gl/aggregation-layers';
import maplibregl from 'maplibre-gl';
import type { Feature, Geometry, FeatureCollection } from 'geojson';
import { countryGeometry, normalizeCode } from '@/services/country-geometry.ts';

import { h } from '@/utils/dom-utils.ts';
import type { ForecastResponse, CountryRiskSummary } from '@/types/api.ts';

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const DARK_STYLE = 'https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json';
// Light style removed -- dark-only theme

/** Layer IDs for toggle state tracking. Exported for external consumers. */
export const LAYER_IDS = [
  'ForecastRiskChoropleth',
  'ActiveForecastMarkers',
  'KnowledgeGraphArcs',
  'GDELTEventHeatmap',
  'ScenarioZones',
] as const;

export type LayerId = (typeof LAYER_IDS)[number];

// ---------------------------------------------------------------------------
// Color utilities
// ---------------------------------------------------------------------------

type RGBA = [number, number, number, number];

/** Blue-to-gray-to-red diverging color scale for risk scores [0, 1]. */
function riskColor(score: number, alpha = 180): RGBA {
  const t = Math.max(0, Math.min(1, score));
  if (t <= 0.5) {
    // Blue [70,130,180] -> Gray [128,128,128]
    const u = t * 2; // 0..1
    return [
      Math.round(70 + (128 - 70) * u),
      Math.round(130 + (128 - 130) * u),
      Math.round(180 + (128 - 180) * u),
      alpha,
    ];
  }
  // Gray [128,128,128] -> Red [220,50,50]
  const u = (t - 0.5) * 2; // 0..1
  return [
    Math.round(128 + (220 - 128) * u),
    Math.round(128 + (50 - 128) * u),
    Math.round(128 + (50 - 128) * u),
    alpha,
  ];
}

/** Accent color for scenario zone highlights (dark theme). */
function accentColor(): RGBA {
  return [0, 212, 255, 60];
}

/** Default country fill (no risk data) -- dark theme. */
function defaultFill(): RGBA {
  return [40, 44, 52, 120];
}

/** Country stroke -- dark theme. */
function countryStroke(): RGBA {
  return [80, 85, 95, 160];
}

// ---------------------------------------------------------------------------
// Data interfaces
// ---------------------------------------------------------------------------

interface MarkerDatum {
  iso: string;
  position: [number, number];
  probability: number;
  question: string;
}

interface ArcDatum {
  sourceIso: string;
  targetIso: string;
  source: [number, number];
  target: [number, number];
}

interface HeatDatum {
  position: [number, number];
  weight: number;
}

// ---------------------------------------------------------------------------
// DeckGLMap
// ---------------------------------------------------------------------------

export class DeckGLMap {
  private readonly container: HTMLElement;
  private map: maplibregl.Map | null = null;
  private overlay: MapboxOverlay | null = null;
  private wrapper: HTMLElement | null = null;
  private tooltip: HTMLElement | null = null;

  // Layer toggle state -- defaults all true (dashboard shows everything).
  // Globe screen calls setLayerDefaults() after construction to override.
  private readonly layerVisible: Record<LayerId, boolean> = {
    ForecastRiskChoropleth: true,
    ActiveForecastMarkers: true,
    KnowledgeGraphArcs: true,
    GDELTEventHeatmap: true,
    ScenarioZones: true,
  };

  // Data stores
  private riskScores = new Map<string, number>();
  private riskTimestamp = 0;
  private markers: MarkerDatum[] = [];
  private arcs: ArcDatum[] = [];
  private heatData: HeatDatum[] = [];
  private scenarioIsos = new Set<string>();

  // Selection state
  private selectedCountry: string | null = null;

  // Event listener references for cleanup
  private readonly onThemeChanged: (e: Event) => void;

  constructor(container: HTMLElement) {
    this.container = container;

    this.onThemeChanged = () => {
      // Dark-only: no basemap switch needed, but rebuild layers
      // in case any color derivations change.
      this.rebuildLayers();
    };

    this.setupDOM();
    this.initMap();

    window.addEventListener('theme-changed', this.onThemeChanged);
  }

  // =========================================================================
  // Public API
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
    this.riskTimestamp = Date.now();
    this.rebuildLayers();
  }

  /**
   * Push forecast data. Updates scatter markers + scenario zones.
   */
  updateForecasts(forecasts: ForecastResponse[]): void {
    this.markers = [];
    for (const f of forecasts) {
      // Extract country ISO from calibration category (convention from backend)
      const iso = f.calibration.category.length === 2
        ? f.calibration.category.toUpperCase()
        : '';
      if (!iso) continue;

      const centroid = countryGeometry.getCentroid(iso);
      if (!centroid) continue;

      this.markers.push({
        iso,
        position: centroid,
        probability: f.probability,
        question: f.question,
      });
    }
    this.rebuildLayers();
  }

  /**
   * Select a country. Shows KnowledgeGraphArcs for that country's forecasts.
   * Pass null to deselect.
   */
  setSelectedCountry(iso: string | null): void {
    this.selectedCountry = iso ? iso.toUpperCase() : null;
    this.buildArcsForCountry();
    this.rebuildLayers();
  }

  /**
   * Select a forecast. Highlights scenario-relevant countries.
   * Pass null to deselect.
   */
  setSelectedForecast(forecast: ForecastResponse | null): void {
    this.scenarioIsos.clear();

    if (forecast) {
      // Collect all entity ISOs from the scenario tree
      const collectEntities = (scenarios: ForecastResponse['scenarios']): void => {
        for (const s of scenarios) {
          for (const entity of s.entities) {
            // Entities might be ISO codes or names; try both
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

    this.rebuildLayers();
  }

  /**
   * Animate camera to the centroid of the given country.
   * Uses maplibre-gl flyTo with Natural Earth LABEL_X/LABEL_Y coordinates.
   * @param iso ISO 3166-1 Alpha-2 code (case-insensitive)
   */
  public flyToCountry(iso: string): void {
    if (!this.map) return;
    const centroid = countryGeometry.getCentroid(iso.toUpperCase());
    if (!centroid) return;
    this.map.flyTo({
      center: [centroid[0], centroid[1]],
      zoom: 4.5,
      duration: 800,
      essential: true,
    });
  }

  /**
   * Toggle visibility of an individual layer. Triggers layer rebuild.
   * @param layerId One of the 5 analytic layer IDs
   * @param visible Whether the layer should be rendered
   */
  public setLayerVisible(layerId: LayerId, visible: boolean): void {
    this.layerVisible[layerId] = visible;
    this.rebuildLayers();
  }

  /**
   * Batch-set layer visibility defaults. Useful for screens that want
   * different initial layer states than the default (all true).
   * Only modifies keys present in the defaults map; others remain unchanged.
   * @param defaults Partial map of LayerId -> visibility
   */
  public setLayerDefaults(defaults: Partial<Record<LayerId, boolean>>): void {
    for (const [id, visible] of Object.entries(defaults)) {
      if (id in this.layerVisible) {
        this.layerVisible[id as LayerId] = visible as boolean;
      }
    }
    this.rebuildLayers();
  }

  /**
   * Query the current visibility state of a layer.
   * @param layerId One of the 5 analytic layer IDs
   */
  public getLayerVisible(layerId: LayerId): boolean {
    return this.layerVisible[layerId];
  }

  /**
   * Access the underlying maplibre-gl Map instance.
   * Returns null if the map hasn't initialized yet or has been destroyed.
   */
  public getMap(): maplibregl.Map | null {
    return this.map;
  }

  /** Clean up map + overlay + event listeners. */
  destroy(): void {
    window.removeEventListener('theme-changed', this.onThemeChanged);

    if (this.overlay && this.map) {
      this.map.removeControl(this.overlay as unknown as maplibregl.IControl);
    }
    this.overlay = null;

    if (this.map) {
      this.map.remove();
      this.map = null;
    }

    if (this.wrapper) {
      this.wrapper.remove();
      this.wrapper = null;
    }

    this.tooltip = null;
  }

  // =========================================================================
  // DOM setup
  // =========================================================================

  private setupDOM(): void {
    this.wrapper = h('div', {
      className: 'deckgl-map-wrapper',
      style: 'position: relative; width: 100%; height: 100%; overflow: hidden;',
    });

    const mapContainer = h('div', {
      className: 'deckgl-basemap',
      style: 'position: absolute; top: 0; left: 0; width: 100%; height: 100%;',
    });
    this.wrapper.appendChild(mapContainer);

    // Tooltip element (positioned absolutely, hidden by default)
    this.tooltip = h('div', {
      className: 'map-tooltip',
      style: 'display: none;',
    });
    this.wrapper.appendChild(this.tooltip);

    // Attribution
    const attribution = h('div', {
      className: 'map-attribution',
    });
    attribution.innerHTML =
      '&copy; <a href="https://carto.com/attributions" target="_blank" rel="noopener">CARTO</a> ' +
      '&copy; <a href="https://www.openstreetmap.org/copyright" target="_blank" rel="noopener">OpenStreetMap</a>';
    this.wrapper.appendChild(attribution);

    this.container.appendChild(this.wrapper);
  }

  // =========================================================================
  // MapLibre + deck.gl initialization
  // =========================================================================

  private initMap(): void {
    const basemapContainer = this.wrapper?.querySelector('.deckgl-basemap') as HTMLElement | null;
    if (!basemapContainer) return;

    this.map = new maplibregl.Map({
      container: basemapContainer,
      style: DARK_STYLE,
      center: [30, 20],
      zoom: 1.8,
      renderWorldCopies: false,
      attributionControl: false,
    });

    this.map.on('load', () => {
      this.initDeckOverlay();
    });

    // WebGL context loss/restore
    const canvas = this.map.getCanvas();
    canvas.addEventListener('webglcontextlost', (e) => {
      e.preventDefault();
      console.warn('[DeckGLMap] WebGL context lost');
    });
    canvas.addEventListener('webglcontextrestored', () => {
      console.info('[DeckGLMap] WebGL context restored');
      this.map?.triggerRepaint();
    });
  }

  private initDeckOverlay(): void {
    if (!this.map) return;

    this.overlay = new MapboxOverlay({
      interleaved: true,
      layers: this.buildLayers(),
      getTooltip: (info: PickingInfo) => this.handleTooltip(info),
      onClick: (info: PickingInfo) => this.handleClick(info),
      pickingRadius: 10,
    });

    this.map.addControl(this.overlay as unknown as maplibregl.IControl);
  }

  // =========================================================================
  // Layer construction
  // =========================================================================

  private rebuildLayers(): void {
    if (!this.overlay) return;
    try {
      this.overlay.setProps({ layers: this.buildLayers() });
    } catch {
      // Map may be mid-teardown
    }
  }

  private buildLayers(): (GeoJsonLayer | ScatterplotLayer | ArcLayer | HeatmapLayer)[] {
    const layers: (GeoJsonLayer | ScatterplotLayer | ArcLayer | HeatmapLayer)[] = [];
    const geoJson = countryGeometry.getFeatureCollection();

    // Layer 1: ForecastRiskChoropleth
    if (this.layerVisible.ForecastRiskChoropleth && geoJson) {
      const riskMap = this.riskScores;
      const ts = this.riskTimestamp;
      const defFill = defaultFill();
      const stroke = countryStroke();

      layers.push(
        new GeoJsonLayer({
          id: 'ForecastRiskChoropleth',
          data: geoJson,
          pickable: true,
          stroked: true,
          filled: true,
          getFillColor: (f: Feature<Geometry>) => {
            const code = normalizeCode(f.properties);
            if (!code) return defFill;
            const score = riskMap.get(code);
            if (score === undefined) return defFill;
            return riskColor(score);
          },
          getLineColor: stroke,
          getLineWidth: 1,
          lineWidthMinPixels: 0.5,
          updateTriggers: {
            getFillColor: [riskMap.size, ts],
          },
        }),
      );
    }

    // Layer 2: ActiveForecastMarkers
    if (this.layerVisible.ActiveForecastMarkers && this.markers.length > 0) {
      layers.push(
        new ScatterplotLayer<MarkerDatum>({
          id: 'ActiveForecastMarkers',
          data: this.markers,
          pickable: true,
          getPosition: (d) => d.position,
          getRadius: (d) => 8 + d.probability * 20,
          getFillColor: (d) => {
            const intensity = Math.round(100 + d.probability * 155);
            return [intensity, 40, 40, 200];
          },
          radiusUnits: 'pixels' as const,
          radiusMinPixels: 4,
          radiusMaxPixels: 30,
          updateTriggers: {
            getPosition: [this.markers.length],
            getRadius: [this.markers.length],
            getFillColor: [this.markers.length],
          },
        }),
      );
    }

    // Layer 3: KnowledgeGraphArcs
    if (this.layerVisible.KnowledgeGraphArcs && this.arcs.length > 0 && this.selectedCountry) {
      layers.push(
        new ArcLayer<ArcDatum>({
          id: 'KnowledgeGraphArcs',
          data: this.arcs,
          pickable: false,
          getSourcePosition: (d) => d.source,
          getTargetPosition: (d) => d.target,
          getSourceColor: [0, 200, 255, 180],
          getTargetColor: [255, 100, 100, 180],
          getWidth: 2,
          greatCircle: true,
        }),
      );
    }

    // Layer 4: GDELTEventHeatmap
    if (this.layerVisible.GDELTEventHeatmap && this.heatData.length > 0) {
      layers.push(
        new HeatmapLayer<HeatDatum>({
          id: 'GDELTEventHeatmap',
          data: this.heatData,
          getPosition: (d) => d.position,
          getWeight: (d) => d.weight,
          radiusPixels: 30,
          intensity: 1,
          threshold: 0.03,
        }),
      );
    }

    // Layer 5: ScenarioZones
    if (this.layerVisible.ScenarioZones && this.scenarioIsos.size > 0 && geoJson) {
      const scenarioFeatures: Feature<Geometry>[] = geoJson.features.filter((f) => {
        const code = normalizeCode(f.properties);
        return code !== null && this.scenarioIsos.has(code);
      });

      if (scenarioFeatures.length > 0) {
        const fc: FeatureCollection<Geometry> = {
          type: 'FeatureCollection',
          features: scenarioFeatures,
        };
        const accent = accentColor();

        layers.push(
          new GeoJsonLayer({
            id: 'ScenarioZones',
            data: fc,
            pickable: false,
            stroked: true,
            filled: true,
            getFillColor: accent,
            getLineColor: [accent[0], accent[1], accent[2], 160],
            getLineWidth: 2,
            lineWidthMinPixels: 1,
            updateTriggers: {
              getFillColor: [this.scenarioIsos.size],
            },
          }),
        );
      }
    }

    return layers;
  }

  // =========================================================================
  // Event handling
  // =========================================================================

  private handleClick(info: PickingInfo): void {
    if (!info.object || !info.layer) return;

    if (info.layer.id === 'ForecastRiskChoropleth') {
      const feature = info.object as Feature<Geometry>;
      const code = normalizeCode(feature.properties);
      if (!code) return;

      this.selectedCountry = code;

      // Dispatch country-selected CustomEvent
      window.dispatchEvent(
        new CustomEvent('country-selected', {
          detail: { iso: code },
          bubbles: true,
        }),
      );

      this.buildArcsForCountry();
      this.rebuildLayers();
    }

    if (info.layer.id === 'ActiveForecastMarkers') {
      const marker = info.object as MarkerDatum;
      window.dispatchEvent(
        new CustomEvent('country-selected', {
          detail: { iso: marker.iso },
          bubbles: true,
        }),
      );
    }
  }

  private handleTooltip(info: PickingInfo): string | null {
    if (!this.tooltip) return null;

    if (!info.object || !info.layer) {
      this.tooltip.style.display = 'none';
      return null;
    }

    let content = '';

    if (info.layer.id === 'ForecastRiskChoropleth') {
      const feature = info.object as Feature<Geometry>;
      const code = normalizeCode(feature.properties);
      const name = countryGeometry.getNameByIso(code ?? '') ?? code ?? 'Unknown';
      const score = code ? this.riskScores.get(code) : undefined;
      const scoreStr = score !== undefined ? (score * 100).toFixed(1) + '%' : 'N/A';
      content = `<strong>${name}</strong><br/>Risk: ${scoreStr}`;
    } else if (info.layer.id === 'ActiveForecastMarkers') {
      const marker = info.object as MarkerDatum;
      const name = countryGeometry.getNameByIso(marker.iso) ?? marker.iso;
      const pct = (marker.probability * 100).toFixed(1);
      const q = marker.question.length > 80
        ? marker.question.slice(0, 77) + '...'
        : marker.question;
      content = `<strong>${name}</strong><br/>${q}<br/>P: ${pct}%`;
    }

    if (content && info.x !== undefined && info.y !== undefined) {
      this.tooltip.innerHTML = content;
      this.tooltip.style.display = 'block';
      this.tooltip.style.left = `${info.x + 12}px`;
      this.tooltip.style.top = `${info.y - 12}px`;
    } else {
      this.tooltip.style.display = 'none';
    }

    // Return null -- we handle tooltip DOM ourselves
    return null;
  }

  // =========================================================================
  // Arc generation
  // =========================================================================

  /**
   * Build arcs from the selected country to other countries that appear
   * in active forecast scenario entities. If no country is selected, clears arcs.
   */
  private buildArcsForCountry(): void {
    this.arcs = [];
    if (!this.selectedCountry) return;

    const sourceCentroid = countryGeometry.getCentroid(this.selectedCountry);
    if (!sourceCentroid) return;

    // Collect target ISOs from markers that match the selected country's forecasts
    const targetIsos = new Set<string>();
    for (const marker of this.markers) {
      if (marker.iso === this.selectedCountry) {
        // This country has a forecast; any scenario entities become arc targets
        // For now, use all other markers as connected nodes
        continue;
      }
      targetIsos.add(marker.iso);
    }

    // Also add scenario entity ISOs
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
        targetIso: targetIso,
        source: sourceCentroid,
        target: targetCentroid,
      });
    }
  }
}
