/**
 * DeckGLMap -- WebGL globe with 5 analytic layers for the Geopol dashboard.
 *
 * Layers:
 *   1. ForecastRiskChoropleth  (GeoJsonLayer)      -- country fill by risk_score
 *   2. ActiveForecastMarkers   (ScatterplotLayer)   -- centroids of forecast targets
 *   3. KnowledgeGraphArcs      (ArcLayer)           -- bilateral arcs (global or per-country)
 *   4. GDELTEventHeatmap       (H3HexagonLayer)     -- H3 hex event density
 *   5. ScenarioZones           (GeoJsonLayer)        -- risk delta regions or scenario highlights
 *
 * NOT a Panel subclass. Standalone map component that renders into a given
 * container element and exposes a public data-push API for screen wiring.
 *
 * Public API for external layer control:
 *   - flyToCountry(iso)           -- animate camera to country centroid
 *   - setLayerVisible(id, v)      -- toggle individual layer visibility
 *   - setLayerDefaults(map)       -- batch-set layer visibility (e.g. globe screen)
 *   - getLayerVisible(id)         -- query layer visibility state
 *   - getMap()                    -- access underlying maplibre-gl Map instance
 *   - updateHeatmapData(data)     -- push H3 hexbin data for heatmap layer
 *   - updateArcData(data)         -- push bilateral arc data for arcs layer
 *   - updateRiskDeltas(deltas)    -- push risk delta data for scenario zones layer
 *
 * Built-in toggle panel removed -- external LayerPillBar (Plan 02) manages
 * layer visibility via setLayerVisible()/getLayerVisible() public API.
 */

import { MapboxOverlay } from '@deck.gl/mapbox';
import type { PickingInfo, Layer } from '@deck.gl/core';
import { GeoJsonLayer, ScatterplotLayer, ArcLayer } from '@deck.gl/layers';
import { H3HexagonLayer } from '@deck.gl/geo-layers';
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
  source: [number, number];
  target: [number, number];
  eventCount: number;
  avgGoldstein: number;
}

/** Risk delta datum for the scenarios/risk-change layer. */
export interface RiskDeltaDatum {
  iso: string;
  delta: number;  // positive = worsening, negative = improving
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

  // Data stores -- choropleth + markers
  private riskScores = new Map<string, number>();
  private riskTimestamp = 0;
  private markers: MarkerDatum[] = [];
  private arcs: ArcDatum[] = [];
  private scenarioIsos = new Set<string>();

  // Data stores -- globe layer data (Phase 24)
  private hexBinData: HexBinDatum[] = [];
  private bilateralArcs: BilateralArcDatum[] = [];
  private riskDeltaIsos = new Map<string, number>(); // iso -> delta

  // Selection state
  private selectedCountry: string | null = null;
  private selectedForecast: ForecastResponse | null = null;

  // Event listener references for cleanup
  private readonly onThemeChanged: (e: Event) => void;
  private resizeObserver: ResizeObserver | null = null;

  constructor(container: HTMLElement) {
    this.container = container;

    this.onThemeChanged = () => {
      // Dark-only: no basemap switch needed, but rebuild layers
      // in case any color derivations change.
      this.rebuildLayers();
    };

    this.setupDOM();
    this.initMap();

    // ResizeObserver: recalculate map viewport when container changes size
    // (flex layout may settle after construction, and window resizes need handling)
    this.resizeObserver = new ResizeObserver(() => {
      this.map?.resize();
    });
    this.resizeObserver.observe(this.container);

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
    this.selectedForecast = forecast;

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
   * Push H3 hexbin data for the heatmap layer.
   * Replaces any existing heatmap data and rebuilds layers.
   */
  updateHeatmapData(data: HexBinDatum[]): void {
    this.hexBinData = data;
    this.rebuildLayers();
  }

  /**
   * Push bilateral arc data for the arcs layer (global view).
   * When no country is selected, these arcs render as the KnowledgeGraphArcs layer.
   */
  updateArcData(data: BilateralArcDatum[]): void {
    this.bilateralArcs = data;
    this.rebuildLayers();
  }

  /**
   * Push risk delta data for the ScenarioZones layer.
   * When no forecast is selected, renders countries with significant risk changes
   * (red = worsening, green = improving).
   */
  updateRiskDeltas(deltas: RiskDeltaDatum[]): void {
    this.riskDeltaIsos.clear();
    for (const d of deltas) {
      this.riskDeltaIsos.set(d.iso.toUpperCase(), d.delta);
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
    this.resizeObserver?.disconnect();
    this.resizeObserver = null;

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
      center: [20, 35],
      zoom: 1.6,
      renderWorldCopies: false,
      attributionControl: false,
    });

    this.map.on('load', () => {
      this.initDeckOverlay();
      // Force MapLibre to recalculate viewport dimensions.
      // Container may not have had final computed dimensions at construction
      // time (e.g., absolutely-positioned parent not yet laid out).
      this.map!.resize();
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

    // Safety net: if data was pushed (updateRiskScores/updateForecasts) or
    // layer defaults were set before the map fired 'load', those rebuildLayers()
    // calls were no-ops (overlay was null). Rebuild now to apply pending state.
    this.rebuildLayers();
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

  private buildLayers(): Layer[] {
    const layers: Layer[] = [];
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
            return riskColor(score / 100);
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
    // Two modes:
    //   a) Country selected: show arcs from selected country to scenario entities
    //   b) No country selected + bilateral data: show global bilateral arcs
    if (this.layerVisible.KnowledgeGraphArcs) {
      if (this.selectedCountry && this.arcs.length > 0) {
        // Mode A: per-country scenario arcs (existing behavior)
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
      } else if (this.bilateralArcs.length > 0) {
        // Mode B: global bilateral arcs with sentiment coloring
        layers.push(
          new ArcLayer<BilateralArcDatum>({
            id: 'KnowledgeGraphArcs',
            data: this.bilateralArcs,
            pickable: true,
            getSourcePosition: (d) => d.source,
            getTargetPosition: (d) => d.target,
            getSourceColor: (d) => d.avgGoldstein < 0
              ? [255, 80, 80, 180] as RGBA    // conflictual (negative Goldstein)
              : [80, 180, 255, 180] as RGBA,   // cooperative (positive Goldstein)
            getTargetColor: (d) => d.avgGoldstein < 0
              ? [255, 80, 80, 180] as RGBA
              : [80, 180, 255, 180] as RGBA,
            getWidth: (d) => Math.max(1, Math.min(5, d.eventCount / 20)),
            greatCircle: true,
            updateTriggers: {
              getSourceColor: [this.bilateralArcs.length],
              getTargetColor: [this.bilateralArcs.length],
              getWidth: [this.bilateralArcs.length],
            },
          }),
        );
      }
    }

    // Layer 4: GDELTEventHeatmap (H3 hexagonal bins)
    if (this.layerVisible.GDELTEventHeatmap && this.hexBinData.length > 0) {
      const maxWeight = this.hexBinData.reduce(
        (max, d) => Math.max(max, d.weight), 1,
      );
      // Capture in local for the accessor closure (avoids `this` reference
      // inside deck.gl accessor which may be called with different `this`)
      const hexData = this.hexBinData;
      const mw = maxWeight;

      layers.push(
        new H3HexagonLayer<HexBinDatum>({
          id: 'GDELTEventHeatmap',
          data: hexData,
          pickable: true,
          getHexagon: (d: HexBinDatum) => d.h3_index,
          getFillColor: (d: HexBinDatum) => {
            // Weight-based intensity: yellow [255,255,0] -> red [255,0,0]
            const t = Math.min(1, d.weight / mw);
            return [
              255,
              Math.round(255 * (1 - t)),
              0,
              Math.round(100 + 120 * t),
            ] as RGBA;
          },
          getElevation: (d: HexBinDatum) => d.weight,
          elevationScale: 1000,
          extruded: false,
          coverage: 0.9,
          updateTriggers: {
            getFillColor: [hexData.length, mw],
          },
        }),
      );
    }

    // Layer 5: ScenarioZones / Risk Delta Regions
    // Two modes:
    //   a) Forecast selected: highlight scenario-relevant countries (accent color)
    //   b) No forecast selected + risk deltas: show risk change zones (red/green)
    if (this.layerVisible.ScenarioZones && geoJson) {
      if (this.selectedForecast && this.scenarioIsos.size > 0) {
        // Mode A: scenario entity highlights (existing behavior)
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
      } else if (this.riskDeltaIsos.size > 0) {
        // Mode B: risk delta visualization (Phase 24)
        const deltaMap = this.riskDeltaIsos;
        const deltaFeatures: Feature<Geometry>[] = geoJson.features.filter((f) => {
          const code = normalizeCode(f.properties);
          return code !== null && deltaMap.has(code);
        });

        if (deltaFeatures.length > 0) {
          const fc: FeatureCollection<Geometry> = {
            type: 'FeatureCollection',
            features: deltaFeatures,
          };

          layers.push(
            new GeoJsonLayer({
              id: 'ScenarioZones',
              data: fc,
              pickable: true,
              stroked: true,
              filled: true,
              getFillColor: (f: Feature<Geometry>) => {
                const code = normalizeCode(f.properties);
                const delta = code ? deltaMap.get(code) ?? 0 : 0;
                // Alpha proportional to |delta| / 50, capped at [60, 160]
                const alpha = Math.round(60 + Math.min(100, (Math.abs(delta) / 50) * 100));
                if (delta > 0) {
                  // Risk worsening: red fill
                  return [220, 50, 50, alpha] as RGBA;
                }
                // Risk improving: green fill
                return [50, 180, 80, alpha] as RGBA;
              },
              getLineColor: (f: Feature<Geometry>) => {
                const code = normalizeCode(f.properties);
                const delta = code ? deltaMap.get(code) ?? 0 : 0;
                return delta > 0
                  ? [220, 50, 50, 120] as RGBA
                  : [50, 180, 80, 120] as RGBA;
              },
              getLineWidth: 1,
              lineWidthMinPixels: 0.5,
              updateTriggers: {
                getFillColor: [deltaMap.size],
                getLineColor: [deltaMap.size],
              },
            }),
          );
        }
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
      const scoreStr = score !== undefined ? score.toFixed(1) + '%' : 'N/A';
      content = `<strong>${name}</strong><br/>Risk: ${scoreStr}`;
    } else if (info.layer.id === 'ActiveForecastMarkers') {
      const marker = info.object as MarkerDatum;
      const name = countryGeometry.getNameByIso(marker.iso) ?? marker.iso;
      const pct = (marker.probability * 100).toFixed(1);
      const q = marker.question.length > 80
        ? marker.question.slice(0, 77) + '...'
        : marker.question;
      content = `<strong>${name}</strong><br/>${q}<br/>P: ${pct}%`;
    } else if (info.layer.id === 'GDELTEventHeatmap') {
      const hex = info.object as HexBinDatum;
      content = `Events: ${hex.event_count}<br/>Weight: ${hex.weight.toFixed(1)}`;
    } else if (info.layer.id === 'KnowledgeGraphArcs' && !this.selectedCountry) {
      // Bilateral arc tooltip
      const arc = info.object as BilateralArcDatum;
      const srcName = countryGeometry.getNameByIso(arc.sourceIso) ?? arc.sourceIso;
      const tgtName = countryGeometry.getNameByIso(arc.targetIso) ?? arc.targetIso;
      const sentiment = arc.avgGoldstein < 0 ? 'Conflictual' : 'Cooperative';
      content = `<strong>${srcName} &harr; ${tgtName}</strong><br/>${sentiment} (${arc.avgGoldstein.toFixed(1)})<br/>Events: ${arc.eventCount}`;
    } else if (info.layer.id === 'ScenarioZones' && this.riskDeltaIsos.size > 0 && !this.selectedForecast) {
      // Risk delta tooltip
      const feature = info.object as Feature<Geometry>;
      const code = normalizeCode(feature.properties);
      const name = countryGeometry.getNameByIso(code ?? '') ?? code ?? 'Unknown';
      const delta = code ? this.riskDeltaIsos.get(code) ?? 0 : 0;
      const direction = delta > 0 ? 'Worsening' : 'Improving';
      content = `<strong>${name}</strong><br/>${direction}: ${delta > 0 ? '+' : ''}${delta.toFixed(1)} pts`;
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
