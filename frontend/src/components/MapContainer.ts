/**
 * MapContainer -- Dual-renderer wrapper dispatching to GlobeMap (3D) or DeckGLMap (2D).
 *
 * Both WebGL contexts remain alive simultaneously. Toggling swaps CSS `display`
 * (no destroy/recreate), so the switch is instant and data is always current.
 * Data is pushed to BOTH renderers on every update -- no re-fetch on toggle.
 *
 * Layer visibility state is independent per view (3D and 2D each have their own
 * toggle configuration). The active view's state is what LayerPillBar reads/writes.
 *
 * Listens for:
 *   - `globe-view-toggle` CustomEvent (NavBar) -> toggleMode()
 *   - `globe-region-change` CustomEvent (GlobeHud) -> flyToRegion()
 *
 * Dispatches:
 *   - `globe-mode-changed` CustomEvent with { mode: '3d' | '2d' } after toggle
 *
 * Preference persisted in localStorage key 'geopol-globe-mode'.
 */

import { h } from '@/utils/dom-utils';
import type { DeckGLMap } from '@/components/DeckGLMap';
import type { GlobeMap } from '@/components/GlobeMap';
import type { LayerId, HexBinDatum, BilateralArcDatum, RiskDeltaDatum } from '@/components/DeckGLMap';
import type { ForecastResponse, CountryRiskSummary } from '@/types/api';

// Re-export types that globe-screen's push functions need
export type { LayerId, HexBinDatum, BilateralArcDatum, RiskDeltaDatum };

// ---------------------------------------------------------------------------
// Region view presets (imported from GlobeMap at runtime, duplicated here
// for the 2D flyTo approximation so we avoid a static import of GlobeMap)
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Default layer state (all ON -- globe screen overrides via setLayerDefaults)
// ---------------------------------------------------------------------------

const DEFAULT_LAYER_STATE: Record<LayerId, boolean> = {
  ForecastRiskChoropleth: true,
  ActiveForecastMarkers: true,
  KnowledgeGraphArcs: true,
  GDELTEventHeatmap: true,
  ScenarioZones: true,
};

type ViewMode = '3d' | '2d';

const STORAGE_KEY = 'geopol-globe-mode';

// ---------------------------------------------------------------------------
// MapContainer
// ---------------------------------------------------------------------------

export class MapContainer {
  private readonly deckContainer: HTMLElement;
  private readonly globeContainer: HTMLElement;

  private deckMap: DeckGLMap | null = null;
  private globeMap: GlobeMap | null = null;

  private activeMode: ViewMode;

  // Independent layer toggle state per view
  private layerState3d: Record<LayerId, boolean>;
  private layerState2d: Record<LayerId, boolean>;

  // Event handler references for cleanup
  private readonly toggleHandler: () => void;
  private readonly regionHandler: EventListener;

  /**
   * @param container  Parent element to host both renderer sub-containers
   * @param deckMap    Pre-constructed DeckGLMap instance (2D renderer)
   * @param globeMap   Pre-constructed GlobeMap instance (3D renderer)
   */
  constructor(
    container: HTMLElement,
    deckMap: DeckGLMap,
    globeMap: GlobeMap,
  ) {

    // Read persisted preference (default 3D for first-time visitors)
    const pref = localStorage.getItem(STORAGE_KEY) ?? '3d';
    this.activeMode = pref === '2d' ? '2d' : '3d';

    // Independent layer states -- both start with defaults
    this.layerState3d = { ...DEFAULT_LAYER_STATE };
    this.layerState2d = { ...DEFAULT_LAYER_STATE };

    // Create two sub-containers occupying the same space
    this.deckContainer = h('div', {
      className: 'map-container-deck',
      style: 'position:absolute;inset:0;',
    });
    this.globeContainer = h('div', {
      className: 'map-container-globe',
      style: 'position:absolute;inset:0;',
    });
    container.appendChild(this.deckContainer);
    container.appendChild(this.globeContainer);

    // Store pre-constructed renderer instances
    this.deckMap = deckMap;
    this.globeMap = globeMap;

    // Apply initial visibility
    this.applyVisibility();

    // Listen for toggle and region events
    this.toggleHandler = () => this.toggleMode();
    window.addEventListener('globe-view-toggle', this.toggleHandler);

    this.regionHandler = ((e: CustomEvent<{ region: string }>) => {
      this.flyToRegion(e.detail.region);
    }) as EventListener;
    window.addEventListener('globe-region-change', this.regionHandler);
  }

  // =========================================================================
  // View toggle
  // =========================================================================

  /** Toggle between 3D and 2D modes. Persists preference and dispatches event. */
  toggleMode(): void {
    this.activeMode = this.activeMode === '3d' ? '2d' : '3d';
    localStorage.setItem(STORAGE_KEY, this.activeMode);
    this.applyVisibility();
    this.syncLayerVisibility();

    // Notify NavBar (and any other listeners) of the new mode
    window.dispatchEvent(new CustomEvent('globe-mode-changed', {
      detail: { mode: this.activeMode },
    }));
  }

  /** Get the currently active view mode. */
  getActiveMode(): ViewMode {
    return this.activeMode;
  }

  // =========================================================================
  // Data dispatch -- push to BOTH renderers (instant toggle, no re-fetch)
  // =========================================================================

  updateRiskScores(summaries: CountryRiskSummary[]): void {
    this.deckMap?.updateRiskScores(summaries);
    this.globeMap?.updateRiskScores(summaries);
  }

  updateForecasts(forecasts: ForecastResponse[]): void {
    this.deckMap?.updateForecasts(forecasts);
    this.globeMap?.updateForecasts(forecasts);
  }

  updateHeatmapData(data: HexBinDatum[]): void {
    this.deckMap?.updateHeatmapData(data);
    this.globeMap?.updateHeatmapData(data);
  }

  updateArcData(data: BilateralArcDatum[]): void {
    this.deckMap?.updateArcData(data);
    this.globeMap?.updateArcData(data);
  }

  updateRiskDeltas(deltas: RiskDeltaDatum[]): void {
    this.deckMap?.updateRiskDeltas(deltas);
    this.globeMap?.updateRiskDeltas(deltas);
  }

  // =========================================================================
  // Selection dispatch -- push to BOTH renderers
  // =========================================================================

  setSelectedCountry(iso: string | null): void {
    this.deckMap?.setSelectedCountry(iso);
    this.globeMap?.setSelectedCountry(iso);
  }

  setSelectedForecast(forecast: ForecastResponse | null): void {
    this.deckMap?.setSelectedForecast(forecast);
    this.globeMap?.setSelectedForecast(forecast);
  }

  // =========================================================================
  // Layer visibility -- independent per view, dispatched to ACTIVE renderer
  // =========================================================================

  setLayerVisible(layerId: LayerId, visible: boolean): void {
    if (this.activeMode === '3d') {
      this.layerState3d[layerId] = visible;
      this.globeMap?.setLayerVisible(layerId, visible);
    } else {
      this.layerState2d[layerId] = visible;
      this.deckMap?.setLayerVisible(layerId, visible);
    }
  }

  getLayerVisible(layerId: LayerId): boolean {
    return this.activeMode === '3d'
      ? this.layerState3d[layerId]
      : this.layerState2d[layerId];
  }

  /**
   * Batch-set layer visibility defaults. Applies to BOTH views since
   * defaults are set before user has interacted with either toggle state.
   */
  setLayerDefaults(defaults: Partial<Record<LayerId, boolean>>): void {
    for (const [id, visible] of Object.entries(defaults)) {
      this.layerState3d[id as LayerId] = visible as boolean;
      this.layerState2d[id as LayerId] = visible as boolean;
    }
    this.deckMap?.setLayerDefaults(defaults);
    this.globeMap?.setLayerDefaults(defaults);
  }

  // =========================================================================
  // Camera control
  // =========================================================================

  /** Fly to a country centroid on the active renderer. */
  flyToCountry(iso: string): void {
    if (this.activeMode === '3d') {
      this.globeMap?.flyToCountry(iso);
    } else {
      this.deckMap?.flyToCountry(iso);
    }
  }

  /**
   * Fly to a named region preset. On 3D, calls globe.pointOfView;
   * on 2D, calls maplibre flyTo with an altitude-to-zoom approximation.
   */
  flyToRegion(region: string): void {
    const pov = VIEW_POVS[region];
    if (!pov) return;

    if (this.activeMode === '3d') {
      this.globeMap?.flyToRegion?.(region);
    } else {
      // Approximate globe altitude to maplibre zoom level
      const zoom = pov.altitude < 1.3 ? 4 : pov.altitude < 1.6 ? 3 : 2;
      this.deckMap?.getMap()?.flyTo({
        center: [pov.lng, pov.lat],
        zoom,
        duration: 1200,
      });
    }
  }

  // =========================================================================
  // Map access (2D compatibility)
  // =========================================================================

  /** Access the underlying maplibre-gl Map instance (2D mode only). */
  getMap(): any {
    return this.deckMap?.getMap() ?? null;
  }

  // =========================================================================
  // Cleanup
  // =========================================================================

  destroy(): void {
    window.removeEventListener('globe-view-toggle', this.toggleHandler);
    window.removeEventListener('globe-region-change', this.regionHandler);

    this.deckMap?.destroy();
    this.globeMap?.destroy();
    this.deckMap = null;
    this.globeMap = null;
  }

  // =========================================================================
  // Private helpers
  // =========================================================================

  /** Apply CSS display to show only the active renderer. */
  private applyVisibility(): void {
    this.deckContainer.style.display = this.activeMode === '2d' ? 'block' : 'none';
    this.globeContainer.style.display = this.activeMode === '3d' ? 'block' : 'none';

    // Pause/resume GlobeMap animation to save GPU when hidden
    if (this.activeMode === '2d') {
      this.globeMap?.pauseAnimation();
    } else {
      this.globeMap?.resumeAnimation();
      // Trigger DeckGLMap resize in case container dimensions changed while hidden
    }

    // Trigger map resize after visibility change -- renderer may need to
    // recalculate viewport dimensions after being display:none.
    requestAnimationFrame(() => {
      if (this.activeMode === '2d') {
        this.deckMap?.getMap()?.resize();
      }
    });
  }

  /**
   * After toggle, apply the correct layer state to the now-active renderer.
   * The renderer may have been created with different defaults or not received
   * the latest per-view layer toggles while it was hidden.
   */
  private syncLayerVisibility(): void {
    const state = this.activeMode === '3d' ? this.layerState3d : this.layerState2d;
    const renderer = this.activeMode === '3d' ? this.globeMap : this.deckMap;

    if (renderer) {
      for (const [id, visible] of Object.entries(state)) {
        renderer.setLayerVisible(id as LayerId, visible);
      }
    }
  }
}
