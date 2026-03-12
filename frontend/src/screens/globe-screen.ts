/**
 * Globe screen -- full-viewport CesiumJS globe with contextual overlays.
 *
 * The globe fills the entire screen container. All UI elements are positioned
 * absolutely over the map:
 *   - GlobeHud: top-left stats (forecast count, countries, last update) + region presets
 *   - LayerPillBar: bottom-center toggle bar for 5 analytic layers
 *   - GlobeDrillDown: right-edge slide-in panel on country click
 *
 * CesiumMap provides a single renderer supporting 3D, Columbus View, and 2D scene
 * modes via CesiumJS scene morphing. No dual-renderer dispatch overhead.
 *
 * CesiumJS bundle is NOT loaded on the dashboard route.
 * The dynamic import() ensures code-splitting at the route level.
 *
 * Event wiring:
 *   country-selected  -> flyToCountry + drillDown.open
 *   forecast-selected -> ScenarioExplorer.open (via modal)
 *   country-brief-requested -> CountryBriefPage.open
 *   globe-view-toggle -> CesiumMap.setSceneMode (via CustomEvent)
 *   globe-region-change -> CesiumMap.flyToRegion (via CustomEvent)
 *
 * Refresh scheduling:
 *   countries  -> every 120s (risk scores + HUD + choropleth)
 *   forecasts  -> every 60s  (markers + scatter layer)
 *   layers     -> every 300s (heatmap hexbins + arcs + risk deltas)
 *
 * Fulfills: SCREEN-03, GLOBE-01, GLOBE-02, GLOBE-03.
 */

import { h } from '@/utils/dom-utils';
import { countryGeometry } from '@/services/country-geometry';
import { forecastClient } from '@/services/forecast-client';
import { RefreshScheduler } from '@/app/refresh-scheduler';
import { GlobeHud } from '@/components/GlobeHud';
import { GlobeDrillDown } from '@/components/GlobeDrillDown';
import type { GeoPolAppContext } from '@/app/app-context';
import type { CesiumMap } from '@/components/CesiumMap';
import type { HexBinDatum, BilateralArcDatum, RiskDeltaDatum } from '@/components/CesiumMap';
import type { CountryRiskSummary, ForecastResponse, HexbinData, ArcData, RiskDeltaData } from '@/types/api';

// Lazy imports -- only resolved when needed
import type { ScenarioExplorer } from '@/components/ScenarioExplorer';
import type { CountryBriefPage } from '@/components/CountryBriefPage';
import type { LayerPillBar } from '@/components/LayerPillBar';

// ---------------------------------------------------------------------------
// Module-scoped state (one instance per screen mount)
// ---------------------------------------------------------------------------

let cesiumMap: CesiumMap | null = null;
let hud: GlobeHud | null = null;
let pillBar: LayerPillBar | null = null;
let drillDown: GlobeDrillDown | null = null;
let scenarioExplorer: ScenarioExplorer | null = null;
let countryBriefPage: CountryBriefPage | null = null;
let scheduler: RefreshScheduler | null = null;

// Event handler references for cleanup
let countrySelectedHandler: EventListener | null = null;
let forecastSelectedHandler: EventListener | null = null;
let countryBriefHandler: EventListener | null = null;

// ---------------------------------------------------------------------------
// Mount
// ---------------------------------------------------------------------------

export async function mountGlobe(
  container: HTMLElement,
  ctx: GeoPolAppContext,
): Promise<void> {
  // Full-viewport wrapper -- position: relative for absolutely-positioned overlays
  const wrapper = h('div', { className: 'globe-screen' });

  // Map element fills the entire wrapper (hosts CesiumMap viewer)
  const mapEl = h('div', {
    style: 'position:absolute;inset:0;',
  });
  wrapper.appendChild(mapEl);
  container.appendChild(wrapper);

  try {
    // Dynamic import: CesiumJS chunk only loads when globe screen mounts.
    await countryGeometry.load();
    const [
      { CesiumMap: CesiumMapClass },
      { LayerPillBar: LayerPillBarClass },
      { ScenarioExplorer: ScenarioExplorerClass },
      { CountryBriefPage: CountryBriefPageClass },
    ] = await Promise.all([
      import('@/components/CesiumMap'),
      import('@/components/LayerPillBar'),
      import('@/components/ScenarioExplorer'),
      import('@/components/CountryBriefPage'),
    ]);

    // Construct CesiumMap -- single renderer handles 3D/Columbus/2D scene modes
    cesiumMap = new CesiumMapClass(mapEl);

    // All layers ON by default -- real data exists for all layers now
    cesiumMap.setLayerDefaults({
      ForecastRiskChoropleth: true,
      ActiveForecastMarkers: true,
      KnowledgeGraphArcs: true,
      GDELTEventHeatmap: true,
      ScenarioZones: true,
    });

    // Construct overlay components
    hud = new GlobeHud();
    pillBar = new LayerPillBarClass(cesiumMap);
    drillDown = new GlobeDrillDown();

    // Append overlays to wrapper (order matters for z-index stacking)
    wrapper.appendChild(hud.getElement());
    wrapper.appendChild(pillBar.getElement());
    wrapper.appendChild(drillDown.getElement());

    // Construct modals (ScenarioExplorer auto-registers forecast-selected listener,
    // CountryBriefPage auto-registers country-selected listener -- we override below)
    scenarioExplorer = new ScenarioExplorerClass();

    // CountryBriefPage: disable auto-open on country-selected (that's dashboard behavior).
    // On the globe, we wire it to country-brief-requested via the drill-down sidebar button.
    countryBriefPage = new CountryBriefPageClass({ autoOpen: false });

    // Wire events
    wireEvents(ctx);

    // Initial data load
    await loadInitialData();

    // Refresh scheduler
    scheduler = new RefreshScheduler(ctx);
    scheduler.init();
    ctx.scheduler = scheduler;

    scheduler.registerAll([
      {
        name: 'globe-countries',
        fn: async () => {
          const countries = await forecastClient.getCountries();
          pushCountries(countries);
        },
        intervalMs: 120_000,
      },
      {
        name: 'globe-forecasts',
        fn: async () => {
          const forecasts = await forecastClient.getTopForecasts(50);
          pushForecasts(forecasts);
        },
        intervalMs: 60_000,
      },
      {
        name: 'globe-layers',
        fn: async () => {
          const [heatmap, arcs, deltas] = await Promise.all([
            forecastClient.getHeatmapData(),
            forecastClient.getArcData(),
            forecastClient.getRiskDeltas(),
          ]);
          pushHeatmap(heatmap);
          pushArcs(arcs);
          pushDeltas(deltas);
        },
        intervalMs: 300_000, // 5 minutes (layer data is hourly-computed on server)
      },
    ]);
  } catch (err) {
    console.error('[GlobeScreen] Failed to load:', err);
    mapEl.innerHTML = '';
    mapEl.appendChild(
      h('div', { className: 'screen-placeholder' }, 'Globe failed to load'),
    );
  }
}

// ---------------------------------------------------------------------------
// Unmount
// ---------------------------------------------------------------------------

export function unmountGlobe(ctx: GeoPolAppContext): void {
  // Remove event listeners
  if (countrySelectedHandler) {
    window.removeEventListener('country-selected', countrySelectedHandler);
    countrySelectedHandler = null;
  }
  if (forecastSelectedHandler) {
    window.removeEventListener('forecast-selected', forecastSelectedHandler);
    forecastSelectedHandler = null;
  }
  if (countryBriefHandler) {
    window.removeEventListener('country-brief-requested', countryBriefHandler);
    countryBriefHandler = null;
  }

  // Destroy scheduler
  if (scheduler) {
    scheduler.destroy();
    scheduler = null;
    ctx.scheduler = null;
  }

  // Destroy overlay components
  if (hud) { hud.destroy(); hud = null; }
  if (pillBar) { pillBar.destroy(); pillBar = null; }
  if (drillDown) { drillDown.destroy(); drillDown = null; }

  // Destroy modals
  if (scenarioExplorer) { scenarioExplorer.destroy(); scenarioExplorer = null; }
  if (countryBriefPage) { countryBriefPage.destroy(); countryBriefPage = null; }

  // Destroy CesiumMap
  if (cesiumMap) { cesiumMap.destroy(); cesiumMap = null; }
}

// ---------------------------------------------------------------------------
// Event wiring
// ---------------------------------------------------------------------------

function wireEvents(_ctx: GeoPolAppContext): void {
  // Country click: fly camera + open drill-down
  countrySelectedHandler = ((e: CustomEvent<{ iso: string }>) => {
    const { iso } = e.detail;
    if (cesiumMap) {
      cesiumMap.flyToCountry(iso);
      cesiumMap.setSelectedCountry(iso);
    }
    if (drillDown) {
      void drillDown.open(iso);
    }
  }) as EventListener;
  window.addEventListener('country-selected', countrySelectedHandler);

  // "View Full Analysis" from expanded card -> ScenarioExplorer
  // ScenarioExplorer already listens for forecast-selected in its constructor,
  // so we don't need to add another handler. But we register one for the
  // globe-specific behavior of updating selected forecast on the map.
  forecastSelectedHandler = ((e: CustomEvent<{ forecast: ForecastResponse }>) => {
    if (cesiumMap) {
      cesiumMap.setSelectedForecast(e.detail.forecast);
    }
  }) as EventListener;
  window.addEventListener('forecast-selected', forecastSelectedHandler);

  // "View Full Country Brief" from drill-down -> CountryBriefPage
  countryBriefHandler = ((e: CustomEvent<{ iso: string }>) => {
    if (countryBriefPage) {
      countryBriefPage.open(e.detail.iso);
    }
  }) as EventListener;
  window.addEventListener('country-brief-requested', countryBriefHandler);
}

// ---------------------------------------------------------------------------
// Data loading
// ---------------------------------------------------------------------------

async function loadInitialData(): Promise<void> {
  try {
    // Load core data (countries + forecasts) and layer data in parallel.
    // Layer data load failures are non-fatal -- layers simply remain empty
    // until the next 5-minute refresh cycle succeeds.
    const [countries, forecasts] = await Promise.all([
      forecastClient.getCountries(),
      forecastClient.getTopForecasts(50),
    ]);
    pushCountries(countries);
    pushForecasts(forecasts);

    // Layer data loaded separately -- circuit breaker handles failures gracefully
    const [heatmap, arcs, deltas] = await Promise.all([
      forecastClient.getHeatmapData(),
      forecastClient.getArcData(),
      forecastClient.getRiskDeltas(),
    ]);
    pushHeatmap(heatmap);
    pushArcs(arcs);
    pushDeltas(deltas);
  } catch (err) {
    console.error('[GlobeScreen] Initial data load failed:', err);
  }
}

function pushCountries(countries: CountryRiskSummary[]): void {
  if (cesiumMap) {
    cesiumMap.updateRiskScores(countries);
  }
  if (hud) {
    hud.update(countries);
  }
}

function pushForecasts(forecasts: ForecastResponse[]): void {
  if (cesiumMap) {
    cesiumMap.updateForecasts(forecasts);
  }
}

/**
 * Convert API HexbinData to HexBinDatum and push to CesiumMap.
 * Field names align 1:1 -- no mapping needed.
 */
function pushHeatmap(data: HexbinData[]): void {
  if (!cesiumMap) return;
  const mapped: HexBinDatum[] = data.map((d) => ({
    h3_index: d.h3_index,
    weight: d.weight,
    event_count: d.event_count,
  }));
  cesiumMap.updateHeatmapData(mapped);
}

/**
 * Convert API ArcData to BilateralArcDatum and push to CesiumMap.
 * Resolves country centroids via countryGeometry. Drops arcs where
 * either centroid is unavailable (unknown ISO code).
 */
function pushArcs(data: ArcData[]): void {
  if (!cesiumMap) return;
  const mapped: BilateralArcDatum[] = [];
  for (const d of data) {
    if (!d.source_iso || !d.target_iso) continue;
    const src = countryGeometry.getCentroid(d.source_iso.toUpperCase());
    const tgt = countryGeometry.getCentroid(d.target_iso.toUpperCase());
    if (!src || !tgt) continue;
    mapped.push({
      sourceIso: d.source_iso.toUpperCase(),
      targetIso: d.target_iso.toUpperCase(),
      source: src,
      target: tgt,
      eventCount: d.event_count,
      avgGoldstein: d.avg_goldstein,
    });
  }
  cesiumMap.updateArcData(mapped);
}

/**
 * Convert API RiskDeltaData to RiskDeltaDatum and push to CesiumMap.
 */
function pushDeltas(data: RiskDeltaData[]): void {
  if (!cesiumMap) return;
  const mapped: RiskDeltaDatum[] = [];
  for (const d of data) {
    if (!d.country_iso) continue;
    mapped.push({ iso: d.country_iso.toUpperCase(), delta: d.delta });
  }
  cesiumMap.updateRiskDeltas(mapped);
}
