/**
 * main.ts -- Geopol dashboard bootstrap.
 *
 * Orchestrates the full application lifecycle:
 *  1. Apply stored theme (FOUC prevention)
 *  2. Create panel grid layout
 *  3. Initialize AppContext
 *  4. Load country geometry (GeoJSON)
 *  5. Create and mount all 6 panels + DeckGLMap
 *  6. Create ScenarioExplorer + CountryBriefPage modals
 *  7. Initial data load (forecasts, countries, health)
 *  8. Wire inter-component events
 *  9. Start RefreshScheduler
 * 10. Visibility-aware background throttling
 */

import { applyStoredTheme, toggleTheme } from '@/utils/theme-manager';
import { createAppContext } from '@/app/app-context';
import { createPanelLayout } from '@/app/panel-layout';
import { RefreshScheduler } from '@/app/refresh-scheduler';
import { countryGeometry } from '@/services/country-geometry';
import { forecastClient } from '@/services/forecast-client';
import { h } from '@/utils/dom-utils';

// Panels
import { ForecastPanel } from '@/components/ForecastPanel';
import { RiskIndexPanel } from '@/components/RiskIndexPanel';
import { EventTimelinePanel } from '@/components/EventTimelinePanel';
import { EnsembleBreakdownPanel } from '@/components/EnsembleBreakdownPanel';
import { SystemHealthPanel } from '@/components/SystemHealthPanel';
import { CalibrationPanel } from '@/components/CalibrationPanel';

// Map
import { DeckGLMap } from '@/components/DeckGLMap';

// Modals
import { ScenarioExplorer } from '@/components/ScenarioExplorer';
import { CountryBriefPage } from '@/components/CountryBriefPage';

// Styles
import 'maplibre-gl/dist/maplibre-gl.css';
import '@/styles/main.css';
import '@/styles/panels.css';

// Types
import type { ForecastResponse, CountryRiskSummary, HealthResponse } from '@/types/api';

// ---------------------------------------------------------------------------
// Header
// ---------------------------------------------------------------------------

function createHeader(): HTMLElement {
  const themeBtn = h('button', {
    className: 'header-theme-toggle',
    'aria-label': 'Toggle theme',
    onClick: () => toggleTheme(),
  }, 'Theme');

  return h('header', { className: 'app-header' },
    h('div', { className: 'header-left' },
      h('span', { className: 'header-logo' }, 'GEOPOL'),
      h('span', { className: 'header-subtitle' }, 'Geopolitical Forecast Dashboard'),
    ),
    h('div', { className: 'header-right' },
      themeBtn,
    ),
  );
}

// ---------------------------------------------------------------------------
// Bootstrap
// ---------------------------------------------------------------------------

async function boot(): Promise<void> {
  applyStoredTheme();

  const app = document.getElementById('app');
  if (!app) throw new Error('Missing #app container');

  const ctx = createAppContext(app);

  // -- Layout --
  app.innerHTML = '';
  app.appendChild(createHeader());

  const { grid, slots } = createPanelLayout();
  app.appendChild(grid);

  // -- Country geometry (async, non-blocking for panel mount) --
  const geoPromise = countryGeometry.load();

  // -- Panels --
  const forecastPanel = new ForecastPanel();
  const riskIndexPanel = new RiskIndexPanel();
  const eventTimelinePanel = new EventTimelinePanel();
  const ensemblePanel = new EnsembleBreakdownPanel();
  const healthPanel = new SystemHealthPanel();
  const calibrationPanel = new CalibrationPanel();

  // Mount panels into grid slots
  slots['forecasts'].appendChild(forecastPanel.getElement());
  slots['ensemble'].appendChild(ensemblePanel.getElement());
  slots['calibration'].appendChild(calibrationPanel.getElement());
  slots['risk-index'].appendChild(riskIndexPanel.getElement());
  slots['system-health'].appendChild(healthPanel.getElement());
  slots['event-timeline'].appendChild(eventTimelinePanel.getElement());

  // Register in context for lifecycle management
  ctx.panels['forecasts'] = forecastPanel;
  ctx.panels['risk-index'] = riskIndexPanel;
  ctx.panels['event-timeline'] = eventTimelinePanel;
  ctx.panels['ensemble'] = ensemblePanel;
  ctx.panels['system-health'] = healthPanel;
  ctx.panels['calibration'] = calibrationPanel;

  // -- DeckGLMap (requires geometry loaded) --
  await geoPromise;
  const deckMap = new DeckGLMap(slots['map']);

  // -- Modals --
  const scenarioExplorer = new ScenarioExplorer();
  const countryBriefPage = new CountryBriefPage();

  // -- Initial data load --
  // Coordinated parallel fetch: update() pushes data to panels and map
  const update = async (): Promise<void> => {
    const [forecasts, countries, health] = await Promise.all([
      forecastClient.getTopForecasts(10),
      forecastClient.getCountries(),
      forecastClient.getHealth(),
    ]);

    pushForecasts(forecasts, forecastPanel, deckMap);
    pushCountries(countries, riskIndexPanel, deckMap);
    pushHealth(health, healthPanel);

    // EventTimeline renders mock data on first load
    eventTimelinePanel.refresh();
  };

  // Fire initial load; circuit breakers handle backend unavailability
  update().catch((err) => {
    console.error('[Geopol] Initial data load failed:', err);
  });

  // -- Event wiring --

  // forecast-selected -> EnsembleBreakdownPanel + CalibrationPanel + map highlight
  window.addEventListener('forecast-selected', ((e: CustomEvent<{ forecast: ForecastResponse }>) => {
    const { forecast } = e.detail;
    ensemblePanel.update(forecast);
    calibrationPanel.update([forecast.calibration]);
    deckMap.setSelectedForecast(forecast);
  }) as EventListener);

  // country-selected -> map highlight + CountryBriefPage (handled internally by CBP)
  window.addEventListener('country-selected', ((e: CustomEvent<{ iso: string }>) => {
    deckMap.setSelectedCountry(e.detail.iso);
  }) as EventListener);

  // -- RefreshScheduler --
  const scheduler = new RefreshScheduler(ctx);
  scheduler.init();
  ctx.scheduler = scheduler;

  const refresh = (): void => {
    scheduler.registerAll([
      {
        name: 'forecasts',
        fn: async () => {
          const forecasts = await forecastClient.getTopForecasts(10);
          pushForecasts(forecasts, forecastPanel, deckMap);
        },
        intervalMs: 60_000,
      },
      {
        name: 'countries',
        fn: async () => {
          const countries = await forecastClient.getCountries();
          pushCountries(countries, riskIndexPanel, deckMap);
        },
        intervalMs: 120_000,
      },
      {
        name: 'health',
        fn: async () => {
          const health = await forecastClient.getHealth();
          pushHealth(health, healthPanel);
        },
        intervalMs: 30_000,
      },
    ]);
  };

  refresh();

  // Visibility-aware: flush stale on tab restore
  document.addEventListener('visibilitychange', () => {
    if (document.visibilityState === 'hidden') {
      scheduler.setHiddenSince(Date.now());
    } else {
      scheduler.flushStaleRefreshes();
    }
  });

  // -- Debug access --
  if (import.meta.env.DEV) {
    (window as unknown as Record<string, unknown>)['__geopol'] = {
      ctx,
      deckMap,
      scenarioExplorer,
      countryBriefPage,
      forecastPanel,
      riskIndexPanel,
      eventTimelinePanel,
      ensemblePanel,
      healthPanel,
      calibrationPanel,
    };
  }
}

// ---------------------------------------------------------------------------
// Data push helpers -- update panels and map in one call
// ---------------------------------------------------------------------------

function pushForecasts(
  forecasts: ForecastResponse[],
  panel: ForecastPanel,
  map: DeckGLMap,
): void {
  panel.update(forecasts);
  map.updateForecasts(forecasts);
}

function pushCountries(
  countries: CountryRiskSummary[],
  panel: RiskIndexPanel,
  map: DeckGLMap,
): void {
  panel.update(countries);
  map.updateRiskScores(countries);
}

function pushHealth(
  health: HealthResponse,
  panel: SystemHealthPanel,
): void {
  panel.update(health);
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

boot().catch((err) => {
  console.error('[Geopol] Boot failed:', err);
  const app = document.getElementById('app');
  if (app) {
    app.innerHTML = `
      <div style="
        display: flex; align-items: center; justify-content: center;
        height: 100vh; color: #e63946; font-family: monospace;
        font-size: 14px; text-align: center; padding: 20px;
      ">
        GEOPOL BOOT FAILURE<br/><br/>
        ${String(err)}
      </div>
    `;
  }
});
