/**
 * Dashboard screen -- 4-column layout with all existing panels.
 *
 * Column assignments:
 *   Col 1: RiskIndexPanel
 *   Col 2: SearchBar, ForecastPanel
 *   Col 3: MyForecastsPanel, SourcesPanel
 *   Col 4: EventTimeline, EnsembleBreakdown, SystemHealth, Calibration
 *
 * Owns the RefreshScheduler lifecycle and all inter-panel event wiring.
 */

import { createDashboardLayout } from '@/app/panel-layout';
import { RefreshScheduler } from '@/app/refresh-scheduler';
import { forecastClient } from '@/services/forecast-client';
import type { GeoPolAppContext } from '@/app/app-context';
import type { ForecastResponse, HealthResponse, CountryRiskSummary, ForecastRequestStatus } from '@/types/api';

// Panels
import { ForecastPanel } from '@/components/ForecastPanel';
import { RiskIndexPanel } from '@/components/RiskIndexPanel';
import { EventTimelinePanel } from '@/components/EventTimelinePanel';
import { EnsembleBreakdownPanel } from '@/components/EnsembleBreakdownPanel';
import { SystemHealthPanel } from '@/components/SystemHealthPanel';
import { CalibrationPanel } from '@/components/CalibrationPanel';
import { MyForecastsPanel } from '@/components/MyForecastsPanel';
import { SourcesPanel } from '@/components/SourcesPanel';

// Search
import { SearchBar } from '@/components/SearchBar';

// Modals
import { ScenarioExplorer } from '@/components/ScenarioExplorer';
import { CountryBriefPage } from '@/components/CountryBriefPage';

// ---------------------------------------------------------------------------
// State scoped to this screen mount
// ---------------------------------------------------------------------------

let scheduler: RefreshScheduler | null = null;
let forecastSelectedHandler: EventListener | null = null;
let countrySelectedHandler: EventListener | null = null;

// Search bar reference -- must be destroyed on unmount
let searchBar: SearchBar | null = null;

// Modal references -- must be destroyed on unmount to remove global listeners
let scenarioExplorer: ScenarioExplorer | null = null;
let countryBriefPage: CountryBriefPage | null = null;

// ---------------------------------------------------------------------------
// Data push helpers
// ---------------------------------------------------------------------------

function pushForecasts(forecasts: ForecastResponse[], panel: ForecastPanel): void {
  panel.update(forecasts);
}

function pushCountries(countries: CountryRiskSummary[], panel: RiskIndexPanel): void {
  panel.update(countries);
}

function pushHealth(health: HealthResponse, panel: SystemHealthPanel): void {
  panel.update(health);
}

function pushRequests(requests: ForecastRequestStatus[], panel: MyForecastsPanel): void {
  panel.update(requests);
}

function pushSources(health: HealthResponse, panel: SourcesPanel): void {
  panel.update(health);
}

// ---------------------------------------------------------------------------
// Mount / Unmount
// ---------------------------------------------------------------------------

export function mountDashboard(container: HTMLElement, ctx: GeoPolAppContext): void {
  const { element, columns } = createDashboardLayout();
  container.appendChild(element);

  // -- Create panels --
  const forecastPanel = new ForecastPanel();
  const riskIndexPanel = new RiskIndexPanel();
  const eventTimelinePanel = new EventTimelinePanel();
  const ensemblePanel = new EnsembleBreakdownPanel();
  const healthPanel = new SystemHealthPanel();
  const calibrationPanel = new CalibrationPanel();
  const myForecastsPanel = new MyForecastsPanel();
  const sourcesPanel = new SourcesPanel();

  // -- Search bar at top of Col 2 --
  searchBar = new SearchBar();
  columns.col2.appendChild(searchBar.getElement());

  // -- Mount panels into columns --
  columns.col1.appendChild(riskIndexPanel.getElement());
  columns.col2.appendChild(forecastPanel.getElement());
  columns.col3.appendChild(myForecastsPanel.getElement());
  columns.col3.appendChild(sourcesPanel.getElement());
  columns.col4.appendChild(eventTimelinePanel.getElement());
  columns.col4.appendChild(ensemblePanel.getElement());
  columns.col4.appendChild(healthPanel.getElement());
  columns.col4.appendChild(calibrationPanel.getElement());

  // -- Register in context --
  ctx.panels['forecasts'] = forecastPanel;
  ctx.panels['risk-index'] = riskIndexPanel;
  ctx.panels['event-timeline'] = eventTimelinePanel;
  ctx.panels['ensemble'] = ensemblePanel;
  ctx.panels['system-health'] = healthPanel;
  ctx.panels['calibration'] = calibrationPanel;
  ctx.panels['my-forecasts'] = myForecastsPanel;
  ctx.panels['sources'] = sourcesPanel;

  // -- Modals (attach global event listeners in constructor) --
  scenarioExplorer = new ScenarioExplorer();
  countryBriefPage = new CountryBriefPage();

  // -- Initial data load --
  const loadInitial = async (): Promise<void> => {
    const [forecasts, countries, health, polymarket, requests] = await Promise.all([
      forecastClient.getTopForecasts(10),
      forecastClient.getCountries(),
      forecastClient.getHealth(),
      forecastClient.getPolymarket(),
      forecastClient.getRequests(),
    ]);

    pushForecasts(forecasts, forecastPanel);
    pushCountries(countries, riskIndexPanel);
    searchBar?.updateCountries(countries);
    pushHealth(health, healthPanel);
    pushSources(health, sourcesPanel);
    pushRequests(requests, myForecastsPanel);
    calibrationPanel.updatePolymarket(polymarket);
    eventTimelinePanel.refresh();
  };

  loadInitial().catch((err) => {
    console.error('[Dashboard] Initial data load failed:', err);
  });

  // -- Event wiring --
  forecastSelectedHandler = ((e: CustomEvent<{ forecast: ForecastResponse }>) => {
    const { forecast } = e.detail;
    ensemblePanel.update(forecast);
    calibrationPanel.update([forecast.calibration]);
  }) as EventListener;
  window.addEventListener('forecast-selected', forecastSelectedHandler);

  countrySelectedHandler = ((e: CustomEvent<{ iso: string }>) => {
    ctx.setCountryFilter(e.detail.iso);
  }) as EventListener;
  window.addEventListener('country-selected', countrySelectedHandler);

  // -- Refresh scheduler --
  scheduler = new RefreshScheduler(ctx);
  scheduler.init();
  ctx.scheduler = scheduler;

  scheduler.registerAll([
    {
      name: 'forecasts',
      fn: async () => {
        const forecasts = await forecastClient.getTopForecasts(10);
        pushForecasts(forecasts, forecastPanel);
      },
      intervalMs: 60_000,
    },
    {
      name: 'countries',
      fn: async () => {
        const countries = await forecastClient.getCountries();
        pushCountries(countries, riskIndexPanel);
        searchBar?.updateCountries(countries);
      },
      intervalMs: 120_000,
    },
    {
      name: 'health',
      fn: async () => {
        const health = await forecastClient.getHealth();
        pushHealth(health, healthPanel);
        pushSources(health, sourcesPanel);
      },
      intervalMs: 30_000,
    },
    {
      name: 'my-forecasts',
      fn: async () => {
        const requests = await forecastClient.getRequests();
        pushRequests(requests, myForecastsPanel);
      },
      intervalMs: 30_000,
    },
    {
      name: 'polymarket',
      fn: async () => {
        const polymarket = await forecastClient.getPolymarket();
        calibrationPanel.updatePolymarket(polymarket);
      },
      intervalMs: 300_000,
    },
  ]);
}

export function unmountDashboard(ctx: GeoPolAppContext): void {
  // Remove event listeners
  if (forecastSelectedHandler) {
    window.removeEventListener('forecast-selected', forecastSelectedHandler);
    forecastSelectedHandler = null;
  }
  if (countrySelectedHandler) {
    window.removeEventListener('country-selected', countrySelectedHandler);
    countrySelectedHandler = null;
  }

  // Destroy scheduler
  if (scheduler) {
    scheduler.destroy();
    scheduler = null;
    ctx.scheduler = null;
  }

  // Destroy panels
  for (const panel of Object.values(ctx.panels)) {
    panel.destroy();
  }
  ctx.panels = {};

  // Destroy search bar
  if (searchBar) {
    searchBar.destroy();
    searchBar = null;
  }

  // Destroy modals (removes global event listeners)
  if (scenarioExplorer) {
    scenarioExplorer.destroy();
    scenarioExplorer = null;
  }
  if (countryBriefPage) {
    countryBriefPage.destroy();
    countryBriefPage = null;
  }
}
