/**
 * Dashboard screen -- 4-column layout with all existing panels.
 *
 * Column assignments (Phase 21):
 *   Col 1 (25%): NewsFeedPanel
 *   Col 2 (30%): SearchBar, ForecastPanel, ComparisonPanel
 *   Col 3 (30%): MyForecastsPanel
 *   Col 4 (15%): RiskIndexPanel, SystemHealth, Polymarket
 *
 * BreakingNewsBanner is a standalone overlay attached to document.body.
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
import { NewsFeedPanel } from '@/components/NewsFeedPanel';
import { SystemHealthPanel } from '@/components/SystemHealthPanel';
import { PolymarketPanel } from '@/components/PolymarketPanel';
import { MyForecastsPanel } from '@/components/MyForecastsPanel';
import { ComparisonPanel } from '@/components/ComparisonPanel';

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

// ---------------------------------------------------------------------------
// Mount / Unmount
// ---------------------------------------------------------------------------

export function mountDashboard(container: HTMLElement, ctx: GeoPolAppContext): void {
  const { element, columns } = createDashboardLayout();
  container.appendChild(element);

  // -- Create panels --
  const forecastPanel = new ForecastPanel();
  const riskIndexPanel = new RiskIndexPanel();
  const newsFeedPanel = new NewsFeedPanel();
  const healthPanel = new SystemHealthPanel();
  const polymarketPanel = new PolymarketPanel();
  const myForecastsPanel = new MyForecastsPanel();
  const comparisonPanel = new ComparisonPanel();

  // -- Search bar at top of Col 2 --
  searchBar = new SearchBar();
  columns.col2.appendChild(searchBar.getElement());

  // -- Mount panels into columns --
  // Col 1 (25%): News feed
  columns.col1.appendChild(newsFeedPanel.getElement());
  // Col 2 (30%): Search + forecasts + comparisons
  columns.col2.appendChild(forecastPanel.getElement());
  columns.col2.appendChild(comparisonPanel.getElement());
  // Col 3 (30%): My forecasts
  columns.col3.appendChild(myForecastsPanel.getElement());
  // Col 4 (15%): Risk index + health + polymarket
  columns.col4.appendChild(riskIndexPanel.getElement());
  columns.col4.appendChild(healthPanel.getElement());
  columns.col4.appendChild(polymarketPanel.getElement());

  // -- Register in context --
  ctx.panels['forecasts'] = forecastPanel;
  ctx.panels['risk-index'] = riskIndexPanel;
  ctx.panels['news-feed'] = newsFeedPanel;
  ctx.panels['system-health'] = healthPanel;
  ctx.panels['polymarket'] = polymarketPanel;
  ctx.panels['my-forecasts'] = myForecastsPanel;
  ctx.panels['comparisons'] = comparisonPanel;

  // -- Modals (attach global event listeners in constructor) --
  scenarioExplorer = new ScenarioExplorer();
  countryBriefPage = new CountryBriefPage();

  // -- Initial data load --
  const loadInitial = async (): Promise<void> => {
    const [forecasts, countries, health, polymarket, requests, comparisons] = await Promise.all([
      forecastClient.getTopForecasts(10),
      forecastClient.getCountries(),
      forecastClient.getHealth(),
      forecastClient.getPolymarketTop(),
      forecastClient.getRequests(),
      forecastClient.getComparisons(),
    ]);

    pushForecasts(forecasts, forecastPanel);
    pushCountries(countries, riskIndexPanel);
    searchBar?.updateCountries(countries);
    pushHealth(health, healthPanel);
    pushRequests(requests, myForecastsPanel);
    polymarketPanel.update(polymarket);
    comparisonPanel.update(comparisons);
  };

  loadInitial().catch((err) => {
    console.error('[Dashboard] Initial data load failed:', err);
  });

  // -- Event wiring --
  forecastSelectedHandler = ((_e: CustomEvent<{ forecast: ForecastResponse }>) => {
    // Reserved for future cross-panel coordination on forecast selection
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
      },
      intervalMs: 30_000,
    },
    {
      name: 'news-feed',
      fn: async () => {
        await newsFeedPanel.refresh();
      },
      intervalMs: 60_000,
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
        const polymarket = await forecastClient.getPolymarketTop();
        polymarketPanel.update(polymarket);
      },
      intervalMs: 300_000,
    },
    {
      name: 'comparisons',
      fn: async () => {
        const comparisons = await forecastClient.getComparisons();
        comparisonPanel.update(comparisons);
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
