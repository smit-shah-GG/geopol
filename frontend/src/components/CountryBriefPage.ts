/**
 * CountryBriefPage -- Full-screen tabbed modal for country-level drill-down.
 *
 * Opens when a country is clicked on the globe (via 'country-selected' CustomEvent).
 * Seven tabs: Overview, Active Forecasts, GDELT Events, Risk Signals,
 * Forecast History, Entity Relations, Calibration.
 *
 * This is FE-06: the deepest data exploration surface. Analysts see everything
 * the system knows about a country -- forecasts, event signals, entity networks,
 * and calibration quality -- in one full-screen modal.
 *
 * Not a Panel subclass. Standalone modal like ScenarioExplorer.
 */

import * as d3 from 'd3';
import { h, clearChildren } from '@/utils/dom-utils';
import { trapFocus } from '@/utils/focus-trap';
import { forecastClient } from '@/services/forecast-client';
import type {
  AdvisoryDTO,
  EventDTO,
  ForecastResponse,
  CountryRiskSummary,
  PaginatedResponse,
  CalibrationDTO,
  ScenarioDTO,
} from '@/types/api';

const SVG_NS = 'http://www.w3.org/2000/svg';

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

type TabId =
  | 'overview'
  | 'forecasts'
  | 'events'
  | 'risk-signals'
  | 'history'
  | 'entities'
  | 'calibration';

const TAB_LABELS: Record<TabId, string> = {
  overview: 'Overview',
  forecasts: 'Active Forecasts',
  events: 'Events',
  'risk-signals': 'Risk Signals',
  history: 'Forecast History',
  entities: 'Entity Relations',
  calibration: 'Calibration',
};

const TAB_ORDER: TabId[] = [
  'overview',
  'forecasts',
  'events',
  'risk-signals',
  'history',
  'entities',
  'calibration',
];

/** CAMEO root categories (01-20). */
const CAMEO_CATEGORIES: Array<{ code: string; name: string; goldsteinBase: number }> = [
  { code: '01', name: 'MAKE PUBLIC STATEMENT', goldsteinBase: 0 },
  { code: '02', name: 'APPEAL', goldsteinBase: 3.0 },
  { code: '03', name: 'EXPRESS INTENT TO COOPERATE', goldsteinBase: 3.5 },
  { code: '04', name: 'CONSULT', goldsteinBase: 1.0 },
  { code: '05', name: 'ENGAGE IN DIPLOMATIC COOPERATION', goldsteinBase: 3.5 },
  { code: '06', name: 'ENGAGE IN MATERIAL COOPERATION', goldsteinBase: 6.0 },
  { code: '07', name: 'PROVIDE AID', goldsteinBase: 7.0 },
  { code: '08', name: 'YIELD', goldsteinBase: 5.0 },
  { code: '09', name: 'INVESTIGATE', goldsteinBase: -0.5 },
  { code: '10', name: 'DEMAND', goldsteinBase: -3.0 },
  { code: '11', name: 'DISAPPROVE', goldsteinBase: -2.0 },
  { code: '12', name: 'REJECT', goldsteinBase: -4.0 },
  { code: '13', name: 'THREATEN', goldsteinBase: -5.0 },
  { code: '14', name: 'PROTEST', goldsteinBase: -6.5 },
  { code: '15', name: 'EXHIBIT MILITARY POSTURE', goldsteinBase: -7.0 },
  { code: '16', name: 'REDUCE RELATIONS', goldsteinBase: -4.0 },
  { code: '17', name: 'COERCE', goldsteinBase: -7.0 },
  { code: '18', name: 'ASSAULT', goldsteinBase: -8.0 },
  { code: '19', name: 'FIGHT', goldsteinBase: -9.0 },
  { code: '20', name: 'ENGAGE IN UNCONVENTIONAL MASS VIOLENCE', goldsteinBase: -10.0 },
];

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** ISO alpha-2 to regional indicator flag emoji. */
function isoToFlag(iso: string): string {
  return iso
    .toUpperCase()
    .split('')
    .map((c) => String.fromCodePoint(0x1f1e6 + c.charCodeAt(0) - 65))
    .join('');
}

/** Format probability as percentage. */
function pctLabel(p: number): string {
  return `${(p * 100).toFixed(0)}%`;
}

/** Severity class based on probability thresholds. */
function severityClass(p: number): string {
  if (p > 0.8) return 'severity-critical';
  if (p > 0.6) return 'severity-high';
  if (p > 0.4) return 'severity-elevated';
  if (p > 0.2) return 'severity-normal';
  return 'severity-low';
}

/** Trend arrow character. */
function trendArrow(trend: 'rising' | 'stable' | 'falling'): string {
  return trend === 'rising' ? '\u2197' : trend === 'falling' ? '\u2198' : '\u2192';
}

/** Goldstein scale color: negative = red, zero = muted, positive = green. */
function goldsteinColor(g: number): string {
  if (g > 0) return 'var(--semantic-success)';
  if (g < 0) return 'var(--semantic-critical)';
  return 'var(--text-muted)';
}

/** Truncate text with ellipsis. */
function truncate(text: string, maxLen: number): string {
  return text.length > maxLen ? text.slice(0, maxLen - 1) + '\u2026' : text;
}

/** Create an SVG element with attributes. */
function svgEl(tag: string, attrs: Record<string, string | number>): SVGElement {
  const el = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs)) {
    el.setAttribute(k, String(v));
  }
  return el;
}

/** Brier score severity classification. */
function brierClass(score: number): string {
  if (score < 0.1) return 'brier-excellent';
  if (score < 0.25) return 'brier-good';
  return 'brier-poor';
}

/** Actor aggregation for entities tab. */
interface ActorCount {
  actor: string;
  count: number;
  lastSeen: string;
}

/** Advisory level color mapping. */
function advisoryLevelColor(level: number): string {
  switch (level) {
    case 1: return 'var(--semantic-success)';
    case 2: return 'var(--semantic-warning, #e5c07b)';
    case 3: return 'var(--semantic-elevated, #d19a66)';
    case 4: return 'var(--semantic-critical)';
    default: return 'var(--text-muted)';
  }
}

/** Advisory level CSS class. */
function advisoryLevelClass(level: number): string {
  switch (level) {
    case 1: return 'advisory-level-1';
    case 2: return 'advisory-level-2';
    case 3: return 'advisory-level-3';
    case 4: return 'advisory-level-4';
    default: return '';
  }
}

/** CAMEO quad_class to severity category for events. */
function quadCategory(quadClass: number | null): string {
  if (quadClass === null) return 'neutral';
  if (quadClass <= 2) return 'cooperative';
  if (quadClass === 3) return 'neutral';
  return 'conflictual';
}

/** CAMEO event_code to severity category. */
function cameoCategory(code: string): string {
  const prefix = parseInt(code.slice(0, 2), 10);
  if (Number.isNaN(prefix)) return 'neutral';
  if (prefix <= 5) return 'cooperative';
  if (prefix <= 9) return 'neutral';
  if (prefix <= 14) return 'conflictual';
  return 'hostile';
}

/** Relative time string from ISO timestamp. */
function relativeTime(iso: string): string {
  const diff = Date.now() - new Date(iso).getTime();
  const mins = Math.floor(diff / 60_000);
  if (mins < 1) return 'now';
  if (mins < 60) return `${mins}m`;
  const hrs = Math.floor(mins / 60);
  if (hrs < 24) return `${hrs}h`;
  const days = Math.floor(hrs / 24);
  return `${days}d`;
}

/** Entity co-occurrence edge for force graph. */
interface EntityEdge {
  source: string;
  target: string;
  weight: number;
  frequency: number;
}

/** D3 force simulation node. */
interface EntityNode extends d3.SimulationNodeDatum {
  id: string;
  frequency: number;
}

// ---------------------------------------------------------------------------
// CountryBriefPage
// ---------------------------------------------------------------------------

export class CountryBriefPage {
  private backdrop: HTMLElement | null = null;
  private modal: HTMLElement | null = null;
  private tabContent: HTMLElement | null = null;
  private activeTab: TabId = 'overview';
  private entityViewMode: 'graph' | 'table' = 'graph';
  private releaseTrap: (() => void) | null = null;
  private triggerElement: HTMLElement | null = null;

  // Loaded data (from loadData -- fetched on open)
  private currentIso: string | null = null;
  private forecasts: ForecastResponse[] = [];
  private countryRisk: CountryRiskSummary | null = null;
  private loading = false;

  // Lazy-loaded per-tab data (null = not loaded, [] = loaded but empty)
  private events: EventDTO[] | null = null;
  private advisories: AdvisoryDTO[] | null = null;
  private actorCounts: ActorCount[] | null = null;

  // Event listeners (stored for cleanup)
  private readonly onCountrySelected: (e: Event) => void;
  private readonly onKeyDown: (e: KeyboardEvent) => void;

  constructor(opts?: { autoOpen?: boolean }) {
    this.onCountrySelected = (e: Event) => {
      const detail = (e as CustomEvent<{ iso: string }>).detail;
      this.open(detail.iso);
    };

    this.onKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') this.close();
    };

    // autoOpen defaults to true (dashboard behavior).
    // Globe screen passes { autoOpen: false } because it wires country-brief-requested instead.
    if (opts?.autoOpen !== false) {
      window.addEventListener('country-selected', this.onCountrySelected);
    }
  }

  // ==================================================================
  // Public API
  // ==================================================================

  public open(iso: string): void {
    // Capture trigger element for focus restoration on close
    this.triggerElement = document.activeElement as HTMLElement | null;

    this.currentIso = iso;
    this.activeTab = 'overview';
    this.forecasts = [];
    this.countryRisk = null;
    this.loading = true;

    // Reset lazy-loaded tab data for fresh country
    this.events = null;
    this.advisories = null;
    this.actorCounts = null;

    this.buildModal(iso);
    document.addEventListener('keydown', this.onKeyDown);
    this.loadData(iso);
  }

  public close(): void {
    // Release focus trap before removing DOM
    this.releaseTrap?.();
    this.releaseTrap = null;

    document.removeEventListener('keydown', this.onKeyDown);
    if (this.backdrop) {
      this.backdrop.remove();
      this.backdrop = null;
    }
    this.modal = null;
    this.tabContent = null;
    this.currentIso = null;

    // Restore focus to the element that triggered the modal
    this.triggerElement?.focus();
    this.triggerElement = null;
  }

  public destroy(): void {
    this.close();
    window.removeEventListener('country-selected', this.onCountrySelected);
  }

  // ==================================================================
  // Data loading
  // ==================================================================

  private async loadData(iso: string): Promise<void> {
    try {
      const [paginatedForecasts, risk] = await Promise.all([
        forecastClient.getForecastsByCountry(iso),
        forecastClient.getCountryRisk(iso),
      ]);

      // Guard against stale responses after user closes/switches
      if (this.currentIso !== iso) return;

      this.forecasts = (paginatedForecasts as PaginatedResponse<ForecastResponse>).items;
      this.countryRisk = risk;
      this.loading = false;
      this.renderActiveTab();
    } catch {
      if (this.currentIso !== iso) return;
      this.loading = false;
      this.renderActiveTab();
    }
  }

  // ==================================================================
  // Modal construction
  // ==================================================================

  private buildModal(iso: string): void {
    if (this.backdrop) this.backdrop.remove();

    // Backdrop
    this.backdrop = h('div', { className: 'country-brief-backdrop' });
    this.backdrop.addEventListener('click', (e: MouseEvent) => {
      if (e.target === this.backdrop) this.close();
    });

    // Modal
    this.modal = h('div', {
      className: 'country-brief-modal',
      role: 'dialog',
      'aria-modal': 'true',
      'aria-label': 'Country Brief',
    });

    // Header
    const flag = isoToFlag(iso);
    const displayName = iso; // Will be replaced by countryRisk.iso_code display name if available

    const header = h('div', { className: 'country-brief-header' },
      h('div', { className: 'country-brief-header-left' },
        h('span', { className: 'country-brief-flag' }, flag),
        h('span', { className: 'country-brief-name' }, displayName),
        h('span', { className: 'country-brief-iso' }, `[${iso}]`),
        h('span', { className: 'country-brief-risk-badge', dataset: { iso } }, 'Loading...'),
      ),
      h('button', {
        className: 'country-brief-close-btn',
        'aria-label': 'Close country brief',
        onclick: () => this.close(),
      }, '\u00d7'),
    );

    // Tab bar
    const tabBar = h('div', { className: 'country-brief-tabs', role: 'tablist' });
    for (const tabId of TAB_ORDER) {
      const isActive = tabId === this.activeTab;
      const btn = h('button', {
        className: `tab-button ${isActive ? 'tab-active' : ''}`,
        dataset: { tab: tabId },
        role: 'tab',
        'aria-selected': String(isActive),
      }, TAB_LABELS[tabId]);
      btn.addEventListener('click', () => this.switchTab(tabId));
      tabBar.appendChild(btn);
    }

    // Tab content area
    this.tabContent = h('div', { className: 'tab-content' });

    this.modal.appendChild(header);
    this.modal.appendChild(tabBar);
    this.modal.appendChild(this.tabContent);
    this.backdrop.appendChild(this.modal);
    document.body.appendChild(this.backdrop);

    // Initial render (loading state)
    this.renderActiveTab();

    // Activate focus trap after DOM is fully populated
    this.releaseTrap = trapFocus(this.modal);
  }

  private switchTab(tabId: TabId): void {
    this.activeTab = tabId;

    // Update tab bar active state and ARIA
    if (this.modal) {
      const buttons = this.modal.querySelectorAll('.tab-button');
      for (const btn of buttons) {
        const el = btn as HTMLElement;
        const isActive = el.dataset['tab'] === tabId;
        el.classList.toggle('tab-active', isActive);
        el.setAttribute('aria-selected', String(isActive));
      }
    }

    this.renderActiveTab();
  }

  private renderActiveTab(): void {
    if (!this.tabContent) return;
    clearChildren(this.tabContent);

    if (this.loading) {
      this.tabContent.appendChild(
        h('div', { className: 'cb-loading' }, 'Loading country data...'),
      );
      return;
    }

    // Update header risk badge now that data may be available
    this.updateRiskBadge();

    switch (this.activeTab) {
      case 'overview':
        this.renderOverviewTab();
        break;
      case 'forecasts':
        this.renderForecastsTab();
        break;
      case 'events':
        this.renderEventsTab();
        break;
      case 'risk-signals':
        this.renderRiskSignalsTab();
        break;
      case 'history':
        this.renderHistoryTab();
        break;
      case 'entities':
        this.renderEntitiesTab();
        break;
      case 'calibration':
        this.renderCalibrationTab();
        break;
    }
  }

  private updateRiskBadge(): void {
    if (!this.modal) return;
    const badge = this.modal.querySelector('.country-brief-risk-badge') as HTMLElement | null;
    if (!badge) return;

    if (this.countryRisk) {
      const score = this.countryRisk.risk_score;
      badge.textContent = `Risk: ${score}`;
      badge.classList.add(severityClass(score / 100));
    } else {
      badge.textContent = 'No risk data';
    }
  }

  // ==================================================================
  // Tab 1: Overview
  // ==================================================================

  private renderOverviewTab(): void {
    if (!this.tabContent) return;

    const grid = h('div', { className: 'overview-grid' });

    // Card 1: Risk Score with component breakdown
    const riskScore = this.countryRisk?.risk_score ?? 0;
    const trend = this.countryRisk?.trend ?? 'stable';
    const forecastRisk = this.countryRisk?.forecast_risk;
    const baselineRisk = this.countryRisk?.baseline_risk ?? 0;
    const breakdownParts: string[] = [];
    if (forecastRisk !== null && forecastRisk !== undefined) {
      breakdownParts.push(`Forecast: ${forecastRisk.toFixed(0)}`);
    }
    breakdownParts.push(`Baseline: ${baselineRisk.toFixed(0)}`);

    grid.appendChild(h('div', { className: 'overview-card' },
      h('div', { className: 'overview-card-label' }, 'RISK SCORE'),
      h('div', { className: `overview-card-value ${severityClass(riskScore / 100)}` },
        String(riskScore)),
      h('div', { className: 'overview-card-sub' },
        h('span', { className: `trend-${trend}` }, `${trendArrow(trend)} ${trend}`),
        ' -- last 7 days',
      ),
      h('div', { className: 'overview-card-breakdown' }, breakdownParts.join(' | ')),
    ));

    // Card 2: Top Forecast
    const topForecast = this.forecasts[0];
    grid.appendChild(h('div', { className: 'overview-card' },
      h('div', { className: 'overview-card-label' }, 'TOP FORECAST'),
      topForecast
        ? h('div', null,
          h('div', { className: 'overview-card-question' },
            truncate(topForecast.question, 100)),
          h('div', { className: 'overview-forecast-bar' },
            h('div', { className: 'overview-forecast-fill', style: `width: ${pctLabel(topForecast.probability)}` }),
          ),
          h('div', { className: 'overview-card-sub' },
            `${pctLabel(topForecast.probability)} probability`,
            h('span', { className: 'overview-confidence' },
              ` | confidence ${topForecast.confidence.toFixed(2)}`),
          ),
        )
        : h('div', { className: 'overview-card-empty' }, 'No active forecasts'),
    ));

    // Card 3: Recent Events (shows count from cached events if loaded)
    const eventCount = this.events !== null ? this.events.length : null;
    grid.appendChild(h('div', { className: 'overview-card' },
      h('div', { className: 'overview-card-label' }, 'RECENT EVENTS'),
      h('div', { className: 'overview-card-value' },
        eventCount !== null ? String(eventCount) : '--'),
      h('div', { className: 'overview-card-sub' },
        eventCount !== null
          ? `${eventCount} events in the last 30 days`
          : 'Switch to Events tab to load event data'),
    ));

    // Card 4: Calibration Snapshot
    const calibration = topForecast?.calibration;
    grid.appendChild(h('div', { className: 'overview-card' },
      h('div', { className: 'overview-card-label' }, 'CALIBRATION SNAPSHOT'),
      calibration
        ? h('div', null,
          h('div', { className: 'overview-card-value' },
            calibration.brier_score !== null
              ? calibration.brier_score.toFixed(4)
              : '--'),
          h('div', { className: 'overview-card-sub' },
            `Brier score | accuracy ${(calibration.historical_accuracy * 100).toFixed(0)}%`
            + ` | n=${calibration.sample_size}`),
        )
        : h('div', { className: 'overview-card-empty' },
          'Calibration data populates as predictions resolve'),
    ));

    this.tabContent.appendChild(grid);

    // Advisory summary section (unconditional -- always rendered)
    this.renderOverviewAdvisorySummary();
  }

  /**
   * Load advisories for overview tab and render summary.
   * Shares cached data with risk-signals tab.
   */
  private renderOverviewAdvisorySummary(): void {
    if (!this.tabContent || !this.currentIso) return;

    const summaryContainer = h('div', { className: 'overview-advisory-summary' },
      h('div', { className: 'section-label' }, 'TRAVEL ADVISORIES'),
    );

    if (this.advisories === null) {
      // Advisories not yet loaded -- trigger lazy load
      summaryContainer.appendChild(
        h('div', { className: 'cb-info-line' }, 'Loading advisories...'),
      );
      this.tabContent.appendChild(summaryContainer);

      // Fire lazy load in background
      this.loadAdvisoriesData().then(() => {
        if (this.activeTab === 'overview' && this.tabContent) {
          // Update only the advisory summary container
          clearChildren(summaryContainer);
          summaryContainer.appendChild(
            h('div', { className: 'section-label' }, 'TRAVEL ADVISORIES'),
          );
          this.fillAdvisorySummaryLines(summaryContainer);
        }
      }).catch(() => {
        if (this.activeTab === 'overview') {
          clearChildren(summaryContainer);
          summaryContainer.appendChild(
            h('div', { className: 'section-label' }, 'TRAVEL ADVISORIES'),
          );
          summaryContainer.appendChild(
            h('div', { className: 'cb-info-line' },
              'No government travel advisories available'),
          );
        }
      });
    } else {
      this.fillAdvisorySummaryLines(summaryContainer);
      this.tabContent.appendChild(summaryContainer);
    }
  }

  private fillAdvisorySummaryLines(container: HTMLElement): void {
    if (this.advisories && this.advisories.length > 0) {
      for (const adv of this.advisories) {
        container.appendChild(
          h('div', { className: `overview-advisory-line ${advisoryLevelClass(adv.level)}` },
            h('span', { className: 'advisory-source-label' }, `${adv.source}:`),
            h('span', null, ` Level ${adv.level} - ${adv.level_description}`),
          ),
        );
      }
    } else {
      container.appendChild(
        h('div', { className: 'cb-info-line' },
          'No government travel advisories available'),
      );
    }
  }

  // ==================================================================
  // Tab 2: Active Forecasts
  // ==================================================================

  private renderForecastsTab(): void {
    if (!this.tabContent) return;

    if (this.forecasts.length === 0) {
      this.tabContent.appendChild(
        h('div', { className: 'cb-empty-tab' }, 'No active forecasts for this country'),
      );
      return;
    }

    for (const forecast of this.forecasts) {
      const card = this.buildForecastDetailCard(forecast);
      this.tabContent.appendChild(card);
    }
  }

  private buildForecastDetailCard(forecast: ForecastResponse): HTMLElement {
    const pct = pctLabel(forecast.probability);
    const sevClass = severityClass(forecast.probability);

    const llmPct = pctLabel(forecast.ensemble_info.llm_probability);
    const tkgPct = forecast.ensemble_info.tkg_probability !== null
      ? pctLabel(forecast.ensemble_info.tkg_probability)
      : 'N/A';

    const created = new Date(forecast.created_at).toLocaleDateString('en-US', {
      year: 'numeric', month: 'short', day: 'numeric',
    });
    const expires = new Date(forecast.expires_at).toLocaleDateString('en-US', {
      year: 'numeric', month: 'short', day: 'numeric',
    });

    const card = h('div', { className: 'forecast-detail-card' },
      h('div', { className: 'forecast-detail-question' }, forecast.question),
      h('div', { className: 'forecast-detail-bar-row' },
        h('div', { className: 'probability-bar' },
          h('div', { className: `probability-fill ${sevClass}`, style: `width: ${pct}` }),
        ),
        h('span', { className: 'probability-badge' }, pct),
      ),
      h('div', { className: 'forecast-detail-meta' },
        h('span', { className: 'forecast-detail-confidence' },
          `Confidence: ${forecast.confidence.toFixed(2)}`),
        h('span', null, `Scenarios: ${forecast.scenarios.length}`),
        h('span', null, `Ensemble: LLM ${llmPct} | TKG ${tkgPct}`),
      ),
      h('div', { className: 'forecast-detail-dates' },
        `Created ${created} -- Expires ${expires}`),
    );

    // Click dispatches forecast-selected to open ScenarioExplorer
    card.addEventListener('click', () => {
      window.dispatchEvent(new CustomEvent('forecast-selected', {
        detail: { forecast },
      }));
    });

    return card;
  }

  // ==================================================================
  // Tab 3: Events (live GDELT + ACLED data)
  // ==================================================================

  private renderEventsTab(): void {
    if (!this.tabContent || !this.currentIso) return;

    // Lazy load: fetch on first activation, cache for modal session
    if (this.events === null) {
      this.tabContent.appendChild(
        h('div', { className: 'cb-loading' }, 'Loading events...'),
      );
      this.loadEventsData().catch((err) => {
        console.error('[CountryBriefPage] events load failed:', err);
        if (this.tabContent && this.activeTab === 'events') {
          clearChildren(this.tabContent);
          this.tabContent.appendChild(
            h('div', { className: 'cb-empty-tab' }, 'Failed to load event data'),
          );
        }
      });
      return;
    }

    this.renderEventsContent();
  }

  private async loadEventsData(): Promise<void> {
    if (!this.currentIso) return;
    const iso = this.currentIso;
    const result = await forecastClient.getEvents({ country: iso, limit: 50 });

    // Guard against stale responses
    if (this.currentIso !== iso) return;

    this.events = result.items;
    if (this.activeTab === 'events' && this.tabContent) {
      clearChildren(this.tabContent);
      this.renderEventsContent();
    }
  }

  private renderEventsContent(): void {
    if (!this.tabContent || !this.events) return;
    const countryName = this.currentIso ?? 'this country';

    if (this.events.length === 0) {
      this.tabContent.appendChild(
        h('div', { className: 'cb-empty-tab' },
          `No events for ${countryName} in the last 30 days`),
      );
      return;
    }

    const table = h('div', { className: 'cb-events-table' });

    for (let i = 0; i < this.events.length; i++) {
      const evt = this.events[i]!;
      const title = evt.title ?? evt.event_code ?? 'Event';
      const sevCategory = evt.quad_class !== null
        ? quadCategory(evt.quad_class)
        : (evt.event_code ? cameoCategory(evt.event_code) : 'neutral');
      const gColor = evt.goldstein_scale !== null ? goldsteinColor(evt.goldstein_scale) : 'var(--text-muted)';
      const gText = evt.goldstein_scale !== null
        ? (evt.goldstein_scale > 0 ? `+${evt.goldstein_scale.toFixed(1)}` : evt.goldstein_scale.toFixed(1))
        : '--';

      table.appendChild(h('div', { className: `cb-event-row ${i % 2 === 0 ? 'even' : 'odd'}` },
        h('span', { className: 'cb-event-cell cb-event-time' }, relativeTime(evt.event_date)),
        h('span', { className: 'cb-event-cell cb-event-desc' }, truncate(title, 80)),
        h('span', { className: `cb-event-cell cameo-badge cameo-${sevCategory}` },
          evt.quad_class !== null ? `Q${evt.quad_class}` : (evt.event_code ?? '')),
        h('span', {
          className: 'cb-event-cell cb-event-goldstein',
          style: `color: ${gColor}`,
        }, gText),
        h('span', { className: `cb-event-cell source-badge source-${evt.source.toLowerCase()}` },
          evt.source.toUpperCase()),
      ));
    }

    this.tabContent.appendChild(table);
  }

  // ==================================================================
  // Tab 4: Risk Signals (advisories + CAMEO category breakdown)
  // ==================================================================

  private renderRiskSignalsTab(): void {
    if (!this.tabContent || !this.currentIso) return;

    // Lazy load advisories on first activation
    if (this.advisories === null) {
      this.tabContent.appendChild(
        h('div', { className: 'cb-loading' }, 'Loading risk signals...'),
      );
      this.loadAdvisoriesData().then(() => {
        if (this.activeTab === 'risk-signals' && this.tabContent) {
          clearChildren(this.tabContent);
          this.renderRiskSignalsContent();
        }
      }).catch((err) => {
        console.error('[CountryBriefPage] advisories load failed:', err);
        if (this.tabContent && this.activeTab === 'risk-signals') {
          clearChildren(this.tabContent);
          this.renderRiskSignalsContent();
        }
      });
      return;
    }

    this.renderRiskSignalsContent();
  }

  private async loadAdvisoriesData(): Promise<void> {
    if (!this.currentIso) return;
    const iso = this.currentIso;
    try {
      this.advisories = await forecastClient.getAdvisories(iso);
    } catch {
      this.advisories = [];
    }
    // Guard against stale responses
    if (this.currentIso !== iso) return;
  }

  private renderRiskSignalsContent(): void {
    if (!this.tabContent) return;

    // Advisory section
    if (this.advisories && this.advisories.length > 0) {
      const advisorySection = h('div', { className: 'advisory-section' },
        h('div', { className: 'section-label' }, 'GOVERNMENT TRAVEL ADVISORIES'),
      );

      for (const adv of this.advisories) {
        const levelColor = advisoryLevelColor(adv.level);
        const levelCls = advisoryLevelClass(adv.level);
        const summaryText = adv.summary.length > 300
          ? adv.summary.slice(0, 297) + '...'
          : adv.summary;

        advisorySection.appendChild(h('div', { className: 'advisory-card' },
          h('div', { className: 'advisory-header' },
            h('span', { className: `advisory-level-badge ${levelCls}`, style: `border-color: ${levelColor}` },
              `Level ${adv.level}`),
            h('span', { className: 'advisory-source' }, adv.source),
            h('span', { className: 'advisory-level-desc' }, adv.level_description),
          ),
          h('div', { className: 'advisory-summary' }, summaryText),
          adv.url
            ? h('a', {
              className: 'advisory-link',
              href: adv.url,
              target: '_blank',
              rel: 'noopener noreferrer',
            }, 'View full advisory')
            : null,
        ));
      }

      this.tabContent.appendChild(advisorySection);
    } else {
      this.tabContent.appendChild(
        h('div', { className: 'cb-info-line' },
          'No government travel advisories available for this country.'),
      );
    }

    // Existing CAMEO category breakdown below advisories
    // Extract CAMEO-related frequency data from forecasts
    const cameoCounts = new Map<string, number>();
    for (const f of this.forecasts) {
      for (const s of f.scenarios) {
        this.extractCameoCounts(s, cameoCounts);
      }
    }

    // Build table header
    const table = h('div', { className: 'risk-signals-table' });
    table.appendChild(h('div', { className: 'cameo-category-row cameo-header' },
      h('span', { className: 'cameo-cell cameo-cell-code' }, 'CODE'),
      h('span', { className: 'cameo-cell cameo-cell-name' }, 'CATEGORY'),
      h('span', { className: 'cameo-cell cameo-cell-freq' }, 'FREQ'),
      h('span', { className: 'cameo-cell cameo-cell-goldstein' }, 'GOLDSTEIN'),
      h('span', { className: 'cameo-cell cameo-cell-trend' }, 'TREND'),
    ));

    // Sort: non-zero counts first (desc by count), then zero counts
    const sorted = [...CAMEO_CATEGORIES].sort((a, b) => {
      const ca = cameoCounts.get(a.code) ?? 0;
      const cb = cameoCounts.get(b.code) ?? 0;
      if (ca !== cb) return cb - ca;
      return parseInt(a.code) - parseInt(b.code);
    });

    for (let i = 0; i < sorted.length; i++) {
      const cat = sorted[i]!;
      const count = cameoCounts.get(cat.code) ?? 0;
      const gColor = goldsteinColor(cat.goldsteinBase);
      const gBarWidth = Math.abs(cat.goldsteinBase) * 10; // 0-100% scale
      const gSign = cat.goldsteinBase > 0 ? '+' : '';

      // Trend: stub (rising/stable/falling) -- would need historical data
      const trend = count > 2 ? 'rising' : count > 0 ? 'stable' : 'stable';
      const trendCls = `trend-${trend}`;

      table.appendChild(h('div', {
        className: `cameo-category-row ${i % 2 === 0 ? 'even' : 'odd'}`,
      },
        h('span', { className: 'cameo-cell cameo-cell-code' }, cat.code),
        h('span', { className: 'cameo-cell cameo-cell-name' }, cat.name),
        h('span', { className: 'cameo-cell cameo-cell-freq' }, String(count)),
        h('span', { className: 'cameo-cell cameo-cell-goldstein' },
          h('span', { className: 'goldstein-bar-track' },
            h('span', {
              className: 'goldstein-bar-fill',
              style: `width: ${gBarWidth}%; background: ${gColor}`,
            }),
          ),
          h('span', { style: `color: ${gColor}` }, `${gSign}${cat.goldsteinBase.toFixed(1)}`),
        ),
        h('span', { className: `cameo-cell cameo-cell-trend ${trendCls}` },
          trendArrow(trend as 'rising' | 'stable' | 'falling')),
      ));
    }

    if (cameoCounts.size === 0) {
      table.appendChild(
        h('div', { className: 'cb-empty-tab' },
          'Risk signals populate from GDELT event classification'),
      );
    }

    // Legend
    table.appendChild(h('div', { className: 'cameo-legend' },
      'Goldstein scale: -10 (conflict) to +10 (cooperation)'));

    this.tabContent.appendChild(table);
  }

  /** Recursively extract CAMEO codes from scenario evidence. */
  private extractCameoCounts(scenario: ScenarioDTO, counts: Map<string, number>): void {
    for (const ev of scenario.evidence_sources) {
      // Evidence source string may contain CAMEO codes
      const match = ev.source.match(/(\d{2})/);
      if (match) {
        const code = match[1]!;
        const catCode = CAMEO_CATEGORIES.find(c => c.code === code);
        if (catCode) {
          counts.set(code, (counts.get(code) ?? 0) + 1);
        }
      }
      // Also look in description for CAMEO references
      const descMatch = ev.description.match(/CAMEO[:\s]*(\d{2})/i);
      if (descMatch) {
        const code = descMatch[1]!;
        counts.set(code, (counts.get(code) ?? 0) + 1);
      }
    }
    for (const child of scenario.child_scenarios) {
      this.extractCameoCounts(child, counts);
    }
  }

  // ==================================================================
  // Tab 5: Forecast History
  // ==================================================================

  private renderHistoryTab(): void {
    if (!this.tabContent) return;

    if (this.forecasts.length === 0) {
      this.tabContent.appendChild(
        h('div', { className: 'cb-empty-tab' },
          'History populates as forecasts accumulate'),
      );
      return;
    }

    // SVG line chart: probability snapshots for each forecast
    const W = 600;
    const H = 250;
    const PAD = { top: 20, right: 20, bottom: 40, left: 50 };
    const PLOT_W = W - PAD.left - PAD.right;
    const PLOT_H = H - PAD.top - PAD.bottom;

    const root = svgEl('svg', {
      width: W, height: H,
      viewBox: `0 0 ${W} ${H}`,
      class: 'history-chart',
    });

    // Plot area border
    root.appendChild(svgEl('rect', {
      x: PAD.left, y: PAD.top, width: PLOT_W, height: PLOT_H,
      fill: 'none', stroke: 'var(--border)', 'stroke-width': 1,
    }));

    // Y-axis grid + labels (probability 0-1)
    for (let i = 0; i <= 5; i++) {
      const frac = i * 0.2;
      const y = PAD.top + (1 - frac) * PLOT_H;

      root.appendChild(svgEl('line', {
        x1: PAD.left, y1: y, x2: PAD.left + PLOT_W, y2: y,
        stroke: 'var(--border-subtle)', 'stroke-width': 0.5,
      }));

      const label = svgEl('text', {
        x: PAD.left - 6, y: y + 3,
        fill: 'var(--text-muted)', 'font-size': 9,
        'font-family': 'var(--font-mono)', 'text-anchor': 'end',
      });
      label.textContent = frac.toFixed(1);
      root.appendChild(label);
    }

    // X-axis label
    const xTitle = svgEl('text', {
      x: PAD.left + PLOT_W / 2, y: H - 4,
      fill: 'var(--text-muted)', 'font-size': 8,
      'font-family': 'var(--font-mono)', 'text-anchor': 'middle',
    });
    xTitle.textContent = 'FORECAST CREATED DATE';
    root.appendChild(xTitle);

    // Y-axis label
    const yTitle = svgEl('text', {
      x: 10, y: PAD.top + PLOT_H / 2,
      fill: 'var(--text-muted)', 'font-size': 8,
      'font-family': 'var(--font-mono)', 'text-anchor': 'middle',
      transform: `rotate(-90, 10, ${PAD.top + PLOT_H / 2})`,
    });
    yTitle.textContent = 'PROBABILITY';
    root.appendChild(yTitle);

    // Plot forecast probability points
    const timestamps = this.forecasts.map(f => new Date(f.created_at).getTime());
    const minTime = Math.min(...timestamps);
    const maxTime = Math.max(...timestamps);
    const timeRange = maxTime - minTime || 1;

    for (let i = 0; i < this.forecasts.length; i++) {
      const f = this.forecasts[i]!;
      const t = new Date(f.created_at).getTime();
      const x = PAD.left + ((t - minTime) / timeRange) * PLOT_W;
      const y = PAD.top + (1 - f.probability) * PLOT_H;

      root.appendChild(svgEl('circle', {
        cx: x, cy: y, r: 5,
        fill: 'var(--accent)', 'fill-opacity': 0.8,
        stroke: 'var(--accent)', 'stroke-width': 1,
      }));

      // Date label on x-axis
      if (this.forecasts.length <= 10 || i % Math.ceil(this.forecasts.length / 8) === 0) {
        const dateStr = new Date(f.created_at).toLocaleDateString('en-US', {
          month: 'short', day: 'numeric',
        });
        const dateLabel = svgEl('text', {
          x, y: PAD.top + PLOT_H + 14,
          fill: 'var(--text-muted)', 'font-size': 8,
          'font-family': 'var(--font-mono)', 'text-anchor': 'middle',
        });
        dateLabel.textContent = dateStr;
        root.appendChild(dateLabel);
      }
    }

    // If multiple points, connect with polyline
    if (this.forecasts.length > 1) {
      const sortedByTime = [...this.forecasts].sort(
        (a, b) => new Date(a.created_at).getTime() - new Date(b.created_at).getTime(),
      );
      const points = sortedByTime.map(f => {
        const t = new Date(f.created_at).getTime();
        const x = PAD.left + ((t - minTime) / timeRange) * PLOT_W;
        const y = PAD.top + (1 - f.probability) * PLOT_H;
        return `${x.toFixed(1)},${y.toFixed(1)}`;
      }).join(' ');

      root.appendChild(svgEl('polyline', {
        points,
        fill: 'none',
        stroke: 'var(--accent)',
        'stroke-width': 1.5,
        'stroke-linejoin': 'round',
        'stroke-linecap': 'round',
        'stroke-opacity': 0.5,
      }));
    }

    const chartWrapper = h('div', { className: 'history-chart-wrapper' },
      h('div', { className: 'section-label' }, 'PROBABILITY OVER TIME'),
    );
    chartWrapper.appendChild(root as unknown as Node);
    this.tabContent.appendChild(chartWrapper);

    // Scrollable past forecasts list below chart
    const listWrapper = h('div', { className: 'history-list' },
      h('div', { className: 'section-label' }, 'ALL FORECASTS'),
    );

    for (const f of this.forecasts) {
      listWrapper.appendChild(this.buildForecastDetailCard(f));
    }

    this.tabContent.appendChild(listWrapper);
  }

  // ==================================================================
  // Tab 6: Entity Relations (top actors + force graph + table toggle)
  // ==================================================================

  private renderEntitiesTab(): void {
    if (!this.tabContent || !this.currentIso) return;

    // Lazy load actor data on first activation
    if (this.actorCounts === null) {
      this.tabContent.appendChild(
        h('div', { className: 'cb-loading' }, 'Loading entity data...'),
      );
      this.loadActorData().then(() => {
        if (this.activeTab === 'entities' && this.tabContent) {
          clearChildren(this.tabContent);
          this.renderEntitiesContent();
        }
      }).catch((err) => {
        console.error('[CountryBriefPage] actor data load failed:', err);
        if (this.tabContent && this.activeTab === 'entities') {
          clearChildren(this.tabContent);
          this.renderEntitiesContent();
        }
      });
      return;
    }

    this.renderEntitiesContent();
  }

  private async loadActorData(): Promise<void> {
    if (!this.currentIso) return;
    const iso = this.currentIso;
    try {
      const result = await forecastClient.getEvents({ country: iso, limit: 200 });
      if (this.currentIso !== iso) return;

      // Aggregate actor codes client-side
      const actorMap = new Map<string, { count: number; lastSeen: string }>();
      for (const evt of result.items) {
        const actors = new Set<string>();
        if (evt.actor1_code) actors.add(evt.actor1_code);
        if (evt.actor2_code) actors.add(evt.actor2_code);

        for (const actor of actors) {
          const existing = actorMap.get(actor);
          if (!existing) {
            actorMap.set(actor, { count: 1, lastSeen: evt.event_date });
          } else {
            existing.count++;
            if (evt.event_date > existing.lastSeen) {
              existing.lastSeen = evt.event_date;
            }
          }
        }
      }

      // Sort by count descending, take top 20
      this.actorCounts = Array.from(actorMap.entries())
        .map(([actor, data]) => ({ actor, count: data.count, lastSeen: data.lastSeen }))
        .sort((a, b) => b.count - a.count)
        .slice(0, 20);
    } catch {
      this.actorCounts = [];
    }
  }

  private renderEntitiesContent(): void {
    if (!this.tabContent) return;

    // Top Actors section (from real event data)
    if (this.actorCounts && this.actorCounts.length > 0) {
      const actorSection = h('div', { className: 'top-actors-section' },
        h('div', { className: 'section-label' }, 'TOP ACTORS'),
      );

      const actorTable = h('div', { className: 'actor-table' });
      actorTable.appendChild(h('div', { className: 'actor-table-row actor-table-header' },
        h('span', { className: 'actor-table-cell actor-cell-code' }, 'ACTOR CODE'),
        h('span', { className: 'actor-table-cell actor-cell-count' }, 'EVENT COUNT'),
        h('span', { className: 'actor-table-cell actor-cell-seen' }, 'LAST SEEN'),
      ));

      for (let i = 0; i < this.actorCounts.length; i++) {
        const ac = this.actorCounts[i]!;
        actorTable.appendChild(h('div', {
          className: `actor-table-row ${i % 2 === 0 ? 'even' : 'odd'}`,
        },
          h('span', { className: 'actor-table-cell actor-cell-code' }, ac.actor),
          h('span', { className: 'actor-table-cell actor-cell-count' }, String(ac.count)),
          h('span', { className: 'actor-table-cell actor-cell-seen' }, relativeTime(ac.lastSeen)),
        ));
      }

      actorSection.appendChild(actorTable);
      this.tabContent.appendChild(actorSection);
    } else if (this.actorCounts !== null) {
      this.tabContent.appendChild(
        h('div', { className: 'cb-info-line' },
          'No actor data available for this country.'),
      );
    }

    // Existing entity co-occurrence graph from forecasts
    // Extract entity co-occurrence data from all forecasts
    const { nodes, edges } = this.extractEntityGraph();

    if (nodes.length === 0 && (!this.actorCounts || this.actorCounts.length === 0)) {
      this.tabContent.appendChild(
        h('div', { className: 'cb-empty-tab' },
          'Entity relationships populate from forecast scenarios'),
      );
      return;
    }

    if (nodes.length === 0) return;

    // View toggle
    const toggleBar = h('div', { className: 'view-toggle' });
    const graphBtn = h('button', {
      className: `view-toggle-btn ${this.entityViewMode === 'graph' ? 'active' : ''}`,
    }, 'Graph View');
    const tableBtn = h('button', {
      className: `view-toggle-btn ${this.entityViewMode === 'table' ? 'active' : ''}`,
    }, 'Table View');

    graphBtn.addEventListener('click', () => {
      this.entityViewMode = 'graph';
      this.renderEntitiesTab();
    });
    tableBtn.addEventListener('click', () => {
      this.entityViewMode = 'table';
      this.renderEntitiesTab();
    });

    toggleBar.appendChild(graphBtn);
    toggleBar.appendChild(tableBtn);
    this.tabContent.appendChild(toggleBar);

    if (this.entityViewMode === 'graph') {
      this.renderEntityGraph(nodes, edges);
    } else {
      this.renderEntityTable(edges);
    }
  }

  /** Extract entity nodes and co-occurrence edges from forecasts. */
  private extractEntityGraph(): { nodes: EntityNode[]; edges: EntityEdge[] } {
    const entityFreq = new Map<string, number>();
    const edgeMap = new Map<string, { weight: number; frequency: number }>();

    for (const f of this.forecasts) {
      for (const s of f.scenarios) {
        this.collectEntitiesFromScenario(s, entityFreq, edgeMap);
      }
    }

    const nodes: EntityNode[] = Array.from(entityFreq.entries()).map(([id, frequency]) => ({
      id,
      frequency,
    }));

    const edges: EntityEdge[] = Array.from(edgeMap.entries()).map(([key, val]) => {
      const [source, target] = key.split('||');
      return { source: source!, target: target!, weight: val.weight, frequency: val.frequency };
    });

    return { nodes, edges };
  }

  /** Recursively collect entity co-occurrences from a scenario tree. */
  private collectEntitiesFromScenario(
    scenario: ScenarioDTO,
    entityFreq: Map<string, number>,
    edgeMap: Map<string, { weight: number; frequency: number }>,
  ): void {
    const entities = scenario.entities;
    for (const e of entities) {
      entityFreq.set(e, (entityFreq.get(e) ?? 0) + 1);
    }

    // Build co-occurrence edges for all entity pairs in this scenario
    for (let i = 0; i < entities.length; i++) {
      for (let j = i + 1; j < entities.length; j++) {
        const pair = [entities[i]!, entities[j]!].sort();
        const key = `${pair[0]}||${pair[1]}`;
        const existing = edgeMap.get(key) ?? { weight: 0, frequency: 0 };
        existing.weight += scenario.probability;
        existing.frequency += 1;
        edgeMap.set(key, existing);
      }
    }

    for (const child of scenario.child_scenarios) {
      this.collectEntitiesFromScenario(child, entityFreq, edgeMap);
    }
  }

  /** Render d3-force graph view for entity relations. */
  private renderEntityGraph(nodes: EntityNode[], edges: EntityEdge[]): void {
    if (!this.tabContent) return;

    const W = 600;
    const H = 400;

    const container = h('div', { className: 'entity-graph-container' });
    const svg = document.createElementNS(SVG_NS, 'svg');
    svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
    svg.setAttribute('width', '100%');
    svg.setAttribute('height', String(H));
    svg.style.maxWidth = `${W}px`;

    // Build d3 force simulation
    const simNodes: EntityNode[] = nodes.map(n => ({ ...n }));
    const simLinks = edges.map(e => ({
      source: e.source,
      target: e.target,
      weight: e.weight,
    }));

    const simulation = d3.forceSimulation<EntityNode>(simNodes)
      .force('link', d3.forceLink<EntityNode, { source: string; target: string; weight: number }>(simLinks)
        .id(d => d.id)
        .distance(80))
      .force('charge', d3.forceManyBody().strength(-120))
      .force('center', d3.forceCenter(W / 2, H / 2))
      .force('collision', d3.forceCollide<EntityNode>().radius(d => 10 + d.frequency * 3));

    // Run simulation synchronously for layout (tick 200 iterations)
    simulation.tick(200);
    simulation.stop();

    // Draw links
    for (const link of simLinks) {
      const source = link.source as unknown as EntityNode;
      const target = link.target as unknown as EntityNode;
      if (source.x == null || source.y == null || target.x == null || target.y == null) continue;

      const line = document.createElementNS(SVG_NS, 'line');
      line.setAttribute('x1', String(source.x));
      line.setAttribute('y1', String(source.y));
      line.setAttribute('x2', String(target.x));
      line.setAttribute('y2', String(target.y));
      line.setAttribute('stroke', 'var(--border)');
      line.setAttribute('stroke-width', String(Math.max(1, link.weight * 2)));
      line.setAttribute('stroke-opacity', '0.5');
      svg.appendChild(line);
    }

    // Draw nodes
    const maxFreq = Math.max(...simNodes.map(n => n.frequency), 1);
    for (const node of simNodes) {
      if (node.x == null || node.y == null) continue;

      const g = document.createElementNS(SVG_NS, 'g');
      g.setAttribute('transform', `translate(${node.x},${node.y})`);

      const radius = 6 + (node.frequency / maxFreq) * 14;
      const circle = document.createElementNS(SVG_NS, 'circle');
      circle.setAttribute('r', String(radius));
      circle.setAttribute('fill', 'var(--accent)');
      circle.setAttribute('fill-opacity', '0.7');
      circle.setAttribute('stroke', 'var(--accent)');
      circle.setAttribute('stroke-width', '1');
      g.appendChild(circle);

      const label = document.createElementNS(SVG_NS, 'text');
      label.setAttribute('y', String(radius + 12));
      label.setAttribute('text-anchor', 'middle');
      label.setAttribute('fill', 'var(--text-secondary)');
      label.setAttribute('font-size', '9');
      label.setAttribute('font-family', 'var(--font-mono)');
      label.textContent = truncate(node.id, 20);
      g.appendChild(label);

      svg.appendChild(g);
    }

    container.appendChild(svg);
    this.tabContent.appendChild(container);
  }

  /** Render table view for entity relations. */
  private renderEntityTable(edges: EntityEdge[]): void {
    if (!this.tabContent) return;

    const sorted = [...edges].sort((a, b) => b.frequency - a.frequency);

    const table = h('div', { className: 'entity-table' });

    // Header
    table.appendChild(h('div', { className: 'entity-table-row entity-table-header' },
      h('span', { className: 'entity-table-cell entity-cell-name' }, 'ENTITY'),
      h('span', { className: 'entity-table-cell entity-cell-name' }, 'CO-OCCURS WITH'),
      h('span', { className: 'entity-table-cell entity-cell-freq' }, 'FREQ'),
      h('span', { className: 'entity-table-cell entity-cell-prob' }, 'AVG PROB'),
    ));

    for (let i = 0; i < sorted.length; i++) {
      const edge = sorted[i]!;
      const avgProb = edge.frequency > 0
        ? (edge.weight / edge.frequency).toFixed(2)
        : '0.00';

      table.appendChild(h('div', {
        className: `entity-table-row ${i % 2 === 0 ? 'even' : 'odd'}`,
      },
        h('span', { className: 'entity-table-cell entity-cell-name' }, edge.source),
        h('span', { className: 'entity-table-cell entity-cell-name' }, edge.target),
        h('span', { className: 'entity-table-cell entity-cell-freq' }, String(edge.frequency)),
        h('span', { className: 'entity-table-cell entity-cell-prob' }, avgProb),
      ));
    }

    this.tabContent.appendChild(table);
  }

  // ==================================================================
  // Tab 7: Calibration (reliability diagram + Brier decomposition)
  // ==================================================================

  private renderCalibrationTab(): void {
    if (!this.tabContent) return;

    // Collect all calibration entries from forecasts
    const calibrations: CalibrationDTO[] = this.forecasts.map(f => f.calibration);

    if (calibrations.length === 0) {
      this.tabContent.appendChild(
        h('div', { className: 'cb-empty-tab' },
          'Calibration data populates as predictions resolve'),
      );
      return;
    }

    // 1. Reliability diagram
    this.tabContent.appendChild(this.buildReliabilityDiagram(calibrations));

    // 2. Brier score decomposition table
    this.tabContent.appendChild(this.buildBrierDecomposition(calibrations));

    // 3. Per-CAMEO category accuracy
    this.tabContent.appendChild(this.buildCameoAccuracy(calibrations));
  }

  /** Reliability diagram SVG (predicted vs observed). */
  private buildReliabilityDiagram(calibrations: CalibrationDTO[]): HTMLElement {
    const W = 300;
    const H = 300;
    const PAD = 40;
    const PLOT_W = W - PAD * 2;
    const PLOT_H = H - PAD * 2;

    const root = svgEl('svg', {
      width: W, height: H,
      viewBox: `0 0 ${W} ${H}`,
      class: 'reliability-diagram',
    });

    // Plot area border
    root.appendChild(svgEl('rect', {
      x: PAD, y: PAD, width: PLOT_W, height: PLOT_H,
      fill: 'none', stroke: 'var(--border)', 'stroke-width': 1,
    }));

    // Grid lines (0.2 intervals)
    for (let i = 1; i < 5; i++) {
      const frac = i * 0.2;
      const x = PAD + frac * PLOT_W;
      const y = PAD + (1 - frac) * PLOT_H;

      root.appendChild(svgEl('line', {
        x1: x, y1: PAD, x2: x, y2: PAD + PLOT_H,
        stroke: 'var(--border-subtle)', 'stroke-width': 0.5,
      }));
      root.appendChild(svgEl('line', {
        x1: PAD, y1: y, x2: PAD + PLOT_W, y2: y,
        stroke: 'var(--border-subtle)', 'stroke-width': 0.5,
      }));
    }

    // Perfect calibration diagonal (dashed)
    root.appendChild(svgEl('line', {
      x1: PAD, y1: PAD + PLOT_H,
      x2: PAD + PLOT_W, y2: PAD,
      stroke: 'var(--text-muted)', 'stroke-width': 1,
      'stroke-dasharray': '4,3',
    }));

    // Axis labels
    const labels = ['0.0', '0.2', '0.4', '0.6', '0.8', '1.0'];
    labels.forEach((label, i) => {
      const frac = i * 0.2;
      const xLabel = svgEl('text', {
        x: PAD + frac * PLOT_W, y: H - 6,
        fill: 'var(--text-muted)', 'font-size': 9,
        'font-family': 'var(--font-mono)', 'text-anchor': 'middle',
      });
      xLabel.textContent = label;
      root.appendChild(xLabel);

      const yLabel = svgEl('text', {
        x: PAD - 6, y: PAD + (1 - frac) * PLOT_H + 3,
        fill: 'var(--text-muted)', 'font-size': 9,
        'font-family': 'var(--font-mono)', 'text-anchor': 'end',
      });
      yLabel.textContent = label;
      root.appendChild(yLabel);
    });

    // Axis titles
    const xTitle = svgEl('text', {
      x: W / 2, y: H,
      fill: 'var(--text-muted)', 'font-size': 8,
      'font-family': 'var(--font-mono)', 'text-anchor': 'middle',
    });
    xTitle.textContent = 'PREDICTED';
    root.appendChild(xTitle);

    // Data dots: temperature as predicted proxy, historical_accuracy as observed
    const maxSampleSize = Math.max(...calibrations.map(c => c.sample_size), 1);
    for (const cal of calibrations) {
      if (cal.brier_score === null) continue;

      const predicted = Math.max(0, Math.min(1, cal.temperature));
      const observed = Math.max(0, Math.min(1, cal.historical_accuracy));

      const cx = PAD + predicted * PLOT_W;
      const cy = PAD + (1 - observed) * PLOT_H;
      const radius = 3 + Math.sqrt(cal.sample_size / maxSampleSize) * 8;

      root.appendChild(svgEl('circle', {
        cx, cy, r: radius,
        fill: 'var(--accent)', 'fill-opacity': 0.7,
        stroke: 'var(--accent)', 'stroke-width': 1,
      }));
    }

    const wrapper = h('div', { className: 'reliability-diagram-wrapper' },
      h('div', { className: 'section-label' }, 'RELIABILITY DIAGRAM'),
    );
    wrapper.appendChild(root as unknown as Node);
    return wrapper;
  }

  /** Brier score decomposition table. */
  private buildBrierDecomposition(calibrations: CalibrationDTO[]): HTMLElement {
    const headerRow = h('div', { className: 'brier-row brier-header' },
      h('span', { className: 'brier-cell brier-cell-cat' }, 'CATEGORY'),
      h('span', { className: 'brier-cell brier-cell-score' }, 'BRIER'),
      h('span', { className: 'brier-cell brier-cell-n' }, 'N'),
    );

    const dataRows = calibrations.map((cal, i) => {
      const score = cal.brier_score;
      const scoreText = score !== null ? score.toFixed(4) : '--';
      const cls = score !== null ? brierClass(score) : '';

      return h('div', { className: `brier-row ${i % 2 === 0 ? 'even' : 'odd'}` },
        h('span', { className: 'brier-cell brier-cell-cat' }, cal.category),
        h('span', { className: `brier-cell brier-cell-score ${cls}` }, scoreText),
        h('span', { className: 'brier-cell brier-cell-n' }, String(cal.sample_size)),
      );
    });

    // Summary row with overall Brier score (average)
    const validScores = calibrations.filter(c => c.brier_score !== null);
    const avgBrier = validScores.length > 0
      ? validScores.reduce((sum, c) => sum + (c.brier_score ?? 0), 0) / validScores.length
      : null;
    const totalN = calibrations.reduce((sum, c) => sum + c.sample_size, 0);

    const summaryRow = h('div', { className: 'brier-row brier-summary' },
      h('span', { className: 'brier-cell brier-cell-cat' }, 'OVERALL'),
      h('span', {
        className: `brier-cell brier-cell-score ${avgBrier !== null ? brierClass(avgBrier) : ''}`,
      }, avgBrier !== null ? avgBrier.toFixed(4) : '--'),
      h('span', { className: 'brier-cell brier-cell-n' }, String(totalN)),
    );

    return h('div', { className: 'brier-table' },
      h('div', { className: 'section-label' }, 'BRIER DECOMPOSITION'),
      headerRow,
      ...dataRows,
      summaryRow,
    );
  }

  /** Per-CAMEO category accuracy table. */
  private buildCameoAccuracy(calibrations: CalibrationDTO[]): HTMLElement {
    const table = h('div', { className: 'cameo-accuracy-table' },
      h('div', { className: 'section-label' }, 'PER-CATEGORY ACCURACY'),
    );

    // Header
    table.appendChild(h('div', { className: 'brier-row brier-header' },
      h('span', { className: 'brier-cell brier-cell-cat' }, 'CATEGORY'),
      h('span', { className: 'brier-cell brier-cell-score' }, 'ACCURACY'),
      h('span', { className: 'brier-cell brier-cell-n' }, 'N'),
    ));

    for (let i = 0; i < calibrations.length; i++) {
      const cal = calibrations[i]!;
      const accuracy = (cal.historical_accuracy * 100).toFixed(1) + '%';
      const cls = cal.historical_accuracy >= 0.7 ? 'brier-excellent'
        : cal.historical_accuracy >= 0.5 ? 'brier-good'
          : 'brier-poor';

      table.appendChild(h('div', { className: `brier-row ${i % 2 === 0 ? 'even' : 'odd'}` },
        h('span', { className: 'brier-cell brier-cell-cat' }, cal.category),
        h('span', { className: `brier-cell brier-cell-score ${cls}` }, accuracy),
        h('span', { className: 'brier-cell brier-cell-n' }, String(cal.sample_size)),
      ));
    }

    return table;
  }
}
