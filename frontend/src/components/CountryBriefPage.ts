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

import { h, clearChildren } from '@/utils/dom-utils';
import { forecastClient } from '@/services/forecast-client';
import type {
  ForecastResponse,
  CountryRiskSummary,
  PaginatedResponse,
  ScenarioDTO,
} from '@/types/api';

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
  events: 'GDELT Events',
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

// ---------------------------------------------------------------------------
// CountryBriefPage
// ---------------------------------------------------------------------------

export class CountryBriefPage {
  private backdrop: HTMLElement | null = null;
  private modal: HTMLElement | null = null;
  private tabContent: HTMLElement | null = null;
  private activeTab: TabId = 'overview';

  // Loaded data
  private currentIso: string | null = null;
  private forecasts: ForecastResponse[] = [];
  private countryRisk: CountryRiskSummary | null = null;
  private loading = false;

  // Event listeners (stored for cleanup)
  private readonly onCountrySelected: (e: Event) => void;
  private readonly onKeyDown: (e: KeyboardEvent) => void;

  constructor() {
    this.onCountrySelected = (e: Event) => {
      const detail = (e as CustomEvent<{ iso: string }>).detail;
      this.open(detail.iso);
    };

    this.onKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') this.close();
    };

    window.addEventListener('country-selected', this.onCountrySelected);
  }

  // ==================================================================
  // Public API
  // ==================================================================

  public open(iso: string): void {
    this.currentIso = iso;
    this.activeTab = 'overview';
    this.forecasts = [];
    this.countryRisk = null;
    this.loading = true;

    this.buildModal(iso);
    document.addEventListener('keydown', this.onKeyDown);
    this.loadData(iso);
  }

  public close(): void {
    document.removeEventListener('keydown', this.onKeyDown);
    if (this.backdrop) {
      this.backdrop.remove();
      this.backdrop = null;
    }
    this.modal = null;
    this.tabContent = null;
    this.currentIso = null;
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
    this.modal = h('div', { className: 'country-brief-modal' });

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
    const tabBar = h('div', { className: 'country-brief-tabs' });
    for (const tabId of TAB_ORDER) {
      const btn = h('button', {
        className: `tab-button ${tabId === this.activeTab ? 'tab-active' : ''}`,
        dataset: { tab: tabId },
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
  }

  private switchTab(tabId: TabId): void {
    this.activeTab = tabId;

    // Update tab bar active state
    if (this.modal) {
      const buttons = this.modal.querySelectorAll('.tab-button');
      for (const btn of buttons) {
        const el = btn as HTMLElement;
        el.classList.toggle('tab-active', el.dataset['tab'] === tabId);
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

    // Card 1: Risk Score
    const riskScore = this.countryRisk?.risk_score ?? 0;
    const trend = this.countryRisk?.trend ?? 'stable';
    grid.appendChild(h('div', { className: 'overview-card' },
      h('div', { className: 'overview-card-label' }, 'RISK SCORE'),
      h('div', { className: `overview-card-value ${severityClass(riskScore / 100)}` },
        String(riskScore)),
      h('div', { className: 'overview-card-sub' },
        h('span', { className: `trend-${trend}` }, `${trendArrow(trend)} ${trend}`),
        ' -- last 7 days',
      ),
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

    // Card 3: Recent Events (placeholder)
    grid.appendChild(h('div', { className: 'overview-card' },
      h('div', { className: 'overview-card-label' }, 'RECENT EVENTS'),
      h('div', { className: 'overview-card-value' }, '--'),
      h('div', { className: 'overview-card-sub' },
        'GDELT event feed -- data available when ingest daemon runs'),
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
  // Tab 3: GDELT Events (mock data placeholder)
  // ==================================================================

  private renderEventsTab(): void {
    if (!this.tabContent) return;

    this.tabContent.appendChild(
      h('div', { className: 'cb-events-notice' },
        'GDELT event data will be available when the ingest daemon is running.'),
    );

    // Mock event rows demonstrating layout
    const mockEvents = [
      { time: '14:32', actor1: 'GOV', action: 'CONSULT', actor2: 'OPP', cameo: '04', goldstein: 1.0 },
      { time: '12:15', actor1: 'MIL', action: 'EXHIBIT POSTURE', actor2: 'REB', cameo: '15', goldstein: -7.0 },
      { time: '09:48', actor1: 'CVL', action: 'PROTEST', actor2: 'GOV', cameo: '14', goldstein: -6.5 },
      { time: '08:20', actor1: 'GOV', action: 'PROVIDE AID', actor2: 'CVL', cameo: '07', goldstein: 7.0 },
      { time: 'YD 22:10', actor1: 'OPP', action: 'DEMAND', actor2: 'GOV', cameo: '10', goldstein: -3.0 },
      { time: 'YD 18:05', actor1: 'MIL', action: 'FIGHT', actor2: 'REB', cameo: '19', goldstein: -9.0 },
    ];

    const table = h('div', { className: 'cb-events-table' });
    // Header
    table.appendChild(h('div', { className: 'cb-event-row cb-event-header' },
      h('span', { className: 'cb-event-cell cb-event-time' }, 'TIME'),
      h('span', { className: 'cb-event-cell cb-event-actors' }, 'ACTORS'),
      h('span', { className: 'cb-event-cell cb-event-cameo' }, 'CAMEO'),
      h('span', { className: 'cb-event-cell cb-event-goldstein' }, 'GOLDSTEIN'),
    ));

    for (let i = 0; i < mockEvents.length; i++) {
      const ev = mockEvents[i]!;
      const gColor = goldsteinColor(ev.goldstein);
      const cameoClass = ev.goldstein > 2 ? 'cameo-cooperative'
        : ev.goldstein > -2 ? 'cameo-neutral'
          : ev.goldstein > -5 ? 'cameo-conflictual'
            : 'cameo-hostile';

      table.appendChild(h('div', { className: `cb-event-row ${i % 2 === 0 ? 'even' : 'odd'}` },
        h('span', { className: 'cb-event-cell cb-event-time' }, ev.time),
        h('span', { className: 'cb-event-cell cb-event-actors' },
          `${ev.actor1} \u2192 ${ev.action} \u2192 ${ev.actor2}`),
        h('span', { className: `cb-event-cell cameo-badge ${cameoClass}` }, ev.cameo),
        h('span', {
          className: 'cb-event-cell cb-event-goldstein',
          style: `color: ${gColor}`,
        }, ev.goldstein > 0 ? `+${ev.goldstein.toFixed(1)}` : ev.goldstein.toFixed(1)),
      ));
    }

    this.tabContent.appendChild(table);
  }

  // ==================================================================
  // Tab 4: Risk Signals (CAMEO category breakdown)
  // ==================================================================

  private renderRiskSignalsTab(): void {
    if (!this.tabContent) return;

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
  // Tab 5: Forecast History (stub -- completed in Task 2)
  // ==================================================================

  private renderHistoryTab(): void {
    if (!this.tabContent) return;
    this.tabContent.appendChild(
      h('div', { className: 'cb-empty-tab' }, 'Loading forecast history...'),
    );
  }

  // ==================================================================
  // Tab 6: Entity Relations (stub -- completed in Task 2)
  // ==================================================================

  private renderEntitiesTab(): void {
    if (!this.tabContent) return;
    this.tabContent.appendChild(
      h('div', { className: 'cb-empty-tab' }, 'Loading entity relations...'),
    );
  }

  // ==================================================================
  // Tab 7: Calibration (stub -- completed in Task 2)
  // ==================================================================

  private renderCalibrationTab(): void {
    if (!this.tabContent) return;
    this.tabContent.appendChild(
      h('div', { className: 'cb-empty-tab' }, 'Loading calibration data...'),
    );
  }
}
