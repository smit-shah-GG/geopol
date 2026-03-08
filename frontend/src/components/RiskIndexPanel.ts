import { Panel } from './Panel';
import { h, replaceChildren } from '@/utils/dom-utils';
import { forecastClient } from '@/services/forecast-client';
import type { CountryRiskSummary } from '@/types/api';

/** Convert ISO code to flag emoji via regional indicator symbols. */
function isoToFlag(iso: string): string {
  const upper = iso.toUpperCase();
  if (upper.length !== 2) return '';
  const a = upper.codePointAt(0);
  const b = upper.codePointAt(1);
  if (a === undefined || b === undefined) return '';
  return String.fromCodePoint(a + 0x1F1A5) + String.fromCodePoint(b + 0x1F1A5);
}

/** Map risk score to severity CSS class. */
function riskSeverity(score: number): string {
  if (score >= 80) return 'critical';
  if (score >= 60) return 'high';
  if (score >= 40) return 'elevated';
  if (score >= 20) return 'normal';
  return 'low';
}

/** Trend arrow element. */
function trendArrow(trend: CountryRiskSummary['trend']): HTMLElement {
  switch (trend) {
    case 'rising':
      return h('span', { className: 'trend-rising' }, '\u2191');
    case 'falling':
      return h('span', { className: 'trend-falling' }, '\u2193');
    case 'stable':
      return h('span', { className: 'trend-stable' }, '\u2192');
  }
}

/** Transient errors get an amber toast; persistent errors get red. */
function isTransientError(err: unknown): boolean {
  const msg = err instanceof Error ? err.message.toLowerCase() : String(err).toLowerCase();
  return /timeout|503|502|504|econnrefused|network|fetch/.test(msg);
}

/**
 * RiskIndexPanel -- per-country aggregate risk scores with trend indicators.
 *
 * Dual API: refresh() self-fetches, update() accepts pre-fetched data.
 * Dispatches 'country-selected' CustomEvent on row click.
 */
export class RiskIndexPanel extends Panel {
  /** Whether real content has been rendered at least once. */
  private hasData = false;

  constructor() {
    super({ id: 'risk-index', title: 'RISK INDEX', showCount: true });
  }

  /** Self-fetch via forecastClient. Used by RefreshScheduler. */
  public async refresh(): Promise<void> {
    if (!this.hasData) {
      this.showSkeleton();
    }
    try {
      const countries = await forecastClient.getCountries();
      this.hasData = true;
      this.dismissToast();
      const state = forecastClient.getDataState('country');
      this.setDataBadge(state.mode);
      this.renderCountries(countries);
    } catch (err: unknown) {
      if (this.isAbortError(err)) return;
      console.error('[RiskIndexPanel] refresh failed:', err);
      if (this.hasData) {
        const severity = isTransientError(err) ? 'amber' : 'red';
        this.showRefreshToast('Failed to refresh -- showing cached data', severity);
      } else {
        this.showErrorWithRetry('Unable to load risk data', () => { void this.refresh(); });
      }
    }
  }

  /** External data injection from main.ts coordinated loads. */
  public update(countries: CountryRiskSummary[]): void {
    this.renderCountries(countries);
  }

  private renderCountries(countries: CountryRiskSummary[]): void {
    this.setCount(countries.length);

    if (countries.length === 0) {
      replaceChildren(this.content,
        h('div', { className: 'empty-state-enhanced' },
          h('div', { className: 'empty-state-icon' }, '\u{1F310}'),
          h('div', { className: 'empty-state-title' }, 'No Risk Data'),
          h('div', { className: 'empty-state-desc' }, 'Country risk scores appear here once forecasts are generated and baseline risk is computed.'),
        ),
      );
      return;
    }

    const sorted = [...countries].sort((a, b) => b.risk_score - a.risk_score);
    const rows = sorted.map((c) => this.buildRow(c));
    replaceChildren(this.content, ...rows);
  }

  private buildRow(c: CountryRiskSummary): HTMLElement {
    const sev = riskSeverity(c.risk_score);
    const raw = c.top_forecast ?? '';
    const question = raw.length > 60 ? raw.slice(0, 57) + '...' : raw;

    const row = h('div', { className: 'risk-row', dataset: { iso: c.iso_code } },
      h('span', { className: 'risk-country' }, `${isoToFlag(c.iso_code)} ${c.iso_code}`),
      h('span', { className: `risk-score severity-${sev}` }, String(c.risk_score)),
      trendArrow(c.trend),
      h('span', { className: 'risk-question' }, question),
    );

    row.addEventListener('click', () => {
      // Bubbles to dashboard screen for map highlight
      this.element.dispatchEvent(
        new CustomEvent('country-selected', {
          detail: { iso: c.iso_code },
          bubbles: true,
        }),
      );
      // Global event for cross-column filtering (ForecastPanel, etc.)
      window.dispatchEvent(
        new CustomEvent('country-filter-changed', {
          detail: { iso: c.iso_code },
        }),
      );
    });

    return row;
  }
}
