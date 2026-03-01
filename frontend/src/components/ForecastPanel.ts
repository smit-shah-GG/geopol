import { Panel } from './Panel';
import { h, replaceChildren } from '@/utils/dom-utils';
import { forecastClient } from '@/services/forecast-client';
import type { ForecastResponse } from '@/types/api';

/** Severity tier derived from probability value. */
function severityClass(p: number): string {
  if (p > 0.8) return 'critical';
  if (p > 0.6) return 'high';
  if (p > 0.4) return 'elevated';
  if (p > 0.2) return 'normal';
  return 'low';
}

/** Convert ISO code to flag emoji via regional indicator symbols. */
function isoToFlag(iso: string): string {
  const upper = iso.toUpperCase();
  if (upper.length !== 2) return '';
  const a = upper.codePointAt(0);
  const b = upper.codePointAt(1);
  if (a === undefined || b === undefined) return '';
  return String.fromCodePoint(a + 0x1F1A5) + String.fromCodePoint(b + 0x1F1A5);
}

/** Relative time string from ISO timestamp. */
function relativeTime(iso: string): string {
  const diff = Date.now() - new Date(iso).getTime();
  const mins = Math.floor(diff / 60_000);
  if (mins < 1) return 'just now';
  if (mins < 60) return `${mins}m ago`;
  const hrs = Math.floor(mins / 60);
  if (hrs < 24) return `${hrs}h ago`;
  const days = Math.floor(hrs / 24);
  return `${days}d ago`;
}

/**
 * ForecastPanel -- top N active forecasts sorted by probability.
 *
 * Dual API: refresh() self-fetches, update() accepts pre-fetched data.
 * Dispatches 'forecast-selected' CustomEvent on card click.
 */
export class ForecastPanel extends Panel {
  constructor() {
    super({ id: 'forecasts', title: 'ACTIVE FORECASTS', showCount: true });
  }

  /** Self-fetch via forecastClient. Used by RefreshScheduler. */
  public async refresh(): Promise<void> {
    this.showLoading();
    try {
      const forecasts = await forecastClient.getTopForecasts(10);
      this.applyBreakerBadge('forecast');
      this.renderForecasts(forecasts);
    } catch (err: unknown) {
      if (this.isAbortError(err)) return;
      console.error('[ForecastPanel] refresh failed:', err);
      this.showError('Failed to load forecasts');
    }
  }

  /** External data injection from main.ts coordinated loads. */
  public update(forecasts: ForecastResponse[]): void {
    this.renderForecasts(forecasts);
  }

  private applyBreakerBadge(endpoint: 'forecast' | 'country' | 'health'): void {
    const state = forecastClient.getDataState(endpoint);
    this.setDataBadge(state.mode);
  }

  private renderForecasts(forecasts: ForecastResponse[]): void {
    this.setCount(forecasts.length);

    if (forecasts.length === 0) {
      replaceChildren(this.content,
        h('div', { className: 'empty-state' }, 'No active forecasts'),
      );
      return;
    }

    // API returns sorted, but guarantee descending probability
    const sorted = [...forecasts].sort((a, b) => b.probability - a.probability);

    const cards = sorted.map((f) => this.buildCard(f));
    replaceChildren(this.content, ...cards);
  }

  private buildCard(f: ForecastResponse): HTMLElement {
    const sev = severityClass(f.probability);
    const pct = `${(f.probability * 100).toFixed(1)}%`;
    const question = f.question.length > 100
      ? f.question.slice(0, 97) + '...'
      : f.question;

    // Extract country ISO from calibration category or use fallback
    const countryIso = f.calibration.category.length === 2
      ? f.calibration.category
      : '';
    const flag = countryIso ? isoToFlag(countryIso) : '';

    const card = h('div', { className: 'forecast-card', dataset: { id: f.forecast_id } },
      h('div', { className: 'forecast-question' }, question),
      h('div', { className: 'forecast-bar-row' },
        h('div', { className: 'probability-bar' },
          h('div', {
            className: `probability-fill severity-${sev}`,
            style: `width: ${f.probability * 100}%`,
          }),
        ),
        h('span', { className: 'probability-badge' }, pct),
      ),
      h('div', { className: 'forecast-meta' },
        flag ? h('span', { className: 'forecast-country' }, `${flag} ${countryIso}`) : null,
        h('span', { className: 'forecast-confidence' }, `conf: ${f.confidence.toFixed(2)}`),
        h('span', { className: 'forecast-scenarios' }, `${f.scenarios.length} scenarios`),
        h('span', { className: 'forecast-time' }, relativeTime(f.created_at)),
      ),
    );

    card.addEventListener('click', () => {
      this.element.dispatchEvent(
        new CustomEvent('forecast-selected', {
          detail: { forecast: f },
          bubbles: true,
        }),
      );
    });

    return card;
  }
}
