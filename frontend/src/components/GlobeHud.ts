/**
 * GlobeHud -- Minimal stats overlay for the globe screen.
 *
 * Positioned in the top-left corner, shows 3 aggregate stats:
 *   - Total active forecasts (sum of forecast_count across countries)
 *   - Number of countries with forecasts
 *   - Last data update timestamp (relative)
 *
 * Pure DOM component. No event listeners, no timers, no external state.
 * The globe screen pushes new data via update().
 */

import { h } from '@/utils/dom-utils';
import { relativeTime } from '@/components/expandable-card';
import type { CountryRiskSummary } from '@/types/api';

export class GlobeHud {
  private readonly element: HTMLElement;
  private readonly countEl: HTMLElement;
  private readonly countriesEl: HTMLElement;
  private readonly updateEl: HTMLElement;

  constructor() {
    this.countEl = h('span', { className: 'hud-value' }, '0');
    this.countriesEl = h('span', { className: 'hud-value' }, '0');
    this.updateEl = h('span', { className: 'hud-value' }, '--');

    this.element = h('div', { className: 'globe-hud' },
      h('div', { className: 'hud-item' },
        h('span', { className: 'hud-label' }, 'FORECASTS'),
        this.countEl,
      ),
      h('div', { className: 'hud-item' },
        h('span', { className: 'hud-label' }, 'COUNTRIES'),
        this.countriesEl,
      ),
      h('div', { className: 'hud-item' },
        h('span', { className: 'hud-label' }, 'UPDATED'),
        this.updateEl,
      ),
    );
  }

  /**
   * Push country risk summaries. Recomputes aggregate stats.
   * Called by globe screen on initial load and on country refresh.
   */
  update(countries: CountryRiskSummary[]): void {
    const totalForecasts = countries.reduce((sum, c) => sum + c.forecast_count, 0);
    this.countEl.textContent = String(totalForecasts);
    this.countriesEl.textContent = String(countries.length);

    if (countries.length > 0) {
      // Find most recently updated country
      const latest = countries.reduce((a, b) =>
        new Date(a.last_updated).getTime() > new Date(b.last_updated).getTime() ? a : b,
      );
      this.updateEl.textContent = relativeTime(latest.last_updated);
    } else {
      this.updateEl.textContent = '--';
    }
  }

  getElement(): HTMLElement {
    return this.element;
  }

  destroy(): void {
    // Pure DOM, no listeners to clean up
  }
}
