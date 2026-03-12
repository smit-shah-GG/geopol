/**
 * GlobeHud -- Minimal stats overlay + region preset selector for the globe screen.
 *
 * Positioned in the top-left corner, shows 3 aggregate stats:
 *   - Total active forecasts (sum of forecast_count across countries)
 *   - Number of countries with forecasts
 *   - Last data update timestamp (relative)
 *
 * Below the stats, 8 region preset buttons dispatch `globe-region-change`
 * CustomEvents. CesiumMap handles the camera fly-to for all scene modes.
 *
 * Pure DOM component. No timers, no external state.
 * The globe screen pushes new data via update().
 */

import { h } from '@/utils/dom-utils';
import { relativeTime } from '@/components/expandable-card';
import type { CountryRiskSummary } from '@/types/api';

// ---------------------------------------------------------------------------
// Region presets for the GlobeHud selector
// ---------------------------------------------------------------------------

const REGION_PRESETS: { id: string; label: string }[] = [
  { id: 'global', label: 'Global' },
  { id: 'america', label: 'Americas' },
  { id: 'eu', label: 'Europe' },
  { id: 'mena', label: 'MENA' },
  { id: 'asia', label: 'Asia' },
  { id: 'latam', label: 'LatAm' },
  { id: 'africa', label: 'Africa' },
  { id: 'oceania', label: 'Oceania' },
];

export { REGION_PRESETS };

export class GlobeHud {
  private readonly element: HTMLElement;
  private readonly countEl: HTMLElement;
  private readonly countriesEl: HTMLElement;
  private readonly updateEl: HTMLElement;

  constructor() {
    this.countEl = h('span', { className: 'hud-value' }, '0');
    this.countriesEl = h('span', { className: 'hud-value' }, '0');
    this.updateEl = h('span', { className: 'hud-value' }, '--');

    // Build region preset bar
    const regionBar = h('div', {
      className: 'hud-regions',
      role: 'toolbar',
      'aria-label': 'Region presets',
    });

    for (const { id, label } of REGION_PRESETS) {
      const btn = h('button', {
        className: `hud-region-btn${id === 'global' ? ' active' : ''}`,
        dataset: { region: id },
        'aria-label': `Fly to ${label}`,
      }, label);

      btn.addEventListener('click', () => {
        window.dispatchEvent(
          new CustomEvent('globe-region-change', {
            detail: { region: id },
          }),
        );
        // Update active state
        regionBar.querySelectorAll('.hud-region-btn').forEach((b) =>
          b.classList.toggle('active', b === btn),
        );
      });

      regionBar.appendChild(btn);
    }

    this.element = h('div', {
      className: 'globe-hud',
      role: 'status',
      'aria-live': 'polite',
    },
      h('div', { className: 'hud-stats' },
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
      ),
      regionBar,
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
    // Pure DOM with inline listeners -- no external cleanup needed
  }
}
