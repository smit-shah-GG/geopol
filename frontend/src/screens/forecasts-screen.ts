/**
 * Forecasts screen -- placeholder for Phase 16 question submission UI.
 */

import { h } from '@/utils/dom-utils';
import type { GeoPolAppContext } from '@/app/app-context';

export function mountForecasts(container: HTMLElement, _ctx: GeoPolAppContext): void {
  const wrapper = h('div', {
    className: 'screen-placeholder',
  }, 'Forecast Submission -- Coming in Phase 16');

  container.appendChild(wrapper);
}

export function unmountForecasts(_ctx: GeoPolAppContext): void {
  // Nothing to clean up -- pure DOM placeholder
}
