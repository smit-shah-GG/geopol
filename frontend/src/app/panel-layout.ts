/**
 * Panel layout manager -- creates the CSS Grid DOM structure for the dashboard.
 *
 * Grid layout (3-column):
 *   Left column (25%):   forecasts, ensemble, calibration
 *   Center column (50%): map (spans full height)
 *   Right column (25%):  risk-index, system-health, event-timeline
 *
 * Returns a Record mapping slot names to their container elements.
 * main.ts mounts Panel.getElement() / DeckGLMap into these slots.
 */

import { h } from '@/utils/dom-utils';

/** Slot names corresponding to dashboard regions. */
export type PanelSlot =
  | 'forecasts'
  | 'ensemble'
  | 'calibration'
  | 'map'
  | 'risk-index'
  | 'system-health'
  | 'event-timeline';

export interface PanelLayoutResult {
  /** The root grid container element to mount into #app */
  grid: HTMLElement;
  /** Map from slot name to its container element */
  slots: Record<PanelSlot, HTMLElement>;
}

/**
 * Create the 3-column panel grid with named slots.
 *
 * CSS grid-template-areas:
 *   "forecasts   map  risk-index"
 *   "ensemble    map  health"
 *   "calibration map  events"
 *
 * The center column (map) spans all 3 rows.
 */
export function createPanelLayout(): PanelLayoutResult {
  const grid = h('div', { className: 'panel-grid' });

  // Left column
  const forecasts = h('div', {
    className: 'grid-slot grid-slot--forecasts',
    style: 'grid-area: forecasts;',
  });
  const ensemble = h('div', {
    className: 'grid-slot grid-slot--ensemble',
    style: 'grid-area: ensemble;',
  });
  const calibration = h('div', {
    className: 'grid-slot grid-slot--calibration',
    style: 'grid-area: calibration;',
  });

  // Center column (map spans full height)
  const map = h('div', {
    className: 'grid-slot grid-slot--map',
    style: 'grid-area: map; position: relative; overflow: hidden;',
  });

  // Right column
  const riskIndex = h('div', {
    className: 'grid-slot grid-slot--risk-index',
    style: 'grid-area: risk-index;',
  });
  const systemHealth = h('div', {
    className: 'grid-slot grid-slot--health',
    style: 'grid-area: health;',
  });
  const eventTimeline = h('div', {
    className: 'grid-slot grid-slot--events',
    style: 'grid-area: events;',
  });

  grid.appendChild(forecasts);
  grid.appendChild(ensemble);
  grid.appendChild(calibration);
  grid.appendChild(map);
  grid.appendChild(riskIndex);
  grid.appendChild(systemHealth);
  grid.appendChild(eventTimeline);

  return {
    grid,
    slots: {
      'forecasts': forecasts,
      'ensemble': ensemble,
      'calibration': calibration,
      'map': map,
      'risk-index': riskIndex,
      'system-health': systemHealth,
      'event-timeline': eventTimeline,
    },
  };
}
