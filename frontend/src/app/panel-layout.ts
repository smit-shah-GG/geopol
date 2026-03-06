/**
 * Dashboard 4-column flexbox layout.
 *
 * Replaces the Phase 12 3-column CSS Grid. Each column scrolls
 * independently and hosts a vertical stack of panels.
 *
 * Column assignments (Phase 21):
 *   Col 1 (25%):  NewsFeedPanel
 *   Col 2 (30%):  SearchBar, ForecastPanel, ComparisonPanel
 *   Col 3 (30%):  MyForecastsPanel
 *   Col 4 (15%):  RiskIndexPanel, SystemHealth, Polymarket
 */

import { h } from '@/utils/dom-utils';

export type DashboardColumn = 'col1' | 'col2' | 'col3' | 'col4';

export interface DashboardLayoutResult {
  /** The outer flex container to mount into the screen container */
  element: HTMLElement;
  /** Map from column key to its scrollable container element */
  columns: Record<DashboardColumn, HTMLElement>;
}

/**
 * Create the 4-column flexbox layout.
 * Column widths are set via CSS classes (.dashboard-col--N).
 */
export function createDashboardLayout(): DashboardLayoutResult {
  const col1 = h('div', { className: 'dashboard-col dashboard-col--1' });
  const col2 = h('div', { className: 'dashboard-col dashboard-col--2' });
  const col3 = h('div', { className: 'dashboard-col dashboard-col--3' });
  const col4 = h('div', { className: 'dashboard-col dashboard-col--4' });

  const container = h('div', { className: 'dashboard-columns' },
    col1, col2, col3, col4,
  );

  return {
    element: container,
    columns: { col1, col2, col3, col4 },
  };
}
