/**
 * Skeleton shimmer placeholder builder.
 *
 * Generates DOM elements matching panel-specific data shapes. Each skeleton
 * container carries role="status" and aria-busy="true" so assistive tech
 * announces the loading state. A visually-hidden "Loading..." span provides
 * the announcement text.
 *
 * CSS shimmer animation is defined in panels.css (@keyframes shimmer).
 * prefers-reduced-motion disables shimmer in main.css.
 */

import { h } from '@/utils/dom-utils';

// -----------------------------------------------------------------------
// Types
// -----------------------------------------------------------------------

export type SkeletonShape =
  | 'card-list'
  | 'row-list'
  | 'bar-pairs'
  | 'text-block'
  | 'news-feed'
  | 'health-grid'
  | 'timeline';

/** Maps panel IDs to their preferred skeleton shape. */
export const PANEL_SKELETON_MAP: Record<string, SkeletonShape> = {
  'forecasts': 'card-list',
  'risk-index': 'row-list',
  'news-feed': 'news-feed',
  'system-health': 'health-grid',
  'polymarket': 'card-list',
  'my-forecasts': 'card-list',
  'comparisons': 'bar-pairs',
  'event-timeline': 'timeline',
  'live-streams': 'text-block',
  'sources': 'row-list',
};

// -----------------------------------------------------------------------
// Shape builders (each returns an array of child elements)
// -----------------------------------------------------------------------

function buildCardList(): HTMLElement[] {
  const cards: HTMLElement[] = [];
  for (let i = 0; i < 3; i++) {
    cards.push(
      h('div', { className: 'skeleton-card' },
        h('div', { className: 'skeleton-line' }),
        h('div', { className: 'skeleton-bar' }),
      ),
    );
  }
  return cards;
}

function buildRowList(): HTMLElement[] {
  const rows: HTMLElement[] = [];
  for (let i = 0; i < 6; i++) {
    rows.push(h('div', { className: 'skeleton-row' }));
  }
  return rows;
}

function buildBarPairs(): HTMLElement[] {
  const pairs: HTMLElement[] = [];
  for (let i = 0; i < 3; i++) {
    pairs.push(h('div', { className: 'skeleton-bar-pair' }));
  }
  return pairs;
}

function buildTextBlock(): HTMLElement[] {
  const widths = ['80%', '100%', '60%', '90%'];
  return widths.map((w) => {
    const line = h('div', { className: 'skeleton-line' });
    line.style.width = w;
    return line;
  });
}

function buildNewsFeed(): HTMLElement[] {
  const articles: HTMLElement[] = [];
  for (let i = 0; i < 5; i++) {
    articles.push(
      h('div', { className: 'skeleton-article' },
        h('div', { className: 'skeleton-line' }),
        h('div', { className: 'skeleton-line short' }),
        h('div', { className: 'skeleton-line short' }),
      ),
    );
  }
  return articles;
}

function buildTimeline(): HTMLElement[] {
  const items: HTMLElement[] = [];
  for (let i = 0; i < 4; i++) {
    items.push(
      h('div', { className: 'skeleton-timeline-item' },
        h('div', { className: 'skeleton-line short' }),
        h('div', { className: 'skeleton-line' }),
      ),
    );
  }
  return items;
}

function buildHealthGrid(): HTMLElement[] {
  const wideLine = h('div', { className: 'skeleton-line wide' });
  const rows: HTMLElement[] = [wideLine];
  for (let i = 0; i < 4; i++) {
    rows.push(h('div', { className: 'skeleton-row' }));
  }
  return rows;
}

// -----------------------------------------------------------------------
// Shape dispatch
// -----------------------------------------------------------------------

const SHAPE_BUILDERS: Record<SkeletonShape, () => HTMLElement[]> = {
  'card-list': buildCardList,
  'row-list': buildRowList,
  'bar-pairs': buildBarPairs,
  'text-block': buildTextBlock,
  'news-feed': buildNewsFeed,
  'health-grid': buildHealthGrid,
  'timeline': buildTimeline,
};

// -----------------------------------------------------------------------
// Public API
// -----------------------------------------------------------------------

/**
 * Build a shimmer skeleton placeholder for the given shape.
 *
 * Returns a `div.panel-skeleton` with `role="status"`, `aria-busy="true"`,
 * and a visually-hidden "Loading..." span for screen readers.
 */
export function buildSkeleton(shape: SkeletonShape): HTMLElement {
  const srOnly = h('span', { className: 'sr-only' }, 'Loading...');
  const children = SHAPE_BUILDERS[shape]();

  const container = h('div', { className: 'panel-skeleton', role: 'status' }, srOnly, ...children);
  container.setAttribute('aria-busy', 'true');
  return container;
}
