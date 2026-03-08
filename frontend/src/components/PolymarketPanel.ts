import { Panel } from './Panel';
import { h, replaceChildren } from '@/utils/dom-utils';
import type { PolymarketTopResponse, PolymarketTopEvent } from '@/types/api';

/** Truncate a string to maxLen, appending ellipsis if truncated. */
function truncate(text: string, maxLen: number): string {
  return text.length > maxLen ? text.slice(0, maxLen - 1) + '\u2026' : text;
}

/** Format large numbers with K/M suffix. */
function formatVolume(vol: number): string {
  if (vol >= 1_000_000) return `${(vol / 1_000_000).toFixed(1)}M`;
  if (vol >= 1_000) return `${(vol / 1_000).toFixed(0)}K`;
  return String(Math.round(vol));
}

/**
 * PolymarketPanel -- top geopolitical prediction markets from Polymarket.
 *
 * Receives PolymarketTopResponse from coordinated loads via update().
 * Shows a ranked table of events with volume, optional Geopol match data.
 */
export class PolymarketPanel extends Panel {
  constructor() {
    super({ id: 'polymarket', title: 'POLYMARKET', showCount: true });
  }

  /** No-op. Data arrives via update(). */
  public refresh(): void {}

  /** Render the Polymarket top events table. */
  public update(data: PolymarketTopResponse): void {
    this.setCount(data.events.length);

    if (data.events.length === 0) {
      this.showPlaceholder();
      return;
    }

    const section = h('div', { className: 'polymarket-section-inner' });
    section.appendChild(this.buildTable(data.events));

    if (data.total_geo_markets > 0) {
      section.appendChild(
        h('div', { className: 'polymarket-footer' },
          `Showing top ${data.events.length} of ${data.total_geo_markets} geopolitical markets`,
        ),
      );
    }

    replaceChildren(this.content, section);
  }

  private showPlaceholder(): void {
    replaceChildren(this.content,
      h('div', { className: 'empty-state-enhanced' },
        h('div', { className: 'empty-state-icon' }, '\u{1F4C8}'),
        h('div', { className: 'empty-state-title' }, 'No Polymarket Data'),
        h('div', { className: 'empty-state-desc' }, 'Active Polymarket geopolitical questions will appear here when the poller runs.'),
      ),
    );
  }

  private buildTable(events: PolymarketTopEvent[]): HTMLElement {
    const header = h('div', { className: 'polymarket-row polymarket-header' },
      h('span', { className: 'polymarket-cell polymarket-cell-question' }, 'QUESTION'),
      h('span', { className: 'polymarket-cell polymarket-cell-vol' }, 'VOL'),
      h('span', { className: 'polymarket-cell polymarket-cell-prob' }, 'GEOPOL'),
      h('span', { className: 'polymarket-cell polymarket-cell-conf' }, 'MATCH'),
    );

    const rows = events.map((evt, i) => {
      const hasMatch = evt.geopol_probability !== null;

      const titleLink = h('a', {
        className: 'polymarket-link',
        href: `https://polymarket.com/event/${evt.slug}`,
        target: '_blank',
        rel: 'noopener noreferrer',
      }, truncate(evt.title, 60));

      return h('div', {
        className: `polymarket-row ${i % 2 === 0 ? 'even' : 'odd'}`,
      },
        h('span', { className: 'polymarket-cell polymarket-cell-question' }, titleLink),
        h('span', { className: 'polymarket-cell polymarket-cell-vol mono' },
          formatVolume(evt.volume),
        ),
        h('span', { className: `polymarket-cell polymarket-cell-prob mono ${hasMatch ? 'polymarket-matched' : ''}` },
          hasMatch ? evt.geopol_probability!.toFixed(3) : '--',
        ),
        h('span', { className: 'polymarket-cell polymarket-cell-conf mono' },
          evt.match_confidence !== null ? evt.match_confidence.toFixed(2) : '',
        ),
      );
    });

    return h('div', { className: 'polymarket-table' },
      header,
      ...rows,
    );
  }
}
