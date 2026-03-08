/**
 * ComparisonPanel -- all Polymarket-vs-Geopol comparisons with dual probability bars.
 *
 * Mounted in Col 2 below ForecastPanel on the dashboard screen.
 * Auto-refreshes every 5 minutes via the RefreshScheduler.
 *
 * Each entry shows:
 *   - Truncated Polymarket question + provenance/status badge
 *   - Dual horizontal bars (GP vs PM) color-coded by divergence magnitude
 *   - Resolved entries: reduced opacity + winner indicator
 */

import { Panel } from './Panel';
import { h, replaceChildren } from '@/utils/dom-utils';
import { forecastClient } from '@/services/forecast-client';
import type { ComparisonPanelResponse, ComparisonPanelItem } from '@/types/api';

/** Truncate text with ellipsis. */
function truncate(text: string, maxLen: number): string {
  return text.length > maxLen ? text.slice(0, maxLen - 1) + '\u2026' : text;
}

/** Transient errors get an amber toast; persistent errors get red. */
function isTransientError(err: unknown): boolean {
  const msg = err instanceof Error ? err.message.toLowerCase() : String(err).toLowerCase();
  return /timeout|503|502|504|econnrefused|network|fetch/.test(msg);
}

/** Divergence CSS class based on absolute magnitude. */
function divClass(divergence: number | null): string {
  if (divergence === null) return 'div-low';
  const abs = Math.abs(divergence);
  if (abs > 0.2) return 'div-high';
  if (abs > 0.1) return 'div-medium';
  return 'div-low';
}

export class ComparisonPanel extends Panel {
  /** Whether real content has been rendered at least once. */
  private hasData = false;

  constructor() {
    super({ id: 'comparisons', title: 'POLYMARKET COMPARISONS', showCount: true });
  }

  public async refresh(): Promise<void> {
    if (!this.hasData) {
      this.showSkeleton();
    }
    try {
      const data = await forecastClient.getComparisons();
      this.hasData = true;
      this.dismissToast();
      this.update(data);
    } catch (err: unknown) {
      if (this.isAbortError(err)) return;
      console.error('[ComparisonPanel] refresh failed:', err);
      if (this.hasData) {
        const severity = isTransientError(err) ? 'amber' : 'red';
        this.showRefreshToast('Failed to refresh -- showing cached data', severity);
      } else {
        this.showErrorWithRetry('Unable to load comparisons', () => { void this.refresh(); });
      }
    }
  }

  public update(data: ComparisonPanelResponse): void {
    this.hasData = true;
    this.setCount(data.total);

    if (data.comparisons.length === 0) {
      this.showEmpty();
      return;
    }

    const entries = data.comparisons.map(comp => this.buildEntry(comp));
    replaceChildren(this.content, ...entries);
  }

  private showEmpty(): void {
    replaceChildren(this.content,
      h('div', { className: 'empty-state-enhanced' },
        h('div', { className: 'empty-state-icon' }, '\u2696'),
        h('div', { className: 'empty-state-title' }, 'No Comparisons Yet'),
        h('div', { className: 'empty-state-desc' }, 'Polymarket comparisons appear automatically when active prediction markets match Geopol forecasts.'),
      ),
    );
  }

  private buildEntry(comp: ComparisonPanelItem): HTMLElement {
    const isResolved = comp.status === 'resolved';
    const dc = divClass(comp.divergence);

    // Badge label and class
    let badgeLabel: string;
    let badgeClass: string;
    if (isResolved) {
      badgeLabel = 'RESOLVED';
      badgeClass = 'badge-resolved';
    } else if (comp.provenance === 'polymarket_driven') {
      badgeLabel = 'DRIVEN';
      badgeClass = 'badge-driven';
    } else {
      badgeLabel = 'TRACKED';
      badgeClass = 'badge-tracked';
    }

    // GP and PM percentage values
    const gpPct = comp.geopol_probability !== null
      ? Math.round(comp.geopol_probability * 100)
      : null;
    const pmPct = comp.polymarket_price !== null
      ? Math.round(comp.polymarket_price * 100)
      : null;

    // Build bars
    const bars = h('div', { className: 'comparison-bars' },
      h('div', { className: 'bar-row' },
        h('span', { className: 'bar-label' }, 'GP'),
        h('div', { className: 'bar-track' },
          h('div', {
            className: `bar-fill bar-geopol ${dc}`,
            style: `width: ${gpPct ?? 0}%`,
          }),
        ),
        h('span', { className: 'bar-value mono' }, gpPct !== null ? `${gpPct}%` : '--'),
      ),
      h('div', { className: 'bar-row' },
        h('span', { className: 'bar-label' }, 'PM'),
        h('div', { className: 'bar-track' },
          h('div', {
            className: `bar-fill bar-polymarket ${dc}`,
            style: `width: ${pmPct ?? 0}%`,
          }),
        ),
        h('span', { className: 'bar-value mono' }, pmPct !== null ? `${pmPct}%` : '--'),
      ),
    );

    // Winner indicator for resolved entries
    let winnerEl: HTMLElement | null = null;
    if (isResolved && comp.geopol_brier !== null && comp.polymarket_brier !== null) {
      const geopWins = comp.geopol_brier < comp.polymarket_brier;
      winnerEl = h('div', { className: `comparison-winner ${geopWins ? 'winner-geopol' : 'winner-market'}` },
        geopWins ? 'Geopol closer' : 'Market closer',
      );
    }

    const entry = h('div', { className: `comparison-entry${isResolved ? ' resolved' : ''}` },
      h('div', { className: 'comparison-title' },
        h('span', {}, truncate(comp.polymarket_title, 50)),
        h('span', { className: `comparison-badge ${badgeClass}` }, badgeLabel),
      ),
      bars,
    );

    if (winnerEl) entry.appendChild(winnerEl);

    return entry;
  }
}
