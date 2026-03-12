/**
 * ComparisonPanel -- all Polymarket-vs-Geopol comparisons with dual probability bars.
 *
 * Mounted in Col 2 below ForecastPanel on the dashboard screen.
 * Auto-refreshes every 5 minutes via the RefreshScheduler.
 *
 * Each entry shows a clickable collapsed header (dual GP/PM bars, badge,
 * divergence) that expands on click to reveal full forecast details via
 * the shared expandable-card system. Forecasts are lazy-fetched on first
 * expand and cached for subsequent toggles.
 */

import { Panel } from './Panel';
import { h, replaceChildren } from '@/utils/dom-utils';
import { forecastClient } from '@/services/forecast-client';
import {
  buildExpandedContent,
  truncate,
  relativeTime,
} from '@/components/expandable-card';
import type { ComparisonPanelResponse, ComparisonPanelItem, ForecastResponse } from '@/types/api';

/** Transient errors get an amber toast; persistent errors get red. */
function isTransientError(err: unknown): boolean {
  const msg = err instanceof Error ? err.message.toLowerCase() : String(err).toLowerCase();
  return /timeout|503|502|504|econnrefused|network|fetch/.test(msg);
}

/** Divergence CSS class based on absolute magnitude (active entries only). */
function divClass(divergence: number | null): string {
  if (divergence === null) return 'div-low';
  const abs = Math.abs(divergence);
  if (abs > 0.2) return 'div-high';
  if (abs > 0.1) return 'div-medium';
  return 'div-low';
}

/**
 * Directional agreement for resolved comparisons.
 * "Agrees" = geopol predicted the same side of 50% as the outcome.
 */
function marketAgreement(comp: ComparisonPanelItem): { text: string; agrees: boolean } | null {
  if (comp.status !== 'resolved' || comp.polymarket_outcome === null || comp.geopol_probability === null) return null;
  const gpYes = comp.geopol_probability >= 0.5;
  const outcomeYes = comp.polymarket_outcome >= 0.5;
  const agrees = gpYes === outcomeYes;
  return { text: agrees ? 'Market agrees' : 'Market disagrees', agrees };
}

export class ComparisonPanel extends Panel {
  /** Whether real content has been rendered at least once. */
  private hasData = false;

  /** IDs of currently expanded comparison entries. */
  private readonly expandedIds = new Set<number>();

  /** Cached ForecastResponse per geopol_prediction_id to avoid re-fetching. */
  private readonly forecastCache = new Map<string, ForecastResponse>();

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

  public override destroy(): void {
    this.forecastCache.clear();
    this.expandedIds.clear();
    super.destroy();
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
    const isExpanded = this.expandedIds.has(comp.id);

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

    // Divergence text (active) or directional agreement (resolved)
    const agreement = marketAgreement(comp);
    const divText = agreement
      ? agreement.text
      : comp.divergence !== null
        ? `${comp.divergence > 0 ? '+' : ''}${(comp.divergence * 100).toFixed(1)}pp`
        : '';

    // Build dual bars -- signature visual of this panel
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

    // Collapsed header -- clickable
    const header = h('div', {
      className: 'comparison-header',
    },
      h('div', { className: 'comparison-title' },
        h('span', {}, truncate(comp.polymarket_title, 50)),
        h('span', { className: `comparison-badge ${badgeClass}` }, badgeLabel),
      ),
      bars,
    );

    // Divergence indicator (active) or agreement indicator (resolved) in header
    if (divText) {
      const indicatorClass = agreement
        ? `comparison-agreement-indicator ${agreement.agrees ? 'agrees' : 'disagrees'}`
        : `comparison-divergence-indicator ${dc}`;
      const indicatorLabel = agreement ? divText : `Divergence: ${divText}`;
      header.appendChild(
        h('div', { className: indicatorClass }, indicatorLabel),
      );
    }

    const entry = h('div', {
      className: `comparison-entry${isResolved ? ' resolved' : ''}${isExpanded ? ' expanded' : ''}`,
      dataset: { compId: String(comp.id) },
    });

    // Keyboard and click accessibility
    header.setAttribute('role', 'button');
    header.setAttribute('tabindex', '0');
    header.setAttribute('aria-expanded', String(isExpanded));
    header.setAttribute('aria-label', `Toggle details for: ${truncate(comp.polymarket_title, 80)}`);

    const toggleExpand = (): void => {
      if (this.expandedIds.has(comp.id)) {
        this.expandedIds.delete(comp.id);
        entry.classList.remove('expanded');
        header.setAttribute('aria-expanded', 'false');
        const expandedSection = entry.querySelector('.comparison-expanded-section');
        if (expandedSection) expandedSection.remove();
      } else {
        this.expandedIds.add(comp.id);
        entry.classList.add('expanded');
        header.setAttribute('aria-expanded', 'true');
        void this.renderExpanded(comp, entry);
      }
    };

    header.addEventListener('click', (e: MouseEvent) => {
      // Don't toggle if clicking "View Full Analysis" button
      if ((e.target as HTMLElement).closest('.view-full-btn')) return;
      toggleExpand();
    });

    header.addEventListener('keydown', (e: KeyboardEvent) => {
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        if ((e.target as HTMLElement).closest('.view-full-btn')) return;
        toggleExpand();
      }
    });

    entry.appendChild(header);

    // If previously expanded, re-render expanded section
    if (isExpanded) {
      void this.renderExpanded(comp, entry);
    }

    return entry;
  }

  /**
   * Render the expanded section below the header. Lazy-fetches the full
   * ForecastResponse on first expand; subsequent toggles use the cache.
   */
  private async renderExpanded(comp: ComparisonPanelItem, entry: HTMLElement): Promise<void> {
    // Remove any existing expanded section (prevents duplicates on re-render)
    const existing = entry.querySelector('.comparison-expanded-section');
    if (existing) existing.remove();

    const section = h('div', { className: 'comparison-expanded-section' });
    entry.appendChild(section);

    const cached = this.forecastCache.get(comp.geopol_prediction_id);
    if (cached) {
      this.fillExpandedSection(section, cached, comp);
      return;
    }

    // Show loading spinner while fetching
    const spinner = h('div', { className: 'comparison-loading' },
      h('div', { className: 'submission-spinner' }),
      h('span', {}, 'Loading forecast details...'),
    );
    section.appendChild(spinner);

    try {
      const forecast = await forecastClient.getForecastById(comp.geopol_prediction_id);
      // Bail if user collapsed while fetch was in-flight
      if (!this.expandedIds.has(comp.id)) return;

      if (!forecast) {
        spinner.remove();
        section.appendChild(
          h('div', { className: 'comparison-fetch-error' }, 'Forecast not found'),
        );
        return;
      }

      this.forecastCache.set(comp.geopol_prediction_id, forecast);
      spinner.remove();
      this.fillExpandedSection(section, forecast, comp);
    } catch (err: unknown) {
      console.error('[ComparisonPanel] Failed to load forecast:', comp.geopol_prediction_id, err);
      // Bail if user collapsed while fetch was in-flight
      if (!this.expandedIds.has(comp.id)) return;
      spinner.remove();
      section.appendChild(
        h('div', { className: 'comparison-fetch-error' }, 'Failed to load forecast details'),
      );
    }
  }

  /**
   * Fill the expanded section with shared forecast content + Polymarket-specific data.
   */
  private fillExpandedSection(
    section: HTMLElement,
    forecast: ForecastResponse,
    comp: ComparisonPanelItem,
  ): void {
    const agreement = marketAgreement(comp);

    // Shared expanded content: ensemble weights, calibration, mini tree, evidence
    section.appendChild(buildExpandedContent(forecast));

    // Polymarket-specific comparison details
    const pmSection = h('div', { className: 'comparison-pm-details' },
      h('div', { className: 'expanded-section-label' }, 'Polymarket Comparison'),
      h('div', { className: 'comparison-pm-grid' },
        h('span', { className: 'comparison-pm-key' }, 'Market Price'),
        h('span', { className: 'comparison-pm-val' },
          comp.polymarket_price !== null
            ? `${(comp.polymarket_price * 100).toFixed(1)}%`
            : '--',
        ),
        h('span', { className: 'comparison-pm-key' }, agreement ? 'Direction' : 'Divergence'),
        h('span', {
          className: agreement
            ? `comparison-pm-val ${agreement.agrees ? 'agrees' : 'disagrees'}`
            : `comparison-pm-val ${divClass(comp.divergence)}`,
        },
          agreement
            ? agreement.text
            : comp.divergence !== null
              ? `${comp.divergence > 0 ? '+' : ''}${(comp.divergence * 100).toFixed(1)}pp`
              : '--',
        ),
        h('span', { className: 'comparison-pm-key' }, 'Provenance'),
        h('span', { className: 'comparison-pm-val' },
          comp.provenance === 'polymarket_driven' ? 'Market-driven' : 'Tracked',
        ),
        h('span', { className: 'comparison-pm-key' }, 'Matched'),
        h('span', { className: 'comparison-pm-val' }, relativeTime(comp.created_at)),
      ),
    );

    // Brier score for resolved entries (only show Geopol's -- PM Brier is
    // always ~0 since the market's final price IS the outcome)
    if (comp.status === 'resolved' && comp.geopol_brier !== null) {
      pmSection.appendChild(
        h('div', { className: 'comparison-pm-brier' },
          h('span', {}, `Geopol Brier: ${comp.geopol_brier.toFixed(4)}`),
        ),
      );
    }

    section.appendChild(pmSection);
  }
}
