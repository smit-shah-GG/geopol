/**
 * SourcesPanel -- auto-discovered data source health from GET /sources.
 *
 * Shows each ingestion source's health dot, name, relative time since last
 * update, events_last_run count, and detail text. Self-refreshes via
 * RefreshScheduler (60s interval) -- no longer push-fed from health response.
 *
 * Fulfills Phase 17 auto-discovered source health requirement.
 */

import { Panel } from './Panel';
import { h, replaceChildren } from '@/utils/dom-utils';
import { forecastClient } from '@/services/forecast-client';
import type { SourceStatusDTO } from '@/types/api';

/** Transient errors get an amber toast; persistent errors get red. */
function isTransientError(err: unknown): boolean {
  const msg = err instanceof Error ? err.message.toLowerCase() : String(err).toLowerCase();
  return /timeout|503|502|504|econnrefused|network|fetch/.test(msg);
}

/** Staleness thresholds in milliseconds. */
const STALE_WARN_MS = 30 * 60_000;    // 30 minutes
const STALE_CRIT_MS = 2 * 60 * 60_000; // 2 hours

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

/** CSS class for staleness based on last_update age. */
function stalenessClass(iso: string | null): string {
  if (!iso) return 'staleness-critical';
  const ageMs = Date.now() - new Date(iso).getTime();
  if (ageMs > STALE_CRIT_MS) return 'staleness-critical';
  if (ageMs > STALE_WARN_MS) return 'staleness-warning';
  return 'staleness-fresh';
}

export class SourcesPanel extends Panel {
  /** Whether real content has been rendered at least once. */
  private hasData = false;

  constructor() {
    super({ id: 'sources', title: 'DATA SOURCES', showCount: false });
  }

  /**
   * Async refresh -- called by RefreshScheduler every 60s.
   * Fetches auto-discovered source health from /sources.
   */
  public async refresh(): Promise<void> {
    if (!this.hasData) {
      this.showSkeleton();
    }
    try {
      const sources = await forecastClient.getSources();
      this.hasData = true;
      this.dismissToast();
      this.renderSources(sources);
    } catch (err: unknown) {
      if (this.isAbortError(err)) return;
      console.error('[SourcesPanel] refresh failed:', err);
      if (this.hasData) {
        const severity = isTransientError(err) ? 'amber' : 'red';
        this.showRefreshToast('Failed to refresh -- showing cached data', severity);
      } else {
        this.showErrorWithRetry('Unable to load source data', () => { void this.refresh(); });
      }
    }
  }

  private renderSources(sources: SourceStatusDTO[]): void {
    if (sources.length === 0) {
      replaceChildren(this.content,
        h('div', { className: 'empty-state-enhanced' },
          h('div', { className: 'empty-state-icon' }, '\u{1F5C4}'),
          h('div', { className: 'empty-state-title' }, 'No Sources'),
          h('div', { className: 'empty-state-desc' }, 'Data source health information will appear once the API is reachable.'),
        ),
      );
      return;
    }

    const rows = sources.map((s) => this.buildSourceRow(s));
    replaceChildren(this.content, ...rows);
  }

  private buildSourceRow(s: SourceStatusDTO): HTMLElement {
    const dotClass = s.healthy ? 'status-dot healthy' : 'status-dot unhealthy';
    const staleness = stalenessClass(s.last_update);
    const lastUpdateText = s.last_update ? relativeTime(s.last_update) : 'never';

    return h('div', { className: 'source-row' },
      h('span', { className: dotClass }),
      h('span', { className: 'source-name' }, s.name),
      h('span', { className: `source-staleness ${staleness}` }, lastUpdateText),
      s.events_last_run > 0
        ? h('span', { className: 'source-count' }, `${s.events_last_run} events`)
        : null,
      s.detail
        ? h('span', { className: 'source-detail' }, s.detail)
        : null,
    );
  }
}
