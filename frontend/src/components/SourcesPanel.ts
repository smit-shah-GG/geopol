/**
 * SourcesPanel -- data source health and staleness indicators.
 *
 * Derives from HealthResponse.subsystems, filtering to data-source-relevant
 * entries (GDELT, RSS, Polymarket, ChromaDB). Shows health dot, source name,
 * staleness indicator with color-coded relative time, and detail text.
 *
 * Updated from the existing health refresh cycle -- no independent fetch.
 * Fulfills FUX-06: data source visibility in Col 3.
 */

import { Panel } from './Panel';
import { h, replaceChildren } from '@/utils/dom-utils';
import type { HealthResponse, SubsystemStatus } from '@/types/api';

/** Subsystem name substrings that identify data sources. */
const SOURCE_KEYWORDS = ['gdelt', 'rss', 'polymarket', 'chromadb'] as const;

/** Staleness thresholds in milliseconds. */
const STALE_WARN_MS = 30 * 60_000;   // 30 minutes
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

/** CSS class for staleness based on checked_at age. */
function stalenessClass(iso: string): string {
  const ageMs = Date.now() - new Date(iso).getTime();
  if (ageMs > STALE_CRIT_MS) return 'staleness-critical';
  if (ageMs > STALE_WARN_MS) return 'staleness-warning';
  return 'staleness-fresh';
}

/** Check if a subsystem name matches one of the data source keywords. */
function isDataSource(name: string): boolean {
  const lower = name.toLowerCase();
  return SOURCE_KEYWORDS.some((kw) => lower.includes(kw));
}

export class SourcesPanel extends Panel {
  constructor() {
    super({ id: 'sources', title: 'DATA SOURCES', showCount: false });
  }

  /** External data injection from health refresh cycle. */
  public update(health: HealthResponse): void {
    this.renderSources(health);
  }

  private renderSources(health: HealthResponse): void {
    const sources = health.subsystems.filter((s) => isDataSource(s.name));

    if (sources.length === 0) {
      replaceChildren(this.content,
        h('div', { className: 'empty-state' },
          'No source data available -- check /api/v1/health',
        ),
      );
      return;
    }

    const rows = sources.map((s) => this.buildSourceRow(s));
    replaceChildren(this.content, ...rows);
  }

  private buildSourceRow(s: SubsystemStatus): HTMLElement {
    const dotClass = s.healthy ? 'status-dot healthy' : 'status-dot unhealthy';
    const staleness = stalenessClass(s.checked_at);
    const checkedAgo = relativeTime(s.checked_at);
    const detail = s.detail ?? '';

    return h('div', { className: 'source-row' },
      h('span', { className: dotClass }),
      h('span', { className: 'source-name' }, s.name),
      h('span', { className: `source-staleness ${staleness}` }, checkedAgo),
      detail ? h('span', { className: 'source-detail' }, detail) : null,
    );
  }
}
