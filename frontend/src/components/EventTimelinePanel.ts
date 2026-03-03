/**
 * EventTimelinePanel -- live GDELT + ACLED event feed with diff-based DOM updates.
 *
 * Fetches real events from GET /events via forecastClient.getEvents().
 * Compact timeline rows: relative timestamp + title + country flag + severity badge + source.
 * Click-to-expand shows full event detail (actors, CAMEO, Goldstein, source URL).
 * Diff-based update preserves expanded card state across 30s refresh cycles.
 *
 * Fulfills Phase 17 live data wiring requirement.
 */

import { Panel } from './Panel';
import { h, clearChildren, replaceChildren } from '@/utils/dom-utils';
import { forecastClient } from '@/services/forecast-client';
import type { EventDTO } from '@/types/api';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** ISO alpha-2 to regional indicator flag emoji. */
function isoToFlag(iso: string): string {
  return iso
    .toUpperCase()
    .split('')
    .map((c) => String.fromCodePoint(0x1f1e6 + c.charCodeAt(0) - 65))
    .join('');
}

/** CAMEO quad_class to severity category. */
function quadCategory(quadClass: number | null): string {
  if (quadClass === null) return 'neutral';
  if (quadClass <= 2) return 'cooperative';   // Verbal/material cooperation
  if (quadClass === 3) return 'neutral';      // Verbal conflict (amber)
  return 'conflictual';                       // Material conflict (4)
}

/** CAMEO code category for event_code string. */
function cameoCategory(code: string): string {
  const prefix = parseInt(code.slice(0, 2), 10);
  if (Number.isNaN(prefix)) return 'neutral';
  if (prefix <= 5) return 'cooperative';
  if (prefix <= 9) return 'neutral';
  if (prefix <= 14) return 'conflictual';
  return 'hostile';
}

/** Goldstein scale indicator element. */
function goldsteinIndicator(value: number): HTMLElement {
  const cls = value > 0 ? 'goldstein-pos' : value < 0 ? 'goldstein-neg' : 'goldstein-zero';
  const sign = value > 0 ? '+' : '';
  return h('span', { className: `goldstein-indicator ${cls}` }, `${sign}${value.toFixed(1)}`);
}

/** Relative time string from ISO timestamp. */
function relativeTime(iso: string): string {
  const diff = Date.now() - new Date(iso).getTime();
  const mins = Math.floor(diff / 60_000);
  if (mins < 1) return 'now';
  if (mins < 60) return `${mins}m`;
  const hrs = Math.floor(mins / 60);
  if (hrs < 24) return `${hrs}h`;
  const days = Math.floor(hrs / 24);
  return `${days}d`;
}

/** Truncate text with ellipsis. */
function truncateText(text: string, maxLen: number): string {
  return text.length > maxLen ? text.slice(0, maxLen - 1) + '\u2026' : text;
}

// ---------------------------------------------------------------------------
// EventTimelinePanel
// ---------------------------------------------------------------------------

export class EventTimelinePanel extends Panel {
  /** Currently displayed events (for diff comparison). */
  private currentEvents: EventDTO[] = [];

  /** Map of event.id -> DOM element for diff-based updates. */
  private eventElements = new Map<number, HTMLElement>();

  /** Currently expanded event ID (only one at a time). */
  private expandedId: number | null = null;

  constructor() {
    super({ id: 'event-timeline', title: 'EVENT FEED', showCount: true });
  }

  /**
   * Async refresh -- called by RefreshScheduler every 30s.
   * Fetches live events from /events API, then diff-updates the DOM.
   */
  public async refresh(): Promise<void> {
    try {
      const result = await forecastClient.getEvents({ limit: 50 });
      const state = forecastClient.getDataState('events');
      this.setDataBadge(state.mode);
      this.updateEvents(result.items);
    } catch (err: unknown) {
      if (this.isAbortError(err)) return;
      console.error('[EventTimelinePanel] refresh failed:', err);
      // Only show error if we have no data at all
      if (this.currentEvents.length === 0) {
        this.showError('Failed to load events');
      }
    }
  }

  // -----------------------------------------------------------------------
  // Diff-based DOM update (follows ForecastPanel pattern)
  // -----------------------------------------------------------------------

  private updateEvents(events: EventDTO[]): void {
    // Sort by event_date descending (newest first)
    const sorted = [...events].sort(
      (a, b) => new Date(b.event_date).getTime() - new Date(a.event_date).getTime(),
    );

    this.currentEvents = sorted;
    this.setCount(sorted.length);

    if (sorted.length === 0) {
      this.eventElements.clear();
      this.expandedId = null;
      replaceChildren(this.content,
        h('div', { className: 'empty-state' },
          'No events in the last 30 days'),
      );
      return;
    }

    const newIds = new Set(sorted.map(e => e.id));
    const existingIds = new Set(this.eventElements.keys());

    // Remove departed events
    for (const id of existingIds) {
      if (!newIds.has(id)) {
        const el = this.eventElements.get(id);
        el?.remove();
        this.eventElements.delete(id);
        if (this.expandedId === id) this.expandedId = null;
      }
    }

    // Build ordered list: preserve existing, create new
    const orderedCards: HTMLElement[] = [];
    for (const evt of sorted) {
      const existing = this.eventElements.get(evt.id);
      if (existing) {
        // Update compact row data in-place (time may have changed)
        this.updateRowInPlace(existing, evt);
        orderedCards.push(existing);
      } else {
        const row = this.buildEventRow(evt);
        this.eventElements.set(evt.id, row);
        orderedCards.push(row);
      }
    }

    // Reorder DOM to match sorted order
    const frag = document.createDocumentFragment();
    for (const card of orderedCards) {
      frag.appendChild(card);
    }
    clearChildren(this.content);
    this.content.appendChild(frag);
  }

  // -----------------------------------------------------------------------
  // Row construction
  // -----------------------------------------------------------------------

  private buildEventRow(evt: EventDTO): HTMLElement {
    const isExpanded = this.expandedId === evt.id;
    const sevCategory = evt.quad_class !== null
      ? quadCategory(evt.quad_class)
      : (evt.event_code ? cameoCategory(evt.event_code) : 'neutral');

    const title = evt.title ?? evt.event_code ?? 'Event';
    const countryFlag = evt.country_iso ? isoToFlag(evt.country_iso) : '';

    const row = h('div', {
      className: `event-row ${isExpanded ? 'expanded' : ''}`,
      dataset: { eventId: String(evt.id) },
    },
      // Compact row
      h('div', { className: 'event-row-compact' },
        h('span', { className: 'event-time' }, relativeTime(evt.event_date)),
        h('span', { className: 'event-desc' }, truncateText(title, 80)),
        countryFlag
          ? h('span', { className: 'event-country-flag' }, countryFlag)
          : null,
        h('span', { className: `cameo-badge cameo-${sevCategory}` },
          evt.quad_class !== null ? `Q${evt.quad_class}` : (evt.event_code ?? '')),
        h('span', { className: `source-badge source-${evt.source.toLowerCase()}` }, evt.source.toUpperCase()),
      ),
    );

    // Click handler for expand/collapse
    const compactRow = row.querySelector('.event-row-compact') as HTMLElement;
    compactRow.addEventListener('click', () => this.toggleExpand(evt, row));

    // If this event was previously expanded, re-expand it
    if (isExpanded) {
      row.appendChild(this.buildExpandedContent(evt));
    }

    return row;
  }

  private buildExpandedContent(evt: EventDTO): HTMLElement {
    const actor1 = evt.actor1_code ?? 'Unknown';
    const actor2 = evt.actor2_code ?? 'Unknown';
    const cameoCode = evt.event_code ?? '--';
    const goldstein = evt.goldstein_scale;

    const detail = h('div', { className: 'event-expanded-content' },
      h('div', { className: 'event-detail-row' },
        h('span', { className: 'event-detail-label' }, 'Actors'),
        h('span', { className: 'event-detail-value' }, `${actor1} vs ${actor2}`),
      ),
      h('div', { className: 'event-detail-row' },
        h('span', { className: 'event-detail-label' }, 'CAMEO Code'),
        h('span', { className: 'event-detail-value' }, cameoCode),
      ),
      goldstein !== null
        ? h('div', { className: 'event-detail-row' },
          h('span', { className: 'event-detail-label' }, 'Goldstein'),
          goldsteinIndicator(goldstein),
        )
        : null,
      h('div', { className: 'event-detail-row' },
        h('span', { className: 'event-detail-label' }, 'Source'),
        h('span', { className: 'event-detail-value' }, evt.source.toUpperCase()),
      ),
      h('div', { className: 'event-detail-row' },
        h('span', { className: 'event-detail-label' }, 'Date'),
        h('span', { className: 'event-detail-value' },
          new Date(evt.event_date).toLocaleDateString('en-US', {
            year: 'numeric', month: 'short', day: 'numeric',
          })),
      ),
      evt.num_mentions !== null
        ? h('div', { className: 'event-detail-row' },
          h('span', { className: 'event-detail-label' }, 'Mentions'),
          h('span', { className: 'event-detail-value' }, String(evt.num_mentions)),
        )
        : null,
      evt.country_iso
        ? h('div', { className: 'event-detail-row' },
          h('span', { className: 'event-detail-label' }, 'Country'),
          h('span', { className: 'event-detail-value' },
            `${isoToFlag(evt.country_iso)} ${evt.country_iso}`),
        )
        : null,
      evt.url
        ? h('div', { className: 'event-detail-row' },
          h('a', {
            className: 'event-source-link',
            href: evt.url,
            target: '_blank',
            rel: 'noopener noreferrer',
          }, 'View source article'),
        )
        : null,
    );

    return detail;
  }

  // -----------------------------------------------------------------------
  // Expand/collapse
  // -----------------------------------------------------------------------

  private toggleExpand(evt: EventDTO, row: HTMLElement): void {
    if (this.expandedId === evt.id) {
      // Collapse current
      this.expandedId = null;
      row.classList.remove('expanded');
      const expanded = row.querySelector('.event-expanded-content');
      expanded?.remove();
    } else {
      // Collapse previous if any
      if (this.expandedId !== null) {
        const prevEl = this.eventElements.get(this.expandedId);
        if (prevEl) {
          prevEl.classList.remove('expanded');
          const prevExpanded = prevEl.querySelector('.event-expanded-content');
          prevExpanded?.remove();
        }
      }
      // Expand this one
      this.expandedId = evt.id;
      row.classList.add('expanded');
      row.appendChild(this.buildExpandedContent(evt));
    }
  }

  // -----------------------------------------------------------------------
  // In-place update (preserves expanded state across refresh)
  // -----------------------------------------------------------------------

  private updateRowInPlace(row: HTMLElement, evt: EventDTO): void {
    // Update the compact row's time (relative time changes over time)
    const timeEl = row.querySelector('.event-time') as HTMLElement | null;
    if (timeEl) {
      timeEl.textContent = relativeTime(evt.event_date);
    }

    // Store latest data on dataset for re-expand
    row.dataset['eventId'] = String(evt.id);
  }
}
