import { Panel } from './Panel';
import { h, replaceChildren } from '@/utils/dom-utils';

/** Mock GDELT event for layout demonstration. */
interface MockEvent {
  timestamp: string;
  description: string;
  cameo_code: string;
  goldstein: number;
  country: string;
}

/** CAMEO code category color class. */
function cameoCategory(code: string): string {
  const prefix = parseInt(code.slice(0, 2), 10);
  if (Number.isNaN(prefix)) return 'neutral';
  if (prefix <= 5) return 'cooperative';   // Material/verbal cooperation
  if (prefix <= 9) return 'neutral';       // Neutral actions
  if (prefix <= 14) return 'conflictual';  // Verbal conflict
  return 'hostile';                        // Material conflict (15-20)
}

/** Goldstein scale indicator: positive = green, negative = red, zero = gray. */
function goldsteinIndicator(value: number): HTMLElement {
  const cls = value > 0 ? 'goldstein-pos' : value < 0 ? 'goldstein-neg' : 'goldstein-zero';
  return h('span', { className: `goldstein-indicator ${cls}` }, value.toFixed(1));
}

const MOCK_EVENTS: MockEvent[] = [
  {
    timestamp: new Date(Date.now() - 15 * 60_000).toISOString(),
    description: 'Head of state makes diplomatic statement',
    cameo_code: '0231',
    goldstein: 3.4,
    country: 'US',
  },
  {
    timestamp: new Date(Date.now() - 45 * 60_000).toISOString(),
    description: 'Military forces mobilize near border region',
    cameo_code: '1511',
    goldstein: -7.2,
    country: 'RU',
  },
  {
    timestamp: new Date(Date.now() - 2 * 3_600_000).toISOString(),
    description: 'Trade agreement signed between nations',
    cameo_code: '0256',
    goldstein: 6.0,
    country: 'CN',
  },
  {
    timestamp: new Date(Date.now() - 5 * 3_600_000).toISOString(),
    description: 'Protest activity reported in capital city',
    cameo_code: '1411',
    goldstein: -5.6,
    country: 'IR',
  },
  {
    timestamp: new Date(Date.now() - 8 * 3_600_000).toISOString(),
    description: 'Humanitarian aid convoy dispatched',
    cameo_code: '0711',
    goldstein: 7.4,
    country: 'UA',
  },
  {
    timestamp: new Date(Date.now() - 12 * 3_600_000).toISOString(),
    description: 'Sanctions imposed on government officials',
    cameo_code: '1630',
    goldstein: -8.0,
    country: 'KP',
  },
];

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

/**
 * EventTimelinePanel -- GDELT event feed.
 *
 * Currently renders mock data; no dedicated API endpoint exists yet.
 * Dual API: refresh() shows mock data, update() accepts future event injection.
 */
export class EventTimelinePanel extends Panel {
  constructor() {
    super({ id: 'event-timeline', title: 'GDELT EVENTS', showCount: true });
  }

  /** Render mock event timeline. Real data requires a dedicated events endpoint. */
  public refresh(): void {
    this.renderEvents(MOCK_EVENTS);
  }

  /** Placeholder for future event data injection. */
  public update(events: unknown[]): void {
    if (events.length === 0) {
      this.renderEvents(MOCK_EVENTS);
    }
    // Future: render real events when endpoint is available
  }

  private renderEvents(events: MockEvent[]): void {
    this.setCount(events.length);

    const notice = h('div', { className: 'event-notice' },
      'GDELT event feed connects when ingest daemon is running',
    );

    const rows = events.map((evt, i) => {
      const cat = cameoCategory(evt.cameo_code);
      return h('div', { className: `event-row ${i % 2 === 0 ? 'even' : 'odd'}` },
        h('span', { className: 'event-time' }, relativeTime(evt.timestamp)),
        h('span', { className: 'event-desc' }, evt.description),
        h('span', { className: `cameo-badge cameo-${cat}` }, evt.cameo_code),
        goldsteinIndicator(evt.goldstein),
        h('span', { className: 'event-country' }, evt.country),
      );
    });

    replaceChildren(this.content, notice, ...rows);
  }
}
