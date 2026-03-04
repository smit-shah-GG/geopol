/**
 * SourceManager -- card grid of data sources with health indicators
 * and enable/disable toggles.
 *
 * Each source renders as a card with name, tier badge, health dot,
 * last-run timestamp, event count, and a toggle switch. Toggle uses
 * optimistic UI (immediate visual flip, revert on API error).
 * Auto-refreshes every 30s, updating text nodes without full re-render.
 */

import { h, clearChildren } from '@/utils/dom-utils';
import type { AdminClient } from '@/admin/admin-client';
import type { SourceInfo } from '@/admin/admin-types';
import type { AdminPanel } from '@/admin/panels/ProcessTable';

/** Format ISO datetime to relative time. */
function relativeTime(iso: string | null): string {
  if (!iso) return 'Never';
  const diff = Date.now() - new Date(iso).getTime();
  if (diff < 0) return 'now';
  const s = Math.floor(diff / 1000);
  if (s < 60) return `${s}s ago`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m}m ago`;
  const hr = Math.floor(m / 60);
  if (hr < 24) return `${hr}h ago`;
  const d = Math.floor(hr / 24);
  return `${d}d ago`;
}

export class SourceManager implements AdminPanel {
  private el: HTMLElement | null = null;
  private grid: HTMLElement | null = null;
  private intervalId: ReturnType<typeof setInterval> | null = null;
  private cards: Map<string, { el: HTMLElement; refs: CardRefs }> = new Map();

  constructor(private readonly client: AdminClient) {}

  async mount(container: HTMLElement): Promise<void> {
    this.el = h('div', { className: 'source-manager' });
    this.grid = h('div', { className: 'source-grid' });
    this.el.appendChild(this.grid);
    container.appendChild(this.el);

    await this.refresh();
    this.intervalId = setInterval(() => { void this.refresh(); }, 30_000);
  }

  destroy(): void {
    if (this.intervalId !== null) clearInterval(this.intervalId);
    this.el?.remove();
    this.el = null;
    this.grid = null;
    this.cards.clear();
  }

  private async refresh(): Promise<void> {
    if (!this.grid) return;
    try {
      const sources = await this.client.getSources();
      this.updateCards(sources);
    } catch {
      // Silently keep stale data
    }
  }

  private updateCards(sources: SourceInfo[]): void {
    if (!this.grid) return;

    // If cards don't exist yet, build them all
    if (this.cards.size === 0) {
      this.buildCards(sources);
      return;
    }

    // Otherwise, update text nodes in-place
    for (const src of sources) {
      const card = this.cards.get(src.name);
      if (!card) {
        // New source appeared -- rebuild all
        this.buildCards(sources);
        return;
      }
      this.updateCardRefs(card.refs, src);
    }
  }

  private buildCards(sources: SourceInfo[]): void {
    if (!this.grid) return;
    clearChildren(this.grid);
    this.cards.clear();

    for (const src of sources) {
      const refs = this.createCardRefs(src);
      const card = this.buildCard(src, refs);
      this.cards.set(src.name, { el: card, refs });
      this.grid.appendChild(card);
    }
  }

  private createCardRefs(src: SourceInfo): CardRefs {
    return {
      healthDot: h('span', {
        className: `status-dot ${src.healthy ? 'status-success' : 'status-failed'}`,
      }),
      lastRun: document.createTextNode(relativeTime(src.last_run)),
      eventsCount: document.createTextNode(String(src.events_count)),
      toggle: null as unknown as HTMLInputElement,
    };
  }

  private buildCard(src: SourceInfo, refs: CardRefs): HTMLElement {
    const card = h('div', { className: 'source-card' });

    // Header: name + tier badge
    const headerRow = h('div', { className: 'source-header' });
    headerRow.appendChild(h('span', { className: 'source-name' }, src.name));
    if (src.tier) {
      headerRow.appendChild(h('span', { className: 'tier-badge' }, src.tier));
    }
    card.appendChild(headerRow);

    // Health row
    const healthRow = h('div', { className: 'source-row' });
    healthRow.appendChild(h('span', { className: 'source-label' }, 'Health'));
    healthRow.appendChild(refs.healthDot);
    card.appendChild(healthRow);

    // Last run
    const lastRunRow = h('div', { className: 'source-row' });
    lastRunRow.appendChild(h('span', { className: 'source-label' }, 'Last Run'));
    const lastRunVal = h('span', { className: 'source-value' });
    lastRunVal.appendChild(refs.lastRun);
    lastRunRow.appendChild(lastRunVal);
    card.appendChild(lastRunRow);

    // Events count
    const eventsRow = h('div', { className: 'source-row' });
    eventsRow.appendChild(h('span', { className: 'source-label' }, 'Events'));
    const eventsVal = h('span', { className: 'source-value' });
    eventsVal.appendChild(refs.eventsCount);
    eventsRow.appendChild(eventsVal);
    card.appendChild(eventsRow);

    // Toggle switch
    const toggleRow = h('div', { className: 'source-row' });
    toggleRow.appendChild(h('span', { className: 'source-label' }, 'Enabled'));
    const toggleLabel = h('label', { className: 'toggle-switch' });
    const toggle = h('input', {
      type: 'checkbox',
      className: 'toggle-input',
    }) as HTMLInputElement;
    toggle.checked = src.enabled;
    toggle.addEventListener('change', () => {
      void this.handleToggle(src.name, toggle);
    });
    refs.toggle = toggle;

    const slider = h('span', { className: 'toggle-slider' });
    toggleLabel.appendChild(toggle);
    toggleLabel.appendChild(slider);
    toggleRow.appendChild(toggleLabel);
    card.appendChild(toggleRow);

    return card;
  }

  private updateCardRefs(refs: CardRefs, src: SourceInfo): void {
    // Health dot
    refs.healthDot.className = `status-dot ${src.healthy ? 'status-success' : 'status-failed'}`;
    // Last run
    refs.lastRun.textContent = relativeTime(src.last_run);
    // Events count
    refs.eventsCount.textContent = String(src.events_count);
    // Toggle (only if not mid-user-interaction)
    if (refs.toggle && document.activeElement !== refs.toggle) {
      refs.toggle.checked = src.enabled;
    }
  }

  private async handleToggle(name: string, toggle: HTMLInputElement): Promise<void> {
    const newState = toggle.checked;
    // Optimistic: already visually toggled by the checkbox
    try {
      await this.client.toggleSource(name, newState);
    } catch {
      // Revert on error
      toggle.checked = !newState;
    }
  }
}

interface CardRefs {
  healthDot: HTMLElement;
  lastRun: Text;
  eventsCount: Text;
  toggle: HTMLInputElement;
}
