/**
 * LogViewer -- real-time log tail with severity pills, subsystem filter,
 * text search, and auto-scroll with pause-on-scroll-up.
 *
 * Polls adminClient.getLogs() every 3s with server-side severity/subsystem
 * filters. Client-side additionally filters by pill toggle state and search
 * text. Diff-based: only appends truly new entries (by timestamp+message).
 */

import { h } from '@/utils/dom-utils';
import type { AdminClient } from '@/admin/admin-client';
import type { LogEntry } from '@/admin/admin-types';
import type { AdminPanel } from '@/admin/panels/ProcessTable';

type Severity = 'ERROR' | 'WARN' | 'INFO';

const SEVERITY_COLORS: Record<Severity, string> = {
  ERROR: '#ef4444',
  WARN: '#f59e0b',
  INFO: '#888',
};

/** Dedupe key for a log entry. */
function entryKey(e: LogEntry): string {
  return `${e.timestamp}|${e.severity}|${e.module}|${e.message}`;
}

export class LogViewer implements AdminPanel {
  private el: HTMLElement | null = null;
  private logContainer: HTMLElement | null = null;
  private intervalId: ReturnType<typeof setInterval> | null = null;
  private autoScroll = true;
  private resumeBtn: HTMLElement | null = null;

  // Filter state
  private severityOn: Record<Severity, boolean> = { ERROR: true, WARN: true, INFO: true };
  private subsystemFilter: string | null = null;
  private searchText = '';
  private searchTimeout: ReturnType<typeof setTimeout> | null = null;

  // UI refs for filter chips
  private subsystemChip: HTMLElement | null = null;
  private pillBtns: Map<Severity, HTMLElement> = new Map();

  // Dedup tracking
  private seenKeys: Set<string> = new Set();

  constructor(private readonly client: AdminClient) {}

  async mount(container: HTMLElement): Promise<void> {
    this.el = h('div', { className: 'log-viewer' });

    // -- Header bar: pills + search + subsystem chip --
    const header = h('div', { className: 'log-header' });

    // Severity pills
    const pillBox = h('div', { className: 'log-pills' });
    for (const sev of ['ERROR', 'WARN', 'INFO'] as Severity[]) {
      const pill = h('button', {
        className: `severity-pill pill-${sev.toLowerCase()} active`,
        dataset: { severity: sev },
      }, sev);
      pill.addEventListener('click', () => this.togglePill(sev, pill));
      this.pillBtns.set(sev, pill);
      pillBox.appendChild(pill);
    }

    // Search input
    const searchInput = h('input', {
      type: 'text',
      className: 'search-input',
      placeholder: 'Search logs...',
    }) as HTMLInputElement;
    searchInput.addEventListener('input', () => {
      if (this.searchTimeout) clearTimeout(this.searchTimeout);
      this.searchTimeout = setTimeout(() => {
        this.searchText = searchInput.value.toLowerCase();
        this.refilter();
      }, 200);
    });

    // Subsystem chip (hidden by default)
    this.subsystemChip = h('div', { className: 'subsystem-chip hidden' });
    const chipClear = h('span', { className: 'chip-clear' }, 'x');
    chipClear.addEventListener('click', () => {
      this.subsystemFilter = null;
      this.subsystemChip!.classList.add('hidden');
      this.refilter();
    });
    this.subsystemChip.appendChild(h('span', { className: 'chip-label' }));
    this.subsystemChip.appendChild(chipClear);

    header.appendChild(pillBox);
    header.appendChild(searchInput);
    header.appendChild(this.subsystemChip);

    // -- Log entries container --
    this.logContainer = h('div', { className: 'log-entries' });
    this.logContainer.addEventListener('scroll', () => this.onScroll());

    // -- Resume button --
    this.resumeBtn = h('button', { className: 'resume-btn hidden' }, 'Resume auto-scroll');
    this.resumeBtn.addEventListener('click', () => {
      this.autoScroll = true;
      this.resumeBtn!.classList.add('hidden');
      this.scrollToBottom();
    });

    this.el.appendChild(header);
    this.el.appendChild(this.logContainer);
    this.el.appendChild(this.resumeBtn);
    container.appendChild(this.el);

    // Initial fetch + start polling
    await this.poll();
    this.intervalId = setInterval(() => { void this.poll(); }, 3_000);
  }

  destroy(): void {
    if (this.intervalId !== null) clearInterval(this.intervalId);
    if (this.searchTimeout) clearTimeout(this.searchTimeout);
    this.el?.remove();
    this.el = null;
    this.logContainer = null;
    this.resumeBtn = null;
    this.subsystemChip = null;
    this.seenKeys.clear();
  }

  // ---------------------------------------------------------------------------
  // Polling & rendering
  // ---------------------------------------------------------------------------

  private async poll(): Promise<void> {
    if (!this.logContainer) return;

    // Build server-side filter params
    const params: { severity?: string; subsystem?: string } = {};
    // Send the single active severity if only one is toggled off
    // Actually, send all active severities comma-separated for server to parse.
    // But the API takes a single severity string -- send lowest enabled.
    // Per the API: severity and subsystem are single-value filters.
    // We'll fetch all from server and filter client-side for pill combinations.
    if (this.subsystemFilter) params.subsystem = this.subsystemFilter;

    try {
      const entries = await this.client.getLogs(params);
      this.appendNewEntries(entries);
    } catch {
      // Skip -- stale data stays visible
    }
  }

  private appendNewEntries(entries: LogEntry[]): void {
    if (!this.logContainer) return;

    let addedAny = false;
    for (const entry of entries) {
      const key = entryKey(entry);
      if (this.seenKeys.has(key)) continue;
      this.seenKeys.add(key);

      const row = this.renderEntry(entry);
      // Apply client-side filter visibility
      const visible = this.entryMatchesFilters(entry);
      if (!visible) row.classList.add('hidden');
      this.logContainer.appendChild(row);
      addedAny = true;
    }

    if (addedAny && this.autoScroll) {
      this.scrollToBottom();
    }
  }

  private renderEntry(entry: LogEntry): HTMLElement {
    const sev = entry.severity.toUpperCase() as Severity;
    const color = SEVERITY_COLORS[sev] ?? '#888';
    const ts = this.formatTime(entry.timestamp);

    const row = h('div', {
      className: 'log-entry',
      dataset: { severity: sev, subsystem: entry.module },
    });

    const tsSpan = h('span', { className: 'log-ts' }, ts);
    const sevSpan = h('span', { className: 'log-severity', style: `color:${color}` }, sev);
    const subsysSpan = h('span', { className: 'subsystem' }, entry.module);
    subsysSpan.addEventListener('click', () => this.setSubsystemFilter(entry.module));
    const msgSpan = h('span', { className: 'log-msg' }, entry.message);

    row.appendChild(tsSpan);
    row.appendChild(sevSpan);
    row.appendChild(subsysSpan);
    row.appendChild(msgSpan);

    return row;
  }

  private formatTime(iso: string): string {
    try {
      const d = new Date(iso);
      return d.toLocaleTimeString('en-US', { hour12: false });
    } catch {
      return iso;
    }
  }

  // ---------------------------------------------------------------------------
  // Client-side filtering
  // ---------------------------------------------------------------------------

  private entryMatchesFilters(entry: LogEntry): boolean {
    const sev = entry.severity.toUpperCase() as Severity;
    if (!this.severityOn[sev]) return false;
    if (this.subsystemFilter && entry.module !== this.subsystemFilter) return false;
    if (this.searchText && !entry.message.toLowerCase().includes(this.searchText)) return false;
    return true;
  }

  /** Re-apply filters to all existing entries (show/hide). */
  private refilter(): void {
    if (!this.logContainer) return;
    const rows = this.logContainer.querySelectorAll('.log-entry');
    for (const row of rows) {
      const el = row as HTMLElement;
      const sev = el.dataset['severity'] as Severity;
      const subsys = el.dataset['subsystem'] ?? '';
      const msg = el.querySelector('.log-msg')?.textContent ?? '';

      let visible = true;
      if (!this.severityOn[sev]) visible = false;
      if (this.subsystemFilter && subsys !== this.subsystemFilter) visible = false;
      if (this.searchText && !msg.toLowerCase().includes(this.searchText)) visible = false;

      el.classList.toggle('hidden', !visible);
    }
  }

  private togglePill(sev: Severity, pill: HTMLElement): void {
    this.severityOn[sev] = !this.severityOn[sev];
    pill.classList.toggle('active', this.severityOn[sev]);
    this.refilter();
  }

  private setSubsystemFilter(subsystem: string): void {
    this.subsystemFilter = subsystem;
    if (this.subsystemChip) {
      const label = this.subsystemChip.querySelector('.chip-label');
      if (label) label.textContent = subsystem;
      this.subsystemChip.classList.remove('hidden');
    }
    this.refilter();
  }

  // ---------------------------------------------------------------------------
  // Auto-scroll
  // ---------------------------------------------------------------------------

  private onScroll(): void {
    if (!this.logContainer) return;
    const { scrollTop, scrollHeight, clientHeight } = this.logContainer;
    const atBottom = scrollTop >= scrollHeight - clientHeight - 50;
    if (!atBottom && this.autoScroll) {
      this.autoScroll = false;
      this.resumeBtn?.classList.remove('hidden');
    } else if (atBottom && !this.autoScroll) {
      this.autoScroll = true;
      this.resumeBtn?.classList.add('hidden');
    }
  }

  private scrollToBottom(): void {
    if (!this.logContainer) return;
    this.logContainer.scrollTop = this.logContainer.scrollHeight;
  }
}
