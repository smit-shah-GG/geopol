/**
 * ProcessTable -- htop-style daemon status table with trigger buttons.
 *
 * Displays all background daemons (GDELT ingest, RSS, TKG training, etc.)
 * with status dots, relative timestamps, success/fail counts, and manual
 * trigger buttons. Auto-refreshes every 15s, only re-renders tbody.
 */

import { h, clearChildren } from '@/utils/dom-utils';
import type { AdminClient } from '@/admin/admin-client';
import type { ProcessInfo } from '@/admin/admin-types';

export interface AdminPanel {
  mount(container: HTMLElement): Promise<void>;
  destroy(): void;
}

/** Format ISO datetime to relative "Xm ago", "Xh ago", "Xd ago". */
function relativeTime(iso: string | null): string {
  if (!iso) return '--';
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

/** Format ISO datetime for "next run" -- show relative if in future, else "--". */
function relativeNext(iso: string | null): string {
  if (!iso) return '--';
  const diff = new Date(iso).getTime() - Date.now();
  if (diff < 0) return 'overdue';
  const s = Math.floor(diff / 1000);
  if (s < 60) return `${s}s`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m}m`;
  const hr = Math.floor(m / 60);
  return `${hr}h`;
}

export class ProcessTable implements AdminPanel {
  private el: HTMLElement | null = null;
  private tbody: HTMLTableSectionElement | null = null;
  private intervalId: ReturnType<typeof setInterval> | null = null;
  private triggering: Set<string> = new Set();

  constructor(private readonly client: AdminClient) {}

  async mount(container: HTMLElement): Promise<void> {
    this.el = h('div', { className: 'process-table-wrap' });

    const table = h('table', { className: 'process-table' });
    const thead = document.createElement('thead');
    thead.innerHTML = `<tr>
      <th></th><th>Name</th><th>Last Run</th><th>Next Run</th><th>OK/Fail</th><th></th>
    </tr>`;
    this.tbody = document.createElement('tbody');
    table.appendChild(thead);
    table.appendChild(this.tbody);
    this.el.appendChild(table);
    container.appendChild(this.el);

    // Initial load
    await this.refresh();

    // Auto-refresh every 15s
    this.intervalId = setInterval(() => { void this.refresh(); }, 15_000);
  }

  destroy(): void {
    if (this.intervalId !== null) clearInterval(this.intervalId);
    this.el?.remove();
    this.el = null;
    this.tbody = null;
  }

  private async refresh(): Promise<void> {
    if (!this.tbody) return;
    try {
      const procs = await this.client.getProcesses();
      this.renderRows(procs);
    } catch {
      // Silently skip -- table keeps stale data, next interval will retry.
    }
  }

  private renderRows(procs: ProcessInfo[]): void {
    if (!this.tbody) return;
    clearChildren(this.tbody);

    for (const p of procs) {
      const isTriggering = this.triggering.has(p.daemon_type);
      const statusClass = isTriggering ? 'status-running' : `status-${p.status}`;
      const row = h('tr', null,
        h('td', null, h('span', { className: `status-dot ${statusClass}` })),
        h('td', { className: 'proc-name' }, p.name),
        h('td', { className: 'proc-time' }, relativeTime(p.last_run)),
        h('td', { className: 'proc-time' }, relativeNext(p.next_run)),
        h('td', { className: 'proc-counts' }, `${p.success_count}/${p.fail_count}`),
        h('td', null, this.makeTriggerBtn(p.daemon_type, isTriggering)),
      );
      this.tbody.appendChild(row);
    }
  }

  private makeTriggerBtn(daemonType: string, isTriggering: boolean): HTMLElement {
    const btn = h('button', {
      className: 'trigger-btn',
      disabled: isTriggering,
    }, isTriggering ? '...' : 'RUN') as HTMLButtonElement;

    btn.addEventListener('click', () => {
      void this.handleTrigger(daemonType);
    });

    return btn;
  }

  private async handleTrigger(daemonType: string): Promise<void> {
    this.triggering.add(daemonType);
    // Optimistic UI: re-render immediately to show running state
    await this.refresh();

    try {
      await this.client.triggerJob(daemonType);
    } catch {
      // Trigger failed -- will show actual state on next refresh
    }

    // Re-fetch after a short delay to pick up the running state
    setTimeout(() => {
      this.triggering.delete(daemonType);
      void this.refresh();
    }, 2_000);
  }
}
