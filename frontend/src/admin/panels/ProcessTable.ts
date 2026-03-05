/**
 * ProcessTable -- htop-style daemon status table with trigger, pause/resume
 * buttons and extended APScheduler job state (failures, duration, errors).
 *
 * Displays all background daemons (GDELT ingest, RSS, TKG training, etc.)
 * with status dots, relative timestamps, success/fail counts, failure badges,
 * last duration, and manual trigger + pause/resume buttons.
 * Auto-refreshes every 15s, only re-renders tbody.
 */

import { h, clearChildren } from '@/utils/dom-utils';
import { showToast } from '@/admin/admin-toast';
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

/** Format seconds to human-readable duration (e.g., "2.3s", "1m 45s"). */
function formatDuration(seconds: number | null): string {
  if (seconds === null || seconds === undefined) return '--';
  if (seconds < 60) return `${seconds.toFixed(1)}s`;
  const m = Math.floor(seconds / 60);
  const s = Math.round(seconds % 60);
  if (m < 60) return `${m}m ${s}s`;
  const hr = Math.floor(m / 60);
  const rm = m % 60;
  return `${hr}h ${rm}m`;
}

/** Auto-pause threshold -- 5 consecutive failures triggers auto-pause. */
const AUTO_PAUSE_THRESHOLD = 5;

export class ProcessTable implements AdminPanel {
  private el: HTMLElement | null = null;
  private tbody: HTMLTableSectionElement | null = null;
  private intervalId: ReturnType<typeof setInterval> | null = null;
  private triggering: Set<string> = new Set();
  private toggling: Set<string> = new Set();

  constructor(private readonly client: AdminClient) {}

  async mount(container: HTMLElement): Promise<void> {
    this.el = h('div', { className: 'process-table-wrap' });

    const table = h('table', { className: 'process-table' });
    const thead = document.createElement('thead');
    thead.innerHTML = `<tr>
      <th></th><th>Name</th><th>Last Run</th><th>Next Run</th><th>Duration</th><th>OK/Fail</th><th></th><th></th>
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
      const isToggling = this.toggling.has(p.daemon_type);

      // Status dot: paused overrides, then triggering, then actual status
      let statusClass: string;
      if (p.paused) {
        statusClass = 'status-paused';
      } else if (isTriggering) {
        statusClass = 'status-running';
      } else {
        statusClass = `status-${p.status}`;
      }

      // Row class: highlight paused rows
      const rowClass = p.paused ? 'process-paused' : '';

      // Build status cell with optional paused badge
      const statusCell = h('td', null,
        h('span', { className: `status-dot ${statusClass}` }),
      );
      if (p.paused) {
        const badge = h('span', { className: 'badge-paused' }, 'PAUSED');
        statusCell.appendChild(badge);
      }

      // Build name cell with failure badge
      const nameCell = h('td', { className: 'proc-name' }, p.name);
      if (p.consecutive_failures > 0) {
        const failClass = p.consecutive_failures >= AUTO_PAUSE_THRESHOLD
          ? 'badge-failures critical'
          : 'badge-failures warning';
        const badge = h('span', { className: failClass }, `${p.consecutive_failures}`);
        nameCell.appendChild(badge);
      }

      // Build error tooltip on last run cell if last_error exists
      const lastRunCell = h('td', { className: 'proc-time' }, relativeTime(p.last_run));
      if (p.last_error) {
        const errorSpan = h('span', { className: 'process-error', title: p.last_error },
          ` ${p.last_error.length > 40 ? p.last_error.slice(0, 40) + '...' : p.last_error}`);
        lastRunCell.appendChild(errorSpan);
      }

      const row = h('tr', { className: rowClass },
        statusCell,
        nameCell,
        lastRunCell,
        h('td', { className: 'proc-time' }, p.paused ? '--' : relativeNext(p.next_run)),
        h('td', { className: 'process-duration' }, formatDuration(p.last_duration)),
        h('td', { className: 'proc-counts' }, `${p.success_count}/${p.fail_count}`),
        h('td', null, this.makePauseResumeBtn(p.daemon_type, p.paused, isToggling)),
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

  private makePauseResumeBtn(daemonType: string, isPaused: boolean, isToggling: boolean): HTMLElement {
    if (isPaused) {
      const btn = h('button', {
        className: 'btn-resume',
        disabled: isToggling,
      }, isToggling ? '...' : 'RESUME') as HTMLButtonElement;
      btn.addEventListener('click', () => { void this.handleResume(daemonType); });
      return btn;
    }

    const btn = h('button', {
      className: 'btn-pause',
      disabled: isToggling,
    }, isToggling ? '...' : 'PAUSE') as HTMLButtonElement;
    btn.addEventListener('click', () => { void this.handlePause(daemonType); });
    return btn;
  }

  private async handleTrigger(daemonType: string): Promise<void> {
    this.triggering.add(daemonType);
    // Optimistic UI: re-render immediately to show running state
    await this.refresh();

    try {
      await this.client.triggerJob(daemonType);
      showToast(`Job triggered: ${daemonType}`);
    } catch (err) {
      this.triggering.delete(daemonType);
      void this.refresh();
      const msg = err instanceof Error ? err.message : 'Trigger failed';
      showToast(msg, true);
      return;
    }

    // Re-fetch after a short delay to pick up the running state
    setTimeout(() => {
      this.triggering.delete(daemonType);
      void this.refresh();
    }, 2_000);
  }

  private async handlePause(daemonType: string): Promise<void> {
    this.toggling.add(daemonType);
    void this.refresh();

    try {
      await this.client.pauseJob(daemonType);
      showToast(`Job paused: ${daemonType}`);
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Pause failed';
      showToast(msg, true);
    } finally {
      this.toggling.delete(daemonType);
      void this.refresh();
    }
  }

  private async handleResume(daemonType: string): Promise<void> {
    this.toggling.add(daemonType);
    void this.refresh();

    try {
      await this.client.resumeJob(daemonType);
      showToast(`Job resumed: ${daemonType}`);
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Resume failed';
      showToast(msg, true);
    } finally {
      this.toggling.delete(daemonType);
      void this.refresh();
    }
  }
}
