import { Panel } from './Panel';
import { h, replaceChildren } from '@/utils/dom-utils';
import { forecastClient } from '@/services/forecast-client';
import type { HealthResponse, SubsystemStatus } from '@/types/api';

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

/** Aggregate status badge class. */
function statusBadgeClass(status: HealthResponse['status']): string {
  switch (status) {
    case 'healthy': return 'health-badge-healthy';
    case 'degraded': return 'health-badge-degraded';
    case 'unhealthy': return 'health-badge-unhealthy';
  }
}

/**
 * SystemHealthPanel -- subsystem health status display.
 *
 * Dual API: refresh() calls /health endpoint, update() accepts pre-fetched data.
 * Health endpoint is public (no API key).
 */
export class SystemHealthPanel extends Panel {
  constructor() {
    super({ id: 'system-health', title: 'SYSTEM HEALTH' });
  }

  /** Self-fetch via forecastClient. Used by RefreshScheduler. */
  public async refresh(): Promise<void> {
    this.showLoading();
    try {
      const health = await forecastClient.getHealth();
      const state = forecastClient.getDataState('health');
      this.setDataBadge(state.mode);
      this.renderHealth(health);
    } catch (err: unknown) {
      if (this.isAbortError(err)) return;
      console.error('[SystemHealthPanel] refresh failed:', err);
      this.showError('Failed to load health status');
    }
  }

  /** External data injection from main.ts coordinated loads. */
  public update(health: HealthResponse): void {
    this.renderHealth(health);
  }

  private renderHealth(health: HealthResponse): void {
    const badgeCls = statusBadgeClass(health.status);
    const badgeText = health.status.toUpperCase();

    const subsystemRows = health.subsystems.map((s) => this.buildSubsystemRow(s));

    replaceChildren(this.content,
      // Aggregate status
      h('div', { className: 'health-aggregate' },
        h('span', { className: `health-status-badge ${badgeCls}` }, badgeText),
        h('span', { className: 'health-version' }, `v${health.version}`),
        h('span', { className: 'health-timestamp' }, relativeTime(health.timestamp)),
      ),
      // Subsystem list
      h('div', { className: 'health-subsystems' }, ...subsystemRows),
    );
  }

  private buildSubsystemRow(s: SubsystemStatus): HTMLElement {
    const dotClass = s.healthy ? 'status-dot healthy' : 'status-dot unhealthy';
    const detail = s.detail ?? '';
    const checked = relativeTime(s.checked_at);

    return h('div', { className: 'health-subsystem' },
      h('span', { className: dotClass }),
      h('span', { className: 'subsystem-name' }, s.name),
      h('span', { className: 'subsystem-detail' }, detail),
      h('span', { className: 'subsystem-checked' }, checked),
    );
  }
}
