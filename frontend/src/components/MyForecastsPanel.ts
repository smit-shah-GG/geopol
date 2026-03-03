/**
 * MyForecastsPanel -- user-submitted forecast requests with status tracking.
 *
 * Connects the Phase 14 submission queue to the frontend. Shows each
 * ForecastRequest with a status badge (pending/confirmed/processing/complete/failed).
 * Completed forecasts are clickable and open ScenarioExplorer.
 *
 * Fulfills FUX-04: user submission tracking panel in Col 3.
 */

import { Panel } from './Panel';
import { h, replaceChildren } from '@/utils/dom-utils';
import { forecastClient } from '@/services/forecast-client';
import type { ForecastRequestStatus } from '@/types/api';

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

/** Truncate text to maxLen chars with ellipsis. */
function truncate(text: string, maxLen: number): string {
  return text.length > maxLen ? text.slice(0, maxLen - 1) + '\u2026' : text;
}

/** CSS class for a given request status. */
function statusCssClass(status: ForecastRequestStatus['status']): string {
  switch (status) {
    case 'pending': return 'status-pending';
    case 'confirmed': return 'status-confirmed';
    case 'processing': return 'status-processing';
    case 'complete': return 'status-complete';
    case 'failed': return 'status-failed';
  }
}

export class MyForecastsPanel extends Panel {
  constructor() {
    super({ id: 'my-forecasts', title: 'MY FORECASTS', showCount: true });
  }

  /** Self-fetch via forecastClient. Used by RefreshScheduler. */
  public async refresh(): Promise<void> {
    try {
      const requests = await forecastClient.getRequests();
      this.renderRequests(requests);
    } catch (err: unknown) {
      if (this.isAbortError(err)) return;
      console.error('[MyForecastsPanel] refresh failed:', err);
      this.showError('Failed to load submissions');
    }
  }

  /** External data injection for coordinated initial loads. */
  public update(requests: ForecastRequestStatus[]): void {
    this.renderRequests(requests);
  }

  private renderRequests(requests: ForecastRequestStatus[]): void {
    this.setCount(requests.length);

    if (requests.length === 0) {
      replaceChildren(this.content,
        h('div', { className: 'empty-state' }, 'No submitted forecasts yet'),
      );
      return;
    }

    // Most recent first
    const sorted = [...requests].sort(
      (a, b) => new Date(b.submitted_at).getTime() - new Date(a.submitted_at).getTime(),
    );

    const rows = sorted.map((r) => this.buildRequestRow(r));
    replaceChildren(this.content, ...rows);
  }

  private buildRequestRow(r: ForecastRequestStatus): HTMLElement {
    const statusClass = statusCssClass(r.status);
    const statusLabel = r.status.toUpperCase();
    const question = truncate(r.question, 80);
    const countries = r.country_iso_list.join(', ');
    const submittedAgo = relativeTime(r.submitted_at);
    const isComplete = r.status === 'complete';
    const isFailed = r.status === 'failed';

    const row = h('div', {
      className: `mf-row${isComplete ? ' mf-row--clickable' : ''}`,
      dataset: { requestId: r.request_id },
    },
      h('div', { className: 'mf-row-top' },
        h('span', { className: 'mf-question' }, question),
        h('span', { className: `mf-status-badge ${statusClass}` }, statusLabel),
      ),
      h('div', { className: 'mf-row-meta' },
        countries ? h('span', { className: 'mf-countries' }, countries) : null,
        h('span', { className: 'mf-time' }, submittedAgo),
        r.horizon_days > 0 ? h('span', { className: 'mf-horizon' }, `${r.horizon_days}d`) : null,
      ),
      isFailed && r.error_message
        ? h('div', { className: 'mf-error' }, r.error_message)
        : null,
    );

    if (isComplete && r.prediction_ids.length > 0) {
      row.addEventListener('click', () => {
        void this.openCompletedForecast(r.prediction_ids[0]!);
      });
    }

    return row;
  }

  /**
   * Fetch the completed forecast and dispatch forecast-selected to open
   * ScenarioExplorer. The window-level listener in ScenarioExplorer picks it up.
   */
  private async openCompletedForecast(predictionId: string): Promise<void> {
    try {
      const forecast = await forecastClient.getForecastById(predictionId);
      if (!forecast) {
        console.warn('[MyForecastsPanel] Forecast not found:', predictionId);
        return;
      }
      window.dispatchEvent(
        new CustomEvent('forecast-selected', {
          detail: { forecast },
          bubbles: false,
        }),
      );
    } catch (err: unknown) {
      console.error('[MyForecastsPanel] Failed to load forecast:', err);
    }
  }
}
