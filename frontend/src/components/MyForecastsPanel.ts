/**
 * MyForecastsPanel -- user-submitted forecast requests with status tracking.
 *
 * Connects the Phase 14 submission queue to the frontend. Shows each
 * ForecastRequest with a status badge (pending/confirmed/processing/complete/failed).
 * Completed forecasts render as expandable cards with progressive disclosure
 * (ensemble weights, calibration, mini scenario tree, evidence summaries).
 *
 * Fulfills FUX-04: user submission tracking panel in Col 3.
 */

import { Panel } from './Panel';
import { h, replaceChildren } from '@/utils/dom-utils';
import { forecastClient } from '@/services/forecast-client';
import {
  buildExpandableCard,
  buildExpandedContent,
  isoToFlag,
  relativeTime,
  truncate,
  type ExpandableCardOptions,
} from '@/components/expandable-card';
import type { ForecastRequestStatus, ForecastResponse } from '@/types/api';

/** CSS class for a given request status. */
function statusCssClass(status: ForecastRequestStatus['status']): string {
  switch (status) {
    case 'pending':
    case 'confirmed':
      return 'status-queued';
    case 'processing': return 'status-processing';
    case 'complete': return 'status-complete';
    case 'failed': return 'status-failed';
  }
}

export class MyForecastsPanel extends Panel {
  private readonly expandedIds = new Set<string>();
  private readonly forecastCache = new Map<string, ForecastResponse>();

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

  public override destroy(): void {
    this.forecastCache.clear();
    this.expandedIds.clear();
    super.destroy();
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
    const isComplete = r.status === 'complete' && r.prediction_ids.length > 0;

    if (isComplete) {
      return this.buildCompleteRow(r);
    }

    return this.buildSimpleRow(r);
  }

  /** Simple row for pending/confirmed/processing/failed statuses. */
  private buildSimpleRow(r: ForecastRequestStatus): HTMLElement {
    const statusClass = statusCssClass(r.status);
    const label = (r.status === 'pending' || r.status === 'confirmed')
      ? 'QUEUED'
      : r.status.toUpperCase();
    const question = truncate(r.question, 80);
    const countries = r.country_iso_list
      .map(iso => `${isoToFlag(iso)} ${iso}`)
      .join(', ');
    const submittedAgo = relativeTime(r.submitted_at);
    const isFailed = r.status === 'failed';

    return h('div', {
      className: 'mf-row',
      dataset: { requestId: r.request_id },
    },
      h('div', { className: 'mf-row-top' },
        h('span', { className: 'mf-question' }, question),
        h('span', { className: `mf-status-badge ${statusClass}` }, label),
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
  }

  /** Expandable card row for completed forecasts. */
  private buildCompleteRow(r: ForecastRequestStatus): HTMLElement {
    const wrapper = h('div', {
      className: 'mf-row mf-row--complete',
      dataset: { requestId: r.request_id },
    });

    // Header with COMPLETE badge and completion time
    const headerRow = h('div', { className: 'mf-row-top' },
      h('span', { className: `mf-status-badge ${statusCssClass(r.status)}` }, 'COMPLETE'),
      r.completed_at
        ? h('span', { className: 'mf-time' }, relativeTime(r.completed_at))
        : null,
    );
    wrapper.appendChild(headerRow);

    const predictionId = r.prediction_ids[0]!;
    const cached = this.forecastCache.get(predictionId);

    if (cached) {
      wrapper.appendChild(this.buildForecastContent(cached));
    } else {
      // Loading placeholder showing the question text
      wrapper.appendChild(
        h('div', { className: 'mf-question mf-loading-placeholder' },
          truncate(r.question, 120),
        ),
      );
      // Fetch forecast in background, then replace placeholder
      void this.loadAndRenderForecast(predictionId, wrapper);
    }

    return wrapper;
  }

  /**
   * Build an expandable forecast card wired with toggle behavior identical
   * to SubmissionQueue and globe drill-down.
   */
  private buildForecastContent(forecast: ForecastResponse): HTMLElement {
    const opts: ExpandableCardOptions = {
      expandedIds: this.expandedIds,
      onToggle: (id: string, cardEl: HTMLElement) => {
        if (this.expandedIds.has(id)) {
          this.expandedIds.delete(id);
          cardEl.classList.remove('expanded');
          const expanded = cardEl.querySelector('.expanded-content');
          if (expanded) expanded.remove();
        } else {
          this.expandedIds.add(id);
          cardEl.classList.add('expanded');
          cardEl.appendChild(buildExpandedContent(forecast));
        }
      },
    };

    return buildExpandableCard(forecast, opts);
  }

  private async loadAndRenderForecast(
    predictionId: string,
    wrapper: HTMLElement,
  ): Promise<void> {
    try {
      const forecast = await forecastClient.getForecastById(predictionId);
      if (!forecast) {
        // Forecast not found -- leave question text placeholder as-is
        return;
      }
      this.forecastCache.set(predictionId, forecast);

      // Replace the loading placeholder with the expandable card
      const placeholder = wrapper.querySelector('.mf-loading-placeholder');
      if (placeholder) {
        placeholder.remove();
      }
      wrapper.appendChild(this.buildForecastContent(forecast));
    } catch (err: unknown) {
      console.error('[MyForecastsPanel] Failed to load forecast:', predictionId, err);
      // Leave placeholder text as-is -- user sees the question at minimum
    }
  }
}
