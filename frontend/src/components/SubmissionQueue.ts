/**
 * SubmissionQueue -- scrollable queue of submitted forecast requests.
 *
 * Displays each submission with lifecycle status badges:
 *   - pending:    neutral badge + relative time + question + meta
 *   - confirmed:  accent badge (same layout as pending)
 *   - processing: amber pulsing badge + MM:SS elapsed counter + question + meta
 *   - complete:   green badge + expandable forecast card via buildExpandableCard
 *   - failed:     red badge + error message
 *
 * Listens for 'submission-confirmed' CustomEvent on window to trigger refresh.
 * Completed forecasts use the shared expandable-card progressive disclosure
 * pattern (identical to dashboard and globe drill-down).
 */

import { h, clearChildren } from '@/utils/dom-utils';
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
    case 'pending': return 'sq-status-pending';
    case 'confirmed': return 'sq-status-confirmed';
    case 'processing': return 'sq-status-processing';
    case 'complete': return 'sq-status-complete';
    case 'failed': return 'sq-status-failed';
  }
}

/** Human-readable status label. */
function statusLabel(status: ForecastRequestStatus['status']): string {
  switch (status) {
    case 'pending': return 'PENDING';
    case 'confirmed': return 'CONFIRMED';
    case 'processing': return 'PROCESSING';
    case 'complete': return 'COMPLETE';
    case 'failed': return 'FAILED';
  }
}

export class SubmissionQueue {
  private readonly container: HTMLElement;
  private readonly list: HTMLElement;
  private requests: ForecastRequestStatus[] = [];
  private readonly expandedIds = new Set<string>();
  private readonly elapsedTimers = new Map<string, number>();
  private readonly forecastCache = new Map<string, ForecastResponse>();
  private readonly onSubmissionConfirmed: EventListener;

  constructor() {
    this.container = h('div', { className: 'submission-queue-wrapper' });

    const header = h('div', { className: 'submission-queue-header' },
      h('span', { className: 'submission-queue-title' }, 'Submission Queue'),
    );

    this.list = h('div', { className: 'submission-queue-list' });

    this.container.append(header, this.list);

    // Render empty state initially
    this.renderEmpty();

    // Listen for new submissions
    this.onSubmissionConfirmed = () => {
      void this.refresh();
    };
    window.addEventListener('submission-confirmed', this.onSubmissionConfirmed);
  }

  getElement(): HTMLElement {
    return this.container;
  }

  // ---------------------------------------------------------------------------
  // Data updates
  // ---------------------------------------------------------------------------

  update(requests: ForecastRequestStatus[]): void {
    // Sort by submitted_at descending (most recent first)
    this.requests = [...requests].sort(
      (a, b) => new Date(b.submitted_at).getTime() - new Date(a.submitted_at).getTime(),
    );
    this.rebuildList();
  }

  async refresh(): Promise<void> {
    try {
      const requests = await forecastClient.getRequests();
      this.update(requests);
    } catch (err: unknown) {
      console.error('[SubmissionQueue] refresh failed:', err);
    }
  }

  destroy(): void {
    // Clear all elapsed timers
    for (const timerId of this.elapsedTimers.values()) {
      window.clearInterval(timerId);
    }
    this.elapsedTimers.clear();
    this.forecastCache.clear();

    // Remove event listener
    window.removeEventListener('submission-confirmed', this.onSubmissionConfirmed);

    clearChildren(this.container);
  }

  // ---------------------------------------------------------------------------
  // Rendering
  // ---------------------------------------------------------------------------

  private rebuildList(): void {
    // Clear previous timers
    for (const timerId of this.elapsedTimers.values()) {
      window.clearInterval(timerId);
    }
    this.elapsedTimers.clear();

    clearChildren(this.list);

    if (this.requests.length === 0) {
      this.renderEmpty();
      return;
    }

    for (const req of this.requests) {
      this.list.appendChild(this.buildCard(req));
    }
  }

  private renderEmpty(): void {
    clearChildren(this.list);
    this.list.appendChild(
      h('div', { className: 'sq-empty-state' },
        h('div', { className: 'sq-empty-icon' }, '?'),
        h('div', { className: 'sq-empty-title' }, 'No submissions yet'),
        h('div', { className: 'sq-empty-message' },
          'Submit a geopolitical question using the form to generate a probability forecast.',
        ),
      ),
    );
  }

  private buildCard(req: ForecastRequestStatus): HTMLElement {
    switch (req.status) {
      case 'pending':
      case 'confirmed':
        return this.buildPendingCard(req);
      case 'processing':
        return this.buildProcessingCard(req);
      case 'complete':
        return this.buildCompleteCard(req);
      case 'failed':
        return this.buildFailedCard(req);
    }
  }

  // ---------------------------------------------------------------------------
  // Status-specific cards
  // ---------------------------------------------------------------------------

  private buildPendingCard(req: ForecastRequestStatus): HTMLElement {
    const countries = req.country_iso_list.map(iso => `${isoToFlag(iso)} ${iso}`).join(', ');

    return h('div', { className: 'sq-card' },
      h('div', { className: 'sq-card-top' },
        h('span', { className: `sq-status-badge ${statusCssClass(req.status)}` },
          statusLabel(req.status),
        ),
        h('span', { className: 'sq-card-time' }, relativeTime(req.submitted_at)),
      ),
      h('div', { className: 'sq-card-question' }, truncate(req.question, 120)),
      h('div', { className: 'sq-card-meta' },
        countries ? h('span', null, countries) : null,
        req.horizon_days > 0
          ? h('span', { className: 'sq-card-horizon' }, `${req.horizon_days}d horizon`)
          : null,
        h('span', { className: 'sq-card-category' }, req.category),
      ),
    );
  }

  private buildProcessingCard(req: ForecastRequestStatus): HTMLElement {
    const countries = req.country_iso_list.map(iso => `${isoToFlag(iso)} ${iso}`).join(', ');

    const elapsedEl = h('span', { className: 'sq-elapsed-time' }, '0:00');

    const card = h('div', { className: 'sq-card sq-card--processing' },
      h('div', { className: 'sq-card-top' },
        h('span', { className: `sq-status-badge ${statusCssClass(req.status)}` },
          statusLabel(req.status),
        ),
        elapsedEl,
      ),
      h('div', { className: 'sq-card-question' }, truncate(req.question, 120)),
      h('div', { className: 'sq-card-meta' },
        countries ? h('span', null, countries) : null,
        req.horizon_days > 0
          ? h('span', { className: 'sq-card-horizon' }, `${req.horizon_days}d horizon`)
          : null,
        h('span', { className: 'sq-card-category' }, req.category),
      ),
    );

    // Start elapsed timer
    const startMs = new Date(req.submitted_at).getTime();
    const updateElapsed = () => {
      const elapsed = Date.now() - startMs;
      const mins = Math.floor(elapsed / 60_000);
      const secs = Math.floor((elapsed % 60_000) / 1000);
      elapsedEl.textContent = `${mins}:${secs.toString().padStart(2, '0')}`;
    };
    updateElapsed(); // Immediate first tick
    const timerId = window.setInterval(updateElapsed, 1000);
    this.elapsedTimers.set(req.request_id, timerId);

    return card;
  }

  private buildCompleteCard(req: ForecastRequestStatus): HTMLElement {
    const card = h('div', {
      className: 'sq-card sq-card--complete',
      dataset: { requestId: req.request_id },
    });

    // Header row with status + completion time
    const headerRow = h('div', { className: 'sq-card-top' },
      h('span', { className: `sq-status-badge ${statusCssClass(req.status)}` },
        statusLabel(req.status),
      ),
      req.completed_at
        ? h('span', { className: 'sq-card-time' }, relativeTime(req.completed_at))
        : null,
    );
    card.appendChild(headerRow);

    // If we have a prediction ID, try to show an expandable forecast card
    if (req.prediction_ids.length > 0) {
      const predictionId = req.prediction_ids[0]!;
      const cached = this.forecastCache.get(predictionId);

      if (cached) {
        // We already have the forecast data -- render expandable card immediately
        card.appendChild(this.buildForecastContent(cached));
      } else {
        // Show question as placeholder while loading
        card.appendChild(
          h('div', { className: 'sq-card-question sq-card-loading' },
            truncate(req.question, 120),
          ),
        );
        // Fetch forecast in background
        void this.loadAndRenderForecast(predictionId, card);
      }
    } else {
      // No prediction ID -- just show question
      card.appendChild(
        h('div', { className: 'sq-card-question' }, truncate(req.question, 120)),
      );
    }

    return card;
  }

  private buildFailedCard(req: ForecastRequestStatus): HTMLElement {
    return h('div', { className: 'sq-card sq-card--failed' },
      h('div', { className: 'sq-card-top' },
        h('span', { className: `sq-status-badge ${statusCssClass(req.status)}` },
          statusLabel(req.status),
        ),
        h('span', { className: 'sq-card-time' }, relativeTime(req.submitted_at)),
      ),
      h('div', { className: 'sq-card-question' }, truncate(req.question, 120)),
      req.error_message
        ? h('div', { className: 'sq-card-error' }, req.error_message)
        : null,
    );
  }

  // ---------------------------------------------------------------------------
  // Expandable forecast card for completed requests
  // ---------------------------------------------------------------------------

  /**
   * Build an expandable forecast card wired with toggle behavior identical
   * to dashboard and globe drill-down. Uses the shared expandable-card utility.
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
    card: HTMLElement,
  ): Promise<void> {
    try {
      const forecast = await forecastClient.getForecastById(predictionId);
      if (!forecast) {
        // Forecast not found -- show question only
        return;
      }
      this.forecastCache.set(predictionId, forecast);

      // Replace the placeholder with the expandable card
      const placeholder = card.querySelector('.sq-card-loading');
      if (placeholder) {
        placeholder.remove();
      }
      card.appendChild(this.buildForecastContent(forecast));
    } catch (err: unknown) {
      console.error('[SubmissionQueue] Failed to load forecast:', predictionId, err);
      // Leave placeholder text as-is -- user sees the question at minimum
    }
  }
}
