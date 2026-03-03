/**
 * Forecasts screen -- two-column layout with question submission form
 * on the left and scrollable submission queue on the right.
 *
 * Implements the two-phase submit/confirm workflow (SCREEN-04):
 *   1. User types a geopolitical question
 *   2. LLM parses into structured fields (countries, horizon, category)
 *   3. User reviews and confirms before committing API budget
 *   4. Confirmed question enters the processing queue
 *   5. Queue auto-refreshes every 15s to show status updates
 *   6. Completed forecasts use shared expandable card progressive disclosure
 *
 * Owns RefreshScheduler lifecycle for queue polling and ScenarioExplorer
 * modal for "View Full Analysis" on completed forecasts.
 */

import { h } from '@/utils/dom-utils';
import { RefreshScheduler } from '@/app/refresh-scheduler';
import { SubmissionForm } from '@/components/SubmissionForm';
import { SubmissionQueue } from '@/components/SubmissionQueue';
import { ScenarioExplorer } from '@/components/ScenarioExplorer';
import type { GeoPolAppContext } from '@/app/app-context';

// ---------------------------------------------------------------------------
// Module-scoped state for screen lifecycle
// ---------------------------------------------------------------------------

let submissionForm: SubmissionForm | null = null;
let submissionQueue: SubmissionQueue | null = null;
let scenarioExplorer: ScenarioExplorer | null = null;
let scheduler: RefreshScheduler | null = null;

// ---------------------------------------------------------------------------
// Mount / Unmount
// ---------------------------------------------------------------------------

export function mountForecasts(container: HTMLElement, ctx: GeoPolAppContext): void {
  // Two-column flex layout
  const wrapper = h('div', { className: 'forecasts-screen' });

  // Left column: submission form (42%)
  const leftCol = h('div', { className: 'forecasts-col-left' });

  // Right column: submission queue (58%)
  const rightCol = h('div', { className: 'forecasts-col-right' });

  // -- Create components --
  submissionForm = new SubmissionForm();
  submissionQueue = new SubmissionQueue();

  leftCol.appendChild(submissionForm.getElement());
  rightCol.appendChild(submissionQueue.getElement());

  wrapper.append(leftCol, rightCol);
  container.appendChild(wrapper);

  // -- ScenarioExplorer modal (listens for forecast-selected on window) --
  scenarioExplorer = new ScenarioExplorer();

  // -- Initial data load --
  submissionQueue.refresh().catch((err: unknown) => {
    console.error('[ForecastsScreen] Initial queue load failed:', err);
  });

  // -- Refresh scheduler: queue status every 15s --
  scheduler = new RefreshScheduler(ctx);
  scheduler.init();

  const queue = submissionQueue;
  scheduler.registerAll([
    {
      name: 'forecasts-queue',
      fn: async () => {
        await queue.refresh();
      },
      intervalMs: 15_000,
    },
  ]);
}

export function unmountForecasts(_ctx: GeoPolAppContext): void {
  // Destroy scheduler
  if (scheduler) {
    scheduler.destroy();
    scheduler = null;
  }

  // Destroy modal
  if (scenarioExplorer) {
    scenarioExplorer.destroy();
    scenarioExplorer = null;
  }

  // Destroy components
  if (submissionQueue) {
    submissionQueue.destroy();
    submissionQueue = null;
  }

  if (submissionForm) {
    submissionForm.destroy();
    submissionForm = null;
  }
}
