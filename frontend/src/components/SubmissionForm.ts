/**
 * SubmissionForm -- three-state question submission form: input -> parsing -> confirm.
 *
 * Implements the two-phase submit/confirm flow where the user types a natural
 * language question, the LLM parses it into structured fields (countries, horizon,
 * category), and the user reviews before confirming. No modal -- the form
 * transforms inline between states.
 *
 * States:
 *   - input:   textarea + "Analyze Question" button
 *   - parsing: spinner + "Analyzing question..." text
 *   - confirm: parsed fields grid + Edit/Confirm buttons
 *
 * On confirm, dispatches 'submission-confirmed' CustomEvent on window so
 * SubmissionQueue can trigger a refresh to pick up the new pending item.
 */

import { h, clearChildren } from '@/utils/dom-utils';
import { forecastClient } from '@/services/forecast-client';
import { isoToFlag } from '@/components/expandable-card';
import type { ParsedQuestionResponse } from '@/types/api';

type FormState = 'input' | 'parsing' | 'confirm';

export class SubmissionForm {
  private readonly container: HTMLElement;
  private formState: FormState = 'input';
  private parsedResponse: ParsedQuestionResponse | null = null;
  private originalQuestion: string = '';
  private errorMessage: string | null = null;

  constructor() {
    this.container = h('div', { className: 'submission-form-wrapper' });
    this.render();
  }

  getElement(): HTMLElement {
    return this.container;
  }

  destroy(): void {
    clearChildren(this.container);
    this.parsedResponse = null;
    this.errorMessage = null;
    // Draft intentionally preserved in sessionStorage across destroy/remount
  }

  // ---------------------------------------------------------------------------
  // State machine
  // ---------------------------------------------------------------------------

  private setState(state: FormState): void {
    this.formState = state;
    this.render();
  }

  private render(): void {
    clearChildren(this.container);

    switch (this.formState) {
      case 'input':
        this.container.appendChild(this.buildInputState());
        break;
      case 'parsing':
        this.container.appendChild(this.buildParsingState());
        break;
      case 'confirm':
        this.container.appendChild(this.buildConfirmState());
        break;
    }
  }

  // ---------------------------------------------------------------------------
  // Input state
  // ---------------------------------------------------------------------------

  private buildInputState(): HTMLElement {
    const section = h('div', { className: 'submission-form' });

    const header = h('div', { className: 'submission-form-header' },
      h('span', { className: 'submission-form-title' }, 'Submit Question'),
      h('span', { className: 'submission-form-subtitle' },
        'Ask a geopolitical forecasting question. The system will parse, validate, and generate a probability forecast.',
      ),
    );

    const textarea = h('textarea', {
      className: 'submission-textarea',
      placeholder: 'e.g. Will Iran conduct a retaliatory strike against Israel within the next 90 days?',
      rows: '4',
    }) as HTMLTextAreaElement;

    // Restore draft from sessionStorage (survives same-route remount)
    const savedDraft = sessionStorage.getItem('geopol-submission-draft');
    if (this.originalQuestion) {
      // Pre-fill with original question if returning from Edit
      textarea.value = this.originalQuestion;
    } else if (savedDraft) {
      textarea.value = savedDraft;
      this.originalQuestion = savedDraft;
    }

    // Persist draft on every keystroke for cross-remount survival
    textarea.addEventListener('input', () => {
      sessionStorage.setItem('geopol-submission-draft', textarea.value);
    });

    const submitBtn = h('button', {
      className: 'submission-analyze-btn',
      type: 'button',
    }, 'Analyze Question') as HTMLButtonElement;

    submitBtn.addEventListener('click', () => {
      const question = textarea.value.trim();
      if (!question) return;
      this.originalQuestion = question;
      void this.handleSubmit(question);
    });

    // Also submit on Ctrl+Enter
    textarea.addEventListener('keydown', (e: KeyboardEvent) => {
      if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
        e.preventDefault();
        submitBtn.click();
      }
    });

    // Show error if any
    if (this.errorMessage) {
      const errorEl = h('div', { className: 'submission-error' }, this.errorMessage);
      section.append(header, textarea, submitBtn, errorEl);
    } else {
      section.append(header, textarea, submitBtn);
    }

    return section;
  }

  // ---------------------------------------------------------------------------
  // Parsing state (spinner)
  // ---------------------------------------------------------------------------

  private buildParsingState(): HTMLElement {
    return h('div', { className: 'submission-form submission-parsing' },
      h('div', { className: 'submission-spinner' }),
      h('span', { className: 'submission-parsing-text' }, 'Analyzing question...'),
      h('span', { className: 'submission-parsing-sub' }, 'Extracting countries, time horizon, and category'),
    );
  }

  // ---------------------------------------------------------------------------
  // Confirm state (parsed fields + Edit/Confirm)
  // ---------------------------------------------------------------------------

  private buildConfirmState(): HTMLElement {
    const parsed = this.parsedResponse;
    if (!parsed) return h('div');

    const section = h('div', { className: 'submission-form submission-confirm' });

    const header = h('div', { className: 'submission-confirm-header' },
      h('span', { className: 'submission-form-title' }, 'Review Parsed Question'),
      h('span', { className: 'submission-form-subtitle' },
        'Verify the system understood your question correctly before submitting.',
      ),
    );

    // Parsed fields grid
    const countriesDisplay = parsed.country_iso_list.length > 0
      ? parsed.country_iso_list.map(iso => `${isoToFlag(iso)} ${iso}`).join(', ')
      : 'Global';

    const fieldsGrid = h('div', { className: 'submission-fields-grid' },
      h('div', { className: 'submission-field' },
        h('span', { className: 'submission-field-label' }, 'Question'),
        h('span', { className: 'submission-field-value submission-field-question' }, parsed.question),
      ),
      h('div', { className: 'submission-field' },
        h('span', { className: 'submission-field-label' }, 'Countries'),
        h('span', { className: 'submission-field-value' }, countriesDisplay),
      ),
      h('div', { className: 'submission-field-row' },
        h('div', { className: 'submission-field' },
          h('span', { className: 'submission-field-label' }, 'Horizon'),
          h('span', { className: 'submission-field-value' }, `${parsed.horizon_days} days`),
        ),
        h('div', { className: 'submission-field' },
          h('span', { className: 'submission-field-label' }, 'Category'),
          h('span', { className: 'submission-field-value' }, parsed.category),
        ),
      ),
    );

    // Action buttons
    const editBtn = h('button', {
      className: 'submission-edit-btn',
      type: 'button',
    }, 'Edit');

    editBtn.addEventListener('click', () => {
      this.handleEdit();
    });

    const confirmBtn = h('button', {
      className: 'submission-confirm-btn',
      type: 'button',
    }, 'Confirm & Submit') as HTMLButtonElement;

    confirmBtn.addEventListener('click', () => {
      void this.handleConfirm(confirmBtn);
    });

    const actions = h('div', { className: 'submission-actions' },
      editBtn,
      confirmBtn,
    );

    // Show error if any
    if (this.errorMessage) {
      const errorEl = h('div', { className: 'submission-error' }, this.errorMessage);
      section.append(header, fieldsGrid, actions, errorEl);
    } else {
      section.append(header, fieldsGrid, actions);
    }

    return section;
  }

  // ---------------------------------------------------------------------------
  // Handlers
  // ---------------------------------------------------------------------------

  private async handleSubmit(question: string): Promise<void> {
    this.errorMessage = null;
    this.setState('parsing');
    try {
      this.parsedResponse = await forecastClient.submitQuestion(question);
      this.setState('confirm');
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Unknown error';
      this.errorMessage = `Failed to analyze question: ${msg}`;
      this.setState('input');
    }
  }

  private async handleConfirm(btn: HTMLButtonElement): Promise<void> {
    if (!this.parsedResponse) return;
    btn.disabled = true;
    this.errorMessage = null;
    try {
      await forecastClient.confirmSubmission(this.parsedResponse.request_id);
      // Reset form to input state and clear saved draft
      this.parsedResponse = null;
      this.originalQuestion = '';
      sessionStorage.removeItem('geopol-submission-draft');
      this.setState('input');
      // Notify queue to refresh
      window.dispatchEvent(new CustomEvent('submission-confirmed'));
    } catch (err: unknown) {
      btn.disabled = false;
      const msg = err instanceof Error ? err.message : 'Unknown error';
      this.errorMessage = `Failed to confirm submission: ${msg}`;
      this.render();
    }
  }

  private handleEdit(): void {
    // Revert to input with original question pre-filled
    this.errorMessage = null;
    this.setState('input');
  }
}
