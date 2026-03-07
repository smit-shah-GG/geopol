/**
 * BacktestingPanel -- historical backtesting dashboard with run list,
 * drill-down view, d3 SVG charts (Brier curves, reliability diagrams),
 * start/cancel controls, and CSV/JSON export.
 *
 * Two views:
 *   A) Run list -- table of all backtest runs with status badges + inline form
 *   B) Drill-down -- summary cards, d3 charts, per-window detail, export buttons
 *
 * Auto-refreshes every 10s (faster than AccuracyPanel's 30s because running
 * backtests need responsive progress updates).
 */

import { h, clearChildren } from '@/utils/dom-utils';
import { showToast } from '@/admin/admin-toast';
import * as d3 from 'd3';
import type { AdminClient } from '@/admin/admin-client';
import type {
  BacktestRun,
  BacktestRunDetail,
  BacktestResult,
  CheckpointInfo,
  StartBacktestRequest,
} from '@/admin/admin-types';
import type { AdminPanel } from '@/admin/panels/ProcessTable';

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const REFRESH_INTERVAL_MS = 10_000;

const STATUS_COLORS: Record<string, string> = {
  pending: '#666',
  running: '#3b82f6',
  completed: '#22c55e',
  cancelled: '#f59e0b',
  failed: '#ef4444',
};

/** Chart color palette for multi-checkpoint comparison. */
const CHART_COLORS = ['#dc2626', '#3b82f6', '#22c55e', '#f59e0b', '#a855f7', '#ec4899'];

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

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

function formatBrier(value: number | null): string {
  if (value === null || value === undefined) return '--';
  return value.toFixed(4);
}

function formatDuration(startIso: string | null, endIso: string | null): string {
  if (!startIso) return '--';
  const start = new Date(startIso).getTime();
  const end = endIso ? new Date(endIso).getTime() : Date.now();
  const sec = Math.floor((end - start) / 1000);
  if (sec < 60) return `${sec}s`;
  const m = Math.floor(sec / 60);
  const s = sec % 60;
  if (m < 60) return `${m}m ${s}s`;
  const hr = Math.floor(m / 60);
  return `${hr}h ${m % 60}m`;
}

function statusBadge(status: string): HTMLElement {
  const color = STATUS_COLORS[status] ?? '#666';
  return h('span', {
    className: 'bt-status-badge',
    style: `background: ${color}20; color: ${color}; border: 1px solid ${color}40;`,
  }, status.toUpperCase());
}

/** Trigger a browser file download from a Blob. */
function downloadBlob(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

// ---------------------------------------------------------------------------
// BacktestingPanel
// ---------------------------------------------------------------------------

export class BacktestingPanel implements AdminPanel {
  private el: HTMLElement | null = null;
  private intervalId: ReturnType<typeof setInterval> | null = null;

  // Current view state
  private selectedRunId: string | null = null;
  private runs: BacktestRun[] = [];
  private detail: BacktestRunDetail | null = null;

  // Start form state
  private formVisible = false;
  private checkpoints: CheckpointInfo[] = [];

  // Container refs for selective re-render
  private listContainer: HTMLElement | null = null;
  private detailContainer: HTMLElement | null = null;
  private formContainer: HTMLElement | null = null;

  constructor(private readonly client: AdminClient) {}

  // -----------------------------------------------------------------------
  // Lifecycle
  // -----------------------------------------------------------------------

  async mount(container: HTMLElement): Promise<void> {
    this.el = h('div', { className: 'backtesting-panel' });
    container.appendChild(this.el);
    this.injectStyles();

    this.listContainer = h('div', { className: 'bt-list-view' });
    this.detailContainer = h('div', { className: 'bt-detail-view' });
    this.detailContainer.style.display = 'none';
    this.el.appendChild(this.listContainer);
    this.el.appendChild(this.detailContainer);

    await this.refresh();
    this.intervalId = setInterval(() => { void this.refresh(); }, REFRESH_INTERVAL_MS);
  }

  destroy(): void {
    if (this.intervalId !== null) {
      clearInterval(this.intervalId);
      this.intervalId = null;
    }
    if (this.el) {
      this.el.remove();
      this.el = null;
    }
    this.listContainer = null;
    this.detailContainer = null;
    this.formContainer = null;
  }

  // -----------------------------------------------------------------------
  // Data refresh
  // -----------------------------------------------------------------------

  private async refresh(): Promise<void> {
    try {
      if (this.selectedRunId) {
        this.detail = await this.client.getBacktestRun(this.selectedRunId);
        this.renderDetail();
      } else {
        this.runs = await this.client.getBacktestRuns();
        this.renderList();
      }
    } catch {
      // Silently skip -- next refresh will retry
    }
  }

  // -----------------------------------------------------------------------
  // View A: Run List
  // -----------------------------------------------------------------------

  private renderList(): void {
    if (!this.listContainer) return;
    clearChildren(this.listContainer);
    this.listContainer.style.display = '';
    if (this.detailContainer) this.detailContainer.style.display = 'none';

    // Toolbar
    const toolbar = h('div', { className: 'bt-toolbar' });
    const startBtn = h('button', { className: 'save-btn' }, '+ START NEW RUN') as HTMLButtonElement;
    startBtn.addEventListener('click', () => { void this.toggleForm(); });
    toolbar.appendChild(startBtn);
    this.listContainer.appendChild(toolbar);

    // Inline form (hidden by default)
    this.formContainer = h('div', { className: 'bt-start-form hidden' });
    this.listContainer.appendChild(this.formContainer);
    if (this.formVisible) {
      void this.renderForm();
    }

    // Empty state
    if (this.runs.length === 0) {
      const empty = h('div', { className: 'accuracy-empty' },
        "No backtest runs. Click 'Start New Run' to begin your first evaluation.",
      );
      this.listContainer.appendChild(empty);
      return;
    }

    // Table
    const wrap = h('div', { className: 'process-table-wrap' });
    const table = h('table', { className: 'process-table' }) as HTMLTableElement;

    const thead = document.createElement('thead');
    thead.innerHTML = `<tr>
      <th>Label</th><th>Status</th><th>Checkpoints</th><th>Windows</th>
      <th>Brier</th><th>MRR</th><th>Created</th><th>Duration</th><th></th>
    </tr>`;
    table.appendChild(thead);

    const tbody = document.createElement('tbody');
    for (const run of this.runs) {
      const tr = document.createElement('tr');
      tr.style.cursor = 'pointer';

      // Label
      const tdLabel = document.createElement('td');
      tdLabel.className = 'proc-name';
      tdLabel.textContent = run.label;
      tr.appendChild(tdLabel);

      // Status badge
      const tdStatus = document.createElement('td');
      tdStatus.appendChild(statusBadge(run.status));
      if (run.status === 'running') {
        const progress = h('div', { className: 'bt-progress-bar' },
          h('div', {
            className: 'bt-progress-fill',
            style: `width: ${run.total_windows > 0 ? (run.completed_windows / run.total_windows * 100) : 0}%`,
          }),
        );
        tdStatus.appendChild(
          h('span', { className: 'proc-time', style: 'margin-left: 6px;' },
            `${run.completed_windows}/${run.total_windows}`,
          ),
        );
        tdStatus.appendChild(progress);
      }
      tr.appendChild(tdStatus);

      // Checkpoints
      const tdCp = document.createElement('td');
      tdCp.className = 'proc-time';
      const cpNames = Object.keys(run.checkpoints_json);
      tdCp.textContent = cpNames.length > 0 ? cpNames.join(', ') : '--';
      tr.appendChild(tdCp);

      // Windows
      const tdWin = document.createElement('td');
      tdWin.className = 'proc-time';
      tdWin.textContent = `${run.completed_windows}/${run.total_windows}`;
      tr.appendChild(tdWin);

      // Brier
      const tdBrier = document.createElement('td');
      tdBrier.textContent = formatBrier(run.aggregate_brier);
      tr.appendChild(tdBrier);

      // MRR
      const tdMrr = document.createElement('td');
      tdMrr.textContent = formatBrier(run.aggregate_mrr);
      tr.appendChild(tdMrr);

      // Created
      const tdCreated = document.createElement('td');
      tdCreated.className = 'proc-time';
      tdCreated.textContent = relativeTime(run.created_at);
      tr.appendChild(tdCreated);

      // Duration
      const tdDur = document.createElement('td');
      tdDur.className = 'proc-time';
      tdDur.textContent = formatDuration(run.started_at, run.completed_at);
      tr.appendChild(tdDur);

      // Cancel button for running/pending
      const tdAction = document.createElement('td');
      if (run.status === 'running' || run.status === 'pending') {
        const cancelBtn = h('button', { className: 'trigger-btn' }, 'CANCEL') as HTMLButtonElement;
        cancelBtn.addEventListener('click', (e) => {
          e.stopPropagation();
          void this.handleCancel(run.id);
        });
        tdAction.appendChild(cancelBtn);
      }
      tr.appendChild(tdAction);

      // Row click -> drill-down
      tr.addEventListener('click', () => { void this.selectRun(run.id); });
      tbody.appendChild(tr);
    }

    table.appendChild(tbody);
    wrap.appendChild(table);
    this.listContainer.appendChild(wrap);
  }

  // -----------------------------------------------------------------------
  // Start Form
  // -----------------------------------------------------------------------

  private async toggleForm(): Promise<void> {
    this.formVisible = !this.formVisible;
    if (this.formContainer) {
      this.formContainer.classList.toggle('hidden', !this.formVisible);
    }
    if (this.formVisible) {
      await this.renderForm();
    }
  }

  private async renderForm(): Promise<void> {
    if (!this.formContainer) return;
    clearChildren(this.formContainer);
    this.formContainer.classList.remove('hidden');

    // Fetch checkpoints if not already loaded
    if (this.checkpoints.length === 0) {
      try {
        this.checkpoints = await this.client.getCheckpoints();
      } catch {
        this.formContainer.appendChild(
          h('div', { className: 'process-error' }, 'Failed to load checkpoints'),
        );
        return;
      }
    }

    // Label
    const labelInput = h('input', {
      className: 'config-input',
      type: 'text',
      placeholder: 'Run label (required)',
    }) as HTMLInputElement;

    // Description
    const descInput = h('textarea', {
      className: 'config-textarea',
      placeholder: 'Description (optional)',
      style: 'min-height: 40px;',
    }) as HTMLTextAreaElement;

    // Window params
    const windowInput = h('input', {
      className: 'config-input',
      type: 'number',
      value: '14',
      style: 'width: 80px;',
    }) as HTMLInputElement;

    const slideInput = h('input', {
      className: 'config-input',
      type: 'number',
      value: '7',
      style: 'width: 80px;',
    }) as HTMLInputElement;

    const minPredInput = h('input', {
      className: 'config-input',
      type: 'number',
      value: '3',
      style: 'width: 80px;',
    }) as HTMLInputElement;

    // Checkpoint checkboxes
    const cpContainer = h('div', { className: 'bt-checkpoint-list' });
    const cpChecks: Map<string, { checkbox: HTMLInputElement; info: CheckpointInfo }> = new Map();

    // Group by model_type
    const byType = new Map<string, CheckpointInfo[]>();
    for (const cp of this.checkpoints) {
      const list = byType.get(cp.model_type) ?? [];
      list.push(cp);
      byType.set(cp.model_type, list);
    }

    for (const [modelType, cps] of byType) {
      const groupLabel = h('div', { className: 'config-desc', style: 'margin-top: 8px; font-weight: 700;' },
        modelType.toUpperCase(),
      );
      cpContainer.appendChild(groupLabel);

      for (const cp of cps) {
        const cb = h('input', { type: 'checkbox', className: 'config-checkbox' }) as HTMLInputElement;
        cpChecks.set(cp.name, { checkbox: cb, info: cp });
        const row = h('label', { className: 'config-checkbox-label' },
          cb,
          `${cp.name}`,
          cp.metrics ? h('span', { className: 'proc-time' }, ` (MRR: ${(cp.metrics['mrr'] ?? 0).toFixed(4)})`) : '',
        );
        cpContainer.appendChild(row);
      }
    }

    if (this.checkpoints.length === 0) {
      cpContainer.appendChild(
        h('div', { className: 'config-desc' }, 'No checkpoints available. Train a model first.'),
      );
    }

    // API call warning
    const apiNote = h('div', { className: 'config-desc', style: 'margin-top: 8px; color: #f59e0b;' },
      'Each prediction burns one Gemini API call. Review window parameters carefully.',
    );

    // Submit button
    const submitBtn = h('button', { className: 'save-btn', style: 'margin-top: 12px;' },
      'START BACKTEST',
    ) as HTMLButtonElement;

    const errorEl = h('div', { className: 'config-error' });

    submitBtn.addEventListener('click', () => {
      const label = labelInput.value.trim();
      if (!label) {
        errorEl.textContent = 'Label is required';
        return;
      }

      const selected: Record<string, string> = {};
      for (const [name, entry] of cpChecks) {
        if (entry.checkbox.checked) {
          selected[name] = entry.info.path;
        }
      }
      if (Object.keys(selected).length === 0) {
        errorEl.textContent = 'Select at least one checkpoint';
        return;
      }

      const config: StartBacktestRequest = {
        label,
        checkpoints: selected,
        window_size_days: parseInt(windowInput.value, 10) || 14,
        slide_step_days: parseInt(slideInput.value, 10) || 7,
        min_predictions_per_window: parseInt(minPredInput.value, 10) || 3,
      };
      const desc = descInput.value.trim();
      if (desc) config.description = desc;

      submitBtn.disabled = true;
      submitBtn.textContent = 'STARTING...';
      void this.handleStart(config);
    });

    // Assemble form
    this.formContainer.appendChild(
      h('div', { className: 'config-form' },
        h('div', { className: 'config-field' },
          h('div', { className: 'config-label' }, 'Label'),
          labelInput,
        ),
        h('div', { className: 'config-field' },
          h('div', { className: 'config-label' }, 'Description'),
          descInput,
        ),
        h('div', { style: 'display: flex; gap: 16px; flex-wrap: wrap;' },
          h('div', { className: 'config-field' },
            h('div', { className: 'config-label' }, 'Window (days)'),
            windowInput,
          ),
          h('div', { className: 'config-field' },
            h('div', { className: 'config-label' }, 'Slide step (days)'),
            slideInput,
          ),
          h('div', { className: 'config-field' },
            h('div', { className: 'config-label' }, 'Min predictions'),
            minPredInput,
          ),
        ),
        h('div', { className: 'config-field' },
          h('div', { className: 'config-label' }, 'Checkpoints'),
          cpContainer,
        ),
        apiNote,
        errorEl,
        submitBtn,
      ),
    );
  }

  // -----------------------------------------------------------------------
  // Actions
  // -----------------------------------------------------------------------

  private async handleStart(config: StartBacktestRequest): Promise<void> {
    try {
      await this.client.startBacktestRun(config);
      showToast('Backtest started');
      this.formVisible = false;
      this.checkpoints = []; // Reset so next form open re-fetches
      await this.refresh();
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Start failed';
      showToast(msg, true);
      await this.refresh();
    }
  }

  private async handleCancel(runId: string): Promise<void> {
    try {
      await this.client.cancelBacktestRun(runId);
      showToast('Backtest cancelled');
      await this.refresh();
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Cancel failed';
      showToast(msg, true);
    }
  }

  private async selectRun(runId: string): Promise<void> {
    this.selectedRunId = runId;
    try {
      this.detail = await this.client.getBacktestRun(runId);
      this.renderDetail();
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Failed to load run';
      showToast(msg, true);
      this.selectedRunId = null;
    }
  }

  private backToList(): void {
    this.selectedRunId = null;
    this.detail = null;
    if (this.detailContainer) this.detailContainer.style.display = 'none';
    void this.refresh();
  }

  // -----------------------------------------------------------------------
  // View B: Drill-Down
  // -----------------------------------------------------------------------

  private renderDetail(): void {
    if (!this.detailContainer || !this.detail) return;
    clearChildren(this.detailContainer);
    this.detailContainer.style.display = '';
    if (this.listContainer) this.listContainer.style.display = 'none';

    const { run, results } = this.detail;

    // Back button
    const backBtn = h('button', { className: 'revert-btn', style: 'margin-bottom: 16px;' },
      '\u2190 All Runs',
    ) as HTMLButtonElement;
    backBtn.addEventListener('click', () => this.backToList());
    this.detailContainer.appendChild(backBtn);

    // Run header
    const header = h('div', { className: 'bt-run-header' },
      h('div', { style: 'display: flex; align-items: center; gap: 12px; margin-bottom: 8px;' },
        h('span', { style: 'font-size: 18px; font-weight: 700; font-family: "Courier New", monospace;' }, run.label),
        statusBadge(run.status),
      ),
      run.description ? h('div', { className: 'config-desc', style: 'margin-bottom: 8px;' }, run.description) : '',
      h('div', { className: 'proc-time' },
        `Windows: ${run.window_size_days}d / ${run.slide_step_days}d step / min ${run.min_predictions_per_window} preds`,
        ` | Checkpoints: ${Object.keys(run.checkpoints_json).join(', ')}`,
        ` | ${formatDuration(run.started_at, run.completed_at)}`,
      ),
      run.error_message ? h('div', { className: 'process-error', style: 'margin-top: 8px;' }, run.error_message) : '',
    );
    this.detailContainer.appendChild(header);

    // Summary stat cards
    const stats = this.computeStats(run, results);
    const statsBar = h('div', { className: 'accuracy-stats', style: 'margin-top: 16px;' });

    const brierCard = this.makeStatCard('Brier Score', formatBrier(run.aggregate_brier), 'bt-card-brier');
    const calCard = this.makeStatCard('Calibration', stats.meanAbsCal !== null ? stats.meanAbsCal.toFixed(4) : '--', 'bt-card-cal');
    const hitCard = this.makeStatCard('Hit Rate', stats.hitRate !== null ? `${(stats.hitRate * 100).toFixed(1)}%` : '--', 'bt-card-hit');

    statsBar.appendChild(brierCard.el);
    statsBar.appendChild(calCard.el);
    statsBar.appendChild(hitCard.el);

    // vs Polymarket card -- only if data exists
    if (run.vs_polymarket_record_json) {
      const rec = run.vs_polymarket_record_json;
      const gw = rec['geopol_wins'] ?? 0;
      const pw = rec['polymarket_wins'] ?? 0;
      const dr = rec['draws'] ?? 0;
      const pmCard = this.makeStatCard('vs Polymarket', `${gw}W / ${pw}L / ${dr}D`, 'bt-card-pm');
      statsBar.appendChild(pmCard.el);
    }

    this.detailContainer.appendChild(statsBar);

    // Chart sections (expandable)
    if (results.length > 0) {
      // Brier curve chart
      const brierSection = this.makeChartSection('Brier Score Over Time', () => this.renderBrierChart(results));
      this.detailContainer.appendChild(brierSection);

      // Calibration reliability diagram
      const calSection = this.makeChartSection('Calibration Reliability Diagram', () => this.renderCalibrationChart(results));
      this.detailContainer.appendChild(calSection);

      // Hit rate per checkpoint
      const hitSection = this.makeChartSection('Hit Rate by Checkpoint', () => this.renderHitRateChart(results));
      this.detailContainer.appendChild(hitSection);

      // Geopol vs Polymarket (only if polymarket data exists)
      const hasPmData = results.some(r => r.polymarket_brier !== null);
      if (hasPmData) {
        const pmSection = this.makeChartSection('Geopol vs Polymarket Brier', () => this.renderPmComparisonChart(results));
        this.detailContainer.appendChild(pmSection);
      }
    }

    // Export buttons
    const exportRow = h('div', { className: 'config-btn-row' });
    const csvBtn = h('button', { className: 'trigger-btn' }, 'EXPORT CSV') as HTMLButtonElement;
    csvBtn.addEventListener('click', () => { void this.handleExport(run.id, 'csv', run.label); });
    const jsonBtn = h('button', { className: 'trigger-btn' }, 'EXPORT JSON') as HTMLButtonElement;
    jsonBtn.addEventListener('click', () => { void this.handleExport(run.id, 'json', run.label); });
    exportRow.appendChild(csvBtn);
    exportRow.appendChild(jsonBtn);

    // Cancel button if still running
    if (run.status === 'running' || run.status === 'pending') {
      const cancelBtn = h('button', { className: 'trigger-btn', style: 'border-color: #ef4444; color: #ef4444;' },
        'CANCEL RUN',
      ) as HTMLButtonElement;
      cancelBtn.addEventListener('click', () => { void this.handleCancel(run.id); });
      exportRow.appendChild(cancelBtn);
    }

    this.detailContainer.appendChild(exportRow);
  }

  // -----------------------------------------------------------------------
  // Stat computation
  // -----------------------------------------------------------------------

  private computeStats(run: BacktestRun, results: BacktestResult[]): {
    meanAbsCal: number | null;
    hitRate: number | null;
  } {
    // Mean absolute calibration error from aggregated bins
    let calSum = 0;
    let calCount = 0;
    for (const r of results) {
      const bins = r.calibration_bins_json;
      if (!bins) continue;
      for (let i = 0; i < bins.bins.length; i++) {
        const pred = bins.predicted_avg[i] ?? null;
        const obs = bins.observed_freq[i] ?? null;
        if (pred !== null && obs !== null) {
          calSum += Math.abs(pred - obs);
          calCount++;
        }
      }
    }
    const meanAbsCal = calCount > 0 ? calSum / calCount : null;

    // Hit rate from prediction_details
    let correct = 0;
    let total = 0;
    for (const r of results) {
      const details = r.prediction_details_json;
      if (!details) continue;
      for (const d of details) {
        total++;
        // Correct if predicted prob > 0.5 and outcome = 1, or predicted prob <= 0.5 and outcome = 0
        const predictedYes = d.predicted_prob > 0.5;
        const actualYes = d.outcome >= 0.5;
        if (predictedYes === actualYes) correct++;
      }
    }
    const hitRate = total > 0 ? correct / total : null;

    // Suppress unused variable warning -- run is accessed via aggregate fields
    void run;

    return { meanAbsCal, hitRate };
  }

  // -----------------------------------------------------------------------
  // Expandable chart sections
  // -----------------------------------------------------------------------

  private makeStatCard(label: string, value: string, _id: string): { el: HTMLElement } {
    const el = h('div', { className: 'accuracy-stat-card' },
      h('div', { className: 'stat-value' }, value),
      h('div', { className: 'stat-label' }, label),
    );
    return { el };
  }

  private makeChartSection(title: string, renderFn: () => HTMLElement): HTMLElement {
    const section = h('div', { className: 'bt-chart-section' });
    const header = h('div', { className: 'bt-chart-header' },
      h('span', null, title),
      h('span', { className: 'bt-chart-toggle' }, '\u25B6'),
    );
    const body = h('div', { className: 'bt-chart-body hidden' });

    let expanded = false;
    header.addEventListener('click', () => {
      expanded = !expanded;
      body.classList.toggle('hidden', !expanded);
      const toggle = header.querySelector('.bt-chart-toggle');
      if (toggle) toggle.textContent = expanded ? '\u25BC' : '\u25B6';
      if (expanded && body.childNodes.length === 0) {
        body.appendChild(renderFn());
      }
    });

    section.appendChild(header);
    section.appendChild(body);
    return section;
  }

  // -----------------------------------------------------------------------
  // d3 Charts
  // -----------------------------------------------------------------------

  private renderBrierChart(results: BacktestResult[]): HTMLElement {
    const container = h('div', { className: 'bt-chart-container' });

    const margin = { top: 20, right: 30, bottom: 40, left: 50 };
    const width = 700;
    const height = 300;
    const innerW = width - margin.left - margin.right;
    const innerH = height - margin.top - margin.bottom;

    const svg = d3.select(container)
      .append('svg')
      .attr('viewBox', `0 0 ${width} ${height}`)
      .attr('preserveAspectRatio', 'xMidYMid meet')
      .style('width', '100%')
      .style('max-width', `${width}px`);

    const g = svg.append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    // Group results by checkpoint_name
    const byCheckpoint = new Map<string, BacktestResult[]>();
    for (const r of results) {
      if (r.brier_score === null) continue;
      const list = byCheckpoint.get(r.checkpoint_name) ?? [];
      list.push(r);
      byCheckpoint.set(r.checkpoint_name, list);
    }

    if (byCheckpoint.size === 0) {
      container.textContent = 'No Brier data available.';
      return container;
    }

    // Sort each group by window_end
    for (const [, list] of byCheckpoint) {
      list.sort((a, b) => new Date(a.window_end).getTime() - new Date(b.window_end).getTime());
    }

    // Scales
    const allDates = results.filter(r => r.brier_score !== null).map(r => new Date(r.window_end));
    const allBrier = results.filter(r => r.brier_score !== null).map(r => r.brier_score!);

    const x = d3.scaleTime()
      .domain(d3.extent(allDates) as [Date, Date])
      .range([0, innerW]);

    const y = d3.scaleLinear()
      .domain([0, Math.max(0.5, d3.max(allBrier) ?? 0.5)])
      .range([innerH, 0])
      .nice();

    // Axes
    g.append('g')
      .attr('transform', `translate(0,${innerH})`)
      .call(d3.axisBottom(x).ticks(6))
      .selectAll('text, line, path')
      .attr('stroke', '#888')
      .attr('fill', '#888');

    g.append('g')
      .call(d3.axisLeft(y).ticks(5))
      .selectAll('text, line, path')
      .attr('stroke', '#888')
      .attr('fill', '#888');

    // Reference line at Brier = 0.25 (random baseline)
    g.append('line')
      .attr('x1', 0)
      .attr('x2', innerW)
      .attr('y1', y(0.25))
      .attr('y2', y(0.25))
      .attr('stroke', '#666')
      .attr('stroke-dasharray', '4,4')
      .attr('stroke-width', 1);

    g.append('text')
      .attr('x', innerW - 4)
      .attr('y', y(0.25) - 4)
      .attr('text-anchor', 'end')
      .attr('fill', '#666')
      .attr('font-size', '10px')
      .text('random (0.25)');

    // Lines per checkpoint
    const line = d3.line<BacktestResult>()
      .x(d => x(new Date(d.window_end)))
      .y(d => y(d.brier_score!));

    let colorIdx = 0;
    for (const [cpName, cpResults] of byCheckpoint) {
      const color = CHART_COLORS[colorIdx % CHART_COLORS.length]!;
      colorIdx++;

      g.append('path')
        .datum(cpResults)
        .attr('fill', 'none')
        .attr('stroke', color)
        .attr('stroke-width', 2)
        .attr('d', line);

      // Dots
      g.selectAll(`.dot-${cpName}`)
        .data(cpResults)
        .enter()
        .append('circle')
        .attr('cx', d => x(new Date(d.window_end)))
        .attr('cy', d => y(d.brier_score!))
        .attr('r', 4)
        .attr('fill', color)
        .append('title')
        .text(d => `${cpName}: ${formatBrier(d.brier_score)} (${d.window_start} - ${d.window_end}, n=${d.num_predictions})`);

      // Legend entry
      g.append('text')
        .attr('x', innerW - 4)
        .attr('y', 12 + (colorIdx - 1) * 14)
        .attr('text-anchor', 'end')
        .attr('fill', color)
        .attr('font-size', '11px')
        .attr('font-family', '"Courier New", monospace')
        .text(cpName);
    }

    return container;
  }

  private renderCalibrationChart(results: BacktestResult[]): HTMLElement {
    const container = h('div', { className: 'bt-chart-container' });

    // Aggregate bins across all windows
    const binMap = new Map<number, { predSum: number; obsSum: number; count: number }>();
    for (const r of results) {
      const bins = r.calibration_bins_json;
      if (!bins) continue;
      for (let i = 0; i < bins.bins.length; i++) {
        const binVal = bins.bins[i]!;
        const pred = bins.predicted_avg[i] ?? null;
        const obs = bins.observed_freq[i] ?? null;
        const cnt = bins.counts[i] ?? 0;
        if (pred === null || obs === null || cnt === 0) continue;
        const existing = binMap.get(binVal) ?? { predSum: 0, obsSum: 0, count: 0 };
        existing.predSum += pred * cnt;
        existing.obsSum += obs * cnt;
        existing.count += cnt;
        binMap.set(binVal, existing);
      }
    }

    if (binMap.size === 0) {
      container.textContent = 'No calibration data available.';
      return container;
    }

    const calPoints = Array.from(binMap.entries())
      .map(([_bin, data]) => ({
        predicted: data.predSum / data.count,
        observed: data.obsSum / data.count,
        count: data.count,
      }))
      .sort((a, b) => a.predicted - b.predicted);

    const margin = { top: 20, right: 30, bottom: 40, left: 50 };
    const width = 500;
    const height = 500;
    const innerW = width - margin.left - margin.right;
    const innerH = height - margin.top - margin.bottom;

    const svg = d3.select(container)
      .append('svg')
      .attr('viewBox', `0 0 ${width} ${height}`)
      .attr('preserveAspectRatio', 'xMidYMid meet')
      .style('width', '100%')
      .style('max-width', `${width}px`);

    const g = svg.append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    const x = d3.scaleLinear().domain([0, 1]).range([0, innerW]);
    const y = d3.scaleLinear().domain([0, 1]).range([innerH, 0]);

    // Axes
    g.append('g')
      .attr('transform', `translate(0,${innerH})`)
      .call(d3.axisBottom(x).ticks(5))
      .selectAll('text, line, path')
      .attr('stroke', '#888')
      .attr('fill', '#888');

    g.append('g')
      .call(d3.axisLeft(y).ticks(5))
      .selectAll('text, line, path')
      .attr('stroke', '#888')
      .attr('fill', '#888');

    // Axis labels
    g.append('text')
      .attr('x', innerW / 2)
      .attr('y', innerH + 35)
      .attr('text-anchor', 'middle')
      .attr('fill', '#888')
      .attr('font-size', '11px')
      .text('Predicted Probability');

    g.append('text')
      .attr('transform', 'rotate(-90)')
      .attr('x', -innerH / 2)
      .attr('y', -38)
      .attr('text-anchor', 'middle')
      .attr('fill', '#888')
      .attr('font-size', '11px')
      .text('Observed Frequency');

    // Perfect calibration diagonal
    g.append('line')
      .attr('x1', x(0))
      .attr('y1', y(0))
      .attr('x2', x(1))
      .attr('y2', y(1))
      .attr('stroke', '#666')
      .attr('stroke-dasharray', '4,4')
      .attr('stroke-width', 1);

    // Faint bar chart for counts
    const maxCount = d3.max(calPoints, d => d.count) ?? 1;
    const barWidth = innerW / (calPoints.length + 1);
    g.selectAll('.cal-bar')
      .data(calPoints)
      .enter()
      .append('rect')
      .attr('x', d => x(d.predicted) - barWidth / 2)
      .attr('y', d => innerH - (d.count / maxCount) * innerH * 0.3)
      .attr('width', barWidth)
      .attr('height', d => (d.count / maxCount) * innerH * 0.3)
      .attr('fill', '#dc262620')
      .attr('stroke', '#dc262640')
      .attr('stroke-width', 0.5);

    // Calibration curve
    const line = d3.line<typeof calPoints[0]>()
      .x(d => x(d.predicted))
      .y(d => y(d.observed));

    g.append('path')
      .datum(calPoints)
      .attr('fill', 'none')
      .attr('stroke', '#dc2626')
      .attr('stroke-width', 2)
      .attr('d', line);

    // Dots
    g.selectAll('.cal-dot')
      .data(calPoints)
      .enter()
      .append('circle')
      .attr('cx', d => x(d.predicted))
      .attr('cy', d => y(d.observed))
      .attr('r', 5)
      .attr('fill', '#dc2626')
      .append('title')
      .text(d => `Predicted: ${d.predicted.toFixed(3)}, Observed: ${d.observed.toFixed(3)}, n=${d.count}`);

    return container;
  }

  private renderHitRateChart(results: BacktestResult[]): HTMLElement {
    const container = h('div', { className: 'bt-chart-container' });

    // Compute per-checkpoint hit rates
    const hitMap = new Map<string, { correct: number; total: number }>();
    for (const r of results) {
      if (!r.prediction_details_json) continue;
      const entry = hitMap.get(r.checkpoint_name) ?? { correct: 0, total: 0 };
      for (const d of r.prediction_details_json) {
        entry.total++;
        const predictedYes = d.predicted_prob > 0.5;
        const actualYes = d.outcome >= 0.5;
        if (predictedYes === actualYes) entry.correct++;
      }
      hitMap.set(r.checkpoint_name, entry);
    }

    if (hitMap.size === 0) {
      container.textContent = 'No prediction detail data available.';
      return container;
    }

    const data = Array.from(hitMap.entries())
      .map(([name, val]) => ({
        name,
        rate: val.total > 0 ? val.correct / val.total : 0,
        correct: val.correct,
        total: val.total,
      }));

    if (data.length <= 1) {
      // Simple stat display for single checkpoint
      const d = data[0];
      if (d) {
        container.appendChild(
          h('div', { style: 'font-family: "Courier New", monospace; font-size: 14px; padding: 12px;' },
            h('div', null, `Checkpoint: ${d.name}`),
            h('div', null, `Total predictions: ${d.total}`),
            h('div', null, `Correct: ${d.correct}`),
            h('div', { style: 'font-size: 20px; font-weight: 700; margin-top: 8px;' },
              `Hit rate: ${(d.rate * 100).toFixed(1)}%`,
            ),
          ),
        );
      }
      return container;
    }

    // Grouped bar chart
    const margin = { top: 20, right: 20, bottom: 50, left: 50 };
    const width = 500;
    const height = 250;
    const innerW = width - margin.left - margin.right;
    const innerH = height - margin.top - margin.bottom;

    const svg = d3.select(container)
      .append('svg')
      .attr('viewBox', `0 0 ${width} ${height}`)
      .attr('preserveAspectRatio', 'xMidYMid meet')
      .style('width', '100%')
      .style('max-width', `${width}px`);

    const g = svg.append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    const x = d3.scaleBand()
      .domain(data.map(d => d.name))
      .range([0, innerW])
      .padding(0.3);

    const y = d3.scaleLinear()
      .domain([0, 1])
      .range([innerH, 0]);

    g.append('g')
      .attr('transform', `translate(0,${innerH})`)
      .call(d3.axisBottom(x))
      .selectAll('text, line, path')
      .attr('stroke', '#888')
      .attr('fill', '#888');

    g.append('g')
      .call(d3.axisLeft(y).ticks(5).tickFormat(d => `${(+d * 100).toFixed(0)}%`))
      .selectAll('text, line, path')
      .attr('stroke', '#888')
      .attr('fill', '#888');

    g.selectAll('.hit-bar')
      .data(data)
      .enter()
      .append('rect')
      .attr('x', d => x(d.name) ?? 0)
      .attr('y', d => y(d.rate))
      .attr('width', x.bandwidth())
      .attr('height', d => innerH - y(d.rate))
      .attr('fill', (_d, i) => CHART_COLORS[i % CHART_COLORS.length]!)
      .attr('rx', 2)
      .append('title')
      .text(d => `${d.name}: ${(d.rate * 100).toFixed(1)}% (${d.correct}/${d.total})`);

    // Value labels on bars
    g.selectAll('.hit-label')
      .data(data)
      .enter()
      .append('text')
      .attr('x', d => (x(d.name) ?? 0) + x.bandwidth() / 2)
      .attr('y', d => y(d.rate) - 4)
      .attr('text-anchor', 'middle')
      .attr('fill', '#e5e5e5')
      .attr('font-size', '11px')
      .attr('font-family', '"Courier New", monospace')
      .text(d => `${(d.rate * 100).toFixed(1)}%`);

    return container;
  }

  private renderPmComparisonChart(results: BacktestResult[]): HTMLElement {
    const container = h('div', { className: 'bt-chart-container' });

    // Filter results with polymarket data, sorted by window_end
    const pmResults = results
      .filter(r => r.polymarket_brier !== null && r.brier_score !== null)
      .sort((a, b) => new Date(a.window_end).getTime() - new Date(b.window_end).getTime());

    if (pmResults.length === 0) {
      container.textContent = 'No Polymarket comparison data available.';
      return container;
    }

    // Summary stats
    let gpWins = 0;
    let pmWins = 0;
    let draws = 0;
    for (const r of pmResults) {
      const gw = r.geopol_vs_pm_wins ?? 0;
      const pw = r.pm_vs_geopol_wins ?? 0;
      gpWins += gw;
      pmWins += pw;
      if (gw === pw) draws++;
    }

    const summaryEl = h('div', { style: 'font-family: "Courier New", monospace; font-size: 13px; margin-bottom: 12px; display: flex; gap: 20px;' },
      h('span', { style: 'color: #22c55e;' }, `Geopol wins: ${gpWins}`),
      h('span', { style: 'color: #ef4444;' }, `PM wins: ${pmWins}`),
      h('span', { style: 'color: #888;' }, `Draws: ${draws}`),
    );
    container.appendChild(summaryEl);

    // Brier comparison chart
    const margin = { top: 20, right: 30, bottom: 40, left: 50 };
    const width = 700;
    const height = 300;
    const innerW = width - margin.left - margin.right;
    const innerH = height - margin.top - margin.bottom;

    const svg = d3.select(container)
      .append('svg')
      .attr('viewBox', `0 0 ${width} ${height}`)
      .attr('preserveAspectRatio', 'xMidYMid meet')
      .style('width', '100%')
      .style('max-width', `${width}px`);

    const g = svg.append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    const allDates = pmResults.map(r => new Date(r.window_end));
    const allBrier = [
      ...pmResults.map(r => r.brier_score!),
      ...pmResults.map(r => r.polymarket_brier!),
    ];

    const x = d3.scaleTime()
      .domain(d3.extent(allDates) as [Date, Date])
      .range([0, innerW]);

    const y = d3.scaleLinear()
      .domain([0, Math.max(0.5, d3.max(allBrier) ?? 0.5)])
      .range([innerH, 0])
      .nice();

    g.append('g')
      .attr('transform', `translate(0,${innerH})`)
      .call(d3.axisBottom(x).ticks(6))
      .selectAll('text, line, path')
      .attr('stroke', '#888')
      .attr('fill', '#888');

    g.append('g')
      .call(d3.axisLeft(y).ticks(5))
      .selectAll('text, line, path')
      .attr('stroke', '#888')
      .attr('fill', '#888');

    // Area fill: green where Geopol < PM (outperforms), red where Geopol > PM
    const area = d3.area<BacktestResult>()
      .x(d => x(new Date(d.window_end)))
      .y0(d => y(d.polymarket_brier!))
      .y1(d => y(d.brier_score!));

    // Single area with neutral fill -- exact per-point coloring requires clipPath
    // which adds complexity. Use a gradient approach: overall winner color.
    const gpAvg = d3.mean(pmResults, d => d.brier_score!) ?? 0;
    const pmAvg = d3.mean(pmResults, d => d.polymarket_brier!) ?? 0;
    const areaColor = gpAvg < pmAvg ? '#22c55e20' : '#ef444420';

    g.append('path')
      .datum(pmResults)
      .attr('fill', areaColor)
      .attr('d', area);

    // Geopol Brier line
    const gpLine = d3.line<BacktestResult>()
      .x(d => x(new Date(d.window_end)))
      .y(d => y(d.brier_score!));

    g.append('path')
      .datum(pmResults)
      .attr('fill', 'none')
      .attr('stroke', '#22c55e')
      .attr('stroke-width', 2)
      .attr('d', gpLine);

    // PM Brier line
    const pmLine = d3.line<BacktestResult>()
      .x(d => x(new Date(d.window_end)))
      .y(d => y(d.polymarket_brier!));

    g.append('path')
      .datum(pmResults)
      .attr('fill', 'none')
      .attr('stroke', '#ef4444')
      .attr('stroke-width', 2)
      .attr('d', pmLine);

    // Legend
    g.append('text')
      .attr('x', innerW - 4)
      .attr('y', 12)
      .attr('text-anchor', 'end')
      .attr('fill', '#22c55e')
      .attr('font-size', '11px')
      .attr('font-family', '"Courier New", monospace')
      .text('Geopol');

    g.append('text')
      .attr('x', innerW - 4)
      .attr('y', 26)
      .attr('text-anchor', 'end')
      .attr('fill', '#ef4444')
      .attr('font-size', '11px')
      .attr('font-family', '"Courier New", monospace')
      .text('Polymarket');

    // Dots for both
    g.selectAll('.gp-dot')
      .data(pmResults)
      .enter()
      .append('circle')
      .attr('cx', d => x(new Date(d.window_end)))
      .attr('cy', d => y(d.brier_score!))
      .attr('r', 3)
      .attr('fill', '#22c55e')
      .append('title')
      .text(d => `Geopol: ${formatBrier(d.brier_score)}`);

    g.selectAll('.pm-dot')
      .data(pmResults)
      .enter()
      .append('circle')
      .attr('cx', d => x(new Date(d.window_end)))
      .attr('cy', d => y(d.polymarket_brier!))
      .attr('r', 3)
      .attr('fill', '#ef4444')
      .append('title')
      .text(d => `Polymarket: ${formatBrier(d.polymarket_brier)}`);

    return container;
  }

  // -----------------------------------------------------------------------
  // Export
  // -----------------------------------------------------------------------

  private async handleExport(runId: string, format: 'csv' | 'json', label: string): Promise<void> {
    try {
      const blob = await this.client.exportBacktestRun(runId, format);
      const filename = `backtest-${label.replace(/\s+/g, '-').toLowerCase()}.${format}`;
      downloadBlob(blob, filename);
      showToast(`Exported ${format.toUpperCase()}`);
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Export failed';
      showToast(msg, true);
    }
  }

  // -----------------------------------------------------------------------
  // Scoped CSS injection
  // -----------------------------------------------------------------------

  private injectStyles(): void {
    if (document.querySelector('#bt-panel-styles')) return;
    const style = document.createElement('style');
    style.id = 'bt-panel-styles';
    style.textContent = `
      /* BacktestingPanel scoped styles */
      .admin-layout .backtesting-panel {
        max-width: 1000px;
      }

      .admin-layout .bt-toolbar {
        display: flex;
        justify-content: flex-end;
        margin-bottom: 16px;
      }

      .admin-layout .bt-start-form {
        background: var(--admin-surface);
        border: 1px solid var(--admin-border);
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 16px;
      }

      .admin-layout .bt-start-form.hidden {
        display: none;
      }

      .admin-layout .bt-status-badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 10px;
        font-weight: 700;
        font-family: 'Courier New', monospace;
        letter-spacing: 0.5px;
      }

      .admin-layout .bt-progress-bar {
        display: inline-block;
        width: 60px;
        height: 4px;
        background: #333;
        border-radius: 2px;
        margin-left: 6px;
        vertical-align: middle;
        overflow: hidden;
      }

      .admin-layout .bt-progress-fill {
        height: 100%;
        background: #3b82f6;
        border-radius: 2px;
        transition: width 0.3s;
        animation: admin-pulse 1s ease-in-out infinite;
      }

      .admin-layout .bt-run-header {
        margin-bottom: 16px;
        padding-bottom: 12px;
        border-bottom: 1px solid var(--admin-border);
      }

      .admin-layout .bt-checkpoint-list {
        display: flex;
        flex-direction: column;
        gap: 4px;
        max-height: 200px;
        overflow-y: auto;
        padding: 4px 0;
      }

      .admin-layout .bt-chart-section {
        border: 1px solid var(--admin-border);
        border-radius: 6px;
        margin-bottom: 12px;
        overflow: hidden;
      }

      .admin-layout .bt-chart-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 10px 14px;
        background: var(--admin-surface);
        cursor: pointer;
        font-family: 'Courier New', monospace;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 1px;
        color: var(--admin-accent);
        user-select: none;
        transition: background 0.15s;
      }

      .admin-layout .bt-chart-header:hover {
        background: var(--admin-surface-hover);
      }

      .admin-layout .bt-chart-toggle {
        font-size: 10px;
        color: var(--admin-text-dim);
      }

      .admin-layout .bt-chart-body {
        padding: 16px;
      }

      .admin-layout .bt-chart-body.hidden {
        display: none;
      }

      .admin-layout .bt-chart-container {
        font-family: 'Courier New', monospace;
        color: var(--admin-text);
      }

      .admin-layout .bt-chart-container svg text {
        font-family: 'Courier New', monospace;
      }

      .admin-layout .bt-chart-container svg .domain,
      .admin-layout .bt-chart-container svg .tick line {
        stroke: #444;
      }

      .admin-layout .bt-chart-container svg .tick text {
        fill: #888;
        font-size: 10px;
      }
    `;
    document.head.appendChild(style);
  }
}
