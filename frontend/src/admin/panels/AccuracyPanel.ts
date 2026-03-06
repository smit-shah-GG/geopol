/**
 * AccuracyPanel -- head-to-head Geopol vs Polymarket accuracy dashboard.
 *
 * Summary stats bar (total resolved, wins/losses/draws, cumulative and
 * rolling 30d Brier scores) above a sortable table of resolved/voided
 * comparisons. Voided rows get reduced opacity and strikethrough.
 *
 * Auto-refreshes every 30s. Sort state persists across refreshes.
 */

import { h, clearChildren } from '@/utils/dom-utils';
import type { AdminClient } from '@/admin/admin-client';
import type { AccuracyData, ResolvedComparison } from '@/admin/admin-types';
import type { AdminPanel } from '@/admin/panels/ProcessTable';

// ---------------------------------------------------------------------------
// Sort column definition
// ---------------------------------------------------------------------------

type SortColumn =
  | 'polymarket_title'
  | 'geopol_probability'
  | 'polymarket_price'
  | 'polymarket_outcome'
  | 'geopol_brier'
  | 'polymarket_brier'
  | 'winner'
  | 'country_iso'
  | 'resolved_at';

type SortDirection = 'asc' | 'desc';

interface ColumnDef {
  key: SortColumn;
  label: string;
  sortable: boolean;
}

const COLUMNS: ColumnDef[] = [
  { key: 'polymarket_title',  label: 'Question',    sortable: true },
  { key: 'geopol_probability', label: 'Geopol P',   sortable: true },
  { key: 'polymarket_price',  label: 'PM Price',    sortable: true },
  { key: 'polymarket_outcome', label: 'Outcome',    sortable: true },
  { key: 'geopol_brier',      label: 'Geopol Brier', sortable: true },
  { key: 'polymarket_brier',  label: 'PM Brier',    sortable: true },
  { key: 'winner',            label: 'Winner',      sortable: true },
  { key: 'country_iso',       label: 'Country',     sortable: true },
  { key: 'resolved_at',       label: 'Resolved',    sortable: true },
];

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

function truncate(text: string, max: number): string {
  return text.length > max ? text.slice(0, max) + '...' : text;
}

function formatPct(value: number | null): string {
  if (value === null || value === undefined) return '--';
  return (value * 100).toFixed(1) + '%';
}

function formatBrier(value: number | null): string {
  if (value === null || value === undefined) return '--';
  return value.toFixed(4);
}

function outcomeLabel(outcome: number | null, status: string): string {
  if (status === 'voided') return 'Voided';
  if (outcome === null || outcome === undefined) return '--';
  return outcome >= 0.5 ? 'Yes' : 'No';
}

// ---------------------------------------------------------------------------
// Stat card builder
// ---------------------------------------------------------------------------

interface StatCardRefs {
  valueNode: Text;
}

function createStatCard(label: string, value: string): { el: HTMLElement; refs: StatCardRefs } {
  const valueNode = document.createTextNode(value);
  const el = h('div', { className: 'accuracy-stat-card' },
    h('div', { className: 'stat-value' }, valueNode),
    h('div', { className: 'stat-label' }, label),
  );
  return { el, refs: { valueNode } };
}

// ---------------------------------------------------------------------------
// AccuracyPanel
// ---------------------------------------------------------------------------

export class AccuracyPanel implements AdminPanel {
  private el: HTMLElement | null = null;
  private tbody: HTMLTableSectionElement | null = null;
  private intervalId: ReturnType<typeof setInterval> | null = null;
  private sortColumn: SortColumn = 'resolved_at';
  private sortDirection: SortDirection = 'desc';
  private headerCells: Map<SortColumn, HTMLTableCellElement> = new Map();
  private comparisons: ResolvedComparison[] = [];

  // Stat card text nodes for in-place update
  private statRefs: Record<string, StatCardRefs> = {};

  constructor(private readonly client: AdminClient) {}

  async mount(container: HTMLElement): Promise<void> {
    this.el = h('div', { className: 'accuracy-panel' });
    container.appendChild(this.el);

    // Build stats bar (populated on first refresh)
    const statsBar = h('div', { className: 'accuracy-stats' });
    const statDefs: Array<{ key: string; label: string }> = [
      { key: 'totalResolved',   label: 'Resolved' },
      { key: 'totalVoided',     label: 'Voided' },
      { key: 'geopolWins',      label: 'Geopol Wins' },
      { key: 'pmWins',          label: 'PM Wins' },
      { key: 'draws',           label: 'Draws' },
      { key: 'geopolBrier',     label: 'Geopol Brier' },
      { key: 'pmBrier',         label: 'PM Brier' },
      { key: 'rolling30Geopol', label: '30d Geopol' },
      { key: 'rolling30PM',     label: '30d PM' },
    ];
    for (const def of statDefs) {
      const card = createStatCard(def.label, '--');
      this.statRefs[def.key] = card.refs;
      statsBar.appendChild(card.el);
    }
    this.el.appendChild(statsBar);

    // Build table
    const tableWrap = h('div', { className: 'process-table-wrap' });
    const table = h('table', { className: 'process-table' }) as HTMLTableElement;

    // Thead
    const thead = document.createElement('thead');
    const headerRow = document.createElement('tr');
    for (const col of COLUMNS) {
      const th = document.createElement('th');
      th.textContent = col.label;
      if (col.sortable) {
        th.classList.add('sortable');
        if (col.key === this.sortColumn) {
          th.classList.add(this.sortDirection === 'asc' ? 'sort-asc' : 'sort-desc');
        }
        th.addEventListener('click', () => this.onSort(col.key));
        this.headerCells.set(col.key, th);
      }
      headerRow.appendChild(th);
    }
    thead.appendChild(headerRow);
    table.appendChild(thead);

    // Tbody
    this.tbody = document.createElement('tbody');
    table.appendChild(this.tbody);

    tableWrap.appendChild(table);
    this.el.appendChild(tableWrap);

    // Initial fetch
    await this.refresh();

    // Auto-refresh every 30s
    this.intervalId = setInterval(() => { void this.refresh(); }, 30_000);
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
    this.tbody = null;
    this.headerCells.clear();
    this.statRefs = {};
  }

  // -----------------------------------------------------------------------
  // Data refresh
  // -----------------------------------------------------------------------

  private async refresh(): Promise<void> {
    let data: AccuracyData;
    try {
      data = await this.client.getAccuracy();
    } catch {
      return; // Silently skip -- next refresh will retry
    }

    this.comparisons = data.comparisons;
    this.updateStats(data);
    this.renderTable();
  }

  // -----------------------------------------------------------------------
  // Stats update (in-place text node mutation)
  // -----------------------------------------------------------------------

  private updateStats(data: AccuracyData): void {
    const s = data.summary;
    this.setStatValue('totalResolved', String(s.total_resolved));
    this.setStatValue('totalVoided', String(s.total_voided));
    this.setStatValue('geopolWins', String(s.geopol_wins));
    this.setStatValue('pmWins', String(s.polymarket_wins));
    this.setStatValue('draws', String(s.draws));
    this.setStatValue('geopolBrier', formatBrier(s.geopol_cumulative_brier));
    this.setStatValue('pmBrier', formatBrier(s.polymarket_cumulative_brier));
    this.setStatValue('rolling30Geopol',
      s.rolling_30d_geopol_brier !== null
        ? `${formatBrier(s.rolling_30d_geopol_brier)} (${s.rolling_30d_count})`
        : '--',
    );
    this.setStatValue('rolling30PM',
      s.rolling_30d_polymarket_brier !== null
        ? `${formatBrier(s.rolling_30d_polymarket_brier)} (${s.rolling_30d_count})`
        : '--',
    );
  }

  private setStatValue(key: string, value: string): void {
    const refs = this.statRefs[key];
    if (refs) refs.valueNode.textContent = value;
  }

  // -----------------------------------------------------------------------
  // Sorting
  // -----------------------------------------------------------------------

  private onSort(column: SortColumn): void {
    if (this.sortColumn === column) {
      this.sortDirection = this.sortDirection === 'asc' ? 'desc' : 'asc';
    } else {
      this.sortColumn = column;
      this.sortDirection = 'desc';
    }

    // Update header classes
    for (const [key, th] of this.headerCells) {
      th.classList.remove('sort-asc', 'sort-desc');
      if (key === this.sortColumn) {
        th.classList.add(this.sortDirection === 'asc' ? 'sort-asc' : 'sort-desc');
      }
    }

    this.renderTable();
  }

  private sortedComparisons(): ResolvedComparison[] {
    const sorted = [...this.comparisons];
    const dir = this.sortDirection === 'asc' ? 1 : -1;
    const col = this.sortColumn;

    sorted.sort((a, b) => {
      const av = a[col];
      const bv = b[col];

      // Nulls sort last
      if (av === null && bv === null) return 0;
      if (av === null) return 1;
      if (bv === null) return -1;

      if (typeof av === 'number' && typeof bv === 'number') {
        return (av - bv) * dir;
      }
      return String(av).localeCompare(String(bv)) * dir;
    });

    return sorted;
  }

  // -----------------------------------------------------------------------
  // Table rendering
  // -----------------------------------------------------------------------

  private renderTable(): void {
    if (!this.tbody || !this.el) return;

    clearChildren(this.tbody);

    if (this.comparisons.length === 0) {
      // Show empty state
      let emptyEl = this.el.querySelector('.accuracy-empty') as HTMLElement | null;
      if (!emptyEl) {
        emptyEl = h('div', { className: 'accuracy-empty' },
          'No resolved comparisons yet. Accuracy data appears after Polymarket questions are resolved.',
        );
        this.el.appendChild(emptyEl);
      }
      return;
    }

    // Remove empty state if present
    const emptyEl = this.el.querySelector('.accuracy-empty');
    if (emptyEl) emptyEl.remove();

    const sorted = this.sortedComparisons();

    for (const comp of sorted) {
      const tr = document.createElement('tr');
      if (comp.status === 'voided') {
        tr.classList.add('voided');
      }

      // Question
      const tdQ = document.createElement('td');
      tdQ.classList.add('question-cell');
      tdQ.textContent = truncate(comp.polymarket_title, 50);
      if (comp.polymarket_title.length > 50) {
        tdQ.title = comp.polymarket_title;
      }
      tr.appendChild(tdQ);

      // Geopol P
      const tdGP = document.createElement('td');
      tdGP.textContent = formatPct(comp.geopol_probability);
      tr.appendChild(tdGP);

      // PM Price
      const tdPM = document.createElement('td');
      tdPM.textContent = formatPct(comp.polymarket_price);
      tr.appendChild(tdPM);

      // Outcome
      const tdOut = document.createElement('td');
      tdOut.textContent = outcomeLabel(comp.polymarket_outcome, comp.status);
      tr.appendChild(tdOut);

      // Geopol Brier
      const tdGB = document.createElement('td');
      tdGB.textContent = formatBrier(comp.geopol_brier);
      tr.appendChild(tdGB);

      // PM Brier
      const tdPB = document.createElement('td');
      tdPB.textContent = formatBrier(comp.polymarket_brier);
      tr.appendChild(tdPB);

      // Winner badge
      const tdW = document.createElement('td');
      const badge = h('span', { className: `winner-badge ${this.winnerClass(comp)}` },
        this.winnerLabel(comp),
      );
      tdW.appendChild(badge);
      tr.appendChild(tdW);

      // Country
      const tdC = document.createElement('td');
      tdC.textContent = comp.country_iso ?? '--';
      tr.appendChild(tdC);

      // Resolved
      const tdR = document.createElement('td');
      tdR.classList.add('proc-time');
      tdR.textContent = relativeTime(comp.resolved_at);
      tr.appendChild(tdR);

      this.tbody.appendChild(tr);
    }
  }

  private winnerClass(comp: ResolvedComparison): string {
    if (comp.status === 'voided') return 'voided';
    if (comp.winner === 'geopol') return 'geopol';
    if (comp.winner === 'polymarket') return 'polymarket';
    if (comp.winner === 'draw') return 'draw';
    return 'draw';
  }

  private winnerLabel(comp: ResolvedComparison): string {
    if (comp.status === 'voided') return 'V';
    if (comp.winner === 'geopol') return 'G';
    if (comp.winner === 'polymarket') return 'PM';
    if (comp.winner === 'draw') return '=';
    return '--';
  }
}
