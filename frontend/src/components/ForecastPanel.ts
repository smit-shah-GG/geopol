/**
 * ForecastPanel -- expandable forecast cards with progressive disclosure.
 *
 * Collapsed card: question + probability bar + country + age.
 * Expanded card: two-column layout with ensemble weights, calibration metadata,
 *   mini d3 scenario tree, evidence summaries, and "View Full Analysis" button.
 *
 * Key behaviors:
 * - Multiple cards expandable simultaneously (NOT accordion).
 * - Expanded state survives 60-second data refresh via diff-based DOM updates.
 * - Listens to search-results/search-cleared events from SearchBar.
 * - "View Full Analysis" dispatches forecast-selected event for ScenarioExplorer.
 *
 * Fulfills FUX-01 (progressive disclosure) requirements.
 */

import { Panel } from './Panel';
import { h, replaceChildren, clearChildren } from '@/utils/dom-utils';
import { forecastClient } from '@/services/forecast-client';
import type { ForecastResponse, EvidenceDTO, ScenarioDTO, SearchResponse } from '@/types/api';
import * as d3 from 'd3';

// ---------------------------------------------------------------------------
// Utility functions
// ---------------------------------------------------------------------------

/** Severity tier derived from probability value. */
function severityClass(p: number): string {
  if (p > 0.8) return 'critical';
  if (p > 0.6) return 'high';
  if (p > 0.4) return 'elevated';
  if (p > 0.2) return 'normal';
  return 'low';
}

/** Convert ISO code to flag emoji via regional indicator symbols. */
function isoToFlag(iso: string): string {
  const upper = iso.toUpperCase();
  if (upper.length !== 2) return '';
  const a = upper.codePointAt(0);
  const b = upper.codePointAt(1);
  if (a === undefined || b === undefined) return '';
  return String.fromCodePoint(a + 0x1F1A5) + String.fromCodePoint(b + 0x1F1A5);
}

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

/** Truncate text with ellipsis. */
function truncate(text: string, maxLen: number): string {
  return text.length > maxLen ? text.slice(0, maxLen - 1) + '\u2026' : text;
}

/** Format probability as percentage string. */
function pctLabel(p: number): string {
  return `${(p * 100).toFixed(1)}%`;
}

/** Extract 2-letter country ISO from calibration category. */
function extractCountryIso(f: ForecastResponse): string {
  return f.calibration.category.length === 2 ? f.calibration.category : '';
}

/** Map evidence source string to a display badge class suffix. */
function sourceClass(source: string): string {
  const lower = source.toLowerCase();
  if (lower.includes('gdelt')) return 'gdelt';
  if (lower.includes('tkg')) return 'tkg';
  if (lower.includes('rag')) return 'rag';
  return 'default';
}

/** Map evidence source string to a display label. */
function sourceLabel(source: string): string {
  const lower = source.toLowerCase();
  if (lower.includes('gdelt')) return 'GDELT';
  if (lower.includes('tkg')) return 'TKG';
  if (lower.includes('rag')) return 'RAG';
  return source;
}

// ---------------------------------------------------------------------------
// Mini tree hierarchy conversion (shared shape with ScenarioExplorer)
// ---------------------------------------------------------------------------

interface MiniTreeDatum {
  name: string;
  probability: number;
  affirmative: boolean;
  children: MiniTreeDatum[];
}

function buildMiniTreeData(f: ForecastResponse): MiniTreeDatum {
  const convert = (s: ScenarioDTO, depth: number): MiniTreeDatum => ({
    name: s.description,
    probability: s.probability,
    affirmative: s.answers_affirmative,
    // Only go 2 levels deep for the mini preview
    children: depth < 2 ? s.child_scenarios.map(c => convert(c, depth + 1)) : [],
  });

  return {
    name: f.question,
    probability: f.probability,
    affirmative: true,
    children: f.scenarios.map(s => convert(s, 1)),
  };
}

// ---------------------------------------------------------------------------
// ForecastPanel
// ---------------------------------------------------------------------------

export class ForecastPanel extends Panel {
  /** IDs of currently expanded cards. */
  private expandedIds = new Set<string>();

  /** Full forecast list from last update() call. */
  private allForecasts: ForecastResponse[] = [];

  /** Map of forecast_id -> card DOM element for diff-based updates. */
  private cardElements = new Map<string, HTMLElement>();

  /** Whether we're showing search results (vs. full list). */
  private isSearchActive = false;

  private readonly onSearchResults: EventListener;
  private readonly onSearchCleared: EventListener;

  constructor() {
    super({ id: 'forecasts', title: 'ACTIVE FORECASTS', showCount: true });

    this.onSearchResults = ((e: CustomEvent<SearchResponse>) => {
      this.isSearchActive = true;
      const forecasts = e.detail.results.map(r => r.forecast);
      this.renderFilteredForecasts(forecasts);
    }) as EventListener;

    this.onSearchCleared = (() => {
      this.isSearchActive = false;
      this.renderFilteredForecasts(this.allForecasts);
    }) as EventListener;

    // Listen on window so SearchBar events (which bubble) reach us
    window.addEventListener('search-results', this.onSearchResults);
    window.addEventListener('search-cleared', this.onSearchCleared);
  }

  /** Self-fetch via forecastClient. Used by RefreshScheduler. */
  public async refresh(): Promise<void> {
    this.showLoading();
    try {
      const forecasts = await forecastClient.getTopForecasts(10);
      this.applyBreakerBadge('forecast');
      this.update(forecasts);
    } catch (err: unknown) {
      if (this.isAbortError(err)) return;
      console.error('[ForecastPanel] refresh failed:', err);
      this.showError('Failed to load forecasts');
    }
  }

  /**
   * External data injection -- diff-based update preserving expanded state.
   * Called by dashboard-screen refresh scheduler every 60s.
   */
  public update(forecasts: ForecastResponse[]): void {
    this.allForecasts = forecasts;

    // If search is active, don't overwrite search results display
    if (this.isSearchActive) return;

    this.renderFilteredForecasts(forecasts);
  }

  public override destroy(): void {
    window.removeEventListener('search-results', this.onSearchResults);
    window.removeEventListener('search-cleared', this.onSearchCleared);
    super.destroy();
  }

  // -----------------------------------------------------------------------
  // Private: rendering
  // -----------------------------------------------------------------------

  private applyBreakerBadge(endpoint: 'forecast' | 'country' | 'health'): void {
    const state = forecastClient.getDataState(endpoint);
    this.setDataBadge(state.mode);
  }

  /**
   * Diff-based render: add new cards, remove stale ones, update existing ones
   * in-place without destroying DOM (preserving expanded state).
   */
  private renderFilteredForecasts(forecasts: ForecastResponse[]): void {
    // Sort by recency (newest first)
    const sorted = [...forecasts].sort(
      (a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime(),
    );
    this.setCount(sorted.length);

    if (sorted.length === 0) {
      this.cardElements.clear();
      replaceChildren(this.content,
        h('div', { className: 'empty-state' },
          this.isSearchActive ? 'No forecasts match your search' : 'No active forecasts',
        ),
      );
      return;
    }

    const newIds = new Set(sorted.map(f => f.forecast_id));
    const existingIds = new Set(this.cardElements.keys());

    // Remove cards no longer in list
    for (const id of existingIds) {
      if (!newIds.has(id)) {
        const el = this.cardElements.get(id);
        el?.remove();
        this.cardElements.delete(id);
        this.expandedIds.delete(id);
      }
    }

    // Build ordered list: update existing, create new
    const orderedCards: HTMLElement[] = [];
    for (const f of sorted) {
      const existing = this.cardElements.get(f.forecast_id);
      if (existing) {
        // Update data in-place without destroying DOM
        this.updateCardInPlace(existing, f);
        orderedCards.push(existing);
      } else {
        const card = this.buildCard(f);
        this.cardElements.set(f.forecast_id, card);
        orderedCards.push(card);
      }
    }

    // Reorder DOM to match sorted order
    const frag = document.createDocumentFragment();
    for (const card of orderedCards) {
      frag.appendChild(card);
    }
    clearChildren(this.content);
    this.content.appendChild(frag);
  }

  /** Update an existing card's data (probability, time) without destroying DOM structure. */
  private updateCardInPlace(card: HTMLElement, f: ForecastResponse): void {
    // Update probability badge
    const badge = card.querySelector('.probability-badge');
    if (badge) badge.textContent = pctLabel(f.probability);

    // Update probability fill bar
    const fill = card.querySelector('.probability-fill') as HTMLElement | null;
    if (fill) {
      const sev = severityClass(f.probability);
      fill.style.width = `${f.probability * 100}%`;
      fill.className = `probability-fill severity-${sev}`;
    }

    // Update time
    const time = card.querySelector('.forecast-time');
    if (time) time.textContent = relativeTime(f.created_at);

    // If expanded, update expanded content too
    if (this.expandedIds.has(f.forecast_id)) {
      const expandedContent = card.querySelector('.expanded-content');
      if (expandedContent) {
        expandedContent.remove();
        card.appendChild(this.buildExpandedContent(f));
      }
    }

    // Store updated reference for "View Full Analysis" click
    card.dataset['forecastData'] = JSON.stringify(f);
  }

  // -----------------------------------------------------------------------
  // Card construction
  // -----------------------------------------------------------------------

  private buildCard(f: ForecastResponse): HTMLElement {
    const sev = severityClass(f.probability);
    const pct = pctLabel(f.probability);
    const question = f.question.length > 100
      ? f.question.slice(0, 97) + '...'
      : f.question;

    const countryIso = extractCountryIso(f);
    const flag = countryIso ? isoToFlag(countryIso) : '';

    const card = h('div', {
      className: 'forecast-card',
      dataset: { id: f.forecast_id, forecastData: JSON.stringify(f) },
    },
      h('div', { className: 'forecast-card-header' },
        h('div', { className: 'forecast-question' }, question),
        h('div', { className: 'forecast-bar-row' },
          h('div', { className: 'probability-bar' },
            h('div', {
              className: `probability-fill severity-${sev}`,
              style: `width: ${f.probability * 100}%`,
            }),
          ),
          h('span', { className: 'probability-badge' }, pct),
        ),
        h('div', { className: 'forecast-meta' },
          flag ? h('span', { className: 'forecast-country' }, `${flag} ${countryIso}`) : null,
          h('span', { className: 'forecast-confidence' }, `conf: ${f.confidence.toFixed(2)}`),
          h('span', { className: 'forecast-scenarios' }, `${f.scenarios.length} scenarios`),
          h('span', { className: 'forecast-time' }, relativeTime(f.created_at)),
        ),
      ),
    );

    // Click header area to toggle expansion
    const header = card.querySelector('.forecast-card-header') as HTMLElement;
    header.addEventListener('click', (e: MouseEvent) => {
      // Don't toggle if clicking "View Full Analysis" button
      if ((e.target as HTMLElement).closest('.view-full-btn')) return;
      this.toggleCard(f.forecast_id, card);
    });

    // If this card was previously expanded, re-expand it
    if (this.expandedIds.has(f.forecast_id)) {
      card.classList.add('expanded');
      card.appendChild(this.buildExpandedContent(f));
    }

    return card;
  }

  // -----------------------------------------------------------------------
  // Expansion toggle
  // -----------------------------------------------------------------------

  private toggleCard(id: string, card: HTMLElement): void {
    if (this.expandedIds.has(id)) {
      // Collapse
      this.expandedIds.delete(id);
      card.classList.remove('expanded');
      const expandedContent = card.querySelector('.expanded-content');
      expandedContent?.remove();
    } else {
      // Expand -- get current data from dataset
      this.expandedIds.add(id);
      card.classList.add('expanded');
      const dataStr = card.dataset['forecastData'];
      if (!dataStr) return;
      const forecast = JSON.parse(dataStr) as ForecastResponse;
      card.appendChild(this.buildExpandedContent(forecast));
    }
  }

  // -----------------------------------------------------------------------
  // Expanded content (two-column layout)
  // -----------------------------------------------------------------------

  private buildExpandedContent(f: ForecastResponse): HTMLElement {
    const sev = severityClass(f.probability);
    const ens = f.ensemble_info;
    const cal = f.calibration;

    // -- Left column: probability + ensemble + calibration --
    const llmPct = (ens.llm_probability * 100).toFixed(1);
    const tkgPct = ens.tkg_probability !== null
      ? (ens.tkg_probability * 100).toFixed(1)
      : 'N/A';

    // Ensemble stacked bar
    const llmWeight = (ens.weights['llm'] ?? 0.5) * 100;
    const tkgWeight = 100 - llmWeight;

    const leftCol = h('div', { className: 'expanded-left' },
      // Larger probability bar
      h('div', { className: 'expanded-prob-section' },
        h('div', { className: 'expanded-prob-label' }, 'Probability'),
        h('div', { className: 'expanded-prob-bar-row' },
          h('div', { className: 'probability-bar expanded-prob-bar' },
            h('div', {
              className: `probability-fill severity-${sev}`,
              style: `width: ${f.probability * 100}%`,
            }),
          ),
          h('span', { className: 'expanded-prob-value' }, pctLabel(f.probability)),
        ),
      ),

      // Ensemble weights
      h('div', { className: 'expanded-ensemble-section' },
        h('div', { className: 'expanded-section-label' }, 'Ensemble Weights'),
        h('div', { className: 'expanded-ensemble-bar' },
          h('div', {
            className: 'expanded-ensemble-llm',
            style: `width: ${llmWeight}%`,
          }),
          h('div', {
            className: 'expanded-ensemble-tkg',
            style: `width: ${tkgWeight}%`,
          }),
        ),
        h('div', { className: 'expanded-ensemble-legend' },
          h('span', { className: 'expanded-ensemble-item' },
            h('span', { className: 'expanded-dot expanded-dot-llm' }),
            `LLM ${llmPct}%`,
          ),
          h('span', { className: 'expanded-ensemble-item' },
            h('span', { className: 'expanded-dot expanded-dot-tkg' }),
            `TKG ${tkgPct}%`,
          ),
        ),
      ),

      // Calibration metadata
      h('div', { className: 'expanded-cal-section' },
        h('div', { className: 'expanded-section-label' }, 'Calibration'),
        h('div', { className: 'expanded-cal-grid' },
          h('span', { className: 'expanded-cal-key' }, 'Category'),
          h('span', { className: 'expanded-cal-val' }, cal.category),
          h('span', { className: 'expanded-cal-key' }, 'Temperature'),
          h('span', { className: 'expanded-cal-val' }, cal.temperature.toFixed(3)),
          h('span', { className: 'expanded-cal-key' }, 'Hist. accuracy'),
          h('span', { className: 'expanded-cal-val' }, (cal.historical_accuracy * 100).toFixed(1) + '%'),
          cal.brier_score !== null
            ? h('span', { className: 'expanded-cal-key' }, 'Brier score')
            : null,
          cal.brier_score !== null
            ? h('span', { className: 'expanded-cal-val' }, cal.brier_score.toFixed(4))
            : null,
        ),
      ),
    );

    // -- Right column: mini tree + evidence + "View Full Analysis" --
    const rightCol = h('div', { className: 'expanded-right' });

    // Mini scenario tree
    if (f.scenarios.length > 0) {
      const treeWrapper = h('div', { className: 'mini-tree-wrapper' });
      this.renderMiniTree(treeWrapper, f);
      rightCol.appendChild(treeWrapper);
    }

    // Top 3 evidence summaries (collect from top-level scenarios)
    const allEvidence: EvidenceDTO[] = [];
    for (const s of f.scenarios) {
      for (const ev of s.evidence_sources) {
        allEvidence.push(ev);
      }
    }
    const topEvidence = allEvidence
      .sort((a, b) => b.confidence - a.confidence)
      .slice(0, 3);

    if (topEvidence.length > 0) {
      const evidenceSection = h('div', { className: 'expanded-evidence-section' },
        h('div', { className: 'expanded-section-label' }, `Evidence (${allEvidence.length})`),
      );
      for (const ev of topEvidence) {
        evidenceSection.appendChild(this.buildMiniEvidenceCard(ev));
      }
      rightCol.appendChild(evidenceSection);
    }

    // "View Full Analysis" button
    const viewBtn = h('button', { className: 'view-full-btn' }, 'View Full Analysis');
    viewBtn.addEventListener('click', (e: MouseEvent) => {
      e.stopPropagation();
      this.element.dispatchEvent(
        new CustomEvent('forecast-selected', {
          detail: { forecast: f },
          bubbles: true,
        }),
      );
    });
    rightCol.appendChild(viewBtn);

    return h('div', { className: 'expanded-content' },
      leftCol,
      rightCol,
    );
  }

  // -----------------------------------------------------------------------
  // Mini evidence card (compact)
  // -----------------------------------------------------------------------

  private buildMiniEvidenceCard(ev: EvidenceDTO): HTMLElement {
    return h('div', { className: 'mini-evidence-card' },
      h('div', { className: 'mini-evidence-header' },
        h('span', { className: `source-badge source-badge--${sourceClass(ev.source)}` },
          sourceLabel(ev.source),
        ),
        h('span', { className: 'mini-evidence-conf' },
          `${(ev.confidence * 100).toFixed(0)}%`,
        ),
      ),
      h('div', { className: 'mini-evidence-desc' },
        truncate(ev.description, 120),
      ),
    );
  }

  // -----------------------------------------------------------------------
  // Mini d3 scenario tree (~150px preview)
  // -----------------------------------------------------------------------

  private renderMiniTree(container: HTMLElement, f: ForecastResponse): void {
    const treeData = buildMiniTreeData(f);
    const root = d3.hierarchy(treeData);

    const treeLayout = d3.tree<MiniTreeDatum>().nodeSize([80, 50]);
    treeLayout(root);

    type MiniTreeNode = d3.HierarchyPointNode<MiniTreeDatum>;
    const nodes = root.descendants() as MiniTreeNode[];
    const links = root.links() as d3.HierarchyPointLink<MiniTreeDatum>[];

    // Compute bounding box
    let minX = Infinity, maxX = -Infinity;
    let minY = Infinity, maxY = -Infinity;
    for (const node of nodes) {
      if (node.x < minX) minX = node.x;
      if (node.x > maxX) maxX = node.x;
      if (node.y < minY) minY = node.y;
      if (node.y > maxY) maxY = node.y;
    }

    const pad = 30;
    const width = maxX - minX + pad * 2;
    const height = maxY - minY + pad * 2;

    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.setAttribute('viewBox', `${minX - pad} ${minY - pad} ${width} ${height}`);
    svg.setAttribute('width', '100%');
    svg.setAttribute('height', '150');
    svg.classList.add('mini-tree-svg');

    // Draw links
    const linkGen = d3.linkVertical<d3.HierarchyPointLink<MiniTreeDatum>, d3.HierarchyPointNode<MiniTreeDatum>>()
      .x(d => d.x)
      .y(d => d.y);

    for (const link of links) {
      const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
      const pathD = linkGen(link);
      if (pathD) path.setAttribute('d', pathD);
      path.setAttribute('fill', 'none');
      path.setAttribute('stroke', 'var(--border)');
      path.setAttribute('stroke-width', '1');
      path.setAttribute('opacity', '0.5');
      svg.appendChild(path);
    }

    // Draw nodes
    for (const node of nodes) {
      const g = document.createElementNS('http://www.w3.org/2000/svg', 'g');
      g.setAttribute('transform', `translate(${node.x},${node.y})`);

      const radius = Math.min(4 + node.data.probability * 10, 8);
      const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
      circle.setAttribute('r', String(radius));
      circle.setAttribute('fill', node.data.affirmative
        ? 'var(--semantic-critical)'
        : 'var(--accent)');
      circle.setAttribute('opacity', '0.8');
      g.appendChild(circle);

      // Short label below (only for non-root, first 20 chars)
      if (node.depth > 0) {
        const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        label.setAttribute('y', String(radius + 10));
        label.setAttribute('text-anchor', 'middle');
        label.setAttribute('fill', 'var(--text-muted)');
        label.setAttribute('font-size', '8');
        label.setAttribute('font-family', 'var(--font-mono)');
        label.textContent = truncate(node.data.name, 20);
        g.appendChild(label);
      }

      svg.appendChild(g);
    }

    container.appendChild(svg);
  }
}
