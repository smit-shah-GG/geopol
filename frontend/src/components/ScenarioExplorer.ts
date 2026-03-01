/**
 * ScenarioExplorer -- Full-screen modal for interactive scenario tree exploration.
 *
 * Renders a forecast's scenario hierarchy as a vertical top-down d3 tree.
 * Evidence sidebar shows source cards for the selected branch node.
 * Opens via 'forecast-selected' CustomEvent dispatched from ForecastPanel.
 *
 * This is FE-04: the core explainability interface where analysts drill into
 * WHY a forecast exists and trace evidence provenance.
 */

import * as d3 from 'd3';
import { h, clearChildren } from '@/utils/dom-utils';
import type { ForecastResponse, ScenarioDTO, EvidenceDTO } from '@/types/api';

/** Maximum tree depth to render before showing a "+N deeper" indicator. */
const MAX_DEPTH = 4;

/** Truncate text to maxLen chars with ellipsis. */
function truncate(text: string, maxLen: number): string {
  return text.length > maxLen ? text.slice(0, maxLen - 1) + '\u2026' : text;
}

/** Format a probability as a percentage string. */
function pctLabel(p: number): string {
  return `${(p * 100).toFixed(0)}%`;
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
  if (lower.includes('tkg')) return 'TKG pattern';
  if (lower.includes('rag')) return 'RAG match';
  return source;
}

// ---- Hierarchy data shape for d3 ----

interface TreeDatum {
  name: string;
  probability: number;
  affirmative: boolean;
  scenario: ScenarioDTO | null;
  children: TreeDatum[];
  /** Indicates children were pruned at MAX_DEPTH. */
  pruned_count: number;
}

type TreeNode = d3.HierarchyPointNode<TreeDatum>;

// ---- ScenarioExplorer ----

export class ScenarioExplorer {
  private backdrop: HTMLElement | null = null;
  private modal: HTMLElement | null = null;
  private treeContainer: HTMLElement | null = null;
  private sidebar: HTMLElement | null = null;

  private readonly onForecastSelected: (e: Event) => void;
  private readonly onKeyDown: (e: KeyboardEvent) => void;

  constructor() {
    this.onForecastSelected = (e: Event) => {
      const detail = (e as CustomEvent<{ forecast: ForecastResponse }>).detail;
      this.open(detail.forecast);
    };

    this.onKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') this.close();
    };

    window.addEventListener('forecast-selected', this.onForecastSelected);
  }

  // ==================================================================
  // Public API
  // ==================================================================

  public open(forecast: ForecastResponse): void {
    this.buildModal(forecast);
    document.addEventListener('keydown', this.onKeyDown);
  }

  public close(): void {
    document.removeEventListener('keydown', this.onKeyDown);
    if (this.backdrop) {
      this.backdrop.remove();
      this.backdrop = null;
    }
    this.modal = null;
    this.treeContainer = null;
    this.sidebar = null;
  }

  public destroy(): void {
    this.close();
    window.removeEventListener('forecast-selected', this.onForecastSelected);
  }

  // ==================================================================
  // Modal construction
  // ==================================================================

  private buildModal(forecast: ForecastResponse): void {
    // Remove any existing modal before building a new one
    if (this.backdrop) this.backdrop.remove();

    // Backdrop -- click to close
    this.backdrop = h('div', { className: 'scenario-explorer-backdrop' });
    this.backdrop.addEventListener('click', (e: MouseEvent) => {
      if (e.target === this.backdrop) this.close();
    });

    // Modal container
    this.modal = h('div', { className: 'scenario-explorer-modal' });

    // Header
    const probColor = forecast.probability >= 0.7
      ? 'var(--prob-high)'
      : forecast.probability >= 0.4
        ? 'var(--prob-medium)'
        : 'var(--prob-low)';

    const header = h('div', { className: 'scenario-header' },
      h('div', { className: 'scenario-header-left' },
        h('span', { className: 'scenario-header-question' }, truncate(forecast.question, 120)),
        h('span', {
          className: 'scenario-header-prob',
          style: `color: ${probColor}`,
        }, pctLabel(forecast.probability)),
      ),
      h('button', {
        className: 'scenario-close-btn',
        'aria-label': 'Close scenario explorer',
        onclick: () => this.close(),
      }, '\u00d7'),
    );

    // Tree container (left 70%)
    this.treeContainer = h('div', { className: 'scenario-tree-container' });

    // Evidence sidebar (right 30%)
    this.sidebar = h('div', { className: 'scenario-evidence-sidebar' },
      h('div', { className: 'scenario-sidebar-placeholder' },
        'Click a scenario node to view evidence',
      ),
    );

    // Content row: tree + sidebar
    const contentRow = h('div', { className: 'scenario-content-row' },
      this.treeContainer,
      this.sidebar,
    );

    this.modal.appendChild(header);
    this.modal.appendChild(contentRow);
    this.backdrop.appendChild(this.modal);
    document.body.appendChild(this.backdrop);

    // Render the tree after the modal is in the DOM
    this.renderTree(forecast);
  }

  // ==================================================================
  // d3 tree rendering
  // ==================================================================

  private buildTreeData(forecast: ForecastResponse): TreeDatum {
    const convertScenario = (s: ScenarioDTO, depth: number): TreeDatum => {
      let children: TreeDatum[] = [];
      let prunedCount = 0;

      if (depth < MAX_DEPTH) {
        children = s.child_scenarios.map(c => convertScenario(c, depth + 1));
      } else {
        prunedCount = this.countDescendants(s);
      }

      return {
        name: s.description,
        probability: s.probability,
        affirmative: s.answers_affirmative,
        scenario: s,
        children,
        pruned_count: prunedCount,
      };
    };

    return {
      name: forecast.question,
      probability: forecast.probability,
      affirmative: true,
      scenario: null,
      children: forecast.scenarios.map(s => convertScenario(s, 1)),
      pruned_count: 0,
    };
  }

  /** Count all nested descendants of a scenario (for pruned indicator). */
  private countDescendants(s: ScenarioDTO): number {
    let count = s.child_scenarios.length;
    for (const child of s.child_scenarios) {
      count += this.countDescendants(child);
    }
    return count;
  }

  private renderTree(forecast: ForecastResponse): void {
    if (!this.treeContainer) return;
    clearChildren(this.treeContainer);

    const treeData = this.buildTreeData(forecast);
    const root = d3.hierarchy(treeData);

    // Layout: vertical top-down, nodeSize controls spacing
    const treeLayout = d3.tree<TreeDatum>().nodeSize([200, 100]);
    treeLayout(root);

    const nodes = root.descendants() as TreeNode[];
    const links = root.links() as d3.HierarchyPointLink<TreeDatum>[];

    // Compute bounding box
    let minX = Infinity, maxX = -Infinity;
    let minY = Infinity, maxY = -Infinity;
    for (const node of nodes) {
      if (node.x < minX) minX = node.x;
      if (node.x > maxX) maxX = node.x;
      if (node.y < minY) minY = node.y;
      if (node.y > maxY) maxY = node.y;
    }

    const pad = 80;
    const width = maxX - minX + pad * 2;
    const height = maxY - minY + pad * 2;

    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.setAttribute('viewBox', `${minX - pad} ${minY - pad} ${width} ${height}`);
    svg.setAttribute('width', '100%');
    svg.setAttribute('height', '100%');
    svg.style.minWidth = `${Math.max(width, 600)}px`;
    svg.style.minHeight = `${Math.max(height, 400)}px`;

    // Defs for glow filter
    const defs = document.createElementNS('http://www.w3.org/2000/svg', 'defs');
    defs.innerHTML = `
      <filter id="node-glow" x="-50%" y="-50%" width="200%" height="200%">
        <feGaussianBlur stdDeviation="3" result="blur"/>
        <feMerge>
          <feMergeNode in="blur"/>
          <feMergeNode in="SourceGraphic"/>
        </feMerge>
      </filter>
    `;
    svg.appendChild(defs);

    // Draw links
    const linkGen = d3.linkVertical<d3.HierarchyPointLink<TreeDatum>, d3.HierarchyPointNode<TreeDatum>>()
      .x(d => d.x)
      .y(d => d.y);

    for (const link of links) {
      const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
      const pathD = linkGen(link);
      if (pathD) path.setAttribute('d', pathD);
      path.setAttribute('fill', 'none');
      path.setAttribute('stroke', 'var(--border)');
      path.setAttribute('stroke-width', '1.5');
      path.setAttribute('opacity', '0.6');
      svg.appendChild(path);
    }

    // Draw nodes
    for (const node of nodes) {
      const g = document.createElementNS('http://www.w3.org/2000/svg', 'g');
      g.setAttribute('transform', `translate(${node.x},${node.y})`);
      g.style.cursor = 'pointer';
      g.classList.add('scenario-node');

      // Circle sized by probability
      const radius = Math.min(8 + node.data.probability * 20, 28);
      const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
      circle.setAttribute('r', String(radius));
      circle.setAttribute('fill', node.data.affirmative
        ? 'var(--semantic-critical)'
        : 'var(--accent)');
      circle.setAttribute('opacity', '0.8');
      circle.setAttribute('stroke', 'var(--border)');
      circle.setAttribute('stroke-width', '1');
      g.appendChild(circle);

      // Probability label above
      const probText = document.createElementNS('http://www.w3.org/2000/svg', 'text');
      probText.setAttribute('y', String(-radius - 6));
      probText.setAttribute('text-anchor', 'middle');
      probText.setAttribute('fill', 'var(--text-muted)');
      probText.setAttribute('font-size', '10');
      probText.setAttribute('font-family', 'var(--font-mono)');
      probText.textContent = pctLabel(node.data.probability);
      g.appendChild(probText);

      // Description label below
      const descText = document.createElementNS('http://www.w3.org/2000/svg', 'text');
      descText.setAttribute('y', String(radius + 16));
      descText.setAttribute('text-anchor', 'middle');
      descText.setAttribute('fill', 'var(--text-secondary)');
      descText.setAttribute('font-size', '11');
      descText.setAttribute('font-family', 'var(--font-mono)');
      descText.textContent = truncate(node.data.name, 40);
      g.appendChild(descText);

      // Pruned indicator
      if (node.data.pruned_count > 0) {
        const pruneText = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        pruneText.setAttribute('y', String(radius + 30));
        pruneText.setAttribute('text-anchor', 'middle');
        pruneText.setAttribute('fill', 'var(--text-muted)');
        pruneText.setAttribute('font-size', '9');
        pruneText.setAttribute('font-family', 'var(--font-mono)');
        pruneText.textContent = `+${node.data.pruned_count} deeper`;
        g.appendChild(pruneText);
      }

      // Click handler -> populate sidebar
      g.addEventListener('click', (e: MouseEvent) => {
        e.stopPropagation();
        this.selectNode(node, svg);
      });

      svg.appendChild(g);
    }

    this.treeContainer.appendChild(svg);
  }

  // ==================================================================
  // Node selection + sidebar population
  // ==================================================================

  private selectNode(node: TreeNode, svg: SVGSVGElement): void {
    // Highlight selected node
    const allNodes = svg.querySelectorAll('.scenario-node circle');
    for (const c of allNodes) {
      c.removeAttribute('filter');
      c.setAttribute('stroke', 'var(--border)');
      c.setAttribute('stroke-width', '1');
    }

    // Find node by position
    for (const g of svg.querySelectorAll('.scenario-node')) {
      const transform = g.getAttribute('transform') ?? '';
      const match = transform.match(/translate\(([-\d.]+),([-\d.]+)\)/);
      if (match) {
        const gx = parseFloat(match[1] ?? '0');
        const gy = parseFloat(match[2] ?? '0');
        if (Math.abs(gx - node.x) < 0.5 && Math.abs(gy - node.y) < 0.5) {
          const circle = g.querySelector('circle');
          if (circle) {
            circle.setAttribute('filter', 'url(#node-glow)');
            circle.setAttribute('stroke', 'var(--accent)');
            circle.setAttribute('stroke-width', '2');
          }
          break;
        }
      }
    }

    this.populateSidebar(node);
  }

  private populateSidebar(node: TreeNode): void {
    if (!this.sidebar) return;
    clearChildren(this.sidebar);

    const scenario = node.data.scenario;

    // Header section: description + probability
    const header = h('div', { className: 'sidebar-node-header' },
      h('div', { className: 'sidebar-node-description' },
        scenario ? scenario.description : node.data.name,
      ),
      h('div', {
        className: 'sidebar-node-prob',
        style: `color: ${node.data.affirmative ? 'var(--semantic-critical)' : 'var(--accent)'}`,
      }, pctLabel(node.data.probability)),
    );
    this.sidebar.appendChild(header);

    if (!scenario) {
      // Root node -- no detailed evidence
      this.sidebar.appendChild(
        h('div', { className: 'scenario-sidebar-placeholder' },
          'Root forecast node. Select a scenario branch to view evidence.',
        ),
      );
      return;
    }

    // Entities as badges
    if (scenario.entities.length > 0) {
      const entitySection = h('div', { className: 'sidebar-section' },
        h('div', { className: 'sidebar-section-label' }, 'Entities'),
        h('div', { className: 'entity-badge-row' },
          ...scenario.entities.map(e => h('span', { className: 'entity-badge' }, e)),
        ),
      );
      this.sidebar.appendChild(entitySection);
    }

    // Timeline
    if (scenario.timeline.length > 0) {
      const timelineSection = h('div', { className: 'sidebar-section' },
        h('div', { className: 'sidebar-section-label' }, 'Timeline'),
        h('ol', { className: 'sidebar-timeline' },
          ...scenario.timeline.map(t => h('li', null, t)),
        ),
      );
      this.sidebar.appendChild(timelineSection);
    }

    // Evidence sources
    const evidenceLabel = h('div', { className: 'sidebar-section-label' },
      `Evidence Sources (${scenario.evidence_sources.length})`,
    );
    this.sidebar.appendChild(h('div', { className: 'sidebar-section' }, evidenceLabel));

    if (scenario.evidence_sources.length === 0) {
      this.sidebar.appendChild(
        h('div', { className: 'scenario-sidebar-placeholder' },
          'No evidence available for this branch',
        ),
      );
    } else {
      for (const ev of scenario.evidence_sources) {
        this.sidebar.appendChild(this.buildEvidenceCard(ev));
      }
    }
  }

  private buildEvidenceCard(ev: EvidenceDTO): HTMLElement {
    const badgeCls = `source-badge source-badge--${sourceClass(ev.source)}`;

    // Confidence bar width
    const confPct = `${(ev.confidence * 100).toFixed(0)}%`;

    const card = h('div', { className: 'evidence-source-card' },
      h('div', { className: 'evidence-card-header' },
        h('span', { className: badgeCls }, sourceLabel(ev.source)),
        h('span', { className: 'evidence-confidence-label' }, confPct),
      ),
      h('div', { className: 'evidence-description' }, ev.description),
      h('div', { className: 'evidence-confidence-bar-track' },
        h('div', {
          className: 'evidence-confidence-bar-fill',
          style: `width: ${confPct}`,
        }),
      ),
    );

    // Optional timestamp
    if (ev.timestamp) {
      const dateStr = new Date(ev.timestamp).toLocaleDateString('en-US', {
        year: 'numeric', month: 'short', day: 'numeric',
      });
      card.appendChild(
        h('div', { className: 'evidence-meta' }, dateStr),
      );
    }

    // Optional GDELT Event ID
    if (ev.gdelt_event_id) {
      card.appendChild(
        h('div', { className: 'evidence-meta evidence-gdelt-id' },
          h('span', null, 'GDELT: '),
          h('code', null, ev.gdelt_event_id),
        ),
      );
    }

    return card;
  }
}
