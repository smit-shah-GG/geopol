/**
 * SettingsModal -- User preferences modal with Sources tab.
 *
 * Provides granular control over which news sources appear in the feed:
 * - Category-level toggle pills (Wire, Mainstream, Defense, etc.)
 * - Individual feed checkboxes with search filter
 * - Select All / Select None bulk actions
 * - Disabled sources persisted in localStorage as JSON array
 * - Dispatches `geopol:sources-changed` CustomEvent on toggle
 *
 * Ported from World Monitor UnifiedSettings, adapted for Geopol's
 * source tier system and dark analyst aesthetic.
 */

import { h, replaceChildren } from '@/utils/dom-utils';
import { trapFocus } from '@/utils/focus-trap';

// ---------------------------------------------------------------------------
// Source data (shared with NewsFeedPanel via static definition)
// ---------------------------------------------------------------------------

/** Category keys matching NewsFeedPanel's CATEGORIES. */
type SourceCategory =
  | 'Wire'
  | 'Mainstream'
  | 'Defense'
  | 'Think Tank'
  | 'Regional'
  | 'Crisis'
  | 'Finance'
  | 'Energy'
  | 'Government'
  | 'Intel';

interface SourceEntry {
  name: string;
  tier: 1 | 2 | 3 | 4;
  category: SourceCategory;
}

/**
 * Complete source catalog. Superset of SOURCE_TIERS in NewsFeedPanel.
 * Organized by category for the settings grid.
 */
const ALL_SOURCES: SourceEntry[] = [
  // Wire (Tier 1)
  { name: 'Reuters', tier: 1, category: 'Wire' },
  { name: 'AP News', tier: 1, category: 'Wire' },
  { name: 'AFP', tier: 1, category: 'Wire' },
  { name: 'Bloomberg', tier: 1, category: 'Wire' },
  { name: 'Wall Street Journal', tier: 1, category: 'Wire' },

  // Mainstream (Tier 2)
  { name: 'BBC World', tier: 2, category: 'Mainstream' },
  { name: 'BBC Middle East', tier: 2, category: 'Mainstream' },
  { name: 'Guardian World', tier: 2, category: 'Mainstream' },
  { name: 'CNN World', tier: 2, category: 'Mainstream' },
  { name: 'Al Jazeera', tier: 2, category: 'Mainstream' },
  { name: 'France 24', tier: 2, category: 'Mainstream' },
  { name: 'DW News', tier: 2, category: 'Mainstream' },
  { name: 'NPR News', tier: 2, category: 'Mainstream' },
  { name: 'Politico', tier: 2, category: 'Mainstream' },
  { name: 'Axios', tier: 2, category: 'Mainstream' },

  // Defense (Tier 2-3)
  { name: 'Military Times', tier: 2, category: 'Defense' },
  { name: 'Defense One', tier: 3, category: 'Defense' },
  { name: 'Breaking Defense', tier: 3, category: 'Defense' },
  { name: 'Janes', tier: 3, category: 'Defense' },

  // Think Tank (Tier 3)
  { name: 'Foreign Policy', tier: 3, category: 'Think Tank' },
  { name: 'Foreign Affairs', tier: 3, category: 'Think Tank' },
  { name: 'Atlantic Council', tier: 3, category: 'Think Tank' },
  { name: 'CSIS', tier: 3, category: 'Think Tank' },
  { name: 'RAND', tier: 3, category: 'Think Tank' },
  { name: 'Brookings', tier: 3, category: 'Think Tank' },
  { name: 'Carnegie', tier: 3, category: 'Think Tank' },
  { name: 'Bellingcat', tier: 3, category: 'Think Tank' },
  { name: 'The Diplomat', tier: 3, category: 'Think Tank' },

  // Crisis (Tier 3)
  { name: 'CrisisWatch', tier: 3, category: 'Crisis' },
  { name: 'ACLED', tier: 3, category: 'Crisis' },
  { name: 'UCDP', tier: 3, category: 'Crisis' },

  // Finance (Tier 1-2)
  { name: 'Financial Times', tier: 2, category: 'Finance' },
  { name: 'CNBC', tier: 2, category: 'Finance' },

  // Government (Tier 1)
  { name: 'UN News', tier: 1, category: 'Government' },
  { name: 'IAEA', tier: 1, category: 'Government' },
  { name: 'WHO', tier: 1, category: 'Government' },

  // Intel (Tier 3 -- state media, flagged separately)
  { name: 'Xinhua', tier: 3, category: 'Intel' },
  { name: 'TASS', tier: 3, category: 'Intel' },
  { name: 'RT', tier: 3, category: 'Intel' },
  { name: 'Sputnik', tier: 3, category: 'Intel' },
  { name: 'CGTN', tier: 3, category: 'Intel' },
  { name: 'Press TV', tier: 3, category: 'Intel' },
  { name: 'KCNA', tier: 3, category: 'Intel' },
  { name: 'Voice of America', tier: 3, category: 'Intel' },
  { name: 'TRT World', tier: 3, category: 'Intel' },
  { name: 'Al Arabiya', tier: 3, category: 'Intel' },
  { name: 'Kyiv Independent', tier: 3, category: 'Intel' },
];

/** Category list for the toggle pill bar. */
const CATEGORIES: SourceCategory[] = [
  'Wire', 'Mainstream', 'Defense', 'Think Tank', 'Regional',
  'Crisis', 'Finance', 'Energy', 'Government', 'Intel',
];

// ---------------------------------------------------------------------------
// localStorage persistence
// ---------------------------------------------------------------------------

const DISABLED_SOURCES_KEY = 'geopol-disabled-sources';

function getDisabledSources(): Set<string> {
  try {
    const raw = localStorage.getItem(DISABLED_SOURCES_KEY);
    if (!raw) return new Set();
    const arr = JSON.parse(raw) as string[];
    return new Set(arr);
  } catch {
    return new Set();
  }
}

function setDisabledSources(sources: Set<string>): void {
  localStorage.setItem(DISABLED_SOURCES_KEY, JSON.stringify(Array.from(sources)));
  // Dispatch event for NewsFeedPanel (listener wired in Plan 05)
  document.dispatchEvent(new CustomEvent('geopol:sources-changed', {
    detail: { disabled: Array.from(sources) },
  }));
}

// ---------------------------------------------------------------------------
// Gear icon SVG
// ---------------------------------------------------------------------------

const GEAR_SVG = `<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"/></svg>`;

// ---------------------------------------------------------------------------
// Helper
// ---------------------------------------------------------------------------

function escapeHtml(text: string): string {
  const div = document.createElement('div');
  div.appendChild(document.createTextNode(text));
  return div.innerHTML;
}

// ---------------------------------------------------------------------------
// SettingsModal
// ---------------------------------------------------------------------------

export class SettingsModal {
  private overlay: HTMLElement;
  private modal: HTMLElement | null = null;
  private escapeHandler: ((e: KeyboardEvent) => void) | null = null;
  private releaseTrap: (() => void) | null = null;
  private triggerElement: HTMLElement | null = null;
  private categoryFilter: SourceCategory | null = null;
  private searchFilter = '';
  private disabled: Set<string>;

  constructor() {
    this.disabled = getDisabledSources();

    // Overlay backdrop
    this.overlay = document.createElement('div');
    this.overlay.className = 'settings-modal-overlay';
    this.overlay.addEventListener('click', (e) => {
      if (e.target === this.overlay) this.close();
    });

    this.render();
  }

  // -------------------------------------------------------------------------
  // Open / Close
  // -------------------------------------------------------------------------

  public open(): void {
    // Capture trigger element for focus restoration on close
    this.triggerElement = document.activeElement as HTMLElement | null;

    // Re-read from localStorage in case another tab changed it
    this.disabled = getDisabledSources();
    this.render();
    this.overlay.classList.add('visible');
    document.body.appendChild(this.overlay);

    this.escapeHandler = (e: KeyboardEvent) => {
      if (e.key === 'Escape') this.close();
    };
    document.addEventListener('keydown', this.escapeHandler);

    // Activate focus trap after modal is in DOM
    if (this.modal) {
      this.releaseTrap = trapFocus(this.modal);
    }
  }

  public close(): void {
    // Release focus trap before removing DOM
    this.releaseTrap?.();
    this.releaseTrap = null;

    this.overlay.classList.remove('visible');
    if (this.overlay.parentNode) {
      this.overlay.parentNode.removeChild(this.overlay);
    }

    if (this.escapeHandler) {
      document.removeEventListener('keydown', this.escapeHandler);
      this.escapeHandler = null;
    }

    // Restore focus to the element that triggered the modal
    this.triggerElement?.focus();
    this.triggerElement = null;
  }

  // -------------------------------------------------------------------------
  // Settings gear button (for NavBar / dashboard header)
  // -------------------------------------------------------------------------

  public createTriggerButton(): HTMLElement {
    const btn = h('button', {
      className: 'settings-gear-btn',
      'aria-label': 'Settings',
    });
    btn.innerHTML = GEAR_SVG;
    btn.addEventListener('click', () => this.open());
    return btn;
  }

  // -------------------------------------------------------------------------
  // Rendering
  // -------------------------------------------------------------------------

  private render(): void {
    const disabledCount = this.disabled.size;

    // Build modal DOM
    this.modal = h('div', {
      className: 'settings-modal',
      role: 'dialog',
      'aria-modal': 'true',
      'aria-label': 'Settings',
    });

    // Header
    const header = h('div', { className: 'settings-modal-header' },
      h('span', { className: 'settings-modal-title' }, 'SETTINGS'),
      this.createCloseButton(),
    );
    this.modal.appendChild(header);

    // Tab bar (single "Sources" tab, extensible)
    const tabBar = h('div', { className: 'settings-tab-bar' },
      h('button', { className: 'settings-tab active', dataset: { tab: 'sources' } }, 'SOURCES'),
    );
    this.modal.appendChild(tabBar);

    // Sources tab content
    const tabContent = h('div', { className: 'settings-tab-content' });

    // Category toggle pills
    const categoryPills = h('div', { className: 'settings-category-pills' });
    this.buildCategoryPills(categoryPills);
    tabContent.appendChild(categoryPills);

    // Search input
    const searchRow = h('div', { className: 'settings-search-row' });
    const searchInput = h('input', {
      className: 'settings-search-input',
      type: 'text',
      placeholder: 'Filter sources...',
    }) as HTMLInputElement;
    searchInput.value = this.searchFilter;
    searchInput.addEventListener('input', () => {
      this.searchFilter = searchInput.value;
      this.renderSourceGrid();
    });
    searchRow.appendChild(searchInput);
    tabContent.appendChild(searchRow);

    // Source checkbox grid
    const grid = h('div', { className: 'settings-source-grid' });
    grid.id = 'settingsSourceGrid';
    tabContent.appendChild(grid);

    // Footer: select all/none + counter
    const footer = h('div', { className: 'settings-sources-footer' });

    const counter = h('span', { className: 'settings-sources-counter' });
    counter.id = 'settingsSourcesCounter';
    counter.textContent = disabledCount > 0
      ? `${disabledCount} source${disabledCount === 1 ? '' : 's'} hidden`
      : 'All sources enabled';
    footer.appendChild(counter);

    const selectAll = h('button', { className: 'settings-bulk-btn' }, 'Select All');
    selectAll.addEventListener('click', () => this.selectAll());
    footer.appendChild(selectAll);

    const selectNone = h('button', { className: 'settings-bulk-btn' }, 'Select None');
    selectNone.addEventListener('click', () => this.selectNone());
    footer.appendChild(selectNone);

    tabContent.appendChild(footer);
    this.modal.appendChild(tabContent);

    // Replace overlay contents
    replaceChildren(this.overlay, this.modal);

    // Render the source grid
    this.renderSourceGrid();
  }

  private createCloseButton(): HTMLElement {
    const btn = h('button', {
      className: 'settings-modal-close',
      'aria-label': 'Close settings',
    }, '\u00D7');
    btn.addEventListener('click', () => this.close());
    return btn;
  }

  // -------------------------------------------------------------------------
  // Category pills
  // -------------------------------------------------------------------------

  private buildCategoryPills(container: HTMLElement): void {
    // "All" pill
    const allPill = h('button', {
      className: `settings-cat-pill${this.categoryFilter === null ? ' active' : ''}`,
    }, 'ALL');
    allPill.addEventListener('click', () => {
      this.categoryFilter = null;
      this.updatePillActive(container);
      this.renderSourceGrid();
    });
    container.appendChild(allPill);

    for (const cat of CATEGORIES) {
      const pill = h('button', {
        className: `settings-cat-pill${this.categoryFilter === cat ? ' active' : ''}`,
        dataset: { category: cat },
      }, cat.toUpperCase());
      pill.addEventListener('click', () => {
        this.categoryFilter = cat;
        this.updatePillActive(container);
        this.renderSourceGrid();
      });
      container.appendChild(pill);
    }
  }

  private updatePillActive(container: HTMLElement): void {
    for (const pill of container.querySelectorAll('.settings-cat-pill')) {
      const el = pill as HTMLElement;
      const cat = el.dataset['category'] ?? null;
      el.classList.toggle('active', cat === (this.categoryFilter ?? null));
    }
  }

  // -------------------------------------------------------------------------
  // Source checkbox grid
  // -------------------------------------------------------------------------

  private getVisibleSources(): SourceEntry[] {
    let sources = ALL_SOURCES;

    // Filter by category
    if (this.categoryFilter) {
      sources = sources.filter((s) => s.category === this.categoryFilter);
    }

    // Filter by search
    if (this.searchFilter) {
      const lower = this.searchFilter.toLowerCase();
      sources = sources.filter((s) => s.name.toLowerCase().includes(lower));
    }

    return sources;
  }

  private renderSourceGrid(): void {
    const grid = document.getElementById('settingsSourceGrid');
    if (!grid) return;

    const visible = this.getVisibleSources();
    replaceChildren(grid);

    if (visible.length === 0) {
      grid.appendChild(
        h('div', { className: 'settings-empty' }, 'No sources match filter'),
      );
      return;
    }

    // Group by category for section headers
    const grouped = new Map<SourceCategory, SourceEntry[]>();
    for (const src of visible) {
      const group = grouped.get(src.category);
      if (group) {
        group.push(src);
      } else {
        grouped.set(src.category, [src]);
      }
    }

    for (const [category, sources] of grouped) {
      // Section header (only if showing all categories)
      if (!this.categoryFilter) {
        const sectionHeader = h('div', { className: 'settings-section-header' }, category);
        grid.appendChild(sectionHeader);
      }

      // Source checkboxes in a flex row
      const row = h('div', { className: 'settings-source-row' });
      for (const src of sources) {
        const isEnabled = !this.disabled.has(src.name);
        const item = h('div', {
          className: `settings-source-item${isEnabled ? ' enabled' : ''}`,
          dataset: { source: src.name },
        });

        const checkbox = h('div', { className: 'settings-source-checkbox' },
          isEnabled ? '\u2713' : '',
        );

        const label = h('span', { className: 'settings-source-label' },
          escapeHtml(src.name),
        );

        const tierBadge = h('span', {
          className: `settings-tier-badge tier-${src.tier}`,
        }, `T${src.tier}`);

        item.appendChild(checkbox);
        item.appendChild(label);
        item.appendChild(tierBadge);

        item.addEventListener('click', () => {
          this.toggleSource(src.name);
        });

        row.appendChild(item);
      }
      grid.appendChild(row);
    }
  }

  // -------------------------------------------------------------------------
  // Source toggle logic
  // -------------------------------------------------------------------------

  private toggleSource(name: string): void {
    if (this.disabled.has(name)) {
      this.disabled.delete(name);
    } else {
      this.disabled.add(name);
    }
    setDisabledSources(this.disabled);
    this.renderSourceGrid();
    this.updateCounter();
  }

  private selectAll(): void {
    const visible = this.getVisibleSources();
    for (const src of visible) {
      this.disabled.delete(src.name);
    }
    setDisabledSources(this.disabled);
    this.renderSourceGrid();
    this.updateCounter();
  }

  private selectNone(): void {
    const visible = this.getVisibleSources();
    for (const src of visible) {
      this.disabled.add(src.name);
    }
    setDisabledSources(this.disabled);
    this.renderSourceGrid();
    this.updateCounter();
  }

  private updateCounter(): void {
    const counter = document.getElementById('settingsSourcesCounter');
    if (!counter) return;
    const count = this.disabled.size;
    counter.textContent = count > 0
      ? `${count} source${count === 1 ? '' : 's'} hidden`
      : 'All sources enabled';
  }

  // -------------------------------------------------------------------------
  // Cleanup
  // -------------------------------------------------------------------------

  public destroy(): void {
    if (this.escapeHandler) {
      document.removeEventListener('keydown', this.escapeHandler);
      this.escapeHandler = null;
    }
    if (this.overlay.parentNode) {
      this.overlay.parentNode.removeChild(this.overlay);
    }
  }
}
