/**
 * NewsFeedPanel -- Clustered news feed with virtual scrolling, category pills,
 * source tier badges, and propaganda risk indicators.
 *
 * Consumes GET /api/v1/articles?sort=recent via forecastClient.getRecentArticles().
 * Articles are clustered by Jaccard similarity on title tokens, enriched with
 * source tier and propaganda risk data, and rendered through WindowedList for
 * scroll performance.
 *
 * Source filtering: reads disabled sources from localStorage (geopol-disabled-sources)
 * set by SettingsModal. Listens for geopol:sources-changed CustomEvent for live updates.
 *
 * Source health indicator: fetches public /api/v1/sources to show a subtle
 * "Some sources temporarily unavailable" label when data pipelines are unhealthy.
 *
 * Replaces EventTimelinePanel in dashboard Col 1 (Phase 21).
 */

import { Panel } from './Panel';
import { WindowedList } from './WindowedList';
import { forecastClient } from '@/services/forecast-client';
import { h, replaceChildren } from '@/utils/dom-utils';
import type { ArticleDTO } from '@/types/api';

// ---------------------------------------------------------------------------
// Source intelligence maps (ported from World Monitor feeds.ts, geopolitical subset)
// ---------------------------------------------------------------------------

/** Source tier: 1 = wire/official, 2 = major outlet, 3 = think tank/specialty, 4 = regional/other */
const SOURCE_TIERS: Record<string, number> = {
  // Tier 1 -- Wire services & official sources
  'Reuters': 1,
  'AP News': 1,
  'AFP': 1,
  'Bloomberg': 1,
  'Wall Street Journal': 1,
  'UN News': 1,
  'IAEA': 1,
  'WHO': 1,

  // Tier 2 -- Major outlets
  'BBC World': 2,
  'BBC Middle East': 2,
  'Guardian World': 2,
  'CNN World': 2,
  'Al Jazeera': 2,
  'Financial Times': 2,
  'France 24': 2,
  'DW News': 2,
  'NPR News': 2,
  'CNBC': 2,
  'Politico': 2,
  'Axios': 2,
  'Military Times': 2,

  // Tier 3 -- Think tanks & specialty
  'Defense One': 3,
  'Breaking Defense': 3,
  'Foreign Policy': 3,
  'The Diplomat': 3,
  'Bellingcat': 3,
  'Atlantic Council': 3,
  'CSIS': 3,
  'RAND': 3,
  'Brookings': 3,
  'Carnegie': 3,
  'CrisisWatch': 3,
  'Foreign Affairs': 3,
  'Janes': 3,

  // Tier 3 -- State media (included for awareness, flagged separately)
  'Xinhua': 3,
  'TASS': 3,
  'RT': 3,
};

/** Propaganda risk profile */
interface PropagandaRisk {
  risk: 'low' | 'medium' | 'high';
  state?: string;
  note?: string;
}

const PROPAGANDA_RISK: Record<string, PropagandaRisk> = {
  'Xinhua': { risk: 'high', state: 'China', note: 'Official CCP news agency' },
  'TASS': { risk: 'high', state: 'Russia', note: 'Russian state news agency' },
  'RT': { risk: 'high', state: 'Russia', note: 'Russian state media' },
  'Sputnik': { risk: 'high', state: 'Russia', note: 'Russian state media' },
  'CGTN': { risk: 'high', state: 'China', note: 'Chinese state broadcaster' },
  'Press TV': { risk: 'high', state: 'Iran', note: 'Iranian state media' },
  'KCNA': { risk: 'high', state: 'North Korea', note: 'North Korean state media' },

  'Al Jazeera': { risk: 'medium', state: 'Qatar', note: 'State-funded, independent editorial' },
  'Al Arabiya': { risk: 'medium', state: 'Saudi Arabia', note: 'Saudi-owned' },
  'TRT World': { risk: 'medium', state: 'Turkey', note: 'Turkish state broadcaster' },
  'France 24': { risk: 'medium', state: 'France', note: 'State-funded, independent editorial' },
  'DW News': { risk: 'medium', state: 'Germany', note: 'State-funded, independent editorial' },
  'Voice of America': { risk: 'medium', state: 'USA', note: 'US government-funded' },
  'Kyiv Independent': { risk: 'medium', note: 'Ukrainian perspective' },
};

function getSourceTier(sourceName: string): number {
  return SOURCE_TIERS[sourceName] ?? 4;
}

function getPropagandaRisk(sourceName: string): PropagandaRisk {
  return PROPAGANDA_RISK[sourceName] ?? { risk: 'low' };
}

// ---------------------------------------------------------------------------
// Category definitions for filter pills
// ---------------------------------------------------------------------------

type CategoryKey = 'all' | 'wire' | 'mainstream' | 'defense' | 'thinktank' | 'regional' | 'crisis' | 'finance';

interface CategoryDef {
  label: string;
  /** Source names belonging to this category (partial match on source_feed) */
  sources: string[];
}

const CATEGORIES: Record<CategoryKey, CategoryDef> = {
  all: { label: 'ALL', sources: [] },
  wire: { label: 'WIRE', sources: ['Reuters', 'AP News', 'AFP', 'Bloomberg'] },
  mainstream: { label: 'NEWS', sources: ['BBC', 'CNN', 'Guardian', 'NPR', 'Al Jazeera', 'France 24', 'DW'] },
  defense: { label: 'DEFENSE', sources: ['Defense One', 'Breaking Defense', 'Janes', 'Military Times', 'War Zone'] },
  thinktank: { label: 'THINK TANK', sources: ['Atlantic Council', 'CSIS', 'RAND', 'Brookings', 'Carnegie', 'Foreign Policy', 'Foreign Affairs', 'CrisisWatch'] },
  regional: { label: 'REGIONAL', sources: [] }, // Tier 4 catch-all
  crisis: { label: 'CRISIS', sources: ['CrisisWatch', 'ACLED', 'UCDP'] },
  finance: { label: 'FINANCE', sources: ['CNBC', 'Financial Times', 'Wall Street Journal', 'MarketWatch', 'Bloomberg'] },
};

// ---------------------------------------------------------------------------
// Article clustering
// ---------------------------------------------------------------------------

/** A cluster of articles about the same story. */
export interface ArticleCluster {
  primaryTitle: string;
  primaryUrl: string;
  primarySource: string;
  sourceCount: number;
  topSources: { name: string; tier: number }[];
  lastUpdated: Date;
  articles: ArticleDTO[];
  snippet: string;
}

/** Common English stop words for Jaccard filtering. */
const STOP_WORDS = new Set([
  'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
  'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been',
  'has', 'have', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
  'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those',
  'it', 'its', 'as', 'not', 'no', 'if', 'than', 'so', 'up', 'out',
  'about', 'into', 'over', 'after', 'before', 'between', 'under',
  'during', 'he', 'she', 'they', 'we', 'you', 'i', 'my', 'your',
  'his', 'her', 'their', 'our', 'all', 'each', 'every', 'both',
  'more', 'most', 'other', 'some', 'such', 'only', 'own', 'same',
  'new', 'old', 'just', 'also', 'now', 'says', 'said',
]);

/** Tokenize a title into a set of lowercase word stems (strip punctuation, drop stop words). */
function tokenize(text: string): Set<string> {
  const words = text
    .toLowerCase()
    .replace(/[^a-z0-9\s'-]/g, '')
    .split(/\s+/)
    .filter((w) => w.length > 2 && !STOP_WORDS.has(w));
  return new Set(words);
}

/** Jaccard similarity between two token sets. */
function jaccard(a: Set<string>, b: Set<string>): number {
  if (a.size === 0 && b.size === 0) return 0;
  let intersection = 0;
  for (const token of a) {
    if (b.has(token)) intersection++;
  }
  const union = a.size + b.size - intersection;
  return union === 0 ? 0 : intersection / union;
}

const JACCARD_THRESHOLD = 0.5;

/**
 * Cluster articles by title similarity using Jaccard on token sets.
 * O(n^2) but n is bounded by API limit (100 articles max).
 */
function clusterArticles(articles: ArticleDTO[]): ArticleCluster[] {
  if (articles.length === 0) return [];

  // Pre-tokenize all titles
  const tokenSets = articles.map((a) => tokenize(a.title));

  // Union-find for clustering
  const parent: number[] = articles.map((_, i) => i);
  function find(x: number): number {
    while (parent[x] !== x) {
      parent[x] = parent[parent[x]!]!; // path compression
      x = parent[x]!;
    }
    return x;
  }
  function union(a: number, b: number): void {
    const ra = find(a);
    const rb = find(b);
    if (ra !== rb) parent[ra] = rb;
  }

  // Pairwise similarity
  for (let i = 0; i < articles.length; i++) {
    for (let j = i + 1; j < articles.length; j++) {
      if (jaccard(tokenSets[i]!, tokenSets[j]!) >= JACCARD_THRESHOLD) {
        union(i, j);
      }
    }
  }

  // Group by root
  const groups = new Map<number, number[]>();
  for (let i = 0; i < articles.length; i++) {
    const root = find(i);
    const group = groups.get(root);
    if (group) {
      group.push(i);
    } else {
      groups.set(root, [i]);
    }
  }

  // Build clusters
  const clusters: ArticleCluster[] = [];
  for (const indices of groups.values()) {
    // Sort by published_at descending within cluster
    indices.sort((a, b) => {
      const dateA = articles[a]!.published_at ?? '';
      const dateB = articles[b]!.published_at ?? '';
      return dateB.localeCompare(dateA);
    });

    const primary = articles[indices[0]!]!;
    const clusterArticles = indices.map((i) => articles[i]!);

    // Unique sources with tiers
    const sourceMap = new Map<string, number>();
    for (const art of clusterArticles) {
      if (!sourceMap.has(art.source_feed)) {
        sourceMap.set(art.source_feed, getSourceTier(art.source_feed));
      }
    }

    const topSources = Array.from(sourceMap.entries())
      .map(([name, tier]) => ({ name, tier }))
      .sort((a, b) => a.tier - b.tier);

    const lastDate = clusterArticles.reduce((latest, art) => {
      const d = art.published_at ? new Date(art.published_at) : new Date(0);
      return d > latest ? d : latest;
    }, new Date(0));

    clusters.push({
      primaryTitle: primary.title,
      primaryUrl: primary.url,
      primarySource: primary.source_feed,
      sourceCount: sourceMap.size,
      topSources,
      lastUpdated: lastDate,
      articles: clusterArticles,
      snippet: primary.snippet?.slice(0, 150) ?? '',
    });
  }

  // Sort clusters by most recent article timestamp descending
  clusters.sort((a, b) => b.lastUpdated.getTime() - a.lastUpdated.getTime());

  return clusters;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function relativeTime(date: Date): string {
  const diff = Date.now() - date.getTime();
  const mins = Math.floor(diff / 60_000);
  if (mins < 1) return 'now';
  if (mins < 60) return `${mins}m`;
  const hrs = Math.floor(mins / 60);
  if (hrs < 24) return `${hrs}h`;
  const days = Math.floor(hrs / 24);
  return `${days}d`;
}

function escapeHtml(text: string): string {
  const div = document.createElement('div');
  div.appendChild(document.createTextNode(text));
  return div.innerHTML;
}

function tierLabel(tier: number): string {
  switch (tier) {
    case 1: return 'WIRE';
    case 2: return 'NEWS';
    case 3: return 'SPEC';
    default: return '';
  }
}

// ---------------------------------------------------------------------------
// Disabled source preferences (mirrors SettingsModal localStorage contract)
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

// ---------------------------------------------------------------------------
// Source health status (from public /api/v1/sources)
// ---------------------------------------------------------------------------

interface SourceHealthStatus {
  name: string;
  healthy: boolean;
  detail: string;
}

async function fetchSourceHealth(): Promise<SourceHealthStatus[]> {
  try {
    const resp = await fetch('/api/v1/sources');
    if (!resp.ok) return [];
    return (await resp.json()) as SourceHealthStatus[];
  } catch {
    return [];
  }
}

// ---------------------------------------------------------------------------
// Error classification
// ---------------------------------------------------------------------------

/** Transient errors get an amber toast; persistent errors get red. */
function isTransientError(err: unknown): boolean {
  const msg = err instanceof Error ? err.message.toLowerCase() : String(err).toLowerCase();
  return /timeout|503|502|504|econnrefused|network|fetch/.test(msg);
}

// ---------------------------------------------------------------------------
// NewsFeedPanel
// ---------------------------------------------------------------------------

export class NewsFeedPanel extends Panel {
  private categoryFilter: CategoryKey = 'all';
  private pillContainer: HTMLElement;
  private healthIndicator: HTMLElement;
  private listContainer: HTMLElement;
  private windowedList: WindowedList<ArticleCluster> | null = null;
  private rawArticles: ArticleDTO[] = [];
  private clusters: ArticleCluster[] = [];
  private disabledSources: Set<string> = getDisabledSources();
  /** Whether real content has been rendered at least once. */
  private hasData = false;
  private readonly onSourcesChanged = () => {
    this.disabledSources = getDisabledSources();
    this.renderArticles();
  };

  constructor() {
    super({ id: 'news-feed', title: 'NEWS FEED', showCount: true });

    // Category pill bar (between header and content)
    this.pillContainer = document.createElement('div');
    this.pillContainer.className = 'news-feed-pills';
    this.element.insertBefore(this.pillContainer, this.content);
    this.buildPills();

    // Source health indicator (below pills, above content)
    this.healthIndicator = document.createElement('div');
    this.healthIndicator.className = 'news-feed-health-indicator hidden';
    this.element.insertBefore(this.healthIndicator, this.content);

    // The content div becomes the scrollable list container
    this.listContainer = this.content;

    // Initialize windowed list
    this.windowedList = new WindowedList<ArticleCluster>(
      { container: this.listContainer, chunkSize: 8, bufferChunks: 1 },
      (cluster, _index) => this.renderClusterHtml(cluster),
    );

    // Listen for source preference changes from SettingsModal
    document.addEventListener('geopol:sources-changed', this.onSourcesChanged);
  }

  // -----------------------------------------------------------------------
  // Category pills
  // -----------------------------------------------------------------------

  private buildPills(): void {
    const keys = Object.keys(CATEGORIES) as CategoryKey[];
    for (const key of keys) {
      const pill = document.createElement('button');
      pill.className = `news-feed-pill${key === 'all' ? ' active' : ''}`;
      pill.textContent = CATEGORIES[key]!.label;
      pill.addEventListener('click', () => {
        this.categoryFilter = key;
        // Update active state
        for (const btn of this.pillContainer.querySelectorAll('.news-feed-pill')) {
          btn.classList.toggle('active', btn === pill);
        }
        this.renderArticles();
      });
      this.pillContainer.appendChild(pill);
    }
  }

  // -----------------------------------------------------------------------
  // Data refresh
  // -----------------------------------------------------------------------

  public async refresh(): Promise<void> {
    if (!this.hasData) {
      this.showSkeleton();
    }
    try {
      // Fetch articles and source health in parallel
      const [articles, sourceHealth] = await Promise.all([
        forecastClient.getRecentArticles(100),
        fetchSourceHealth(),
      ]);

      this.rawArticles = articles;
      this.hasData = true;
      this.dismissToast();
      // Enrich with source intelligence
      for (const art of this.rawArticles) {
        art.source_tier = getSourceTier(art.source_feed);
        art.propaganda_risk = getPropagandaRisk(art.source_feed).risk;
      }

      // Refresh disabled sources from localStorage
      this.disabledSources = getDisabledSources();

      this.setDataBadge('live');
      this.renderArticles();
      this.updateHealthIndicator(sourceHealth);
    } catch (err: unknown) {
      if (this.isAbortError(err)) return;
      console.error('[NewsFeedPanel] refresh failed:', err);
      if (this.hasData) {
        const severity = isTransientError(err) ? 'amber' : 'red';
        this.showRefreshToast('Failed to refresh -- showing cached data', severity);
      } else {
        this.showErrorWithRetry('Unable to load news', () => { void this.refresh(); });
      }
    }
  }

  // -----------------------------------------------------------------------
  // Rendering pipeline
  // -----------------------------------------------------------------------

  private renderArticles(): void {
    // Step 1: Filter articles by disabled source preferences
    let visible: ArticleDTO[];
    if (this.disabledSources.size > 0) {
      visible = this.rawArticles.filter(
        (a) => !this.disabledSources.has(a.source_feed),
      );
    } else {
      visible = this.rawArticles;
    }

    // Step 2: Filter by category
    let filtered: ArticleDTO[];
    if (this.categoryFilter === 'all') {
      filtered = visible;
    } else if (this.categoryFilter === 'regional') {
      // Regional = tier 4 (not in any named category)
      filtered = visible.filter(
        (a) => (a.source_tier ?? getSourceTier(a.source_feed)) >= 4,
      );
    } else {
      const categoryDef = CATEGORIES[this.categoryFilter]!;
      filtered = visible.filter((a) =>
        categoryDef.sources.some((src) =>
          a.source_feed.toLowerCase().includes(src.toLowerCase()),
        ),
      );
    }

    // Step 3: Cluster
    this.clusters = clusterArticles(filtered);
    this.setCount(this.clusters.length);

    if (this.clusters.length === 0) {
      if (this.categoryFilter !== 'all' || this.disabledSources.size > 0) {
        replaceChildren(
          this.listContainer,
          h('div', { className: 'empty-state-enhanced' },
            h('div', { className: 'empty-state-icon' }, '\u{1F50D}'),
            h('div', { className: 'empty-state-title' }, 'No Articles Match'),
            h('div', { className: 'empty-state-desc' }, 'No articles match the current filter. Try a different category.'),
          ),
        );
      } else {
        replaceChildren(
          this.listContainer,
          h('div', { className: 'empty-state-enhanced' },
            h('div', { className: 'empty-state-icon' }, '\u{1F4F0}'),
            h('div', { className: 'empty-state-title' }, 'No Articles'),
            h('div', { className: 'empty-state-desc' }, 'News articles from curated RSS sources will appear here.'),
          ),
        );
      }
      return;
    }

    // Step 4: Render via windowed list
    if (this.windowedList) {
      this.windowedList.setItems(this.clusters);
    }
  }

  // -----------------------------------------------------------------------
  // Cluster HTML rendering
  // -----------------------------------------------------------------------

  private renderClusterHtml(cluster: ArticleCluster): string {
    const tier = getSourceTier(cluster.primarySource);
    const propRisk = getPropagandaRisk(cluster.primarySource);

    // Tier badge
    const tLabel = tierLabel(tier);
    const tierBadgeHtml = tier <= 3
      ? `<span class="tier-badge tier-${tier}" title="Tier ${tier}: ${escapeHtml(tLabel)}">${tLabel}</span>`
      : '';

    // Source count badge with tooltip
    const sourceCountHtml = cluster.sourceCount > 1
      ? `<span class="source-count-badge" title="${escapeHtml(cluster.topSources.map((s) => s.name).join(', '))}">${cluster.sourceCount} sources</span>`
      : '';

    // Propaganda risk
    const propHtml = propRisk.risk !== 'low'
      ? `<span class="propaganda-badge ${propRisk.risk}" title="${escapeHtml(propRisk.note ?? propRisk.state ?? '')}">${propRisk.risk === 'high' ? 'STATE MEDIA' : 'CAUTION'}</span>`
      : '';

    const timeStr = relativeTime(cluster.lastUpdated);
    const snippetHtml = cluster.snippet
      ? `<div class="news-cluster-snippet">${escapeHtml(cluster.snippet)}</div>`
      : '';

    return `<div class="news-cluster">
      <div class="news-cluster-header">
        <a class="news-cluster-title" href="${escapeHtml(cluster.primaryUrl)}" target="_blank" rel="noopener noreferrer">${escapeHtml(cluster.primaryTitle)}</a>
      </div>
      <div class="news-cluster-meta">
        ${tierBadgeHtml}
        <span class="news-cluster-source">${escapeHtml(cluster.primarySource)}</span>
        ${sourceCountHtml}
        ${propHtml}
        <span class="news-cluster-time">${timeStr}</span>
      </div>
      ${snippetHtml}
    </div>`;
  }

  // -----------------------------------------------------------------------
  // Source health indicator
  // -----------------------------------------------------------------------

  private updateHealthIndicator(sources: SourceHealthStatus[]): void {
    const unhealthy = sources.filter((s) => !s.healthy);
    if (unhealthy.length === 0) {
      this.healthIndicator.classList.add('hidden');
      return;
    }

    // Build indicator with expandable detail
    this.healthIndicator.classList.remove('hidden');
    this.healthIndicator.innerHTML = '';

    const label = document.createElement('span');
    label.className = 'health-indicator-label';
    label.textContent = 'Some sources temporarily unavailable';
    label.title = unhealthy.map((s) => `${s.name}: ${s.detail}`).join('\n');

    const detailList = document.createElement('div');
    detailList.className = 'health-indicator-details hidden';
    for (const src of unhealthy) {
      const item = document.createElement('div');
      item.className = 'health-indicator-item';
      item.textContent = `${src.name}: ${src.detail}`;
      detailList.appendChild(item);
    }

    label.addEventListener('click', () => {
      detailList.classList.toggle('hidden');
    });

    this.healthIndicator.appendChild(label);
    this.healthIndicator.appendChild(detailList);
  }

  // -----------------------------------------------------------------------
  // Cleanup
  // -----------------------------------------------------------------------

  public override destroy(): void {
    document.removeEventListener('geopol:sources-changed', this.onSourcesChanged);
    this.windowedList?.destroy();
    this.windowedList = null;
    super.destroy();
  }
}
