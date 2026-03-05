# Phase 21: Source Expansion & Feed Management - Research

**Researched:** 2026-03-05
**Domain:** Frontend news panels (WM port), admin feed management, cross-source dedup, RSS pipeline refactoring
**Confidence:** HIGH (primary source: actual codebase reading of both geopol and WM reference)

## Summary

Phase 21 is a major frontend-heavy phase that replaces the information-sparse EventTimelinePanel with a full WM-style news ecosystem (NewsFeedPanel, LiveStreamsPanel, BreakingNewsBanner), adds user source controls, moves SourcesPanel to admin-only, refactors RSS feed config from hardcoded Python to a PostgreSQL `rss_feeds` table with admin CRUD, and adds cross-source dedup at the knowledge graph layer.

The WM reference implementation is well-understood from direct code reading: NewsPanel (654 lines) uses `analysisWorker.clusterNews()` for web-worker-based clustering, `WindowedList` for virtual scrolling, `getSourceTier()`/`getSourcePropagandaRisk()` for credibility badges, and `enrichWithVelocityML()` for velocity tracking. The BreakingNewsBanner (297 lines) is a standalone class appended to `document.body` using CustomEvents (`wm:breaking-news`). LiveNewsPanel (1200+ lines) embeds YouTube iframes via the YouTube IFrame API and HLS via native `<video>` or hls.js, with idle detection and region pills.

The geopol frontend is vanilla TypeScript with a Panel base class pattern, 4-column flexbox layout, `h()` DOM helper, and RefreshScheduler for periodic data refresh. The admin panel uses `AdminClient` with `X-Admin-Key` auth and section-based panel mounting. The RSS pipeline currently uses hardcoded `FeedSource` dataclasses in `feed_config.py`, ingests via `RSSDaemon.poll_feeds()`, and indexes into ChromaDB `rss_articles` collection with `ArticleIndexer`.

**Primary recommendation:** Port WM patterns with geopol-specific simplifications -- no i18n, no web worker for clustering (inline is fine for geopol's feed volume), use existing Panel base class, and leverage the existing ChromaDB article pipeline as the data source for NewsFeedPanel.

## Standard Stack

### Core (Already in Codebase)
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| Panel base class | N/A | All dashboard panels extend this | Established pattern in geopol |
| `h()` DOM helper | N/A | `@/utils/dom-utils` | All panels use this for DOM construction |
| AdminClient | N/A | Admin API calls with X-Admin-Key | Phase 19-20 pattern |
| ChromaDB | Installed | Article storage and retrieval | RSS pipeline already indexes here |
| SQLAlchemy 2.0 | Installed | PostgreSQL ORM for rss_feeds table | All DB models use this |
| Alembic | Installed | DB migrations | Migration infrastructure exists |
| APScheduler 3.11.2 | Installed | RSS job scheduling | Phase 20 scheduler package |

### New (Frontend)
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| YouTube IFrame API | External CDN | YouTube live stream embedding | LiveStreamsPanel |
| hls.js | npm | HLS stream playback in browsers without native HLS | LiveStreamsPanel for non-Safari |

### Not Needed
| Instead of | Don't Use | Reason |
|------------|-----------|--------|
| Web Workers | analysisWorker | WM uses a web worker for clustering -- overkill for geopol's feed volume. Inline clustering is sufficient. |
| trafilatura (new) | Already present | Article extraction already works via `article_processor.py` |
| sentence-transformers (new) | Already present | Embeddings already computed via ChromaDB's SentenceTransformerEmbeddingFunction |

## Architecture Patterns

### Dashboard Layout Change
```
CURRENT (15-35-30-20):
Col 1 (15%): RiskIndexPanel
Col 2 (35%): SearchBar, ForecastPanel, ComparisonPanel
Col 3 (30%): MyForecastsPanel, SourcesPanel
Col 4 (20%): EventTimelinePanel, SystemHealthPanel, PolymarketPanel

NEW (25-30-30-15):
Col 1 (25%): NewsFeedPanel, LiveStreamsPanel
Col 2 (30%): SearchBar, ForecastPanel, ComparisonPanel (unchanged)
Col 3 (30%): MyForecastsPanel (SourcesPanel REMOVED)
Col 4 (15%): RiskIndexPanel (moved+compacted), SystemHealthPanel, PolymarketPanel
+ BreakingNewsBanner (full-width, above columns)
- EventTimelinePanel (DELETED)
- SourcesPanel (moved to admin-only, already exists as SourceManager)
```

**Files to modify:**
- `frontend/src/styles/main.css` lines 253-256: Change column widths from `15/35/30/20` to `25/30/30/15`
- `frontend/src/screens/dashboard-screen.ts`: Rewire panel mounting to new columns, add NewsFeedPanel/LiveStreamsPanel/BreakingNewsBanner, remove EventTimelinePanel/SourcesPanel
- `frontend/src/app/panel-layout.ts`: Update comment header

### Pattern 1: NewsFeedPanel (WM NewsPanel Port)
**What:** Clustered news feed showing articles from ChromaDB, with source tier badges, velocity indicators, sentiment, propaganda risk, and virtual scrolling.
**Data source:** Existing `/api/v1/articles` endpoint (keyword mode), extended to return recent articles sorted by time. The ChromaDB `rss_articles` collection stores all ingested articles with `source_name`, `title`, `source_url`, `published_at` metadata.
**Clustering:** Simple title-similarity clustering inline (no web worker). Group articles by normalized headline similarity. WM uses `analysisWorker.clusterNews()` but that requires a separate worker build.
**Virtual scrolling:** Port WM's `WindowedList<T>` class (chunk-based, variable height). WM's `VirtualList` is fixed-height; `WindowedList` is the right choice for news items.

```typescript
// Pattern: NewsFeedPanel extends Panel
export class NewsFeedPanel extends Panel {
  private windowedList: WindowedList<PreparedCluster> | null = null;
  private categoryFilter: string = 'all';

  constructor() {
    super({ id: 'news-feed', title: 'NEWS FEED', showCount: true });
    this.initCategoryPills();
    this.initWindowedList();
  }

  public async refresh(): Promise<void> {
    const articles = await forecastClient.getArticles({ limit: 100 });
    const clusters = clusterArticles(articles.items);
    this.renderClusters(clusters);
  }
}
```

### Pattern 2: BreakingNewsBanner (WM Port)
**What:** Full-width banner above dashboard columns. Triggered by CustomEvent.
**WM pattern:** Appends container to `document.body`, listens for `wm:breaking-news` CustomEvent. Max 3 concurrent alerts. Auto-dismiss: 60s critical, 30s high. Sound with 5-min cooldown. Visibility API pause/resume of dismiss timers.
**Geopol adaptation:** Use `geopol:breaking-news` event name. Trigger from two sources:
1. GDELT events with extreme Goldstein scale values (checked during refresh)
2. Forecast probability spikes (detected during forecast refresh)

```typescript
// Pattern: BreakingNewsBanner is standalone, not a Panel
const banner = new BreakingNewsBanner();
// Triggered via:
document.dispatchEvent(new CustomEvent('geopol:breaking-news', {
  detail: { id, headline, source, threatLevel: 'critical'|'high', timestamp }
}));
```

### Pattern 3: LiveStreamsPanel
**What:** YouTube iframes + HLS streams with region pills and idle detection.
**WM pattern:** Loads YouTube IFrame API script dynamically, creates `YT.Player` instances. HLS via direct `<video>` tag (Safari) or hls.js fallback. Idle detection pauses after 5 min.
**Geopol adaptation:** Curate 10-20 channels from WM's 160+ list. Keep the same `LiveChannel` interface. Store channel list in localStorage (no backend persistence needed).

### Pattern 4: User Source Controls (localStorage)
**What:** Category pills on NewsFeedPanel + settings modal Sources tab.
**WM pattern:** `UnifiedSettings` with Sources tab: region pills, search input, checkbox grid, Select All/None. `getDisabledSources()` returns `Set<string>`, stored in localStorage.
**Geopol adaptation:** Create a `SettingsModal` with Sources tab. Store disabled sources in `localStorage.getItem('geopol-disabled-sources')`. Category pills directly on NewsFeedPanel filter the displayed articles without changing the ingestion.

### Pattern 5: Admin Feed Management (PostgreSQL)
**What:** CRUD for RSS feeds via admin panel, replacing hardcoded `feed_config.py`.
**Schema:**
```sql
CREATE TABLE rss_feeds (
  id SERIAL PRIMARY KEY,
  name VARCHAR(255) NOT NULL UNIQUE,
  url TEXT NOT NULL,
  tier INTEGER NOT NULL DEFAULT 2 CHECK (tier IN (1, 2)),
  category VARCHAR(50) NOT NULL DEFAULT 'regional',
  lang VARCHAR(10) NOT NULL DEFAULT 'en',
  enabled BOOLEAN NOT NULL DEFAULT true,
  last_poll_at TIMESTAMPTZ,
  last_error TEXT,
  error_count INTEGER NOT NULL DEFAULT 0,
  articles_24h INTEGER NOT NULL DEFAULT 0,
  articles_total INTEGER NOT NULL DEFAULT 0,
  avg_articles_per_poll REAL NOT NULL DEFAULT 0.0,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  deleted_at TIMESTAMPTZ  -- soft delete
);
```
**Migration:** Alembic migration that creates the table AND seeds it from current `feed_config.py` `ALL_FEEDS` list.
**RSS daemon change:** `RSSDaemon.poll_feeds()` must read from `rss_feeds` table instead of `TIER_1_FEEDS`/`TIER_2_FEEDS` constants. The `job_wrappers.py` functions `rss_poll_tier1()` and `rss_poll_tier2()` must query the DB for enabled feeds of the appropriate tier.

### Pattern 6: Cross-Source Dedup at Graph Insertion
**What:** Prevent GDELT + ACLED duplicate events from creating duplicate knowledge graph triples.
**Current state:** `deduplication.py` already does GDELT-internal dedup using `(actor1_code, actor2_code, event_code, location)` content hash + time window. The knowledge graph builder (`graph_builder.py`) calls `add_event_from_db_row()` per event -- there's no cross-source awareness.
**New layer:** Before `TemporalKnowledgeGraph.add_event_from_db_row()`, compute a cross-source fingerprint: `(date, country_iso, event_type)` where `event_type` maps CAMEO codes to a coarser category. When a fingerprint collision is found across sources, prefer ACLED (human-coded). Log every suppressed duplicate with both event IDs.

### Anti-Patterns to Avoid
- **Don't re-fetch articles from the network in NewsFeedPanel** -- use the existing ChromaDB-backed `/api/v1/articles` endpoint. The RSS daemon already handles ingestion.
- **Don't embed the YouTube API script at page load** -- load it lazily when LiveStreamsPanel mounts (WM does this).
- **Don't store user source preferences in the backend** -- no user accounts exist, use localStorage only.
- **Don't auto-disable feeds without audit trail** -- every auto-disable must log the reason and error count.
- **Don't use innerHTML for user-controlled content** -- WM uses `escapeHtml()` throughout. Geopol has `safeHtml()` in `@/utils/dom-utils`.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Virtual scrolling | Custom scroll handler | Port WM's `WindowedList<T>` | Handles chunk rendering, scroll RAF, buffer zones, cleanup |
| YouTube embedding | Raw iframes | YouTube IFrame API (`YT.Player`) | Handles ready state, error recovery, quality control, mute/unmute |
| HLS playback | Custom video code | hls.js library | HLS is not natively supported in Chrome/Firefox without a library |
| RSS parsing | Custom XML parser | feedparser (already used) | Handles all RSS/Atom variants |
| Article extraction | Custom scraping | trafilatura (already used) | Handles boilerplate removal, encoding, etc. |
| Breaking news sound | Custom audio code | WM's base64 WAV approach | Tiny inline WAV avoids CORS and loading issues |

**Key insight:** WM has already solved every frontend problem this phase faces. The task is porting, not inventing.

## Common Pitfalls

### Pitfall 1: ChromaDB Article Query Performance
**What goes wrong:** `/api/v1/articles` currently uses `collection.get()` which returns unordered results. NewsFeedPanel needs recent articles sorted by time.
**Why it happens:** ChromaDB `get()` doesn't support ORDER BY. The current endpoint has no time-sorted retrieval.
**How to avoid:** Extend the articles endpoint to support `sort=recent` mode. Fetch with `where={"published_at": {"$gte": cutoff_24h}}` and sort Python-side. Consider adding a `GET /api/v1/articles/recent` convenience endpoint.
**Warning signs:** NewsFeedPanel shows articles in random order.

### Pitfall 2: YouTube IFrame API Loading Race
**What goes wrong:** `YT.Player` constructor fails if called before `onYouTubeIframeAPIReady`.
**Why it happens:** YouTube IFrame API is loaded asynchronously. If the panel tries to create players before the script loads, it crashes.
**How to avoid:** WM pattern: set `window.onYouTubeIframeAPIReady` callback before injecting the script tag. Queue player creation until callback fires.
**Warning signs:** "YT is not defined" errors in console.

### Pitfall 3: Feed Config Migration Data Loss
**What goes wrong:** Switching from `feed_config.py` to DB without seeding the table leaves the RSS daemon with zero feeds.
**Why it happens:** The daemon starts reading from `rss_feeds` table which is empty.
**How to avoid:** Alembic migration must seed the `rss_feeds` table from the current `ALL_FEEDS` list as part of the migration. Make the daemon fall back to `feed_config.py` if the table is empty (graceful degradation).
**Warning signs:** Zero articles ingested after deploy.

### Pitfall 4: HLS CORS Issues
**What goes wrong:** HLS streams fail to load due to CORS restrictions.
**Why it happens:** Many HLS manifests don't serve CORS headers for browser playback.
**How to avoid:** Use YouTube fallback for channels where HLS fails. WM's `DIRECT_HLS_MAP` has curated URLs that work, but some may break over time. For geopol's curated 10-20 channels, prefer YouTube-only initially.
**Warning signs:** Video element shows error, console shows CORS blocked.

### Pitfall 5: BreakingNewsBanner Stacking with Dashboard
**What goes wrong:** Banner overlaps dashboard content or causes layout shift.
**Why it happens:** Banner is appended to `document.body` and uses absolute/fixed positioning. Dashboard doesn't account for its height.
**How to avoid:** WM uses `--breaking-alert-offset` CSS custom property and `has-breaking-alert` body class. Dashboard columns need `padding-top` or `margin-top` that responds to this variable.
**Warning signs:** First dashboard row hidden behind banner.

### Pitfall 6: Cross-Source Dedup False Merges
**What goes wrong:** Conservative fingerprint `(date, country, event_type)` merges distinct events in the same country on the same day.
**Why it happens:** Multiple real events of the same type can happen in one country on one day (e.g., two separate protests in different cities).
**How to avoid:** The CONTEXT.md explicitly says "miss some dupes rather than false-merge distinct events." Keep the fingerprint conservative. Log all suppressed duplicates so false merges can be audited and the algorithm tuned.
**Warning signs:** Event count drops dramatically after enabling dedup.

### Pitfall 7: Admin CSS Loading Order
**What goes wrong:** Admin feed management panel styles don't load.
**Why it happens:** Memory note: "Admin CSS loading: must import admin-styles.css BEFORE rendering AuthModal." Same applies to new feed management panel.
**How to avoid:** Ensure admin-styles.css import happens before any admin panel DOM is created. The dynamic import boundary is `admin-screen.ts`.
**Warning signs:** Unstyled admin panel content.

## Code Examples

### WindowedList Port (from WM VirtualList.ts)
```typescript
// Source: /home/kondraki/personal/worldmonitor/src/components/VirtualList.ts
// The WindowedList<T> class (lines 244-406) should be ported to:
// frontend/src/components/WindowedList.ts
// Key interface:
export class WindowedList<T> {
  constructor(
    options: { container: HTMLElement; chunkSize?: number; bufferChunks?: number },
    renderItem: (item: T, index: number) => string,
    onRendered?: () => void
  );
  setItems(items: T[]): void;  // Full replace + re-render
  refresh(): void;              // Re-render visible chunks
  destroy(): void;              // Cleanup scroll listeners
}
```

### NewsFeedPanel Article Clustering (Simplified)
```typescript
// Simplified clustering -- no web worker needed for geopol's volume
interface ClusteredArticle {
  primaryTitle: string;
  primaryUrl: string;
  primarySource: string;
  sourceCount: number;
  topSources: { name: string; tier: number }[];
  lastUpdated: Date;
  articles: ArticleDTO[];
}

function clusterArticles(articles: ArticleDTO[]): ClusteredArticle[] {
  // Group by normalized title similarity (Jaccard on word tokens)
  // Threshold: 0.6 similarity -> same cluster
  // Sort clusters by most recent article timestamp
}
```

### Admin Feed Management Card
```typescript
// Extends existing SourceManager pattern (frontend/src/admin/panels/SourceManager.ts)
// New SourceManager replaces current simple card grid with richer metadata:
interface FeedInfo {
  id: number;
  name: string;
  url: string;
  tier: number;
  category: string;
  lang: string;
  enabled: boolean;
  last_poll_at: string | null;
  last_error: string | null;
  error_count: number;
  articles_24h: number;
  articles_total: number;
  avg_articles_per_poll: number;
}
// AdminClient gets new methods:
// getFeeds(): Promise<FeedInfo[]>
// addFeed(data: AddFeedRequest): Promise<FeedInfo>
// updateFeed(id: number, data: UpdateFeedRequest): Promise<FeedInfo>
// deleteFeed(id: number, purge?: boolean): Promise<void>
```

### Cross-Source Dedup Fingerprint
```python
# Source: new code in src/deduplication.py or new src/knowledge_graph/cross_source_dedup.py
import hashlib
from typing import Optional

def cross_source_fingerprint(
    event_date: str,  # YYYY-MM-DD
    country_iso: Optional[str],
    event_type: str,  # Coarse category from CAMEO code
) -> str:
    """Generate (date, country, event_type) fingerprint for cross-source dedup."""
    date_part = event_date[:10] if event_date else "unknown"
    country = (country_iso or "UNK").upper()
    etype = event_type.upper()
    raw = f"{date_part}|{country}|{etype}"
    return hashlib.sha256(raw.encode()).hexdigest()[:32]

def cameo_to_coarse_type(cameo_code: str) -> str:
    """Map CAMEO event code to coarse category for dedup fingerprinting."""
    prefix = int(cameo_code[:2]) if cameo_code and cameo_code[:2].isdigit() else 0
    if prefix <= 5: return "cooperation"
    if prefix <= 9: return "diplomacy"
    if prefix <= 14: return "conflict"
    return "force"
```

### Breaking News Trigger (GDELT + Forecast)
```typescript
// In dashboard-screen.ts, after pushing forecast/event data:
function checkBreakingNews(events: EventDTO[]): void {
  for (const evt of events) {
    if (evt.goldstein_scale !== null && Math.abs(evt.goldstein_scale) >= 8.0) {
      document.dispatchEvent(new CustomEvent('geopol:breaking-news', {
        detail: {
          id: `gdelt-${evt.id}`,
          headline: evt.title || `${evt.source} event: ${evt.event_code}`,
          source: evt.source,
          threatLevel: Math.abs(evt.goldstein_scale) >= 9.0 ? 'critical' : 'high',
          timestamp: new Date(evt.event_date),
        }
      }));
    }
  }
}
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Hardcoded feeds in `feed_config.py` | DB-backed `rss_feeds` table with admin CRUD | This phase | Admin can add/remove feeds without code deploy |
| EventTimelinePanel (raw GDELT events) | NewsFeedPanel (clustered articles with metadata) | This phase | Rich news experience replaces sparse event list |
| SourcesPanel in user dashboard | Admin-only source health | This phase | Cleaner user UI, operational controls stay in admin |
| No cross-source dedup | Fingerprint-based dedup at graph insertion | This phase | Prevents double-counting events in knowledge graph |

**Preserved:**
- RSS daemon architecture (tier-based polling) -- unchanged, just reads from DB instead of constants
- ChromaDB article storage -- unchanged, remains the data source
- ArticleIndexer and extraction pipeline -- unchanged
- Admin panel structure (AdminSidebar + section routing) -- extended with richer feed management

## Open Questions

1. **hls.js Dependency**
   - What we know: WM uses native HLS for Safari + hls.js for other browsers. WM's HLS URLs are curated and some may be geo-restricted.
   - What's unclear: Whether geopol should include hls.js as a dependency or just use YouTube-only for the initial 10-20 channels.
   - Recommendation: Start YouTube-only for all channels (simpler). Add HLS later if needed for channels that don't have YouTube streams.

2. **AI Summary Implementation**
   - What we know: WM's NewsPanel has a "summarize" button that calls `generateSummary()` with top 8 headlines. This uses an LLM endpoint.
   - What's unclear: Whether geopol should expose a similar LLM summary endpoint. The Gemini budget (25/day) is already tight.
   - Recommendation: Implement the UI button but make it optional/disabled by default. Can use the existing Gemini integration if budget allows.

3. **Article Country Tagging**
   - What we know: ChromaDB articles currently have no `country_iso` metadata (the articles endpoint returns `country_iso: null`). NewsFeedPanel category pills can't filter by country/region without this.
   - What's unclear: Whether to add country extraction to the RSS pipeline or filter by source category only.
   - Recommendation: Filter by source category (wire/mainstream/defense/etc.) and feed name, not by country. Country extraction is a separate concern.

4. **Propaganda Risk Data Source**
   - What we know: `feed_config.py` has a minimal `PROPAGANDA_RISK` dict (6 entries). WM's `feeds.ts` has a comprehensive `SOURCE_PROPAGANDA_RISK` map (~20 entries with detailed profiles).
   - What's unclear: Whether to port WM's full propaganda risk data or keep geopol's minimal version.
   - Recommendation: Port WM's full `SOURCE_PROPAGANDA_RISK` data. It's static config data, relevant for geopolitical credibility assessment.

5. **Feed Auto-Disable Threshold**
   - What we know: CONTEXT.md says "Feed auto-disable on N consecutive failures."
   - What's unclear: What N should be.
   - Recommendation: Use N=5 (same as JobFailureTracker in `src/scheduler/retry.py`). Expose as a config setting via system_config.

## Sources

### Primary (HIGH confidence)
- `/home/kondraki/personal/worldmonitor/src/components/NewsPanel.ts` -- 654 lines, full clustering/rendering implementation
- `/home/kondraki/personal/worldmonitor/src/components/BreakingNewsBanner.ts` -- 297 lines, full alert system
- `/home/kondraki/personal/worldmonitor/src/components/LiveNewsPanel.ts` -- 1200+ lines, YouTube/HLS embedding
- `/home/kondraki/personal/worldmonitor/src/components/VirtualList.ts` -- 406 lines, VirtualList + WindowedList
- `/home/kondraki/personal/worldmonitor/src/config/feeds.ts` -- 1245 lines, source tiers, types, propaganda risk
- `/home/kondraki/personal/worldmonitor/src/components/UnifiedSettings.ts` -- 527 lines, Sources tab UI
- `/home/kondraki/personal/geopol/frontend/src/screens/dashboard-screen.ts` -- current dashboard layout and panel wiring
- `/home/kondraki/personal/geopol/frontend/src/components/Panel.ts` -- 385 lines, base class for all panels
- `/home/kondraki/personal/geopol/frontend/src/components/EventTimelinePanel.ts` -- 314 lines, being deleted
- `/home/kondraki/personal/geopol/frontend/src/components/SourcesPanel.ts` -- 91 lines, being moved to admin
- `/home/kondraki/personal/geopol/frontend/src/admin/admin-layout.ts` -- admin section routing
- `/home/kondraki/personal/geopol/frontend/src/admin/panels/SourceManager.ts` -- 199 lines, existing admin source cards
- `/home/kondraki/personal/geopol/frontend/src/admin/admin-client.ts` -- admin API client
- `/home/kondraki/personal/geopol/src/ingest/feed_config.py` -- 264 lines, hardcoded feeds being moved to DB
- `/home/kondraki/personal/geopol/src/ingest/rss_daemon.py` -- 379 lines, RSS polling daemon
- `/home/kondraki/personal/geopol/src/ingest/article_processor.py` -- 390 lines, ChromaDB indexing
- `/home/kondraki/personal/geopol/src/deduplication.py` -- 333 lines, current GDELT dedup
- `/home/kondraki/personal/geopol/src/knowledge_graph/graph_builder.py` -- graph insertion point for dedup
- `/home/kondraki/personal/geopol/src/scheduler/job_wrappers.py` -- RSS job wrappers
- `/home/kondraki/personal/geopol/src/api/routes/v1/articles.py` -- article search endpoint
- `/home/kondraki/personal/geopol/src/db/models.py` -- SQLAlchemy models (no rss_feeds yet)

### Secondary (MEDIUM confidence)
- YouTube IFrame API documentation (well-known, stable API)
- hls.js library (well-known, but deferred to later)

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- direct codebase reading, all libraries already installed
- Architecture: HIGH -- WM reference implementation fully read, geopol frontend patterns fully understood
- Pitfalls: HIGH -- identified from actual code patterns and established failure modes
- Cross-source dedup: MEDIUM -- algorithm is simple but tuning the coarse event type mapping needs empirical validation

**Research date:** 2026-03-05
**Valid until:** 2026-04-05 (30 days -- stable domain, no external API changes expected)
