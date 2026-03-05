# Phase 21: Source Expansion & Feed Management - Context

**Gathered:** 2026-03-05
**Status:** Ready for planning

<domain>
## Phase Boundary

Bring live news content to dashboard users with WM-style panels, user source controls, and admin feed management. Replace the information-sparse EventTimelinePanel with a full news ecosystem. Add cross-source dedup at the knowledge graph layer and move source health exclusively to admin.

UCDP poller is **deferred** until API token is procured (email-gated, unpredictable turnaround). All UCDP-specific decisions (conflict type mapping, fatality schema, polling frequency) are deferred with it.

</domain>

<decisions>
## Implementation Decisions

### Dashboard Layout Redesign
- **Column widths change from 15-35-30-20 to 25-30-30-15**
- Col 1 (25%): NewsFeedPanel + LiveStreamsPanel (replaces RiskIndexPanel)
- Col 2 (30%): SearchBar, ForecastPanel, ComparisonPanel (unchanged)
- Col 3 (30%): MyForecastsPanel (SourcesPanel removed from user-facing UI)
- Col 4 (15%): RiskIndexPanel (moved here, compacted), SystemHealthPanel, PolymarketPanel
- BreakingNewsBanner spans full width at top of page, above all columns
- EventTimelinePanel is **deleted** (replaced by NewsFeedPanel)
- SourcesPanel is **removed from dashboard** (moved to admin-only)

### News Feed Panel (Col 1, replaces EventTimeline)
- **Full WM port**: clustering, velocity badges, sentiment, source tiers, AI summaries, propaganda risk
- Article clustering groups same-story articles across sources — shows source count badge, primary headline, "also reported by" list
- Virtual scrolling for performance (WM's WindowedList pattern)
- Category filter pills directly on the panel for quick filtering
- Full source management available in settings modal (category + individual feed toggles)
- Data source: existing ChromaDB articles from RSS ingestion pipeline

### Breaking News Banner (full-width, top of page)
- **Dual trigger**: high-severity GDELT events (Goldstein extremes / high significance) AND forecast probability spikes
- WM-style: auto-dismiss, severity color coding, sound optional
- Max concurrent alerts on screen (WM uses 3)
- Clickable — navigates to relevant forecast or event

### Live Streams Panel (Col 1, below news feed)
- **YouTube iframes + native HLS** (full WM approach — YouTube for most channels, HLS for Sky/Euronews/DW direct feeds)
- **Curated 10-20 channels** — hand-picked geopolitical news (Al Jazeera, BBC, CNN, Sky, France24, DW, etc.)
- Region filter pills (even with small channel count — helps users find relevant streams)
- Idle detection: pause stream after 5 min inactive, resume on interaction
- **Always visible** — not collapsible, monitoring dashboard feel
- Mute button, live indicator, fullscreen button

### User Source Controls
- **Dual interface**: category filter pills on news panel (quick) + full settings modal Sources tab (granular)
- **Both category-level and feed-level control**: toggle categories for quick control, drill into individual feeds for fine-tuning
- Checkbox grid with search, select all/none (WM pattern)
- **localStorage persistence** — no user accounts, browser-local
- Source preferences affect NewsFeedPanel display only (not backend ingestion)

### Cross-Source Dedup
- **Conservative matching**: exact (date, country, event_type) fingerprint — miss some dupes rather than false-merge distinct events
- **ACLED preferred**: when GDELT and ACLED report same event, ACLED version wins (human-coded, higher accuracy)
- **Dedup at graph insertion**: raw events stored in SQLite as-is from both sources; dedup filter runs before knowledge graph triple creation — preserves raw data for audit
- **Full audit logging**: every suppressed duplicate gets a log entry with both event IDs and fingerprint match

### Per-Source Health (Admin Only)
- Health metrics (staleness, error rate, event counts) exposed via API but displayed only in admin dashboard
- Consistent with SourcesPanel moving out of user-facing UI
- Feed auto-disable on N consecutive failures — admin dashboard shows alert
- **Subtle user indicator**: news feed shows small "some sources unavailable" note when feeds are auto-disabled (not silent, not intrusive)

### Admin Feed Management
- Admin dashboard gets a feed management panel: add URL + name + tier, enable/disable, see per-feed rich metadata
- Rich metadata per feed: name, URL, tier, enabled/disabled, last poll time, error count, articles ingested (24h/total), avg articles per poll, last error message, feed language
- Soft delete with optional purge: disabling stops polling but retains ChromaDB articles; separate "delete feed" action offers to purge articles
- Per-feed tier assignment (tier1=15min, tier2=60min)
- Feeds persisted to rss_feeds PostgreSQL table

### Claude's Discretion
- Exact clustering algorithm for news articles (can reference WM's analysisWorker approach)
- Specific channel list for live streams (within the 10-20 curated constraint)
- Breaking news banner auto-dismiss timing and max concurrent count
- AI summary implementation (WM uses LLM for top-8 headline summaries)
- Virtual scrolling chunk size and buffer strategy
- How to compact RiskIndexPanel for the narrower Col 4

</decisions>

<specifics>
## Specific Ideas

- "Wiring sources into the frontend (like WM)" — the original v3.0 vision statement. This phase delivers the user-facing news experience.
- WM's NewsPanel (654 lines), LiveNewsPanel (1200+ lines), BreakingNewsBanner (297 lines), UnifiedSettings Sources tab (527 lines), and feeds.ts (1245 lines) are the reference implementations in `/home/kondraki/personal/worldmonitor/src/`.
- WM source tier system: Tier 1 (wire services: Reuters, AP, AFP, Bloomberg), Tier 2 (major outlets: BBC, Guardian, CNN), Tier 3 (specialty: Foreign Policy, CSIS), Tier 4 (aggregators: Hacker News, The Verge).
- WM's propaganda risk tracking (state-affiliated source detection) should port to geopol — relevant for geopolitical news credibility.

</specifics>

<deferred>
## Deferred Ideas

- **UCDP poller + all UCDP-specific decisions** — deferred until API token procured from mertcan.yilmaz@pcr.uu.se. Includes: conflict type taxonomy (CAMEO mapping vs parallel), fatality estimates (best/high/low), polling frequency. Can be a sub-phase or added to Phase 24.
- **Positive news feed** (WM's PositiveNewsFeedPanel) — not included, geopol is a geopolitical forecasting tool, not a general news monitor.
- **160+ channel list** — start with curated 10-20, expand later if needed.

</deferred>

---

*Phase: 21-source-expansion-feed-mgmt*
*Context gathered: 2026-03-05*
