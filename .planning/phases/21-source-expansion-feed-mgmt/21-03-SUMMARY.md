---
phase: 21
plan: 03
subsystem: frontend
tags: [news-feed, virtual-scrolling, clustering, jaccard, source-tiers, propaganda-risk, breaking-news, dashboard-layout]
depends_on:
  requires:
    - "21-02 (articles ?sort=recent endpoint)"
  provides:
    - "NewsFeedPanel with article clustering, source intelligence, category filtering"
    - "WindowedList<T> generic virtual scrolling utility"
    - "BreakingNewsBanner with auto-dismiss, visibility-aware timers"
    - "Dashboard 25/30/30/15 layout with RiskIndex in Col 4"
    - "Probability spike detection for breaking news dispatch"
  affects:
    - "21-04+ (further feed management panels may use WindowedList)"
    - "Future source management UI (source tiers/propaganda maps available)"
tech-stack:
  added: []
  patterns:
    - "Chunk-based windowed rendering (WindowedList<T>)"
    - "Jaccard similarity clustering on title tokens"
    - "Union-find for cluster grouping"
    - "CustomEvent-based alert dispatch (geopol:breaking-news)"
    - "Visibility-aware timer pause/resume"
    - "Frontend source intelligence enrichment (tier + propaganda risk)"
key-files:
  created:
    - frontend/src/components/NewsFeedPanel.ts
    - frontend/src/components/WindowedList.ts
    - frontend/src/components/BreakingNewsBanner.ts
  modified:
    - frontend/src/styles/main.css
    - frontend/src/app/panel-layout.ts
    - frontend/src/screens/dashboard-screen.ts
    - frontend/src/services/forecast-client.ts
    - frontend/src/types/api.ts
decisions:
  - id: "21-03-01"
    decision: "Jaccard threshold 0.5 for article clustering"
    rationale: "Empirically reasonable for news headline similarity; lower threshold over-clusters, higher threshold fragments related coverage"
  - id: "21-03-02"
    decision: "Source tier/propaganda maps are frontend-only static data"
    rationale: "No backend source metadata API exists yet; static maps ported from WM provide immediate value, can be migrated to backend in future phase"
  - id: "21-03-03"
    decision: "Probability spike threshold 0.15 for breaking news alerts"
    rationale: "15% probability change is a significant forecast shift; >30% delta triggers critical severity"
  - id: "21-03-04"
    decision: "Alert sound default: off (opt-in via localStorage)"
    rationale: "Audio autoplay blocked by most browsers; users who want sound must explicitly enable it"
metrics:
  duration: "7 minutes"
  completed: "2026-03-06"
  tasks: "2/2"
  tests_added: 0
  tests_passing: "N/A (frontend build verification only)"
---

# Phase 21 Plan 03: Dashboard NewsFeedPanel & BreakingNewsBanner Summary

**One-liner:** Dashboard reorganized to 25/30/30/15 with Jaccard-clustered NewsFeedPanel (source tiers, propaganda risk, category pills, WindowedList virtual scrolling) in Col 1 and BreakingNewsBanner with visibility-aware auto-dismiss for probability spikes.

## What Was Done

### Task 1: Dashboard layout + WindowedList port + NewsFeedPanel scaffold

1. **CSS column widths**: Changed from `15/35/30/20` to `25/30/30/15` with proportional min-widths. Added `--breaking-alert-offset` CSS custom property and `has-breaking-alert` body class rule.

2. **panel-layout.ts**: Updated column assignment comments to reflect new Phase 21 layout.

3. **dashboard-screen.ts panel rewiring**:
   - Removed `EventTimelinePanel` and `SourcesPanel` imports and instantiation
   - Moved `RiskIndexPanel` to Col 4 (before SystemHealth, Polymarket)
   - Removed `events` and `sources` entries from `scheduler.registerAll()`
   - Removed `event-timeline` and `sources` from `ctx.panels` registration
   - Files EventTimelinePanel.ts and SourcesPanel.ts preserved (may be referenced elsewhere)

4. **WindowedList<T>**: Ported from WM's `WindowedList` class (179 lines). Chunk-based rendering with `requestAnimationFrame`-throttled scroll handling. CSS containment on chunks. `setItems()`, `refresh()`, `destroy()` API.

5. **ArticleDTO enrichment**: Added `source_tier` and `propaganda_risk` optional fields.

6. **forecastClient.getRecentArticles()**: New convenience method calling `GET /articles?sort=recent&limit=N` via the existing `forecastBreaker`.

7. **NewsFeedPanel (474 lines)**:
   - Extends `Panel` base class with `id: 'news-feed'`
   - Category pills: All, Wire, News, Defense, Think Tank, Regional, Crisis, Finance
   - `refresh()` calls `forecastClient.getRecentArticles(100)`, enriches with source intelligence
   - `clusterArticles()` uses Jaccard similarity (threshold 0.5) on tokenized titles with union-find grouping
   - Source tier badges (T1=gold/WIRE, T2=blue/NEWS, T3=green/SPEC, T4=gray)
   - Propaganda risk indicators (high=STATE MEDIA, medium=CAUTION)
   - Source count badges with tooltip listing all sources
   - Renders through WindowedList with 8 items per chunk
   - Registered in scheduler with 60s refresh interval

### Task 2: BreakingNewsBanner + probability spike detection

1. **BreakingNewsBanner (340 lines)**:
   - Standalone class, not a Panel subclass -- appended to `document.body`
   - Listens for `geopol:breaking-news` CustomEvent on document
   - Max 3 concurrent alerts; oldest evicted when 4th arrives
   - Critical alerts evict existing high-priority alerts
   - Auto-dismiss: critical=60s, high=30s
   - Visibility-aware: pauses timers on `document.visibilitychange`, resumes on visible
   - Severity colors: critical=#dc2626 (red), high=#d97706 (amber), white text
   - Fixed position below nav bar, z-index above modal layer
   - Slide-down animation on appear
   - Dismiss button (x) per alert, 30min cooldown before same alert ID can reappear
   - Optional alert sound (base64 WAV, 5min cooldown, off by default, `geopol-alert-sound` localStorage key)
   - Click headline dispatches `geopol:navigate-alert` for future wiring
   - `--breaking-alert-offset` CSS variable updated on alert add/remove
   - `has-breaking-alert` body class toggles dashboard padding-top transition

2. **Probability spike detection**:
   - Tracks previous forecast probabilities across refresh cycles
   - Fires `geopol:breaking-news` event when any forecast probability delta > 0.15
   - Delta > 0.30 triggers `critical` severity, otherwise `high`
   - Wired into both initial data load and 60s forecast refresh scheduler

3. **CSS**: Full banner styling (fixed positioning, severity colors, animation), cluster card styles (borders, hover, source badges, propaganda indicators, windowed list chunks).

## Deviations from Plan

None -- plan executed exactly as written.

## Decisions Made

| # | Decision | Rationale |
|---|----------|-----------|
| 21-03-01 | Jaccard threshold 0.5 for clustering | Balanced: lower over-clusters, higher fragments related coverage |
| 21-03-02 | Source tier/propaganda maps are frontend-only | No backend source metadata API; static maps from WM provide immediate value |
| 21-03-03 | Spike threshold 0.15 for alerts | 15% probability change is significant; >30% is critical |
| 21-03-04 | Alert sound default: off | Browser autoplay policies block; explicit opt-in via localStorage |

## Verification

1. `cd frontend && npm run build` -- zero errors, built in 3.97s
2. Dashboard columns: 25%/30%/30%/15% (grep confirmed)
3. NewsFeedPanel mounted in Col 1 (grep confirmed)
4. BreakingNewsBanner container attached to body (constructor verified)
5. EventTimelinePanel and SourcesPanel: 0 references in dashboard-screen.ts
6. RiskIndexPanel in Col 4 (grep confirmed)
7. WindowedList imported and used by NewsFeedPanel (grep confirmed)
8. Source tier map: 37 entries, Reuters/AP/AFP/Bloomberg at tier 1
9. clusterArticles function: Jaccard + union-find clustering

## Next Phase Readiness

Plan 21-03 delivers the frontend news ecosystem. Remaining plans in this phase:
- 21-04: UCDP integration (requires API token)
- 21-05: Admin source management panel
