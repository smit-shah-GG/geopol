# Phase 26: Operational Fixes & UX Polish - Context

**Gathered:** 2026-03-08
**Status:** Ready for planning

<domain>
## Phase Boundary

Fix operational bugs (Polymarket binary-only filtering, poller enablement), fill the scenario tree root node with news + narrative, overhaul scenario tree text rendering, make all forecast card types clickable with progressive disclosure, and force-refresh data on route navigation. No new features — polish and fix what exists.

</domain>

<decisions>
## Implementation Decisions

### Scenario tree root node content
- Root node displays news articles + LLM-generated narrative summary
- News sourced via ChromaDB semantic search using the forecast question text
- Narrative generated at forecast time, stored with the Prediction row (zero latency on open)
- 2-3 articles shown by default, expandable "Show more" link reveals additional matches
- Hover tooltip remains for any truncated article text

### Scenario tree text rendering overhaul
- Replace current ~40-char truncated labels with 2-3 line paragraphs (~80-120 chars) per node
- Alternating sides layout: left subtree text extends left, right subtree text extends right (classic dendrogram)
- Dynamic node separation: d3-tree spacing calculated from text block height to prevent overlap
- Probability and confidence stay as visual indicators on the node circle (not inline with text)
- Pan + zoom for dense trees (5+ scenarios): fixed scale, user can pan/zoom the SVG canvas
- Curved bezier link lines replacing current straight connectors
- Hover tooltip remains for full description when text exceeds 2-3 lines

### Clickable forecast entries
- My Forecasts entries use identical expandable card layout as Active Forecasts (ensemble weights, evidence summaries, calibration metadata, "View Full Analysis" button)
- Polymarket comparison entries use same expandable card plus inline market price, divergence (pp), and dual-line sparkline
- Extract shared ForecastExpandableCard component used by all three panels (Active Forecasts, My Forecasts, Polymarket Comparisons)
- Full badge parity: My Forecasts entries show "P" badge when linked to Polymarket market, identical to Active Forecasts

### Route refresh behavior
- All three screens (/dashboard, /globe, /forecasts) refresh on every navigation — even clicking the same nav link
- Full cache bust: circuit breaker stale-data cache invalidated on navigate, forces fresh API calls
- Skeleton loading states shown during refresh (no stale-then-swap)
- Globe resets to default position/zoom on /globe navigation (true ctrl+r equivalent)
- Fetch triggers immediately on route change, not after view transition
- Exception: /forecasts submission form input text preserved across same-route refresh; queue and completed forecasts still refresh

### Polymarket binary filter
- Strict `outcomes=['Yes','No']` check — only markets with exactly these two outcome labels are eligible
- Filter applied at forecast trigger stage: non-binary markets are matched and logged, but not forecasted
- Existing non-binary comparisons in DB marked as excluded (status flag) — excluded from accuracy metrics but data retained
- Non-binary matches logged for visibility (admin can see what was skipped)

### Poller enablement
- Enable Polymarket poller (already registered as APScheduler job, just needs activation)
- Baseline risk poller fires hourly (heavy_baseline_risk already wired as APScheduler heavy job)
- No gray areas — mechanical enablement

### Claude's Discretion
- Exact skeleton shimmer timing during route refresh
- ForecastExpandableCard internal component structure and prop interface
- Bezier curve tension values for tree links
- Pan/zoom library choice (d3-zoom vs custom SVG transforms)

</decisions>

<specifics>
## Specific Ideas

- "Just simulate someone hitting ctrl+r whenever the user navigates to dashboard or globe" — full page-reload equivalent, not incremental
- "The root node in the full forecast view is too empty, we'll fill it in with relevant news and a rundown of the scenario" (from March 3 conversation)
- "My forecasts panel's entries need to be clickable in the same way as the active forecasts are" (from March 3 conversation)
- "Polymarket prediction needs to ONLY be for questions with yes or no options, not who's the next supreme leader of Iran"
- Tree text: "lines of text coinciding and running out way too early" — current 40-char truncation is counterproductive
- Tree text: "multiline small paragraph alongside the node, in a direction pointed away from the tree or maybe from its centroid" — led to alternating sides decision

</specifics>

<deferred>
## Deferred Ideas

- NewsFeedPanel rendering bug ("failed to refresh" toast persists after successful fetch) — documented in memory, not scoped into Phase 26
- On-the-fly narrative regeneration when user reopens stale forecasts — potential future enhancement

</deferred>

---

*Phase: 26-operational-fixes-ux-polish*
*Context gathered: 2026-03-08*
