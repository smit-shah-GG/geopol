---
phase: 17-live-data-feeds-country-depth
plan: 03
subsystem: ui
tags: [typescript, forecast-client, circuit-breaker, diff-dom, events, advisories, sources, country-brief]

# Dependency graph
requires:
  - phase: 17-01
    provides: EventDTO, ArticleDTO, AdvisoryDTO, SourceStatusDTO Pydantic schemas
  - phase: 17-02
    provides: GET /events, /articles, /sources, /advisories API routes
provides:
  - TypeScript interfaces for all new DTOs (EventDTO, ArticleDTO, AdvisoryDTO, SourceStatusDTO)
  - forecast-client methods (getEvents, getArticles, getSources, getAdvisories) with circuit breakers
  - Live EventTimelinePanel with diff-based DOM updates
  - Auto-discovered SourcesPanel from /sources endpoint
  - CountryBriefPage events, risk-signals, entities tabs populated with real data
  - CountryBriefPage overview tab unconditional advisory summary
affects: [phase-18]

# Tech tracking
tech-stack:
  added: []
  patterns: [lazy-tab-loading-with-null-sentinel, actor-aggregation-client-side, advisory-level-badges]

key-files:
  created: []
  modified:
    - frontend/src/types/api.ts
    - frontend/src/services/forecast-client.ts
    - frontend/src/components/EventTimelinePanel.ts
    - frontend/src/components/SourcesPanel.ts
    - frontend/src/components/CountryBriefPage.ts
    - frontend/src/screens/dashboard-screen.ts

key-decisions:
  - "Events breaker (maxFailures=2, cooldown=15s, cacheTtl=30s) independent from forecast breaker for failure isolation"
  - "SourcesPanel self-refreshes via /sources endpoint at 60s interval -- no longer push-fed from health response"
  - "EventTimelinePanel renamed from GDELT EVENTS to EVENT FEED -- now includes ACLED data"
  - "CountryBriefPage tabs lazy-load data on activation with null sentinel (null=not loaded, []=loaded empty)"
  - "Actor aggregation done client-side from 200-event limit -- avoids dedicated backend endpoint"
  - "Overview tab advisory summary fires background load if advisories not yet cached, shares data with risk-signals tab"

patterns-established:
  - "Lazy tab loading: null sentinel pattern for per-tab cached API data in modal components"
  - "Advisory level badges: 1=green, 2=yellow, 3=orange, 4=red CSS classes"

# Metrics
duration: 7min
completed: 2026-03-04
---

# Phase 17 Plan 03: Frontend Wiring Summary

**TypeScript types + forecast-client methods for 4 new endpoints, EventTimelinePanel rewritten with diff-based DOM updates, SourcesPanel rewired to /sources auto-discovery, CountryBriefPage events/risk-signals/entities tabs populated with live API data, advisory summary unconditionally wired to overview tab.**

## Performance

- **Duration:** 7 min
- **Started:** 2026-03-03T19:52:33Z
- **Completed:** 2026-03-03T19:59:33Z
- **Tasks:** 3
- **Files modified:** 6

## Accomplishments

- All frontend panels now display real live data -- no mock data remains
- EventTimelinePanel preserves expanded card state across 30s refresh via diff-based DOM updates
- CountryBriefPage tabs lazy-load data on activation with per-session caching
- Advisory summary lines render unconditionally on overview tab (data or "no advisories" info)
- Four new forecast-client methods with circuit breakers and deduplication

## Task Commits

Each task was committed atomically:

1. **Task 1: TypeScript types + forecast-client API methods** - `e545cd5` (feat)
2. **Task 2: EventTimelinePanel + SourcesPanel rewiring** - `11c789d` (feat)
3. **Task 3: CountryBriefPage tab population with real data** - `7a3cfc2` (feat)

## Files Created/Modified

- `frontend/src/types/api.ts` - Added EventDTO, ArticleDTO, AdvisoryDTO, SourceStatusDTO interfaces
- `frontend/src/services/forecast-client.ts` - Added getEvents(), getArticles(), getSources(), getAdvisories() with eventsBreaker
- `frontend/src/components/EventTimelinePanel.ts` - Complete rewrite: live events, diff-based DOM, expandable rows, country flags
- `frontend/src/components/SourcesPanel.ts` - Rewrite: self-refreshing via /sources endpoint, auto-discovered health
- `frontend/src/components/CountryBriefPage.ts` - Events/risk-signals/entities tabs with lazy loading, overview advisory summary
- `frontend/src/screens/dashboard-screen.ts` - Removed pushSources, added events+sources to RefreshScheduler

## Decisions Made

- Events breaker independent from forecast breaker (different failure modes -- events are high-frequency, forecasts are low-frequency)
- SourcesPanel self-refreshes at 60s via scheduler (decoupled from health push which was 30s and bundled all subsystems)
- Tab label changed from "GDELT Events" to "Events" (now includes ACLED)
- Panel title changed from "GDELT EVENTS" to "EVENT FEED" (multi-source)
- Client-side actor aggregation from 200-event window avoids need for dedicated backend endpoint
- Lazy tab loading with null sentinel avoids fetching all tab data on modal open

## Deviations from Plan

None -- plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None -- no external service configuration required.

## Next Phase Readiness

Phase 17 is now complete (all 3 plans delivered):
- Plan 01: Data layer (schema, queries, DTOs, settings)
- Plan 02: Backend API routes + ingestion daemons
- Plan 03: Frontend wiring (this plan)

The entire pipeline from database through API to frontend is wired end-to-end. All panels display real data. Phase 18 (Polymarket-Driven Forecasting) can proceed when ready.

---
*Phase: 17-live-data-feeds-country-depth*
*Completed: 2026-03-04*
