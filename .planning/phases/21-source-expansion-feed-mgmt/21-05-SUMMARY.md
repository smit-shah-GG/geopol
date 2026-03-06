# Plan 21-05 Summary: Admin SourceManager + NewsFeedPanel Source Filtering

## Status: Complete

## What was built

1. **AdminClient feed CRUD methods** (`admin-client.ts`): `getFeeds()`, `addFeed()`, `updateFeed()`, `deleteFeed()` — all wired to `/api/v1/admin/feeds` with X-Admin-Key header.

2. **Admin types** (`admin-types.ts`): `FeedInfo`, `AddFeedRequest`, `UpdateFeedRequest` interfaces.

3. **Rich SourceManager panel** (`SourceManager.ts`): Complete rewrite from simple card grid to full feed management:
   - Feed card grid with rich metadata: articles_24h, articles_total, avg_articles_per_poll, last_error, error_count
   - Add Feed form (expandable) with URL, name, tier selector
   - Per-feed tier toggle (T1/T2), enable/disable toggle, delete with optional purge
   - Auto-disabled alert banner for feeds with repeated errors
   - Stats footer with totals
   - 30s auto-refresh

4. **NewsFeedPanel source filtering** (`NewsFeedPanel.ts`): Reads `geopol-disabled-sources` from localStorage (set by SettingsModal), filters articles in render pipeline. Listens for `geopol:sources-changed` CustomEvent for live updates.

5. **Source health indicator**: Fetches public `/api/v1/sources`, shows subtle "Some sources temporarily unavailable" label with expandable detail when sources are unhealthy.

## Human verification

Verified by user during checkpoint. Issues found and fixed:
- ChromaDB `_recent_search()` was broken — `$gte` operator fails on string metadata. Fixed with Python-side date parsing via `_parse_date_lenient()` and stale-data fallback (commit `b85080f`).
- LiveStreamsPanel moved from col 1 to col 3 for more space.
- LiveStreamsPanel rewritten from 16-channel grid to single-stream player with channel-name pills and gear icon region filter (commit `a394a71`).
- Submission worker race condition: `flush()` → `commit()` before scheduling to prevent `SKIP LOCKED` from silently skipping uncommitted rows (commit `a394a71`).

## Commits

- `4ad2b78` feat(21-05): NewsFeedPanel source filtering + health indicator
- `16efa9a` feat(21-05): AdminClient feed CRUD + rich SourceManager panel
- `b85080f` fix(21): articles API recent search — proper date parsing + stale fallback
- `a394a71` fix(21): single-stream LiveStreamsPanel + submission worker race fix
