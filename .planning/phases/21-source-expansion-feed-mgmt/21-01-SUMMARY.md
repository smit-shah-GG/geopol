---
phase: 21-source-expansion-feed-mgmt
plan: 01
subsystem: ingest
tags: [rss, admin-api, postgresql, alembic, feed-management]
dependency_graph:
  requires: [phase-20-daemon-consolidation]
  provides: [rss-feed-crud-api, db-backed-rss-polling, per-feed-health-metrics]
  affects: [phase-21-plan-02, phase-21-plan-03, admin-frontend-feeds-panel]
tech_stack:
  added: []
  patterns: [soft-delete, db-fallback, per-feed-health-tracking, auto-disable]
key_files:
  created:
    - alembic/versions/20260305_007_rss_feeds.py
  modified:
    - src/db/models.py
    - src/api/routes/v1/admin.py
    - src/api/schemas/admin.py
    - src/api/services/admin_service.py
    - src/scheduler/job_wrappers.py
    - src/ingest/rss_daemon.py
    - src/ingest/feed_config.py
decisions:
  - Per-feed health callback pattern: RSSDaemon returns PerFeedResult list, callers persist to DB (decoupled from SQLAlchemy)
  - Auto-disable threshold: 5 consecutive failures disables a feed
  - DB fallback: if rss_feeds query fails, job wrappers fall back to feed_config.py constants
  - Soft-delete default: DELETE /feeds/{id} sets deleted_at, ?purge=true for hard delete
metrics:
  duration: 7min
  completed: 2026-03-06
---

# Phase 21 Plan 01: RSS Feed Management Backend Summary

**DB-backed RSS feed registry with admin CRUD, per-feed health metrics, and auto-disable on consecutive failures.**

## What Was Done

### Task 1: RSSFeed SQLAlchemy Model + Alembic Migration
- Added `RSSFeed` model to `src/db/models.py` with 15 columns: id, name, url, tier, category, lang, enabled, last_poll_at, last_error, error_count, articles_24h, articles_total, avg_articles_per_poll, created_at, deleted_at.
- Added `CHECK(tier IN (1, 2))` constraint (`ck_rss_feeds_tier`).
- Created Alembic migration `007` that creates the `rss_feeds` table and seeds 101 rows from `feed_config.py ALL_FEEDS` via `op.bulk_insert()`.

### Task 2: Admin Feed CRUD API Endpoints
- Added Pydantic schemas: `FeedInfo` (14 fields), `AddFeedRequest` (with Literal[1,2] tier validation), `UpdateFeedRequest` (all Optional).
- Added `AdminService` methods: `get_feeds()`, `add_feed()`, `update_feed()`, `delete_feed()`.
- Registered 4 endpoints on the admin router:
  - `GET /api/v1/admin/feeds` -- list all non-deleted feeds with health
  - `POST /api/v1/admin/feeds` -- add feed (201, 409 on duplicate name)
  - `PUT /api/v1/admin/feeds/{feed_id}` -- update feed (404 if missing)
  - `DELETE /api/v1/admin/feeds/{feed_id}?purge=false` -- soft/hard delete (204)

### Task 3: DB-Backed RSS Polling + Health Metric Updates
- Rewrote `rss_poll_tier1()` and `rss_poll_tier2()` in `job_wrappers.py` to query `rss_feeds` table for enabled, non-deleted feeds of the appropriate tier.
- Added `_get_feeds_from_db(tier)` helper that returns `list[FeedSource]` or `None` on DB failure.
- Added `_update_feed_health()` that persists per-feed outcomes to `rss_feeds` after each poll cycle: `last_poll_at`, `error_count`, `last_error`, `articles_24h`, `articles_total`, `avg_articles_per_poll`.
- Feeds auto-disable after 5 consecutive failures (sets `enabled=False`).
- Added `PerFeedResult` dataclass to `rss_daemon.py` for per-feed outcome tracking.
- `CycleMetrics` now includes `per_feed: list[PerFeedResult]` populated during `poll_feeds()`.
- `feed_config.py` docstring updated to note it's now primarily a seed/fallback source.

## Decisions Made

| Decision | Rationale |
|----------|-----------|
| Callback pattern over direct SQLAlchemy coupling | RSSDaemon returns PerFeedResult list; callers (job_wrappers) persist to DB. Keeps daemon testable without DB. |
| 5-failure auto-disable threshold | Prevents wasting poll cycles on permanently broken feeds. Admin can re-enable manually. |
| DB query fallback to feed_config.py | If PostgreSQL is unreachable during polling, fall back to hardcoded constants rather than polling zero feeds. |
| Soft-delete by default | Preserves feed history and allows recovery. Hard delete via `?purge=true`. |
| 101 seed rows from ALL_FEEDS | Migration seeds all existing feeds immediately -- zero manual setup needed. |

## Deviations from Plan

None -- plan executed exactly as written.

## Verification Results

1. `RSSFeed` model imports cleanly with all 15 columns and tier CHECK constraint.
2. Migration file passes AST parse check.
3. Feed CRUD endpoints registered on admin router: `/feeds`, `/feeds/{feed_id}`.
4. Test suite: 284 passed, 10 skipped, 41 failed (all pre-existing CUDA OOM / TKG predictor / stale test issues).
5. `job_wrappers.py` TIER_1_FEEDS/TIER_2_FEEDS imports only in fallback path.

## SRC-06 Satisfaction

Per-source health (daemon-level) is exposed via existing `GET /api/v1/sources`. Per-feed granular health metrics are now exposed via `GET /api/v1/admin/feeds` (admin-only). This satisfies requirement SRC-06 as designed in the phase context.

## Next Phase Readiness

Plan 21-02 (UCDP integration) and Plan 21-03 (admin feed management UI) can proceed. The `rss_feeds` table and CRUD API are the foundation for the admin feeds panel.
