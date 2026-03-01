---
phase: 10-ingest-forecast-pipeline
plan: 03
subsystem: ingest
tags: [rss, trafilatura, chromadb, feedparser, aiohttp, asyncio, systemd]

# Dependency graph
requires:
  - phase: 09-api-foundation
    provides: IngestRun model with daemon_type='rss', Settings with rss_poll_interval_tier1/tier2
provides:
  - Tiered RSS feed config (101 geopolitical sources from WM's 298-domain list)
  - Article text extraction via trafilatura with paragraph-boundary chunking
  - ArticleIndexer wrapping ChromaDB rss_articles collection (all-mpnet-base-v2)
  - Async RSS polling daemon with tiered scheduling and SIGTERM handling
  - systemd unit file for production deployment
  - 24-test unit test suite
affects: [10-04, 11-tkg-predictor, 13-calibration]

# Tech tracking
tech-stack:
  added: [trafilatura, feedparser, aiohttp]
  patterns: [tiered-polling, paragraph-chunking, chromadb-collection-per-domain, semaphore-bounded-concurrency]

key-files:
  created:
    - src/ingest/feed_config.py
    - src/ingest/article_processor.py
    - src/ingest/rss_daemon.py
    - scripts/rss_daemon.py
    - deploy/systemd/geopol-rss-daemon.service
    - tests/test_rss_daemon.py
  modified: []

key-decisions:
  - "101 feeds selected from WM's 298: excluded startup/VC, lifestyle, podcast, finance-only"
  - "Two-tier system: 31 tier-1 (wire/major/gov, 15-min), 70 tier-2 (think tank/regional/defense, 60-min)"
  - "Separate ChromaDB collection 'rss_articles' with same embedding model (all-mpnet-base-v2) as graph_patterns"
  - "Paragraph-boundary chunking with sentence-level fallback for oversized blocks"
  - "Propaganda risk metadata exposed for downstream weighting (Xinhua, TASS = high)"

patterns-established:
  - "FeedSource frozen dataclass with tier/category/lang for feed definitions"
  - "ArticleIndexer as ChromaDB wrapper with URL-based dedup and retention pruning"
  - "RSSDaemon tick-based loop with 10s responsive shutdown"
  - "Semaphore-bounded concurrency (max 10) for parallel feed/article fetching"

# Metrics
duration: 6min
completed: 2026-03-01
---

# Phase 10 Plan 03: RSS Feed Ingestion Daemon Summary

**101 geopolitical RSS feeds with tiered polling, trafilatura extraction, paragraph chunking, and ChromaDB indexing via all-mpnet-base-v2**

## Performance

- **Duration:** 6 min
- **Started:** 2026-03-01T13:57:19Z
- **Completed:** 2026-03-01T14:03:32Z
- **Tasks:** 3
- **Files created:** 6

## Accomplishments
- Extracted 101 geopolitical feeds from WM's 298-domain list, organized into 31 tier-1 (15-min) and 70 tier-2 (60-min) sources
- Built article extraction + chunking pipeline with trafilatura and paragraph-boundary splitting
- Implemented async RSS daemon with tiered scheduling, bounded concurrency, URL dedup, and 90-day pruning
- 24 tests covering all pipeline components, all passing

## Task Commits

Each task was committed atomically:

1. **Task 1: Feed configuration and article processor** - `992d1f2` (feat)
2. **Task 2: RSS polling daemon with tiered scheduling and systemd unit** - `3251c9b` (feat)
3. **Task 3: Unit tests for RSS processing pipeline** - `1a5de96` (test)

## Files Created/Modified
- `src/ingest/feed_config.py` - 101 tiered geopolitical RSS feeds with FeedSource dataclass, validation, propaganda risk metadata
- `src/ingest/article_processor.py` - trafilatura extraction, paragraph-boundary chunking, ArticleIndexer wrapping ChromaDB
- `src/ingest/rss_daemon.py` - Async RSS daemon with tiered scheduling, SIGTERM handling, IngestRun recording
- `scripts/rss_daemon.py` - CLI entry point with configurable intervals
- `deploy/systemd/geopol-rss-daemon.service` - systemd unit with MemoryMax=1G, security hardening
- `tests/test_rss_daemon.py` - 24 unit tests (chunking, extraction, dedup, indexing, config, lifecycle, concurrency, fetching, pruning, metrics)

## Decisions Made
- 101 feeds from WM's 298: filtered to geopolitically relevant only (wire, government, defense, think tanks, crisis, regional, energy, finance)
- Two-tier polling: tier-1 = wire services + major outlets + government (15 min), tier-2 = think tanks + regional + defense (60 min)
- Separate ChromaDB collection "rss_articles" with same all-mpnet-base-v2 embedding as existing graph_patterns
- Paragraph-boundary chunking with 800-char target, sentence-level fallback for oversized paragraphs, 80-char minimum
- Propaganda risk metadata from WM's SOURCE_PROPAGANDA_RISK for downstream weighting

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- RSS daemon ready for production deployment via systemd
- ArticleIndexer ready for RAG integration (Phase 10-04 daily pipeline can query rss_articles collection)
- Feed list can be extended by adding FeedSource entries to TIER_1_FEEDS or TIER_2_FEEDS

---
*Phase: 10-ingest-forecast-pipeline*
*Completed: 2026-03-01*
