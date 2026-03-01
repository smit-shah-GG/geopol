# Phase 10: Ingest & Forecast Pipeline - Context

**Gathered:** 2026-03-01
**Status:** Ready for planning

<domain>
## Phase Boundary

Continuous GDELT micro-batch ingestion (15-min), RSS feed ingestion for RAG enrichment, daily automated forecast generation with outcome tracking, real API endpoints replacing Phase 9 mock fixtures, and API hardening (Redis caching, input sanitization, rate limiting). 14 requirements: INGEST-01..06, AUTO-01..05, API-04..06.

</domain>

<decisions>
## Implementation Decisions

### Ingest daemon architecture
- Two separate OS processes: GDELT poller and RSS daemon as independent systemd units
- Fault isolation: one crashing/restarting does not affect the other
- Each process gets its own systemd unit file, independent restart logic

### Failure recovery & backoff
- Conservative exponential backoff on GDELT feed failure: 1min -> 2min -> 4min -> max 30min
- On SIGTERM (systemd stop): abort immediately, mark current run as 'interrupted' in ingest_runs table
- On startup: configurable backfill from last successful run (default: backfill enabled, --no-backfill flag for clean start)
- Interrupted runs reconciled automatically via backfill on next startup

### Daily forecast question generation
- LLM-generated: feed top-N recent high-significance events (Goldstein scale, actor importance) to Gemini, which formulates yes/no geopolitical forecast questions
- Event-driven volume: more questions on major-event days, fewer on quiet days (still needs a ceiling to prevent budget blowout)
- Medium time horizon: 14-30 day forecast windows — enough for events to unfold, short enough to accumulate calibration data for Phase 13

### Budget exhaustion handling
- When daily Gemini budget is exhausted mid-pipeline: persist unprocessed questions to a queue
- Next day's run prioritizes queued carryover questions before generating fresh ones
- No forecast is lost, only delayed

### Mock-to-real API transition
- Parallel endpoints: mock fixtures stay at /api/v1/fixtures/*, real forecasts at /api/v1/forecasts/*
- Frontend decides which to hit — zero ambiguity about data source
- Phase 9 mock fixture routes preserved indefinitely for development/testing

### Redis caching
- TTL-based invalidation only — no write-through coupling between pipeline and cache
- 10-minute LRU (in-memory, 100 entries), 1-hour Redis TTL for summaries, 6-hour for full forecasts
- Daily forecast pipeline does not explicitly invalidate cache; natural TTL expiry handles freshness

### Input sanitization (API-05)
- Moderate defense depth: blocklist of known injection patterns + structural validation (question must look like a geopolitical forecast question) + optional LLM pre-check for adversarial intent
- Length cap on question input (~500 chars)
- No system internals leaked in error responses

### Rate limiting (API-06)
- Per-API-key rate limits using Redis counters
- Each key gets N requests/day (configurable per key)

### RSS-to-RAG enrichment
- Semantic chunking: split articles on natural paragraph/section boundaries, not fixed token windows
- Tiered feed polling: top-50 major outlets (Reuters, AP, BBC, Al Jazeera, etc.) at 15-min frequency; remaining ~248 regional/niche sources at hourly or daily
- Separate ChromaDB collections: one for RSS article chunks, existing one for GDELT event descriptions; merged at query time with configurable source weighting
- 90-day rolling window: articles older than 90 days pruned from ChromaDB; balances historical depth for slow-moving situations with manageable index size

### Claude's Discretion
- Exact ceiling for event-driven daily forecast volume
- trafilatura configuration for article text extraction
- ChromaDB embedding model choice for article chunks
- Specific top-50 feed list curation from WM's 298 domains
- systemd unit file details (restart policies, resource limits)
- Redis connection pooling and error handling specifics

</decisions>

<specifics>
## Specific Ideas

- Backfill + abort-on-SIGTERM form a self-healing loop: daemon stops fast, restarts with gap recovery
- Question queue table implies a new PostgreSQL table (pending_questions) not in original schema — needed for budget-exhaustion carryover
- Parallel fixture/real endpoints mean frontend can be developed against fixtures even after real data flows — useful for Phase 12 development

</specifics>

<deferred>
## Deferred Ideas

- ACLED data integration — explicitly deferred until Phase 10 proves micro-batch ingest architecture and Phase 11 validates TKG replacement (ADV-05 in backlog)
- ICEWS data expansion — same deferral condition (ADV-08)

</deferred>

---

*Phase: 10-ingest-forecast-pipeline*
*Context gathered: 2026-03-01*
