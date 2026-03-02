# Phase 14: Backend API Hardening - Context

**Gathered:** 2026-03-02
**Status:** Ready for planning

<domain>
## Phase Boundary

API endpoints transition from mock fixtures to real PostgreSQL-backed data, add a question submission queue with LLM parsing, and expose full-text search. Scope: BAPI-01 (kill fixtures), BAPI-02 (real country risk), BAPI-03 (question submission queue), BAPI-04 (full-text search). No frontend changes — this is pure backend.

</domain>

<decisions>
## Implementation Decisions

### Question Parsing Behavior
- Multi-country support: LLM can map a single question to multiple `country_iso` codes (e.g., "Middle East conflict" → [IL, PS, IR, LB])
- Show parsed form to user for confirmation before queueing — user sees extracted (countries, horizon, category) and confirms or edits
- Horizon is LLM-inferred from natural language ("next month" → 30 days, "by end of year" → remaining days) — no explicit horizon input from user
- When CAMEO category is unclear, default to GENERAL — ensemble falls back to global alpha weight

### Queue Processing Model
- Hybrid processing: immediate async worker for user-submitted questions, daily pipeline for auto-generated questions
- Bounded parallelism: 2-3 concurrent user submissions processing simultaneously
- Retry policy: 3 retries with exponential backoff (e.g., 30s, 2min, 10min) before marking as `failed`
- Queue visibility: private per API key — each user sees only their own submissions and results

### Country Risk Scoring
- Composite index (0-100 scale) combining: forecast count + average probability + Goldstein severity weighting
- Trend determination: 7-day delta — compare current score to score 7 days ago, threshold-based rising/stable/falling classification
- Time decay: exponential decay on prediction contributions — older predictions contribute less to current risk score
- `top_forecast` field: most recently generated forecast for each country

### Fixture Removal Strategy
- Zero predictions for a country: return `{"forecasts": [], "count": 0}` with HTTP 200 — frontend handles empty state display
- Search with no matches: return empty results plus LLM-generated search suggestions (similar queries, related categories)
- Fixture code (`_guess_country_iso`, fixture JSON, fallback paths): keep behind dev env flag (`USE_FIXTURES=1`), production never activates it
- Cold start handled by existing `--seed-countries` mode in `scripts/daily_forecast.py` — no new mechanism needed

### Claude's Discretion
- Exact exponential decay half-life for risk score time weighting
- Composite index formula weights (how to balance count vs probability vs severity)
- Search suggestion implementation (LLM prompt design, caching strategy)
- Full-text search GIN index configuration and query parsing
- Async worker implementation (asyncio task vs separate process vs Celery-lite)

</decisions>

<specifics>
## Specific Ideas

- FRONTEND_REDESIGN.md prescribes `forecast_requests` table schema: id, question, country_iso, horizon_days, category, status, submitted_by, submitted_at, prediction_id FK
- Existing `--seed-countries` CLI mode has a hardcoded SEED_COUNTRIES list of geopolitically significant countries — this serves as the cold-start mechanism
- The Myanmar-under-Syria bleed-through bug is caused by `_guess_country_iso` parsing UUIDs incorrectly — fixture removal eliminates this entirely

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 14-backend-api-hardening*
*Context gathered: 2026-03-02*
