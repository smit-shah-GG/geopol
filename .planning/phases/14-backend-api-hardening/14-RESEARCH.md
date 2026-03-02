# Phase 14: Backend API Hardening - Research

**Researched:** 2026-03-03
**Domain:** FastAPI backend -- fixture removal, PostgreSQL aggregation, queue processing, full-text search
**Confidence:** HIGH (entirely codebase research, no external dependencies)

## Summary

Phase 14 replaces mock fixture data with real PostgreSQL-backed data across four requirements: kill fixture fallback (BAPI-01), real country risk aggregation (BAPI-02), question submission queue (BAPI-03), and full-text search (BAPI-04).

The codebase is well-structured for these changes. The existing `ForecastService` already queries PostgreSQL correctly -- the problem is exclusively in the route layer (`forecasts.py`) where fixture fallback code catches empty PostgreSQL results and substitutes mock data. The `countries.py` route is 100% hardcoded mock data with no PostgreSQL integration at all. Both must be rewritten. A new `forecast_requests` table, submission endpoint, LLM parsing step, and async worker are needed for BAPI-03. PostgreSQL 16 provides native `tsvector` + GIN index support for BAPI-04.

**Primary recommendation:** Attack in dependency order: BAPI-01 (fixture removal) first since it's pure deletion, then BAPI-04 (search -- needs migration but is self-contained), then BAPI-02 (country risk -- complex SQL), then BAPI-03 (queue -- most new code, depends on understanding the prediction pipeline).

## Existing Codebase Analysis

### Current Route Architecture

**File: `src/api/routes/v1/forecasts.py` (369 lines)**

Four endpoints:
1. `GET /forecasts/top` -- cache -> PostgreSQL -> **fixture fallback** (lines 154-159)
2. `GET /forecasts/country/{iso_code}` -- cache -> PostgreSQL -> **fixture fallback** (lines 213-236, the Myanmar-Syria bug source)
3. `GET /forecasts/{forecast_id}` -- cache -> PostgreSQL -> **fixture fallback** (lines 271-276) -> 404
4. `POST /forecasts` -- live EnsemblePredictor (no fixtures)

The fixture fallback path uses:
- `_fixture_cache` (module-level global dict, lazy-loaded)
- `_get_fixture_cache()` -- loads SY/UA/MM from JSON files + generates IR/TW/SD mocks
- `_guess_country_iso()` -- parses forecast_id convention `fc-{iso}-{hash}` (line 363-368). This is the Myanmar-Syria bleed-through bug: UUID-based IDs from real predictions don't follow this convention, so `parts[1].upper()` returns garbage.
- `load_fixture()`, `load_all_fixtures()`, `create_mock_forecast()` from `src/api/fixtures/factory.py`

**Imports to remove** (lines 33-37):
```python
from src.api.fixtures.factory import (
    create_mock_forecast,
    load_all_fixtures,
    load_fixture,
)
```

**File: `src/api/routes/v1/countries.py` (100 lines)**

Two endpoints, both entirely mock:
1. `GET /countries` -- returns hardcoded `_MOCK_COUNTRIES` list (8 countries with fake risk scores)
2. `GET /countries/{iso_code}` -- looks up from same mock cache

Zero PostgreSQL integration. No `AsyncSession` dependency. No imports from `src.db.models`. This file must be substantially rewritten.

**Import from `src.api.fixtures.factory`:**
```python
from src.api.fixtures.factory import create_mock_country_risk
```

### Current Schema (PostgreSQL)

**`predictions` table** (3 migrations: 001, 002, 003):
```
id              String(36) PK
question        Text NOT NULL
prediction      Text NOT NULL
probability     Float NOT NULL
confidence      Float NOT NULL
horizon_days    Integer NOT NULL (default 30)
category        String(32) NOT NULL, indexed
reasoning_summary Text NOT NULL
evidence_count  Integer (default 0)
scenarios_json  JSON NOT NULL
ensemble_info_json JSON NOT NULL
calibration_json JSON NOT NULL
entities        JSON NOT NULL
country_iso     String(3) nullable, indexed
cameo_root_code String(4) nullable, indexed (added Phase 13)
created_at      DateTime(tz) indexed, default now()
expires_at      DateTime(tz) NOT NULL
```

**Existing indexes:**
- `ix_predictions_category` (category)
- `ix_predictions_country_iso` (country_iso)
- `ix_predictions_created_at` (created_at)
- `ix_predictions_country_created` (country_iso, created_at) -- composite
- `ix_predictions_cameo_root` (cameo_root_code)

**Missing for Phase 14:**
- No `tsvector` column or GIN index on `question` (needed for BAPI-04)
- No `forecast_requests` table (needed for BAPI-03)

**`pending_questions` table** (existing, from migration 002):
```
id              Integer PK autoincrement
question        Text NOT NULL
country_iso     String(3) nullable
horizon_days    Integer NOT NULL (default 21)
category        String(32) NOT NULL
priority        Integer (default 0)
created_at      DateTime(tz) default now()
status          String(20) default 'pending' (pending | processing | completed)
```

This is the pipeline's internal budget-overflow queue. It lacks `submitted_by`, `prediction_id` FK, and the `failed` status. The CONTEXT.md specifies a **new** `forecast_requests` table for user-submitted questions, which is correct -- `pending_questions` serves a different purpose (auto-generated questions from the daily pipeline when Gemini budget is exhausted).

**`api_keys` table:**
```
id              Integer PK
key             String(64) NOT NULL UNIQUE, indexed
client_name     String(100) NOT NULL
created_at      DateTime(tz) default now()
revoked         Boolean default false
```

The `client_name` field is what `verify_api_key()` returns. Queue visibility filtering (`submitted_by`) should use this value.

### Current DTO Schema

**`CountryRiskSummary`** (src/api/schemas/country.py):
```python
iso_code: str           # 2-3 char
risk_score: float       # 0.0-1.0  <-- BAPI-02 says 0-100!
forecast_count: int
top_question: str       # <-- BAPI-02 says "top_forecast" (most recent)
top_probability: float
trend: "rising" | "stable" | "falling"
last_updated: datetime
```

**Breaking changes needed for BAPI-02:**
1. `risk_score` range: currently 0.0-1.0, requirement says 0-100. Either change the DTO range or scale internally. Recommendation: change to 0-100 integer scale (matches requirement spec exactly).
2. `top_question` -> `top_forecast`: requirement says "top_forecast (most recent)" but the current field name is `top_question`. The existing name is arguably better (it's a question text, not a full forecast object). Decision needed: keep `top_question` naming or change to include a mini forecast summary.
3. Frontend TypeScript type `CountryRiskSummary` in `frontend/src/types/api.ts` must be updated to match.

**`ForecastResponse`** (src/api/schemas/forecast.py) -- no changes needed. Already has all fields for search results.

### Forecasting Pipeline Architecture

**EnsemblePredictor.predict()** (synchronous, CPU-bound):
```python
def predict(
    self,
    question: str,
    context: Optional[List[str]] = None,
    entity1: Optional[str] = None,
    relation: Optional[str] = None,
    entity2: Optional[str] = None,
    category: Optional[str] = None,
    alpha_override: Optional[float] = None,
    cameo_root_code: Optional[str] = None,
) -> Tuple[EnsemblePrediction, ForecastOutput]:
```

This is the core prediction call. It:
1. Calls Gemini LLM (2-3 minutes per question due to multiple scenario generation calls)
2. Calls TKG predictor (sub-second)
3. Blends probabilities with alpha weighting
4. Returns (EnsemblePrediction, ForecastOutput) tuple

**ForecastService.persist_forecast()** (async):
Takes `ForecastOutput` + `EnsemblePrediction` and creates a `Prediction` ORM row.

**Current POST /forecasts flow:**
1. Validate + sanitize input
2. Check Gemini budget
3. `asyncio.to_thread(predictor.predict, question=...)` -- runs sync predict in thread pool
4. `service.persist_forecast(...)` -- async persist to PostgreSQL
5. Commit + cache

**For BAPI-03 (queue), the new flow is:**
1. `POST /forecasts/submit` accepts natural language question
2. LLM parses -> structured form (country_iso[], horizon_days, category)
3. Return request_id + parsed form to user (for confirmation)
4. Background worker picks up confirmed request
5. Runs `EnsemblePredictor.predict()` per country
6. Persists via `ForecastService.persist_forecast()`
7. Updates `forecast_requests.status` to `complete` and links `prediction_id`

### Goldstein Scale Data

Available in the codebase:
- `GoldsteinScale`: Float from -10 (conflict: military attack) to +10 (cooperation: formal agreement)
- Stored in GDELT events (SQLite `events.db`)
- Referenced via `cameo_root_code` on `predictions` table
- CAMEO quadrant mapping in `src/calibration/priors.py`:
  - Quadrant 1 (verbal cooperation, codes 01-05): Goldstein +1 to +3.4
  - Quadrant 2 (material cooperation, codes 06-09): Goldstein +3.5 to +8.3
  - Quadrant 3 (verbal conflict, codes 10-14): Goldstein -3.4 to -0.1
  - Quadrant 4 (material conflict, codes 15-20): Goldstein -10 to -3.5

**For BAPI-02 risk scoring:** The `cameo_root_code` on predictions maps to Goldstein severity via the CAMEO quadrant taxonomy. However, `cameo_root_code` is nullable (added in Phase 13) and may be NULL for older predictions. The risk score formula needs a fallback for predictions without CAMEO codes.

**Goldstein severity weighting for risk score:**
Since we need a "severity" component (how bad the events are), we can map CAMEO root codes to their typical Goldstein scale ranges. Higher absolute values = more severe. Material conflict (codes 15-20) should weight highest in a risk score.

### Fixture Code Inventory

**Files that contain fixture/mock code:**

1. **`src/api/routes/v1/forecasts.py`** -- fixture fallback logic (lines 64-91, 154-159, 213-236, 271-276, 363-368)
2. **`src/api/routes/v1/countries.py`** -- entirely mock (lines 27-58, entire file essentially)
3. **`src/api/fixtures/factory.py`** -- fixture factory (376 lines)
4. **`src/api/fixtures/__init__.py`** -- exports
5. **`src/api/fixtures/scenarios/syria.json`** -- hand-crafted fixture
6. **`src/api/fixtures/scenarios/ukraine.json`** -- hand-crafted fixture
7. **`src/api/fixtures/scenarios/myanmar.json`** -- hand-crafted fixture

**CONTEXT decision:** Keep fixture code behind `USE_FIXTURES=1` dev env flag, production never activates it. So we DON'T delete the fixture files -- we gate their usage.

### Settings Configuration

`src/settings.py` -- Pydantic BaseSettings with `.env` loading. No `USE_FIXTURES` flag exists yet. Needs addition:
```python
use_fixtures: bool = False  # Enable fixture fallback (dev only)
```

Or simpler: read `os.environ.get("USE_FIXTURES") == "1"` inline.

### Frontend Impact Analysis

**`frontend/src/types/api.ts`** -- TypeScript interfaces that mirror Pydantic DTOs:
- `CountryRiskSummary` -- needs `risk_score` range update if changed
- `ForecastResponse` -- no changes needed
- New interfaces needed: `ForecastRequest`, `SubmitQuestionRequest`, `ParsedQuestionResponse`

**`frontend/src/services/forecast-client.ts`** -- API client:
- `getCountries()` -- no endpoint change needed, response shape changes
- New methods needed: `submitQuestion()`, `getMyRequests()`, `searchForecasts()`

**Phase 14 is backend-only per CONTEXT, but DTO changes will affect Phase 15/16 frontend work.**

## Architecture Patterns

### BAPI-01: Fixture Removal Pattern

```python
# BEFORE (forecasts.py:get_forecasts_by_country)
try:
    service = ForecastService(db)
    result = await service.get_forecasts_by_country(...)
    if result.items:
        return result
except Exception as exc:
    logger.warning("PostgreSQL country query failed...")

# Fall back to fixtures  <-- THIS BLOCK DELETED
fixture_cache = _get_fixture_cache()
...

# AFTER
service = ForecastService(db)
result = await service.get_forecasts_by_country(...)
return result  # Empty items = empty items. No fallback.
```

Key: The `ForecastService.get_forecasts_by_country()` already returns `PaginatedResponse(items=[], ...)` when no rows match. The fixture fallback was compensating for a non-problem.

For the dev flag:
```python
if settings.use_fixtures and not result.items:
    # Dev-only fixture fallback
    ...
```

### BAPI-02: SQL Aggregation Pattern

The country risk endpoint needs a single SQL query that:
1. Groups predictions by `country_iso`
2. Counts active (non-expired) predictions per country
3. Computes weighted average probability with time decay
4. Maps `cameo_root_code` to severity weight
5. Computes composite risk score (0-100)
6. Computes trend via 7-day delta

**Recommended: single CTE-based query.**

```sql
WITH active_predictions AS (
    SELECT
        country_iso,
        probability,
        cameo_root_code,
        created_at,
        question,
        -- Exponential decay: half-life of 7 days
        EXP(-0.693 * EXTRACT(EPOCH FROM (NOW() - created_at)) / (7 * 86400))
            AS decay_weight
    FROM predictions
    WHERE country_iso IS NOT NULL
      AND expires_at > NOW()
),
current_scores AS (
    SELECT
        country_iso,
        COUNT(*) AS forecast_count,
        -- Weighted probability (decay-adjusted)
        SUM(probability * decay_weight) / NULLIF(SUM(decay_weight), 0)
            AS avg_probability,
        -- Severity from CAMEO (material conflict = high severity)
        SUM(
            CASE
                WHEN cameo_root_code IN ('15','16','17','18','19','20') THEN 1.0
                WHEN cameo_root_code IN ('10','11','12','13','14') THEN 0.6
                WHEN cameo_root_code IN ('06','07','08','09') THEN 0.3
                WHEN cameo_root_code IN ('01','02','03','04','05') THEN 0.1
                ELSE 0.5  -- unknown CAMEO
            END * decay_weight
        ) / NULLIF(SUM(decay_weight), 0) AS avg_severity,
        -- Most recent forecast
        (SELECT question FROM active_predictions ap2
         WHERE ap2.country_iso = active_predictions.country_iso
         ORDER BY created_at DESC LIMIT 1) AS top_question,
        (SELECT probability FROM active_predictions ap2
         WHERE ap2.country_iso = active_predictions.country_iso
         ORDER BY created_at DESC LIMIT 1) AS top_probability
    FROM active_predictions
    GROUP BY country_iso
),
-- 7-day-ago scores for trend
past_scores AS (
    SELECT
        country_iso,
        SUM(probability * EXP(-0.693 * EXTRACT(EPOCH FROM
            (NOW() - INTERVAL '7 days' - created_at)) / (7 * 86400)))
            / NULLIF(SUM(EXP(-0.693 * EXTRACT(EPOCH FROM
            (NOW() - INTERVAL '7 days' - created_at)) / (7 * 86400))), 0)
            AS past_avg_probability
    FROM predictions
    WHERE country_iso IS NOT NULL
      AND expires_at > NOW() - INTERVAL '7 days'
      AND created_at <= NOW() - INTERVAL '7 days'
    GROUP BY country_iso
)
SELECT
    cs.*,
    ps.past_avg_probability,
    -- Composite risk: count_component + probability_component + severity_component
    LEAST(100, (
        LEAST(20, cs.forecast_count * 4) +  -- count: 0-20 points (caps at 5 forecasts)
        cs.avg_probability * 50 +             -- probability: 0-50 points
        cs.avg_severity * 30                  -- severity: 0-30 points
    )) AS risk_score
FROM current_scores cs
LEFT JOIN past_scores ps ON cs.country_iso = ps.country_iso
ORDER BY risk_score DESC;
```

This is illustrative. The exact formula weights are Claude's discretion per CONTEXT.md.

### BAPI-03: Queue Processing Pattern

**New table: `forecast_requests`**
```python
class ForecastRequest(Base):
    __tablename__ = "forecast_requests"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_uuid)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    country_iso_list: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    horizon_days: Mapped[int] = mapped_column(Integer, nullable=False, default=30)
    category: Mapped[str] = mapped_column(String(32), nullable=False, default="GENERAL")
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="pending")
        # pending | confirmed | processing | complete | failed
    submitted_by: Mapped[str] = mapped_column(String(100), nullable=False)  # client_name from api_key
    submitted_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    prediction_ids: Mapped[list] = mapped_column(JSON, nullable=False, default=list)  # FK list
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    retry_count: Mapped[int] = mapped_column(Integer, default=0)
    parsed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
```

Note: `country_iso_list` is a JSON array because CONTEXT.md specifies multi-country support (e.g., "Middle East conflict" -> [IL, PS, IR, LB]). Each country gets its own prediction, all linked back to one request.

**Two-phase submission flow:**
1. `POST /forecasts/submit` -- LLM parses question, returns parsed form + request_id
2. `POST /forecasts/submit/{request_id}/confirm` -- user confirms parsed form, status -> `confirmed`, worker picks up
   (Or: single endpoint if we skip confirmation for MVP. CONTEXT says "show parsed form to user for confirmation" which implies the frontend does the confirm step.)

**Async worker architecture (Claude's discretion):**

Recommendation: **asyncio.create_task with bounded semaphore**. Rationale:
- No external dependency (no Celery, no Redis-based queue)
- FastAPI already runs an asyncio event loop
- Bounded parallelism via `asyncio.Semaphore(3)` matches the 2-3 concurrent requirement
- Retry via simple `asyncio.sleep()` with exponential backoff
- Process lifetime matches API server lifetime (no orphan workers)

```python
_worker_semaphore = asyncio.Semaphore(3)
_active_tasks: set[asyncio.Task] = set()

async def process_forecast_request(request_id: str):
    async with _worker_semaphore:
        # ... run prediction pipeline ...
```

### BAPI-04: Full-Text Search Pattern

**PostgreSQL 16 tsvector approach:**

New migration adds:
1. Generated tsvector column on `predictions`
2. GIN index on the tsvector column

```sql
-- Migration
ALTER TABLE predictions
    ADD COLUMN question_tsv tsvector
    GENERATED ALWAYS AS (to_tsvector('english', question)) STORED;

CREATE INDEX ix_predictions_question_tsv ON predictions USING GIN (question_tsv);
```

**Query pattern:**
```sql
SELECT * FROM predictions
WHERE question_tsv @@ plainto_tsquery('english', :query)
  AND (:country IS NULL OR country_iso = :country)
  AND (:category IS NULL OR category = :category)
ORDER BY ts_rank(question_tsv, plainto_tsquery('english', :query)) DESC
LIMIT :limit;
```

**SQLAlchemy 2.0 approach:**
```python
from sqlalchemy import func
from sqlalchemy.dialects.postgresql import TSVECTOR

# In model:
question_tsv: Mapped[Any] = mapped_column(
    TSVECTOR,
    Computed("to_tsvector('english', question)", persisted=True),
)

# In query:
query_ts = func.plainto_tsquery('english', search_query)
stmt = (
    select(Prediction)
    .where(Prediction.question_tsv.op('@@')(query_ts))
    .order_by(func.ts_rank(Prediction.question_tsv, query_ts).desc())
    .limit(limit)
)
```

**Performance:** GIN index provides sub-200ms on 1000+ rows easily. The success criterion of "sub-200ms on 1000+" is trivially met with GIN indexes.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Full-text search | Custom LIKE/ILIKE queries | PostgreSQL `tsvector` + GIN index | Stemming, ranking, boolean operators, sub-ms on millions of rows |
| Question parsing | Regex extraction of country/horizon/category | Gemini LLM structured output | Natural language is too varied for regex. "by end of year" -> horizon_days requires date math + context |
| Exponential decay | Python-side time weighting in a loop | SQL `EXP()` in aggregation query | One query vs N+1 pattern. PostgreSQL handles the math natively |
| Background task queue | Redis-backed Celery/RQ | `asyncio.create_task` + `Semaphore` | Overkill for 2-3 concurrent tasks. No external process to manage |
| Country-to-Goldstein mapping | Hardcoded severity lookup table | Existing `src/calibration/priors.py` CAMEO_TO_SUPER mapping + CAMEO quadrant taxonomy | Already codified in the codebase |

## Common Pitfalls

### Pitfall 1: N+1 Query in Country Risk Aggregation
**What goes wrong:** Computing risk scores by iterating Python-side over per-country prediction lists, issuing one query per country.
**Why it happens:** Natural to think "get countries, then for each country, get predictions."
**How to avoid:** Single CTE-based SQL query that does all aggregation server-side. The composite index `ix_predictions_country_created` already exists.
**Warning signs:** Response time > 500ms on the countries endpoint.

### Pitfall 2: Fixture Fallback Not Actually Gated
**What goes wrong:** Fixture code is "removed" but the imports remain, or the dev flag is checked inconsistently across endpoints.
**Why it happens:** The fixture fallback exists in 3 separate endpoints with different patterns.
**How to avoid:** Grep for all `_fixture_cache`, `_get_fixture_cache`, `load_fixture`, `create_mock_forecast`, `_guess_country_iso` references. Gate them all behind the same flag.
**Warning signs:** Myanmar still shows Syria's forecasts in production.

### Pitfall 3: Missing tsvector Backfill for Existing Rows
**What goes wrong:** `GENERATED ALWAYS AS` column is computed for new inserts, but existing rows may not be backfilled if the migration doesn't handle it.
**Why it happens:** PostgreSQL `GENERATED STORED` columns ARE automatically backfilled on `ALTER TABLE ADD COLUMN` -- this is actually fine. But if using a trigger-based approach instead, backfill is needed.
**How to avoid:** Use `GENERATED ALWAYS AS (to_tsvector('english', question)) STORED` which auto-computes for all existing rows during migration.

### Pitfall 4: LLM Question Parsing Returning Invalid Country Codes
**What goes wrong:** Gemini returns "MIDDLE_EAST" or "EU" instead of valid ISO codes. Or returns ISO-3 codes when the system uses ISO-2.
**Why it happens:** LLMs hallucinate country codes confidently. The existing `QuestionGenerator._parse_response()` already has this problem (line 332: `country_iso=str(item.get("country_iso", "XX")).upper()[:3]`).
**How to avoid:** Validate parsed country codes against a known-good set. The `SEED_COUNTRIES` list in `daily_forecast.py` has 32 entries. Maintain a comprehensive ISO code validation set.
**Warning signs:** `forecast_requests` rows with country_iso values like "EU", "XX", "MID".

### Pitfall 5: Concurrent Worker Exhausting Gemini Budget
**What goes wrong:** 3 concurrent user submissions each call Gemini, exhausting the daily budget (25 calls) in minutes.
**Why it happens:** `BudgetTracker` exists but the user submission path bypasses it.
**How to avoid:** The async worker must check `BudgetTracker.has_budget()` before invoking `EnsemblePredictor.predict()`. If budget exhausted, set status to `failed` with error "Budget exhausted, will retry tomorrow".
**Warning signs:** `gemini_budget_remaining` returns 0 but requests keep processing.

### Pitfall 6: Race Condition in Request Status Updates
**What goes wrong:** Two workers pick up the same `forecast_request` because status update from `pending` to `processing` isn't atomic.
**Why it happens:** Read-check-update pattern without row-level locking.
**How to avoid:** Use `SELECT ... FOR UPDATE SKIP LOCKED` to claim requests atomically:
```sql
UPDATE forecast_requests SET status = 'processing'
WHERE id = (
    SELECT id FROM forecast_requests
    WHERE status = 'confirmed'
    ORDER BY submitted_at
    LIMIT 1
    FOR UPDATE SKIP LOCKED
) RETURNING *;
```

## Code Examples

### Example 1: Fixture-Free Country Endpoint
```python
# src/api/routes/v1/forecasts.py -- get_forecasts_by_country after BAPI-01
@router.get("/country/{iso_code}", response_model=PaginatedResponse[ForecastResponse])
async def get_forecasts_by_country(
    iso_code: str,
    cursor: Optional[str] = Query(default=None),
    limit: int = Query(default=10, ge=1, le=50),
    _client: str = Depends(verify_api_key),
    cache: ForecastCache = Depends(get_cache),
    db: AsyncSession = Depends(get_db),
) -> PaginatedResponse[ForecastResponse]:
    iso_upper = iso_code.upper()

    # Cache check (first page only)
    if cursor is None:
        key = cache_key_for_country(iso_upper)
        cached = await cache.get(key)
        if cached is not None:
            items = [ForecastResponse(**item) for item in cached.get("items", [])]
            return PaginatedResponse[ForecastResponse](
                items=items[:limit],
                next_cursor=cached.get("next_cursor"),
                has_more=cached.get("has_more", False),
            )

    # PostgreSQL -- no fallback
    service = ForecastService(db)
    result = await service.get_forecasts_by_country(
        country_iso=iso_upper, cursor=cursor, limit=limit,
    )

    # Cache first page
    if cursor is None and result.items:
        data = {
            "items": [item.model_dump(mode="json") for item in result.items],
            "next_cursor": result.next_cursor,
            "has_more": result.has_more,
        }
        await cache.set(cache_key_for_country(iso_upper), data, ttl=SUMMARY_TTL)

    return result  # Empty items = empty items. Frontend handles it.
```

### Example 2: tsvector Search Endpoint
```python
@router.get("/search", response_model=PaginatedResponse[ForecastResponse])
async def search_forecasts(
    q: str = Query(..., min_length=2, max_length=200),
    country: Optional[str] = Query(default=None),
    category: Optional[str] = Query(default=None),
    limit: int = Query(default=20, ge=1, le=50),
    _client: str = Depends(verify_api_key),
    db: AsyncSession = Depends(get_db),
) -> PaginatedResponse[ForecastResponse]:
    query_ts = func.plainto_tsquery('english', q)
    stmt = (
        select(Prediction)
        .where(Prediction.question_tsv.op('@@')(query_ts))
    )
    if country:
        stmt = stmt.where(Prediction.country_iso == country.upper())
    if category:
        stmt = stmt.where(Prediction.category == category.lower())

    stmt = stmt.order_by(
        func.ts_rank(Prediction.question_tsv, query_ts).desc()
    ).limit(limit + 1)

    result = await db.execute(stmt)
    rows = result.scalars().all()
    has_more = len(rows) > limit
    items = [ForecastService.prediction_to_dto(r) for r in rows[:limit]]

    return PaginatedResponse[ForecastResponse](
        items=items, next_cursor=None, has_more=has_more,
    )
```

### Example 3: Question Parsing via Gemini
```python
_PARSE_QUESTION_PROMPT = """\
Parse this geopolitical forecast question into structured form.

QUESTION: {question}

Return ONLY a JSON object with:
- "country_iso_list": array of ISO 3166-1 alpha-2 codes (uppercase)
- "horizon_days": integer (7-365), inferred from temporal language
- "category": one of "conflict", "diplomatic", "economic", "security", "political", "GENERAL"

If the question mentions a region (e.g., "Middle East"), expand to relevant country codes.
If no time horizon is specified, default to 30 days.
If category is ambiguous, use "GENERAL".

Return ONLY the JSON object. No markdown.
"""
```

## Migration Plan

**New Alembic migration (004):**

```python
def upgrade() -> None:
    # 1. forecast_requests table
    op.create_table(
        "forecast_requests",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("question", sa.Text(), nullable=False),
        sa.Column("country_iso_list", JSON(), nullable=False, server_default="[]"),
        sa.Column("horizon_days", sa.Integer(), nullable=False, server_default="30"),
        sa.Column("category", sa.String(32), nullable=False, server_default="'GENERAL'"),
        sa.Column("status", sa.String(20), nullable=False, server_default="'pending'"),
        sa.Column("submitted_by", sa.String(100), nullable=False),
        sa.Column("submitted_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
        sa.Column("prediction_ids", JSON(), nullable=False, server_default="[]"),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("retry_count", sa.Integer(), server_default="0"),
        sa.Column("parsed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_forecast_requests_submitted_by", "forecast_requests", ["submitted_by"])
    op.create_index("ix_forecast_requests_status", "forecast_requests", ["status"])

    # 2. tsvector generated column + GIN index on predictions.question
    op.execute("""
        ALTER TABLE predictions
        ADD COLUMN question_tsv tsvector
        GENERATED ALWAYS AS (to_tsvector('english', question)) STORED
    """)
    op.execute("""
        CREATE INDEX ix_predictions_question_tsv
        ON predictions USING GIN (question_tsv)
    """)
```

## Open Questions

1. **Confirmation step UX flow:** CONTEXT says "show parsed form to user for confirmation before queueing." This implies a two-step API: parse then confirm. But the CONTEXT also says this is backend-only phase. Should we build both endpoints now (submit + confirm) or just submit with auto-confirm? Recommendation: build both, the frontend will use them in Phase 15.

2. **Risk score caching strategy:** Country risk scores involve an expensive aggregation query. How often should they be recomputed? Options:
   - On every request with cache (current SUMMARY_TTL = 1 hour)
   - Materialized view refreshed by daily pipeline
   - Hybrid: cache with 15-minute TTL, background refresh
   Recommendation: Cache with 15-minute TTL. The query is fast with proper indexes.

3. **Search suggestions on empty results:** CONTEXT says "return empty results plus LLM-generated search suggestions." This requires a Gemini call on empty search results, which adds latency and budget cost. Should this be deferred or implemented with a simpler heuristic (e.g., return related categories from the predictions table)?

4. **CountryRiskSummary DTO breaking change:** The `risk_score` field needs to change from 0.0-1.0 to 0-100 per BAPI-02. The frontend TypeScript type must be updated. Since Phase 14 is backend-only, do we document this as a breaking change for Phase 15 to handle?

## Sources

### Primary (HIGH confidence -- direct codebase analysis)
- `src/api/routes/v1/forecasts.py` -- fixture fallback code, endpoint structure
- `src/api/routes/v1/countries.py` -- 100% mock data, no DB integration
- `src/db/models.py` -- all ORM models including Prediction schema
- `src/api/services/forecast_service.py` -- persist + query patterns
- `src/api/schemas/forecast.py` -- ForecastResponse DTO contract
- `src/api/schemas/country.py` -- CountryRiskSummary DTO contract
- `src/api/fixtures/factory.py` -- fixture generation code
- `alembic/versions/*` -- 3 existing migrations (001-003)
- `src/forecasting/ensemble_predictor.py` -- predict() signature and flow
- `src/pipeline/question_generator.py` -- existing LLM question parsing
- `src/calibration/priors.py` -- CAMEO super-category mapping
- `src/settings.py` -- Settings class, no USE_FIXTURES flag
- `frontend/src/types/api.ts` -- TypeScript DTO mirrors
- `frontend/src/services/forecast-client.ts` -- frontend API client
- `docker-compose.yml` -- PostgreSQL 16-alpine confirmed
- `scripts/daily_forecast.py` -- seed-countries mode, pipeline init

## Metadata

**Confidence breakdown:**
- Fixture removal (BAPI-01): HIGH -- direct code reading, exact lines identified
- Country risk aggregation (BAPI-02): HIGH -- schema known, SQL pattern clear, CAMEO mapping exists
- Question submission queue (BAPI-03): HIGH -- pipeline architecture understood, existing patterns to follow
- Full-text search (BAPI-04): HIGH -- PostgreSQL 16 tsvector support confirmed, SQLAlchemy 2.0 pattern clear

**Research date:** 2026-03-03
**Valid until:** N/A (codebase research, valid as long as codebase doesn't change)
