---
phase: 09-api-foundation
plan: 01
subsystem: database, infra
tags: [postgresql, asyncpg, sqlalchemy, alembic, docker, redis, pydantic-settings, fastapi]

requires:
  - phase: 08-graph-partitioning
    provides: "v1.1 complete codebase with SQLite-only persistence"
provides:
  - "PostgreSQL ORM models (5 tables: predictions, outcome_records, calibration_weights, ingest_runs, api_keys)"
  - "Async SQLAlchemy engine with connection pooling (asyncpg driver)"
  - "Alembic async migration environment with initial schema"
  - "SQLite WAL-mode connection manager for GDELT store"
  - "Docker Compose dev environment (postgres:16, redis:7, api service)"
  - "Centralized settings via pydantic-settings with env file support"
  - "Dockerfile for API container"
affects: [09-02, 09-03, 09-05, 09-06, 10-ingest-pipeline, 12-frontend, 13-calibration-monitoring]

tech-stack:
  added: [fastapi, uvicorn, asyncpg, alembic, redis, pydantic-settings, python-multipart]
  removed: [jraph]
  patterns: ["async SQLAlchemy session via generator with auto commit/rollback", "pydantic-settings BaseSettings singleton", "dual database managers (PostgreSQL async + SQLite sync)"]

key-files:
  created:
    - src/settings.py
    - src/db/__init__.py
    - src/db/models.py
    - src/db/postgres.py
    - src/db/sqlite.py
    - docker-compose.yml
    - Dockerfile
    - .dockerignore
    - alembic.ini
    - alembic/env.py
    - alembic/script.py.mako
    - alembic/versions/20260301_1443_001_initial_schema.py
  modified:
    - pyproject.toml
    - .env.example

key-decisions:
  - "extra=ignore in pydantic-settings to coexist with legacy .env vars (GEMINI_API_KEY, etc.)"
  - "DateTime(timezone=True) for all timestamp columns -- timezone-aware from the start"
  - "Manual Alembic migration instead of autogenerate due to Docker unavailability during execution"
  - "Prediction.id is String(36) UUID, not Integer -- stable IDs across systems"

patterns-established:
  - "get_async_session() generator: yields session, auto-commits on success, rolls back on exception"
  - "SQLiteConnection context manager with WAL/busy_timeout/synchronous pragmas"
  - "Settings singleton: get_settings() caches a single Settings instance"
  - "Docker Compose healthcheck with service_healthy condition for dependency ordering"

duration: 6min
completed: 2026-03-01
---

# Phase 9 Plan 01: Dependencies, Docker, Database Foundation Summary

**Dual-database persistence layer (PostgreSQL async via asyncpg + SQLite WAL for GDELT), Docker Compose dev environment, Alembic migration for 5 tables, and pydantic-settings configuration**

## Performance

- **Duration:** 6 min
- **Started:** 2026-03-01T09:09:44Z
- **Completed:** 2026-03-01T09:15:44Z
- **Tasks:** 2
- **Files modified:** 16

## Accomplishments

- 5 PostgreSQL ORM models using SQLAlchemy 2.0 `Mapped[T]` + `mapped_column()` with timezone-aware datetimes and proper indices
- Async engine factory with connection pooling (pool_size=5, max_overflow=10) and session generator with auto commit/rollback
- SQLite connection manager retaining WAL mode with 30s busy_timeout for GDELT event store
- Alembic async migration environment with hand-written initial schema (verified via offline SQL generation)
- Docker Compose with postgres:16-alpine, redis:7-alpine, and API service with healthcheck-gated dependencies
- Centralized settings via pydantic-settings with env file support and singleton pattern
- jraph removed from pyproject.toml (archived by DeepMind), version bumped to 2.0.0-dev

## Task Commits

Each task was committed atomically:

1. **Task 1: Dependencies, Settings, and Docker infrastructure** - `16c7657` (feat)
2. **Task 2: PostgreSQL ORM models, async connection manager, and Alembic migrations** - `f7d8889` (feat)

## Files Created/Modified

- `pyproject.toml` - Bumped to 2.0.0-dev, added 7 new deps, removed jraph, added sqlalchemy[asyncio] extra
- `src/settings.py` - Centralized pydantic-settings with DATABASE_URL, REDIS_URL, GDELT_DB_PATH, CORS, logging config
- `docker-compose.yml` - PostgreSQL 16, Redis 7, API service with healthcheck-gated dependencies
- `Dockerfile` - Python 3.11-slim + uv, copies app code and alembic
- `.env.example` - Updated with v2.0 config vars alongside v1.x legacy settings
- `.dockerignore` - Excludes data/, .git/, tests/, .planning/, .env
- `src/db/__init__.py` - Package docstring for dual-database layer
- `src/db/models.py` - 5 ORM models: Prediction, OutcomeRecord, CalibrationWeight, IngestRun, ApiKey
- `src/db/postgres.py` - Async engine factory, session generator, init_db()/close_db() lifecycle
- `src/db/sqlite.py` - SQLiteConnection with WAL mode, busy_timeout, synchronous=NORMAL
- `alembic.ini` - Alembic configuration pointing to alembic/ directory
- `alembic/env.py` - Async migration environment loading DATABASE_URL from settings
- `alembic/script.py.mako` - Migration template
- `alembic/versions/20260301_1443_001_initial_schema.py` - Initial migration creating all 5 tables

## Decisions Made

1. **`extra="ignore"` in pydantic-settings** -- The existing `.env` file contains `GEMINI_API_KEY` and other legacy vars not defined in Settings. Without `extra="ignore"`, Settings construction fails with a validation error. This is the correct approach -- Settings only validates what it cares about.

2. **`DateTime(timezone=True)` on all timestamp columns** -- Avoids the `datetime.utcnow()` deprecation issue (already noted in STATE.md as deferred). All timestamps are timezone-aware from the start.

3. **Manual Alembic migration** -- Docker daemon was not running and requires sudo to start. The migration was hand-written to match the ORM models exactly, then verified via `alembic upgrade head --sql` (offline mode) which generates correct PostgreSQL DDL for all 5 tables. This is structurally equivalent to autogenerate and avoids drift since both derive from the same models.

4. **Prediction.id as String(36) UUID** -- Forecasts need stable IDs that work across systems (PostgreSQL, API responses, frontend references). Integer auto-increment creates coupling to a single database instance.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] pydantic-settings rejected extra env vars**
- **Found during:** Task 1 (Settings verification)
- **Issue:** `.env` contains `GEMINI_API_KEY` which is not defined in Settings. pydantic-settings defaults to `extra="forbid"`, causing a ValidationError.
- **Fix:** Added `extra="ignore"` to model_config. Settings now ignores env vars it doesn't recognize.
- **Files modified:** `src/settings.py`
- **Verification:** `get_settings().database_url` returns correct default
- **Committed in:** `16c7657` (Task 1 commit)

---

**Total deviations:** 1 auto-fixed (1 bug)
**Impact on plan:** Minor config fix. No scope creep.

## Issues Encountered

**Docker daemon unavailable** -- The Docker daemon was inactive (`systemctl is-active docker` returned "inactive") and the user is not in the `docker` group. Starting Docker requires `sudo systemctl start docker`, which requires a password. This prevented:
- `docker compose up -d postgres redis` (container startup)
- `alembic upgrade head` against a live PostgreSQL (migration application)
- The async session connection test (`SELECT 1`)

**Mitigation:** All code artifacts are complete and verified via non-Docker means:
- Migration SQL verified via `alembic upgrade head --sql` (offline mode) -- generates correct DDL
- All Python modules import cleanly
- SQLite WAL mode verified against actual database
- ORM model metadata confirmed: 5 tables with correct names

**To complete Docker verification:**
```bash
sudo systemctl start docker
docker compose up -d postgres redis
uv run alembic upgrade head
uv run python -c "
import asyncio
from src.db.postgres import get_async_session
from sqlalchemy import text
async def test():
    async for session in get_async_session():
        result = await session.execute(text('SELECT 1'))
        print('Connection OK:', result.scalar())
        break
asyncio.run(test())
"
```

## User Setup Required

**Docker daemon must be started before PostgreSQL/Redis are available:**
```bash
sudo systemctl start docker
sudo usermod -aG docker $USER  # Optional: avoid sudo for future sessions
docker compose up -d postgres redis
uv run alembic upgrade head
```

## Next Phase Readiness

- All ORM models ready for DTO mapping in Plan 09-02
- Async session pattern ready for FastAPI dependency injection in Plan 09-05
- Alembic migration ready to apply once Docker is available
- Settings module ready for import by all subsequent Phase 9 plans
- Docker verification deferred to user (requires `sudo systemctl start docker`)

---
*Phase: 09-api-foundation*
*Completed: 2026-03-01*
