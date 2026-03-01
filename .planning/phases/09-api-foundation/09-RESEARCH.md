# Phase 9: API Foundation & Infrastructure - Research

**Researched:** 2026-03-01
**Domain:** FastAPI + SQLAlchemy async + PostgreSQL + Docker + jraph elimination + structured logging
**Confidence:** HIGH

## Summary

Phase 9 establishes the entire v2.0 foundation: a dual-database persistence layer (PostgreSQL for forecasts, SQLite retained for GDELT events), a headless FastAPI server with mock fixtures defining the DTO contract, structured logging replacing all `print()` statements, jraph elimination from the JAX training code, and a `TKGModelProtocol` for swappable model backends.

The existing codebase is well-structured for this transition. The database layer (`src/database/`) is currently SQLite-only with raw connection management. The calibration module (`src/calibration/prediction_store.py`) already uses SQLAlchemy ORM with a `Prediction` model -- this is a direct template for the PostgreSQL migration. The jraph dependency is isolated to exactly two files (`src/training/models/regcn_jraph.py` and `src/training/train_jraph.py`) plus one script (`scripts/train_tkg_jraph.py`). The pure-JAX model (`regcn_jax.py`) already uses a local `GraphSnapshot` NamedTuple and `jax.lax.fori_loop` with `jnp.zeros().at[].add()` instead of `jraph.segment_sum` -- it is the template for the replacement. There are 60+ `print()` statements in production code across 8 files that need conversion to `logging` calls.

**Primary recommendation:** Use SQLAlchemy 2.0 async ORM with asyncpg driver for PostgreSQL, Alembic async migrations, FastAPI with dependency-injected async sessions, and hand-rolled RFC 9457 error responses (no third-party library needed). The jraph elimination is a 2-hour mechanical replacement since `regcn_jax.py` already demonstrates the pattern.

## Standard Stack

### Core (New Dependencies)

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `fastapi` | `>=0.115` | Async web framework | De facto Python API framework, auto-generates OpenAPI from Pydantic models |
| `uvicorn[standard]` | `>=0.34` | ASGI server | FastAPI's recommended production server |
| `asyncpg` | `>=0.30` | Native async PostgreSQL driver | 3-5x faster than psycopg2-async, required for SQLAlchemy async |
| `sqlalchemy[asyncio]` | `>=2.0` (already in deps) | Async ORM + connection pooling | Already a dependency, just needs async extension |
| `alembic` | `>=1.14` | Database migrations | Standard SQLAlchemy migration tool, supports async via template |
| `redis[hiredis]` | `>=5.0` | Async Redis client (aioredis merged) | `redis-py` absorbed aioredis; hiredis parser for performance |
| `python-multipart` | `>=0.0.9` | Form data parsing for FastAPI | Required by FastAPI for form/file uploads |
| `httptools` | `>=0.6` | Fast HTTP parsing for uvicorn | Part of uvicorn[standard], significant performance improvement |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `pydantic` | `>=2.0` (already in deps) | DTO validation, OpenAPI schema generation | All API models |
| `pydantic-settings` | `>=2.0` | Settings management from env vars | Configuration loading |

### Already Present (No Change)

| Library | Purpose | Notes |
|---------|---------|-------|
| `sqlalchemy>=2.0.0` | ORM, already used in `prediction_store.py` | Add `[asyncio]` extra |
| `pydantic>=2.0` | Already used in `src/forecasting/models.py` | Perfect for DTOs |
| `python-dotenv>=1.0.0` | Environment variable loading | Already present |

### Remove

| Library | Reason | Replacement |
|---------|--------|-------------|
| `jraph>=0.0.6.dev0` | Archived by Google DeepMind 2025-05-21, read-only | Local `GraphsTuple` NamedTuple + `jax.ops.segment_sum` |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| asyncpg | psycopg3 (async) | psycopg3 is more Pythonic but asyncpg is 3-5x faster for connection pooling; asyncpg is the de facto choice for SQLAlchemy async + PostgreSQL |
| fastapi-rfc7807 | Hand-rolled exception handlers | fastapi-rfc7807 is a thin wrapper with minimal maintenance activity; hand-rolling 30 lines of exception handler code is simpler and avoids a dependency |
| fastapi-pagination | Manual cursor pagination | The library adds offset/cursor/keyset but is overkill for 3 list endpoints; manual implementation is ~40 lines and fully controlled |
| redis-py | aioredis | aioredis was merged into redis-py 5.0+; use `redis[hiredis]` directly |

**Installation:**
```bash
uv add fastapi "uvicorn[standard]" asyncpg "sqlalchemy[asyncio]" alembic "redis[hiredis]" pydantic-settings python-multipart
uv remove jraph
```

## Architecture Patterns

### Recommended Project Structure

```
src/
├── api/                          # NEW - FastAPI application
│   ├── __init__.py
│   ├── app.py                    # FastAPI app factory, middleware registration
│   ├── deps.py                   # Dependency injection (DB sessions, auth)
│   ├── errors.py                 # RFC 9457 Problem Details exception handlers
│   ├── middleware/
│   │   ├── __init__.py
│   │   ├── auth.py               # API key validation middleware
│   │   └── cors.py               # CORS configuration
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── v1/
│   │   │   ├── __init__.py
│   │   │   ├── router.py         # v1 router aggregator
│   │   │   ├── health.py         # GET /api/v1/health
│   │   │   ├── forecasts.py      # Forecast CRUD endpoints
│   │   │   └── countries.py      # Country risk endpoints
│   ├── schemas/                   # Pydantic DTOs
│   │   ├── __init__.py
│   │   ├── forecast.py           # ForecastResponse, ScenarioDTO, etc.
│   │   ├── health.py             # HealthResponse
│   │   ├── country.py            # CountryRiskSummary
│   │   └── common.py             # ProblemDetail, PaginatedResponse
│   └── fixtures/                  # Mock data
│       ├── __init__.py
│       ├── factory.py            # Random forecast factory
│       └── scenarios/
│           ├── syria.json
│           ├── ukraine.json
│           └── myanmar.json
├── db/                            # NEW - Database layer (replaces parts of src/database/)
│   ├── __init__.py
│   ├── postgres.py               # Async engine, session factory, connection pool
│   ├── sqlite.py                 # SQLite connection (retained for GDELT, with WAL+busy_timeout)
│   ├── models.py                 # SQLAlchemy ORM models (Prediction, OutcomeRecord, etc.)
│   └── migrations/               # Alembic
│       ├── alembic.ini
│       ├── env.py
│       └── versions/
├── protocols/                     # NEW - Protocol definitions
│   ├── __init__.py
│   └── tkg.py                    # TKGModelProtocol
├── database/                      # EXISTING - GDELT event storage (SQLite, retained as-is)
├── forecasting/                   # EXISTING - ensemble, TKG, etc.
├── training/                      # EXISTING - models get jraph eliminated
└── ...
```

### Pattern 1: Async Database Session via Dependency Injection

**What:** Each FastAPI request gets its own async SQLAlchemy session, automatically committed/rolled back.
**When to use:** Every route handler that touches PostgreSQL.

```python
# src/db/postgres.py
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

engine = create_async_engine(
    "postgresql+asyncpg://user:pass@localhost:5432/geopol",
    pool_size=5,          # Concurrent connections in pool
    max_overflow=10,      # Extra connections above pool_size
    pool_timeout=30,      # Seconds to wait for connection
    pool_recycle=1800,    # Recycle connections every 30 min
    echo=False,           # Set True for SQL debugging
)

async_session_factory = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,  # Allow access to ORM objects after commit
    autoflush=False,         # Explicit flush/commit only
)

async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    async with async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
```

```python
# src/api/deps.py
from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession
from src.db.postgres import get_async_session

async def get_db(session: AsyncSession = Depends(get_async_session)) -> AsyncSession:
    return session
```

```python
# src/api/routes/v1/forecasts.py
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from src.api.deps import get_db

router = APIRouter()

@router.get("/forecasts/{forecast_id}")
async def get_forecast(forecast_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Prediction).where(Prediction.forecast_id == forecast_id))
    prediction = result.scalar_one_or_none()
    if not prediction:
        raise HTTPException(status_code=404, detail="Forecast not found")
    return ForecastResponse.from_orm(prediction)
```

### Pattern 2: RFC 9457 Problem Details (Hand-Rolled)

**What:** Standardized error response format for all API errors.
**When to use:** All error responses from the API.

```python
# src/api/errors.py
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel
from typing import Optional

class ProblemDetail(BaseModel):
    type: str = "about:blank"
    title: str
    status: int
    detail: Optional[str] = None
    instance: Optional[str] = None

async def problem_detail_handler(request: Request, exc: Exception) -> JSONResponse:
    if isinstance(exc, RequestValidationError):
        return JSONResponse(
            status_code=422,
            content=ProblemDetail(
                type="/errors/validation",
                title="Validation Error",
                status=422,
                detail=str(exc),
                instance=str(request.url),
            ).model_dump(),
            media_type="application/problem+json",
        )
    # ... other exception types

def register_error_handlers(app: FastAPI) -> None:
    app.add_exception_handler(RequestValidationError, problem_detail_handler)
    app.add_exception_handler(404, not_found_handler)
    # etc.
```

### Pattern 3: API Key Authentication Middleware

**What:** Validate `X-API-Key` header against the `api_keys` table.
**When to use:** All endpoints except `GET /api/v1/health` (public).

```python
# src/api/middleware/auth.py
from fastapi import Security, HTTPException
from fastapi.security import APIKeyHeader

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def verify_api_key(
    api_key: str = Security(api_key_header),
    db: AsyncSession = Depends(get_db),
) -> str:
    if not api_key:
        raise HTTPException(status_code=401, detail="Missing API key")
    result = await db.execute(
        select(ApiKey).where(ApiKey.key == api_key, ApiKey.revoked == False)
    )
    key_record = result.scalar_one_or_none()
    if not key_record:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return key_record.client_name
```

### Pattern 4: Dual Database Managers

**What:** Separate, explicit connection managers for PostgreSQL and SQLite.
**When to use:** The calling code always knows which database it is targeting.

```python
# PostgreSQL: async sessions via src/db/postgres.py (see Pattern 1)
# SQLite: synchronous, retained for GDELT events

# src/db/sqlite.py
import sqlite3
from contextlib import contextmanager

class SQLiteConnection:
    def __init__(self, db_path: str = "data/events.db"):
        self.db_path = db_path

    @contextmanager
    def get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA busy_timeout = 30000")  # 30s busy timeout
        conn.execute("PRAGMA synchronous = NORMAL")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
```

### Pattern 5: Cursor-Based Pagination

**What:** List endpoints use opaque cursor tokens, not offset/limit.
**When to use:** `GET /api/v1/forecasts/country/{iso}`, `GET /api/v1/forecasts/top`.

```python
# src/api/schemas/common.py
from pydantic import BaseModel
from typing import Generic, TypeVar, Optional, List
import base64, json

T = TypeVar("T")

class PaginatedResponse(BaseModel, Generic[T]):
    items: List[T]
    next_cursor: Optional[str] = None
    has_more: bool = False

def encode_cursor(forecast_id: str, created_at: str) -> str:
    return base64.urlsafe_b64encode(
        json.dumps({"id": forecast_id, "ts": created_at}).encode()
    ).decode()

def decode_cursor(cursor: str) -> dict:
    return json.loads(base64.urlsafe_b64decode(cursor.encode()).decode())
```

### Anti-Patterns to Avoid

- **Unified repository abstraction over PostgreSQL + SQLite:** The CONTEXT.md explicitly rejects this. Two databases, two connection managers, calling code knows which it talks to.
- **Sync database operations in async handlers:** All PostgreSQL operations must use `await`. Never use synchronous SQLAlchemy with FastAPI async endpoints.
- **N+1 queries in forecast responses:** The fully-nested DTO means a single `GET /forecasts/{id}` must load the complete tree. Use `selectinload()` or `joinedload()` for eager loading in one query.
- **Shared mutable state between workers:** FastAPI with uvicorn workers shares nothing. Database is the communication channel.
- **`response_model` with ORM objects directly:** Always map ORM models to Pydantic DTOs explicitly. SQLAlchemy models and API schemas are separate layers.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Database migrations | Manual ALTER TABLE scripts | Alembic with `--autogenerate` | Schema drift, rollback support, team collaboration |
| Connection pooling | Manual connection recycling | SQLAlchemy async engine pool (`pool_size`, `max_overflow`) | Race conditions, leak detection, health checks built-in |
| OpenAPI documentation | Manual swagger.json | FastAPI auto-generates from Pydantic DTOs | Always in sync with code, zero maintenance |
| ASGI server | Custom event loop | uvicorn[standard] | Production-grade, handles signals, graceful shutdown |
| JSON serialization for DTOs | Manual `dict()` calls | Pydantic `.model_dump()` / `.model_validate()` | Handles nested models, datetime serialization, validation |
| Password/key hashing | Custom hash function | `secrets.token_urlsafe(32)` for API key generation | Cryptographically secure, standard library |

**Key insight:** FastAPI + Pydantic + SQLAlchemy async is a battle-tested stack. The framework handles serialization, validation, documentation, and dependency injection. Custom solutions for any of these are strictly worse.

## Common Pitfalls

### Pitfall 1: SQLAlchemy Async Session Lifecycle Mismanagement

**What goes wrong:** Sessions left open, connections leak, pool exhaustion under load.
**Why it happens:** Forgetting `await session.close()` in error paths, or reusing sessions across requests.
**How to avoid:** Always use `async with` or FastAPI's `Depends()` pattern which guarantees cleanup via generator lifecycle. The `get_async_session` generator pattern (shown above) handles commit/rollback/close automatically.
**Warning signs:** `TimeoutError` from connection pool, growing PostgreSQL connection count.

### Pitfall 2: Alembic Async Migration env.py Configuration

**What goes wrong:** Alembic migration commands fail with "asyncio" errors or hang.
**Why it happens:** Default Alembic env.py uses synchronous engine. Must use `alembic init -t async` or manually update `run_migrations_online()` to use `run_async()` with an async engine.
**How to avoid:** Initialize with `alembic init -t async alembic` to get the correct template. The env.py must use `connectable = async_engine_from_config(...)` and `async with connectable.connect() as connection: await connection.run_sync(do_run_migrations)`.
**Warning signs:** `RuntimeError: no current event loop` during migration.

### Pitfall 3: Mixed Sync/Async Database Access

**What goes wrong:** Blocking the event loop by calling synchronous SQLite operations from async FastAPI handlers.
**Why it happens:** The GDELT SQLite store uses synchronous `sqlite3` connections, but FastAPI handlers are async.
**How to avoid:** SQLite operations (GDELT reads) that must be called from async context should use `asyncio.to_thread()` or `run_in_executor()`. Alternatively, keep SQLite access in separate synchronous functions called from non-async background tasks.
**Warning signs:** API latency spikes when GDELT queries run, event loop blocked warnings.

### Pitfall 4: jraph Import Breakage During Elimination

**What goes wrong:** Removing jraph from `pyproject.toml` breaks `import src.training.models.regcn_jraph` which is imported by `train_jraph.py` and `scripts/train_tkg_jraph.py`.
**Why it happens:** The jraph elimination must be atomic -- remove the dependency AND update all importing code in the same commit.
**How to avoid:** The replacement is mechanical. `regcn_jax.py` (which does NOT use jraph) already demonstrates the pattern: a local `GraphSnapshot` NamedTuple with `edge_index`, `edge_type`, `num_edges` fields, and `jnp.zeros().at[target_idx].add()` instead of `jraph.segment_sum`. Apply this same pattern to `regcn_jraph.py`.
**Warning signs:** `ModuleNotFoundError: No module named 'jraph'` at import time.

### Pitfall 5: Pydantic V2 Forward Reference in Recursive DTOs

**What goes wrong:** `ScenarioDTO.child_scenarios: list[ScenarioDTO]` fails with forward reference error.
**Why it happens:** Pydantic V2 handles self-referential models differently than V1.
**How to avoid:** Use `model_rebuild()` after class definition, or define the field with `Optional[list["ScenarioDTO"]]` and call `ScenarioDTO.model_rebuild()` at module level.
**Warning signs:** `PydanticUserError` about undefined type during schema generation.

### Pitfall 6: Docker Compose Postgres Readiness Race

**What goes wrong:** FastAPI container starts before PostgreSQL is ready, connection refused errors.
**Why it happens:** `depends_on` in docker-compose only waits for container start, not service readiness.
**How to avoid:** Use `depends_on` with `condition: service_healthy` and a `healthcheck` on the postgres service (`pg_isready`). Or use a wait-for-it script in the API entrypoint.
**Warning signs:** Connection refused errors on first startup, works after manual restart.

## Code Examples

### PostgreSQL ORM Models

```python
# src/db/models.py
import uuid
from datetime import datetime
from typing import Optional
from sqlalchemy import (
    Column, String, Float, Integer, Text, DateTime, Boolean, JSON, ForeignKey, Index
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.dialects.postgresql import UUID

class Base(DeclarativeBase):
    pass

class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    question: Mapped[str] = mapped_column(Text, nullable=False)
    prediction: Mapped[str] = mapped_column(Text, nullable=False)
    probability: Mapped[float] = mapped_column(Float, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    horizon_days: Mapped[int] = mapped_column(Integer, nullable=False, default=30)
    category: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    reasoning_summary: Mapped[str] = mapped_column(Text, nullable=False)
    evidence_count: Mapped[int] = mapped_column(Integer, default=0)
    # Full DTO stored as JSON for reconstruction
    scenarios_json: Mapped[dict] = mapped_column(JSON, nullable=False, default=list)
    ensemble_info_json: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    calibration_json: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    entities: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    country_iso: Mapped[Optional[str]] = mapped_column(String(3), nullable=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)
    expires_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)

    __table_args__ = (
        Index("ix_predictions_country_created", "country_iso", "created_at"),
    )

class OutcomeRecord(Base):
    __tablename__ = "outcome_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    prediction_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("predictions.id"), nullable=False, index=True
    )
    outcome: Mapped[float] = mapped_column(Float, nullable=False)  # 0.0 or 1.0
    resolution_date: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    resolution_method: Mapped[str] = mapped_column(String(50), nullable=False)
    evidence_gdelt_ids: Mapped[list] = mapped_column(JSON, default=list)
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

class CalibrationWeight(Base):
    __tablename__ = "calibration_weights"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    cameo_code: Mapped[str] = mapped_column(String(10), nullable=False, unique=True)
    alpha: Mapped[float] = mapped_column(Float, nullable=False)  # LLM weight
    sample_size: Mapped[int] = mapped_column(Integer, nullable=False)
    brier_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

class IngestRun(Base):
    __tablename__ = "ingest_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    started_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    status: Mapped[str] = mapped_column(String(20), nullable=False)  # running/success/failed
    events_fetched: Mapped[int] = mapped_column(Integer, default=0)
    events_new: Mapped[int] = mapped_column(Integer, default=0)
    events_duplicate: Mapped[int] = mapped_column(Integer, default=0)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

class ApiKey(Base):
    __tablename__ = "api_keys"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    key: Mapped[str] = mapped_column(String(64), nullable=False, unique=True, index=True)
    client_name: Mapped[str] = mapped_column(String(100), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    revoked: Mapped[bool] = mapped_column(Boolean, default=False)
```

### TKGModelProtocol Definition

```python
# src/protocols/tkg.py
from typing import List, Optional, Protocol, Tuple, runtime_checkable
from jax import Array

@runtime_checkable
class TKGModelProtocol(Protocol):
    """Protocol for temporal knowledge graph prediction models.

    Any TKG model (RE-GCN, TiRGN, etc.) must satisfy this interface
    to be used by the EnsemblePredictor and training infrastructure.
    """

    num_entities: int
    num_relations: int
    embedding_dim: int

    def evolve_embeddings(
        self,
        snapshots: list,
        training: bool = False,
    ) -> Array:
        """Evolve entity embeddings through temporal graph snapshots.

        Returns:
            Entity embeddings array of shape (num_entities, embedding_dim)
        """
        ...

    def compute_scores(
        self,
        entity_emb: Array,
        triples: Array,
    ) -> Array:
        """Score triples given entity embeddings.

        Args:
            entity_emb: (num_entities, embedding_dim)
            triples: (batch, 3) of [subject, relation, object]

        Returns:
            Scores array of shape (batch,)
        """
        ...

    def compute_loss(
        self,
        snapshots: list,
        pos_triples: Array,
        neg_triples: Array,
        margin: float = 1.0,
        **kwargs,
    ) -> Array:
        """Compute training loss for positive and negative triples.

        Returns:
            Scalar loss value
        """
        ...
```

**Verification pattern:**
```python
from src.protocols.tkg import TKGModelProtocol
from src.training.models.regcn_jax import REGCN

# Structural check (no explicit inheritance needed)
model = REGCN(num_entities=100, num_relations=20, rngs=nnx.Rngs(0))
assert isinstance(model, TKGModelProtocol)
```

### jraph Elimination: GraphsTuple Replacement

```python
# In regcn_jraph.py, replace:
#   import jraph
#   jraph.GraphsTuple(...)
#   jraph.segment_sum(...)

# With local NamedTuple (already demonstrated by regcn_jax.py's GraphSnapshot):
from typing import NamedTuple, Optional
import jax
import jax.numpy as jnp

class GraphsTuple(NamedTuple):
    """Local replacement for jraph.GraphsTuple (archived library)."""
    nodes: Optional[jax.Array]
    edges: Optional[jax.Array]
    senders: jax.Array
    receivers: jax.Array
    n_node: jax.Array
    n_edge: jax.Array
    globals: Optional[jax.Array] = None

# Replace jraph.segment_sum with jax.ops.segment_sum (identical API):
def segment_sum(data, segment_ids, num_segments):
    return jax.ops.segment_sum(data, segment_ids, num_segments=num_segments)

# The RGCNLayer body_fn changes from:
#   rel_aggregated = jraph.segment_sum(masked_messages, receivers, num_segments=num_nodes)
# To:
#   rel_aggregated = jax.ops.segment_sum(masked_messages, receivers, num_segments=num_nodes)
```

### Docker Compose

```yaml
# docker-compose.yml
services:
  postgres:
    image: postgres:16-alpine
    environment:
      POSTGRES_DB: geopol
      POSTGRES_USER: geopol
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD:-geopol_dev}
    ports:
      - "5432:5432"
    volumes:
      - pgdata:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U geopol"]
      interval: 5s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 5s
      timeout: 3s
      retries: 5

  api:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    environment:
      DATABASE_URL: postgresql+asyncpg://geopol:${POSTGRES_PASSWORD:-geopol_dev}@postgres:5432/geopol
      REDIS_URL: redis://redis:6379/0
      ENVIRONMENT: development
      GDELT_DB_PATH: /app/data/events.db
    volumes:
      - ./data:/app/data  # Mount GDELT SQLite DB
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy

volumes:
  pgdata:
```

### Structured Logging Configuration

```python
# src/logging_config.py
import logging
import sys
from typing import Optional

def setup_logging(level: str = "INFO", json_format: bool = False) -> None:
    """Configure structured logging for all production code.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        json_format: If True, emit JSON log lines (for production)
    """
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    handler = logging.StreamHandler(sys.stderr)

    if json_format:
        # JSON structured logging for production
        import json
        class JSONFormatter(logging.Formatter):
            def format(self, record: logging.LogRecord) -> str:
                return json.dumps({
                    "timestamp": self.formatTime(record),
                    "level": record.levelname,
                    "module": record.module,
                    "message": record.getMessage(),
                    "logger": record.name,
                })
        handler.setFormatter(JSONFormatter())
    else:
        # Human-readable for development
        handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))

    root.handlers.clear()
    root.addHandler(handler)
```

## Existing Codebase Findings

### Files Requiring jraph Elimination

| File | jraph Usage | Replacement Strategy |
|------|-------------|---------------------|
| `src/training/models/regcn_jraph.py` | `import jraph`, `jraph.GraphsTuple`, `jraph.segment_sum` | Replace with local `GraphsTuple` NamedTuple + `jax.ops.segment_sum` |
| `src/training/train_jraph.py` | Imports from `regcn_jraph` | Update imports after `regcn_jraph.py` is fixed |
| `scripts/train_tkg_jraph.py` | Imports from `regcn_jraph` and `train_jraph` | Update imports after source files are fixed |
| `pyproject.toml` line 48 | `"jraph>=0.0.6.dev0"` | Remove this dependency |

**Critical observation:** `regcn_jax.py` (the pure-JAX implementation) does NOT use jraph at all. It already has a local `GraphSnapshot` NamedTuple and uses `jnp.zeros().at[target_idx].add()` for message aggregation. This file is the living template for the jraph replacement in `regcn_jraph.py`. The code in `regcn_jraph.py` uses `jraph.segment_sum` in exactly 2 locations (lines 159 and 170).

### Files Requiring print() -> logging Conversion

| File | print() Count | Severity |
|------|---------------|----------|
| `src/knowledge_graph/vector_store.py` | 18 | High - production code |
| `src/knowledge_graph/evaluation.py` | 13 | Medium - used in evaluation scripts |
| `src/knowledge_graph/embedding_trainer.py` | 12 | High - production training |
| `src/knowledge_graph/test_integration.py` | 27 | Low - test file, but should still use logging |
| `src/forecasting/ensemble_predictor.py` | 4 | High - core prediction path (lines 150, 152, 161, 163) |
| `src/forecasting/forecast_engine.py` | 3 | High - core engine (lines 174, 177, 181) |
| `src/forecasting/scenario_generator.py` | 2 | Medium - error fallback paths |
| `src/forecasting/gemini_client.py` | 1 | Medium - error handling |
| `src/bootstrap/checkpoint.py` | 6 | Medium - bootstrap status reporting |
| `src/training/data_processor.py` | 2 | Low - debug output |
| `src/knowledge_graph/test_*.py` (various) | ~15 | Low - test files |

**Total production code print() statements to convert:** ~61 across 11 files.
**Test files:** ~42 additional print() statements (lower priority but should be converted for consistency).

### Existing SQLAlchemy ORM Usage

The file `src/calibration/prediction_store.py` already has a working SQLAlchemy ORM pattern with:
- `declarative_base()` (line 39)
- `Prediction` model with proper column types (lines 42-76)
- `sessionmaker` with context manager (lines 96-145)
- CRUD operations with proper error handling

This is the direct template for the PostgreSQL models, but needs migration from:
- Sync SQLAlchemy -> Async SQLAlchemy
- SQLite engine -> asyncpg PostgreSQL engine
- `Base = declarative_base()` -> `class Base(DeclarativeBase): pass` (modern pattern)

### EnsemblePredictor Integration Point

`src/forecasting/ensemble_predictor.py` line 115-225: The `predict()` method returns `Tuple[EnsemblePrediction, ForecastOutput]`. Phase 9 must add a persistence step after this returns to write to PostgreSQL. The `EnsemblePrediction` dataclass (line 40-52) maps cleanly to `EnsembleInfoDTO`. The `ForecastOutput` model (line 117-128 in `models.py`) maps to `ForecastResponse`.

Key mapping:
- `ForecastOutput.question` -> `ForecastResponse.question`
- `ForecastOutput.prediction` -> `ForecastResponse.prediction`
- `ForecastOutput.probability` -> `ForecastResponse.probability`
- `ForecastOutput.confidence` -> `ForecastResponse.confidence`
- `ForecastOutput.scenario_tree.scenarios` -> `ForecastResponse.scenarios` (needs transformation to `ScenarioDTO` list)
- `EnsemblePrediction.llm_prediction.probability` -> `EnsembleInfoDTO.llm_probability`
- `EnsemblePrediction.tkg_prediction.probability` -> `EnsembleInfoDTO.tkg_probability`

### DTO Contract Spec (from WORLDMONITOR_INTEGRATION.md lines 328-382)

Six DTOs are defined and locked:
1. **ForecastResponse** - 12 fields including nested scenarios, ensemble_info, calibration
2. **ScenarioDTO** - Self-referential (child_scenarios), 8 fields
3. **EvidenceDTO** - 5 fields, links to GDELT events
4. **CalibrationDTO** - 6 fields, per-category calibration metadata
5. **CountryRiskSummary** - 7 fields, aggregate country risk
6. **EnsembleInfoDTO** - 4 fields, LLM/TKG weight breakdown

These are already valid Pydantic V2 models. The only adjustment needed is `ScenarioDTO.child_scenarios: list[ScenarioDTO]` self-reference (requires `model_rebuild()`).

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `declarative_base()` function | `class Base(DeclarativeBase): pass` | SQLAlchemy 2.0 (2023) | Modern mapped_column syntax, better type checking |
| `aioredis` package | `redis[hiredis]` package | redis-py 5.0 (2024) | aioredis merged into redis-py, single package |
| RFC 7807 | RFC 9457 | July 2023 | RFC 9457 obsoletes 7807, same format but clarified semantics |
| jraph for GNN ops | Local NamedTuples + `jax.ops.segment_sum` | jraph archived May 2025 | jraph was a thin wrapper; JAX has all the primitives built-in |
| Sync SQLAlchemy + SQLite | Async SQLAlchemy + asyncpg + PostgreSQL | SQLAlchemy 2.0 (2023) | Full async support, connection pooling, concurrent access |

**Deprecated/outdated:**
- `jraph`: Archived by Google DeepMind 2025-05-21, read-only repository. Must eliminate.
- `aioredis`: Merged into redis-py 5.0. Do not install separately.
- `fastapi-rfc7807`: Last updated 2023, minimal activity. Hand-roll the 30 lines instead.

## Open Questions

1. **PostgreSQL data for `EnsemblePredictor.predict()` persistence**
   - What we know: The predict() method returns `(EnsemblePrediction, ForecastOutput)`. The task needs to wrap this with persistence.
   - What's unclear: Should persistence be added inside `predict()` (side-effect) or in a new method/wrapper (pure function + separate persist call)?
   - Recommendation: Create a new `ForecastService` class that calls `predict()` then persists. Keep `EnsemblePredictor` pure -- it shouldn't know about PostgreSQL. This follows the existing architecture where `PredictionStore` is separate from the predictor.

2. **Existing `PredictionStore` (calibration) migration path**
   - What we know: `src/calibration/prediction_store.py` has a working SQLAlchemy ORM `Prediction` model backed by SQLite.
   - What's unclear: Should this be migrated to PostgreSQL in Phase 9, or left as-is until Phase 13 (calibration)?
   - Recommendation: Migrate it. The PostgreSQL `predictions` table supersedes the SQLite-backed `prediction_store.py`. The existing code becomes the blueprint, then the old SQLite predictions store is removed. This avoids having two different prediction storage mechanisms.

3. **Alembic directory placement**
   - What we know: Alembic needs a directory for migration scripts and an `env.py`.
   - Options: `src/db/migrations/` (close to models) or top-level `alembic/` (project standard)
   - Recommendation: `alembic/` at project root with `alembic.ini` at project root. This is the Alembic convention and avoids nested package import issues. The `env.py` imports models from `src.db.models`.

4. **Mock fixture scope for country endpoints**
   - What we know: `GET /api/v1/forecasts/country/{iso}` and country risk endpoints need mock data.
   - What's unclear: How many countries need mock data? Just Syria (SY), Ukraine (UA), Myanmar (MM)?
   - Recommendation: 3 countries with 2-3 forecasts each is sufficient for frontend contract development. Include one country with zero forecasts to test empty state.

## Sources

### Primary (HIGH confidence)

- Codebase analysis: Direct reading of all `src/` files, `pyproject.toml`, `WORLDMONITOR_INTEGRATION.md`
- `.planning/research/STACK.md` - Prior jraph elimination analysis with `jax.ops.segment_sum` verification
- `.planning/research/SUMMARY.md` - Prior v2.0 domain research
- `src/training/models/regcn_jax.py` - Living template for jraph-free JAX GNN code
- `src/calibration/prediction_store.py` - Existing SQLAlchemy ORM pattern in codebase
- [jax.ops.segment_sum docs](https://docs.jax.dev/en/latest/_autosummary/jax.ops.segment_sum.html) - Verified identical API to `jraph.segment_sum`
- [jraph GraphsTuple API](https://jraph.readthedocs.io/en/latest/api.html) - 7-field NamedTuple definition

### Secondary (MEDIUM confidence)

- [FastAPI SQLAlchemy async patterns](https://dev.to/akarshan/asynchronous-database-sessions-in-fastapi-with-sqlalchemy-1o7e) - Session management best practices
- [Alembic async template](https://alembic.sqlalchemy.org/en/latest/cookbook.html) - `-t async` initialization
- [FastAPI Docker deployment](https://fastapi.tiangolo.com/deployment/docker/) - Official FastAPI Docker guidance
- [RFC 9457](https://datatracker.ietf.org/doc/rfc9457/) - Current Problem Details standard (obsoletes RFC 7807)
- [Python Protocols (PEP 544)](https://peps.python.org/pep-0544/) - `@runtime_checkable` structural typing
- [fastapi-rfc7807](https://github.com/vapor-ware/fastapi-rfc7807) - Evaluated and rejected (hand-roll instead)

### Tertiary (LOW confidence)

- Docker Compose patterns from web search - Community patterns, verified against official docs

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - All libraries verified against official docs and existing codebase compatibility
- Architecture: HIGH - Patterns derived from codebase analysis and FastAPI/SQLAlchemy official documentation
- Pitfalls: HIGH - Based on direct codebase inspection (print() audit, jraph import analysis, session lifecycle patterns)
- Code examples: HIGH - PostgreSQL models derived from existing `prediction_store.py`; jraph elimination derived from existing `regcn_jax.py`

**Research date:** 2026-03-01
**Valid until:** 2026-04-01 (stable stack, no fast-moving dependencies)
