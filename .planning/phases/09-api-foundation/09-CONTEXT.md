# Phase 9: API Foundation & Infrastructure - Context

**Gathered:** 2026-02-27
**Status:** Ready for planning

<domain>
## Phase Boundary

Persistence layer (PostgreSQL + SQLite coexistence), headless FastAPI server with mock fixtures establishing the DTO contract, structured logging, jraph elimination, and TKGModelProtocol definition. This is the foundation that enables Phases 10, 11, and 12 to develop in parallel — the API contract (DTOs + mock endpoints) is the critical output.

Does NOT include: real forecast generation (Phase 10), TiRGN implementation (Phase 11), or frontend code (Phase 12). Mock data only.

</domain>

<decisions>
## Implementation Decisions

### Database Architecture
- **Dual database**: PostgreSQL for forecast-related tables (predictions, outcome_records, calibration_weights, ingest_runs, api_keys). SQLite retained for GDELT event store and partition index.
- **Separate connection managers**: Two explicit connection pools — one for PostgreSQL, one for SQLite. Calling code knows which it's talking to. No unified repository abstraction.
- **SQLAlchemy ORM (declarative)**: Define Prediction, OutcomeRecord, CalibrationWeight, etc. as SQLAlchemy ORM classes. Alembic for version-controlled migrations with autogeneration from model changes.
- **Full async**: asyncpg as native async PostgreSQL driver. SQLAlchemy async session. aioredis for Redis. FastAPI handlers are all async def.
- **Local Postgres via Docker**: docker-compose with postgres:16 for development. Production hosting decided later (Railway/Fly.io/VPS deferred to Phase 10+).

### API Contract & Mock Fixtures
- **Mock strategy: both static files and factory functions**: Hand-crafted JSON fixture files for key scenarios (Syria, Ukraine, Myanmar) as golden samples. Plus a factory function for generating N random forecasts with valid probability distributions and plausible timelines.
- **Cursor-based pagination**: List endpoints (GET /forecasts/country/{iso}, GET /forecasts/top) use next_cursor token in response, not offset/limit.
- **Fully nested DTOs**: GET /forecasts/{id} returns the complete tree — scenarios with child_scenarios, evidence sources, calibration, ensemble info — all in one response. No separate sub-resource endpoints.
- **URL path versioning**: /api/v1/ prefix. When v2 comes, v1 keeps working at the old path.

### Deployment & Process Model
- **Docker from Phase 9**: Dockerfile for API server + docker-compose.yml (postgres, redis, api) shipped in this phase. `docker-compose up` is the dev command from day one. Frontend (Phase 12) adds its own service later. Ingest daemon (Phase 10) adds its container later.
- **Production deploy target deferred**: Phase 9 sets up Docker. Actual deploy target (Railway vs Fly.io vs VPS) decided when real data flows in Phase 10+.

### Auth & Security Posture
- **Per-client API keys**: Each client gets a unique key stored in an `api_keys` table. Individual revocation. Audit trail via key identity in request logs.
- **CORS: permissive in dev, strict in prod**: CORS allows all origins when ENVIRONMENT=development. Strict allowlist (configured via env var) in production.
- **RFC 7807 Problem Details**: Error responses follow the standard format: {type, title, status, detail, instance}. Machine-parseable, well-documented.
- **Health endpoint: public, no auth**: GET /api/v1/health returns full subsystem status without authentication. Standard for load balancers, uptime monitors, health probes.

### Claude's Discretion
- Connection pool sizing for asyncpg
- Redis key prefix naming conventions
- Exact Alembic directory structure
- Mock fixture data content (as long as it's geopolitically plausible and structurally valid)
- Structured logging format and handler configuration
- jraph elimination implementation details (NamedTuple design for GraphsTuple replacement)
- TKGModelProtocol method signatures

</decisions>

<specifics>
## Specific Ideas

- DTO contract spec is already defined in WORLDMONITOR_INTEGRATION.md lines 328-382 (ForecastResponse, ScenarioDTO, CalibrationDTO, CountryRiskSummary, EnsembleInfoDTO, EvidenceDTO) — use these as the starting point
- docker-compose should include postgres:16, redis:7-alpine, and the API service
- Mock fixtures should feel real enough that a frontend developer building against them wouldn't notice they're fake

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 09-api-foundation*
*Context gathered: 2026-02-27*
