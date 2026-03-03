# Plan 17-02 Summary: Backend API Routes + Ingestion Daemons

**Phase:** 17-live-data-feeds-country-depth
**Plan:** 02
**Status:** Complete
**Duration:** ~12min (7min agent + 5min manual Task 4 recovery)

## Deliverables

| File | What it provides |
|------|-----------------|
| `src/ingest/advisory_store.py` | `AdvisoryStore` shared in-memory cache (reader: route, writer: poller) |
| `src/api/routes/v1/events.py` | `GET /events` with 9-parameter filter + keyset cursor pagination |
| `src/api/routes/v1/articles.py` | `GET /articles` with keyword (collection.get) + semantic (collection.query) dual-mode ChromaDB |
| `src/api/routes/v1/sources.py` | `GET /sources` auto-discovery from IngestRun table |
| `src/api/routes/v1/advisories.py` | `GET /advisories` from AdvisoryStore cache with country filter |
| `src/api/routes/v1/router.py` | All 4 new routes wired with correct prefixes and tags |
| `src/ingest/acled_poller.py` | `ACLEDPoller` daily daemon: API key+email auth, iso3-to-alpha2 mapping, unified Event schema |
| `src/ingest/advisory_poller.py` | `AdvisoryPoller` daily daemon: StateDeptClient + FCDOClient, rate-limited, country-name-to-ISO mapping |

## Commits

| Hash | Description |
|------|-------------|
| `2c7ccbc` | feat(17-02): API routes for events, articles, sources, advisories + AdvisoryStore + router wiring |
| `cc3d099` | feat(17-02): ACLED conflict event poller daemon |
| `384c32e` | feat(17-02): government advisory poller daemon |

## Key Decisions

- AdvisoryStore uses classmethod-based in-memory cache pattern (no Pydantic coupling) -- import-safe for both route and poller
- Events route uses `asyncio.to_thread()` for sync SQLite calls inside async FastAPI handlers
- Sources route queries IngestRun table per daemon_type with known-types fallback ("never run" for missing)
- Articles route handles ChromaDB unavailability gracefully (empty list, no 500)
- ACLED uses key+email query params (NOT OAuth2 -- per ACLED API docs)
- ACLED iso3 field mapped via static 130-country ISO3_TO_ISO2 dict
- FCDO per-country fetches bounded by asyncio.Semaphore(5) with 0.3s delay
- EU EEAS deliberately excluded (no structured API) with code comment

## Deviations

- Task 4 (advisory poller) was completed manually after content filter blocked the executor agent
- Pre-existing test failure: `test_default_is_regcn` expects "regcn" but Settings default is "tirgn" since Phase 11

## Verification

- All 8 new modules importable without errors
- No circular imports between advisory route and advisory poller (shared advisory_store)
- 264 tests passed, 9 skipped, 1 pre-existing failure (unrelated to Phase 17)
