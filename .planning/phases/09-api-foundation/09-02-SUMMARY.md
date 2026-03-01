---
phase: 09-api-foundation
plan: 02
subsystem: api
tags: [pydantic, dto, fastapi, mock-fixtures, openapi]

# Dependency graph
requires:
  - phase: 09-01
    provides: FastAPI/Pydantic dependencies and settings infrastructure
provides:
  - ForecastResponse, ScenarioDTO, EvidenceDTO, CalibrationDTO, EnsembleInfoDTO DTOs
  - CountryRiskSummary, HealthResponse, SubsystemStatus DTOs
  - ProblemDetail (RFC 9457), PaginatedResponse[T] generic, cursor encode/decode
  - Hand-crafted JSON fixtures (Syria, Ukraine, Myanmar) with realistic geopolitical content
  - Factory functions for generating arbitrary mock forecasts and country risk summaries
affects: [10-forecast-pipeline, 12-frontend, 09-05-routes]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Pydantic V2 ConfigDict(from_attributes=True) on all DTOs for ORM compatibility"
    - "ScenarioDTO.model_rebuild() for recursive self-reference resolution"
    - "Base64url cursor encoding for keyset pagination"
    - "Hand-crafted golden-sample fixtures + random factory dual strategy"

key-files:
  created:
    - src/api/fixtures/__init__.py
    - src/api/fixtures/factory.py
    - src/api/fixtures/scenarios/syria.json
    - src/api/fixtures/scenarios/ukraine.json
    - src/api/fixtures/scenarios/myanmar.json
  modified: []

key-decisions:
  - "Schema files already existed from prior 09-03 execution — verified identical to spec, no re-commit needed"
  - "Syria fixture: 3 scenarios with 2 recursive child_scenarios on primary (military escalation) branch"
  - "Ukraine fixture: 3 scenarios with diplomatic/military/escalation split, child scenario modeling negotiation collapse"
  - "Myanmar fixture: 2 scenarios only (testing smaller response) — junta collapse vs Chinese diplomatic intervention"
  - "Factory uses weighted random scenario generation with plausible probability distributions"

patterns-established:
  - "Fixture validation: all JSON fixtures are validated through ForecastResponse.model_validate() on load — structural errors fail at load time, not at API response time"
  - "Factory output validation: create_mock_forecast() validates against ForecastResponse before returning"

# Metrics
duration: 6min
completed: 2026-03-01
---

# Phase 9 Plan 02: DTO Schemas and Mock Fixtures Summary

**Pydantic V2 DTO contract (6 DTOs + generics) with 3 country-specific mock fixtures (SY, UA, MM) and random forecast factory**

## Performance

- **Duration:** 6 min
- **Started:** 2026-03-01T09:15:01Z
- **Completed:** 2026-03-01T09:21:18Z
- **Tasks:** 2
- **Files created:** 5 (fixtures + factory; schemas pre-existed from 09-03)

## Accomplishments

- All 6 forecast-related DTOs (ForecastResponse, ScenarioDTO, EvidenceDTO, CalibrationDTO, EnsembleInfoDTO, CountryRiskSummary) validated and generating correct OpenAPI schema
- HealthResponse with 8 canonical subsystem status entries (database, redis, gdelt_store, graph_partitions, tkg_model, last_ingest, last_prediction, api_budget)
- ProblemDetail (RFC 9457), PaginatedResponse[T] generic, base64url cursor encode/decode with round-trip verification
- Three hand-crafted fixtures with realistic geopolitical content: Syria (Turkish-backed offensive, 3 scenarios, recursive children), Ukraine (ceasefire negotiations, 3 scenarios), Myanmar (junta stability, 2 scenarios)
- Factory function generating arbitrary valid ForecastResponse objects with plausible probability distributions

## Task Commits

Each task was committed atomically:

1. **Task 1: Pydantic V2 DTO schemas** - Already committed in `5f57d12` (prior 09-03 execution produced identical files)
2. **Task 2: Mock fixtures and factory** - `48ae0aa` (feat)

## Files Created/Modified

- `src/api/__init__.py` - API package root (pre-existed)
- `src/api/schemas/__init__.py` - Re-exports all public DTOs (pre-existed)
- `src/api/schemas/forecast.py` - ForecastResponse, ScenarioDTO, EvidenceDTO, CalibrationDTO, EnsembleInfoDTO (pre-existed)
- `src/api/schemas/country.py` - CountryRiskSummary DTO (pre-existed)
- `src/api/schemas/health.py` - HealthResponse, SubsystemStatus, SUBSYSTEM_NAMES (pre-existed)
- `src/api/schemas/common.py` - ProblemDetail, PaginatedResponse[T], cursor encode/decode (pre-existed)
- `src/api/fixtures/__init__.py` - Fixture public API exports
- `src/api/fixtures/factory.py` - load_fixture, load_all_fixtures, create_mock_forecast, create_mock_country_risk, get_empty_country_response
- `src/api/fixtures/scenarios/syria.json` - 159 lines, 3 scenarios with recursive branching, GDELT/TKG/RAG evidence
- `src/api/fixtures/scenarios/ukraine.json` - 137 lines, 3 scenarios (diplomatic/military/escalation), ceasefire negotiation focus
- `src/api/fixtures/scenarios/myanmar.json` - 86 lines, 2 scenarios (junta collapse vs Chinese intervention)

## Decisions Made

- Schema files were already committed identically by a prior 09-03 plan execution. Verified content matches spec exactly — no re-commit needed, no divergence.
- Syria fixture uses CAMEO 190 (military force) and CAMEO 036 (negotiate) event codes with realistic frequency multipliers from GDELT baseline patterns.
- Factory generates probabilities in the 0.15-0.85 range to avoid unrealistic extreme predictions, with scenario probabilities distributed via decreasing-share algorithm.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Task 1 schema files already committed**
- **Found during:** Task 1 (DTO schemas)
- **Issue:** All 6 schema files plus `src/api/__init__.py` were already tracked by git (committed in `5f57d12` from the 09-03 plan execution). The Write tool produced identical content.
- **Fix:** Verified content matches plan spec exactly via diff. No separate commit needed — prior execution already satisfied Task 1 requirements.
- **Files modified:** None (files were identical)
- **Verification:** All DTOs import and validate correctly, OpenAPI schema generates without errors
- **Committed in:** N/A (pre-existing)

---

**Total deviations:** 1 auto-fixed (1 blocking — already-committed files)
**Impact on plan:** No scope creep. Task 1 deliverables were already in place from prior execution. Task 2 (fixtures/factory) was the actual new work.

## Issues Encountered

None — fixtures loaded and validated on first attempt, factory stress-tested with 10 random generations.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- DTO contract fully established — Phases 10 (backend pipeline) and 12 (frontend) can develop against these schemas
- Mock fixtures provide realistic data for frontend rendering before real forecasts exist
- Factory enables generating arbitrary test data for API endpoint integration tests
- Route implementation (Plan 05) can wire these DTOs to FastAPI endpoints

---
*Phase: 09-api-foundation*
*Completed: 2026-03-01*
