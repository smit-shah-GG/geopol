# Phase 12 Plan 02: Service Layer (CircuitBreaker + ForecastServiceClient) Summary

**One-liner:** CircuitBreaker with failure counting/cooldown/TTL cache + ForecastServiceClient with typed methods for all 7 API endpoints, in-flight deduplication, and data state reporting.

## What Was Done

### Task 1: CircuitBreaker class
- Adapted from WorldMonitor's `circuit-breaker.ts` (279 lines -> 182 lines)
- Stripped: IndexedDB persistent cache, Tauri offline detection, Beacon API offline queue
- Kept: CircuitState, CacheEntry, BreakerDataMode, BreakerDataState, execute() with stale-while-revalidate
- Added: `createCircuitBreaker<T>()` factory function
- Exports: `CircuitBreaker`, `createCircuitBreaker`, `BreakerDataMode`, `BreakerDataState`, `CircuitBreakerOptions`
- Commit: `635e479`

### Task 2: ForecastServiceClient
- 7 public methods matching all FastAPI endpoints:
  1. `getTopForecasts(limit?)` -- GET /forecasts/top
  2. `getForecastsByCountry(iso, cursor?, limit?)` -- GET /forecasts/country/{iso}
  3. `getForecastById(id)` -- GET /forecasts/{id} (404 -> null)
  4. `getCountries()` -- GET /countries
  5. `getCountryRisk(iso)` -- GET /countries/{iso} (404 -> null)
  6. `getHealth()` -- GET /health (no API key)
  7. `createForecast(question, countryIso, horizonDays?)` -- POST /forecasts
- Circuit breakers per endpoint group: forecast (2 failures/30s/60s TTL), country (2/30s/120s), health (3/15s/30s)
- In-flight deduplication via `Map<string, Promise<unknown>>`
- POST bypasses dedup + circuit breaker (mutations fire every time)
- `getDataState(endpoint)` for UI badge rendering
- Singleton export: `forecastClient`
- Fulfills FE-01 DataLoaderManager requirement
- Commit: `6c5e201`

## Verification

- `tsc --noEmit` passes (strict mode, noUncheckedIndexedAccess)
- All 5 circuit-breaker exports present
- All 7 API methods present with correct TypeScript signatures
- No IndexedDB/Tauri/offline queue code in production
- ForecastServiceClient + singleton exported

## Deviations from Plan

None -- plan executed exactly as written.

## Decisions Made

| Decision | Rationale | Ref |
|----------|-----------|-----|
| POST createForecast bypasses dedup + circuit breaker | Mutations must fire every time; user expects immediate feedback on failure, not stale cache | 12-02 Task 2 |
| CircuitBreaker uses `unknown` generic for client breakers | Per-group breaker wraps heterogeneous return types; casting at call site is type-safe | 12-02 Task 2 |
| ApiError class for non-2xx responses | Structured error with status/statusText/url enables 404 -> null mapping and future error UI | 12-02 Task 2 |

## Files

### Created
- `frontend/src/utils/circuit-breaker.ts` (182 lines)
- `frontend/src/services/forecast-client.ts` (281 lines)

### Modified
None.

## Metrics

- **Tasks:** 2/2
- **Duration:** ~2 minutes
- **Commits:** 2 (task) + 1 (metadata)
