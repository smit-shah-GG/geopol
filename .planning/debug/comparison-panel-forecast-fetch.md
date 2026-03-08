---
status: diagnosed
trigger: "ComparisonPanel lazy forecast fetch shows 'failed to load forecast' despite API returning 200"
created: 2026-03-09T00:00:00Z
updated: 2026-03-09T00:00:00Z
---

## Current Focus

hypothesis: Circuit breaker caches `null` as a successful result, then returns it on subsequent calls; but the primary issue is the breaker's shared cache poisoning getForecastById with results from getTopForecasts
test: Trace the full data flow from ComparisonPanel -> forecastClient.getForecastById -> forecastBreaker.execute
expecting: The circuit breaker's cache or fallback mechanism returns null, which ComparisonPanel treats as "Forecast not found"
next_action: Document root cause

## Symptoms

expected: Expanding a ComparisonPanel entry should lazy-fetch the full ForecastResponse and render it inline
actual: Shows "Failed to load forecast details" (catch block) or "Forecast not found" (null check) on every expand
errors: Console shows `[ComparisonPanel] Failed to load forecast:` with the geopol_prediction_id
reproduction: Click any ComparisonPanel entry to expand
started: Since ComparisonPanel was built

## Eliminated

- hypothesis: Type mismatch between geopol_prediction_id and forecast API parameter (numeric vs string)
  evidence: Both sides use string UUIDs. ComparisonPanelItem.geopol_prediction_id is `string` in both backend Pydantic model and frontend TS type. Prediction.id is String(36). The GET /forecasts/{forecast_id} endpoint accepts a string path parameter.
  timestamp: 2026-03-09

- hypothesis: The API endpoint returns 404 for valid prediction IDs
  evidence: User reports server logs show only 200 responses, no 500s or errors. The endpoint is found and responding.
  timestamp: 2026-03-09

## Evidence

- timestamp: 2026-03-09
  checked: ComparisonPanel.renderExpanded() at line 264
  found: Calls `forecastClient.getForecastById(comp.geopol_prediction_id)` which is a string UUID -- correct
  implication: The ID being passed is correct

- timestamp: 2026-03-09
  checked: ForecastServiceClient.getForecastById() at line 195-203
  found: Uses `forecastBreaker.execute()` with fallback `null as unknown as ForecastResponse`. The inner function is `fetchJsonNullable` which returns `ForecastResponse | null`. On 404, returns null. On success, returns the parsed JSON.
  implication: The forecastBreaker is shared across getTopForecasts, getForecastById, getComparisons, getPolymarket, getPolymarketTop, getRecentArticles, and getArticles

- timestamp: 2026-03-09
  checked: CircuitBreaker.execute() at lines 134-183
  found: The circuit breaker has a SINGLE cache slot (`this.cache: CacheEntry<T> | null`). It is NOT keyed by URL or request path. `recordSuccess(data)` overwrites the single cache entry with whatever data was last fetched. `getCached()` returns the single cache entry if within TTL.
  implication: **CRITICAL** -- Any successful call through the forecastBreaker poisons the cache for ALL other calls using the same breaker

- timestamp: 2026-03-09
  checked: ForecastServiceClient.dedup() at lines 543-553
  found: Deduplication IS keyed by path string. But dedup only prevents duplicate in-flight requests -- it does NOT affect the circuit breaker's cache. Once the promise resolves, `dedup` removes the key. The circuit breaker's `execute()` method is what caches.
  implication: Dedup doesn't save us -- the circuit breaker's shared cache is the problem

- timestamp: 2026-03-09
  checked: Typical call sequence on dashboard load
  found: Dashboard loads -> getTopForecasts() fires -> forecastBreaker.execute() succeeds with ForecastResponse[] -> recordSuccess(ForecastResponse[]) -> cache = {data: ForecastResponse[], timestamp: now}. Then user expands ComparisonPanel entry -> getForecastById() fires -> forecastBreaker.execute() -> getCached() returns the ForecastResponse[] array (still within TTL) -> this array is returned as if it were a ForecastResponse|null.
  implication: getForecastById() returns a ForecastResponse[] (an array of forecasts from getTopForecasts) instead of a single ForecastResponse. The ComparisonPanel tries to use this as a ForecastResponse object, which either crashes in buildExpandedContent() or gets treated as truthy-but-wrong-shape.

- timestamp: 2026-03-09
  checked: ComparisonPanel.renderExpanded() lines 268-274
  found: `if (!forecast)` -- an array is truthy even if empty, so this check passes. Then `this.forecastCache.set(comp.geopol_prediction_id, forecast)` caches the wrong-type object. Then `this.fillExpandedSection(section, forecast, comp)` tries to use it as ForecastResponse. `buildExpandedContent(forecast)` receives an array where it expects an object with .scenarios, .ensemble_info, etc. This crashes, triggering the catch block at line 279-287 which shows "Failed to load forecast details".
  implication: The catch block catches the TypeError from buildExpandedContent and displays the error message

- timestamp: 2026-03-09
  checked: Alternative scenario -- fresh page, no prior forecastBreaker calls
  found: If getForecastById is the FIRST call through forecastBreaker (no cache), it works correctly: fetchJsonNullable fetches the real data, recordSuccess caches the single ForecastResponse, returns it. But on any SUBSEQUENT getForecastById call for a DIFFERENT prediction ID within the cache TTL (60s), the breaker returns the FIRST prediction's cached data (wrong prediction). This is a secondary bug.
  implication: Even in the best case (no prior getTopForecasts call), the second expand returns wrong data

## Resolution

root_cause: |
  The `forecastBreaker` circuit breaker has a **single cache slot** (not keyed by URL/request).
  Seven different endpoint methods share this one breaker: getTopForecasts, getForecastById,
  getComparisons, getPolymarket, getPolymarketTop, getRecentArticles, and getArticles.

  When the dashboard loads, getTopForecasts() fires first and caches a `ForecastResponse[]`
  array in the breaker's single cache slot. When getForecastById() subsequently fires (on
  ComparisonPanel expand), the breaker's `execute()` method hits `getCached()` which returns
  the ForecastResponse[] array (still within the 60-second TTL). This wrong-type data is
  returned to ComparisonPanel, which tries to render it as a single ForecastResponse object.
  The property access on the array (.scenarios, .ensemble_info, etc.) throws a TypeError,
  caught by the catch block, which shows "Failed to load forecast details".

  This is the same class of bug already documented in the code comments at lines 173-176
  and 218-221 of forecast-client.ts: getForecastsByCountry and getCountryRisk deliberately
  bypass their respective breakers with a comment explaining the poisoning problem:
  "the shared breaker caches by breaker (not by URL), so getTopForecasts() would poison
  this response with a ForecastResponse[] instead of PaginatedResponse<ForecastResponse>."

  getForecastById was NOT given this same treatment -- it still goes through
  forecastBreaker.execute(), which is the bug.

fix: |
  Change getForecastById() to bypass forecastBreaker.execute(), matching the pattern
  already used by getForecastsByCountry() and getCountryRisk(). Replace the breaker-wrapped
  call with a direct fetchJsonNullable() call with a try/catch that returns null on failure:

  ```typescript
  async getForecastById(id: string): Promise<ForecastResponse | null> {
    const path = `/forecasts/${encodeURIComponent(id)}`;
    try {
      return await this.fetchJsonNullable<ForecastResponse>(path);
    } catch (e: unknown) {
      console.warn('[forecast-client] getForecastById failed:', e);
      return null;
    }
  }
  ```

verification:
files_changed: []
