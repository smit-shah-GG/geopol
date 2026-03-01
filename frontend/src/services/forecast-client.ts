/**
 * ForecastServiceClient -- typed, resilient access to the Geopol FastAPI backend.
 *
 * Architecture:
 * - Circuit breakers per endpoint group (forecast, country, health)
 * - In-flight request deduplication (same URL = one fetch)
 * - BreakerDataState reporting per group for UI badges
 *
 * This client + CircuitBreaker architecture fulfills the FE-01 DataLoaderManager
 * requirement. A separate DataLoaderManager class would be redundant wrapping.
 */

import type {
  CountryRiskSummary,
  ForecastResponse,
  HealthResponse,
  PaginatedResponse,
} from '@/types/api.ts';
import {
  type BreakerDataState,
  createCircuitBreaker,
} from '@/utils/circuit-breaker.ts';

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

const API_BASE: string =
  (import.meta.env.VITE_API_BASE as string | undefined) ?? '/api/v1';
const API_KEY: string =
  (import.meta.env.VITE_API_KEY as string | undefined) ?? '';

// ---------------------------------------------------------------------------
// Error types
// ---------------------------------------------------------------------------

class ApiError extends Error {
  constructor(
    public readonly status: number,
    public readonly statusText: string,
    public readonly url: string,
  ) {
    super(`API ${status} ${statusText}: ${url}`);
    this.name = 'ApiError';
  }
}

// ---------------------------------------------------------------------------
// Default fallbacks for circuit breaker (returned when API + cache both fail)
// ---------------------------------------------------------------------------

const EMPTY_FORECASTS: ForecastResponse[] = [];
const EMPTY_COUNTRIES: CountryRiskSummary[] = [];
const EMPTY_PAGINATED: PaginatedResponse<ForecastResponse> = {
  items: [],
  next_cursor: null,
  has_more: false,
};
const FALLBACK_HEALTH: HealthResponse = {
  status: 'unhealthy',
  subsystems: [],
  timestamp: new Date().toISOString(),
  version: 'unknown',
};

// ---------------------------------------------------------------------------
// ForecastServiceClient
// ---------------------------------------------------------------------------

export class ForecastServiceClient {
  // Circuit breakers per endpoint group
  private readonly forecastBreaker = createCircuitBreaker<unknown>({
    name: 'forecast',
    maxFailures: 2,
    cooldownMs: 30_000,
    cacheTtlMs: 60_000,
  });

  private readonly countryBreaker = createCircuitBreaker<unknown>({
    name: 'country',
    maxFailures: 2,
    cooldownMs: 30_000,
    cacheTtlMs: 120_000,
  });

  private readonly healthBreaker = createCircuitBreaker<unknown>({
    name: 'health',
    maxFailures: 3,
    cooldownMs: 15_000,
    cacheTtlMs: 30_000,
  });

  // In-flight deduplication map
  private readonly inFlight = new Map<string, Promise<unknown>>();

  // -----------------------------------------------------------------------
  // Public API methods
  // -----------------------------------------------------------------------

  /** GET /forecasts/top?limit=N */
  async getTopForecasts(limit?: number): Promise<ForecastResponse[]> {
    const params = limit !== undefined ? `?limit=${limit}` : '';
    const key = `/forecasts/top${params}`;
    return this.dedup(key, () =>
      this.forecastBreaker.execute(
        () => this.fetchJson<ForecastResponse[]>(key),
        EMPTY_FORECASTS,
      ),
    ) as Promise<ForecastResponse[]>;
  }

  /** GET /forecasts/country/{iso}?cursor=&limit= */
  async getForecastsByCountry(
    iso: string,
    cursor?: string,
    limit?: number,
  ): Promise<PaginatedResponse<ForecastResponse>> {
    const searchParams = new URLSearchParams();
    if (cursor !== undefined) searchParams.set('cursor', cursor);
    if (limit !== undefined) searchParams.set('limit', String(limit));
    const qs = searchParams.toString();
    const path = `/forecasts/country/${encodeURIComponent(iso)}${qs ? `?${qs}` : ''}`;
    return this.dedup(path, () =>
      this.forecastBreaker.execute(
        () => this.fetchJson<PaginatedResponse<ForecastResponse>>(path),
        EMPTY_PAGINATED,
      ),
    ) as Promise<PaginatedResponse<ForecastResponse>>;
  }

  /** GET /forecasts/{id} -- returns null on 404 */
  async getForecastById(id: string): Promise<ForecastResponse | null> {
    const path = `/forecasts/${encodeURIComponent(id)}`;
    return this.dedup(path, () =>
      this.forecastBreaker.execute(
        () => this.fetchJsonNullable<ForecastResponse>(path),
        null as unknown as ForecastResponse,
      ),
    ) as Promise<ForecastResponse | null>;
  }

  /** GET /countries */
  async getCountries(): Promise<CountryRiskSummary[]> {
    const key = '/countries';
    return this.dedup(key, () =>
      this.countryBreaker.execute(
        () => this.fetchJson<CountryRiskSummary[]>(key),
        EMPTY_COUNTRIES,
      ),
    ) as Promise<CountryRiskSummary[]>;
  }

  /** GET /countries/{iso} -- returns null on 404 */
  async getCountryRisk(iso: string): Promise<CountryRiskSummary | null> {
    const path = `/countries/${encodeURIComponent(iso)}`;
    return this.dedup(path, () =>
      this.countryBreaker.execute(
        () => this.fetchJsonNullable<CountryRiskSummary>(path),
        null as unknown as CountryRiskSummary,
      ),
    ) as Promise<CountryRiskSummary | null>;
  }

  /** GET /health (no API key required) */
  async getHealth(): Promise<HealthResponse> {
    const key = '/health';
    return this.dedup(key, () =>
      this.healthBreaker.execute(
        () => this.fetchJson<HealthResponse>(key, { skipAuth: true }),
        FALLBACK_HEALTH,
      ),
    ) as Promise<HealthResponse>;
  }

  /**
   * POST /forecasts -- create a new forecast.
   * Not deduplicated (mutations must fire every time).
   * Not circuit-broken (user expects immediate feedback on POST failure).
   */
  async createForecast(
    question: string,
    countryIso: string,
    horizonDays?: number,
  ): Promise<ForecastResponse> {
    const body: Record<string, unknown> = {
      question,
      country_iso: countryIso,
    };
    if (horizonDays !== undefined) {
      body.horizon_days = horizonDays;
    }
    return this.fetchJson<ForecastResponse>('/forecasts', {
      method: 'POST',
      body: JSON.stringify(body),
    });
  }

  // -----------------------------------------------------------------------
  // Data state for UI badges
  // -----------------------------------------------------------------------

  /** Return the circuit breaker data state for a given endpoint group. */
  getDataState(endpoint: 'forecast' | 'country' | 'health'): BreakerDataState {
    switch (endpoint) {
      case 'forecast':
        return this.forecastBreaker.getDataState();
      case 'country':
        return this.countryBreaker.getDataState();
      case 'health':
        return this.healthBreaker.getDataState();
    }
  }

  // -----------------------------------------------------------------------
  // Private helpers
  // -----------------------------------------------------------------------

  /**
   * In-flight deduplication. If a request for `key` is already pending,
   * return the same promise instead of firing a duplicate fetch.
   */
  private dedup<R>(key: string, fn: () => Promise<R>): Promise<R> {
    const existing = this.inFlight.get(key);
    if (existing) {
      return existing as Promise<R>;
    }
    const promise = fn().finally(() => {
      this.inFlight.delete(key);
    });
    this.inFlight.set(key, promise);
    return promise;
  }

  /**
   * Fetch JSON from the API. Adds auth header unless skipAuth is set.
   * Throws ApiError on non-2xx responses.
   */
  private async fetchJson<R>(
    path: string,
    options?: RequestInit & { skipAuth?: boolean },
  ): Promise<R> {
    const url = `${API_BASE}${path}`;
    const headers: Record<string, string> = {};

    if (!options?.skipAuth && API_KEY) {
      headers['X-API-Key'] = API_KEY;
    }

    if (options?.method === 'POST') {
      headers['Content-Type'] = 'application/json';
    }

    const response = await fetch(url, {
      ...options,
      headers: { ...headers, ...(options?.headers as Record<string, string> | undefined) },
    });

    if (!response.ok) {
      throw new ApiError(response.status, response.statusText, url);
    }

    return (await response.json()) as R;
  }

  /**
   * Fetch JSON with 404 -> null mapping. All other errors propagate.
   */
  private async fetchJsonNullable<R>(path: string): Promise<R | null> {
    try {
      return await this.fetchJson<R>(path);
    } catch (e: unknown) {
      if (e instanceof ApiError && e.status === 404) {
        return null;
      }
      throw e;
    }
  }
}

/** Singleton instance for panel consumption. */
export const forecastClient = new ForecastServiceClient();
