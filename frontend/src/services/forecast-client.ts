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
  AdvisoryDTO,
  ArcData,
  ArticleDTO,
  ComparisonPanelResponse,
  ConfirmSubmissionResponse,
  CountryRiskSummary,
  EventDTO,
  ForecastRequestStatus,
  ForecastResponse,
  HealthResponse,
  HexbinData,
  PaginatedResponse,
  ParsedQuestionResponse,
  PolymarketComparisonResponse,
  PolymarketTopResponse,
  RiskDeltaData,
  SearchResponse,
  SnapshotResponse,
  SourceStatusDTO,
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
const EMPTY_REQUESTS: ForecastRequestStatus[] = [];
const FALLBACK_POLYMARKET: PolymarketComparisonResponse = {
  active: [],
  resolved: [],
  summary: { active_count: 0, resolved_count: 0, geopol_avg_brier: null, polymarket_avg_brier: null, geopol_wins: 0 },
  seeking_more_matches: true,
};
const FALLBACK_POLYMARKET_TOP: PolymarketTopResponse = {
  events: [],
  total_geo_markets: 0,
};
const EMPTY_EVENTS: PaginatedResponse<EventDTO> = { items: [], next_cursor: null, has_more: false };
const EMPTY_ARTICLES: PaginatedResponse<ArticleDTO> = { items: [], next_cursor: null, has_more: false };
const EMPTY_SOURCES: SourceStatusDTO[] = [];
const EMPTY_ADVISORIES: AdvisoryDTO[] = [];
const EMPTY_HEXBINS: HexbinData[] = [];
const EMPTY_ARCS: ArcData[] = [];
const EMPTY_DELTAS: RiskDeltaData[] = [];
const EMPTY_COMPARISONS: ComparisonPanelResponse = { comparisons: [], total: 0 };

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

  private readonly eventsBreaker = createCircuitBreaker<unknown>({
    name: 'events',
    maxFailures: 2,
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

  /**
   * GET /forecasts/country/{iso}?cursor=&limit=
   *
   * Does NOT use forecastBreaker.execute() — the shared breaker caches by
   * breaker (not by URL), so getTopForecasts() would poison this response
   * with a ForecastResponse[] instead of PaginatedResponse<ForecastResponse>.
   */
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
    try {
      return await this.fetchJson<PaginatedResponse<ForecastResponse>>(path);
    } catch (e: unknown) {
      console.warn('[forecast-client] getForecastsByCountry failed:', e);
      return EMPTY_PAGINATED;
    }
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

  /**
   * GET /countries/{iso} -- returns null on 404.
   *
   * Does NOT use countryBreaker.execute() — the shared breaker caches by
   * breaker (not by URL), so getCountries() would poison this response
   * with a CountryRiskSummary[] instead of a single CountryRiskSummary.
   */
  async getCountryRisk(iso: string): Promise<CountryRiskSummary | null> {
    const path = `/countries/${encodeURIComponent(iso)}`;
    try {
      return await this.fetchJsonNullable<CountryRiskSummary>(path);
    } catch (e: unknown) {
      console.warn('[forecast-client] getCountryRisk failed:', e);
      return null;
    }
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

  /** GET /calibration/polymarket */
  async getPolymarket(): Promise<PolymarketComparisonResponse> {
    const key = '/calibration/polymarket';
    return this.dedup(key, () =>
      this.forecastBreaker.execute(
        () => this.fetchJson<PolymarketComparisonResponse>(key),
        FALLBACK_POLYMARKET,
      ),
    ) as Promise<PolymarketComparisonResponse>;
  }

  /** GET /calibration/polymarket/top */
  async getPolymarketTop(): Promise<PolymarketTopResponse> {
    const key = '/calibration/polymarket/top';
    return this.dedup(key, () =>
      this.forecastBreaker.execute(
        () => this.fetchJson<PolymarketTopResponse>(key),
        FALLBACK_POLYMARKET_TOP,
      ),
    ) as Promise<PolymarketTopResponse>;
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

  /**
   * GET /forecasts/search -- full-text search with optional country/category filters.
   * NOT deduplicated (search terms change rapidly with each keystroke).
   * NOT circuit-broken (user expects immediate search feedback).
   * Caller passes AbortSignal for race condition prevention via AbortController.
   */
  async search(
    q: string,
    options?: { country?: string; category?: string; limit?: number; signal?: AbortSignal },
  ): Promise<SearchResponse> {
    const params = new URLSearchParams({ q });
    if (options?.country) params.set('country', options.country);
    if (options?.category) params.set('category', options.category);
    if (options?.limit) params.set('limit', String(options.limit));
    const path = `/forecasts/search?${params.toString()}`;
    return this.fetchJson<SearchResponse>(path, { signal: options?.signal });
  }

  // -----------------------------------------------------------------------
  // Submission flow (Phase 14 backend)
  // -----------------------------------------------------------------------

  /**
   * POST /forecasts/submit -- submit a natural language question for LLM parsing.
   * NOT deduplicated (mutation, user expects immediate feedback).
   * NOT circuit-broken (user expects immediate error on failure).
   */
  async submitQuestion(question: string): Promise<ParsedQuestionResponse> {
    return this.fetchJson<ParsedQuestionResponse>('/forecasts/submit', {
      method: 'POST',
      body: JSON.stringify({ question }),
    });
  }

  /**
   * POST /forecasts/submit/{id}/confirm -- confirm a parsed question for processing.
   * NOT deduplicated, NOT circuit-broken (mutation).
   */
  async confirmSubmission(requestId: string): Promise<ConfirmSubmissionResponse> {
    return this.fetchJson<ConfirmSubmissionResponse>(
      `/forecasts/submit/${encodeURIComponent(requestId)}/confirm`,
      { method: 'POST' },
    );
  }

  /**
   * GET /forecasts/requests -- list user's submitted forecast requests.
   * Deduplicated (polled periodically). Uses health breaker (low-priority).
   */
  async getRequests(statusFilter?: string): Promise<ForecastRequestStatus[]> {
    const qs = statusFilter ? `?status_filter=${encodeURIComponent(statusFilter)}` : '';
    const key = `/forecasts/requests${qs}`;
    return this.dedup(key, () =>
      this.healthBreaker.execute(
        () => this.fetchJson<ForecastRequestStatus[]>(key),
        EMPTY_REQUESTS,
      ),
    ) as Promise<ForecastRequestStatus[]>;
  }

  // -----------------------------------------------------------------------
  // Events, articles, sources, advisories (Phase 17)
  // -----------------------------------------------------------------------

  /** GET /events with optional filter parameters and keyset cursor pagination. */
  async getEvents(params?: {
    country?: string;
    start_date?: string;
    end_date?: string;
    cameo_code?: string;
    source?: string;
    limit?: number;
    cursor?: string;
  }): Promise<PaginatedResponse<EventDTO>> {
    const sp = new URLSearchParams();
    if (params) {
      for (const [k, v] of Object.entries(params)) {
        if (v !== undefined) sp.set(k, String(v));
      }
    }
    const qs = sp.toString();
    const key = `/events${qs ? `?${qs}` : ''}`;
    return this.dedup(key, () =>
      this.eventsBreaker.execute(
        () => this.fetchJson<PaginatedResponse<EventDTO>>(key),
        EMPTY_EVENTS as unknown as PaginatedResponse<EventDTO>,
      ),
    ) as Promise<PaginatedResponse<EventDTO>>;
  }

  /**
   * GET /articles?sort=recent&limit=N -- recent articles for NewsFeedPanel.
   * Convenience wrapper around getArticles() with sort=recent default.
   */
  async getRecentArticles(limit?: number): Promise<ArticleDTO[]> {
    const sp = new URLSearchParams({ sort: 'recent' });
    if (limit !== undefined) sp.set('limit', String(limit));
    const key = `/articles?${sp.toString()}`;
    const result = await this.dedup(key, () =>
      this.forecastBreaker.execute(
        () => this.fetchJson<PaginatedResponse<ArticleDTO>>(key),
        EMPTY_ARTICLES as unknown as PaginatedResponse<ArticleDTO>,
      ),
    ) as PaginatedResponse<ArticleDTO>;
    return result.items;
  }

  /** GET /articles with optional filter parameters. */
  async getArticles(params?: {
    country?: string;
    text?: string;
    semantic?: boolean;
    limit?: number;
  }): Promise<PaginatedResponse<ArticleDTO>> {
    const sp = new URLSearchParams();
    if (params) {
      for (const [k, v] of Object.entries(params)) {
        if (v !== undefined) sp.set(k, String(v));
      }
    }
    const qs = sp.toString();
    const key = `/articles${qs ? `?${qs}` : ''}`;
    return this.dedup(key, () =>
      this.forecastBreaker.execute(
        () => this.fetchJson<PaginatedResponse<ArticleDTO>>(key),
        EMPTY_ARTICLES as unknown as PaginatedResponse<ArticleDTO>,
      ),
    ) as Promise<PaginatedResponse<ArticleDTO>>;
  }

  /** GET /sources (public, no auth). Auto-discovered ingestion source health. */
  async getSources(): Promise<SourceStatusDTO[]> {
    const key = '/sources';
    return this.dedup(key, () =>
      this.healthBreaker.execute(
        () => this.fetchJson<SourceStatusDTO[]>(key, { skipAuth: true }),
        EMPTY_SOURCES,
      ),
    ) as Promise<SourceStatusDTO[]>;
  }

  /** GET /advisories with optional country filter. */
  async getAdvisories(country?: string): Promise<AdvisoryDTO[]> {
    const qs = country ? `?country=${encodeURIComponent(country)}` : '';
    const key = `/advisories${qs}`;
    return this.dedup(key, () =>
      this.healthBreaker.execute(
        () => this.fetchJson<AdvisoryDTO[]>(key),
        EMPTY_ADVISORIES,
      ),
    ) as Promise<AdvisoryDTO[]>;
  }

  // -----------------------------------------------------------------------
  // Globe layer data (Phase 24)
  // -----------------------------------------------------------------------

  /** GET /globe/heatmap -- pre-computed H3 hexbin event density. */
  async getHeatmapData(): Promise<HexbinData[]> {
    const key = '/globe/heatmap';
    return this.dedup(key, () =>
      this.eventsBreaker.execute(
        async () => {
          const envelope = await this.fetchJson<{ hexbins: HexbinData[] }>(key);
          return envelope.hexbins ?? [];
        },
        EMPTY_HEXBINS,
      ),
    ) as Promise<HexbinData[]>;
  }

  /** GET /globe/arcs -- top bilateral relationships with sentiment. */
  async getArcData(): Promise<ArcData[]> {
    const key = '/globe/arcs';
    return this.dedup(key, () =>
      this.eventsBreaker.execute(
        async () => {
          const envelope = await this.fetchJson<{ arcs: ArcData[] }>(key);
          return envelope.arcs ?? [];
        },
        EMPTY_ARCS,
      ),
    ) as Promise<ArcData[]>;
  }

  /** GET /globe/deltas -- countries with significant risk score changes. */
  async getRiskDeltas(): Promise<RiskDeltaData[]> {
    const key = '/globe/deltas';
    return this.dedup(key, () =>
      this.eventsBreaker.execute(
        async () => {
          const envelope = await this.fetchJson<{ deltas: RiskDeltaData[] }>(key);
          return envelope.deltas ?? [];
        },
        EMPTY_DELTAS,
      ),
    ) as Promise<RiskDeltaData[]>;
  }

  // -----------------------------------------------------------------------
  // Polymarket comparisons (Phase 18)
  // -----------------------------------------------------------------------

  /** GET /calibration/polymarket/comparisons -- all comparisons for ComparisonPanel. */
  async getComparisons(): Promise<ComparisonPanelResponse> {
    const key = '/calibration/polymarket/comparisons';
    return this.dedup(key, () =>
      this.forecastBreaker.execute(
        () => this.fetchJson<ComparisonPanelResponse>(key),
        EMPTY_COMPARISONS,
      ),
    ) as Promise<ComparisonPanelResponse>;
  }

  /** GET /calibration/polymarket/comparisons/{id}/snapshots -- sparkline data. */
  async getSnapshots(comparisonId: number, limit?: number): Promise<SnapshotResponse> {
    const params = limit ? `?limit=${limit}` : '';
    const path = `/calibration/polymarket/comparisons/${comparisonId}/snapshots${params}`;
    // NOT deduplicated (called on card expand, not polled)
    // NOT circuit-broken (user-initiated, expects immediate feedback)
    try {
      return await this.fetchJson<SnapshotResponse>(path);
    } catch (e: unknown) {
      console.warn('[forecast-client] getSnapshots failed:', e);
      return { comparison_id: comparisonId, snapshots: [], total_available: 0 };
    }
  }

  // -----------------------------------------------------------------------
  // Data state for UI badges
  // -----------------------------------------------------------------------

  /** Return the circuit breaker data state for a given endpoint group. */
  getDataState(endpoint: 'forecast' | 'country' | 'health' | 'events'): BreakerDataState {
    switch (endpoint) {
      case 'forecast':
        return this.forecastBreaker.getDataState();
      case 'country':
        return this.countryBreaker.getDataState();
      case 'health':
        return this.healthBreaker.getDataState();
      case 'events':
        return this.eventsBreaker.getDataState();
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
