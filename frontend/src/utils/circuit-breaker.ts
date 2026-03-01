/**
 * Circuit breaker for API resilience.
 *
 * Adapted from WorldMonitor's circuit-breaker.ts. Stripped:
 * - IndexedDB persistent cache (overkill for single-API client)
 * - Tauri offline detection (web-only target)
 * - Beacon API offline queue
 *
 * Kept: failure counting, cooldown, TTL cache, stale-while-revalidate,
 * BreakerDataState reporting for panel UI badges.
 */

interface CircuitState {
  failures: number;
  cooldownUntil: number;
  lastError?: string;
}

interface CacheEntry<T> {
  data: T;
  timestamp: number;
}

export type BreakerDataMode = 'live' | 'cached' | 'unavailable';

export interface BreakerDataState {
  mode: BreakerDataMode;
  timestamp: number | null;
}

export interface CircuitBreakerOptions {
  name: string;
  maxFailures?: number;
  cooldownMs?: number;
  cacheTtlMs?: number;
}

const DEFAULT_MAX_FAILURES = 2;
const DEFAULT_COOLDOWN_MS = 5 * 60 * 1000; // 5 minutes
const DEFAULT_CACHE_TTL_MS = 10 * 60 * 1000; // 10 minutes

export class CircuitBreaker<T> {
  private state: CircuitState = { failures: 0, cooldownUntil: 0 };
  private cache: CacheEntry<T> | null = null;
  private readonly name: string;
  private readonly maxFailures: number;
  private readonly cooldownMs: number;
  private readonly cacheTtlMs: number;
  private lastDataState: BreakerDataState = { mode: 'unavailable', timestamp: null };

  constructor(options: CircuitBreakerOptions) {
    this.name = options.name;
    this.maxFailures = options.maxFailures ?? DEFAULT_MAX_FAILURES;
    this.cooldownMs = options.cooldownMs ?? DEFAULT_COOLDOWN_MS;
    this.cacheTtlMs = options.cacheTtlMs ?? DEFAULT_CACHE_TTL_MS;
  }

  /** True when the circuit is open (too many failures, waiting for cooldown). */
  isOnCooldown(): boolean {
    if (Date.now() < this.state.cooldownUntil) {
      return true;
    }
    // Cooldown expired -- reset circuit to half-open
    if (this.state.cooldownUntil > 0) {
      this.state = { failures: 0, cooldownUntil: 0 };
    }
    return false;
  }

  /** Seconds remaining until cooldown expires. */
  getCooldownRemaining(): number {
    return Math.max(0, Math.ceil((this.state.cooldownUntil - Date.now()) / 1000));
  }

  /** Human-readable status string for diagnostics. */
  getStatus(): string {
    if (this.isOnCooldown()) {
      return `temporarily unavailable (retry in ${this.getCooldownRemaining()}s)`;
    }
    return 'ok';
  }

  /** Current data mode for UI badge rendering. */
  getDataState(): BreakerDataState {
    return { ...this.lastDataState };
  }

  /** Return cached data if within TTL, else null. */
  getCached(): T | null {
    if (this.cache && Date.now() - this.cache.timestamp < this.cacheTtlMs) {
      return this.cache.data;
    }
    return null;
  }

  /** Record a successful fetch: reset failures, update cache, set mode=live. */
  recordSuccess(data: T): void {
    this.state = { failures: 0, cooldownUntil: 0 };
    this.cache = { data, timestamp: Date.now() };
    this.lastDataState = { mode: 'live', timestamp: Date.now() };
  }

  /** Record a failed fetch: increment failures, open circuit if threshold reached. */
  recordFailure(error?: string): void {
    this.state.failures++;
    this.state.lastError = error;
    if (this.state.failures >= this.maxFailures) {
      this.state.cooldownUntil = Date.now() + this.cooldownMs;
      console.warn(
        `[${this.name}] On cooldown for ${this.cooldownMs / 1000}s after ${this.state.failures} failures`,
      );
    }
  }

  /**
   * Execute a fetch through the circuit breaker.
   *
   * Flow:
   * 1. If circuit is open (on cooldown), return cached data or fallback
   * 2. If fresh cache exists (within TTL), return it
   * 3. If stale cache exists (outside TTL), return stale + background refresh
   * 4. Otherwise, attempt live fetch -- on success, cache + return;
   *    on failure, record failure + return fallback
   */
  async execute(fetchFn: () => Promise<T>, fallback: T): Promise<T> {
    // Circuit open -- serve cached or fallback
    if (this.isOnCooldown()) {
      console.log(`[${this.name}] Currently unavailable, ${this.getCooldownRemaining()}s remaining`);
      const cachedFallback = this.getCached();
      if (cachedFallback !== null) {
        this.lastDataState = { mode: 'cached', timestamp: this.cache?.timestamp ?? null };
        return cachedFallback;
      }
      // Stale cache (outside TTL but data exists)
      if (this.cache !== null) {
        this.lastDataState = { mode: 'cached', timestamp: this.cache.timestamp };
        return this.cache.data;
      }
      this.lastDataState = { mode: 'unavailable', timestamp: null };
      return fallback;
    }

    // Fresh cache hit
    const cached = this.getCached();
    if (cached !== null) {
      this.lastDataState = { mode: 'cached', timestamp: this.cache?.timestamp ?? null };
      return cached;
    }

    // Stale-while-revalidate: serve stale data immediately, refresh in background
    if (this.cache !== null) {
      this.lastDataState = { mode: 'cached', timestamp: this.cache.timestamp };
      fetchFn()
        .then((result) => this.recordSuccess(result))
        .catch((e: unknown) => {
          console.warn(`[${this.name}] Background refresh failed:`, e);
          this.recordFailure(String(e));
        });
      return this.cache.data;
    }

    // No cache at all -- must fetch synchronously
    try {
      const result = await fetchFn();
      this.recordSuccess(result);
      return result;
    } catch (e: unknown) {
      const msg = String(e);
      console.error(`[${this.name}] Failed:`, msg);
      this.recordFailure(msg);
      this.lastDataState = { mode: 'unavailable', timestamp: null };
      return fallback;
    }
  }
}

/**
 * Factory function for creating circuit breakers with less boilerplate.
 */
export function createCircuitBreaker<T>(options: CircuitBreakerOptions): CircuitBreaker<T> {
  return new CircuitBreaker<T>(options);
}
