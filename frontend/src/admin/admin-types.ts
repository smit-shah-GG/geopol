/**
 * TypeScript interfaces mirroring the backend admin Pydantic DTOs.
 *
 * These map 1:1 to the schemas in src/api/schemas/admin.py.
 * Keep in sync manually -- the admin surface is small enough that
 * codegen would be overkill.
 */

/** Status summary for a background daemon / scheduled job. */
export interface ProcessInfo {
  name: string;
  daemon_type: string;
  status: string; // running | scheduled | paused | success | failed | unknown
  last_run: string | null;
  next_run: string | null;
  success_count: number;
  fail_count: number;
  last_duration: number | null; // seconds
  last_error: string | null;
  consecutive_failures: number;
  paused: boolean;
}

/** Single runtime-adjustable configuration value. */
export interface ConfigEntry {
  key: string;
  value: unknown;
  type: string; // int | float | str | bool | list
  editable: boolean;
  dangerous: boolean;
  description: string;
}

/** Structured log entry from the in-memory ring buffer. */
export interface LogEntry {
  timestamp: string;
  severity: string;
  module: string;
  message: string;
}

/** Per-source health and enable/disable state. */
export interface SourceInfo {
  name: string;
  daemon_type: string;
  enabled: boolean;
  healthy: boolean;
  last_run: string | null;
  events_count: number;
  tier: string | null;
}

/** Admin-facing RSS feed with health metrics (mirrors backend FeedInfo). */
export interface FeedInfo {
  id: number;
  name: string;
  url: string;
  tier: number;
  category: string;
  lang: string;
  enabled: boolean;
  last_poll_at: string | null;
  last_error: string | null;
  error_count: number;
  articles_24h: number;
  articles_total: number;
  avg_articles_per_poll: number;
  created_at: string;
}

/** Payload for POST /admin/feeds. */
export interface AddFeedRequest {
  name: string;
  url: string;
  tier: 1 | 2;
  category?: string;
  lang?: string;
}

/** Payload for PUT /admin/feeds/{feed_id}. All fields optional. */
export interface UpdateFeedRequest {
  name?: string;
  url?: string;
  tier?: 1 | 2;
  category?: string;
  lang?: string;
  enabled?: boolean;
}

/** Single resolved or voided Polymarket comparison for the accuracy table. */
export interface ResolvedComparison {
  id: number;
  polymarket_title: string;
  polymarket_event_id: string;
  geopol_probability: number | null;
  polymarket_price: number | null;
  polymarket_outcome: number | null;
  geopol_brier: number | null;
  polymarket_brier: number | null;
  winner: string | null; // "geopol" | "polymarket" | "draw" | null
  status: string; // "resolved" | "voided"
  resolved_at: string | null;
  created_at: string;
  country_iso: string | null;
  category: string | null;
}

/** Aggregate accuracy stats. */
export interface AccuracySummary {
  total_resolved: number;
  total_voided: number;
  geopol_wins: number;
  polymarket_wins: number;
  draws: number;
  geopol_cumulative_brier: number | null;
  polymarket_cumulative_brier: number | null;
  rolling_30d_geopol_brier: number | null;
  rolling_30d_polymarket_brier: number | null;
  rolling_30d_count: number;
}

/** Full accuracy endpoint response. */
export interface AccuracyData {
  summary: AccuracySummary;
  comparisons: ResolvedComparison[];
}

/** Admin sidebar navigation sections. */
export type AdminSection = 'processes' | 'config' | 'logs' | 'sources' | 'accuracy';
