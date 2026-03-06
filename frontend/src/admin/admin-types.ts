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

/** Admin sidebar navigation sections. */
export type AdminSection = 'processes' | 'config' | 'logs' | 'sources';
