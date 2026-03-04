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
  status: string; // running | success | failed | unknown
  last_run: string | null;
  next_run: string | null;
  success_count: number;
  fail_count: number;
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

/** Admin sidebar navigation sections. */
export type AdminSection = 'processes' | 'config' | 'logs' | 'sources';
