/**
 * AdminClient -- typed access to the admin API (/api/v1/admin/*).
 *
 * Unlike ForecastServiceClient, this is deliberately simple: no circuit
 * breakers, no in-flight dedup, no caching. Admin is low-traffic,
 * operator-facing, and should always show fresh data.
 */

import type {
  ConfigEntry,
  LogEntry,
  ProcessInfo,
  SourceInfo,
} from '@/admin/admin-types';

const API_BASE = '/api/v1/admin';

class AdminApiError extends Error {
  constructor(
    public readonly status: number,
    public readonly path: string,
  ) {
    super(`Admin API ${status}: ${path}`);
    this.name = 'AdminApiError';
  }
}

export class AdminClient {
  private readonly key: string;

  constructor(adminKey: string) {
    this.key = adminKey;
  }

  // -----------------------------------------------------------------------
  // Public API
  // -----------------------------------------------------------------------

  /** POST /verify -- lightweight auth check. */
  async verify(): Promise<{ status: string }> {
    return this.request<{ status: string }>('/verify', { method: 'POST' });
  }

  /** GET /processes -- daemon status table. */
  async getProcesses(): Promise<ProcessInfo[]> {
    return this.request<ProcessInfo[]>('/processes');
  }

  /** POST /processes/{daemon_type}/trigger -- spawn one-shot job. */
  async triggerJob(daemonType: string): Promise<{ status: string; daemon_type: string }> {
    return this.request<{ status: string; daemon_type: string }>(
      `/processes/${encodeURIComponent(daemonType)}/trigger`,
      { method: 'POST' },
    );
  }

  /** GET /config -- all runtime settings. */
  async getConfig(): Promise<ConfigEntry[]> {
    return this.request<ConfigEntry[]>('/config');
  }

  /** PUT /config -- batch update settings. */
  async updateConfig(updates: Record<string, unknown>): Promise<{ status: string; keys: string[] }> {
    return this.request<{ status: string; keys: string[] }>('/config', {
      method: 'PUT',
      body: JSON.stringify({ updates }),
    });
  }

  /** DELETE /config -- reset all overrides to defaults. */
  async resetConfig(): Promise<{ status: string }> {
    return this.request<{ status: string }>('/config', { method: 'DELETE' });
  }

  /** GET /logs -- ring buffer entries with optional filters. */
  async getLogs(params?: { severity?: string; subsystem?: string }): Promise<LogEntry[]> {
    const sp = new URLSearchParams();
    if (params?.severity) sp.set('severity', params.severity);
    if (params?.subsystem) sp.set('subsystem', params.subsystem);
    const qs = sp.toString();
    return this.request<LogEntry[]>(`/logs${qs ? `?${qs}` : ''}`);
  }

  /** GET /sources -- per-source health. */
  async getSources(): Promise<SourceInfo[]> {
    return this.request<SourceInfo[]>('/sources');
  }

  /** PUT /sources/{name}/toggle -- enable/disable a source. */
  async toggleSource(name: string, enabled: boolean): Promise<{ status: string }> {
    return this.request<{ status: string }>(
      `/sources/${encodeURIComponent(name)}/toggle`,
      {
        method: 'PUT',
        body: JSON.stringify({ enabled }),
      },
    );
  }

  // -----------------------------------------------------------------------
  // Private
  // -----------------------------------------------------------------------

  private async request<T>(path: string, init?: RequestInit): Promise<T> {
    const url = `${API_BASE}${path}`;
    const headers: Record<string, string> = {
      'X-Admin-Key': this.key,
      'Content-Type': 'application/json',
    };

    const response = await fetch(url, {
      ...init,
      headers: {
        ...headers,
        ...(init?.headers as Record<string, string> | undefined),
      },
    });

    if (!response.ok) {
      throw new AdminApiError(response.status, path);
    }

    return (await response.json()) as T;
  }
}
