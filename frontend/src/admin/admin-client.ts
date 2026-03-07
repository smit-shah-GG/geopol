/**
 * AdminClient -- typed access to the admin API (/api/v1/admin/*).
 *
 * Unlike ForecastServiceClient, this is deliberately simple: no circuit
 * breakers, no in-flight dedup, no caching. Admin is low-traffic,
 * operator-facing, and should always show fresh data.
 */

import type {
  AccuracyData,
  AddFeedRequest,
  BacktestRun,
  BacktestRunDetail,
  CheckpointInfo,
  ConfigEntry,
  FeedInfo,
  LogEntry,
  ProcessInfo,
  SourceInfo,
  StartBacktestRequest,
  UpdateFeedRequest,
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

  /** POST /processes/{daemon_type}/pause -- pause a scheduled job. */
  async pauseJob(daemonType: string): Promise<{ status: string; daemon_type: string }> {
    return this.request<{ status: string; daemon_type: string }>(
      `/processes/${encodeURIComponent(daemonType)}/pause`,
      { method: 'POST' },
    );
  }

  /** POST /processes/{daemon_type}/resume -- resume a paused job. */
  async resumeJob(daemonType: string): Promise<{ status: string; daemon_type: string }> {
    return this.request<{ status: string; daemon_type: string }>(
      `/processes/${encodeURIComponent(daemonType)}/resume`,
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
  // Accuracy (22-03)
  // -----------------------------------------------------------------------

  /** GET /accuracy -- head-to-head Brier accuracy data. */
  async getAccuracy(): Promise<AccuracyData> {
    return this.request<AccuracyData>('/accuracy');
  }

  // -----------------------------------------------------------------------
  // Feed CRUD (21-05)
  // -----------------------------------------------------------------------

  /** GET /feeds -- list all RSS feeds with health metrics. */
  async getFeeds(): Promise<FeedInfo[]> {
    return this.request<FeedInfo[]>('/feeds');
  }

  /** POST /feeds -- add a new RSS feed. */
  async addFeed(data: AddFeedRequest): Promise<FeedInfo> {
    return this.request<FeedInfo>('/feeds', {
      method: 'POST',
      body: JSON.stringify(data),
    });
  }

  /** PUT /feeds/{id} -- update feed properties. */
  async updateFeed(id: number, data: UpdateFeedRequest): Promise<FeedInfo> {
    return this.request<FeedInfo>(`/feeds/${id}`, {
      method: 'PUT',
      body: JSON.stringify(data),
    });
  }

  /** DELETE /feeds/{id} -- soft-delete by default, hard delete with purge=true. */
  async deleteFeed(id: number, purge = false): Promise<void> {
    await this.request<void>(`/feeds/${id}${purge ? '?purge=true' : ''}`, {
      method: 'DELETE',
    });
  }

  // -----------------------------------------------------------------------
  // Backtesting (23-03)
  // -----------------------------------------------------------------------

  /** GET /backtesting/runs -- list all backtest runs. */
  async getBacktestRuns(): Promise<BacktestRun[]> {
    return this.request<BacktestRun[]>('/backtesting/runs');
  }

  /** POST /backtesting/runs -- start a new backtest run. */
  async startBacktestRun(config: StartBacktestRequest): Promise<BacktestRun> {
    return this.request<BacktestRun>('/backtesting/runs', {
      method: 'POST',
      body: JSON.stringify(config),
    });
  }

  /** GET /backtesting/runs/{runId} -- detail with per-window results. */
  async getBacktestRun(runId: string): Promise<BacktestRunDetail> {
    return this.request<BacktestRunDetail>(`/backtesting/runs/${encodeURIComponent(runId)}`);
  }

  /** POST /backtesting/runs/{runId}/cancel -- cancel running/pending run. */
  async cancelBacktestRun(runId: string): Promise<BacktestRun> {
    return this.request<BacktestRun>(
      `/backtesting/runs/${encodeURIComponent(runId)}/cancel`,
      { method: 'POST' },
    );
  }

  /** GET /backtesting/runs/{runId}/export?format={format} -- download as blob. */
  async exportBacktestRun(runId: string, format: 'csv' | 'json'): Promise<Blob> {
    const url = `${API_BASE}/backtesting/runs/${encodeURIComponent(runId)}/export?format=${format}`;
    const response = await fetch(url, {
      headers: { 'X-Admin-Key': this.key },
    });
    if (!response.ok) {
      throw new Error(`Export failed: ${response.status}`);
    }
    return response.blob();
  }

  /** GET /backtesting/checkpoints -- list available model checkpoints. */
  async getCheckpoints(): Promise<CheckpointInfo[]> {
    return this.request<CheckpointInfo[]>('/backtesting/checkpoints');
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
