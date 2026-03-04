# Phase 19: Admin Dashboard Foundation - Research

**Researched:** 2026-03-05
**Domain:** Admin UI (vanilla TypeScript SPA) + FastAPI admin API endpoints + PostgreSQL config persistence + in-memory log ring buffer
**Confidence:** HIGH

## Summary

Phase 19 adds a `/admin` route to the existing vanilla TypeScript SPA with a full operator control panel: auth gating, process table, manual job triggers, configuration editor, log viewer, and source management. The entire admin UI must be dynamic-import code-split so zero bytes ship in the public bundle.

The existing codebase provides strong, well-established patterns that this phase extends. The Router class (`frontend/src/app/router.ts`) already supports `addRoute()` with mount/unmount lifecycle. The globe screen (`globe-screen.ts`) demonstrates the exact dynamic `import()` pattern needed for code splitting. The Panel base class, dom-utils `h()` helper, RefreshScheduler, and forecastClient patterns are all reusable without modification.

The backend needs 6 new API endpoints under `/api/v1/admin/` with admin-key auth (separate from the existing API key system), a new `system_config` PostgreSQL table + Alembic migration, an in-memory ring buffer logging handler, and job trigger endpoints that invoke existing ingest/pipeline functions.

**Primary recommendation:** Follow the existing globe-screen dynamic import pattern exactly for admin code splitting. Build all admin panels as standalone classes (not extending Panel -- admin uses a different two-column layout, not the dashboard's four-column grid). Reuse `h()`, `replaceChildren()`, and `forecastClient` patterns but create a dedicated `adminClient` service.

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| Vanilla TypeScript | 5.7+ | Admin frontend | Already used, no framework to add |
| FastAPI | existing | Admin API endpoints | Already the API framework |
| SQLAlchemy 2.0 | existing | system_config table ORM | Already used for all persistence |
| Alembic | existing | Migration for system_config | Already used for all schema changes |
| Pydantic | existing | Admin API DTOs | Already used for all API schemas |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| logging (stdlib) | 3.11+ | Ring buffer handler | Custom `logging.Handler` subclass for in-memory buffer |
| collections.deque | stdlib | Ring buffer data structure | Fixed-size O(1) append/popleft for log entries |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Custom ring buffer handler | python-loguru | Adds dependency for something stdlib handles in ~30 lines |
| sessionStorage auth | Cookie-based sessions | sessionStorage is per-tab (decided), cookies leak across tabs |
| Dedicated admin auth endpoint | Reuse existing verify_api_key | Admin needs separate auth flow -- modal overlay, not header-based |

**Installation:**
No new dependencies required. Everything uses existing stack.

## Architecture Patterns

### Recommended Project Structure
```
frontend/src/
├── screens/
│   └── admin-screen.ts         # Dynamic import entry, auth gate, layout shell
├── admin/                       # NEW: all admin code in separate directory for clean code-split
│   ├── admin-client.ts          # API client for /api/v1/admin/* endpoints
│   ├── admin-types.ts           # TypeScript interfaces for admin API responses
│   ├── panels/
│   │   ├── ProcessTable.ts      # ADMIN-02: job status table
│   │   ├── ConfigEditor.ts      # ADMIN-04: settings editor
│   │   ├── LogViewer.ts         # ADMIN-05: ring buffer log viewer
│   │   └── SourceManager.ts     # ADMIN-06: source health + controls
│   ├── components/
│   │   ├── AuthModal.ts         # ADMIN-01: auth overlay
│   │   └── AdminSidebar.ts      # Sidebar nav for section switching
│   └── admin-styles.css         # Red-accented admin theme
src/api/
├── routes/v1/
│   └── admin.py                 # NEW: admin API router
├── schemas/
│   └── admin.py                 # NEW: admin Pydantic DTOs
├── services/
│   └── admin_service.py         # NEW: admin business logic
└── log_buffer.py                # NEW: ring buffer logging handler
src/db/
└── models.py                    # ADD: SystemConfig model
alembic/versions/
└── 20260305_006_system_config.py  # NEW: system_config table
```

### Pattern 1: Dynamic Import Code Splitting (from globe-screen.ts)
**What:** The admin screen file in `screens/` contains only the auth modal and a dynamic `import()` call. All admin panel code lives in `frontend/src/admin/` which Vite tree-shakes out of the main bundle.
**When to use:** Always -- this is the ADMIN-01 zero-bytes-in-public-bundle requirement.
**Example:**
```typescript
// screens/admin-screen.ts -- this file IS in the main bundle but is tiny
// Source: existing pattern from frontend/src/screens/globe-screen.ts

export async function mountAdmin(container: HTMLElement): Promise<void> {
  // Show auth modal first
  const authModal = createAuthModal(container);
  const adminKey = await authModal.waitForAuth();

  // Only after auth succeeds, dynamic-import the admin module
  const { mountAdminDashboard } = await import('@/admin/panels/ProcessTable');
  // ... import all admin panels
  await mountAdminDashboard(container, adminKey);
}
```

### Pattern 2: Admin Route Registration (extends existing Router)
**What:** Add `/admin` as a fourth route in `main.ts`. The NavBar does NOT show an admin link -- admin is accessed by direct URL only.
**When to use:** Route setup in main.ts boot sequence.
**Example:**
```typescript
// main.ts addition -- admin screen uses dynamic import at route level
router.addRoute({
  path: '/admin',
  mount: (container) => mountAdmin(container),
  unmount: () => unmountAdmin(),
});
```

### Pattern 3: Two-Column Admin Layout
**What:** Admin uses sidebar nav (left) + content area (right), not the dashboard's four-column grid. The sidebar has clickable section links (Process Table, Config, Logs, Sources). Clicking a section swaps the content area.
**When to use:** Admin layout construction after auth.
**Example:**
```typescript
function createAdminLayout(): { sidebar: HTMLElement; content: HTMLElement; wrapper: HTMLElement } {
  const wrapper = h('div', { className: 'admin-layout' });
  const sidebar = h('nav', { className: 'admin-sidebar' });
  const content = h('div', { className: 'admin-content' });
  wrapper.appendChild(sidebar);
  wrapper.appendChild(content);
  return { sidebar, content, wrapper };
}
```

### Pattern 4: Admin Auth -- Modal Overlay with Promise Resolution
**What:** Navigating to `/admin` shows a dimmed page with centered auth modal. The modal returns a Promise that resolves with the admin key on success, or rejects on dismiss. Rate limiting (5 attempts, 30s cooldown) is client-side.
**When to use:** Auth gating before loading any admin code.
**Example:**
```typescript
class AuthModal {
  private attempts = 0;
  private cooldownUntil = 0;

  waitForAuth(): Promise<string> {
    return new Promise((resolve, reject) => {
      // Show modal, wire submit handler
      // On success: sessionStorage.setItem('admin_key', key); resolve(key);
      // On 5 failures: disable input for 30s
    });
  }
}
```

### Pattern 5: Ring Buffer Logging Handler (Python)
**What:** A custom `logging.Handler` that stores the last N log records in a `collections.deque(maxlen=1000)`. Each entry is a structured dict: `{timestamp, severity, module, message}`. The handler is added to the root logger during app lifespan startup. The admin API endpoint reads from this deque.
**When to use:** ADMIN-05 log viewer backend.
**Example:**
```python
import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone

@dataclass
class LogEntry:
    timestamp: str
    severity: str
    module: str
    message: str

class RingBufferHandler(logging.Handler):
    def __init__(self, maxlen: int = 1000):
        super().__init__()
        self.buffer: deque[LogEntry] = deque(maxlen=maxlen)

    def emit(self, record: logging.LogRecord) -> None:
        entry = LogEntry(
            timestamp=datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            severity=record.levelname,
            module=record.name,
            message=record.getMessage(),
        )
        self.buffer.append(entry)

    def get_entries(self, severity: str | None = None, subsystem: str | None = None) -> list[LogEntry]:
        entries = list(self.buffer)
        if severity:
            entries = [e for e in entries if e.severity == severity]
        if subsystem:
            entries = [e for e in entries if subsystem in e.module]
        return entries
```

### Pattern 6: system_config Table Design
**What:** Key-value store with JSON values for runtime-adjustable settings. Each row has a `key` (unique), `value` (JSON), `updated_at` timestamp, and `updated_by` identifier. On startup, missing keys are seeded from `Settings` defaults.
**When to use:** ADMIN-04 config persistence.
**Example:**
```python
class SystemConfig(Base):
    __tablename__ = "system_config"

    key: Mapped[str] = mapped_column(String(100), primary_key=True)
    value: Mapped[dict] = mapped_column(JSON, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_by: Mapped[str] = mapped_column(String(100), default="system")
```

### Pattern 7: Admin API Auth (Separate from User API Key)
**What:** Admin endpoints use a dedicated `ADMIN_KEY` environment variable, validated via a FastAPI dependency. This is NOT the same as the existing `verify_api_key` which checks the `api_keys` table. Admin auth is a simple string comparison against a single env var.
**When to use:** All `/api/v1/admin/*` endpoints.
**Example:**
```python
async def verify_admin_key(
    x_admin_key: str | None = Header(None, alias="X-Admin-Key"),
) -> None:
    settings = get_settings()
    if not settings.admin_key:
        raise HTTPException(503, "Admin access not configured")
    if x_admin_key != settings.admin_key:
        raise HTTPException(401, "Invalid admin key")
```

### Anti-Patterns to Avoid
- **Importing admin modules statically in main.ts:** This defeats code splitting. The `import()` must be dynamic.
- **Reusing the Panel base class for admin panels:** Panel has resize handles, span classes, grid layout assumptions. Admin panels use a different layout. Build admin panel classes from scratch (they're simpler).
- **Reading logs from the filesystem:** ADMIN-05 explicitly says "not filesystem reads". Use the in-memory ring buffer.
- **Storing admin auth in localStorage:** Decided: sessionStorage (per-tab). localStorage persists across tabs and browser restarts.
- **Adding admin link to NavBar:** Admin is accessed by direct URL `/admin` only. The NavBar should not expose it.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Dynamic import code splitting | Manual chunk loading | Vite's native `import()` support | Vite handles chunking, hashing, and loading automatically |
| Ring buffer | Array with manual index tracking | `collections.deque(maxlen=N)` | O(1) append, automatic eviction, thread-safe for single-writer |
| Admin API auth | Custom middleware | FastAPI `Depends()` with `Header()` | Consistent with existing auth pattern, just different key source |
| Config persistence | Custom file-based config | PostgreSQL table + SQLAlchemy | Consistent with all other persistence, survives restarts |
| Client-side rate limiting | Complex token bucket | Simple counter + timestamp | 5 attempts with 30s cooldown is trivially implementable |

**Key insight:** The admin dashboard is architecturally simple -- it's CRUD over existing data (ingest_runs, settings, logs) with a new UI. No novel algorithms or complex state management needed.

## Common Pitfalls

### Pitfall 1: Code Splitting Leaks
**What goes wrong:** Admin code ends up in the main bundle because a top-level `import` sneaks in (e.g., importing admin types in a shared file).
**Why it happens:** TypeScript `import type` is safe (erased at compile time), but `import { AdminPanel }` is not.
**How to avoid:** Keep ALL admin runtime code in `frontend/src/admin/`. The only import from `screens/admin-screen.ts` should be dynamic `import()`. Verify with `npx vite build && ls -la dist/assets/` -- admin chunk should be separate.
**Warning signs:** No separate admin chunk in build output; main bundle size increases.

### Pitfall 2: Ring Buffer Thread Safety
**What goes wrong:** Concurrent log writes from async handlers corrupt the deque.
**Why it happens:** `deque` is thread-safe for append/popleft in CPython (GIL protects), but NOT safe across multiple processes.
**How to avoid:** `uvicorn --workers 1` is already a hard constraint (APScheduler in-process). Single-process = deque is safe. Document this constraint explicitly.
**Warning signs:** Garbled log entries (won't happen with workers=1).

### Pitfall 3: Settings Override Ordering
**What goes wrong:** Changing a setting in the admin UI has no effect because `Settings` is a cached singleton created from env vars at startup.
**Why it happens:** `get_settings()` returns a frozen singleton. The system_config table values are never consulted at runtime.
**How to avoid:** Build a `RuntimeSettings` layer that checks `system_config` table first, falls back to `Settings` for missing keys. Cache with short TTL or invalidate on write. Settings.admin_key and database URLs remain env-only (never overridable via UI).
**Warning signs:** Config changes not taking effect without restart.

### Pitfall 4: Auth Modal and Route Lifecycle
**What goes wrong:** Navigating away from `/admin` and back shows a flash of admin content before the auth modal appears.
**Why it happens:** The `unmount()` function cleaned up the modal but `sessionStorage` still has the key from the previous session.
**How to avoid:** On mount, always check sessionStorage first. If key exists, validate it with a lightweight `/api/v1/admin/verify` endpoint before showing content. If invalid, clear sessionStorage and show modal.
**Warning signs:** Stale auth letting users see admin without re-entering key.

### Pitfall 5: Process Table Data Source Gap
**What goes wrong:** The process table can't show "next scheduled run" or "currently running" because `ingest_runs` only records completed runs.
**Why it happens:** `ingest_runs` is an audit table (records after completion), not a job scheduler state table. There's no "next_run_at" or "is_running" column.
**How to avoid:** For Phase 19 (pre-APScheduler consolidation in Phase 20), derive status from `ingest_runs` data:
  - **Last run:** `MAX(started_at)` per daemon_type
  - **Status:** Latest row's status field (success/failed)
  - **Success/fail counts:** `COUNT` grouped by status per daemon_type
  - **Next run:** Calculate from last_run + poll_interval (from Settings)
  - **Currently running:** Check for rows with `status='running'` and `completed_at IS NULL`
Phase 20 (daemon consolidation with APScheduler) will provide proper scheduler state. Phase 19 should use the best-effort derivation above.
**Warning signs:** "Next run" showing incorrect times because it's calculated, not from scheduler state.

### Pitfall 6: Admin Styles Bleeding into Main App
**What goes wrong:** Red admin accent colors appear on the dashboard/globe/forecasts screens.
**Why it happens:** CSS loaded via dynamic import persists in the DOM after the admin screen unmounts.
**How to avoid:** Scope all admin CSS under a `.admin-layout` parent selector. On unmount, remove the admin stylesheet `<link>` element (or use CSS module scoping). Alternatively, load admin CSS as a `<style>` tag appended during mount and removed during unmount.
**Warning signs:** Red accent colors visible on non-admin screens after visiting admin.

## Code Examples

### Admin API Router Structure
```python
# Source: follows pattern from src/api/routes/v1/router.py

from fastapi import APIRouter, Depends, Header, HTTPException
from src.settings import get_settings

router = APIRouter()

async def verify_admin(
    x_admin_key: str | None = Header(None, alias="X-Admin-Key"),
) -> None:
    settings = get_settings()
    if not settings.admin_key or x_admin_key != settings.admin_key:
        raise HTTPException(401, "Invalid admin key")

# All admin endpoints require admin auth
@router.get("/processes", dependencies=[Depends(verify_admin)])
async def get_processes(db: AsyncSession = Depends(get_db)):
    """Process table data: last run, status, counts per daemon type."""
    ...

@router.post("/processes/{daemon_type}/trigger", dependencies=[Depends(verify_admin)])
async def trigger_job(daemon_type: str):
    """Manual trigger for a specific daemon job."""
    ...

@router.get("/config", dependencies=[Depends(verify_admin)])
async def get_config(db: AsyncSession = Depends(get_db)):
    """All runtime-adjustable settings with current values."""
    ...

@router.put("/config", dependencies=[Depends(verify_admin)])
async def update_config(updates: dict, db: AsyncSession = Depends(get_db)):
    """Persist config changes to system_config table."""
    ...

@router.get("/logs", dependencies=[Depends(verify_admin)])
async def get_logs(severity: str | None = None, subsystem: str | None = None):
    """Recent log entries from in-memory ring buffer."""
    ...

@router.post("/verify", dependencies=[Depends(verify_admin)])
async def verify_auth():
    """Lightweight auth verification (returns 200 or 401)."""
    return {"status": "authenticated"}
```

### Admin Client (Frontend)
```typescript
// Source: follows pattern from frontend/src/services/forecast-client.ts

const ADMIN_BASE = '/api/v1/admin';

class AdminClient {
  private adminKey: string;

  constructor(adminKey: string) {
    this.adminKey = adminKey;
  }

  private async fetch<T>(path: string, init?: RequestInit): Promise<T> {
    const res = await fetch(`${ADMIN_BASE}${path}`, {
      ...init,
      headers: {
        'X-Admin-Key': this.adminKey,
        'Content-Type': 'application/json',
        ...(init?.headers ?? {}),
      },
    });
    if (!res.ok) throw new Error(`Admin API ${res.status}: ${path}`);
    return res.json();
  }

  getProcesses(): Promise<ProcessInfo[]> { return this.fetch('/processes'); }
  triggerJob(daemon: string): Promise<void> { return this.fetch(`/processes/${daemon}/trigger`, { method: 'POST' }); }
  getConfig(): Promise<ConfigEntry[]> { return this.fetch('/config'); }
  updateConfig(updates: Record<string, unknown>): Promise<void> {
    return this.fetch('/config', { method: 'PUT', body: JSON.stringify(updates) });
  }
  getLogs(params?: { severity?: string; subsystem?: string }): Promise<LogEntry[]> {
    const qs = new URLSearchParams();
    if (params?.severity) qs.set('severity', params.severity);
    if (params?.subsystem) qs.set('subsystem', params.subsystem);
    return this.fetch(`/logs?${qs}`);
  }
  verify(): Promise<{ status: string }> { return this.fetch('/verify', { method: 'POST' }); }
}
```

### Process Table Row Rendering
```typescript
// Source: follows h() pattern from frontend/src/utils/dom-utils.ts
function renderProcessRow(proc: ProcessInfo): HTMLElement {
  const statusDot = h('span', {
    className: `status-dot status-${proc.status}`,
  });
  return h('tr', { className: 'process-row' },
    h('td', null, statusDot, proc.name),
    h('td', null, proc.last_run ?? 'Never'),
    h('td', null, proc.next_run ?? '--'),
    h('td', null, `${proc.success_count}/${proc.fail_count}`),
    h('td', null,
      h('button', {
        className: 'trigger-btn',
        onclick: () => adminClient.triggerJob(proc.daemon_type),
      }, 'Run'),
    ),
  );
}
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Polling-only daemons (asyncio.sleep loops) | APScheduler (Phase 20) | Phase 20 (next phase) | Phase 19 must derive scheduler state from ingest_runs; Phase 20 will provide native scheduler introspection |
| File-based logging only | File + ring buffer handler | Phase 19 (this phase) | Admin log viewer reads from memory, not filesystem |
| Env-var-only config | Env vars + system_config table | Phase 19 (this phase) | Runtime changes without restart for non-secret settings |

**Deprecated/outdated:**
- Nothing specific to deprecate. This phase adds new infrastructure.

## Open Questions

1. **Job trigger mechanism pre-APScheduler**
   - What we know: Current daemons run as `asyncio.sleep` loops (Polymarket in `app.py`, GDELT/RSS presumably similar). There's no centralized job registry.
   - What's unclear: How to trigger a job "immediately" when it's sleeping in a loop. Options: (a) cancel the sleep task and restart, (b) set a flag that the loop checks, (c) run a one-shot invocation in a new task alongside the loop.
   - Recommendation: Option (c) -- spawn a new `asyncio.Task` that runs the job function once. The existing loop continues independently. Risk of concurrent execution is low (jobs are idempotent via ingest_runs dedup). Phase 20's APScheduler will replace this with proper `scheduler.run_job()`.

2. **Which settings are "dangerous" (require confirmation dialog)?**
   - Recommendation: `gdelt_poll_interval < 60`, `gemini_daily_budget = 0`, `polymarket_enabled = false`, `rss_poll_interval_tier1 < 60`. These could either hammer external APIs or disable core functionality.

3. **Source management granularity for RSS**
   - What we know: ADMIN-06 requires "feed-level controls for RSS". The feed list is currently hardcoded in `feed_config.py` (84 feeds across 2 tiers).
   - What's unclear: Whether feed enable/disable state should persist to the `system_config` table or a separate `feed_overrides` table.
   - Recommendation: Store `disabled_feeds: string[]` (list of feed names) in `system_config` as a single JSON key. The RSS daemon checks this list before polling each feed. Simple, no schema change beyond the already-planned `system_config` table.

## Sources

### Primary (HIGH confidence)
- Codebase analysis: `frontend/src/main.ts`, `frontend/src/app/router.ts`, `frontend/src/screens/globe-screen.ts` -- dynamic import and routing patterns
- Codebase analysis: `frontend/src/components/Panel.ts` -- base panel pattern (identified as NOT suitable for admin)
- Codebase analysis: `src/api/app.py`, `src/api/routes/v1/router.py` -- API structure and endpoint patterns
- Codebase analysis: `src/db/models.py` -- ORM model patterns, especially `IngestRun` schema
- Codebase analysis: `src/settings.py` -- all runtime-adjustable setting fields
- Codebase analysis: `src/logging_config.py` -- existing logging infrastructure
- Codebase analysis: `src/api/middleware/auth.py` -- existing API key auth pattern
- Codebase analysis: `src/ingest/feed_config.py` -- RSS feed structure for source management

### Secondary (MEDIUM confidence)
- Python stdlib docs: `collections.deque` thread safety under CPython GIL
- Vite documentation: dynamic `import()` code splitting behavior

### Tertiary (LOW confidence)
- None

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- all technologies already in use, no new dependencies
- Architecture: HIGH -- patterns derived directly from existing codebase (globe-screen, Panel, Router)
- Pitfalls: HIGH -- identified through direct codebase analysis (settings singleton, ingest_runs schema gaps, CSS scoping)
- Process table data source: MEDIUM -- "next run" calculation is best-effort until Phase 20 APScheduler

**Research date:** 2026-03-05
**Valid until:** 2026-04-05 (stable -- no external dependencies to go stale)
