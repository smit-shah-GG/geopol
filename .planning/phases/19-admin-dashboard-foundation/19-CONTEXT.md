# Phase 19: Admin Dashboard Foundation - Context

**Gathered:** 2026-03-05
**Status:** Ready for planning

<domain>
## Phase Boundary

Browser-based operator control panel at `/admin` route. Provides real-time visibility into all running jobs, data source health, system configuration, and recent log output. Covers ADMIN-01 through ADMIN-06: route with auth gating, process table, manual triggers, config editor, log viewer, source management panel. This is the observation layer that every subsequent v3.0 phase extends.

</domain>

<decisions>
## Implementation Decisions

### Dashboard layout & density
- Two-column layout with sidebar nav (left sidebar with section links, right area shows selected section)
- Process table uses compact table rows — one row per job, all info inline: name, status dot, last run, next run, success/fail counts, trigger button (dense like htop)
- Distinct red-accented admin theme — visually differentiated from the main app's dark blue theme so operators know they're in admin mode. Same CSS variable system, red primary accent instead of blue.
- Source management panel uses card grid — each source as a card showing health, tier badge, and controls

### Auth gating & session flow
- Modal overlay on `/admin` — navigating to /admin shows page dimmed with centered auth modal. Key accepted -> modal dismisses, dashboard appears.
- sessionStorage (per-tab) — auth dies when tab closes. Each new tab requires re-entering the key. Most secure for single-operator use.
- Rate limit after 5 failed attempts — 30s cooldown delay before next attempt. Prevents brute force without permanently locking out the operator.
- No session timeout — session stays active until tab close. Operator keeps dashboard open all day without re-auth.

### Log viewer behavior
- Auto-scroll by default, pause on scroll-up — tails logs live. If operator scrolls up to inspect older entries, auto-scroll pauses. Resume button to jump back to bottom.
- Color-coded rows with toggle pill buttons — ERROR=red, WARN=amber, INFO=neutral. Clickable pill buttons at top to toggle each severity on/off independently.
- Client-side text search — search box filters the in-memory ring buffer entries by substring match. Instant, no server round-trip.
- Subsystem column with clickable inline filter — each log entry shows its subsystem (gdelt, rss, pipeline, polymarket, etc.). Clicking a subsystem name in any row filters to that subsystem only.

### Config editor guardrails
- All runtime settings editable — everything in the settings module that doesn't require a restart. Secrets (API keys, database URLs) displayed as read-only masked values.
- Inline validation errors per field — red border + error message directly under the invalid field. Save button disabled until all fields valid.
- Revert to defaults button — single button restores all settings to hardcoded defaults. No per-change undo history.
- Confirmation dialog only for risky changes — normal saves go through immediately. Settings marked dangerous (e.g. disabling all polling, zeroing daily caps) show a confirm dialog with description of the impact.

### Claude's Discretion
- Sidebar nav section ordering and icons
- Exact red accent hue (as long as it's clearly distinct from main app blue)
- Log viewer ring buffer poll interval
- Which specific settings are marked "dangerous" requiring confirmation
- Process table column widths and responsive breakpoints
- Card grid layout for sources (2-col vs 3-col)

</decisions>

<specifics>
## Specific Ideas

- Red-themed admin to visually distinguish from the main app — operator should immediately know they're in admin mode
- Process table density inspired by htop — maximum information per row, no wasted space
- Log viewer should feel like a terminal tail — live-scrolling, instant filter response

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 19-admin-dashboard-foundation*
*Context gathered: 2026-03-05*
