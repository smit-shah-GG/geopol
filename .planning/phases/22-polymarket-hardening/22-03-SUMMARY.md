---
phase: 22-polymarket-hardening
plan: 03
subsystem: admin-ui, api
tags: [polymarket, brier-score, accuracy, admin-panel, sortable-table, fastapi]

# Dependency graph
requires:
  - phase: 22-polymarket-hardening
    provides: "PolymarketComparison schema with geopol_brier/polymarket_brier/status columns (22-01)"
  - phase: 19-admin-dashboard-foundation
    provides: "Admin layout, sidebar, client, CSS architecture"
provides:
  - "GET /admin/accuracy endpoint with AccuracyResponse DTO"
  - "AccuracyPanel: sortable table + summary stats for head-to-head comparison"
  - "Admin sidebar 'Accuracy' section (5th nav item)"
affects: [23-backtesting-engine, 24-global-seeding]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "In-place text node stat updates (no innerHTML rebuild)"
    - "Client-side sortable table with direction state preservation"
    - "Winner determination: lower Brier = better"

key-files:
  created:
    - frontend/src/admin/panels/AccuracyPanel.ts
  modified:
    - src/api/schemas/admin.py
    - src/api/services/admin_service.py
    - src/api/routes/v1/admin.py
    - frontend/src/admin/admin-types.ts
    - frontend/src/admin/components/AdminSidebar.ts
    - frontend/src/admin/admin-layout.ts
    - frontend/src/admin/admin-client.ts
    - frontend/src/admin/admin-styles.css

key-decisions:
  - "Live-computed accuracy from polymarket_comparisons table, not from polymarket_accuracy ledger"
  - "Winner = lower Brier score (geopol_brier < polymarket_brier = geopol wins)"
  - "Outerjoin to predictions for country_iso and category metadata"
  - "200-row limit on comparisons list (resolved + voided)"
  - "Client-side sorting (data volume low, avoids round-trip)"

patterns-established:
  - "Sortable table header pattern: .sortable class, .sort-asc/.sort-desc with CSS ::after arrows"
  - "Winner badge pattern: green G (geopol), red PM (polymarket), gray = (draw), strikethrough V (voided)"

# Metrics
duration: 6min
completed: 2026-03-06
---

# Phase 22 Plan 03: Admin Accuracy Panel Summary

**Head-to-head Geopol vs Polymarket accuracy panel with 9-card summary stats bar, sortable 9-column table, winner badges, and 30s auto-refresh**

## Performance

- **Duration:** 6 min
- **Started:** 2026-03-06T14:35:45Z
- **Completed:** 2026-03-06T14:41:24Z
- **Tasks:** 2
- **Files modified:** 9

## Accomplishments
- Backend GET /admin/accuracy endpoint returning live-computed summary stats (cumulative + rolling 30d Brier, win/loss/draw counts) with resolved/voided comparison list joined to prediction metadata
- Frontend AccuracyPanel (397 lines) with 9 stat cards (resolved, voided, geopol wins, PM wins, draws, cumulative Brier both, rolling 30d both), sortable table with 9 columns, voided row styling, and empty state
- Full admin registration: types, sidebar nav, layout switch, client method, panel CSS

## Task Commits

Each task was committed atomically:

1. **Task 1: Backend accuracy endpoint + DTOs** - `9f3226e` (feat)
2. **Task 2: Frontend AccuracyPanel + admin registration** - `0a4eacb` (feat)

## Files Created/Modified
- `src/api/schemas/admin.py` - AccuracyResponse, AccuracySummary, ResolvedComparisonDTO Pydantic DTOs
- `src/api/services/admin_service.py` - get_accuracy() with summary aggregates + comparison list query
- `src/api/routes/v1/admin.py` - GET /accuracy endpoint
- `frontend/src/admin/admin-types.ts` - AccuracyData, AccuracySummary, ResolvedComparison TS interfaces + AdminSection union
- `frontend/src/admin/components/AdminSidebar.ts` - 5th nav item (Accuracy with bullseye icon)
- `frontend/src/admin/admin-layout.ts` - AccuracyPanel import + createPanel case + SECTION_TITLES
- `frontend/src/admin/admin-client.ts` - getAccuracy() method
- `frontend/src/admin/panels/AccuracyPanel.ts` - Full accuracy panel implementation (397 lines)
- `frontend/src/admin/admin-styles.css` - Accuracy stats bar, winner badges, sortable headers, voided rows, empty state

## Decisions Made
- Live-computed accuracy from polymarket_comparisons table rather than polymarket_accuracy ledger (ledger is for historical trend; admin shows real-time data)
- Winner determination: lower Brier score wins (geopol_brier < polymarket_brier = geopol wins)
- 200-row limit on comparisons list -- sufficient for admin inspection, avoids unbounded queries
- Client-side sorting -- data volume is low (max 200 rows), avoids unnecessary server round-trips
- In-place text node mutation for stat values on refresh (no innerHTML rebuild)

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- Phase 22 complete: all 3 plans delivered (schema/bug fix, Brier scoring engine, accuracy panel)
- Admin dashboard now has 5 sections: Processes, Config, Logs, Sources, Accuracy
- Ready for Phase 23 (Backtesting Engine) which will consume accuracy data patterns

---
*Phase: 22-polymarket-hardening*
*Completed: 2026-03-06*
