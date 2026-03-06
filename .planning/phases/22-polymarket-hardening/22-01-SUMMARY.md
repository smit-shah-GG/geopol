---
phase: 22
plan: 01
subsystem: polymarket
tags: [polymarket, schema, alembic, bug-fix, brier-score]
dependency_graph:
  requires: [18]
  provides: [reforecasted_at-column, polymarket-accuracy-table, fixed-cap-tracking]
  affects: [22-02, 22-03]
tech_stack:
  added: []
  patterns: [immutable-timestamp, append-only-ledger]
key_files:
  created:
    - alembic/versions/20260306_008_polymarket_hardening.py
  modified:
    - src/db/models.py
    - src/polymarket/auto_forecaster.py
decisions:
  - Prediction.created_at is now immutable -- reforecasted_at tracks reforecast activity
  - count_today_reforecasts queries reforecasted_at directly -- no subtraction heuristic
  - PolymarketAccuracy is append-only (no UPDATE, no DELETE -- insert on each resolution)
metrics:
  duration: 5min
  completed: 2026-03-06
---

# Phase 22 Plan 01: Schema + Bug Fix (POLY-01, POLY-02 schema, POLY-04 schema) Summary

**One-liner:** Fixed created_at overwrite bug, added reforecasted_at column and polymarket_accuracy table via Alembic 008

## What Was Done

### Task 1: Alembic migration 008 + ORM model updates
- Added `Prediction.reforecasted_at` column: nullable `DateTime(timezone=True)`, indexed, placed after `polymarket_event_id`
- Added `PolymarketAccuracy` model: append-only ledger with cumulative Brier scores, rolling 30-day windows, win/loss/draw counts, triggered_by_comparison_id for traceability
- Updated `PolymarketComparison` status comment to document `voided` alongside `active` and `resolved`
- Created Alembic migration 008 chaining from 007: adds column + index on predictions, creates polymarket_accuracy table
- Updated module docstring to include polymarket_accuracy

### Task 2: Fix reforecast_active() bug and rewrite cap tracking
- **Bug fix (POLY-01):** Line 601 of `auto_forecaster.py` was `existing.created_at = datetime.now(timezone.utc)` -- replaced with `existing.reforecasted_at = datetime.now(timezone.utc)`. The `created_at` timestamp is now immutable.
- **count_today_reforecasts() rewrite:** Replaced the fragile subtraction heuristic (`total_polymarket_today - new_today`) with a direct query on `Prediction.reforecasted_at >= today_start`. The old approach was doubly broken: it relied on the very bug that was just fixed (overwritten `created_at`), and even when the bug existed, it double-counted new forecasts as reforecasts if they were created on the same day.
- **count_today_new_forecasts():** Left unchanged -- it already correctly queries `Prediction.created_at >= today_start` with `provenance == "polymarket_driven"`. The bug fix makes this function correct again because `created_at` is no longer overwritten during reforecast.
- AST verification confirms zero Store-context assignments to `created_at` in auto_forecaster.py.

## Deviations from Plan

None -- plan executed exactly as written.

## Verification Results

| Check | Result |
|-------|--------|
| `Prediction.reforecasted_at` column exists | PASS |
| `PolymarketAccuracy.__tablename__` == "polymarket_accuracy" | PASS |
| Alembic head == "008", down_revision == "007" | PASS |
| No Store assignment to created_at (AST check) | PASS (empty output) |
| `reforecasted_at` used in reforecast_active() and count_today_reforecasts() | PASS |
| `created_at` only READ, never WRITTEN in auto_forecaster.py | PASS |
| Test suite: no regressions (48 pre-existing failures, 285 pass, 2 skip) | PASS |

## Commits

| Task | Commit | Description |
|------|--------|-------------|
| 1 | 7e1ae05 | feat(22-01): alembic migration 008 + ORM model updates |
| 2 | e671c2d | fix(22-01): fix created_at overwrite bug and rewrite cap tracking |

## Decisions Made

1. **Prediction.created_at immutability:** Enforced by convention (no code writes to it after initial creation). No DB-level trigger needed -- the only writer was the reforecast_active() method, now fixed.
2. **Direct reforecasted_at query:** Replaces subtraction heuristic entirely. Cleaner, no edge cases, no dependency on created_at semantics.
3. **PolymarketAccuracy append-only pattern:** Each resolution appends a snapshot row. No updates, no deletes. Enables time-series accuracy curves without recomputing from comparison rows.

## Next Phase Readiness

Plan 22-02 (Brier scoring engine + resolution pipeline) can now proceed:
- `polymarket_accuracy` table ready for inserts
- `reforecasted_at` column available for temporal queries
- `voided` status documented for comparison lifecycle
- Cap tracking functions are correct and ready for production use
