---
phase: 22-polymarket-hardening
plan: 02
subsystem: polymarket
tags: [brier-score, resolution, circuit-breaker, rate-limiting, gamma-api, accuracy]

# Dependency graph
requires:
  - phase: 22-01
    provides: PolymarketAccuracy table, reforecasted_at column, created_at immutability fix
  - phase: 18-polymarket-driven-forecasting
    provides: PolymarketAutoForecaster, PolymarketComparisonService, PolymarketClient
  - phase: 20-daemon-consolidation
    provides: heavy_runner.py with run_polymarket_cycle
provides:
  - Resolution engine using Gamma API closed/resolutionSource metadata
  - Voided market detection (_is_voided_market)
  - Cumulative accuracy snapshots (PolymarketAccuracy rows)
  - Reordered cycle (resolve BEFORE reforecast)
  - Top-10 active set model replacing shotgun volume threshold
  - 429-specific Retry-After backoff in client
  - Circuit breaker state exposure (circuit_state, consecutive_failures)
affects: [22-03, 23-backtesting, admin-api]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Resolution via API metadata fields (closed, resolutionSource) rather than price convergence heuristic"
    - "Append-only accuracy ledger pattern: compute_accuracy_snapshot() after each resolution batch"
    - "Cycle ordering as race-condition prevention: resolve -> forecast -> reforecast"
    - "Top-N active set model: fetch_top_geopolitical(limit=10) instead of volume threshold filtering"

key-files:
  created: []
  modified:
    - src/polymarket/comparison.py
    - src/polymarket/client.py
    - src/scheduler/heavy_runner.py
    - src/polymarket/auto_forecaster.py

key-decisions:
  - "Resolution uses Gamma API closed/resolutionSource/umaResolutionStatus instead of price convergence alone"
  - "Voided markets detected via ambiguous prices [0.5, 0.5] OR void/cancel keywords in resolution metadata"
  - "Accuracy snapshot computed only when comparisons are resolved (not voided-only)"
  - "Volume threshold filter removed from auto_forecaster.run() -- caller provides pre-filtered top-10 set"
  - "reforecast_active() scoped to top-10 event IDs for focused budget allocation"

patterns-established:
  - "resolve_completed() returns dict with resolved/voided/accuracy_snapshot_computed counts"
  - "429 backoff: sleep(min(Retry-After, 60)) before raising for tenacity retry"
  - "circuit_state property for admin status: closed | open | half-open"

# Metrics
duration: 12min
completed: 2026-03-06
---

# Phase 22 Plan 02: Brier Scoring Engine + Resolution Pipeline Summary

**Resolution engine with voided market detection via Gamma API metadata, cumulative accuracy snapshots, race-safe cycle ordering, and top-10 active set model**

## Performance

- **Duration:** 12 min
- **Started:** 2026-03-06T14:34:08Z
- **Completed:** 2026-03-06T14:46:08Z
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments
- Rewrote resolve_completed() to use Gamma API resolution metadata (closed, resolutionSource, umaResolutionStatus) instead of price convergence alone
- Added voided market detection with three heuristics: price ambiguity, resolution source keywords, UMA status
- Implemented compute_accuracy_snapshot() that persists cumulative Brier scores, win/loss/draw counts, and 30-day rolling window to PolymarketAccuracy table
- Reordered polymarket cycle to resolve -> forecast -> reforecast, eliminating the race condition where reforecasted probabilities could be scored instead of pre-resolution probabilities
- Replaced shotgun volume-threshold approach with top-10 active set model using fetch_top_geopolitical(limit=10)
- Added 429-specific handling with Retry-After backoff capped at 60s
- Exposed circuit_state and consecutive_failures properties for admin status

## Task Commits

Each task was committed atomically:

1. **Task 1: Resolution engine with voided detection + accuracy computation** - `6c5bc4c` (feat)
2. **Task 2: Reorder polymarket cycle + top-10 active set model** - `0c4ed17` (feat)

## Files Created/Modified
- `src/polymarket/comparison.py` - Enhanced resolve_completed() with voided detection, added _is_voided_market() and compute_accuracy_snapshot()
- `src/polymarket/client.py` - Added fetch_event_details(), 429 Retry-After handling, circuit_state/consecutive_failures properties
- `src/scheduler/heavy_runner.py` - Reordered cycle: match -> snapshot -> resolve -> forecast -> reforecast; top-10 model
- `src/polymarket/auto_forecaster.py` - Removed volume threshold filter from run(), added active_event_ids parameter to reforecast_active()

## Decisions Made
- Resolution uses Gamma API fields (closed, resolutionSource, umaResolutionStatus) rather than price convergence -- more reliable for detecting voided/cancelled markets
- Voided detection uses three independent heuristics (any match triggers void): ambiguous prices near [0.5, 0.5], void/cancel/invalid keywords in resolutionSource, void/resolved_too_early in umaResolutionStatus
- compute_accuracy_snapshot() called only when comparisons are actually resolved (not on voided-only batches) -- accuracy metrics should only update when there's new scoring data
- polymarket_volume_threshold setting kept in Settings for backward compatibility even though run() no longer uses it
- reforecast_active() accepts optional active_event_ids set for top-10 scoping; None value maintains backward compatibility

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Resolution engine and accuracy computation fully wired into the polymarket cycle
- Circuit breaker state properties ready for admin API exposure in Phase 22-03
- PolymarketAccuracy rows will accumulate as markets resolve, providing data for admin dashboard accuracy curves
- resolve_completed() returns dict (not int) -- any callers expecting int return type should be updated

---
*Phase: 22-polymarket-hardening*
*Completed: 2026-03-06*
