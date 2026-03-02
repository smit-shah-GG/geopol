---
phase: 13-calibration-monitoring-hardening
plan: 02
subsystem: calibration
tags: [scipy, l-bfgs-b, brier-score, cameo, ensemble-weights, caching]

requires:
  - phase: 13-01
    provides: CalibrationWeight + CalibrationWeightHistory ORM models, Settings calibration fields
  - phase: 09-api-foundation
    provides: Prediction + OutcomeRecord ORM models, ensemble_info_json schema
provides:
  - CAMEO super-category taxonomy mapping (20 root codes -> 4 quadrants)
  - Literature-derived cold-start alpha priors for ensemble bootstrap
  - L-BFGS-B Brier score optimizer producing per-CAMEO alpha weights
  - Weekly calibration pipeline with deviation guardrails and history audit
  - Hierarchical weight loader with 5-level fallback and TTL cache
affects:
  - 13-06 (pipeline scheduling hooks will invoke WeightOptimizer.run_weekly_calibration)
  - 13-04 (Polymarket comparison may reference weight resolution levels)
  - future EnsemblePredictor integration (WeightLoader.resolve_alpha replaces fixed alpha=0.6)

tech-stack:
  added: []
  patterns:
    - "L-BFGS-B for bounded single-variable optimization (alpha in [0,1])"
    - "Hierarchical key resolution: specific -> category -> global -> cold-start"
    - "TTL-based in-memory cache for long-lived singleton loaders"
    - "Guardrail pattern: flag weights exceeding deviation threshold"

key-files:
  created:
    - src/calibration/priors.py
    - src/calibration/weight_optimizer.py
    - src/calibration/weight_loader.py
  modified: []

key-decisions:
  - "CAMEO quadrant mapping follows standard taxonomy: 01-05 verbal_coop, 06-09 material_coop, 10-14 verbal_conflict, 15-20 material_conflict"
  - "Cold-start priors are asymmetric: verbal_coop=0.65 (LLM-strong) to material_conflict=0.50 (TKG-strong)"
  - "Guardrails use relative deviation (not absolute) so a 0.50->0.60 change (20%) is treated differently than 0.90->1.00 (11%)"
  - "WeightLoader uses monotonic clock for TTL (immune to system clock adjustments)"
  - "Under-sampled CAMEO codes aggregate to super-category for optimization rather than being skipped entirely"

patterns-established:
  - "Hierarchical key namespace: CAMEO root codes, 'super:{category}', 'global' in calibration_weights"
  - "CalibrationResult partitions weights into applied/held for downstream audit"
  - "_CachedWeight slotted class for memory-efficient in-memory cache entries"

duration: 3min
completed: 2026-03-02
---

# Phase 13 Plan 02: Calibration Weight Recompute Summary

**L-BFGS-B alpha optimizer with CAMEO-hierarchical fallback, 20% deviation guardrails, and TTL-cached weight loader**

## Performance

- **Duration:** 3 min
- **Started:** 2026-03-02T06:19:09Z
- **Completed:** 2026-03-02T06:22:30Z
- **Tasks:** 3/3
- **Files created:** 3

## Accomplishments

- CAMEO root codes mapped to 4 super-categories with literature-derived cold-start alpha priors that replace naive alpha=0.6
- L-BFGS-B optimizer minimizes ensemble Brier score per-CAMEO with graceful fallback for under-sampled categories
- WeightLoader provides 5-level hierarchical resolution with 5-minute TTL cache for production singleton use
- Weekly calibration pipeline handles empty data, optimizer failures, and deviation guardrails without crashing

## Task Commits

Each task was committed atomically:

1. **Task 1: Create priors module** - `255f822` (feat)
2. **Task 2: Create weight optimizer** - `001de78` (feat)
3. **Task 3: Create weight loader** - `f569720` (feat)

## Files Created

- `src/calibration/priors.py` - CAMEO-to-super-category mapping, cold-start alpha priors, keyword bridge, infer_super_category helper
- `src/calibration/weight_optimizer.py` - optimize_alpha_for_category (L-BFGS-B), CalibrationResult dataclass, WeightOptimizer class with full weekly pipeline
- `src/calibration/weight_loader.py` - WeightLoader with hierarchical resolve_alpha, TTL cache, get_weight_info diagnostics

## Decisions Made

- **CAMEO quadrant taxonomy**: Standard mapping (01-05 verbal_coop through 15-20 material_conflict). Not controversial -- this is the authoritative CAMEO classification.
- **Asymmetric cold-start priors**: LLM weight ranges from 0.65 (verbal coop, where LLMs excel at diplomatic language analysis) to 0.50 (material conflict, where TKG structural patterns dominate). Global fallback 0.58.
- **Relative deviation guardrails**: 20% relative change threshold means a weight at 0.50 can move to 0.40-0.60 without flagging, while a weight at 0.90 can only move to 0.72-1.00. More conservative at extreme values.
- **Under-sampled aggregation**: CAMEO codes with fewer than min_samples pairs are aggregated to their super-category rather than skipped, maximizing data utilization.
- **Monotonic clock for TTL**: time.monotonic() is immune to NTP adjustments and system clock skew. The cache can never appear to go backwards.
- **ensemble_info_json parsing**: Handles both dict and JSON string forms defensively, since asyncpg may return either depending on driver configuration.

## Deviations from Plan

None -- plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None -- no external service configuration required.

## Next Phase Readiness

- Calibration subsystem is self-contained and ready for integration
- Plan 06 (pipeline hooks) will wire WeightOptimizer.run_weekly_calibration into the scheduler
- EnsemblePredictor needs to be updated to use WeightLoader.resolve_alpha instead of fixed alpha=0.6 (future integration task)
- No blockers

---
*Phase: 13-calibration-monitoring-hardening*
*Completed: 2026-03-02*
