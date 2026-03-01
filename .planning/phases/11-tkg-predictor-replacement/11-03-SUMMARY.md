---
phase: 11-tkg-predictor-replacement
plan: 03
subsystem: forecasting
tags: [tkg, tirgn, regcn, backend-dispatch, scheduler, jax, flax-nnx]

# Dependency graph
requires:
  - phase: 11-01
    provides: TiRGN model module (create_tirgn_model, TKGModelProtocol)
  - phase: 11-02
    provides: TiRGN training pipeline (train_tirgn, save_tirgn_checkpoint, TiRGNTrainingConfig)
provides:
  - Config-only backend swap via TKG_BACKEND envvar
  - TKGPredictor backend dispatch (TiRGN or RE-GCN)
  - Model-agnostic RetrainingScheduler
  - TiRGN checkpoint loading with model_type validation
  - 16 integration tests for dispatch chain
affects: [13-calibration, api-routes, daily-pipeline]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Backend dispatch via Settings.tkg_backend property read at init"
    - "JSON sidecar model_type discriminator for checkpoint cross-load prevention"
    - "Scheduler _train_tirgn/_train_regcn delegation pattern"

key-files:
  created:
    - tests/test_tkg_backend_dispatch.py
  modified:
    - src/settings.py
    - src/forecasting/tkg_predictor.py
    - src/training/scheduler.py
    - config/retraining.yaml
    - scripts/retrain_tkg.py

key-decisions:
  - "TKG_BACKEND read once at TKGPredictor.__init__; requires process restart to switch"
  - "TiRGN mode sets self.model = None (no REGCNWrapper), uses self._tirgn_model"
  - "TiRGN checkpoint restore: nnx.split -> flatten leaves -> map npz keys -> unflatten -> nnx.merge"
  - "TiRGN predict_object uses raw_decoder directly (not fused distribution) for speed"
  - "Scheduler config uses model_tirgn section alongside existing model section"
  - "retrain_tkg.py --backend override resets Settings singleton to pick up new envvar"

patterns-established:
  - "Backend dispatch: Settings.tkg_backend -> conditional init in TKGPredictor"
  - "Checkpoint model_type validation: JSON sidecar must match configured backend"
  - "Model-agnostic scheduler: _train_new_model delegates to backend-specific method"

# Metrics
duration: 8min
completed: 2026-03-01
---

# Phase 11 Plan 03: Integration + Backend Dispatch Summary

**Config-only TiRGN/RE-GCN swap via TKG_BACKEND envvar with model-agnostic scheduler and 16 integration tests**

## Performance

- **Duration:** 8 min
- **Started:** 2026-03-01T17:51:23Z
- **Completed:** 2026-03-01T17:58:53Z
- **Tasks:** 2
- **Files modified:** 6

## Accomplishments
- TKGPredictor dispatches to TiRGN or RE-GCN based on TKG_BACKEND envvar with zero changes to EnsemblePredictor or API routes
- TiRGN checkpoint loading validates model_type discriminator; cross-loading (tirgn checkpoint with regcn backend) logs error and falls back
- RetrainingScheduler dispatches to train_tirgn or train_regcn, with backend-aware backup/restore/validation
- scripts/retrain_tkg.py accepts --backend override for one-off retraining runs
- 16 integration tests verify full dispatch chain end-to-end

## Task Commits

Each task was committed atomically:

1. **Task 1: Settings, TKGPredictor backend dispatch, and checkpoint loading** - `557c3ab` (feat)
2. **Task 2: Model-agnostic scheduler, retraining config, and integration tests** - `d503dd0` (feat)

## Files Created/Modified
- `src/settings.py` - Added tkg_backend: Literal["tirgn", "regcn"] = "regcn"
- `src/forecasting/tkg_predictor.py` - Backend dispatch in __init__, TiRGN checkpoint loading, TiRGN prediction methods
- `src/training/scheduler.py` - Model-agnostic _train_new_model, backend-aware backup/validate/cleanup
- `config/retraining.yaml` - Added model_tirgn section with TiRGN hyperparameters
- `scripts/retrain_tkg.py` - Added --backend CLI argument with settings singleton reset
- `tests/test_tkg_backend_dispatch.py` - 16 integration tests (414 lines)

## Decisions Made
- TKG_BACKEND read once at TKGPredictor.__init__; requires process restart to switch (per CONTEXT.md)
- TiRGN mode sets self.model = None (no REGCNWrapper instantiation) -- saves memory
- TiRGN checkpoint restore uses nnx.split/unflatten/merge pattern (same as train_jax.py save path)
- TiRGN predict_object uses raw_decoder directly for speed (not fused distribution, which needs history vocab)
- Scheduler uses model_tirgn config section alongside existing model section (no breaking changes)
- --backend CLI override resets Settings singleton before scheduler initialization

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Config versioning.model_dir overrides constructor model_dir in tests**
- **Found during:** Task 2 (integration test test_tirgn_backup)
- **Issue:** RetrainingScheduler.__init__ applies config overrides from YAML after setting self.model_dir from constructor, so the real config/retraining.yaml's versioning.model_dir clobbered the test's tmp_path model_dir
- **Fix:** Tests write a minimal config YAML to tmp_path that points versioning.model_dir to the test's model_dir
- **Files modified:** tests/test_tkg_backend_dispatch.py
- **Verification:** test_tirgn_backup passes
- **Committed in:** d503dd0 (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (1 bug)
**Impact on plan:** Test infrastructure fix only. No production code affected.

## Issues Encountered
None.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Phase 11 is complete. All 3 plans delivered:
  - 11-01: TiRGN model architecture (490 lines)
  - 11-02: TiRGN training pipeline with observability (557 lines)
  - 11-03: Integration + backend dispatch (config-only swap)
- TiRGN is fully wired into production pipeline behind TKG_BACKEND envvar
- RE-GCN remains default (zero regressions)
- Ready for parallel Phase 12 (or Phase 13 once outcome data accumulates)

---
*Phase: 11-tkg-predictor-replacement*
*Completed: 2026-03-01*
