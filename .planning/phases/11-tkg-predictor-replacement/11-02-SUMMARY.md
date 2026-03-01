---
phase: 11-tkg-predictor-replacement
plan: 02
subsystem: training
tags: [tirgn, training-loop, tensorboard, wandb, early-stopping, vram, mrr, comparison]

# Dependency graph
requires:
  - phase: 11-tkg-predictor-replacement (plan 01)
    provides: TiRGN model module (tirgn_jax.py), create_tirgn_model factory, build_history_vocabulary, TKGModelProtocol compliance
  - phase: 05-tkg-training
    provides: RE-GCN training infrastructure (train_jax.py -- load_gdelt_data, create_graph_snapshots, negative_sampling, compute_mrr, save_checkpoint)
provides:
  - TiRGN training loop with data loading, negative sampling, evaluation, checkpointing, early stopping
  - TrainingLogger abstraction (TensorBoard always-on, W&B optional)
  - compare_models() infrastructure for TiRGN vs RE-GCN MRR evaluation
  - CLI entry point scripts/train_tirgn.py with all hyperparameters
  - TiRGN-specific checkpoint format with model_type discriminator
affects: [11-tkg-predictor-replacement plan 03 (integration + backend dispatch), 13-calibration]

# Tech tracking
tech-stack:
  added: [tensorboardX>=2.6, wandb>=0.16 (optional)]
  patterns: [TrainingLogger abstraction, TiRGN checkpoint format with model_type discriminator, history vocabulary pre-computation, _evaluate_tirgn adapter for fused distribution MRR]

key-files:
  created:
    - src/training/training_logger.py
    - src/training/train_tirgn.py
    - scripts/train_tirgn.py
    - src/training/compare_models.py
    - tests/test_train_tirgn.py
  modified:
    - pyproject.toml

key-decisions:
  - "tensorboardX over torch.utils.tensorboard -- no PyTorch dependency, pure-Python TensorBoard writer"
  - "wandb as optional[observability] extra, not core dependency -- never crashes training if missing"
  - "TiRGN checkpoint JSON includes model_type: 'tirgn' discriminator for downstream model loading"
  - "compare_models loads GDELT data ONCE, both models evaluated on same val_triples (no re-splitting)"
  - "ComparisonResult pass_threshold defaults to -5.0% -- TiRGN ships if within 5% of RE-GCN MRR"
  - "_evaluate_tirgn uses model._compute_fused_distribution directly, evolves embeddings once for all triples"
  - "Early stopping tracks epochs_without_improvement, incremented by eval_interval per non-improving evaluation"

patterns-established:
  - "TrainingLogger: context-manager pattern for multi-backend metrics logging (TensorBoard + optional W&B)"
  - "save_tirgn_checkpoint: .npz + .json sidecar with model_type discriminator and TiRGN-specific config"
  - "Metric naming: train/*, eval/*, system/* namespaces for TensorBoard organization"

# Metrics
duration: 6min
completed: 2026-03-01
---

# Phase 11 Plan 02: TiRGN Training Pipeline Summary

**TiRGN training loop with TensorBoard/W&B observability, early stopping on validation MRR, VRAM monitoring, and RE-GCN baseline comparison report**

## Performance

- **Duration:** 6 min
- **Started:** 2026-03-01T17:40:58Z
- **Completed:** 2026-03-01T17:47:08Z
- **Tasks:** 3
- **Files modified:** 6

## Accomplishments

- TrainingLogger abstraction writes to TensorBoard always, W&B when WANDB_API_KEY is set, never crashes on missing W&B
- TiRGN training loop reuses load_gdelt_data, create_graph_snapshots, negative_sampling from train_jax.py (zero duplication)
- compare_models() evaluates both TiRGN and RE-GCN on identical held-out split, produces comparison_report.json with pass/fail status
- 8 unit tests pass without GPU, covering config, early stopping, history vocab, checkpoints, logger, and comparison logic
- CLI entry point exposes all TiRGN-specific hyperparameters (--history-rate, --history-window, --patience)

## Task Commits

Each task was committed atomically:

1. **Task 1: Dependencies and TrainingLogger abstraction** - `191e6a0` (feat)
2. **Task 2: TiRGN training loop with early stopping and VRAM monitoring** - `d37fdf8` (feat)
3. **Task 3: RE-GCN baseline evaluation and MRR comparison report** - `ffbe5bd` (feat)

## Files Created/Modified

- `pyproject.toml` - Added tensorboardX (core) and wandb (optional) dependencies
- `src/training/training_logger.py` - TrainingLogger: TensorBoard always-on, W&B optional, context manager support
- `src/training/train_tirgn.py` - TiRGN training loop: data loading, NLL loss, early stopping, VRAM monitoring, checkpointing
- `scripts/train_tirgn.py` - CLI entry point with argparse for all hyperparameters
- `src/training/compare_models.py` - compare_models(): dual evaluation on shared held-out split, comparison_report.json output
- `tests/test_train_tirgn.py` - 8 unit tests for training infrastructure and comparison logic

## Decisions Made

- **tensorboardX over torch.utils.tensorboard**: No PyTorch dependency. Pure-Python TensorBoard writer aligns with our JAX-only training stack.
- **wandb as optional extra**: Added under `[project.optional-dependencies].observability`. Core training never depends on it. All wandb calls wrapped in try/except.
- **model_type discriminator in checkpoint JSON**: Enables downstream code (Plan 03 integration) to dispatch on model type when loading checkpoints.
- **Single data load for comparison**: `compare_models()` loads GDELT data once via `load_gdelt_data()`. Both models are evaluated on the same `val_triples` array. No re-splitting allowed.
- **-5.0% pass threshold**: Per CONTEXT.md failure strategy, TiRGN ships if MRR is within 5% of RE-GCN. `ComparisonResult.passed = delta_pct >= pass_threshold`.
- **_evaluate_tirgn adapter**: Calls `model._compute_fused_distribution` directly with pre-evolved entity embeddings, avoiding per-triple embedding evolution. More efficient than the RE-GCN `compute_mrr` path.
- **Early stopping increments by eval_interval**: `epochs_without_improvement` grows by `config.eval_interval` per non-improving evaluation round, not by 1. This means patience is measured in epochs, not evaluation rounds.

## Deviations from Plan

None -- plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

**Optional**: If W&B cloud logging is desired, set `WANDB_API_KEY` environment variable. TensorBoard works without it.

## Next Phase Readiness

- Training loop is runnable -- `scripts/train_tirgn.py` trains TiRGN to completion on GDELT data
- compare_models() ready for Plan 03 integration into the backend model dispatch logic
- Requirements TKG-03 (MRR comparison) and TKG-04 (GPU envelope with mixed precision + gradient checkpointing) are structurally covered
- train_jax.py was NOT modified (only imported from)
- 17 total TiRGN-related tests passing (9 model + 8 training)

---
*Phase: 11-tkg-predictor-replacement*
*Completed: 2026-03-01*
