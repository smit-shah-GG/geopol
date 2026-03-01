---
phase: 11-tkg-predictor-replacement
plan: 01
subsystem: tkg-model
tags: [jax, flax-nnx, tirgn, temporal-knowledge-graph, convtranse, copy-generation, mixed-precision, bfloat16]

# Dependency graph
requires:
  - phase: 09-api-foundation
    provides: TKGModelProtocol, RelationalGraphConv, GRUCell, GraphSnapshot
provides:
  - TiRGN nnx.Module (full TiRGN architecture in JAX/Flax NNX)
  - GlobalHistoryEncoder (sparse binary vocabulary + history-constrained scoring)
  - TimeConvTransEDecoder (4-channel temporal ConvTransE)
  - build_history_vocabulary() preprocessing function
  - create_tirgn_model() factory function
affects: [11-02 (training loop), 11-03 (integration + backend dispatch)]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Sparse history vocabulary via dict-of-keys (avoid dense E*R,E matrix)"
    - "Copy-generation fusion: linear interpolation of raw + history softmax distributions"
    - "NLL loss over fused distribution (not margin ranking loss)"
    - "Mixed precision: bfloat16 compute via nnx.Conv/nnx.Linear dtype param, float32 master weights"
    - "Modern Flax NNX param access via [...] instead of deprecated .value"

key-files:
  created:
    - src/training/models/components/__init__.py
    - src/training/models/components/global_history.py
    - src/training/models/components/time_conv_transe.py
    - src/training/models/tirgn_jax.py
    - tests/test_tirgn_model.py
  modified: []

key-decisions:
  - "Sparse history vocab (dict[tuple[int,int], set[int]]) instead of dense (E*R, E) matrix -- 28GB saved for GDELT scale"
  - "history_rate is a fixed hyperparameter (default 0.3), not a learned parameter -- per CONTEXT.md"
  - "History vocab passed via kwargs to compute_loss, not stored in model -- training loop owns data lifecycle"
  - "Relation GRU uses projection layer + existing GRUCell instead of modifying shared GRUCell class"
  - "neg_triples and margin accepted but ignored in compute_loss for protocol compatibility"
  - "Used modern param[...] access pattern instead of deprecated .value in new code"

patterns-established:
  - "TiRGN component decomposition: components/ subpackage for reusable decoder/encoder modules"
  - "History mask construction: get_history_mask() builds per-batch (batch, E) boolean from sparse vocab"
  - "Protocol compliance: TiRGN satisfies same TKGModelProtocol as REGCN despite different loss semantics"

# Metrics
duration: 7min
completed: 2026-03-01
---

# Phase 11 Plan 01: TiRGN Model Architecture Summary

**TiRGN nnx.Module with global history encoder, Time-ConvTransE decoder, copy-generation fusion, sparse history vocab, and bfloat16 mixed precision -- 9 unit tests passing, TKGModelProtocol compliant**

## Performance

- **Duration:** 7 min
- **Started:** 2026-03-01T17:28:59Z
- **Completed:** 2026-03-01T17:36:07Z
- **Tasks:** 2/2
- **Files created:** 5

## Accomplishments

- Full TiRGN architecture (Li et al., IJCAI 2022) implemented as Flax NNX module with all subcomponents
- Sparse history vocabulary avoids 28GB dense matrix via dictionary-of-keys representation
- Mixed precision support (bfloat16 compute, float32 master weights) built into Time-ConvTransE decoder
- TKGModelProtocol satisfied at runtime -- drop-in replacement for REGCN in the forecasting pipeline
- 9 comprehensive unit tests covering shapes, loss, protocol, components, fusion, and precision

## Task Commits

Each task was committed atomically:

1. **Task 1: Global history encoder and Time-ConvTransE decoder components** - `0901903` (feat)
2. **Task 2: TiRGN model module with protocol compliance and unit tests** - `753fef6` (feat)

## Files Created/Modified

- `src/training/models/components/__init__.py` - Package exports for GlobalHistoryEncoder, TimeConvTransEDecoder
- `src/training/models/components/global_history.py` - Sparse history vocabulary construction, per-batch mask generation, GlobalHistoryEncoder nnx.Module
- `src/training/models/components/time_conv_transe.py` - 4-channel Time-ConvTransE decoder with learned periodic/non-periodic time embeddings, nnx.Conv, nnx.BatchNorm
- `src/training/models/tirgn_jax.py` - TiRGN nnx.Module composing R-GCN, entity GRU, relation GRU, raw decoder, global history encoder, copy-generation fusion; factory function
- `tests/test_tirgn_model.py` - 9 unit tests for forward shapes, loss, protocol compliance, attributes, component standalone, history mask, fusion distribution, mixed precision

## Decisions Made

- **Sparse history vocab**: Dictionary-of-keys `dict[(s, r)] -> set[o]` instead of dense `(E*R, E)` boolean matrix. For GDELT scale (~500K entities, ~300 relations), the dense matrix would require ~28GB. The sparse representation only stores non-empty entries and constructs per-batch masks on demand.
- **History vocab not stored in model**: Passed via `**kwargs` in `compute_loss()` (key: `history_vocab`). The training loop is responsible for building and passing it. This keeps the model stateless w.r.t. data preprocessing.
- **Relation GRU via projection + existing GRUCell**: The reference implementation uses `nn.GRUCell(h_dim*2, h_dim)`. Instead of modifying the shared GRUCell (used by both REGCN and TiRGN), added a `nnx.Linear` projection layer before the existing GRUCell. Same representational capacity, zero impact on existing code.
- **Modern Flax NNX param access**: Used `param[...]` in new code instead of deprecated `.value`. The existing regcn_jax.py still uses `.value` (out of scope for this plan).
- **Fixed history_rate=0.3 default**: Per CONTEXT.md, this is a configurable hyperparameter, not a learned parameter. Tuning happens in Plan 02 (training loop).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] gitignore `models/` pattern blocks new files**
- **Found during:** Task 1 (staging components for commit)
- **Issue:** `.gitignore` has `models/` pattern that matches `src/training/models/components/`. Existing model files were previously added with `-f`.
- **Fix:** Used `git add -f` for new files in the components directory, matching the existing pattern.
- **Files affected:** All files in `src/training/models/components/` and `src/training/models/tirgn_jax.py`
- **Verification:** Files tracked in git.
- **Committed in:** 0901903, 753fef6

**2. [Rule 1 - Bug] Fixed Flax NNX `.value` deprecation warnings in new code**
- **Found during:** Task 2 (running tests showed deprecation warnings from new code)
- **Issue:** Using `.value` on `nnx.Param` is deprecated in Flax 0.12+. Should use `[...]` or `.get_value()`.
- **Fix:** Replaced all `.value` accesses in time_conv_transe.py and tirgn_jax.py with `[...]` syntax.
- **Files modified:** `src/training/models/components/time_conv_transe.py`, `src/training/models/tirgn_jax.py`, `tests/test_tirgn_model.py`
- **Verification:** Warnings from new code eliminated (remaining warnings are from existing regcn_jax.py).
- **Committed in:** 753fef6

---

**Total deviations:** 2 auto-fixed (1 blocking, 1 bug)
**Impact on plan:** Both fixes necessary for clean commits and forward compatibility. No scope creep.

## Issues Encountered

None -- plan executed smoothly. All shapes, protocol compliance, and loss computation verified on first pass.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- TiRGN model architecture is complete and tested
- Ready for Plan 02: training loop with TensorBoard/W&B observability, GDELT data loading, mixed precision training step
- Ready for Plan 03: backend dispatch (TKG_BACKEND envvar), retraining scheduler integration, evaluation pipeline
- Requirements TKG-01 (model architecture) and TKG-02 (protocol compliance) are covered

---
*Phase: 11-tkg-predictor-replacement*
*Completed: 2026-03-01*
