---
phase: 09-api-foundation
plan: 03
subsystem: infra
tags: [jax, jraph, protocol, logging, tkg, regcn]

# Dependency graph
requires:
  - phase: 05-tkg-training
    provides: RE-GCN model implementation (regcn_jraph.py, train_jraph.py)
provides:
  - jraph-free RE-GCN using local GraphsTuple + jax.ops.segment_sum
  - TKGModelProtocol (@runtime_checkable) for swappable TKG backends
  - StubTiRGN satisfying the protocol (placeholder for Phase 11)
  - Structured logging config (setup_logging) with human + JSON modes
affects: [11-tirgn, 09-04-logging-sweep, 10-forecast-quality]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Protocol-based model contracts via typing.Protocol + @runtime_checkable"
    - "Local NamedTuple replacing archived library types (GraphsTuple)"
    - "Idempotent logging setup with handler deduplication"

key-files:
  created:
    - src/protocols/__init__.py
    - src/protocols/tkg.py
    - src/logging_config.py
  modified:
    - src/training/models/regcn_jraph.py
    - src/training/train_jraph.py
    - scripts/train_tkg_jraph.py

key-decisions:
  - "jax.ops.segment_sum over manual scatter-add -- same XLA kernel as jraph.segment_sum, zero behavior change"
  - "Protocol uses **kwargs for implementation-specific args (rng_key) -- avoids forcing every backend to declare JAX-specific params"
  - "StubTiRGN returns zeros -- contract verification only, not a functional model"

patterns-established:
  - "TKGModelProtocol: any TKG model must expose evolve_embeddings, compute_scores, compute_loss"
  - "setup_logging(level, json_format): single entry point for all log configuration"

# Metrics
duration: 4min
completed: 2026-03-01
---

# Phase 9 Plan 3: jraph Elimination, TKGModelProtocol, and Structured Logging Summary

**Eliminated archived jraph dependency via local GraphsTuple + jax.ops.segment_sum, defined @runtime_checkable TKGModelProtocol verified against REGCNJraph and StubTiRGN, created setup_logging() with human-readable and JSON output modes**

## Performance

- **Duration:** 4 min
- **Started:** 2026-03-01T09:11:10Z
- **Completed:** 2026-03-01T09:15:31Z
- **Tasks:** 2
- **Files modified:** 6

## Accomplishments
- Zero `import jraph` statements remain in the codebase -- archived dependency fully excised
- RE-GCN model creation and forward pass verified identical behavior post-migration
- TKGModelProtocol defined with @runtime_checkable; both REGCNJraph and StubTiRGN pass isinstance()
- Structured logging with idempotent handler management, human-readable and JSON modes, ValueError on invalid levels

## Task Commits

Each task was committed atomically:

1. **Task 1: Eliminate jraph -- replace with local JAX equivalents** - `3ac5bb0` (refactor)
2. **Task 2: TKGModelProtocol + stub TiRGN + structured logging config** - `9774bbb` (feat)

## Files Created/Modified
- `src/training/models/regcn_jraph.py` - Replaced jraph.GraphsTuple with local NamedTuple, jraph.segment_sum with jax.ops.segment_sum
- `src/training/train_jraph.py` - Updated docstrings to reflect jraph-free architecture
- `scripts/train_tkg_jraph.py` - Updated docstrings and description
- `src/protocols/__init__.py` - New package for protocol definitions
- `src/protocols/tkg.py` - TKGModelProtocol + StubTiRGN
- `src/logging_config.py` - setup_logging() with human/JSON modes

## Decisions Made
- Used `jax.ops.segment_sum` as the direct replacement for `jraph.segment_sum` -- identical XLA kernel, no behavior change, no additional dependencies
- Protocol method signatures use `**kwargs` for implementation-specific parameters (e.g. `rng_key`) rather than forcing all backends to declare JAX-specific arguments
- StubTiRGN returns zero tensors -- its sole purpose is protocol contract verification, not functional modeling
- Logging uses `sys.stderr` (not stdout) to avoid polluting data pipelines

## Deviations from Plan

None -- plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None -- no external service configuration required.

## Next Phase Readiness
- Plan 04 (print->logging sweep) can proceed: `setup_logging()` is ready to import
- Phase 11 (TiRGN JAX port) has its protocol contract defined: implement TKGModelProtocol
- jraph can be removed from pyproject.toml dependencies (if still listed) in a future cleanup task

---
*Phase: 09-api-foundation*
*Completed: 2026-03-01*
