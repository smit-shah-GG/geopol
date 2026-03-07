---
phase: 23
plan: 01
subsystem: backtesting
tags: [backtesting, walk-forward, brier-score, chromadb, calibration, postgresql, alembic]
dependency-graph:
  requires: [phase-22-polymarket-hardening, phase-13-calibration]
  provides: [backtesting-engine, backtest-orm-models, backtest-migration-009]
  affects: [phase-23-plan-02-api, phase-23-plan-03-admin-panel]
tech-stack:
  added: []
  patterns: [walk-forward-evaluation, temporal-chromadb-index, calibration-weight-snapshot, db-based-cancellation]
key-files:
  created:
    - src/backtesting/__init__.py
    - src/backtesting/schemas.py
    - src/backtesting/evaluator.py
    - src/backtesting/temporal_index.py
    - src/backtesting/weight_snapshot.py
    - src/backtesting/runner.py
    - src/backtesting/export.py
    - alembic/versions/20260307_009_backtest_tables.py
  modified:
    - src/db/models.py
decisions:
  - DB-based cancellation (not threading.Event) -- process-safe, survives restarts
  - Python-side date filtering for ChromaDB (not $lte string comparison) -- handles mixed date formats
  - Naive datetime comparison for temporal index cutoff -- avoids timezone mismatch between sources
  - MRR left as None in window results -- TKG ranking data unavailable from re-prediction path
metrics:
  duration: 13min
  completed: 2026-03-07
---

# Phase 23 Plan 01: Backtesting Engine Core Summary

Walk-forward evaluation engine with PostgreSQL schema, temporal bias prevention, metric computation, and investor-grade CSV/JSON export.

## What Was Done

### Task 1: Database Schema + ORM Models + Alembic Migration (33e44da)

Added BacktestRun and BacktestResult ORM models to `src/db/models.py` following existing patterns. BacktestRun tracks run lifecycle (pending -> running -> completed/cancelled/failed), configuration (window size, slide step, checkpoints), progress counters, and aggregate metrics. BacktestResult stores per-window metrics (Brier, MRR, Hits@k), calibration bins as JSON, per-prediction details, Polymarket head-to-head data, and weight snapshots. Composite index on (run_id, window_start) for efficient drill-down queries.

Created Alembic migration 009 with correct revision chain (down_revision = "008").

Created `src/backtesting/schemas.py` with three dataclasses:
- **BacktestRunConfig**: Serializes to/from JSON for ProcessPoolExecutor transport (no unpickleable objects). Fields: label, checkpoints, window_size_days (14), slide_step_days (7), min_predictions_per_window (3), description, run_id.
- **WindowResult**: Per-window metric container including Brier, calibration bins, prediction details, Polymarket comparison, and weight snapshot.
- **BacktestRunResult**: Aggregate run result with status, window list, and Polymarket record.

### Task 2: Backtesting Engine Modules (4baf926)

**evaluator.py** -- Stateless metric computation:
- `compute_brier_score()`: np.mean((p - o)^2), validated at 0.01 for [0.9, 0.1] vs [1.0, 0.0].
- `compute_calibration_bins()`: 10-bin histogram with None for empty bins, handles edge cases (last bin includes 1.0).
- `compute_hit_rate()`: Binary classification at threshold 0.5.
- `compute_mrr()`: Mean reciprocal rank, returns 0.0 for empty input.

**temporal_index.py** -- Ephemeral ChromaDB index builder:
- Opens persistent source read-only, creates in-memory ephemeral client via `chromadb.Client()`.
- Paginated fetch (BATCH_SIZE=500) of ALL chunks, Python-side date filtering via `datetime.fromisoformat()` (not ChromaDB string $lte).
- Excludes chunks with empty/missing/unparseable published_at metadata.
- Handles ISO 8601 variants: "Z" suffix, "+00:00", date-only.
- No disk caching -- each call builds fresh.

**weight_snapshot.py** -- Calibration weight time-travel:
- Queries `calibration_weight_history` for latest auto_applied entry per cameo_code where computed_at <= as_of.
- Falls back to cold-start priors from `src/calibration/priors.py` for codes with no history.
- Returns complete dict covering all CAMEO codes (01-20), super-categories, and global.

**runner.py** (373 lines) -- Central BacktestRunner:
- Walk-forward loop: generates sliding windows from prediction date range, iterates per (window, checkpoint) pair.
- DB-based cancellation: polls backtest_runs.status between windows (not threading.Event -- process-safe).
- Per-window: snapshots calibration weights, builds ephemeral ChromaDB index, queries resolved predictions, re-predicts each via fresh EnsemblePredictor.
- Caches heavy components (RAG, TKG, orchestrator) across predictions within a window.
- Persists WindowResult as BacktestResult row after each window.
- Computes aggregate Brier, MRR, and Polymarket head-to-head record on completion.
- Handles partial results on cancellation.

**export.py** -- CSV/JSON export:
- `export_run_json()`: Run metadata + all window results + methodology section.
- `export_run_csv()`: Methodology as comment block + CSV headers + per-window rows.
- `export_multi_run_json()`: Merges multiple runs with labels as group identifiers.
- `METHODOLOGY_TEMPLATE` (3060 chars): Documents walk-forward methodology, bias prevention (weight snapshots + temporal indexes + fresh predictors), metric definitions (Brier formula, calibration bins, hit rate, MRR), inclusion criteria, and data sources.

## Decisions Made

| Decision | Rationale |
|----------|-----------|
| DB-based cancellation instead of threading.Event | ProcessPoolExecutor runs in separate process; DB status polling is process-safe and survives restarts |
| Python-side date filtering for temporal index | ChromaDB $lte on strings breaks with mixed date formats (YYYY-MM-DD vs YYYY-MM-DDTHH:MM:SSZ) across RSS feeds |
| Naive datetime comparison for cutoff | Mixed timezone metadata; stripping tzinfo for comparison avoids false exclusions |
| MRR = None in window results | TKG ranking data not available from the re-prediction code path; MRR requires raw link prediction rankings |

## Deviations from Plan

None -- plan executed exactly as written.

## Verification Results

- All 7 modules in `src/backtesting/` import cleanly
- BacktestRunConfig JSON round-trip verified
- Evaluator produces correct Brier score (0.0100 for [0.9, 0.1] vs [1.0, 0.0])
- ORM models register in SQLAlchemy metadata
- Alembic migration has correct revision chain (009 -> 008)
- METHODOLOGY_TEMPLATE is 3060 chars (requirement: >200)
- 287 existing tests pass (41 pre-existing failures: CUDA OOM, Docker, JAX/jraph)

## Next Phase Readiness

Plan 23-02 (API + scheduler wiring) can proceed. The engine is fully importable and the BacktestRunner.run() interface is stable. The API layer will:
1. Create BacktestRun rows and call run_backtest() via ProcessPoolExecutor.
2. Expose CRUD endpoints for run management.
3. Wire cancellation via status='cancelling' update.
4. Serve export endpoints using export_run_json/csv.
