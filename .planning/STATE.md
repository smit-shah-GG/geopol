# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-01-23)

**Core value:** Explainability — every forecast must provide clear, traceable reasoning paths
**Current focus:** v1.0 Shipped — Planning next milestone

## Current Position

Phase: v1.0 complete
Plan: N/A
Status: **MILESTONE SHIPPED**
Last activity: 2026-01-23 — v1.0 milestone archived

Progress: v1.0 ████████████████ 100% (5 phases, 16 plans)

## Performance Metrics

**Velocity:**
- Total plans completed: 16
- Average duration: 25 minutes
- Total execution time: 6.6 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 01-data-foundation | 3 | 30min | 10min |
| 02-knowledge-graph | 3 | 180min | 60min |
| 03-hybrid-forecasting | 4 | 95min | 24min |
| 04-calibration-evaluation | 2 | 44min | 22min |
| 05-tkg-training | 4 | 64min | 16min |

**Recent Trend:**
- Last 3 plans: 05-02 (9min), 05-03 (25min), 05-04 (18min)
- Trend: All phases complete

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Key decisions made throughout project:

- Python selected as implementation language
- Focus on conflicts and diplomatic events (QuadClass 1 & 4)
- Evaluation on recent 2023-2024 events
- Using uv for Python package management (user-specified)
- Using google-genai SDK instead of deprecated google-generativeai
- RE-GCN chosen over TiRGN for mature implementation (03-03)
- CPU-only PyTorch for production compatibility (03-03)
- 60/40 graph/RAG confidence weighting (03-03)
- Ensemble weights: 0.6 LLM, 0.4 TKG (configurable) (03-04)
- Temperature scaling for confidence calibration (03-04)
- CLI with JSON/text/summary output formats (03-04)
- SQLite for lightweight prediction tracking (04-01)
- Per-category calibration curves (conflict/diplomatic/economic) (04-01)
- Isotonic + temperature scaling complementary approach (04-01)
- Keyword-based category inference for simplicity (04-01)
- All QuadClasses (1-4) included for comprehensive pattern learning (05-01)
- Parquet format for efficient TKG data loading (05-01)
- Composite relation format: EventCode_QuadClass (05-01)
- JAX/jraph training for memory efficiency (05-03)
- Weekly retraining schedule (configurable to monthly) (05-04)
- Time-based retraining over performance-triggered (05-04)

### Deferred Issues

None.

### Blockers/Concerns

None.

### Roadmap Evolution

- Phase 5 added (2026-01-13): TKG predictor training with RE-GCN implementation and GDELT data pipeline
- Project completed (2026-01-23): All 5 phases delivered

## Session Continuity

Last session: 2026-01-23
Stopped at: v1.0 milestone shipped
Resume file: None
Next: `/gsd:new-milestone` when ready for v1.1

## Technical Debt (Carried to v1.1)

1. **UAT-005:** Fix NetworkX shortest_path API (minor) — use `single_source_shortest_path`
2. **Bootstrap script:** Production script for Phase 1 → Phase 2 → Phase 3 connection
3. **Scalability:** Graph partitioning for >1M events

## v1.0 Deliverables

See `.planning/MILESTONES.md` for full details.

**Key artifacts:**
- `forecast.py` — Main CLI for running forecasts
- `src/forecasting/` — Ensemble forecaster, TKG predictor, LLM integration
- `src/knowledge_graph/` — Knowledge graph construction and querying
- `src/training/` — Data collection, model training, retraining scheduler
- `models/tkg/regcn_trained.pt` — Trained TKG model checkpoint
- `config/retraining.yaml` — Retraining schedule configuration
