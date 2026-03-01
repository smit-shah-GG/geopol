# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-27)

**Core value:** Explainability -- every forecast must provide clear, traceable reasoning paths
**Current focus:** Phase 9 - API Foundation & Infrastructure

## Current Position

Milestone: v2.0 Operationalization & Forecast Quality
Phase: 9 of 13 (API Foundation & Infrastructure)
Plan: 04 of 6 (in phase 9) -- 09-01, 09-03, 09-04 complete
Status: In progress
Last activity: 2026-03-01 -- Completed 09-04-PLAN.md (print() -> structured logging across 9 modules)

Progress: [####################....................] 50% (8/16 phases lifetime)
v2.0:    [#.........] ~10% (0/5 phases, plans 01+03+04 of 6 in phase 9 complete)

## Performance Metrics

**Velocity:**
- Total plans completed: 25
- Average duration: 18 minutes
- Total execution time: 7.4 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 01-data-foundation | 3 | 30min | 10min |
| 02-knowledge-graph | 3 | 180min | 60min |
| 03-hybrid-forecasting | 4 | 95min | 24min |
| 04-calibration-evaluation | 2 | 44min | 22min |
| 05-tkg-training | 4 | 64min | 16min |
| 06-networkx-fix | 1 | 2min | 2min |
| 07-bootstrap-pipeline | 2 | 12min | 6min |
| 08-graph-partitioning | 2 | 12min | 6min |
| 09-api-foundation | 3 | 17min | 6min |

**Recent Trend:**
- Last 3 plans: 09-03 (4min), 09-01 (6min), 09-04 (7min)
- Trend: Fast (infrastructure and mechanical sweeps)

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Key decisions affecting current work:

- JAX for TKG training -- jraph eliminated in 09-03, local GraphsTuple + jax.ops.segment_sum
- TKGModelProtocol defined -- @runtime_checkable, REGCNJraph + StubTiRGN verified (09-03)
- Fixed 60/40 alpha -- being replaced by per-CAMEO dynamic weights in v2.0
- Retain Gemini over local Llama -- frontier reasoning, negligible cost (2026-02-14)
- Micro-batch ingest (15-min GDELT), daily predict -- v2.0 architecture
- WM-derived TypeScript frontend over Streamlit -- vanilla TS + deck.gl + Panel system (2026-02-27)
- Headless API-first backend -- FastAPI + Pydantic DTOs as mandatory bridge (2026-02-27)
- PostgreSQL for forecast persistence -- SQLite retained only for GDELT events/partition index (2026-02-27)
- Contract-first parallel execution -- DTOs + mock fixtures in Phase 9 enable Phases 10/11/12 in parallel (2026-02-27)
- RSS feeds from WM's 298-domain list -> RAG enrichment in Phase 10 (2026-02-27)
- Polymarket comparison for calibration validation in Phase 13 (2026-02-27)
- extra="ignore" in pydantic-settings to coexist with legacy .env vars (2026-03-01, 09-01)
- DateTime(timezone=True) for all PostgreSQL timestamp columns (2026-03-01, 09-01)
- Prediction.id as String(36) UUID for cross-system stability (2026-03-01, 09-01)

### Deferred Issues

- Docker daemon requires sudo to start -- user must `sudo systemctl start docker` before running containers or Alembic migrations

### Quick Tasks Completed

| # | Description | Date | Commit | Directory |
|---|-------------|------|--------|-----------|
| 001 | Fix README + add forecast CLI + preflight check | 2026-02-12 | 03dfe49 | [001-fix-readme-add-forecast-cli-preflight](./quick/001-fix-readme-add-forecast-cli-preflight/) |

### Blockers/Concerns

- TiRGN JAX port has no published reference implementation (research-phase may be needed for Phase 11)
- Gemini API cost exposure under public traffic (rate limiting mandatory before deployment)
- jraph archived by Google DeepMind (RESOLVED: eliminated in 09-03)
- Polyglot tax: Python + TypeScript + Rust/Tauri -- three ecosystems for single developer
- Docker daemon not auto-started -- verification of PostgreSQL/Redis containers and Alembic migration deferred

## Session Continuity

Last session: 2026-03-01
Stopped at: Completed 09-04-PLAN.md
Resume file: None
Next: Execute 09-02-PLAN.md (Pydantic V2 DTOs and mock fixtures)
