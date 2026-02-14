# Requirements: Geopol

**Core Value:** Explainability — every forecast must provide clear, traceable reasoning paths

## Delivered Requirements

### v1.0 MVP (shipped 2026-01-23)

- [x] Event data ingestion from GDELT API with custom enrichment pipeline
- [x] Temporal knowledge graph construction from event streams
- [x] Hybrid model ensemble combining TKG algorithms (RE-GCN/TiRGN) with LLM reasoning
- [x] Explainable reasoning chain generation for each prediction
- [x] Probability calibration system with Brier score optimization
- [x] Evaluation framework against historical events

### v1.1 Tech Debt (shipped 2026-01-30)

- [x] Fix NetworkX shortest_path API to use single_source_shortest_path
- [x] Production bootstrap script connecting data ingestion -> graph build -> RAG indexing
- [x] Graph partitioning for scalability beyond 1M events

## v2.0 Requirements

**Defined:** 2026-02-14
**Core Value:** Explainability — every forecast must provide clear, traceable reasoning paths
**Milestone Goal:** Transform research prototype into publicly demonstrable system with automated operations, upgraded TKG predictor, and self-improving calibration

### Infrastructure & Persistence

- [ ] **INFRA-01**: System persists all forecasts (question, prediction, probability, reasoning chain, timestamp) in SQLite `predictions` table
- [ ] **INFRA-02**: System persists outcome records comparing past predictions against GDELT ground truth in SQLite `outcome_records` table
- [ ] **INFRA-03**: System persists per-CAMEO calibration weights in SQLite `calibration_weights` table
- [ ] **INFRA-04**: System tracks micro-batch ingest runs in SQLite `ingest_runs` table with status, event counts, and timestamps
- [ ] **INFRA-05**: System eliminates archived jraph dependency, replacing with raw JAX ops (local NamedTuple for GraphsTuple, `jax.ops.segment_sum`)
- [ ] **INFRA-06**: System runs Streamlit, ingest daemon, and prediction pipeline as separate OS processes communicating exclusively through SQLite
- [ ] **INFRA-07**: System uses structured logging (Python `logging` module) replacing all `print()` statements in production code paths
- [ ] **INFRA-08**: SQLite connections use WAL mode with `busy_timeout=30000ms`; Streamlit uses read-only connections (`PRAGMA query_only = ON`)

### Web Frontend

- [ ] **WEB-01**: Streamlit app displays current forecasts with full reasoning chains on the main page
- [ ] **WEB-02**: Streamlit app displays historical track record: past predictions vs outcomes with Brier scores
- [ ] **WEB-03**: Streamlit app displays calibration plots (reliability diagrams, per-category accuracy)
- [ ] **WEB-04**: Streamlit app provides interactive query input where visitor submits a geopolitical question and receives a live Gemini+TKG forecast with reasoning chain
- [ ] **WEB-05**: Streamlit app enforces per-IP rate limiting (max 3 interactive queries per hour) via slowapi middleware
- [ ] **WEB-06**: Streamlit app manages session memory with explicit cleanup to prevent OOM under sustained public traffic
- [ ] **WEB-07**: Streamlit app includes methodology page explaining the hybrid LLM+TKG approach, data sources, and calibration method
- [ ] **WEB-08**: Streamlit app sanitizes all user query input to prevent prompt injection and leakage of system internals (API keys, file paths, model config)
- [ ] **WEB-09**: Streamlit app enforces Gemini API budget cap with usage tracking; queries rejected when daily budget exceeded

### Micro-batch Ingest

- [ ] **INGEST-01**: System polls GDELT 15-minute update feed (`lastupdate.txt`) on a configurable schedule via daemon process
- [ ] **INGEST-02**: System performs incremental graph update appending only new events, avoiding full graph rebuild (O(N_new) not O(N_total))
- [ ] **INGEST-03**: System deduplicates events across micro-batches and daily dumps using GDELT `GlobalEventID`
- [ ] **INGEST-04**: System handles missing/late GDELT feed updates gracefully with exponential backoff and logged warnings
- [ ] **INGEST-05**: Ingest daemon tracks per-run metrics (events fetched, events new, events duplicate, duration) in `ingest_runs` table

### Daily Forecast Automation

- [ ] **AUTO-01**: System runs daily forecast pipeline on a configurable schedule (default 06:00) via systemd timer
- [ ] **AUTO-02**: Daily pipeline generates forecast questions from recent high-significance events in the knowledge graph
- [ ] **AUTO-03**: Daily pipeline runs full Gemini+TKG ensemble prediction for each generated question and persists results
- [ ] **AUTO-04**: Daily pipeline resolves outcomes for past predictions by comparing against subsequent GDELT events within a configurable time window
- [ ] **AUTO-05**: Daily pipeline handles failures with retry logic and sends alerts on consecutive failures

### Dynamic Calibration

- [ ] **CAL-01**: System computes per-CAMEO-category ensemble alpha weights from accumulated outcome data, replacing fixed alpha=0.6
- [ ] **CAL-02**: System uses hierarchical fallback: 4 super-categories (Verbal Coop, Material Coop, Verbal Conflict, Material Conflict) → 20 CAMEO root codes as data accumulates
- [ ] **CAL-03**: System falls back to global alpha weight when insufficient outcome data exists for a category (configurable minimum sample threshold)
- [ ] **CAL-04**: System optimizes alpha weights via scipy L-BFGS-B minimizing Brier score per category
- [ ] **CAL-05**: Ensemble predictor dynamically loads per-CAMEO weights from `calibration_weights` table at prediction time

### TKG Predictor Replacement

- [ ] **TKG-01**: System implements TiRGN algorithm in JAX, porting the global history encoder while reusing existing RE-GCN local encoder
- [ ] **TKG-02**: System defines `TKGModelProtocol` (Python Protocol class) abstracting `predict_future_events()` interface for swappable TKG backends
- [ ] **TKG-03**: TiRGN implementation achieves measurable MRR improvement over RE-GCN baseline on held-out GDELT test set
- [ ] **TKG-04**: TiRGN training completes within 24 hours on RTX 3060 12GB for full GDELT dataset
- [ ] **TKG-05**: System supports weekly automated retraining of TiRGN model with the existing retraining scheduler

### Monitoring & Hardening

- [ ] **MON-01**: System monitors GDELT feed freshness and alerts when no new data arrives for >1 hour
- [ ] **MON-02**: System monitors calibration drift (Brier score trend) and alerts when prediction quality degrades beyond threshold
- [ ] **MON-03**: System monitors Gemini API usage and cost, with daily budget cap enforcement
- [ ] **MON-04**: System exposes health endpoint reporting status of all subsystems (ingest daemon, last prediction, graph freshness, API budget remaining)
- [ ] **MON-05**: System logs all errors and warnings to structured log files with rotation

## Future Requirements (Backlog)

Tracked ideas not yet assigned to a milestone.

### Potential Improvements

- **ADV-03**: Multi-language support (non-English GDELT sources)
- **ADV-05**: Multi-source data expansion (ACLED, ICEWS) — deferred from v2.0 until micro-batch architecture is proven
- **ADV-06**: Knowledge graph visualization in Streamlit — high complexity, deferred from v2.0
- **ADV-07**: Real-time prediction (sub-daily forecast cycle) — requires incremental TKG inference

### Out of Scope

Explicitly excluded. Documented to prevent scope creep.

| Feature | Reason |
|---------|--------|
| Human forecaster integration | Pure AI system; adding humans changes product category to prediction market platform |
| Cross-sector impact modeling | Financial/supply chain requires completely different data pipeline and causal models |
| Multi-source data (v2.0) | Deferred until micro-batch architecture proven and TKG replacement validated |
| Real-time prediction (v2.0) | Micro-batch ingest yes, but predictions remain daily; real-time would require incremental TKG inference |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| INFRA-01 | — | Pending |
| INFRA-02 | — | Pending |
| INFRA-03 | — | Pending |
| INFRA-04 | — | Pending |
| INFRA-05 | — | Pending |
| INFRA-06 | — | Pending |
| INFRA-07 | — | Pending |
| INFRA-08 | — | Pending |
| WEB-01 | — | Pending |
| WEB-02 | — | Pending |
| WEB-03 | — | Pending |
| WEB-04 | — | Pending |
| WEB-05 | — | Pending |
| WEB-06 | — | Pending |
| WEB-07 | — | Pending |
| WEB-08 | — | Pending |
| WEB-09 | — | Pending |
| INGEST-01 | — | Pending |
| INGEST-02 | — | Pending |
| INGEST-03 | — | Pending |
| INGEST-04 | — | Pending |
| INGEST-05 | — | Pending |
| AUTO-01 | — | Pending |
| AUTO-02 | — | Pending |
| AUTO-03 | — | Pending |
| AUTO-04 | — | Pending |
| AUTO-05 | — | Pending |
| CAL-01 | — | Pending |
| CAL-02 | — | Pending |
| CAL-03 | — | Pending |
| CAL-04 | — | Pending |
| CAL-05 | — | Pending |
| TKG-01 | — | Pending |
| TKG-02 | — | Pending |
| TKG-03 | — | Pending |
| TKG-04 | — | Pending |
| TKG-05 | — | Pending |
| MON-01 | — | Pending |
| MON-02 | — | Pending |
| MON-03 | — | Pending |
| MON-04 | — | Pending |
| MON-05 | — | Pending |

**Coverage:**
- v2.0 requirements: 32 total
- Mapped to phases: 0 (pending roadmap creation)
- Unmapped: 32

---
*Requirements defined: 2026-02-14*
*Last updated: 2026-02-14 after v2.0 milestone definition*
