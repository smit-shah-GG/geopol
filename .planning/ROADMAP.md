# Roadmap: Explainable Geopolitical Forecasting Engine

## Milestones

- v1.0 MVP -- Phases 1-5 (shipped 2026-01-23)
- v1.1 Tech Debt Remediation -- Phases 6-8 (shipped 2026-01-30)
- v2.0 Operationalization & Forecast Quality -- Phases 9-13 (in progress)

## Phases

<details>
<summary>v1.0 MVP (Phases 1-5) - SHIPPED 2026-01-23</summary>

### Phase 1: Data Foundation
**Goal**: GDELT event ingestion pipeline with intelligent sampling
**Plans**: 3 plans

Plans:
- [x] 01-01: GDELT API client and event schema
- [x] 01-02: Event filtering and sampling strategy
- [x] 01-03: Storage layer and data pipeline

### Phase 2: Knowledge Graph Engine
**Goal**: Temporal knowledge graph construction from event streams
**Plans**: 3 plans

Plans:
- [x] 02-01: Entity extraction and resolution
- [x] 02-02: Relationship extraction and graph construction
- [x] 02-03: Graph querying and traversal

### Phase 3: Hybrid Forecasting
**Goal**: Ensemble prediction combining LLM reasoning with TKG patterns
**Plans**: 4 plans

Plans:
- [x] 03-01: RAG indexing pipeline
- [x] 03-02: LLM integration with Gemini
- [x] 03-03: RE-GCN TKG predictor
- [x] 03-04: Ensemble forecaster and CLI

### Phase 4: Calibration & Evaluation
**Goal**: Probability calibration and historical evaluation
**Plans**: 2 plans

Plans:
- [x] 04-01: Isotonic and temperature scaling calibration
- [x] 04-02: Evaluation framework against historical events

### Phase 5: TKG Training
**Goal**: RE-GCN training pipeline on GDELT data
**Plans**: 4 plans

Plans:
- [x] 05-01: GDELT data collection for training
- [x] 05-02: TKG data preprocessing
- [x] 05-03: JAX/jraph RE-GCN training
- [x] 05-04: Retraining scheduler

</details>

<details>
<summary>v1.1 Tech Debt Remediation (Phases 6-8) - SHIPPED 2026-01-30</summary>

### Phase 6: NetworkX API Fix
**Goal**: Graph entity queries return valid results without API errors
**Plans**: 1 plan

Plans:
- [x] 06-01: Fix NetworkX API call and update tests

### Phase 7: Bootstrap Pipeline
**Goal**: A single command takes the system from zero data to fully operational
**Plans**: 2 plans

Plans:
- [x] 07-01: Bootstrap orchestration module with stage definitions
- [x] 07-02: Checkpoint/resume and dual idempotency

### Phase 8: Graph Partitioning
**Goal**: Knowledge graph scales beyond 1M events through partitioning
**Plans**: 2 plans

Plans:
- [x] 08-01: Partition index (SQLite) and partition manager (temporal-first partitioning, LRU cache)
- [x] 08-02: Boundary resolver, scatter-gather query router, PartitionedTemporalGraph interface

</details>

### v2.0 Operationalization & Forecast Quality (In Progress)

**Milestone Goal:** Transform research prototype into publicly demonstrable system with a WM-derived TypeScript dashboard, headless FastAPI backend, automated operations, upgraded TKG predictor, and self-improving calibration.

**Architecture Decision (2026-02-27):** World Monitor used as reference architecture and code quarry — not as a live integration target. Geopol becomes a headless Python forecast engine with its own WM-derived TypeScript frontend. See `.planning/research/WM_AS_REPOSITORY.md` for full analysis and `WORLDMONITOR_INTEGRATION.md` for DTO contract spec.

**Phase Numbering:** Starts at 9 (v1.1 ended at Phase 8; cancelled Llama-TGL phases 9-14 archived in `.planning/archive/v2.0-llama-cancelled.md`).

**Execution Model:** Parallel after Phase 9. Phases 10, 11, and 12 run concurrently once Phase 9 establishes the API contract (DTOs + mock fixtures). Phase 13 waits for all three.

```
Phase 9 (API + DB foundation) ─── critical path, everything gates on this
    │
    ├──► Phase 10 (ingest + pipeline + real API data)
    │
    ├──► Phase 11 (TKG replacement) ──── parallelizable
    │
    └──► Phase 12 (frontend against mock API → real API when Phase 10 lands)
                │
                └──────────► Phase 13 (monitoring + calibration + hardening)
```

- [x] **Phase 9: API Foundation & Infrastructure** — PostgreSQL, FastAPI skeleton with DTOs and mock fixtures, structured logging, jraph elimination, TKGModelProtocol
- [ ] **Phase 10: Ingest & Forecast Pipeline** — Micro-batch GDELT ingest, daily forecast automation, real API endpoints replacing mocks, Redis caching
- [ ] **Phase 11: TKG Predictor Replacement** — TiRGN JAX port replacing RE-GCN for improved accuracy (parallelizable with Phases 10 and 12)
- [ ] **Phase 12: WM-Derived Frontend** — TypeScript dashboard scaffolded from World Monitor patterns: deck.gl globe, forecast panels, scenario explorer, country briefs, map layers
- [ ] **Phase 13: Calibration, Monitoring & Hardening** — Dynamic per-CAMEO calibration from accumulated outcome data, system health observability, alerting, operational resilience

## Phase Details

### Phase 9: API Foundation & Infrastructure
**Goal**: System has the persistence layer, headless API server with mock data, and cleaned dependency tree that every v2.0 feature requires. The API DTOs and mock fixtures establish the contract that Phases 10 and 12 develop against independently.
**Depends on**: Phase 8 (v1.1 complete)
**Requirements**: INFRA-01, INFRA-02, INFRA-03, INFRA-04, INFRA-05, INFRA-06, INFRA-07, INFRA-08, API-01, API-02, API-03, API-07
**Success Criteria** (what must be TRUE):
  1. `GET /api/v1/health` returns JSON reporting subsystem status (database connected, graph partition count, model loaded)
  2. `GET /api/v1/forecasts/country/SY` returns a mock `ForecastResponse` with structurally valid scenarios, evidence, and calibration data — the response matches the DTO contract spec
  3. `POST /api/v1/forecasts` with a valid API key accepts a question and returns a mock forecast; without a valid API key returns 401
  4. Running `EnsemblePredictor.predict()` persists the forecast to PostgreSQL and the row is retrievable via both SQL and `GET /api/v1/forecasts/{id}`
  5. Three separate Python processes (FastAPI server, simulated ingest daemon, simulated prediction pipeline) can read/write the PostgreSQL database concurrently without errors or data corruption
  6. All production code paths emit structured log messages (no `print()` statements remain) with severity, timestamp, and module name
  7. Importing the TKG training module succeeds without jraph installed — all jraph references replaced with local JAX equivalents
  8. `TKGModelProtocol` is defined and both the existing RE-GCN wrapper and a stub TiRGN class satisfy it (verified by `isinstance` or Protocol structural check)
**Plans**: 6 plans

Plans:
- [x] 09-01-PLAN.md — Dependencies, Docker, PostgreSQL ORM models, Alembic migrations, settings
- [x] 09-02-PLAN.md — Pydantic V2 DTOs (contract spec, 8-subsystem health schema) and mock fixtures (SY, UA, MM)
- [x] 09-03-PLAN.md — jraph elimination, TKGModelProtocol, structured logging config module
- [x] 09-04-PLAN.md — print() to logging conversion sweep (9 production files)
- [x] 09-05-PLAN.md — FastAPI app, full subsystem health endpoint (8 checks), routes, auth middleware, error handling
- [x] 09-06-PLAN.md — ForecastService persistence bridge, route wiring, multi-process concurrent DB tests, table smoke writes

### Phase 10: Ingest & Forecast Pipeline
**Goal**: System continuously ingests GDELT events every 15 minutes and produces daily automated forecasts with outcome tracking. API endpoints serve real forecast data (replacing Phase 9 mock fixtures). Redis caching prevents redundant computation.
**Depends on**: Phase 9
**Requirements**: INGEST-01, INGEST-02, INGEST-03, INGEST-04, INGEST-05, INGEST-06, AUTO-01, AUTO-02, AUTO-03, AUTO-04, AUTO-05, API-04, API-05, API-06
**Success Criteria** (what must be TRUE):
  1. Ingest daemon runs for 1+ hours, fetching GDELT updates every 15 minutes, and the `ingest_runs` table shows sequential successful runs with non-zero event counts and no duplicate `GlobalEventID` values across runs
  2. A missed or failed GDELT feed fetch triggers exponential backoff retries with logged warnings, and the daemon resumes normal operation when the feed recovers
  3. Daily pipeline (triggered via systemd timer or manual invocation) generates forecast questions from recent high-significance events, runs Gemini+TKG ensemble predictions, and persists results to the `predictions` table
  4. `GET /api/v1/forecasts/country/{iso}` returns real forecasts generated by the daily pipeline (not mock fixtures); responses are served from Redis cache on subsequent requests within TTL
  5. `POST /api/v1/forecasts` generates a live forecast with input sanitization — crafted prompt-injection inputs do not leak system internals; requests are rejected when daily Gemini budget is exhausted
  6. After sufficient time passes, the daily pipeline resolves past predictions against GDELT ground truth and writes outcome records to the `outcome_records` table
  7. A consecutive daily pipeline failure triggers an alert (log or notification) and the system recovers on the next scheduled run without manual intervention
  8. RSS ingest daemon polls WM-curated feed list, and querying ChromaDB for a recent geopolitical topic returns article chunks from RSS sources (not just GDELT event descriptions)
**Plans**: 4 plans

Plans:
- [ ] 10-01-PLAN.md — GDELT micro-batch poller daemon with backoff, metrics, incremental graph update
- [ ] 10-02-PLAN.md — API hardening: three-tier cache, per-key rate limiting, input sanitization
- [ ] 10-03-PLAN.md — RSS feed ingestion daemon with tiered polling and ChromaDB indexing
- [ ] 10-04-PLAN.md — Daily forecast pipeline, outcome resolution, real API route wiring

### Phase 11: TKG Predictor Replacement
**Goal**: TiRGN replaces RE-GCN as the TKG backend, delivering measurable accuracy improvement while fitting the RTX 3060 training envelope and weekly retraining cadence
**Depends on**: Phase 9 (TKGModelProtocol defined, jraph eliminated)
**Requirements**: TKG-01, TKG-02, TKG-03, TKG-04, TKG-05
**Research flag**: YES — no published JAX implementation of TiRGN exists. Plan-phase should evaluate whether `/gsd:research-phase` is needed before implementation.
**Success Criteria** (what must be TRUE):
  1. TiRGN model trains to completion on the full GDELT dataset within 24 hours on RTX 3060 12GB without OOM
  2. TiRGN achieves higher MRR than RE-GCN on a held-out GDELT test set (improvement logged and reproducible)
  3. Swapping `TKGPredictor` from RE-GCN to TiRGN requires only a config change — no downstream code modifications to `EnsemblePredictor`, calibration, or the daily pipeline
  4. Weekly automated retraining (via existing scheduler) completes successfully with the TiRGN model
**Plans**: TBD

### Phase 12: WM-Derived Frontend
**Goal**: External visitors see a production-quality dashboard with deck.gl globe, forecast panels, interactive scenario exploration, and country briefs — all consuming Geopol's FastAPI backend. Architecturally derived from World Monitor's vanilla TypeScript patterns but purpose-built for geopolitical forecasting.
**Depends on**: Phase 9 (API contract with mock fixtures — real data from Phase 10 is not required to start; frontend develops against mocks)
**Requirements**: FE-01, FE-02, FE-03, FE-04, FE-05, FE-06, FE-07, FE-08
**WM Reference**: `.planning/research/WM_AS_REPOSITORY.md` documents salvageable components, panel mappings, and layer mappings
**Success Criteria** (what must be TRUE):
  1. Dashboard loads in browser showing a deck.gl globe with `ForecastRiskChoropleth` layer coloring countries by aggregate forecast risk, and at least 3 additional map layers toggleable via UI
  2. `ForecastPanel` displays top N active forecasts (from API or mock fixtures) with question, probability (color-coded), confidence, scenario count, and last-updated timestamp
  3. Clicking a forecast opens `ScenarioExplorer` modal showing interactive scenario tree with expandable branches, probability per node, and evidence sidebar with source links
  4. Clicking a country on the globe opens Country Brief page with at least 4 tabs (Active Forecasts, GDELT Events, Forecast History, Calibration), each populated from API data
  5. `RiskIndexPanel` displays per-country aggregate risk scores with trend indicators (rising/stable/falling), derived from the CII scoring pattern
  6. `CalibrationPanel` displays reliability diagram and Brier score decomposition (populated from real data when Phase 10 is complete, placeholder when running against mocks)
  7. Dark/light theme toggle works correctly with semantic severity colors consistent across all panels and map layers
  8. `ForecastServiceClient` implements circuit breaker pattern — API failures result in stale-data display with "unavailable" indicator, not a broken UI
**Plans**: TBD

### Phase 13: Calibration, Monitoring & Hardening
**Goal**: System self-improves its ensemble weights from accumulated outcome data, health is continuously observable, operational failures trigger alerts, and the system runs unattended for days without degradation
**Depends on**: Phase 10 (prediction-outcome data must have accumulated), Phase 12 (frontend exists to display health data)
**Requirements**: CAL-01, CAL-02, CAL-03, CAL-04, CAL-05, CAL-06, MON-01, MON-02, MON-03, MON-04, MON-05
**Success Criteria** (what must be TRUE):
  1. The `calibration_weights` table contains per-CAMEO alpha weights that differ from the default 0.6, computed from accumulated outcome data via L-BFGS-B optimization
  2. The `EnsemblePredictor` loads and uses per-CAMEO weights at prediction time (verified by checking prediction logs show varying alpha per category)
  3. Hierarchical fallback works: categories with <N outcome samples use super-category weights; super-categories with <N samples use global weight
  4. `GET /api/v1/health` returns JSON reporting status of every subsystem (ingest daemon up/down, last prediction timestamp, graph freshness age, Gemini API budget remaining) and the `SystemHealthPanel` in the frontend displays this information
  5. Stopping the GDELT feed for >1 hour triggers a logged alert; Brier score degradation beyond a configured threshold triggers a calibration drift alert
  6. All errors and warnings across all subsystems are written to structured log files with automatic rotation (no unbounded log growth)
  7. Polymarket prediction market data is fetched for at least 5 geopolitical questions that overlap with Geopol forecasts, and a comparison table shows Geopol probability vs. market probability for each
**Plans**: TBD

## Progress

**Execution Order:**
Phase 9 first (critical path). Then Phases 10, 11, 12 in parallel. Phase 13 after convergence.

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|----------------|--------|-----------|
| 1. Data Foundation | v1.0 | 3/3 | Complete | 2026-01-11 |
| 2. Knowledge Graph | v1.0 | 3/3 | Complete | 2026-01-13 |
| 3. Hybrid Forecasting | v1.0 | 4/4 | Complete | 2026-01-17 |
| 4. Calibration | v1.0 | 2/2 | Complete | 2026-01-19 |
| 5. TKG Training | v1.0 | 4/4 | Complete | 2026-01-23 |
| 6. NetworkX Fix | v1.1 | 1/1 | Complete | 2026-01-28 |
| 7. Bootstrap Pipeline | v1.1 | 2/2 | Complete | 2026-01-30 |
| 8. Graph Partitioning | v1.1 | 2/2 | Complete | 2026-01-30 |
| 9. API Foundation | v2.0 | 6/6 | Complete | 2026-03-01 |
| 10. Ingest & Pipeline | v2.0 | 0/4 | Planned | - |
| 11. TKG Replacement | v2.0 | 0/TBD | Not started | - |
| 12. WM-Derived Frontend | v2.0 | 0/TBD | Not started | - |
| 13. Calibration & Monitoring | v2.0 | 0/TBD | Not started | - |

**Total:** 9 phases complete (v1.0 + v1.1 + Phase 9), 27 plans delivered. v2.0: 1/5 phases complete, 48 requirements, Phases 10/11/12 now parallelizable.
