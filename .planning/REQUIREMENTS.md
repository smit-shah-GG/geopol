# Requirements: Geopol

**Core Value:** Explainability -- every forecast must provide clear, traceable reasoning paths

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

## Removed Requirements

### WEB-* (Streamlit Frontend) — Removed 2026-02-27

**Rationale:** Streamlit frontend replaced by WM-derived TypeScript dashboard. The "WM as repository" strategy uses World Monitor's architecture (Vite + TypeScript + deck.gl + Panel system) as a code quarry to build Geopol's own frontend. This produces a dramatically better product than Streamlit — 3D globe, interactive panels, Tauri desktop — while the polyglot cost (TypeScript + Python) is mitigated by WM's zero-framework vanilla architecture. See `.planning/research/WM_AS_REPOSITORY.md` for full analysis.

Removed requirements:
- ~~WEB-01~~: Streamlit forecasts display → replaced by FE-03 (ForecastPanel)
- ~~WEB-02~~: Streamlit track record → replaced by FE-08 (CalibrationPanel includes track record)
- ~~WEB-03~~: Streamlit calibration plots → replaced by FE-08 (CalibrationPanel)
- ~~WEB-04~~: Streamlit interactive queries → replaced by FE-04 (ScenarioExplorer) + API endpoint
- ~~WEB-05~~: Streamlit rate limiting → replaced by API-06 (rate limiting on FastAPI)
- ~~WEB-06~~: Streamlit session memory → N/A (TypeScript frontend is stateless client)
- ~~WEB-07~~: Streamlit methodology page → replaced by FE-07 (Country Brief includes methodology context per forecast)
- ~~WEB-08~~: Streamlit prompt injection → replaced by API-05 (input sanitization on API layer)
- ~~WEB-09~~: Streamlit Gemini budget cap → replaced by API-06 (rate limiting + budget enforcement)

## v2.0 Requirements

**Defined:** 2026-02-14. **Restructured:** 2026-02-27 (WM-derived frontend replaces Streamlit).
**Core Value:** Explainability -- every forecast must provide clear, traceable reasoning paths
**Milestone Goal:** Transform research prototype into a publicly demonstrable system with a WM-derived TypeScript dashboard, headless FastAPI backend, automated operations, upgraded TKG predictor, and self-improving calibration

### Infrastructure & Persistence

- [ ] **INFRA-01**: System persists all forecasts (question, prediction, probability, reasoning chain, timestamps) in PostgreSQL `predictions` table, queryable via SQL and exposed through FastAPI endpoints
- [ ] **INFRA-02**: System persists outcome records comparing past predictions against GDELT ground truth in PostgreSQL `outcome_records` table
- [ ] **INFRA-03**: System persists per-CAMEO calibration weights in PostgreSQL `calibration_weights` table
- [ ] **INFRA-04**: System tracks micro-batch ingest runs in PostgreSQL `ingest_runs` table with status, event counts, and timestamps
- [ ] **INFRA-05**: System eliminates archived jraph dependency, replacing with raw JAX ops (local NamedTuple for GraphsTuple, `jax.ops.segment_sum`)
- [ ] **INFRA-06**: System runs FastAPI server, ingest daemon, and prediction pipeline as separate OS processes communicating through PostgreSQL (GDELT event store remains SQLite with WAL mode)
- [ ] **INFRA-07**: System uses structured logging (Python `logging` module) replacing all `print()` statements in production code paths
- [ ] **INFRA-08**: PostgreSQL connections use connection pooling (asyncpg pool or SQLAlchemy async); SQLite GDELT store retains WAL mode with `busy_timeout=30000ms`

### API Layer

- [ ] **API-01**: FastAPI server (`src/api/`) with versioned routes (`/api/v1/`), health endpoint, and OpenAPI schema auto-generated from Pydantic DTOs
- [ ] **API-02**: Pydantic DTOs: `ForecastResponse`, `ScenarioDTO`, `CalibrationDTO`, `CountryRiskSummary`, `EnsembleInfoDTO`, `EvidenceDTO` — matching the contract spec in `WORLDMONITOR_INTEGRATION.md` lines 328-382
- [ ] **API-03**: API key authentication middleware validating `X-API-Key` header; CORS middleware configured for frontend origins
- [ ] **API-04**: Redis forecast response caching with three-tier hierarchy: in-memory LRU (100 entries, 10-min TTL) → Redis (1-hour TTL for summaries, 6-hour for full forecasts) → PostgreSQL (cold storage)
- [ ] **API-05**: Input sanitization on `POST /api/v1/forecasts` preventing prompt injection; no system internals (API keys, file paths, model config) leaked in responses
- [ ] **API-06**: Rate limiting on `POST /api/v1/forecasts` (on-demand generation) with Gemini API daily budget enforcement; requests rejected when budget exhausted
- [ ] **API-07**: Mock/fixture responses for all endpoints — hardcoded realistic `ForecastResponse` objects (Syria, Ukraine, Myanmar scenarios) enabling contract-first frontend development before Phase 10 delivers real data

### Frontend (WM-Derived TypeScript Dashboard)

- [x] **FE-01**: WM-derived scaffold: Vite build system, TypeScript strict mode, `Panel` base class with `refresh()` lifecycle, `AppContext` singleton, `DataLoaderManager` skeleton, `RefreshScheduler`, `h()` DOM helper — stripped of all WM-specific panels/services/data sources
- [x] **FE-02**: Forecast service client (`services/forecast/client.ts`) consuming Geopol REST API with circuit breaker, freshness tracking, `inFlight` request deduplication, and response caching
- [x] **FE-03**: Dashboard panels: `ForecastPanel` (top N active forecasts globally), `RiskIndexPanel` (per-country aggregate risk, CII-derived pattern), `EventTimelinePanel` (recent GDELT events), `EnsembleBreakdownPanel` (LLM vs TKG weights), `SystemHealthPanel` (ingest freshness, graph size, API budget)
- [x] **FE-04**: `ScenarioExplorer` modal: full-screen interactive scenario tree visualization with click-to-expand branches, probability as node size, evidence sidebar with GDELT event links, historical precedent panel
- [x] **FE-05**: Map layers: `ForecastRiskChoropleth` (GeoJsonLayer coloring countries by aggregate forecast risk), `ActiveForecastMarkers` (ScatterplotLayer sized by probability), `GDELTEventHeatmap` (density of recent events)
- [x] **FE-06**: Country Brief page: full-screen modal (WM pattern) with tabs — Active Forecasts (probability bars + scenario trees), GDELT Events, Forecast History, Risk Signals (CAMEO categories), Entity Relations (knowledge graph subgraph), Calibration (per-category accuracy)
- [x] **FE-07**: Dark/light theme with WM's CSS variable system and semantic severity colors (critical/high/elevated/normal/low mapped to forecast confidence bands)
- [x] **FE-08**: `CalibrationPanel`: reliability diagrams, Brier score decomposition, per-CAMEO category accuracy, prediction track record over time

### Micro-batch Ingest

- [ ] **INGEST-01**: System polls GDELT 15-minute update feed (`lastupdate.txt`) on a configurable schedule via daemon process
- [ ] **INGEST-02**: System performs incremental graph update appending only new events, avoiding full graph rebuild (O(N_new) not O(N_total))
- [ ] **INGEST-03**: System deduplicates events across micro-batches and daily dumps using GDELT `GlobalEventID`
- [ ] **INGEST-04**: System handles missing/late GDELT feed updates gracefully with exponential backoff and logged warnings
- [ ] **INGEST-05**: Ingest daemon tracks per-run metrics (events fetched, events new, events duplicate, duration) in `ingest_runs` table
- [ ] **INGEST-06**: Ingest daemon polls WM-curated RSS feed list (298 domains) on 15-minute cycle, extracts article text via `trafilatura`, chunks and indexes into ChromaDB for RAG enrichment — giving the LLM full narrative context beyond GDELT's terse event descriptions

### Daily Forecast Automation

- [ ] **AUTO-01**: System runs daily forecast pipeline on a configurable schedule (default 06:00) via systemd timer
- [ ] **AUTO-02**: Daily pipeline generates forecast questions from recent high-significance events in the knowledge graph
- [ ] **AUTO-03**: Daily pipeline runs full Gemini+TKG ensemble prediction for each generated question and persists results
- [ ] **AUTO-04**: Daily pipeline resolves outcomes for past predictions by comparing against subsequent GDELT events within a configurable time window
- [ ] **AUTO-05**: Daily pipeline handles failures with retry logic and sends alerts on consecutive failures

### Dynamic Calibration

- [ ] **CAL-01**: System computes per-CAMEO-category ensemble alpha weights from accumulated outcome data, replacing fixed alpha=0.6
- [ ] **CAL-02**: System uses hierarchical fallback: 4 super-categories (Verbal Coop, Material Coop, Verbal Conflict, Material Conflict) -> 20 CAMEO root codes as data accumulates
- [ ] **CAL-03**: System falls back to global alpha weight when insufficient outcome data exists for a category (configurable minimum sample threshold)
- [ ] **CAL-04**: System optimizes alpha weights via scipy L-BFGS-B minimizing Brier score per category
- [ ] **CAL-05**: Ensemble predictor dynamically loads per-CAMEO weights from `calibration_weights` table at prediction time
- [ ] **CAL-06**: System fetches Polymarket prediction market data for geopolitical questions and displays side-by-side comparison with Geopol's calibrated forecasts — tracking which source is more accurate over time

### TKG Predictor Replacement

- [x] **TKG-01**: System implements TiRGN algorithm in JAX, porting the global history encoder while reusing existing RE-GCN local encoder
- [x] **TKG-02**: System defines `TKGModelProtocol` (Python Protocol class) abstracting `predict_future_events()` interface for swappable TKG backends
- [x] **TKG-03**: TiRGN implementation achieves measurable MRR improvement over RE-GCN baseline on held-out GDELT test set
- [x] **TKG-04**: TiRGN training completes within 24 hours on RTX 3060 12GB for full GDELT dataset
- [x] **TKG-05**: System supports weekly automated retraining of TiRGN model with the existing retraining scheduler

### Monitoring & Hardening

- [ ] **MON-01**: System monitors GDELT feed freshness and alerts when no new data arrives for >1 hour
- [ ] **MON-02**: System monitors calibration drift (Brier score trend) and alerts when prediction quality degrades beyond threshold
- [ ] **MON-03**: System monitors Gemini API usage and cost, with daily budget cap enforcement
- [ ] **MON-04**: System exposes health endpoint reporting status of all subsystems (ingest daemon, last prediction, graph freshness, API budget remaining)
- [ ] **MON-05**: System logs all errors and warnings to structured log files with rotation

## Future Requirements (Backlog)

Tracked ideas not yet assigned to a milestone. Includes WM-derived features for future cherry-picking.

### WM Feature Backlog (Reference Implementations Exist)

- **WM-FEAT-01**: Variant system — regional forecast builds (MENA, Europe, Asia-Pacific, Africa) from single codebase via `VITE_VARIANT`
- **WM-FEAT-02**: Prediction markets panel — Polymarket data alongside Geopol forecasts for calibration comparison
- **WM-FEAT-03**: Command palette (⌘K) — search forecasts, countries, entities
- **WM-FEAT-04**: Playback mode — replay how forecasts evolved over time (time travel)
- **WM-FEAT-05**: i18n / RTL — multi-language forecast display
- **WM-FEAT-06**: Virtual scrolling — for large forecast history lists
- **WM-FEAT-07**: Export (JSON, CSV, PNG) — forecast report generation
- **WM-FEAT-08**: URL state binding — shareable deep links (`?country=SY&forecast=abc123`)
- **WM-FEAT-09**: Tauri desktop — native desktop app with keychain secrets and local API sidecar
- **WM-FEAT-10**: PWA offline support — cached forecasts viewable without network

### Potential Improvements

- **ADV-03**: Multi-language support (non-English GDELT sources)
- **ADV-05**: ACLED → knowledge graph (structured conflict events with actor dyads map naturally to CAMEO-like triples; needs CAMEO code mapping layer) — deferred until Phase 10 proves micro-batch ingest architecture
- **ADV-08**: ICEWS data expansion — deferred from v2.0 until micro-batch architecture is proven
- **ADV-06**: Knowledge graph visualization in dashboard -- high complexity, deferred from v2.0
- **ADV-07**: Real-time prediction (sub-daily forecast cycle) -- requires incremental TKG inference

### Out of Scope

Explicitly excluded. Documented to prevent scope creep.

| Feature | Reason |
|---------|--------|
| Human forecaster integration | Pure AI system; adding humans changes product category to prediction market platform |
| Cross-sector impact modeling | Financial/supply chain requires completely different data pipeline and causal models |
| Multi-source data (v2.0) | Deferred until micro-batch architecture proven and TKG replacement validated |
| Real-time prediction (v2.0) | Micro-batch ingest yes, but predictions remain daily; real-time would require incremental TKG inference |
| World Monitor live integration | WM used as reference architecture and code quarry, not as a runtime dependency; no service-to-service HTTP calls between WM and Geopol |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| INFRA-01 | Phase 9 | Complete |
| INFRA-02 | Phase 9 | Complete |
| INFRA-03 | Phase 9 | Complete |
| INFRA-04 | Phase 9 | Complete |
| INFRA-05 | Phase 9 | Complete |
| INFRA-06 | Phase 9 | Complete |
| INFRA-07 | Phase 9 | Complete |
| INFRA-08 | Phase 9 | Complete |
| API-01 | Phase 9 | Complete |
| API-02 | Phase 9 | Complete |
| API-03 | Phase 9 | Complete |
| API-04 | Phase 10 | Complete |
| API-05 | Phase 10 | Complete |
| API-06 | Phase 10 | Complete |
| API-07 | Phase 9 | Complete |
| FE-01 | Phase 12 | Complete |
| FE-02 | Phase 12 | Complete |
| FE-03 | Phase 12 | Complete |
| FE-04 | Phase 12 | Complete |
| FE-05 | Phase 12 | Complete |
| FE-06 | Phase 12 | Complete |
| FE-07 | Phase 12 | Complete |
| FE-08 | Phase 12 | Complete |
| INGEST-01 | Phase 10 | Complete |
| INGEST-02 | Phase 10 | Complete |
| INGEST-03 | Phase 10 | Complete |
| INGEST-04 | Phase 10 | Complete |
| INGEST-05 | Phase 10 | Complete |
| INGEST-06 | Phase 10 | Complete |
| AUTO-01 | Phase 10 | Complete |
| AUTO-02 | Phase 10 | Complete |
| AUTO-03 | Phase 10 | Complete |
| AUTO-04 | Phase 10 | Complete |
| AUTO-05 | Phase 10 | Complete |
| CAL-01 | Phase 13 | Pending |
| CAL-02 | Phase 13 | Pending |
| CAL-03 | Phase 13 | Pending |
| CAL-04 | Phase 13 | Pending |
| CAL-05 | Phase 13 | Pending |
| CAL-06 | Phase 13 | Pending |
| TKG-01 | Phase 11 | Complete |
| TKG-02 | Phase 11 | Complete |
| TKG-03 | Phase 11 | Complete |
| TKG-04 | Phase 11 | Complete |
| TKG-05 | Phase 11 | Complete |
| MON-01 | Phase 13 | Pending |
| MON-02 | Phase 13 | Pending |
| MON-03 | Phase 13 | Pending |
| MON-04 | Phase 13 | Pending |
| MON-05 | Phase 13 | Pending |

**Coverage:**
- v2.0 requirements: 50 total (8 INFRA + 7 API + 8 FE + 6 INGEST + 5 AUTO + 6 CAL + 5 TKG + 5 MON)
- Mapped to phases: 50/50
- Unmapped: 0

**Phase distribution:**
- Phase 9: 11 requirements (INFRA-01..08, API-01..03, API-07)
- Phase 10: 14 requirements (INGEST-01..06, AUTO-01..05, API-04..06)
- Phase 11: 5 requirements (TKG-01..05)
- Phase 12: 8 requirements (FE-01..08)
- Phase 13: 11 requirements (CAL-01..06, MON-01..05)

---
*Requirements defined: 2026-02-14*
*Restructured: 2026-02-27 — WM-derived frontend replaces Streamlit; API layer added; PostgreSQL replaces SQLite for forecast persistence; CAL-* moved to Phase 13; parallel execution model adopted*
*Updated: 2026-02-27 — Added INGEST-06 (RSS feed → RAG enrichment from WM's 298-domain feed list) and CAL-06 (Polymarket calibration comparison)*
