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

### v2.0 Operationalization & Forecast Quality (shipped 2026-03-02)

50 requirements delivered: 8 INFRA + 7 API + 8 FE + 6 INGEST + 5 AUTO + 6 CAL + 5 TKG + 5 MON.
See detailed v2.0 requirements section below for full list.

## Removed Requirements

### WEB-* (Streamlit Frontend) -- Removed 2026-02-27

**Rationale:** Streamlit frontend replaced by WM-derived TypeScript dashboard. The "WM as repository" strategy uses World Monitor's architecture (Vite + TypeScript + deck.gl + Panel system) as a code quarry to build Geopol's own frontend. This produces a dramatically better product than Streamlit -- 3D globe, interactive panels, Tauri desktop -- while the polyglot cost (TypeScript + Python) is mitigated by WM's zero-framework vanilla architecture. See `.planning/research/WM_AS_REPOSITORY.md` for full analysis.

Removed requirements:
- ~~WEB-01~~: Streamlit forecasts display -> replaced by FE-03 (ForecastPanel)
- ~~WEB-02~~: Streamlit track record -> replaced by FE-08 (CalibrationPanel includes track record)
- ~~WEB-03~~: Streamlit calibration plots -> replaced by FE-08 (CalibrationPanel)
- ~~WEB-04~~: Streamlit interactive queries -> replaced by FE-04 (ScenarioExplorer) + API endpoint
- ~~WEB-05~~: Streamlit rate limiting -> replaced by API-06 (rate limiting on FastAPI)
- ~~WEB-06~~: Streamlit session memory -> N/A (TypeScript frontend is stateless client)
- ~~WEB-07~~: Streamlit methodology page -> replaced by FE-07 (Country Brief includes methodology context per forecast)
- ~~WEB-08~~: Streamlit prompt injection -> replaced by API-05 (input sanitization on API layer)
- ~~WEB-09~~: Streamlit Gemini budget cap -> replaced by API-06 (rate limiting + budget enforcement)

## v2.0 Requirements

**Defined:** 2026-02-14. **Restructured:** 2026-02-27 (WM-derived frontend replaces Streamlit).
**Core Value:** Explainability -- every forecast must provide clear, traceable reasoning paths
**Milestone Goal:** Transform research prototype into a publicly demonstrable system with a WM-derived TypeScript dashboard, headless FastAPI backend, automated operations, upgraded TKG predictor, and self-improving calibration

### Infrastructure & Persistence

- [x] **INFRA-01**: System persists all forecasts (question, prediction, probability, reasoning chain, timestamps) in PostgreSQL `predictions` table, queryable via SQL and exposed through FastAPI endpoints
- [x] **INFRA-02**: System persists outcome records comparing past predictions against GDELT ground truth in PostgreSQL `outcome_records` table
- [x] **INFRA-03**: System persists per-CAMEO calibration weights in PostgreSQL `calibration_weights` table
- [x] **INFRA-04**: System tracks micro-batch ingest runs in PostgreSQL `ingest_runs` table with status, event counts, and timestamps
- [x] **INFRA-05**: System eliminates archived jraph dependency, replacing with raw JAX ops (local NamedTuple for GraphsTuple, `jax.ops.segment_sum`)
- [x] **INFRA-06**: System runs FastAPI server, ingest daemon, and prediction pipeline as separate OS processes communicating through PostgreSQL (GDELT event store remains SQLite with WAL mode)
- [x] **INFRA-07**: System uses structured logging (Python `logging` module) replacing all `print()` statements in production code paths
- [x] **INFRA-08**: PostgreSQL connections use connection pooling (asyncpg pool or SQLAlchemy async); SQLite GDELT store retains WAL mode with `busy_timeout=30000ms`

### API Layer

- [x] **API-01**: FastAPI server (`src/api/`) with versioned routes (`/api/v1/`), health endpoint, and OpenAPI schema auto-generated from Pydantic DTOs
- [x] **API-02**: Pydantic DTOs: `ForecastResponse`, `ScenarioDTO`, `CalibrationDTO`, `CountryRiskSummary`, `EnsembleInfoDTO`, `EvidenceDTO` -- matching the contract spec in `WORLDMONITOR_INTEGRATION.md` lines 328-382
- [x] **API-03**: API key authentication middleware validating `X-API-Key` header; CORS middleware configured for frontend origins
- [x] **API-04**: Redis forecast response caching with three-tier hierarchy: in-memory LRU (100 entries, 10-min TTL) -> Redis (1-hour TTL for summaries, 6-hour for full forecasts) -> PostgreSQL (cold storage)
- [x] **API-05**: Input sanitization on `POST /api/v1/forecasts` preventing prompt injection; no system internals (API keys, file paths, model config) leaked in responses
- [x] **API-06**: Rate limiting on `POST /api/v1/forecasts` (on-demand generation) with Gemini API daily budget enforcement; requests rejected when budget exhausted
- [x] **API-07**: Mock/fixture responses for all endpoints -- hardcoded realistic `ForecastResponse` objects (Syria, Ukraine, Myanmar scenarios) enabling contract-first frontend development before Phase 10 delivers real data

### Frontend (WM-Derived TypeScript Dashboard)

- [x] **FE-01**: WM-derived scaffold: Vite build system, TypeScript strict mode, `Panel` base class with `refresh()` lifecycle, `AppContext` singleton, `DataLoaderManager` skeleton, `RefreshScheduler`, `h()` DOM helper -- stripped of all WM-specific panels/services/data sources
- [x] **FE-02**: Forecast service client (`services/forecast/client.ts`) consuming Geopol REST API with circuit breaker, freshness tracking, `inFlight` request deduplication, and response caching
- [x] **FE-03**: Dashboard panels: `ForecastPanel` (top N active forecasts globally), `RiskIndexPanel` (per-country aggregate risk, CII-derived pattern), `EventTimelinePanel` (recent GDELT events), `EnsembleBreakdownPanel` (LLM vs TKG weights), `SystemHealthPanel` (ingest freshness, graph size, API budget)
- [x] **FE-04**: `ScenarioExplorer` modal: full-screen interactive scenario tree visualization with click-to-expand branches, probability as node size, evidence sidebar with GDELT event links, historical precedent panel
- [x] **FE-05**: Map layers: `ForecastRiskChoropleth` (GeoJsonLayer coloring countries by aggregate forecast risk), `ActiveForecastMarkers` (ScatterplotLayer sized by probability), `GDELTEventHeatmap` (density of recent events)
- [x] **FE-06**: Country Brief page: full-screen modal (WM pattern) with tabs -- Active Forecasts (probability bars + scenario trees), GDELT Events, Forecast History, Risk Signals (CAMEO categories), Entity Relations (knowledge graph subgraph), Calibration (per-category accuracy)
- [x] **FE-07**: Dark/light theme with WM's CSS variable system and semantic severity colors (critical/high/elevated/normal/low mapped to forecast confidence bands)
- [x] **FE-08**: `CalibrationPanel`: reliability diagrams, Brier score decomposition, per-CAMEO category accuracy, prediction track record over time

### Micro-batch Ingest

- [x] **INGEST-01**: System polls GDELT 15-minute update feed (`lastupdate.txt`) on a configurable schedule via daemon process
- [x] **INGEST-02**: System performs incremental graph update appending only new events, avoiding full graph rebuild (O(N_new) not O(N_total))
- [x] **INGEST-03**: System deduplicates events across micro-batches and daily dumps using GDELT `GlobalEventID`
- [x] **INGEST-04**: System handles missing/late GDELT feed updates gracefully with exponential backoff and logged warnings
- [x] **INGEST-05**: Ingest daemon tracks per-run metrics (events fetched, events new, events duplicate, duration) in `ingest_runs` table
- [x] **INGEST-06**: Ingest daemon polls WM-curated RSS feed list (298 domains) on 15-minute cycle, extracts article text via `trafilatura`, chunks and indexes into ChromaDB for RAG enrichment -- giving the LLM full narrative context beyond GDELT's terse event descriptions

### Daily Forecast Automation

- [x] **AUTO-01**: System runs daily forecast pipeline on a configurable schedule (default 06:00) via systemd timer
- [x] **AUTO-02**: Daily pipeline generates forecast questions from recent high-significance events in the knowledge graph
- [x] **AUTO-03**: Daily pipeline runs full Gemini+TKG ensemble prediction for each generated question and persists results
- [x] **AUTO-04**: Daily pipeline resolves outcomes for past predictions by comparing against subsequent GDELT events within a configurable time window
- [x] **AUTO-05**: Daily pipeline handles failures with retry logic and sends alerts on consecutive failures

### Dynamic Calibration

- [x] **CAL-01**: System computes per-CAMEO-category ensemble alpha weights from accumulated outcome data, replacing fixed alpha=0.6
- [x] **CAL-02**: System uses hierarchical fallback: 4 super-categories (Verbal Coop, Material Coop, Verbal Conflict, Material Conflict) -> 20 CAMEO root codes as data accumulates
- [x] **CAL-03**: System falls back to global alpha weight when insufficient outcome data exists for a category (configurable minimum sample threshold)
- [x] **CAL-04**: System optimizes alpha weights via scipy L-BFGS-B minimizing Brier score per category
- [x] **CAL-05**: Ensemble predictor dynamically loads per-CAMEO weights from `calibration_weights` table at prediction time
- [x] **CAL-06**: System fetches Polymarket prediction market data for geopolitical questions and displays side-by-side comparison with Geopol's calibrated forecasts -- tracking which source is more accurate over time

### TKG Predictor Replacement

- [x] **TKG-01**: System implements TiRGN algorithm in JAX, porting the global history encoder while reusing existing RE-GCN local encoder
- [x] **TKG-02**: System defines `TKGModelProtocol` (Python Protocol class) abstracting `predict_future_events()` interface for swappable TKG backends
- [x] **TKG-03**: TiRGN implementation achieves measurable MRR improvement over RE-GCN baseline on held-out GDELT test set
- [x] **TKG-04**: TiRGN training completes within 24 hours on RTX 3060 12GB for full GDELT dataset
- [x] **TKG-05**: System supports weekly automated retraining of TiRGN model with the existing retraining scheduler

### Monitoring & Hardening

- [x] **MON-01**: System monitors GDELT feed freshness and alerts when no new data arrives for >1 hour
- [x] **MON-02**: System monitors calibration drift (Brier score trend) and alerts when prediction quality degrades beyond threshold
- [x] **MON-03**: System monitors Gemini API usage and cost, with daily budget cap enforcement
- [x] **MON-04**: System exposes health endpoint reporting status of all subsystems (ingest daemon, last prediction, graph freshness, API budget remaining)
- [x] **MON-05**: System logs all errors and warnings to structured log files with rotation

## v2.1 Requirements

**Defined:** 2026-03-02
**Core Value:** Explainability -- every forecast must provide clear, traceable reasoning paths
**Milestone Goal:** Restructure the single-screen dashboard into a three-screen URL-routed application with progressive disclosure, real data-driven country risk, user-submitted forecast questions, and full-text search -- transforming the v2.0 demo into a usable analytical tool.
**Design document:** `.planning/research/FRONTEND_REDESIGN.md`

### Screen Architecture

- [x] **SCREEN-01**: Frontend uses URL-based routing (`/dashboard`, `/globe`, `/forecasts`) with browser history support -- each screen is bookmarkable and shareable
- [x] **SCREEN-02**: Dashboard screen (`/dashboard`) uses scrollable column layout with collapsible sections, no globe -- freed space allocated to feeds, sources, search, and expanded forecast cards
- [x] **SCREEN-03**: Globe screen (`/globe`) renders full-viewport deck.gl globe with contextual overlay panels that appear on interaction
- [x] **SCREEN-04**: Forecasts screen (`/forecasts`) displays question submission form with LLM-parsed confirmation and queue showing pending/processing/complete status

### Forecast UX

- [x] **FUX-01**: Clicking a forecast card inline-expands to reveal probability bar, ensemble weights, evidence count + top 2-3 evidence summaries, horizon/expiry, calibration metadata, and a "View Full Analysis" button that opens ScenarioExplorer modal
- [x] **FUX-02**: Scenario tree node labels show short text (~40 chars) by default with tooltip on hover revealing full scenario description
- [x] **FUX-03**: Full-text search UI over active forecasts filtering by question text, country, and category -- usable at 30+ forecasts
- [x] **FUX-04**: "My Forecasts" section on Dashboard showing user-submitted questions with status badges (pending -> processing -> complete) and links to completed results
- [x] **FUX-05**: Event Feed panel showing GDELT event stream and RSS article headlines in compact timeline format
- [x] **FUX-06**: Sources panel showing active data sources (GDELT, RSS tiers, Polymarket) with health/staleness indicators

### Globe Interaction

- [x] **GLOBE-01**: Country click on globe opens slide-in panel showing all forecasts for that country, risk timeline, and GDELT event sparkline
- [x] **GLOBE-02**: Choropleth layer colors countries by real aggregate risk score derived from predictions table (not mock data)
- [x] **GLOBE-03**: Layer toggle UI controls for forecast markers, conflict arcs, heatmap, and scenario zone layers

### Backend API

- [x] **BAPI-01**: Remove mock fixture fallback in `forecasts.py` -- return empty results when PostgreSQL has no data for a country; fixes Myanmar-under-Syria bleed-through
- [x] **BAPI-02**: Real country risk aggregation endpoint (`GET /api/v1/countries`) returns per-country forecast_count, risk_score (composite 0-100 index combining count + probability + Goldstein severity, with exponential time decay), trend (rising/stable/falling via 7-day delta), and top_forecast (most recent) -- all computed from `predictions` table
- [x] **BAPI-03**: Question submission queue: new `forecast_requests` table, `POST /api/v1/forecasts/submit` accepting natural language question, LLM parsing to structured form (country_iso, horizon_days, category), `GET /api/v1/forecasts/requests` for status listing
- [x] **BAPI-04**: Full-text search endpoint `GET /api/v1/forecasts/search?q=...&category=...&country=...` using PostgreSQL `ts_vector` + GIN index on `predictions.question`

### Live Data Feeds & Country Depth (Phase 17)

*Requirements TBD -- define during `/gsd:discuss-phase 17`*

Scope areas:
- Backend event/article API endpoints (expose GDELT events and RSS articles as first-class API resources)
- Wire EventTimelinePanel to real GDELT data (replace mock events)
- Additional data source ingestion beyond GDELT + RSS
- Country screen subpages fleshed out with real data (economic indicators, entity graphs, event timelines, source coverage)

### Polymarket-Driven Forecasting (Phase 18)

*Requirements TBD -- define during `/gsd:discuss-phase 18`*

Scope areas:
- Poll Polymarket for active geopolitical questions
- Run Geopol forecasting pipeline on matched Polymarket questions
- Track Geopol probability vs. market price over time
- Dashboard Col 2 panel showing Polymarket questions with Geopol's competing forecasts
- Calibration comparison: head-to-head accuracy tracking

## v3.0 Requirements

**Defined:** 2026-03-04
**Core Value:** Explainability -- every forecast must provide clear, traceable reasoning paths
**Milestone Goal:** Full backend control via admin dashboard, source expansion with feed management, daemon consolidation, historical backtesting, global map seeding, Polymarket operational hardening, and frontend polish -- preparing the system for v4.0 production deployment.
**Research:** `.planning/research/SUMMARY.md` (v3.0), `.planning/research/STACK_V3.md`, `FEATURES.md`, `ARCHITECTURE.md`, `PITFALLS.md`

### Admin Dashboard

- [ ] **ADMIN-01**: Admin screen at `/admin` route with route-level auth gating (API key or password) -- dynamic import code-split so admin code is zero bytes in public bundle
- [ ] **ADMIN-02**: Process table showing all running daemons/jobs (GDELT poller, RSS poller, daily pipeline, Polymarket forecaster, TKG retrainer) with status, last run time, next scheduled run, and success/failure counts
- [ ] **ADMIN-03**: Manual trigger buttons for each daemon job -- operator can force-run any job immediately outside its schedule
- [ ] **ADMIN-04**: Configuration editor for runtime-adjustable settings (polling intervals, daily caps, eval samples, history rate) with validation and persistence to PostgreSQL `system_config` table
- [ ] **ADMIN-05**: Log viewer showing recent structured log entries filterable by severity (ERROR/WARN/INFO) and subsystem -- last 1000 entries from in-memory ring buffer, not filesystem reads
- [ ] **ADMIN-06**: Source management panel showing all data sources (GDELT, RSS feeds, ACLED, UCDP, Polymarket, advisories) with per-source health, enable/disable toggles, and feed-level controls for RSS

### Daemon Consolidation

- [ ] **DAEMON-01**: Single APScheduler 3.x AsyncIOScheduler instance mounted in FastAPI lifespan, replacing all separate systemd timers and daemon processes -- in-memory jobstore with `coalesce=True` for missed runs
- [ ] **DAEMON-02**: All existing pollers (GDELT 15-min, RSS 15-min, daily forecast pipeline, Polymarket auto-forecaster, ACLED daily) registered as APScheduler jobs with configurable schedules
- [ ] **DAEMON-03**: Heavy jobs (daily forecast pipeline, TKG retraining) execute in `ProcessPoolExecutor` to retain OS-level memory isolation -- event loop never blocks
- [ ] **DAEMON-04**: Admin API endpoints for job control: `POST /api/v1/admin/jobs/{id}/pause`, `/resume`, `/trigger` -- exposed to admin dashboard
- [ ] **DAEMON-05**: Graceful shutdown: APScheduler shuts down before FastAPI, in-flight jobs complete or timeout within 30 seconds, no orphaned processes

### Source Expansion & Feed Management

- [ ] **SRC-01**: UCDP event poller fetching armed conflict events from UCDP GED API (token-authenticated), mapping to unified Event schema with "UCDP-" prefixed IDs, inserting into SQLite event store
- [ ] **SRC-02**: UCDP events include fatality counts, conflict type classification, and geographic coordinates (lat/lon) -- enriching the knowledge graph with casualty severity signal
- [ ] **SRC-03**: WM-style RSS feed management: admin can add/remove/categorize RSS feeds, assign tier (1=flagship, 2=regional, 3=niche), set polling frequency per tier, view per-feed health/article counts
- [ ] **SRC-04**: RSS feed configuration stored in PostgreSQL `rss_feeds` table (replacing hardcoded feed list in code) -- admin changes take effect on next polling cycle without restart
- [ ] **SRC-05**: Cross-source deduplication layer preventing the same real-world event from being counted multiple times across GDELT + ACLED + UCDP -- dedup by (date, country, event_type) with configurable similarity threshold
- [ ] **SRC-06**: Per-source health metrics exposed via `/api/v1/sources` endpoint: last poll time, events ingested (24h), error rate, staleness status -- auto-discovered from `ingest_runs` table

### Polymarket Hardening

- [ ] **POLY-01**: Fix `reforecast_active()` bug where `Prediction.created_at` overwrite corrupts daily cap tracking -- reforecast must preserve original creation timestamp
- [ ] **POLY-02**: Cumulative Brier score tracking: system computes rolling Brier score for Geopol predictions vs Polymarket market prices on all resolved matched questions, stored in `polymarket_accuracy` table
- [ ] **POLY-03**: Head-to-head accuracy dashboard panel showing Geopol vs Polymarket Brier score curves over time, per-category breakdown, and win/loss record
- [ ] **POLY-04**: Resolution tracking: system monitors Polymarket question resolution status, detects ambiguous/voided resolutions, and handles them gracefully (exclude from accuracy metrics)
- [ ] **POLY-05**: Polymarket polling reliability: retry logic with exponential backoff, API rate limit compliance, graceful degradation when Polymarket API is unavailable -- no silent failures

### Historical Backtesting

- [ ] **BTEST-01**: Walk-forward evaluation harness: train on [t₀,t₁], predict [t₁,t₂], slide window forward, producing MRR/Brier curves over time -- uses static model weights for v3.0 (full TKG retraining per window deferred)
- [ ] **BTEST-02**: Model comparison framework: run identical evaluation windows against TiRGN and RE-GCN checkpoints, producing side-by-side accuracy tables and delta charts
- [ ] **BTEST-03**: Calibration audit: reliability diagrams computed over sliding time windows showing how calibration quality evolves -- detects calibration drift before it degrades live predictions
- [ ] **BTEST-04**: Backtesting results stored in PostgreSQL `backtest_runs` and `backtest_results` tables -- queryable from admin dashboard, not ephemeral console output
- [ ] **BTEST-05**: Look-ahead bias prevention: backtesting harness uses calibration weight snapshots from each evaluation window (not current weights) and excludes ChromaDB articles published after the prediction date

### Global Seeding & Globe Layers

- [ ] **SEED-01**: Baseline risk computation for all ~195 countries using composite score: GDELT event density + ACLED conflict intensity + UCDP fatality signal + government travel advisory levels, with configurable weights and exponential time decay
- [ ] **SEED-02**: `baseline_country_risk` table in PostgreSQL storing per-country composite scores, updated daily -- API merges with active forecast risk via `COALESCE` (forecast risk overrides baseline when available)
- [ ] **SEED-03**: Globe choropleth renders all ~195 countries with color intensity from merged risk scores -- no more empty/neutral countries with zero data
- [ ] **GLYR-01**: Heatmap layer populated with real GDELT event locations -- requires adding `lat`/`lon` columns to SQLite events schema (currently discarded during ingestion) and a new `/api/v1/events/geo` endpoint returning geocoded events
- [ ] **GLYR-02**: Arcs layer showing bilateral country relationships from knowledge graph edges -- new `/api/v1/countries/relations` endpoint returning top-N country pairs by edge weight
- [ ] **GLYR-03**: Scenarios layer showing geographic zones for active scenario branches -- polygon generation from country ISO codes of scenario-relevant entities

### Frontend Finalization

- [ ] **POLISH-01**: Loading states for all panels and screens -- skeleton placeholders during API fetches, not blank space or frozen UI
- [ ] **POLISH-02**: Error boundaries per panel -- a failed API call in one panel does not crash the entire screen; shows inline error with retry button
- [ ] **POLISH-03**: Empty states for all data-dependent panels -- meaningful messages when no forecasts, no events, no comparisons exist (not blank panels)
- [ ] **POLISH-04**: Performance optimization: lazy-load heavy panels (ScenarioExplorer, CalibrationPanel), debounce search/filter inputs, minimize DOM operations on refresh cycles
- [ ] **POLISH-05**: Accessibility basics: keyboard navigation for all interactive elements, ARIA labels on map controls, sufficient color contrast ratios, focus management on modal open/close

## Future Requirements (Backlog)

Tracked ideas not yet assigned to a milestone. Includes WM-derived features for future cherry-picking.

### WM Feature Backlog (Reference Implementations Exist)

- **WM-FEAT-01**: Variant system -- regional forecast builds (MENA, Europe, Asia-Pacific, Africa) from single codebase via `VITE_VARIANT`
- **WM-FEAT-02**: Prediction markets panel -- Polymarket data alongside Geopol forecasts for calibration comparison
- **WM-FEAT-03**: Command palette (Cmd+K) -- search forecasts, countries, entities *(partially addressed by FUX-03 in v2.1)*
- **WM-FEAT-04**: Playback mode -- replay how forecasts evolved over time (time travel)
- **WM-FEAT-05**: i18n / RTL -- multi-language forecast display
- **WM-FEAT-06**: Virtual scrolling -- for large forecast history lists
- **WM-FEAT-07**: Export (JSON, CSV, PNG) -- forecast report generation
- **WM-FEAT-08**: URL state binding -- shareable deep links (`?country=SY&forecast=abc123`) *(partially addressed by SCREEN-01 URL routing in v2.1)*
- **WM-FEAT-09**: Tauri desktop -- native desktop app with keychain secrets and local API sidecar
- **WM-FEAT-10**: PWA offline support -- cached forecasts viewable without network

### Potential Improvements

- **ADV-03**: Multi-language support (non-English GDELT sources)
- **ADV-05**: ACLED -> knowledge graph (structured conflict events with actor dyads map naturally to CAMEO-like triples; needs CAMEO code mapping layer) -- deferred until Phase 10 proves micro-batch ingest architecture
- **ADV-08**: ~~ICEWS data expansion~~ ICEWS discontinued April 2023. Replaced by POLECAT (PLOVER ontology, incompatible with CAMEO). PLOVER-to-CAMEO mapping feasibility unverified -- defer to v3.1+ after UCDP integration proves multi-source value
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
| Mobile/responsive layout (v2.1) | Three screens + globe = poor mobile experience; desktop-first |
| Real-time collaboration (v2.1) | Multiple users viewing same forecast; single-user system |
| Forecast comparison view (v2.1) | Side-by-side two forecasts; out of scope for this milestone |
| Historical forecast replay (v2.1) | Time travel through past forecasts; see WM-FEAT-04 backlog |
| SSE/WebSocket streaming (v2.1) | Partial results as pipeline progresses; defer unless UX testing shows wait is intolerable |
| Docker/containerization (v3.0) | Deferred to v4.0; gates on daemon consolidation completing first |
| POLECAT/ICEWS replacement (v3.0) | ICEWS dead since April 2023; POLECAT uses incompatible PLOVER ontology; mapping feasibility unverified; defer to v3.1+ |
| Walk-forward TKG retraining per window (v3.0) | Compute-prohibitive for v3.0; backtesting uses static model weights; full walk-forward with per-window retraining is v4.0+ |
| APScheduler 4.x (v3.0) | v4.0.0a6 is alpha with incompatible API rewrite; author warns against production use; pin to 3.x |
| Multi-user admin (v3.0) | Single operator; no RBAC or audit trail beyond structured logs |

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
| CAL-01 | Phase 13 | Complete |
| CAL-02 | Phase 13 | Complete |
| CAL-03 | Phase 13 | Complete |
| CAL-04 | Phase 13 | Complete |
| CAL-05 | Phase 13 | Complete |
| CAL-06 | Phase 13 | Complete |
| TKG-01 | Phase 11 | Complete |
| TKG-02 | Phase 11 | Complete |
| TKG-03 | Phase 11 | Complete |
| TKG-04 | Phase 11 | Complete |
| TKG-05 | Phase 11 | Complete |
| MON-01 | Phase 13 | Complete |
| MON-02 | Phase 13 | Complete |
| MON-03 | Phase 13 | Complete |
| MON-04 | Phase 13 | Complete |
| MON-05 | Phase 13 | Complete |
| BAPI-01 | Phase 14 | Complete |
| BAPI-02 | Phase 14 | Complete |
| BAPI-03 | Phase 14 | Complete |
| BAPI-04 | Phase 14 | Complete |
| SCREEN-01 | Phase 15 | Complete |
| SCREEN-02 | Phase 15 | Complete |
| FUX-01 | Phase 15 | Complete |
| FUX-02 | Phase 15 | Complete |
| FUX-03 | Phase 15 | Complete |
| FUX-04 | Phase 15 | Complete |
| FUX-05 | Phase 15 | Complete |
| FUX-06 | Phase 15 | Complete |
| SCREEN-03 | Phase 16 | Complete |
| SCREEN-04 | Phase 16 | Complete |
| GLOBE-01 | Phase 16 | Complete |
| GLOBE-02 | Phase 16 | Complete |
| GLOBE-03 | Phase 16 | Complete |

**v2.0 Coverage:**
- v2.0 requirements: 50 total (8 INFRA + 7 API + 8 FE + 6 INGEST + 5 AUTO + 6 CAL + 5 TKG + 5 MON)
- Mapped to phases: 50/50
- Unmapped: 0

**v2.0 Phase distribution:**
- Phase 9: 11 requirements (INFRA-01..08, API-01..03, API-07)
- Phase 10: 14 requirements (INGEST-01..06, AUTO-01..05, API-04..06)
- Phase 11: 5 requirements (TKG-01..05)
- Phase 12: 8 requirements (FE-01..08)
- Phase 13: 11 requirements (CAL-01..06, MON-01..05)

**v2.1 Coverage:**
- v2.1 requirements: 17 defined + TBD (Phases 17-18 requirements pending discuss-phase)
- Mapped to phases: 17/17 (defined), Phases 17-18 scope areas identified
- Unmapped: 0

**v2.1 Phase distribution:**
- Phase 14: 4 requirements (BAPI-01..04) -- Complete
- Phase 15: 8 requirements (SCREEN-01, SCREEN-02, FUX-01..06) -- Complete
- Phase 16: 5 requirements (SCREEN-03, SCREEN-04, GLOBE-01..03) -- Complete
- Phase 17: TBD (Live Data Feeds & Country Depth) -- Scope identified, requirements pending
- Phase 18: TBD (Polymarket-Driven Forecasting) -- Scope identified, requirements pending

---
*Requirements defined: 2026-02-14*
*Restructured: 2026-02-27 -- WM-derived frontend replaces Streamlit; API layer added; PostgreSQL replaces SQLite for forecast persistence; CAL-* moved to Phase 13; parallel execution model adopted*
*Updated: 2026-02-27 -- Added INGEST-06 (RSS feed -> RAG enrichment from WM's 298-domain feed list) and CAL-06 (Polymarket calibration comparison)*
*Updated: 2026-03-02 -- v2.0 shipped; v2.1 requirements defined (17 requirements across 4 categories)*
*Updated: 2026-03-02 -- v2.1 roadmap created; 17 requirements mapped to Phases 14-16*
*Updated: 2026-03-03 -- v2.1 extended with Phases 17-18 (Live Data Feeds + Polymarket-Driven Forecasting); milestone scope 14-18*
*Updated: 2026-03-04 -- v2.1 shipped; v3.0 requirements defined (37 requirements across 7 categories). ICEWS marked dead, POLECAT deferred.*
