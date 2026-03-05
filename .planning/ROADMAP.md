# Roadmap: Explainable Geopolitical Forecasting Engine

## Milestones

- v1.0 MVP -- Phases 1-5 (shipped 2026-01-23)
- v1.1 Tech Debt Remediation -- Phases 6-8 (shipped 2026-01-30)
- v2.0 Operationalization & Forecast Quality -- Phases 9-13 (shipped 2026-03-02)
- v2.1 Production UX & Live Data Integration -- Phases 14-18 (shipped 2026-03-04)
- v3.0 Operational Command & Verification -- Phases 19-25 (in progress)

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

<details>
<summary>v2.0 Operationalization & Forecast Quality (Phases 9-13) - SHIPPED 2026-03-02</summary>

**Milestone Goal:** Transform research prototype into publicly demonstrable system with a WM-derived TypeScript dashboard, headless FastAPI backend, automated operations, upgraded TKG predictor, and self-improving calibration.

**Architecture Decision (2026-02-27):** World Monitor used as reference architecture and code quarry -- not as a live integration target. Geopol becomes a headless Python forecast engine with its own WM-derived TypeScript frontend. See `.planning/research/WM_AS_REPOSITORY.md` for full analysis and `WORLDMONITOR_INTEGRATION.md` for DTO contract spec.

**Phase Numbering:** Starts at 9 (v1.1 ended at Phase 8; cancelled Llama-TGL phases 9-14 archived in `.planning/archive/v2.0-llama-cancelled.md`).

**Execution Model:** Parallel after Phase 9. Phases 10, 11, and 12 run concurrently once Phase 9 establishes the API contract (DTOs + mock fixtures). Phase 13 waits for all three.

```
Phase 9 (API + DB foundation) --- critical path, everything gates on this
    |
    |---> Phase 10 (ingest + pipeline + real API data)
    |
    |---> Phase 11 (TKG replacement) ---- parallelizable
    |
    +---> Phase 12 (frontend against mock API -> real API when Phase 10 lands)
                |
                +-----------> Phase 13 (monitoring + calibration + hardening)
```

- [x] **Phase 9: API Foundation & Infrastructure** -- PostgreSQL, FastAPI skeleton with DTOs and mock fixtures, structured logging, jraph elimination, TKGModelProtocol
- [x] **Phase 10: Ingest & Forecast Pipeline** -- Micro-batch GDELT ingest, daily forecast automation, real API endpoints replacing mocks, Redis caching
- [x] **Phase 11: TKG Predictor Replacement** -- TiRGN JAX port replacing RE-GCN for improved accuracy (parallelizable with Phases 10 and 12)
- [x] **Phase 12: WM-Derived Frontend** -- TypeScript dashboard scaffolded from World Monitor patterns: deck.gl globe, forecast panels, scenario explorer, country briefs, map layers
- [x] **Phase 13: Calibration, Monitoring & Hardening** -- Dynamic per-CAMEO calibration from accumulated outcome data, system health observability, alerting, operational resilience

### Phase 9: API Foundation & Infrastructure
**Goal**: System has the persistence layer, headless API server with mock data, and cleaned dependency tree that every v2.0 feature requires. The API DTOs and mock fixtures establish the contract that Phases 10 and 12 develop against independently.
**Depends on**: Phase 8 (v1.1 complete)
**Requirements**: INFRA-01, INFRA-02, INFRA-03, INFRA-04, INFRA-05, INFRA-06, INFRA-07, INFRA-08, API-01, API-02, API-03, API-07
**Success Criteria** (what must be TRUE):
  1. `GET /api/v1/health` returns JSON reporting subsystem status (database connected, graph partition count, model loaded)
  2. `GET /api/v1/forecasts/country/SY` returns a mock `ForecastResponse` with structurally valid scenarios, evidence, and calibration data -- the response matches the DTO contract spec
  3. `POST /api/v1/forecasts` with a valid API key accepts a question and returns a mock forecast; without a valid API key returns 401
  4. Running `EnsemblePredictor.predict()` persists the forecast to PostgreSQL and the row is retrievable via both SQL and `GET /api/v1/forecasts/{id}`
  5. Three separate Python processes (FastAPI server, simulated ingest daemon, simulated prediction pipeline) can read/write the PostgreSQL database concurrently without errors or data corruption
  6. All production code paths emit structured log messages (no `print()` statements remain) with severity, timestamp, and module name
  7. Importing the TKG training module succeeds without jraph installed -- all jraph references replaced with local JAX equivalents
  8. `TKGModelProtocol` is defined and both the existing RE-GCN wrapper and a stub TiRGN class satisfy it (verified by `isinstance` or Protocol structural check)
**Plans**: 6 plans

Plans:
- [x] 09-01-PLAN.md -- Dependencies, Docker, PostgreSQL ORM models, Alembic migrations, settings
- [x] 09-02-PLAN.md -- Pydantic V2 DTOs (contract spec, 8-subsystem health schema) and mock fixtures (SY, UA, MM)
- [x] 09-03-PLAN.md -- jraph elimination, TKGModelProtocol, structured logging config module
- [x] 09-04-PLAN.md -- print() to logging conversion sweep (9 production files)
- [x] 09-05-PLAN.md -- FastAPI app, full subsystem health endpoint (8 checks), routes, auth middleware, error handling
- [x] 09-06-PLAN.md -- ForecastService persistence bridge, route wiring, multi-process concurrent DB tests, table smoke writes

### Phase 10: Ingest & Forecast Pipeline
**Goal**: System continuously ingests GDELT events every 15 minutes and produces daily automated forecasts with outcome tracking. API endpoints serve real forecast data (replacing Phase 9 mock fixtures). Redis caching prevents redundant computation.
**Depends on**: Phase 9
**Requirements**: INGEST-01, INGEST-02, INGEST-03, INGEST-04, INGEST-05, INGEST-06, AUTO-01, AUTO-02, AUTO-03, AUTO-04, AUTO-05, API-04, API-05, API-06
**Success Criteria** (what must be TRUE):
  1. Ingest daemon runs for 1+ hours, fetching GDELT updates every 15 minutes, and the `ingest_runs` table shows sequential successful runs with non-zero event counts and no duplicate `GlobalEventID` values across runs
  2. A missed or failed GDELT feed fetch triggers exponential backoff retries with logged warnings, and the daemon resumes normal operation when the feed recovers
  3. Daily pipeline (triggered via systemd timer or manual invocation) generates forecast questions from recent high-significance events, runs Gemini+TKG ensemble predictions, and persists results to the `predictions` table
  4. `GET /api/v1/forecasts/country/{iso}` returns real forecasts generated by the daily pipeline (not mock fixtures); responses are served from Redis cache on subsequent requests within TTL
  5. `POST /api/v1/forecasts` generates a live forecast with input sanitization -- crafted prompt-injection inputs do not leak system internals; requests are rejected when daily Gemini budget is exhausted
  6. After sufficient time passes, the daily pipeline resolves past predictions against GDELT ground truth and writes outcome records to the `outcome_records` table
  7. A consecutive daily pipeline failure triggers an alert (log or notification) and the system recovers on the next scheduled run without manual intervention
  8. RSS ingest daemon polls WM-curated feed list, and querying ChromaDB for a recent geopolitical topic returns article chunks from RSS sources (not just GDELT event descriptions)
**Plans**: 4 plans

Plans:
- [x] 10-01-PLAN.md -- GDELT micro-batch poller daemon with backoff, metrics, incremental graph update
- [x] 10-02-PLAN.md -- API hardening: three-tier cache, per-key rate limiting, input sanitization
- [x] 10-03-PLAN.md -- RSS feed ingestion daemon with tiered polling and ChromaDB indexing
- [x] 10-04-PLAN.md -- Daily forecast pipeline, outcome resolution, real API route wiring

### Phase 11: TKG Predictor Replacement
**Goal**: TiRGN replaces RE-GCN as the TKG backend, delivering measurable accuracy improvement while fitting the RTX 3060 training envelope and weekly retraining cadence
**Depends on**: Phase 9 (TKGModelProtocol defined, jraph eliminated)
**Requirements**: TKG-01, TKG-02, TKG-03, TKG-04, TKG-05
**Research flag**: YES -- no published JAX implementation of TiRGN exists. Plan-phase should evaluate whether `/gsd:research-phase` is needed before implementation.
**Success Criteria** (what must be TRUE):
  1. TiRGN model trains to completion on the full GDELT dataset within 24 hours on RTX 3060 12GB without OOM
  2. TiRGN achieves higher MRR than RE-GCN on a held-out GDELT test set (improvement logged and reproducible)
  3. Swapping `TKGPredictor` from RE-GCN to TiRGN requires only a config change -- no downstream code modifications to `EnsemblePredictor`, calibration, or the daily pipeline
  4. Weekly automated retraining (via existing scheduler) completes successfully with the TiRGN model
**Plans**: 3 plans

Plans:
- [x] 11-01-PLAN.md -- TiRGN model module (global history encoder, Time-ConvTransE decoder, copy-generation fusion, protocol compliance)
- [x] 11-02-PLAN.md -- TiRGN training loop with observability (TensorBoard + W&B, early stopping, VRAM monitoring)
- [x] 11-03-PLAN.md -- Backend dispatch, scheduler integration, config-only swap, integration tests

### Phase 12: WM-Derived Frontend
**Goal**: External visitors see a production-quality dashboard with deck.gl globe, forecast panels, interactive scenario exploration, and country briefs -- all consuming Geopol's FastAPI backend. Architecturally derived from World Monitor's vanilla TypeScript patterns but purpose-built for geopolitical forecasting.
**Depends on**: Phase 9 (API contract with mock fixtures -- real data from Phase 10 is not required to start; frontend develops against mocks)
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
  8. `ForecastServiceClient` implements circuit breaker pattern -- API failures result in stale-data display with "unavailable" indicator, not a broken UI
**Plans**: 7 plans

Plans:
- [x] 12-01-PLAN.md -- Project scaffold (Vite, TypeScript, Panel base class, AppContext, h() helper, theme system, API types)
- [x] 12-02-PLAN.md -- ForecastServiceClient with CircuitBreaker (typed API access, resilience, deduplication)
- [x] 12-03-PLAN.md -- DeckGLMap globe with 5 map layers (choropleth, markers, arcs, heatmap, scenario zones)
- [x] 12-04-PLAN.md -- Dashboard panels (ForecastPanel, RiskIndexPanel, EventTimelinePanel, EnsembleBreakdownPanel, SystemHealthPanel, CalibrationPanel)
- [x] 12-05-PLAN.md -- ScenarioExplorer modal (d3-hierarchy tree visualization, evidence sidebar)
- [x] 12-06-PLAN.md -- Country Brief Page (6-tab full-screen modal with entity graph, calibration charts)
- [x] 12-07-PLAN.md -- Integration wiring (main.ts, panel layout, RefreshScheduler, end-to-end verification)

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
**Plans**: 7 plans

Plans:
- [x] 13-01-PLAN.md -- DB schema extensions (3 new tables, widened CalibrationWeight, Prediction.cameo_root_code) + settings
- [x] 13-02-PLAN.md -- Calibration optimizer (L-BFGS-B), weight loader (hierarchical fallback), cold-start priors
- [x] 13-03-PLAN.md -- Monitoring package (alert manager, feed/drift/budget/disk monitors)
- [x] 13-04-PLAN.md -- Structured log rotation + systemd service units
- [x] 13-05-PLAN.md -- Polymarket client, matcher, comparison service
- [x] 13-06-PLAN.md -- Integration wiring (ensemble dynamic alpha, health enrichment, pipeline hooks)
- [x] 13-07-PLAN.md -- Calibration API routes, daily digest, frontend CalibrationPanel Polymarket display

</details>

<details>
<summary>v2.1 Production UX & Live Data Integration (Phases 14-18) - SHIPPED 2026-03-04</summary>

**Milestone Goal:** Restructure the single-screen dashboard into a three-screen URL-routed application with progressive disclosure, real data-driven country risk, user-submitted forecast questions, full-text search, live data feeds, deep country drill-down, and Polymarket-driven comparative forecasting -- transforming the v2.0 demo into a usable analytical tool.

**Design document:** `.planning/research/FRONTEND_REDESIGN.md`

**Phase Numbering:** Starts at 14 (v2.0 ended at Phase 13).

**Execution Model:** Sequential. Phase 14 (backend) unblocks Phases 15 and 16. Phase 15 (routing + dashboard) establishes the screen scaffold that Phase 16 (globe + forecasts screens) builds on. Phases 17 and 18 are independent of each other but both depend on Phase 16 (screens exist).

```
Phase 14 (backend API hardening) --- unblocks real data for frontend
    |
    +---> Phase 15 (URL routing + dashboard screen) --- establishes 3-screen scaffold
              |
              +---> Phase 16 (globe + forecasts screens) --- completes remaining screens
                        |
                        |---> Phase 17 (live data feeds + country depth) --- makes screens data-rich
                        |
                        +---> Phase 18 (Polymarket-driven forecasting) --- comparative calibration
```

- [x] **Phase 14: Backend API Hardening** -- Kill fixture fallback, real country risk aggregation, question submission queue, full-text search endpoint
- [x] **Phase 15: URL Routing & Dashboard Screen** -- Three-screen URL routing, information-dense dashboard with progressive disclosure, search UI, event feed, sources panel
- [x] **Phase 16: Globe & Forecasts Screens** -- Full-viewport globe with contextual drill-down, forecast submission queue UI, layer toggle controls
- [x] **Phase 17: Live Data Feeds & Country Depth** -- Backend event/article API endpoints, wire EventTimelinePanel to real data, additional data source ingestion, country screen subpages with real data
- [x] **Phase 18: Polymarket-Driven Forecasting** -- Poll Polymarket for active geopolitical questions, run Geopol pipeline on matched questions, comparison tracking over time, Col 2 Polymarket panel

### Phase 14: Backend API Hardening
**Goal**: API endpoints serve real aggregated data instead of mock fixtures, accept user-submitted forecast questions via a processing queue, and support full-text search over the predictions table. The frontend can request country risk, submit questions, and search forecasts against production data.
**Depends on**: Phase 13 (v2.0 complete)
**Requirements**: BAPI-01, BAPI-02, BAPI-03, BAPI-04
**Success Criteria** (what must be TRUE):
  1. `GET /api/v1/forecasts/country/MM` returns an empty result set (not Syria's forecasts) when no Myanmar predictions exist in PostgreSQL -- the fixture fallback code path is gated behind `USE_FIXTURES=1` (default off), not active in production
  2. `GET /api/v1/countries` returns a JSON array where each entry has `country_iso`, `forecast_count`, `risk_score` (composite 0-100 index: count + probability + Goldstein severity with exponential time decay), `trend` (rising/stable/falling via 7-day delta), and `top_forecast` (most recent) -- all computed from the `predictions` table via SQL aggregation, not hardcoded
  3. `POST /api/v1/forecasts/submit` accepts a natural language question, returns a `request_id` and LLM-parsed structured form (country_iso_list, horizon_days, category) supporting multi-country questions, and the request appears in the `forecast_requests` table with status `pending`
  4. `GET /api/v1/forecasts/search?q=conflict&country=UA` returns forecasts matching the query using PostgreSQL `ts_vector` full-text search with sub-200ms response time on 1000+ predictions
  5. `GET /api/v1/forecasts/requests` returns the user's submitted questions with current status (pending/processing/complete/failed) and links to completed forecast results
**Plans**: 4 plans

Plans:
- [x] 14-01-PLAN.md -- DB schema (forecast_requests + tsvector), fixture removal (BAPI-01), USE_FIXTURES dev flag
- [x] 14-02-PLAN.md -- Real country risk aggregation from PostgreSQL with CTE-based scoring (BAPI-02)
- [x] 14-03-PLAN.md -- Full-text search endpoint using tsvector + GIN index (BAPI-04)
- [x] 14-04-PLAN.md -- Question submission queue: LLM parsing, submit/confirm flow, async worker (BAPI-03)

### Phase 15: URL Routing & Dashboard Screen
**Goal**: The single-screen layout is replaced by three URL-routed screens. The Dashboard screen (`/dashboard`) is a dense information display with progressive disclosure on forecast cards, full-text search, event feed, data source health, and a "My Forecasts" section showing user-submitted questions.
**Depends on**: Phase 14 (real country risk, search endpoint, submission queue)
**Requirements**: SCREEN-01, SCREEN-02, FUX-01, FUX-02, FUX-03, FUX-04, FUX-05, FUX-06
**Success Criteria** (what must be TRUE):
  1. Navigating to `/dashboard`, `/globe`, and `/forecasts` renders three distinct screens; the browser URL updates on navigation; back/forward buttons work; direct URL entry loads the correct screen
  2. The Dashboard screen displays scrollable columns with collapsible sections and no globe -- the freed viewport space is filled with forecast cards, event feed, and source health panels
  3. Clicking a forecast card inline-expands to show ensemble weights, top evidence summaries, calibration metadata, and a "View Full Analysis" button that opens the ScenarioExplorer modal
  4. Typing in the search bar filters visible forecasts in real-time by question text, country, or category; results update as the user types (debounced) against the BAPI-04 search endpoint
  5. The "My Forecasts" section shows user-submitted questions with status badges (pending/processing/complete) and clicking a completed forecast navigates to its full result
  6. Scenario tree node labels display short text (~40 chars) by default; hovering over a node shows a tooltip with the full scenario description
  7. An Event Feed panel renders recent GDELT events and RSS article headlines in a compact timeline format with timestamps and source attribution
  8. A Sources panel displays active data sources (GDELT, RSS tiers, Polymarket) with health/staleness indicators showing last-updated time and status
**Plans**: 3 plans

Plans:
- [x] 15-01-PLAN.md -- Router + NavBar + 4-column layout + screen lifecycle + theme simplification + top_forecast rename
- [x] 15-02-PLAN.md -- ForecastPanel expandable cards + SearchBar + ScenarioExplorer tooltip (FUX-01, FUX-02, FUX-03)
- [x] 15-03-PLAN.md -- MyForecastsPanel + SourcesPanel + dashboard Col 3 wiring (FUX-04, FUX-05, FUX-06)

### Phase 16: Globe & Forecasts Screens
**Goal**: The Globe screen provides full-viewport geospatial exploration with contextual drill-down on country click. The Forecasts screen provides a question submission workflow with queue status display. All three screens are complete and the v2.1 milestone is deliverable.
**Depends on**: Phase 15 (URL routing scaffold, dashboard screen complete)
**Requirements**: SCREEN-03, SCREEN-04, GLOBE-01, GLOBE-02, GLOBE-03
**Success Criteria** (what must be TRUE):
  1. The Globe screen (`/globe`) renders deck.gl at full viewport with overlay panels that appear contextually on interaction -- no permanent sidebar competing for space
  2. Clicking a country on the globe opens a slide-in panel showing that country's active forecasts, a risk score timeline, and a GDELT event sparkline -- all from live API data via BAPI-02
  3. The choropleth layer colors countries by real aggregate risk scores from the `predictions` table (via BAPI-02); countries with no forecasts render as neutral/uncolored
  4. Layer toggle controls are visible on the Globe screen allowing the user to enable/disable forecast markers, conflict arcs, heatmap, and scenario zone layers independently
  5. The Forecasts screen (`/forecasts`) displays a question submission form; submitting a question shows LLM-parsed confirmation (country, horizon, category) and the question enters the processing queue with real-time status updates
**Plans**: 3 plans

Plans:
- [x] 16-01-PLAN.md -- Shared expandable card utility extraction + DeckGLMap public API extensions (flyToCountry, setLayerVisible, remove built-in toggles)
- [x] 16-02-PLAN.md -- Globe screen: full-viewport globe with GlobeHud, LayerPillBar, GlobeDrillDown, data loading, refresh scheduling (SCREEN-03, GLOBE-01, GLOBE-02, GLOBE-03)
- [x] 16-03-PLAN.md -- Forecasts screen: SubmissionForm (three-state inline transform), SubmissionQueue (status badges, elapsed timer, expandable completed forecasts), two-column layout (SCREEN-04)

### Phase 17: Live Data Feeds & Country Depth
**Goal**: Every panel and country screen displays real, live data from multiple sources. Backend exposes event and article API endpoints with full filter surfaces. EventTimelinePanel shows real GDELT+ACLED events. Country screens are populated with meaningful content across all tabs. ACLED conflict data and US/UK government travel advisories are ingested as new data sources. SourcesPanel auto-discovers active sources from the backend.
**Depends on**: Phase 16 (all screens exist)
**Success Criteria** (what must be TRUE):
  1. `GET /api/v1/events?country=UA&limit=50` returns paginated GDELT+ACLED events for Ukraine with cursor-based pagination, filterable by date range, CAMEO code, actor, Goldstein range, text search, and source
  2. `GET /api/v1/articles?text=conflict&semantic=true` returns ChromaDB vector similarity results for articles; keyword mode returns metadata-filtered results
  3. `GET /api/v1/sources` returns health/staleness for all data sources (gdelt, rss, acled, advisory) auto-discovered from IngestRun table -- adding a new source on the backend auto-appears without frontend changes
  4. `GET /api/v1/advisories?country=UA` returns US State Dept and UK FCDO travel advisories for Ukraine with normalized risk levels (1-4)
  5. EventTimelinePanel on the dashboard shows real events from /events API with diff-based DOM updates preserving expanded card state across 30s refresh cycles -- no mock data remains
  6. SourcesPanel displays auto-discovered data sources from /sources endpoint, not filtered health subsystems
  7. CountryBriefPage events tab shows events filtered by country; risk-signals tab shows government advisories; entities tab shows top actors with event counts -- all from real API data
  8. ACLED poller daemon fetches armed conflict events (Battles, Explosions, Violence against civilians) daily, maps to unified Event schema, and inserts into SQLite with "ACLED-" prefixed IDs
  9. Advisory poller daemon fetches US State Dept + UK FCDO travel advisories daily and populates in-memory cache served by /advisories endpoint
**Plans**: 3 plans

Plans:
- [x] 17-01-PLAN.md -- Data layer foundation: SQLite schema migration (country_iso + source columns, backfill, indexes), Event model update, EventStorage query expansion, Pydantic DTOs, settings
- [x] 17-02-PLAN.md -- Backend API routes (events, articles, sources, advisories) + ACLED poller + advisory poller
- [x] 17-03-PLAN.md -- Frontend wiring: TypeScript types, forecast-client methods, EventTimelinePanel, SourcesPanel, CountryBriefPage tabs

### Phase 18: Polymarket-Driven Forecasting
**Goal**: The system actively polls Polymarket for geopolitical questions, runs Geopol's forecasting pipeline on matching questions, and tracks probability comparisons over time. A dedicated dashboard panel shows Polymarket questions alongside Geopol's competing forecasts, providing direct calibration signal and demonstrable accuracy comparison.
**Depends on**: Phase 16 (screens exist); Phase 13 (existing Polymarket client, matcher, comparison service)
**Success Criteria** (what must be TRUE):
  1. Unmatched Polymarket geopolitical questions above $100K volume automatically trigger full EnsemblePredictor pipeline runs, persisting results with `provenance="polymarket_driven"` and creating PolymarketComparison tracking rows
  2. Active comparisons are re-forecasted daily (overwriting existing Prediction rows), with probability changes captured in polymarket_snapshots time-series
  3. Daily caps (3 new + 5 re-forecasts) and Gemini budget checks prevent runaway API costs; same Polymarket question never triggers duplicate pipeline runs
  4. Forecast cards in the dashboard show a "P" badge on collapsed state when linked to a Polymarket market; expanding the card shows market price, divergence (pp), and a dual-line sparkline (lazy-loaded from /calibration/polymarket/comparisons/{id}/snapshots)
  5. ComparisonPanel in Col 2 (below Active Forecasts) displays all active and resolved comparisons with dual probability bars, divergence color coding, provenance badges, and resolved status indicators
  6. `GET /calibration/polymarket/comparisons` returns all comparisons with provenance and divergence data; `GET /calibration/polymarket/comparisons/{id}/snapshots` returns sampled sparkline data (30 points)
**Plans**: 3 plans

Plans:
- [x] 18-01-PLAN.md -- DB schema (provenance + polymarket_event_id on Prediction), PolymarketAutoForecaster (volume filter, tiered extraction, pipeline trigger, cap tracking, dedup), app.py wiring, comparison.py snapshot query
- [x] 18-02-PLAN.md -- ForecastResponse DTO extension (polymarket_comparison field), forecast enrichment, comparison panel + snapshot API endpoints
- [x] 18-03-PLAN.md -- Frontend: TypeScript types, forecast-client methods, badge + inline comparison on expandable cards, ComparisonPanel, dashboard wiring

</details>

### v3.0 Operational Command & Verification

**Milestone Goal:** Full backend control via admin dashboard, source expansion with feed management, daemon consolidation, historical backtesting, global map seeding, Polymarket operational hardening, and frontend polish -- preparing the system for v4.0 production deployment.

**Research:** `.planning/research/SUMMARY.md` (v3.0), `.planning/research/STACK_V3.md`, `FEATURES.md`, `ARCHITECTURE.md`, `PITFALLS.md`

**Phase Numbering:** Starts at 19 (v2.1 ended at Phase 18).

**Execution Model:** Phase 19 (admin) establishes the observation layer. Phase 20 (daemon consolidation) is the critical path -- everything after it registers APScheduler jobs. Phases 21 and 22 are independent parallel tracks after Phase 20. Phase 23 depends on Phase 22 (clean resolution data). Phase 24 depends on Phase 21 (UCDP data feeds baseline risk). Phase 25 depends on Phases 23 and 24 (all features exist to polish).

```
Phase 19 (Admin Dashboard) --- observability layer first
    |
    +---> Phase 20 (Daemon Consolidation) --- critical path
              |
              |---> Phase 21 (Source Expansion) ---> Phase 24 (Global Seeding + Globe Layers)
              |                                                  |
              +---> Phase 22 (Polymarket Hardening) ---> Phase 23 (Backtesting)
                                                                 |
                                                                 +---> Phase 25 (Frontend Polish)
```

Phases 21 and 22 are independent tracks after Phase 20 and can run in parallel.

**Hard constraint:** `uvicorn --workers 1` -- APScheduler in-process is not safe with multiple workers (each worker creates its own scheduler, running every job N times).

**Pre-phase blocker (Phase 21):** UCDP API token must be requested from the UCDP team before implementation begins. Email-gated, takes days to weeks.

- [x] **Phase 19: Admin Dashboard Foundation** -- `/admin` route with auth gating, process table, manual triggers, config editor, log viewer, source management panel
- [x] **Phase 20: Daemon Consolidation** -- APScheduler AsyncIOScheduler in FastAPI lifespan, all pollers registered as jobs, ProcessPoolExecutor for heavy jobs, admin API for job control, systemd retirement
- [ ] **Phase 21: Source Expansion & Feed Management** -- UCDP poller, RSS feed management via admin, cross-source dedup layer, per-source health metrics, feed health auto-disable
- [ ] **Phase 22: Polymarket Hardening** -- Fix created_at overwrite bug, cumulative Brier score tracking, head-to-head accuracy panel, resolution tracking, polling reliability
- [ ] **Phase 23: Historical Backtesting** -- Walk-forward evaluation harness, model comparison (TiRGN vs RE-GCN), calibration audit (reliability diagrams over time), look-ahead bias prevention
- [ ] **Phase 24: Global Seeding & Globe Layers** -- Baseline risk for all ~195 countries, heatmap/arcs/scenarios data wiring, advisory-level risk floors, active forecast override
- [ ] **Phase 25: Frontend Finalization** -- Loading states, error boundaries, empty states, performance optimization, accessibility basics

## Phase Details

### Phase 19: Admin Dashboard Foundation
**Goal**: Operators can monitor and control the entire system from a browser without SSH access. The `/admin` route provides real-time visibility into all running jobs, data source health, system configuration, and recent log output -- establishing the observation layer that every subsequent phase extends with new panels and controls.
**Depends on**: Phase 18 (v2.1 complete)
**Requirements**: ADMIN-01, ADMIN-02, ADMIN-03, ADMIN-04, ADMIN-05, ADMIN-06
**Success Criteria** (what must be TRUE):
  1. Navigating to `/admin` prompts for authentication; entering a valid admin key loads the admin dashboard; entering an invalid key shows an error and loads nothing -- admin TypeScript code is zero bytes in the public bundle (dynamic import code-split)
  2. The process table displays all running daemons/jobs (GDELT poller, RSS poller, daily pipeline, Polymarket forecaster, TKG retrainer) with current status, last run time, next scheduled run, and success/failure counts -- data sourced from `ingest_runs` table and job metadata
  3. Manual trigger buttons next to each job allow the operator to force-run any job immediately; clicking a trigger button initiates the job and the process table updates to reflect the running state
  4. The configuration editor displays runtime-adjustable settings (polling intervals, daily caps, eval samples) with input validation; saving persists changes to the `system_config` PostgreSQL table and changes take effect on the next job execution without restart
  5. The log viewer shows the most recent structured log entries (last 1000 from in-memory ring buffer) filterable by severity (ERROR/WARN/INFO) and subsystem -- not reading from the filesystem
**Plans**: 3 plans

Plans:
- [x] 19-01-PLAN.md -- Admin backend: ring buffer, SystemConfig model, admin API endpoints
- [x] 19-02-PLAN.md -- Admin frontend shell: route, auth modal, layout, admin client, styles
- [x] 19-03-PLAN.md -- Admin panels: ProcessTable, ConfigEditor, LogViewer, SourceManager

### Phase 20: Daemon Consolidation
**Goal**: All background jobs run under a single APScheduler instance inside the FastAPI process, replacing scattered systemd timers and standalone daemon processes. The admin dashboard's pause/resume/trigger controls become functional. Heavy jobs retain OS-level memory isolation via ProcessPoolExecutor.
**Depends on**: Phase 19 (admin dashboard provides observability for the consolidated scheduler)
**Requirements**: DAEMON-01, DAEMON-02, DAEMON-03, DAEMON-04, DAEMON-05
**Success Criteria** (what must be TRUE):
  1. Starting the FastAPI server also starts the APScheduler instance; `GET /api/v1/admin/jobs` returns a list of all registered jobs with their trigger type, next run time, and current state -- no separate systemd services need to be started for GDELT polling, RSS polling, or the daily forecast pipeline
  2. The GDELT poller runs every 15 minutes, the RSS poller runs every 15 minutes (tiered), and the daily forecast pipeline runs at 06:00 -- all as APScheduler jobs with the same behavior as their predecessor implementations, verified by checking `ingest_runs` table entries
  3. The daily forecast pipeline and TKG retraining execute in a `ProcessPoolExecutor` (not on the event loop) -- confirmed by the FastAPI server remaining responsive to HTTP requests during pipeline execution
  4. `POST /api/v1/admin/jobs/{id}/pause` pauses a job (no future executions until resumed); `POST /api/v1/admin/jobs/{id}/resume` resumes it; `POST /api/v1/admin/jobs/{id}/trigger` fires it immediately -- all reflected in the admin dashboard process table within seconds
  5. Stopping the FastAPI server gracefully shuts down APScheduler first, waits for in-flight jobs to complete (up to 30 seconds), then shuts down the API -- no orphaned processes or database connections remain
**Plans**: 3 plans

Plans:
- [x] 20-01-PLAN.md -- Scheduler package: APScheduler core, dependency container, 9 job wrappers, failure tracking
- [x] 20-02-PLAN.md -- FastAPI integration: lifespan mount, admin API rewire, graceful shutdown
- [x] 20-03-PLAN.md -- Frontend ProcessTable update with pause/resume controls + verification

### Phase 21: Source Expansion & Feed Management
**Goal**: The system ingests UCDP armed conflict events as a new data source, provides admin-level RSS feed management (enable/disable/tier per feed), prevents cross-source event duplication in the knowledge graph, and exposes per-source health metrics through the API and admin dashboard.
**Depends on**: Phase 20 (new pollers register as APScheduler jobs)
**Pre-phase blocker**: UCDP API token must be requested before implementation. Fallback: bulk CSV download from ucdp.uu.se/downloads/ if token is delayed.
**Requirements**: SRC-01, SRC-02, SRC-03, SRC-04, SRC-05, SRC-06
**Success Criteria** (what must be TRUE):
  1. The UCDP poller fetches armed conflict events daily from the UCDP GED API, maps them to the unified Event schema with "UCDP-" prefixed IDs, and inserts them into the SQLite event store -- events include fatality counts, conflict type classification, and geographic coordinates
  2. An admin can enable/disable individual RSS feeds and change their tier assignment from the admin dashboard; changes persist to the `rss_feeds` PostgreSQL table and take effect on the next polling cycle without server restart
  3. The same real-world conflict event appearing in both GDELT and UCDP does not create duplicate triples in the knowledge graph -- the cross-source dedup layer filters by (date, country, event_type) fingerprint hash before graph insertion
  4. `GET /api/v1/sources` returns per-source health metrics (last poll time, events ingested in 24h, error rate, staleness status) for all active data sources including UCDP -- auto-discovered from the `ingest_runs` table
  5. A feed that fails N consecutive times is automatically disabled and flagged in the admin dashboard source management panel with an alert
**Plans**: 3 plans

Plans:
- [ ] 20-01-PLAN.md -- Scheduler package: APScheduler core, dependency container, 9 job wrappers, failure tracking
- [ ] 20-02-PLAN.md -- FastAPI integration: lifespan mount, admin API rewire, graceful shutdown
- [ ] 20-03-PLAN.md -- Frontend ProcessTable update with pause/resume controls + verification

### Phase 22: Polymarket Hardening
**Goal**: Polymarket-driven forecasting operates reliably with correct budget tracking, and the system maintains rigorous cumulative accuracy metrics comparing Geopol predictions against Polymarket market prices on all resolved questions.
**Depends on**: Phase 20 (Polymarket poller registered as APScheduler job)
**Requirements**: POLY-01, POLY-02, POLY-03, POLY-04, POLY-05
**Success Criteria** (what must be TRUE):
  1. Re-forecasting an active Polymarket comparison preserves the original `created_at` timestamp on the Prediction row -- the `reforecasted_at` column tracks the re-forecast time separately, and daily cap tracking correctly distinguishes new forecasts from re-forecasts
  2. The `polymarket_accuracy` table contains cumulative Brier scores computed from all resolved Polymarket comparisons, updated after each resolution -- both Geopol and Polymarket scores are tracked independently
  3. The admin dashboard displays a head-to-head accuracy panel showing Geopol vs Polymarket Brier score curves over time, per-category breakdown, and a win/loss record on resolved questions
  4. The system detects ambiguous or voided Polymarket question resolutions and excludes them from accuracy metrics -- these cases are logged and visible in the admin dashboard
  5. Polymarket API failures trigger exponential backoff retries; extended Polymarket unavailability degrades gracefully (existing comparisons continue with stale market prices, no silent data loss) -- the poller status shows "degraded" in the admin process table
**Plans**: 3 plans

Plans:
- [ ] 20-01-PLAN.md -- Scheduler package: APScheduler core, dependency container, 9 job wrappers, failure tracking
- [ ] 20-02-PLAN.md -- FastAPI integration: lifespan mount, admin API rewire, graceful shutdown
- [ ] 20-03-PLAN.md -- Frontend ProcessTable update with pause/resume controls + verification

### Phase 23: Historical Backtesting
**Goal**: The system can evaluate its own historical prediction accuracy through walk-forward evaluation, compare TiRGN vs RE-GCN model performance, and audit calibration quality over time -- all as an internal reporting system accessible from the admin dashboard, not public-facing.
**Depends on**: Phase 22 (clean resolution data needed for meaningful Brier scoring)
**Requirements**: BTEST-01, BTEST-02, BTEST-03, BTEST-04, BTEST-05
**Success Criteria** (what must be TRUE):
  1. The walk-forward evaluation harness trains on [t0, t1], predicts [t1, t2], slides the window forward, and produces MRR and Brier score curves over time -- using static model weights (no per-window TKG retraining in v3.0)
  2. The model comparison framework runs identical evaluation windows against TiRGN and RE-GCN checkpoints, producing side-by-side accuracy tables showing MRR, Hits@1, Hits@10, and Brier scores with delta highlighting
  3. Calibration audit generates reliability diagrams computed over sliding time windows, visible in the admin BacktestingPanel -- an operator can observe whether calibration quality is improving, stable, or degrading over time
  4. Backtesting results persist in the `backtest_runs` and `backtest_results` PostgreSQL tables -- results are queryable from the admin dashboard, not ephemeral console output that disappears on restart
  5. The backtesting harness uses calibration weight snapshots from each evaluation window (not current weights) and excludes ChromaDB articles published after the prediction date -- preventing look-ahead bias from corrupting accuracy metrics
**Plans**: 3 plans

Plans:
- [ ] 20-01-PLAN.md -- Scheduler package: APScheduler core, dependency container, 9 job wrappers, failure tracking
- [ ] 20-02-PLAN.md -- FastAPI integration: lifespan mount, admin API rewire, graceful shutdown
- [ ] 20-03-PLAN.md -- Frontend ProcessTable update with pause/resume controls + verification

### Phase 24: Global Seeding & Globe Layers
**Goal**: The globe choropleth renders meaningful risk data for all ~195 countries (not just those with active forecasts), and the three currently-empty globe layers (heatmap, arcs, scenarios) display real data from the event store and knowledge graph.
**Depends on**: Phase 21 (UCDP data feeds baseline risk computation; can start with GDELT+ACLED only but UCDP improves signal)
**Requirements**: SEED-01, SEED-02, SEED-03, GLYR-01, GLYR-02, GLYR-03
**Success Criteria** (what must be TRUE):
  1. The `baseline_country_risk` table contains composite risk scores for all ~195 countries, computed from GDELT event density + ACLED conflict intensity + UCDP fatality signal + government travel advisory levels with configurable weights and exponential time decay -- updated every 6 hours via APScheduler job
  2. `GET /api/v1/countries` returns merged risk scores: active forecast risk overrides baseline when available (`COALESCE(forecast_risk, baseline_risk)`), so countries with forecasts show prediction-derived risk while countries without forecasts still show meaningful baseline risk
  3. The globe choropleth colors all ~195 countries with intensity proportional to their merged risk scores -- no more empty/neutral countries with zero data; high-risk conflict zones visually stand out
  4. The heatmap layer displays real GDELT event locations on the globe -- events include `lat`/`lon` coordinates (added to SQLite schema), served via `/api/v1/events/geo` with server-side 0.5-degree grid aggregation for performance
  5. The arcs layer renders bilateral country relationships from knowledge graph edges via `/api/v1/countries/relations` -- showing top-N country pairs by edge weight as great-circle arcs on the globe
**Plans**: 3 plans

Plans:
- [ ] 20-01-PLAN.md -- Scheduler package: APScheduler core, dependency container, 9 job wrappers, failure tracking
- [ ] 20-02-PLAN.md -- FastAPI integration: lifespan mount, admin API rewire, graceful shutdown
- [ ] 20-03-PLAN.md -- Frontend ProcessTable update with pause/resume controls + verification

### Phase 25: Frontend Finalization
**Goal**: Every screen and panel handles loading, error, and empty states gracefully. Heavy components lazy-load. Interactive elements are keyboard-accessible. The frontend is ready for external users who encounter edge cases, slow connections, and assistive technology.
**Depends on**: Phase 23 (backtesting), Phase 24 (global seeding + globe layers) -- all features exist to polish
**Requirements**: POLISH-01, POLISH-02, POLISH-03, POLISH-04, POLISH-05
**Success Criteria** (what must be TRUE):
  1. Every panel displays a skeleton placeholder during API fetches -- no blank space, no frozen UI, no layout shift when data arrives
  2. A failed API call in one panel shows an inline error message with a retry button in that panel only -- other panels on the same screen continue operating normally
  3. Screens with no data (no forecasts, no events, no comparisons) display contextual empty-state messages explaining what the panel will show once data exists -- not blank white space
  4. ScenarioExplorer and CalibrationPanel are lazy-loaded (not in the initial bundle); search and filter inputs are debounced; refresh cycles use diff-based DOM updates -- confirmed by Lighthouse performance score above 80 on the dashboard route
  5. All interactive elements (map controls, layer toggles, modal open/close, forecast card expand/collapse) are reachable via keyboard navigation; map controls have ARIA labels; color contrast ratios meet WCAG AA on the dark theme
**Plans**: 3 plans

Plans:
- [ ] 20-01-PLAN.md -- Scheduler package: APScheduler core, dependency container, 9 job wrappers, failure tracking
- [ ] 20-02-PLAN.md -- FastAPI integration: lifespan mount, admin API rewire, graceful shutdown
- [ ] 20-03-PLAN.md -- Frontend ProcessTable update with pause/resume controls + verification

## Progress

**Execution Order:**
Phase 19 -> Phase 20. Then parallel: Phase 21 + Phase 22. Then Phase 23 (after 22), Phase 24 (after 21). Finally Phase 25 (after 23 and 24).

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
| 10. Ingest & Pipeline | v2.0 | 4/4 | Complete | 2026-03-01 |
| 11. TKG Replacement | v2.0 | 3/3 | Complete | 2026-03-01 |
| 12. WM-Derived Frontend | v2.0 | 7/7 | Complete | 2026-03-02 |
| 13. Calibration & Monitoring | v2.0 | 7/7 | Complete | 2026-03-02 |
| 14. Backend API Hardening | v2.1 | 4/4 | Complete | 2026-03-03 |
| 15. URL Routing & Dashboard | v2.1 | 3/3 | Complete | 2026-03-03 |
| 16. Globe & Forecasts Screens | v2.1 | 3/3 | Complete | 2026-03-03 |
| 17. Live Data Feeds & Country Depth | v2.1 | 3/3 | Complete | 2026-03-04 |
| 18. Polymarket-Driven Forecasting | v2.1 | 3/3 | Complete | 2026-03-04 |
| 19. Admin Dashboard Foundation | v3.0 | 3/3 | Complete | 2026-03-05 |
| 20. Daemon Consolidation | v3.0 | 3/3 | Complete | 2026-03-05 |
| 21. Source Expansion & Feed Mgmt | v3.0 | 0/TBD | Not started | - |
| 22. Polymarket Hardening | v3.0 | 0/TBD | Not started | - |
| 23. Historical Backtesting | v3.0 | 0/TBD | Not started | - |
| 24. Global Seeding & Globe Layers | v3.0 | 0/TBD | Not started | - |
| 25. Frontend Finalization | v3.0 | 0/TBD | Not started | - |

**Total:** 20 phases complete (v1.0 + v1.1 + v2.0 + v2.1 + v3.0 partial), 71 plans delivered. v3.0: 2/7 phases complete.
