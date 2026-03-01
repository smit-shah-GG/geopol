# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-27)

**Core value:** Explainability -- every forecast must provide clear, traceable reasoning paths
**Current focus:** Phase 12 in progress. Frontend scaffold delivered (plan 01/07). 6 panel plans remain.

## Current Position

Milestone: v2.0 Operationalization & Forecast Quality
Phase: 12 of 13 (WM-Derived Frontend) -- IN PROGRESS
Plan: 01 of 7 (in phase 12) -- COMPLETE
Status: In progress
Last activity: 2026-03-02 -- Completed 12-01-PLAN.md (frontend scaffold + build system)

Progress: [##############################..........] 75% (12/16 phases lifetime)
v2.0:    [######....] 60% (3/5 phases complete, 12 in progress)

## Performance Metrics

**Velocity:**
- Total plans completed: 36
- Average duration: 14 minutes
- Total execution time: 8.58 hours

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
| 09-api-foundation | 6 | 33min | 6min |
| 10-ingest-forecast-pipeline | 4 | 27min | 7min |
| 11-tkg-predictor-replacement | 3 | 21min | 7min |
| 12-wm-derived-frontend | 1 | 7min | 7min |

**Recent Trend:**
- Last 4 plans: 11-01 (7min), 11-02 (6min), 11-03 (8min), 12-01 (7min)
- Trend: Consistent (scaffold + integration, reuse-heavy from WM)

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
- Auth as per-route Depends, not global ASGI middleware -- health endpoint public (2026-03-01, 09-05)
- Health status derivation: unhealthy only if database down, degraded for other failures (2026-03-01, 09-05)
- ForecastService takes session, does NOT call predict() -- callers invoke predict() then pass results (2026-03-01, 09-06)
- URL-dedup fast path: events_fetched=0 is a valid successful run when lastupdate.txt URL unchanged (2026-03-01, 10-01)
- asyncio.to_thread() for all synchronous v1.0 code in async daemons (2026-03-01, 10-01)
- IngestRun recording tolerates PostgreSQL downtime -- daemon stays up even if metrics DB is down (2026-03-01, 10-01)
- Cache key generators centralize all prefixing; get/set use keys as-is -- prevents double-prefix bugs (2026-03-01, 10-02)
- NullRedis stub enables graceful degradation without Optional branching in hot path (2026-03-01, 10-02)
- Rate limiter fail-open: Redis failure allows request through (rate limiter must never kill the API) (2026-03-01, 10-02)
- pytest asyncio_mode=auto configured globally in pyproject.toml (2026-03-01, 10-02)
- 101 geopolitical feeds from WM's 298-domain list: 31 tier-1 (15-min), 70 tier-2 (60-min) (2026-03-01, 10-03)
- Separate ChromaDB collection 'rss_articles' with same embedding model as graph_patterns (2026-03-01, 10-03)
- Paragraph-boundary chunking with sentence fallback; propaganda risk metadata for downstream weighting (2026-03-01, 10-03)
- LLM-based outcome resolution primary, heuristic (actor+event_code matching) as fallback (2026-03-01, 10-04)
- Carryover questions get priority=1, processed before fresh generation (2026-03-01, 10-04)
- Consecutive failure alerting at >= 2 failures emits CRITICAL log (2026-03-01, 10-04)
- POST /forecasts creates EnsemblePredictor per-request (stateless) (2026-03-01, 10-04)
- Redis lifecycle in app.py lifespan: init on startup, close on shutdown (2026-03-01, 10-04)
- Sparse history vocab (dict[(s,r)] -> set[o]) instead of dense (E*R, E) matrix -- 28GB saved at GDELT scale (2026-03-01, 11-01)
- history_rate is fixed hyperparameter (default 0.3), not learned -- per CONTEXT.md (2026-03-01, 11-01)
- History vocab passed via kwargs to compute_loss, not stored in model (2026-03-01, 11-01)
- Relation GRU uses projection + existing GRUCell, not modified GRUCell (2026-03-01, 11-01)
- TiRGN uses NLL loss; neg_triples/margin accepted but ignored for protocol compat (2026-03-01, 11-01)
- Modern Flax NNX param[...] access in new code, deprecated .value in existing code untouched (2026-03-01, 11-01)
- tensorboardX over torch.utils.tensorboard -- no PyTorch dependency, pure-Python TensorBoard writer (2026-03-01, 11-02)
- wandb as optional[observability] extra, not core dependency -- never crashes training (2026-03-01, 11-02)
- TiRGN checkpoint JSON includes model_type: "tirgn" discriminator for downstream model loading (2026-03-01, 11-02)
- compare_models loads GDELT data ONCE, both models evaluated on same val_triples -- no re-splitting (2026-03-01, 11-02)
- ComparisonResult pass_threshold defaults to -5.0% -- TiRGN ships if within 5% of RE-GCN MRR (2026-03-01, 11-02)
- _evaluate_tirgn uses _compute_fused_distribution directly, evolves embeddings once for all triples (2026-03-01, 11-02)
- Early stopping increments by eval_interval per non-improving evaluation, not by 1 (2026-03-01, 11-02)
- TKG_BACKEND envvar read once at TKGPredictor.__init__; process restart to switch (2026-03-01, 11-03)
- TiRGN mode: self.model = None (no REGCNWrapper), saves memory (2026-03-01, 11-03)
- TiRGN checkpoint restore: nnx.split -> flatten -> map npz keys -> unflatten -> nnx.merge (2026-03-01, 11-03)
- TiRGN predict_object uses raw_decoder directly (not fused distribution) for inference speed (2026-03-01, 11-03)
- Scheduler model_tirgn config section alongside existing model section (2026-03-01, 11-03)
- retrain_tkg.py --backend override resets Settings singleton before scheduler init (2026-03-01, 11-03)
- localStorage key 'geopol-theme' for theme persistence, 'geopol-panel-spans' for resize state (2026-03-02, 12-01)
- No i18n, no Tauri, no analytics in Panel -- web-only, English-only (2026-03-02, 12-01)
- GeoPolAppContext ~8 fields (focused) vs WM's 100+ (2026-03-02, 12-01)
- Monospace analyst aesthetic: SF Mono, #0a0e14 dark bg, terminal-inspired (2026-03-02, 12-01)
- TypeScript DTO fields use snake_case matching JSON wire format (2026-03-02, 12-01)

### Deferred Issues

- Docker daemon requires sudo to start -- user must `sudo systemctl start docker` before running containers or Alembic migrations
- PostgreSQL tests (8 tests in test_forecast_persistence.py and test_concurrent_db.py) skip until Docker is running

### Quick Tasks Completed

| # | Description | Date | Commit | Directory |
|---|-------------|------|--------|-----------|
| 001 | Fix README + add forecast CLI + preflight check | 2026-02-12 | 03dfe49 | [001-fix-readme-add-forecast-cli-preflight](./quick/001-fix-readme-add-forecast-cli-preflight/) |

### Blockers/Concerns

- TiRGN JAX port has no published reference implementation (RESOLVED: model architecture ported successfully in 11-01, research doc sufficient)
- Gemini API cost exposure under public traffic (RESOLVED: rate limiter + Gemini budget tracking in 10-02, enforcement wired to endpoints in 10-04, budget exhaustion returns 429)
- jraph archived by Google DeepMind (RESOLVED: eliminated in 09-03)
- Polyglot tax: Python + TypeScript -- two ecosystems for single developer (Rust/Tauri dropped, web-only)
- Docker daemon not auto-started -- verification of PostgreSQL/Redis containers and Alembic migration deferred

## Session Continuity

Last session: 2026-03-02
Stopped at: Completed 12-01-PLAN.md (frontend scaffold + build system)
Resume file: None
Next: 12-02-PLAN.md through 12-07-PLAN.md (6 remaining panel plans in Phase 12)
