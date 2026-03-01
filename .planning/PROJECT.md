# Explainable Geopolitical Forecasting Engine

## What This Is

An AI-powered geopolitical forecasting engine that combines multiple models to predict political events with clear reasoning paths. The system ingests event data from public APIs and custom sources, processes it through hybrid intelligence algorithms, and produces calibrated probability estimates with explainable reasoning chains. Inspired by systems like IARPA SAGE but optimized for transparency over raw performance.

## Core Value

Explainability — every forecast must provide clear, traceable reasoning paths showing why specific predictions were made.

## Requirements

### Validated

- Event data ingestion from GDELT API with custom enrichment pipeline — v1.0
- Temporal knowledge graph construction from event streams — v1.0
- Hybrid model ensemble combining TKG algorithms (RE-GCN/TiRGN) with LLM reasoning — v1.0
- Explainable reasoning chain generation for each prediction — v1.0
- Probability calibration system with Brier score optimization — v1.0
- Evaluation framework against historical events — v1.0
- Fix NetworkX shortest_path API to use single_source_shortest_path — v1.1
- Production bootstrap script connecting data ingestion → graph build → RAG indexing — v1.1
- Graph partitioning for scalability beyond 1M events — v1.1

### Active

- WM-derived TypeScript dashboard as public-facing frontend with deck.gl globe, forecast panels, scenario explorer, country briefs, and calibration visualizations
- Headless FastAPI backend serving forecast data via versioned REST API with Pydantic DTOs
- Scheduled daily forecast automation with cron/systemd orchestration
- System health monitoring: data freshness, calibration drift, pipeline status
- TKG predictor replacement: research and implement best-accuracy JAX-compatible algorithm (candidates: HisMatch, TiRGN, others) optimized for weekly retraining on large datasets
- Micro-batch GDELT ingest on 15-minute cycle to keep knowledge graph current between daily prediction runs
- Dynamic per-CAMEO-category ensemble calibration: outcome feedback loop replacing fixed 60/40 alpha with learned weights per CAMEO root category

## Current Milestone: v2.0 Operationalization & Forecast Quality

**Goal:** Transform the research prototype into a publicly demonstrable system with a WM-derived TypeScript dashboard, headless FastAPI backend, automated operations, while upgrading the TKG predictor and closing the calibration feedback loop.

**Architecture:** Geopol is a headless Python forecast engine. Its frontend is a TypeScript dashboard architecturally derived from World Monitor (vanilla TS + deck.gl + Panel system) but purpose-built for geopolitical forecasting. WM is used as a reference architecture and code quarry — not as a runtime dependency. See `WORLDMONITOR_INTEGRATION.md` for DTO contract spec and `.planning/research/WM_AS_REPOSITORY.md` for salvageability analysis.

**Execution model:** Parallel after Phase 9. The API DTOs and mock fixtures established in Phase 9 serve as the contract that backend (Phase 10) and frontend (Phase 12) develop against independently.

**Target features:**
- WM-derived TypeScript frontend: deck.gl globe with forecast risk choropleth, forecast panels, interactive scenario explorer, country brief pages with forecast/evidence/calibration tabs
- Headless FastAPI backend: versioned REST API, Pydantic DTOs, Redis caching, API key auth
- Scheduled daily forecast automation
- System health monitoring: data freshness, calibration drift, pipeline status
- TKG predictor replacement: best-accuracy JAX-compatible algorithm, weekly-retrainable on large GDELT data
- Micro-batch GDELT ingest: 15-minute update cycle keeping graph current, predictions remain daily
- Dynamic calibration: per-CAMEO-category ensemble weights learned from outcome feedback loop, replacing static alpha=0.6

### Out of Scope

- Human crowdsourcing — pure AI system without human forecaster integration; adding human forecasters changes the product category from "AI forecasting engine" to "prediction market platform," which is a fundamentally different system with different trust models, incentive structures, and UX requirements
- Multi-language support — English sources only; GDELT's non-English coverage uses machine translation with variable quality, and entity resolution across languages introduces ambiguity that degrades graph quality more than it improves coverage for the event types we care about (conflicts, diplomacy)
- Financial market modeling — no cross-sector impact propagation; financial contagion modeling requires a completely different data pipeline (market feeds, SEC filings, supply chain graphs) and the causal models are domain-specific — bolting this onto a geopolitical engine would dilute both
- Multi-source data expansion (ACLED, ICEWS) — deferred from v2.0; adding sources before the micro-batch architecture is proven would compound integration complexity; revisit for v3.0 once the streaming ingest pipeline is stable and the TKG predictor replacement is validated
- Real-time prediction — v2.0 implements micro-batch ingest (15-min GDELT updates) but predictions remain daily; real-time prediction would require incremental TKG inference (the encoder would need to update embeddings without full recomputation), which is architecturally different from the batch predict model and not justified until the micro-batch ingest proves its value for graph freshness
- World Monitor live integration — WM is used as a reference architecture and code quarry, not as a runtime dependency; no service-to-service HTTP calls between WM and Geopol in v2.0

## Context

Drawing from the comprehensive technical reference in geopol.md, this project implements a geopolitical forecasting engine combining temporal knowledge graphs with LLM reasoning. v1.0-v1.1 established the core pipeline as a research prototype; v2.0 transitions to a publicly demonstrable system.

Key technical inspirations:
- IARPA SAGE's hybrid architecture (10% improvement over human baselines)
- TKG algorithms: RE-GCN (40.4% MRR), TiRGN (44.0% MRR), HisMatch (46.4% MRR)
- Explainable approaches like xERTE with reasoning paths
- Calibration methods from superforecaster research
- GDELT 15-minute update feed for near-real-time event coverage
- World Monitor's vanilla TypeScript architecture (Panel system, deck.gl globe, AppContext pattern, CSS variable theming) as frontend reference

**v2.0 context shift:** The system moves from research prototype (CLI-only, manual runs, fixed parameters) to public demo (web dashboard, automated daily runs, learned calibration). The audience expands from "just me" to "external visitors who can see live forecasts and explore scenarios." This raises the bar for reliability, error handling, and presentation quality across every component.

**Frontend context (2026-02-27):** Streamlit was originally chosen for development speed but replaced by a WM-derived TypeScript dashboard. World Monitor's zero-framework vanilla TypeScript architecture (no React/Vue/Svelte) makes it uniquely suitable as a code quarry — the patterns carry without framework baggage. ~35-40% of WM's code is salvageable by volume (~70% by architectural value): build system, Panel base class, deck.gl + maplibre-gl setup, AppContext singleton, CSS variable theming, Tauri desktop scaffold. The 45 WM-specific panels and 50+ API integrations are replaced entirely with forecast-specific implementations. This introduces a polyglot tax (Python + TypeScript + Rust/Tauri) mitigated by WM's architectural simplicity.

**TKG predictor context:** The current RE-GCN implementation (JAX/jraph, 200-dim embeddings, 40.4% MRR) works but is resource-intensive to train and may not be the best accuracy/efficiency tradeoff for weekly retraining on large GDELT datasets. The cancelled v2.0 plan included TKG-03 (RE-GCN → HisMatch, ~46.4% MRR); this upgrade is being revisited independently of the cancelled Llama integration. Research should evaluate candidates on accuracy, JAX compatibility, training time on large data, and suitability for weekly retraining cycles.

**Calibration context:** The current ensemble uses fixed alpha=0.6 (60% Gemini LLM, 40% RE-GCN TKG) set at design time and never updated from production outcomes. The calibration infrastructure (isotonic + temperature scaling) exists but lacks a feedback loop. v2.0 closes this gap: past predictions are compared against GDELT ground truth, and per-CAMEO-category alpha weights are optimized automatically. This makes the system self-improving as prediction history accumulates.

## Constraints

- **Compute**: Limited GPU resources (RTX 3060 12GB) — JAX TKG training constrained, no local LLM feasible; TKG predictor replacement must fit this envelope for weekly retraining
- **Data volume**: GDELT produces 500K-1M articles/day; micro-batch ingest processes the 15-minute update feed but still requires selective sampling to avoid overwhelming graph construction; the full firehose is not feasible
- **LLM**: Gemini API (frontier-class reasoning, negligible cost at current query volume); interactive queries add API cost pressure — rate limiting is mandatory to prevent abuse
- **Frontend**: WM-derived TypeScript dashboard (vanilla TS + Vite + deck.gl). Accepts the polyglot tax (Python backend + TypeScript frontend + Rust/Tauri desktop) in exchange for a dramatically better product than Streamlit. Mitigated by WM's zero-framework architecture — no React/Vue dependency churn.
- **Public exposure**: Web frontend is externally accessible — API endpoints must handle malformed input, rate limiting must be enforced, and no internal system details (API keys, file paths, model internals) should leak through error messages or reasoning chains

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Hybrid intelligence over pure ML | Combines strengths of multiple approaches for robustness — LLM brings world knowledge and reasoning, TKG brings structural pattern detection. Neither alone matches their combination. | Good |
| Mixed data approach (APIs + custom) | Balances development speed with unique value creation — GDELT API provides breadth, custom enrichment pipeline provides depth and domain-specific filtering | Good |
| Explainability as core value | Trust and interpretability matter more than raw accuracy — a forecast nobody understands is useless for decision-making; explainability is the product differentiator vs black-box competitors | Good |
| Batch processing over real-time (v1.0-v1.1) | Reduced complexity and compute requirements significantly for the research prototype phase; daily batch was sufficient when the only consumer was the developer | Good — revisited in v2.0: micro-batch ingest added, predictions remain daily |
| Python for backend | Scientific computing ecosystem — JAX, PyTorch, NetworkX, FastAPI all Python-native; backend remains pure Python | Good |
| RE-GCN over TiRGN (v1.0) | More mature implementation available at time of v1.0; TiRGN's local+global recurrent encoding was more complex to implement in JAX/jraph without clear accuracy justification over RE-GCN for the MVP | Good — under review for v2.0 TKG replacement |
| Fixed 60/40 LLM/TKG ensemble weighting (v1.0) | Simple, interpretable starting point; balanced reasoning (LLM) with pattern matching (TKG) without overcomplicating the MVP | Good — validated in v1.1.1 but being replaced in v2.0 with per-CAMEO dynamic weights |
| Retain Gemini over local Llama | Frontier-class Gemini reasoning dramatically outperforms 4-bit Llama2-7B constrained to RTX 3060 12GB VRAM; API costs negligible at current query volume (~$0.01/forecast); engineering effort for local LLM (6 phases) targeted uncertain gains on an inferior reasoning engine; Gemini improves for free with each model generation | Good — v2.0 Llama plan cancelled 2026-02-14 |
| JAX/jraph for TKG training | Memory efficiency on CPU — JAX's XLA compilation and jraph's sparse graph representation keep training feasible on RTX 3060; v2.0 TKG replacement must maintain JAX compatibility to avoid framework migration overhead | Good |
| Weekly retraining schedule | Captures evolving geopolitical patterns — geopolitical dynamics shift on weekly-to-monthly timescales, so weekly retraining balances freshness vs compute cost; v2.0 TKG replacement must support this cadence on large datasets | Good |
| Micro-batch ingest, daily predict (v2.0) | GDELT updates every 15 minutes; ingesting continuously keeps the knowledge graph current for when daily predictions run, capturing fast-moving conflicts that a daily dump would miss by up to 24 hours. Predictions remain daily because the Gemini+TKG pipeline is too expensive to run every 15 minutes and most geopolitical forecasts don't need sub-daily resolution. | — Pending |
| Replace fixed alpha with per-CAMEO dynamic calibration (v2.0) | The current fixed alpha=0.6 treats all event types identically, but the LLM and TKG have different strengths: TKG likely excels at repetitive conflict patterns (CAMEO 18-20) where historical graphs are dense, while LLM likely excels at novel diplomatic events (CAMEO 03-06) where reasoning from context matters more than pattern matching. Per-CAMEO weights (20 root categories) are granular enough to capture this without overfitting, and the feedback loop comparing predictions to GDELT ground truth makes the system self-improving. | — Pending |
| TKG predictor replacement via research (v2.0) | RE-GCN at 40.4% MRR is functional but not best-in-class; HisMatch (46.4% MRR) and TiRGN (44.0% MRR) offer meaningful accuracy gains. The choice must balance accuracy (king), JAX compatibility (non-negotiable — avoids framework migration), training time on large GDELT datasets (weekly retraining cadence), and RTX 3060 12GB VRAM fit. Research phase will evaluate candidates before committing. | — Pending |
| WM-derived TypeScript frontend over Streamlit (v2.0) | World Monitor's vanilla TypeScript architecture (Vite + deck.gl + Panel system + CSS variables) provides a dramatically better frontend than Streamlit — 3D globe, interactive panels, Tauri desktop, custom visualizations. WM's zero-framework approach means patterns are salvageable without dependency baggage (~35-40% code reuse, ~70% architectural reuse). Accepts polyglot tax (Python + TypeScript + Rust) mitigated by WM's architectural simplicity. Replaces Streamlit decision from 2026-02-14. | — Pending |
| Headless API-first backend (v2.0) | FastAPI REST server with Pydantic DTOs as the mandatory bridge between Python engine and TypeScript frontend. Contract-first design: DTOs defined in Phase 9 enable parallel development of backend (Phase 10) and frontend (Phase 12) against mock fixtures. More general than Streamlit monolith — any frontend can consume the API (WM integration, mobile app, third-party). | — Pending |
| PostgreSQL for forecast persistence (v2.0) | API server needs a real database for concurrent access from multiple processes (FastAPI + ingest daemon + prediction pipeline). SQLite is viable for single-writer workloads (GDELT events, partition index) but PostgreSQL handles connection pooling, concurrent writes, and deployment to Railway/Fly.io without the WAL mode constraints. SQLite retained for GDELT event store and partition index where single-writer semantics are natural. | — Pending |
| Public-facing demo as target audience (v2.0) | Shifts quality bar from "works for me" to "works for strangers" — requires error handling, input validation, rate limiting, and visual polish that a research prototype doesn't need. This is a deliberate choice to make the system demonstrable, which forces operational maturity that benefits all use cases. | — Pending |

## Current State

**Version:** v1.1 (shipped 2026-01-30) — v2.0 in progress

**Tech Stack:**
- **Backend**: Python 3.11+ with uv package management
- PyTorch (CPU-only) for inference
- JAX for TKG training (jraph being eliminated in Phase 9)
- NetworkX for graph operations
- SQLite for GDELT event store and partition index
- PostgreSQL for forecast persistence, calibration, and outcome tracking (v2.0)
- FastAPI + Pydantic for REST API (v2.0)
- Redis (Upstash) for forecast response caching (v2.0)
- Gemini API via google-genai SDK
- **Frontend** (v2.0): TypeScript 5.7+ with Vite 6
- deck.gl 9 + maplibre-gl for globe rendering
- Vanilla TypeScript (no framework — WM-derived Panel system, AppContext, h() helper)
- CSS variables for dark/light theming
- Tauri for desktop app (future backlog)

**Codebase:**
- ~100 Python source files, 40,257 lines
- 8 phases, 21 plans delivered across 2 milestones (v1.0 + v1.1)
- v2.0: 5 phases planned, 48 requirements, parallel execution after Phase 9

**Known Issues:**
- datetime.utcnow() deprecated in Python 3.12+ (minor, in bootstrap code)

**Reference Projects:**
- World Monitor (`/home/kondraki/personal/worldmonitor/`) — TypeScript OSINT dashboard used as frontend reference architecture. See `.planning/research/WM_AS_REPOSITORY.md`.

---
*Last updated: 2026-02-27 — v2.0 restructured: WM-derived TypeScript frontend replaces Streamlit, headless FastAPI backend, parallel execution model, PostgreSQL for forecast persistence*
