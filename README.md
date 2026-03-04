# Geopol — AI Geopolitical Forecasting Engine

Explainable probabilistic forecasting system combining Temporal Knowledge Graphs, LLM reasoning, and ensemble prediction. Ingests real-time event streams from GDELT, ACLED, RSS feeds, and government advisories. Produces calibrated forecasts with full reasoning chains, scenario trees, and Polymarket comparison tracking.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  Frontend (TypeScript/Vite)                                     │
│  Three-screen SPA: /dashboard  /globe  /forecasts               │
│  deck.gl + maplibre (globe), d3 (visualizations)                │
└────────────────────────┬────────────────────────────────────────┘
                         │ /api/v1
┌────────────────────────▼────────────────────────────────────────┐
│  FastAPI Backend                                                │
│  Forecast submission queue · Search · Country risk · Calibration│
│  Polymarket matching loop · RFC 9457 errors · API key auth      │
├─────────────┬──────────────┬──────────────┬─────────────────────┤
│ PostgreSQL  │    Redis     │   SQLite     │     ChromaDB        │
│ Predictions │  Response    │ GDELT/ACLED  │  Article + pattern  │
│ Comparisons │  cache       │ events       │  embeddings (RAG)   │
│ IngestRuns  │              │ (1.37M rows) │                     │
└─────────────┴──────────────┴──────────────┴─────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────────┐
│  Forecast Engine                                                │
│  EnsemblePredictor: Gemini LLM + TiRGN TKG + calibration       │
│  Per-CAMEO dynamic calibration via L-BFGS-B optimization        │
│  Scenario tree generation · Reasoning chain extraction           │
└─────────────────────────────────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────────┐
│  Data Ingestion Daemons                                         │
│  GDELT poller (15 min) · RSS daemon (100+ feeds, tiered)        │
│  ACLED conflict poller (daily) · Advisory poller (daily)        │
│  Polymarket matching + auto-forecast (hourly, in-process)       │
└─────────────────────────────────────────────────────────────────┘
```

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/smit-shah-GG/geopol.git
cd geopol
uv sync
cd frontend && npm ci && cd ..

# 2. Configure
cp .env.example .env
# Set GEMINI_API_KEY in .env

# 3. Infrastructure
docker compose up -d postgres redis
uv run alembic upgrade head

# 4. Bootstrap data (15-30 min)
uv run python scripts/bootstrap.py

# 5. Start
uv run uvicorn src.api.app:create_app --factory --reload --port 8000  # Terminal 1
cd frontend && npm run dev                                             # Terminal 2
```

Open http://localhost:5173. API docs at http://localhost:8000/docs.

See [RUN_GUIDE.md](RUN_GUIDE.md) for the complete deployment walkthrough including training, pollers, and production setup.

## Key Features

**Forecasting**
- Hybrid ensemble: Gemini LLM reasoning + TiRGN temporal knowledge graph predictions
- Per-CAMEO calibration weights optimized via L-BFGS-B (not fixed alpha)
- Scenario tree generation with branching probability paths
- Full explainability: every forecast traces evidence → reasoning → probability

**Data Pipeline**
- GDELT micro-batch polling with gap recovery and backfill
- 100+ RSS feeds (tiered: wire services at 15 min, think tanks at 60 min)
- ACLED armed conflict events (battles, remote violence, civilian targeting)
- US State Dept + UK FCDO travel advisory aggregation
- ChromaDB RAG store with `all-mpnet-base-v2` embeddings for article retrieval

**Frontend**
- Three bookmarkable screens: Dashboard, Globe, Forecasts
- deck.gl globe with country risk heatmap, forecast scatter, and arc layers
- Expandable forecast cards with scenario trees and reasoning chains
- Question submission with two-phase submit/confirm flow
- Polymarket comparison badges and divergence tracking
- Dark theme, View Transition API crossfades, diff-based DOM updates

**Polymarket Integration**
- Automatic matching of internal forecasts to Polymarket prediction markets
- Auto-forecast generation for high-volume unmatched geopolitical questions
- Daily re-forecasting of active comparisons with Brier score tracking
- Divergence analysis (geopol probability vs market price)

## Tech Stack

| Layer | Technology |
|-------|-----------|
| API | FastAPI, Pydantic, SQLAlchemy (async), asyncpg |
| Frontend | TypeScript, Vite, deck.gl, maplibre-gl, d3 |
| Database | PostgreSQL 16 (forecasts, calibration), SQLite (events), Redis 7 (cache) |
| LLM | Google Gemini (primary: gemini-3-pro-preview, fallback: gemini-2.5-pro) |
| TKG | TiRGN (JAX/Flax) — RE-GCN + copy-generation + Time-ConvTransE |
| RAG | LlamaIndex, ChromaDB, sentence-transformers (all-mpnet-base-v2) |
| Ingest | aiohttp, feedparser, trafilatura, GDELT API, ACLED API |
| Infra | Docker Compose, uv, Alembic |

## Project Structure

```
src/
├── api/                  # FastAPI routes, schemas, middleware, services
│   ├── routes/v1/        # Versioned endpoints (forecasts, calibration, events, etc.)
│   ├── schemas/          # Pydantic DTOs
│   └── services/         # Business logic (forecast service, submission worker)
├── forecasting/          # Hybrid prediction engine
│   ├── ensemble_predictor.py   # LLM + TKG weighted ensemble
│   ├── gemini_client.py        # Gemini API with retry + fallback
│   ├── scenario_generator.py   # Branching scenario trees
│   └── tkg_predictor.py        # TiRGN inference wrapper
├── polymarket/           # Polymarket integration
│   ├── client.py               # Gamma API client with circuit breaker
│   ├── matcher.py              # Semantic matching (Gemini)
│   ├── comparison.py           # Comparison service + snapshots
│   └── auto_forecaster.py      # Auto-forecast pipeline
├── ingest/               # Data ingestion daemons
│   ├── gdelt_poller.py         # GDELT micro-batch polling
│   ├── rss_daemon.py           # Tiered RSS feed aggregation
│   ├── acled_poller.py         # ACLED armed conflict events
│   ├── advisory_poller.py      # Government travel advisories
│   └── article_processor.py    # Trafilatura extraction + ChromaDB indexing
├── knowledge_graph/      # TKG construction and persistence
├── calibration/          # Probability calibration (isotonic, temperature, per-CAMEO)
├── training/             # TKG model training pipeline
├── database/             # SQLite event storage (GDELT/ACLED)
├── db/                   # PostgreSQL models + async session management
└── settings.py           # Centralized env-var config (pydantic-settings)

frontend/src/
├── app/                  # Router, panel layout, screen lifecycle
├── components/           # Panel classes (ForecastPanel, ComparisonPanel, etc.)
├── screens/              # DashboardScreen, GlobeScreen, ForecastsScreen
├── services/             # API client, circuit breaker, dedup cache
├── types/                # TypeScript API type definitions
└── utils/                # DOM helpers, formatting, keyset cursor

scripts/
├── bootstrap.py          # Zero-to-operational (5-stage pipeline)
├── preflight.py          # System readiness validator
├── forecast.py           # CLI single-question forecast
├── daily_forecast.py     # 4-phase daily pipeline (cron/timer target)
├── gdelt_poller.py       # GDELT daemon entry point
├── rss_daemon.py         # RSS daemon entry point
├── acled_poller.py       # ACLED daemon entry point
├── advisory_poller.py    # Advisory daemon entry point
├── train_tirgn.py        # TiRGN model training (JAX)
├── collect_training_data.py  # GDELT historical data collection
└── retrain_tkg.py        # Automated retraining
```

## Development

```bash
# Run tests
uv run pytest tests/ -v

# Coverage
uv run pytest tests/ --cov=src --cov-report=term-missing

# Single forecast (CLI)
uv run python scripts/forecast.py -q "Will NATO expand?" --verbose

# Preflight check
uv run python scripts/preflight.py
```

## Version History

| Version | Date | Milestone |
|---------|------|-----------|
| v1.0 | 2026-01-23 | MVP: GDELT ingest, TKG, hybrid forecasting, calibration |
| v1.1 | 2026-02 | Infrastructure hardening: graph partitioning, checkpoint/resume |
| v2.0 | 2026-03-02 | Headless API + frontend: FastAPI, PostgreSQL, TypeScript SPA |
| v2.1 | 2026-03-04 | Production UX: three-screen routing, live data feeds, Polymarket integration |

## License

Private repository. All rights reserved.
