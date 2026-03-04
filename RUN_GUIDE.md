# Geopol Run Guide

From-clone to fully operational. Every step, every terminal, every command.

---

## Prerequisites

| Tool | Version | Install |
|------|---------|---------|
| Python | 3.10+ | System package manager |
| uv | latest | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| Node.js | 18+ | nvm, fnm, or system package |
| Docker + Compose | v2+ | Docker Desktop or `docker.io` + `docker-compose-v2` |

Verify:

```bash
python --version    # >= 3.10
uv --version
node --version      # >= 18
docker compose version  # v2+
```

---

## 1. Clone & Install

```bash
git clone https://github.com/smit-shah-GG/geopol.git
cd geopol
```

### Python dependencies

```bash
# Full install (training + LLM + ingest + dev tools):
uv sync

# Production API only (no torch/jax/trafilatura):
uv sync --no-dev --no-group training

# API + forecasting (no training, no ingest daemons):
uv sync --extra llm
```

**Dependency groups** (defined in `pyproject.toml`):

| Group | What it pulls | Size |
|-------|---------------|------|
| Core (always) | FastAPI, SQLAlchemy, asyncpg, Redis, Pydantic, aiohttp | ~200 MB |
| `training` | JAX, Flax, Torch, Optax, scikit-learn, tensorboardX | ~3.5 GB |
| `llm` | google-genai, LlamaIndex, ChromaDB, sentence-transformers | ~2 GB |
| `ingest` | gdeltdoc, pandas, Dask, feedparser, trafilatura | ~1.5 GB |
| `all` | All of the above | ~7 GB |
| `dev` | pytest, pytest-cov, ruff, mypy | ~100 MB |

### Frontend dependencies

```bash
cd frontend
npm ci
cd ..
```

---

## 2. Configure

```bash
cp .env.example .env
```

Edit `.env`:

```bash
# === REQUIRED (no default) ===
GEMINI_API_KEY=your_key_here          # https://aistudio.google.com/apikey

# === Infrastructure (defaults work for docker-compose) ===
DATABASE_URL=postgresql+asyncpg://geopol:geopol_dev@localhost:5432/geopol
POSTGRES_PASSWORD=geopol_dev
REDIS_URL=redis://localhost:6379/0
GDELT_DB_PATH=data/events.db

# === Runtime ===
ENVIRONMENT=development
LOG_LEVEL=INFO

# === Polymarket (disable if Gemini budget is tight) ===
POLYMARKET_ENABLED=false              # Set true when ready
```

Frontend env (already populated if cloned):

```bash
# frontend/.env
VITE_API_KEY=dev-api-key-geopol-2026
VITE_API_BASE=/api/v1
```

### Complete environment variable reference

<details>
<summary>All settings (from <code>src/settings.py</code>)</summary>

**Database & Services**

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | `postgresql+asyncpg://geopol:geopol_dev@localhost:5432/geopol` | PostgreSQL connection |
| `GDELT_DB_PATH` | `data/events.db` | SQLite event store path |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis connection |

**Runtime**

| Variable | Default | Description |
|----------|---------|-------------|
| `ENVIRONMENT` | `development` | `development` \| `production` \| `testing` |
| `USE_FIXTURES` | `false` | Mock data fallback (dev only) |
| `LOG_LEVEL` | `INFO` | `DEBUG` \| `INFO` \| `WARNING` \| `ERROR` |
| `LOG_JSON` | `false` | Structured JSON logs (production) |

**API**

| Variable | Default | Description |
|----------|---------|-------------|
| `API_KEY_HEADER` | `X-API-Key` | Auth header name |
| `CORS_ORIGINS` | `["http://localhost:5173","http://localhost:3000"]` | Allowed origins |

**Gemini LLM**

| Variable | Default | Description |
|----------|---------|-------------|
| `GEMINI_API_KEY` | `""` | **Required.** Google Gemini API key |
| `GEMINI_MODEL` | `models/gemini-3-pro-preview` | Primary model |
| `GEMINI_FALLBACK_MODEL` | `models/gemini-2.5-pro` | Fallback on 503 |
| `GEMINI_MAX_RPM` | `25` | Requests per minute cap |
| `GEMINI_DAILY_BUDGET` | `25` | Max questions per UTC day |

**GDELT Ingest**

| Variable | Default | Description |
|----------|---------|-------------|
| `GDELT_POLL_INTERVAL` | `900` | Poll interval (seconds, 15 min) |
| `GDELT_BACKFILL_ON_START` | `true` | Recover missed events on boot |

**RSS Ingest**

| Variable | Default | Description |
|----------|---------|-------------|
| `RSS_POLL_INTERVAL_TIER1` | `900` | Tier-1 feeds (15 min) |
| `RSS_POLL_INTERVAL_TIER2` | `3600` | Tier-2 feeds (60 min) |
| `RSS_ARTICLE_RETENTION_DAYS` | `90` | Article retention |

**ACLED**

| Variable | Default | Description |
|----------|---------|-------------|
| `ACLED_EMAIL` | `""` | ACLED account email |
| `ACLED_PASSWORD` | `""` | ACLED API key |
| `ACLED_POLL_INTERVAL` | `86400` | Poll interval (24h) |

**Advisories**

| Variable | Default | Description |
|----------|---------|-------------|
| `ADVISORY_POLL_INTERVAL` | `86400` | Poll interval (24h) |

**TKG Model**

| Variable | Default | Description |
|----------|---------|-------------|
| `TKG_BACKEND` | `tirgn` | `tirgn` \| `regcn` |

**Calibration**

| Variable | Default | Description |
|----------|---------|-------------|
| `CALIBRATION_MIN_SAMPLES` | `10` | Min predictions for calibration |
| `CALIBRATION_MAX_DEVIATION` | `0.20` | Max drift tolerance |
| `CALIBRATION_RECOMPUTE_DAY` | `0` | 0=Monday |

**Polymarket**

| Variable | Default | Description |
|----------|---------|-------------|
| `POLYMARKET_ENABLED` | `true` | Enable background matching loop |
| `POLYMARKET_POLL_INTERVAL` | `3600` | Matching cycle interval (1h) |
| `POLYMARKET_MATCH_THRESHOLD` | `0.6` | Semantic match confidence |
| `POLYMARKET_VOLUME_THRESHOLD` | `100000` | Min USD volume for auto-forecast |
| `POLYMARKET_DAILY_NEW_FORECAST_CAP` | `3` | New PM-driven forecasts/day |
| `POLYMARKET_DAILY_REFORECAST_CAP` | `5` | Re-forecasts of active matches/day |

**Monitoring/Alerting**

| Variable | Default | Description |
|----------|---------|-------------|
| `SMTP_HOST` | `""` | Mail server |
| `SMTP_PORT` | `587` | Mail port |
| `ALERT_RECIPIENT` | `""` | Alert email destination |
| `FEED_STALENESS_HOURS` | `1.0` | Alert if feed older than N hours |
| `DRIFT_THRESHOLD_PCT` | `10.0` | Alert if calibration drift exceeds % |
| `DISK_WARNING_PCT` | `80.0` | Disk usage warning threshold |
| `DISK_CRITICAL_PCT` | `90.0` | Disk usage critical threshold |

</details>

---

## 3. Infrastructure

### Start PostgreSQL + Redis

```bash
docker compose up -d postgres redis
```

Wait for healthy:

```bash
docker compose ps
# Both should show "healthy" within 20 seconds
```

### Run database migrations

```bash
uv run alembic upgrade head
```

This applies 5 migrations:

| Migration | What it creates |
|-----------|----------------|
| `001_initial_schema` | `predictions`, `outcome_records`, `api_keys`, `ingest_runs` |
| `002_pending_questions` | `question_queue`, `daemon_type` tracking |
| `phase13_schema` | Forecast request tracking, async response queue |
| `004_forecast_requests_tsvector` | Full-text search (`tsvector`) on questions |
| `005_polymarket_provenance` | `polymarket_comparisons`, `polymarket_snapshots` |

---

## 4. Bootstrap (Data Foundation)

This populates the event database, knowledge graph, and RAG vector store. Required for forecasting to work.

```bash
uv run python scripts/bootstrap.py
```

**Duration:** 15-30 minutes (GDELT API speed dependent).

**5 stages, executed sequentially:**

```
1. collect   → Fetch 30 days of GDELT events from API
2. process   → Transform to TKG format, deduplicate, load into SQLite
3. graph     → Build temporal knowledge graph (entities + relations)
4. persist   → Save graph to data/graphs/*.graphml
5. index     → Embed patterns into ChromaDB vector store (chroma_db/)
```

**Outputs:**

| Path | Contents |
|------|----------|
| `data/events.db` | SQLite with ~30K-100K GDELT events |
| `data/graphs/*.graphml` | Temporal knowledge graph partitions |
| `chroma_db/` | ChromaDB vector embeddings (graph patterns) |
| `data/bootstrap_state.json` | Checkpoint state (for resume) |

**Options:**

```bash
# Preview what would run:
uv run python scripts/bootstrap.py --dry-run

# Re-run a specific stage:
uv run python scripts/bootstrap.py --force-stage collect

# Resume after interrupt (automatic — completed stages are skipped):
uv run python scripts/bootstrap.py
```

---

## 5. Training (TKG Model)

Optional but improves forecast quality. The system works in LLM-only mode without a trained model.

### 5a. Collect training data

```bash
uv run python scripts/collect_training_data.py
```

Fetches 30 days of GDELT events (all QuadClasses 1-4) and writes Parquet to `data/gdelt/processed/events.parquet`.

### 5b. Train TiRGN

```bash
uv run python scripts/train_tirgn.py
```

**Default hyperparameters:**

| Flag | Default | Description |
|------|---------|-------------|
| `--epochs` | 100 | Training iterations |
| `--lr` | 0.001 | Learning rate |
| `--batch-size` | 1024 | Samples per batch |
| `--embedding-dim` | 200 | Entity/relation embedding dimension |
| `--num-layers` | 2 | R-GCN layers |
| `--history-rate` | 0.3 | Copy-generation fusion (0=raw, 1=history only) |
| `--history-window` | 50 | Past snapshots for history vocabulary |
| `--patience` | 15 | Early stopping epochs without MRR improvement |
| `--logdir` | `runs/tirgn` | TensorBoard log directory |
| `--num-days` | 30 | Days of data to include |
| `--data-path` | `data/gdelt/processed/events.parquet` | Input data |
| `--model-dir` | `models/tkg` | Checkpoint output directory |

**Full example with tuning:**

```bash
uv run python scripts/train_tirgn.py \
  --epochs 50 \
  --patience 10 \
  --batch-size 512 \
  --history-rate 0.3 \
  --history-window 50
```

**Duration:** Estimated 25-50 minutes on CPU. Faster with JAX+CUDA.

**Outputs:**

| Path | Contents |
|------|----------|
| `models/tkg/tirgn_best.json` | Best checkpoint metadata (epoch, MRR, config) |
| `models/tkg/tirgn_final.jax` | Final JAX model parameters |
| `runs/tirgn/` | TensorBoard event files |

**Monitor training:**

```bash
uv run tensorboard --logdir runs/tirgn
# Open http://localhost:6006
```

### 5c. Scheduled retraining (optional)

```bash
uv run python scripts/retrain_tkg.py
```

Automates collection + retraining. Designed for cron/systemd timer (e.g., weekly at 2 AM UTC).

---

## 6. Preflight Check

Before starting the system, validate everything:

```bash
uv run python scripts/preflight.py
```

Checks: Python imports, `.env` file, `GEMINI_API_KEY`, event database, knowledge graphs, RAG vector store, TKG model.

Example output:

```
[PASS] Import google.generativeai
[PASS] Import llama_index
[PASS] Environment file (.env)
[PASS] GEMINI_API_KEY
[PASS] Event database (data/events.db)
[PASS] Knowledge graphs (3 .graphml files)
[PASS] RAG vector store (1245 entries)
[SKIP] TKG model not found — forecasting works without it (LLM-only mode)
```

---

## 7. Start the System

### Terminal 1: API server

```bash
uv run uvicorn src.api.app:create_app --factory --reload --host 0.0.0.0 --port 8000
```

The `--factory` flag tells uvicorn that `create_app` is a callable returning the app, not the app itself. This enables the lifespan hooks (DB init, Redis init, dev API key seeding, Polymarket loop).

**Startup sequence inside `create_app`:**
1. Structured logging
2. PostgreSQL engine + connection pool (5 base, 10 overflow)
3. Redis client (degrades to NullRedis if unavailable)
4. Dev API key seeded (`dev-api-key-geopol-2026`) in development mode
5. Polymarket background loop (if `POLYMARKET_ENABLED=true`)

**Verify:** http://localhost:8000/docs (OpenAPI/Swagger UI)

### Terminal 2: Frontend dev server

```bash
cd frontend
npm run dev
```

Runs Vite at http://localhost:5173. Proxies `/api/*` to `http://localhost:8000` (configured in `vite.config.ts`).

### Terminal 3+: Poller daemons

Each poller runs as a long-lived process. Start whichever you need:

```bash
# GDELT event polling (every 15 min)
uv run python scripts/gdelt_poller.py

# RSS news aggregation (tier-1 every 15 min, tier-2 every 60 min)
# WARNING: First run is CPU-heavy (embedding ~5000+ article chunks). Expect 10-25 min.
uv run python scripts/rss_daemon.py

# ACLED armed conflict events (daily)
# Requires ACLED_EMAIL + ACLED_PASSWORD in .env
uv run python scripts/acled_poller.py

# Government travel advisories (daily)
# No API keys needed (US State Dept + UK FCDO are public)
uv run python scripts/advisory_poller.py
```

All pollers:
- Write `IngestRun` rows to PostgreSQL (powers the `/api/v1/sources` health endpoint)
- Handle SIGINT/SIGTERM gracefully (Ctrl+C is clean)
- Resume correctly after restart (dedup prevents reprocessing)

### Optional: Daily forecast pipeline

```bash
uv run python scripts/daily_forecast.py
```

Generates new forecast questions from recent events, predicts, and persists to PostgreSQL. Designed for cron/timer, not continuous running.

**Flags:**

| Flag | Description |
|------|-------------|
| `--max-questions N` | Override daily budget |
| `--skip-outcomes` | Skip outcome resolution phase |
| `--dry-run` | Generate questions only, don't predict or persist |
| `--seed-countries` | One forecast per tracked country (initial seeding) |

---

## 8. Production Frontend Build

For serving static files instead of Vite dev server:

```bash
cd frontend
npm run build
```

Outputs to `frontend/dist/`. Serve via nginx, caddy, or any static file server. Configure reverse proxy for `/api/*` -> uvicorn.

**Nginx example:**

```nginx
server {
    listen 80;

    root /path/to/geopol/frontend/dist;

    location / {
        try_files $uri $uri/ /index.html;
    }

    location /api/ {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

---

## 9. CLI Forecast (Quick Test)

Test the forecast engine directly without the API:

```bash
uv run python scripts/forecast.py -q "Will NATO expand in the next 6 months?"
```

**Flags:**

| Flag | Description |
|------|-------------|
| `-q "..."` | Question text (required) |
| `--verbose` | Full reasoning trace |
| `--json` | JSON output |
| `--no-tkg` | LLM-only mode (skip TKG) |

---

## 10. Docker Deployment (All-in-One)

```bash
docker compose up -d --build
```

This starts `postgres`, `redis`, and `api` containers. The API image uses `uv sync --no-dev` and runs uvicorn with `--factory`.

Note: Poller daemons are **not** containerized. Run them on the host or add your own Dockerfile/compose entries.

---

## Startup Dependency Graph

```
Docker Compose
  ├── PostgreSQL (port 5432)
  └── Redis (port 6379)
         │
    alembic upgrade head
         │
    bootstrap.py (optional, ~20 min)
         │
    ┌────┴────┐
    │  uvicorn │ ← API server (port 8000)
    │  --factory
    └─────────┘
         │
    ┌────┴─────────────────────────┐
    │  Vite dev server (port 5173) │ ← Frontend
    │  proxies /api → :8000        │
    └──────────────────────────────┘

Independent (start anytime after PostgreSQL):
    scripts/gdelt_poller.py     → SQLite (data/events.db)
    scripts/rss_daemon.py       → ChromaDB (chroma_db/)
    scripts/acled_poller.py     → SQLite (data/events.db) + PostgreSQL
    scripts/advisory_poller.py  → PostgreSQL (IngestRun) + in-memory cache
    scripts/daily_forecast.py   → PostgreSQL + Gemini API
```

---

## Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| `alembic upgrade` fails | Docker not running | `docker compose up -d postgres` |
| `ModuleNotFoundError: google.genai` | Missing LLM deps | `uv sync --extra llm` |
| `ModuleNotFoundError: feedparser` | Missing ingest deps | `uv sync --extra ingest` |
| RSS daemon pins CPU for 10+ min | First-run cold start (embedding thousands of articles) | Wait. Subsequent cycles take seconds. |
| `/sources` shows "Never run" | Pollers haven't completed a cycle | Start the relevant poller script |
| Polymarket 429 errors | Gemini rate limit exceeded | Set `POLYMARKET_ENABLED=false`, restart uvicorn |
| Bootstrap hangs on "collect" | GDELT API slow/down | Retry later, or `--force-stage collect` |
| Frontend CORS error | API not running or wrong port | Verify uvicorn on :8000, check `CORS_ORIGINS` |
| `async_generator does not support async context manager` | Bug in pre-fix rss_daemon.py | Pull latest (commit `bbb955c` fixes it) |

---

## Quick Start Checklist

```
[ ] git clone + cd geopol
[ ] uv sync
[ ] cd frontend && npm ci && cd ..
[ ] cp .env.example .env → set GEMINI_API_KEY
[ ] docker compose up -d postgres redis
[ ] uv run alembic upgrade head
[ ] uv run python scripts/bootstrap.py
[ ] uv run python scripts/preflight.py        ← verify everything
[ ] Terminal 1: uv run uvicorn src.api.app:create_app --factory --reload --port 8000
[ ] Terminal 2: cd frontend && npm run dev
[ ] Open http://localhost:5173
[ ] Optional: uv run python scripts/gdelt_poller.py
[ ] Optional: uv run python scripts/rss_daemon.py
[ ] Optional: uv run python scripts/acled_poller.py
[ ] Optional: uv run python scripts/advisory_poller.py
[ ] Optional: uv run python scripts/daily_forecast.py --seed-countries
[ ] Optional: uv run python scripts/train_tirgn.py
```
