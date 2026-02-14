# Architecture Patterns: v2.0 Integration with Existing Codebase

**Domain:** Geopolitical forecasting engine operationalization
**Researched:** 2026-02-14
**Confidence:** HIGH (derived from codebase inspection + official SQLite/Streamlit documentation)
**Supersedes:** Previous ARCHITECTURE.md (2026-01-31, Llama-TGL deep fusion -- cancelled)

---

## 1. Current Architecture (v1.1 Baseline)

Before defining integration points, the existing architecture must be stated precisely.

### Process Model

Single-process, batch-invoked. No daemons, no long-running processes. Everything runs as:

```
uv run python scripts/bootstrap.py        # One-shot pipeline
uv run python scripts/train_tkg_jax.py    # One-shot training
```

### Data Flow (v1.1)

```
GDELT Events API
    |
    v
[GDELTHistoricalCollector]  ->  data/gdelt/raw/*.csv
    |
    v
[GDELTDataProcessor]  ->  data/gdelt/processed/events.parquet
    |
    v
[EventStorage]  ->  data/events.db (SQLite, WAL mode)
    |
    v
[TemporalKnowledgeGraph]  ->  NetworkX MultiDiGraph (in-memory)
    |                          + data/graphs/knowledge_graph.graphml (persisted)
    |
    +---> [PartitionManager]  ->  data/partitions/{window_id}/graph.graphml
    |
    v
[DataAdapter]  ->  numpy quadruples (subject, relation, object, timestep)
    |
    v
[REGCNWrapper / REGCN]  ->  models/tkg/regcn_trained.pt (PyTorch checkpoint)
    |
    v
[TKGPredictor]  ->  link prediction scores
    |
    +---> [ReasoningOrchestrator]  ->  Gemini API  ->  ScenarioTree
    |
    v
[EnsemblePredictor]  ->  EnsemblePrediction + ForecastOutput
    |
    v
[TemperatureScaler]  ->  calibrated confidence
    |
    v
(stdout / return value)  <-- NO persistent forecast storage exists
```

### Critical Observation: Missing Forecast Persistence

The current system has **no table for storing forecast results**. `data/events.db` has `events` and `ingestion_stats` tables only. Predictions are returned as in-memory `ForecastOutput` objects and printed to stdout. This is the single largest gap for v2.0: both Streamlit and dynamic calibration require historical forecast records.

### Concurrency Model (v1.1)

None. Single process, single thread (aside from NetworkX internals). SQLite connections are opened-and-closed per operation via `DatabaseConnection.get_connection()` context manager. WAL mode is already enabled (`src/database/connection.py`, line 51: `PRAGMA journal_mode = WAL`).

---

## 2. Recommended v2.0 Architecture

### Process Topology

v2.0 introduces **three long-running processes** on a single server:

```
Process 1: Streamlit Frontend          (streamlit run scripts/app.py)
Process 2: Ingest Daemon               (uv run python scripts/ingest_daemon.py)
Process 3: Daily Pipeline              (systemd timer -> scripts/daily_pipeline.py)
```

Plus one periodic batch job:

```
Weekly Job: TKG Retraining             (systemd timer -> scripts/retrain_tkg.py)
```

**Why separate processes, not threads within Streamlit:**

1. Streamlit's execution model reruns the entire script on every user interaction. Embedding a scheduler inside Streamlit would restart on every page click.
2. The ingest daemon must run even when nobody has a browser open.
3. The daily pipeline runs a Gemini API call loop that takes minutes. Blocking a Streamlit thread would freeze the UI.
4. Process isolation means a crash in ingest does not kill the frontend.

### Concurrency Model

```
                    SQLite (WAL mode)
                   /        |         \
          [Streamlit]  [Ingest]  [Daily Pipeline]
          READ-ONLY    WRITES    WRITES (serialized)

Constraint: Only ONE writer at a time.
WAL mode allows unlimited concurrent readers alongside one writer.
```

**Rules:**
- Streamlit opens connections with `PRAGMA query_only = ON` -- read-only mode.
- Ingest daemon is the high-frequency writer (every 15 min).
- Daily pipeline writes predictions. If it collides with an active ingest write, SQLite's busy_timeout (set to 30s) handles the wait.
- Weekly retraining reads from SQLite, writes model checkpoints to filesystem. No SQLite write contention.

### Directory Structure (New and Modified)

```
src/
    ingest/                          # NEW MODULE
        __init__.py
        gdelt_fetcher.py             # 15-min GDELT update feed client
        incremental_updater.py       # Graph update without full rebuild
        daemon.py                    # APScheduler-based daemon loop
    monitoring/                      # NEW MODULE
        __init__.py
        health_checks.py             # Data freshness, model staleness, disk usage
        metrics.py                   # In-memory metric counters
    calibration/
        temperature_scaler.py        # MODIFIED: migrate persistence to SQLite
        outcome_tracker.py           # NEW: compare predictions vs GDELT ground truth
        weight_optimizer.py          # NEW: scipy.optimize per-CAMEO alpha weights
    forecasting/
        ensemble_predictor.py        # MODIFIED: accept dynamic alpha dict
        tkg_predictor.py             # MODIFIED: abstract model interface
        tkg_models/
            regcn_wrapper.py         # MODIFIED: implement TKGModelProtocol
            model_protocol.py        # NEW: Protocol class for pluggable TKG models
            hismatch_wrapper.py      # NEW: replacement model (post-research)
            data_adapter.py          # LIKELY MODIFIED: handle new model's format
    database/
        schema.sql                   # MODIFIED: add predictions + calibration tables
        models.py                    # MODIFIED: add Prediction, CalibrationRecord models
        storage.py                   # MODIFIED: add prediction CRUD methods
        connection.py                # MINOR: add read-only connection factory
    frontend/                        # NEW MODULE
        __init__.py
        app.py                       # Streamlit entry point
        pages/
            forecast.py              # Live forecast display
            history.py               # Historical track record + calibration plots
            query.py                 # Interactive on-demand queries
            health.py                # System health monitoring
        components/
            scenario_tree.py         # Scenario tree visualization
            calibration_plot.py      # Calibration reliability diagrams
            rate_limiter.py          # Per-session rate limiting
scripts/
    app.py                           # NEW: `streamlit run scripts/app.py`
    ingest_daemon.py                 # NEW: starts ingest loop
    daily_pipeline.py                # NEW: daily forecast orchestration
    retrain_tkg.py                   # EXISTS: may need minor updates
    bootstrap.py                     # EXISTS: no changes needed
```

---

## 3. Integration Point Analysis

### 3.1 Streamlit Integration

**Architecture decision: Separate process, read-only SQLite access.**

#### Where Streamlit fits

Streamlit runs as its own process (`streamlit run scripts/app.py`). It does NOT wrap the existing CLI. It does NOT share a process with the forecast pipeline. It reads all data from SQLite and the filesystem.

#### How Streamlit reads forecast results

Direct SQLite access in read-only mode. No API layer needed for a single-server deployment:

```python
# src/frontend/app.py (conceptual)
import sqlite3
import streamlit as st

@st.cache_data(ttl=300)  # Cache 5 min, avoid re-querying on every widget interaction
def get_latest_forecasts(n: int = 10):
    conn = sqlite3.connect("data/events.db")
    conn.execute("PRAGMA query_only = ON")
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM predictions ORDER BY created_at DESC LIMIT ?", (n,)
    ).fetchall()
    conn.close()
    return rows
```

**Why not an API layer:** Adding FastAPI between Streamlit and SQLite adds process management complexity, deployment surface, and latency for zero benefit on a single server. The data source is a local file. Read-only SQLite connections are safe for concurrent access under WAL.

**Why `st.cache_data` with TTL:** Streamlit reruns the script on every interaction. Without caching, every widget click triggers a database query. A 5-minute TTL balances freshness with load.

#### Concurrent access handling

SQLite WAL mode (already enabled in `src/database/connection.py` line 51) guarantees:
- Unlimited concurrent readers
- One writer at a time
- Readers never blocked by writers
- Writers never blocked by readers
- Readers get snapshot isolation -- they see the database state at the start of their transaction, even if writes happen during the read.

Streamlit opens read-only connections. The ingest daemon and daily pipeline are the writers. No read-write conflicts.

**New helper in `src/database/connection.py`:**

```python
def get_readonly_connection(db_path: str = "data/events.db") -> sqlite3.Connection:
    """Read-only connection for Streamlit and other readers."""
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA query_only = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.row_factory = sqlite3.Row
    return conn
```

#### Session management and rate limiting

Streamlit has built-in session state (`st.session_state`). Rate limiting uses this:

```python
# src/frontend/components/rate_limiter.py
import time
import streamlit as st

MAX_QUERIES_PER_HOUR = 10  # Per browser session

def check_rate_limit() -> bool:
    if "query_timestamps" not in st.session_state:
        st.session_state.query_timestamps = []
    now = time.time()
    st.session_state.query_timestamps = [
        ts for ts in st.session_state.query_timestamps if now - ts < 3600
    ]
    if len(st.session_state.query_timestamps) >= MAX_QUERIES_PER_HOUR:
        return False
    st.session_state.query_timestamps.append(now)
    return True
```

**Nuance:** Streamlit session state is per-browser-tab, not per-IP. Bypass-by-new-tab is trivial. For a demo, this is acceptable. For hardened production, add IP-based rate limiting at the reverse proxy (nginx) level.

#### Entry point

New file `scripts/app.py`, mirroring the existing `scripts/bootstrap.py` pattern:

```python
#!/usr/bin/env python
"""Streamlit entry point. Run: streamlit run scripts/app.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.frontend.app import main
main()
```

### 3.2 Micro-batch Ingest Architecture

**Architecture decision: Separate daemon process using APScheduler, writing to existing SQLite schema, performing incremental graph updates.**

#### How 15-minute ingest integrates with bootstrap

It does NOT integrate with the bootstrap pipeline. The bootstrap pipeline (`src/bootstrap/`) is a one-shot zero-to-operational initialization. The ingest daemon is a steady-state operation that runs AFTER bootstrap completes.

```
Bootstrap (one-shot):     collect -> process -> graph -> persist -> index
Ingest daemon (ongoing):  fetch -> deduplicate -> insert -> incremental_graph_update
```

The bootstrap pipeline remains unchanged. The ingest daemon is a new, independent pipeline.

#### Database model changes

The existing `events` table and `ingestion_stats` table are sufficient for micro-batch ingest. The `INSERT OR IGNORE` with `content_hash + time_window` deduplication (`src/database/storage.py`, line 83-84) already handles duplicate events gracefully. No schema changes needed for the events table.

**New table for daemon health monitoring:**

```sql
CREATE TABLE IF NOT EXISTS ingest_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at TEXT NOT NULL,
    completed_at TEXT,
    status TEXT NOT NULL DEFAULT 'running',  -- running, completed, failed
    events_fetched INTEGER DEFAULT 0,
    events_inserted INTEGER DEFAULT 0,
    error_message TEXT,
    duration_seconds REAL
);
```

#### Incremental graph update

This is the hardest integration point. The current `TemporalKnowledgeGraph.add_events_batch()` (`src/knowledge_graph/graph_builder.py`, line 160) opens a raw SQLite connection and reads ALL events with `ORDER BY event_date DESC`. For 15-minute updates, we need incremental addition.

**Approach:** New method on `TemporalKnowledgeGraph`:

```python
def add_events_incremental(self, events: List[Dict]) -> Dict:
    """Add new events to existing in-memory graph without full rebuild.

    Reuses the existing add_event_from_db_row() method, which already
    handles individual event normalization, classification, and graph insertion.
    """
    stats = {"added": 0, "skipped": 0}
    for event in events:
        result = self.add_event_from_db_row(event)
        if result:
            stats["added"] += 1
        else:
            stats["skipped"] += 1
    return stats
```

The `add_event_from_db_row()` method (`graph_builder.py`, line 54) already handles individual event addition including entity resolution, relation classification, and edge creation. It just needs to be called incrementally rather than through the batch-from-SQLite path.

**File modified:** `src/knowledge_graph/graph_builder.py` -- add `add_events_incremental()` method.

#### Partition routing for new events

New events must be routed to the correct partition. The `PartitionManager` (`partition_manager.py`, line 65) uses temporal windowing -- each event goes to the partition matching its timestamp.

**Approach:** After incremental graph update, if the event's time window maps to an existing partition, append to that partition's GraphML. If no partition exists for the window, create a new one.

**New method on `PartitionManager`:**

```python
def add_events_to_partition(
    self, events: List[Dict], graph: nx.MultiDiGraph
) -> str:
    """Route new events to appropriate partition, creating if needed."""
```

**File modified:** `src/knowledge_graph/partition_manager.py`

#### Process management: APScheduler within a standalone daemon

**Why APScheduler, not systemd timer:** The 15-minute cycle is too frequent for systemd timer overhead (process startup, Python import, SQLite connection setup on every tick). APScheduler runs a persistent Python process with the GDELT client already initialized, connections warm.

**Why not APScheduler within Streamlit:** Streamlit reruns scripts on user interaction. An embedded scheduler would restart on every page load. Separate process is mandatory.

**File:** `scripts/ingest_daemon.py`

```python
#!/usr/bin/env python
"""GDELT micro-batch ingest daemon. Run as: uv run python scripts/ingest_daemon.py"""
from apscheduler.schedulers.blocking import BlockingScheduler

def ingest_cycle():
    """Single 15-minute ingest cycle."""
    # 1. Fetch latest GDELT 15-min update
    # 2. Deduplicate against existing events
    # 3. Insert new events to SQLite
    # 4. Incrementally update in-memory graph
    # 5. Record ingest stats in ingest_runs table

scheduler = BlockingScheduler()
scheduler.add_job(ingest_cycle, 'interval', minutes=15, max_instances=1)
scheduler.start()
```

`max_instances=1` prevents overlap if a cycle takes longer than 15 minutes.

**Supervision via systemd:**

```ini
[Unit]
Description=Geopol GDELT Ingest Daemon
After=network.target

[Service]
Type=simple
WorkingDirectory=/home/kondraki/personal/geopol
ExecStart=/home/kondraki/.local/bin/uv run python scripts/ingest_daemon.py
Restart=on-failure
RestartSec=30

[Install]
WantedBy=multi-user.target
```

### 3.3 TKG Predictor Replacement

**Architecture decision: Define a Protocol interface, implement existing RE-GCN against it, implement new algorithm against it, swap via configuration.**

#### Protocol interface

New file `src/forecasting/tkg_models/model_protocol.py`:

```python
from typing import Protocol, List, Tuple, Optional
import numpy as np
from pathlib import Path

class TKGModelProtocol(Protocol):
    """Interface for pluggable TKG prediction models."""

    @property
    def trained(self) -> bool: ...

    def fit(self, quadruples: np.ndarray, **kwargs) -> None: ...

    def predict_object(
        self, subject_id: int, relation_id: int, k: int
    ) -> List[Tuple[int, float]]: ...

    def predict_relation(
        self, subject_id: int, object_id: int, k: int
    ) -> List[Tuple[int, float]]: ...

    def score_triple(
        self, subject_id: int, relation_id: int, object_id: int
    ) -> float: ...

    def save(self, path: Path) -> None: ...

    def load(self, path: Path) -> None: ...
```

#### What changes in `src/training/`

The training pipeline (JAX/jraph) remains. The RE-GCN model in `src/training/models/regcn.py` (PyTorch, 623 lines) is the inference model. The JAX training script (`scripts/train_tkg_jax.py`) produces a checkpoint that the PyTorch model loads.

**For a new algorithm (e.g., HisMatch):**

1. New training implementation: `src/training/models/hismatch_jax.py` (JAX/jraph, for training on GPU)
2. New inference wrapper: `src/forecasting/tkg_models/hismatch_wrapper.py` (implements `TKGModelProtocol`)
3. New training script: `scripts/train_hismatch_jax.py`
4. Checkpoint format may differ -- the `DataAdapter` stays the same (it converts NetworkX to quadruples regardless of model).

**Files modified:**
- `src/forecasting/tkg_predictor.py`: Change from directly importing `REGCNWrapper` to accepting any `TKGModelProtocol`. The `TKGPredictor.__init__()` currently takes `model: Optional[REGCNWrapper]` (line 160) -- change type hint to `model: Optional[TKGModelProtocol]`.
- `src/forecasting/tkg_models/regcn_wrapper.py`: Add explicit Protocol compliance (no structural changes needed -- it already implements `predict_object`, `predict_relation`, `score_triple`, `save_model`, `load_model`; method names may need aliasing).

#### Embedding dimension impact

Current: 200-dim embeddings (`regcn_wrapper.py` line 80: `embedding_dim: int = 200`). If the new model uses a different dimension:

- `REGCNWrapper` stores `embedding_dim` as an instance attribute. The `TKGPredictor` does not depend on embedding dimension at all -- it only sees `confidence` floats from predictions.
- `EnsemblePredictor` does not touch embeddings.
- The only consumer of raw embeddings is `REGCNWrapper.get_embedding()` (line 634), which is not used in the prediction pipeline.

**Conclusion:** Embedding dimension change has ZERO downstream impact. The abstraction boundary at `TKGPredictor.predict_future_events()` returns `List[Dict[str, Union[str, float]]]` -- no embedding leaks through.

#### Data format compatibility

The `DataAdapter` (`data_adapter.py`) converts `NetworkX MultiDiGraph -> numpy (N, 4) quadruples [subject_id, relation_id, object_id, timestep]`. This is the standard TKG format used by RE-GCN, TiRGN, HisMatch, and every other TKG algorithm in the literature. No conversion needed.

### 3.4 Dynamic Calibration Module

**Architecture decision: New `src/calibration/outcome_tracker.py` and `src/calibration/weight_optimizer.py` modules. Per-CAMEO weights stored in SQLite. Weight optimization runs after each daily prediction cycle.**

#### New SQLite tables

```sql
-- Forecast results for tracking and display
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    question TEXT NOT NULL,
    probability REAL NOT NULL,
    confidence REAL NOT NULL,
    category TEXT,              -- conflict/diplomatic/economic
    cameo_root TEXT,            -- CAMEO root code (01-20)
    llm_probability REAL,
    tkg_probability REAL,
    alpha_used REAL,
    temperature_used REAL,
    reasoning_summary TEXT,
    scenario_tree_json TEXT,    -- serialized ScenarioTree (Pydantic .model_dump_json())
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    resolved_at TEXT,           -- when outcome was determined
    outcome INTEGER             -- 1=occurred, 0=did not occur, NULL=pending
);

CREATE INDEX IF NOT EXISTS idx_predictions_created ON predictions(created_at);
CREATE INDEX IF NOT EXISTS idx_predictions_cameo ON predictions(cameo_root);
CREATE INDEX IF NOT EXISTS idx_predictions_outcome ON predictions(outcome);

-- Per-CAMEO calibration weights
CREATE TABLE IF NOT EXISTS calibration_weights (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cameo_root TEXT NOT NULL,
    alpha REAL NOT NULL,         -- LLM weight for this category
    temperature REAL NOT NULL,
    sample_count INTEGER NOT NULL,
    brier_score REAL,
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(cameo_root)
);

-- Outcome tracking for calibration feedback loop
CREATE TABLE IF NOT EXISTS outcome_records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    prediction_id INTEGER NOT NULL REFERENCES predictions(id),
    gdelt_event_ids TEXT,       -- JSON array of matching GDELT event IDs
    resolution_method TEXT,     -- 'gdelt_match', 'manual', 'timeout'
    resolved_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Ingest daemon health tracking
CREATE TABLE IF NOT EXISTS ingest_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at TEXT NOT NULL,
    completed_at TEXT,
    status TEXT NOT NULL DEFAULT 'running',
    events_fetched INTEGER DEFAULT 0,
    events_inserted INTEGER DEFAULT 0,
    error_message TEXT,
    duration_seconds REAL
);
```

**File modified:** `src/database/schema.sql` -- additive only, existing tables untouched.

#### Integration with existing TemperatureScaler

The existing `TemperatureScaler` (`src/calibration/temperature_scaler.py`) already supports per-category temperatures via `self.temperatures: Dict[str, float]` (line 58). It stores temperatures as pickle files in `./data/temperature/temperatures.pkl` (lines 300-311).

**Change:** Move temperature persistence from pickle to the `calibration_weights` SQLite table. This unifies all calibration state in one place and makes it queryable by the monitoring dashboard.

**Files modified:**
- `src/calibration/temperature_scaler.py`: Replace `_save_temperatures()` (line 300) and `load_temperatures()` (line 313) with SQLite read/write. The `fit()` and `calibrate()` interfaces stay the same.

#### Per-CAMEO weight storage and consumption

Weights stored in `calibration_weights` table. `EnsemblePredictor` loads them at initialization:

```python
# Modified EnsemblePredictor.__init__() -- new parameter
def __init__(
    self,
    ...,
    category_weights: Optional[Dict[str, float]] = None,
):
    ...
    self.category_weights = category_weights or {}
```

The `_combine_predictions()` method (line 517) changes from:
```python
prob = self.alpha * llm_pred.probability + (1 - self.alpha) * tkg_pred.probability
```
to:
```python
effective_alpha = self.category_weights.get(category, self.alpha)
prob = effective_alpha * llm_pred.probability + (1 - effective_alpha) * tkg_pred.probability
```

**File modified:** `src/forecasting/ensemble_predictor.py` -- `__init__` signature and `_combine_predictions()` method.

#### When weight optimization runs

After each daily prediction cycle completes. The daily pipeline script calls `weight_optimizer.optimize()` which:

1. Queries `predictions` table for resolved predictions (where `outcome IS NOT NULL`)
2. Groups by CAMEO root category
3. For each category with >= 20 samples: runs `scipy.optimize.minimize` to find optimal alpha minimizing Brier score
4. Writes updated weights to `calibration_weights` table
5. Next prediction cycle loads fresh weights

**New file:** `src/calibration/weight_optimizer.py`

### 3.5 Daily Automation

**Architecture decision: systemd timer (not cron) calling a Python script. Separate from the ingest daemon.**

#### Orchestration

systemd timer fires once daily at a configured time (e.g., 06:00 UTC):

```ini
# /etc/systemd/system/geopol-daily.timer
[Timer]
OnCalendar=*-*-* 06:00:00
Persistent=true

[Install]
WantedBy=timers.target
```

```ini
# /etc/systemd/system/geopol-daily.service
[Service]
Type=oneshot
WorkingDirectory=/home/kondraki/personal/geopol
ExecStart=/home/kondraki/.local/bin/uv run python scripts/daily_pipeline.py
```

**Why systemd timer over cron:** `Persistent=true` ensures missed runs (server was off) execute on next boot. systemd logging integrates with journalctl. Dependencies can be declared (`After=geopol-ingest.service`).

**Why systemd timer over APScheduler for daily:** Unlike the 15-min ingest, a daily job does not benefit from a warm process. Cold start overhead (2-3s) is negligible relative to the minutes-long pipeline. systemd provides crash logging, restart semantics, and timer persistence for free.

#### Daily pipeline script

New file `scripts/daily_pipeline.py`:

```python
#!/usr/bin/env python
"""Daily forecast pipeline. Triggered by systemd timer."""
# 1. Load current knowledge graph (from latest partition or full graph)
# 2. Load trained TKG model
# 3. Load dynamic calibration weights from SQLite
# 4. Generate N forecast questions (from GDELT trends or curated list)
# 5. For each question:
#    a. Run EnsemblePredictor.predict() with dynamic weights
#    b. Store result in predictions table
# 6. Run outcome resolution on past predictions
# 7. Run weight optimization if enough resolved predictions
# 8. Write health check timestamp
```

#### Interaction with ingest daemon

The daily pipeline reads from SQLite (events table populated by ingest daemon). It does not coordinate with the ingest daemon directly. If a 15-min ingest cycle and the daily pipeline write simultaneously, SQLite's WAL mode handles it: the daily pipeline's write will wait (up to busy_timeout) for the ingest write to complete.

**Recommended:** Set `busy_timeout` to 30000ms (30 seconds) in all writer connections:

```python
conn.execute("PRAGMA busy_timeout = 30000")
```

This should be added to `DatabaseConnection.get_connection()` in `src/database/connection.py`.

#### Monitoring

New file `src/monitoring/health_checks.py`:

```python
def check_data_freshness(db_path: str) -> Dict:
    """Check how recent the latest events are.
    Alert if > 2 hours stale (ingest daemon may be down)."""

def check_prediction_freshness(db_path: str) -> Dict:
    """Check when last prediction was made.
    Alert if > 26 hours stale (daily pipeline may have failed)."""

def check_model_staleness(model_path: str) -> Dict:
    """Check model checkpoint age.
    Alert if > 8 days stale (weekly retraining may have failed)."""

def check_disk_usage(data_dir: str) -> Dict:
    """Check data/ directory size.
    Alert if > 80% of available disk."""

def check_ingest_health(db_path: str) -> Dict:
    """Check last N ingest_runs for failures.
    Alert if > 3 consecutive failures."""
```

**Log management:** All processes write to systemd journal via stdout. `journalctl -u geopol-ingest` and `journalctl -u geopol-daily` for inspection. No separate log files needed on a single-server deployment.

---

## 4. Complete v2.0 Data Flow

```
                    GDELT 15-min Feed
                          |
                          v
              +-------------------------+
              |  src/ingest/            |
              |  gdelt_fetcher.py       |  (every 15 min via APScheduler)
              |  incremental_updater.py |
              +-------------------------+
                          |
                    [INSERT events]
                          |
                          v
            +----------------------------+
            |    data/events.db          |
            |    (SQLite, WAL mode)      |
            |                            |
            |  tables:                   |
            |    events                  |  <-- ingest writes here
            |    predictions             |  <-- daily pipeline writes here
            |    calibration_weights     |  <-- weight optimizer writes here
            |    outcome_records         |  <-- outcome tracker writes here
            |    ingestion_stats         |  <-- ingest writes here (existing)
            |    ingest_runs             |  <-- ingest daemon health tracking
            +----------------------------+
                  |            |
         [daily]  |            |  [streamlit, read-only]
                  v            v
    +---------------------+   +---------------------+
    | scripts/             |   | src/frontend/        |
    | daily_pipeline.py    |   | app.py               |
    |                      |   |                      |
    | 1. Load graph        |   | Pages:               |
    | 2. Load TKG model    |   |  - forecast.py       |
    | 3. Load dyn. weights |   |  - history.py        |
    | 4. Generate Qs       |   |  - query.py          |
    | 5. EnsemblePredictor |   |  - health.py         |
    |    -> Gemini API     |   |                      |
    |    -> TKG inference  |   | Reads: predictions,  |
    | 6. Store predictions |   |   events, calibration|
    | 7. Resolve outcomes  |   |   health checks      |
    | 8. Optimize weights  |   +---------------------+
    +---------------------+
              |
              v
    +---------------------+
    | On-demand queries    |
    | (Streamlit query.py) |
    |                      |
    | User submits question|
    | -> Rate limit check  |
    | -> EnsemblePredictor |
    |    -> Gemini API     |
    |    -> TKG inference  |
    | -> Display result    |
    | -> Store in preds    |
    +---------------------+
```

### Data Flow Changes: Before vs After

| Stage | v1.1 (Before) | v2.0 (After) |
|-------|--------------|-------------|
| Data collection | One-shot, 30 days backfill | Continuous 15-min micro-batch + initial backfill |
| Event storage | Same | Same (`INSERT OR IGNORE` handles overlap) |
| Graph construction | Full rebuild from SQLite | Incremental update + periodic full rebuild |
| TKG training | Manual CLI invocation | Weekly automated via systemd timer |
| Prediction | Manual CLI invocation | Daily automated + on-demand via Streamlit |
| Forecast storage | None (stdout only) | SQLite `predictions` table |
| Calibration | Fixed alpha=0.6, pickle temps | Dynamic per-CAMEO alpha, SQLite storage |
| Outcome tracking | None | Automated GDELT-based resolution |
| Monitoring | None | Health checks exposed via Streamlit |
| Frontend | None (CLI only) | Streamlit multi-page app |

---

## 5. New vs Modified Components

### New Files (22 files)

| File | Purpose | Depends On |
|------|---------|------------|
| `src/ingest/__init__.py` | Module init | -- |
| `src/ingest/gdelt_fetcher.py` | GDELT 15-min update feed client | Reuses patterns from `src/training/data_collector.py` |
| `src/ingest/incremental_updater.py` | Incremental graph update logic | `src/knowledge_graph/graph_builder.py` |
| `src/ingest/daemon.py` | APScheduler loop + health tracking | `gdelt_fetcher.py`, `incremental_updater.py` |
| `src/monitoring/__init__.py` | Module init | -- |
| `src/monitoring/health_checks.py` | Data/model/disk freshness checks | `src/database/connection.py` |
| `src/monitoring/metrics.py` | In-memory counters for ingest/predict cycles | -- |
| `src/calibration/outcome_tracker.py` | Compare predictions vs GDELT ground truth | `src/database/storage.py` |
| `src/calibration/weight_optimizer.py` | Per-CAMEO alpha optimization via scipy | `scipy.optimize`, `src/database/connection.py` |
| `src/forecasting/tkg_models/model_protocol.py` | Protocol for pluggable TKG models | -- |
| `src/frontend/__init__.py` | Module init | -- |
| `src/frontend/app.py` | Streamlit main app with page routing | All read paths |
| `src/frontend/pages/forecast.py` | Live forecast display page | `src/database/storage.py` |
| `src/frontend/pages/history.py` | Track record + calibration plots | `src/database/storage.py`, `matplotlib` |
| `src/frontend/pages/query.py` | Interactive on-demand queries | `src/forecasting/ensemble_predictor.py` |
| `src/frontend/pages/health.py` | System health dashboard | `src/monitoring/health_checks.py` |
| `src/frontend/components/scenario_tree.py` | Scenario tree visualization widget | `streamlit` |
| `src/frontend/components/calibration_plot.py` | Reliability diagrams | `matplotlib`, `numpy` |
| `src/frontend/components/rate_limiter.py` | Session-based rate limiting | `streamlit` |
| `scripts/app.py` | Streamlit entry point | `src/frontend/app.py` |
| `scripts/ingest_daemon.py` | Ingest daemon entry point | `src/ingest/daemon.py` |
| `scripts/daily_pipeline.py` | Daily pipeline entry point | `src/forecasting/ensemble_predictor.py` |

### Modified Files (10 files)

| File | Change | Risk |
|------|--------|------|
| `src/database/schema.sql` | Add `predictions`, `calibration_weights`, `outcome_records`, `ingest_runs` tables + indexes | LOW -- additive only, existing tables untouched |
| `src/database/models.py` | Add `Prediction`, `CalibrationRecord`, `IngestRun` dataclasses | LOW -- additive |
| `src/database/storage.py` | Add prediction CRUD, calibration weight read/write, ingest run recording | LOW -- additive methods |
| `src/database/connection.py` | Add `get_readonly_connection()` factory, explicit `busy_timeout = 30000` | LOW -- new function, minor change to existing |
| `src/forecasting/ensemble_predictor.py` | Accept `Dict[str, float]` for per-category alpha; load from `category_weights` parameter | MEDIUM -- modifies `_combine_predictions()` core logic |
| `src/forecasting/tkg_predictor.py` | Change `model` type hint from `REGCNWrapper` to `TKGModelProtocol` | LOW -- type annotation only |
| `src/forecasting/tkg_models/regcn_wrapper.py` | Ensure Protocol compliance (may need method name aliases for `save`/`load`) | LOW |
| `src/calibration/temperature_scaler.py` | Migrate persistence from pickle to SQLite `calibration_weights` table | MEDIUM -- changes I/O paths |
| `src/knowledge_graph/graph_builder.py` | Add `add_events_incremental(events: List[Dict])` method | LOW -- new method, existing code untouched |
| `src/knowledge_graph/partition_manager.py` | Add `add_events_to_partition()` method | LOW -- new method |

### Unchanged Files (17+ key files)

These require NO modifications:

- `src/bootstrap/*` -- bootstrap remains a one-shot initializer
- `src/training/*` -- training pipeline runs independently
- `src/forecasting/reasoning_orchestrator.py` -- Gemini integration unchanged
- `src/forecasting/scenario_generator.py` -- scenario generation unchanged
- `src/forecasting/gemini_client.py` -- Gemini API client unchanged
- `src/forecasting/models.py` -- Pydantic models (`ForecastOutput`, `Scenario`, etc.) unchanged
- `src/forecasting/rag_pipeline.py` -- RAG pipeline unchanged
- `src/forecasting/graph_validator.py` -- graph validation unchanged
- `src/knowledge_graph/entity_normalization.py` -- normalization unchanged
- `src/knowledge_graph/relation_classification.py` -- classification unchanged
- `src/knowledge_graph/persistence.py` -- GraphML persistence unchanged
- `src/knowledge_graph/partition_index.py` -- partition index unchanged
- `src/knowledge_graph/boundary_resolver.py` -- cross-partition queries unchanged
- `scripts/bootstrap.py` -- bootstrap script unchanged
- `scripts/train_tkg_jax.py` -- training script unchanged

---

## 6. Suggested Build Order Based on Dependencies

The dependency graph dictates build order. Each phase produces a working increment.

```
Phase 1: Foundation (no dependencies on other phases)
    - Schema changes: predictions, calibration_weights, outcome_records, ingest_runs tables
    - Forecast persistence: store ForecastOutput -> predictions table after ensemble predict
    - Read-only DB connection factory in connection.py
    - TKGModelProtocol definition
    - Data models: Prediction, CalibrationRecord dataclasses

Phase 2: Streamlit MVP (depends on Phase 1: predictions table must exist)
    - Streamlit app skeleton with multi-page navigation
    - Forecast display page (reads predictions table)
    - History page (reads predictions + outcome data)
    - Health page (basic: event count, latest date, model age)
    - Manual seed: run a few forecasts via CLI to populate predictions table for demo

Phase 3: Ingest Daemon (depends on Phase 1: schema; independent of Phase 2)
    - GDELT 15-min update feed fetcher
    - Incremental graph updater (add_events_incremental)
    - APScheduler-based daemon loop
    - Ingest health tracking (ingest_runs table)
    - systemd service definition

Phase 4: Daily Automation (depends on Phase 1: prediction persistence)
    - Daily pipeline script
    - Question generation from GDELT trends or curated list
    - systemd timer setup
    - Integration with existing EnsemblePredictor (still using fixed alpha)

Phase 5: Dynamic Calibration (depends on Phase 4: needs prediction history with outcomes)
    - Outcome tracker: compare predictions vs GDELT events
    - Weight optimizer: per-CAMEO alpha optimization via scipy
    - Migrate TemperatureScaler persistence to SQLite
    - Connect dynamic weights to EnsemblePredictor._combine_predictions()

Phase 6: TKG Replacement (independent; can start after Phase 1, can parallel Phases 3-5)
    - Research best algorithm (separate research task, already scoped)
    - Implement new model training (JAX/jraph)
    - Implement new inference wrapper (TKGModelProtocol)
    - Validate against RE-GCN baseline on held-out data
    - Swap via configuration

Phase 7: Interactive Queries + Polish (depends on Phase 2 + Phase 4)
    - Streamlit query page with rate limiter
    - On-demand EnsemblePredictor invocation from Streamlit
    - Scenario tree visualization component
    - Calibration reliability diagram component
    - Input sanitization for public-facing queries
```

**Critical path:** Phase 1 -> Phase 2 -> Phase 4 -> Phase 5. This is the shortest path to a demonstrable, self-improving system.

**Parallelizable:** Phase 3 (ingest) and Phase 6 (TKG replacement) can run in parallel with the critical path after Phase 1 completes.

---

## 7. Anti-Patterns to Avoid

### Anti-Pattern 1: Shared In-Memory Graph Across Processes

**What:** Keeping a single NetworkX graph in memory and sharing it between Streamlit and the ingest daemon via multiprocessing.Value or shared memory.

**Why bad:** NetworkX graphs are not pickle-serializable in a way that supports concurrent modification. Shared memory requires manual locking. A graph with 50K nodes and 200K edges consumes ~400MB -- duplicating this across processes wastes memory.

**Instead:** Each process that needs graph data reads from GraphML files or queries SQLite. The ingest daemon maintains its own in-memory graph for incremental updates and periodically persists to GraphML. Streamlit never loads the full graph -- it reads aggregated statistics from SQLite or loads specific partitions on demand via the existing `PartitionManager`.

### Anti-Pattern 2: Streamlit as the Scheduler

**What:** Running APScheduler or background threads inside the Streamlit app to handle ingest and daily predictions.

**Why bad:** Streamlit reruns the entire script on every widget interaction. Background threads would be recreated on every rerun. The scheduler would have multiple instances fighting each other.

**Instead:** Separate daemon processes managed by systemd.

### Anti-Pattern 3: Polling SQLite from Streamlit for "Real-Time" Updates

**What:** Using `st.rerun()` in a loop or `time.sleep()` to poll the database for new predictions.

**Why bad:** Creates unnecessary database load. Burns Streamlit server resources spinning on polls.

**Instead:** Use `st.cache_data(ttl=300)` for data that updates every 15 minutes. Use `st.cache_data(ttl=3600)` for data that updates daily. Let users manually refresh via a button when they want the latest.

### Anti-Pattern 4: Monolithic Daily Pipeline

**What:** A single script that does ingest + graph build + train + predict + calibrate all in sequence.

**Why bad:** If training fails (GPU OOM), prediction never runs. If prediction fails (Gemini API down), calibration never runs. Components have different failure modes and cadences (ingest=15min, train=weekly, predict=daily, calibrate=daily).

**Instead:** Separate concerns: ingest is continuous (15-min daemon), training is weekly (systemd timer), prediction is daily (systemd timer), calibration runs after prediction (same script, but graceful degradation -- prediction results are stored even if calibration fails).

### Anti-Pattern 5: Embedding Dimension Coupling

**What:** Having downstream components (ensemble predictor, calibration, Streamlit) depend on TKG embedding dimensionality.

**Why bad:** Makes TKG model replacement require changes across the codebase.

**Instead:** The existing abstraction boundary is correct and must be maintained: `TKGPredictor.predict_future_events()` returns `List[Dict]` with string entities and float confidences. Embeddings never leak past `tkg_predictor.py`. Do not introduce any code path that passes raw embeddings to components outside `src/forecasting/tkg_models/`.

---

## 8. Scalability Considerations

| Concern | Current (v1.1) | v2.0 Design | Future (v3.0+) |
|---------|----------------|-------------|-----------------|
| Event volume | ~50K events (30 days) | ~200K events (90 days rolling) | Multi-source (ACLED, ICEWS) |
| SQLite writes | Bootstrap only | 15-min micro-batch + daily | Consider write batching or async WAL |
| Graph memory | Full graph in-memory (~400MB) | Same, with partition LRU cache | Move to out-of-core graph (DuckDB?) |
| Prediction storage | None | Daily + on-demand (~30-100/day) | Millions of rows over years -- needs archival |
| Concurrent users | 0 | 1-10 Streamlit sessions | Reverse proxy (nginx) + multiple Streamlit workers |
| Gemini API cost | Manual only | Daily automation + interactive | Rate limit strictly, consider response caching |
| TKG training | Manual, 30-day window | Weekly automated, 90-day window | Incremental training (fine-tune, not retrain) |
| Disk usage | ~500MB (data + graphs) | ~2-5GB (+ predictions, partitions) | Archive old partitions, compress |

---

## Sources

- [SQLite WAL mode official documentation](https://sqlite.org/wal.html) -- concurrent reader/writer semantics (HIGH confidence)
- [SQLite isolation documentation](https://sqlite.org/isolation.html) -- snapshot isolation guarantees (HIGH confidence)
- [Streamlit execution model](https://docs.streamlit.io/develop/concepts/architecture) -- script rerun semantics (HIGH confidence)
- [Streamlit threading documentation](https://docs.streamlit.io/develop/concepts/design/multithreading) -- process/thread model (HIGH confidence)
- [Streamlit caching](https://docs.streamlit.io/develop/api-reference/caching-and-state/st.cache_data) -- TTL-based caching (HIGH confidence)
- [APScheduler GitHub](https://github.com/agronholm/apscheduler) -- scheduler architecture, max_instances (HIGH confidence)
- [SQLite concurrent writes patterns](https://tenthousandmeters.com/blog/sqlite-concurrent-writes-and-database-is-locked-errors/) -- busy_timeout patterns (MEDIUM confidence)
- [Streamlit separate process computation](https://discuss.streamlit.io/t/make-apps-faster-by-moving-heavy-computation-to-a-separate-process/68541) -- process separation pattern (MEDIUM confidence)
- Codebase inspection: All file-level integration points verified against actual source code (HIGH confidence)
