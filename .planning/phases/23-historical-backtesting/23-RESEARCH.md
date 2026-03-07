# Phase 23: Historical Backtesting - Research

**Researched:** 2026-03-07
**Domain:** Walk-forward evaluation, model comparison, calibration audit, d3 visualization, bias prevention
**Confidence:** HIGH (codebase-derived research -- all findings verified against actual source files)

## Summary

Phase 23 builds an investor-grade backtesting system that re-runs predictions across historical time windows, compares TiRGN vs RE-GCN checkpoints, audits calibration drift, and presents results in a d3-powered admin panel with CSV/JSON export. The locked decisions specify 14-day overlapping windows, live re-prediction via EnsemblePredictor, per-window calibration weight snapshots, ephemeral ChromaDB temporal indexes, on-demand triggering via ProcessPoolExecutor, and cancellable background jobs.

The codebase provides strong foundations: `PolymarketAutoForecaster._build_predictor()` already demonstrates fresh-EnsemblePredictor construction with component caching. The `WeightLoader` + `CalibrationWeight` table hold the calibration state that needs snapshotting. ChromaDB's `rss_articles` collection stores `published_at` metadata per chunk, enabling date-filtered temporal index builds. The admin panel system (AdminPanel interface, AdminClient, AdminSidebar, admin route) provides a well-defined extension point. The `heavy_job_lock` + `ProcessPoolExecutor` pattern handles mutual exclusion for heavy background work.

**Primary recommendation:** Model the backtesting engine as `src/backtesting/` package with a `BacktestRunner` class that orchestrates window sliding, prediction execution, and result persistence. Wire it as a heavy job through the existing scheduler infrastructure. Build `BacktestingPanel` as a new admin section following the `AccuracyPanel` pattern.

## Standard Stack

### Core (Already in Project)

| Library | Version | Purpose | How Used in Phase 23 |
|---------|---------|---------|---------------------|
| chromadb | PersistentClient | Vector store for RAG articles | Temporal index creation per evaluation window |
| d3 | ^7.9.0 | SVG chart rendering | Brier curves, reliability diagrams, comparison charts |
| SQLAlchemy 2.0 | async ORM | PostgreSQL persistence | backtest_runs, backtest_results tables |
| Alembic | migrations | Schema evolution | Migration 009 for backtest tables |
| APScheduler 3.11.2 | AsyncIOScheduler | Background job dispatch | Heavy job wrapper for backtest runs |
| FastAPI | async API | Admin endpoints | /api/v1/admin/backtesting/* |
| Pydantic | DTOs | Request/response schemas | BacktestRunDTO, BacktestResultDTO |

### Supporting (Already in Project)

| Library | Version | Purpose | When Used |
|---------|---------|---------|-----------|
| numpy | -- | Brier score computation | Per-window metric calculation |
| scipy | -- | L-BFGS-B optimizer | Calibration weight restoration (read-only in backtest) |
| sentence-transformers | all-mpnet-base-v2 | Embedding function | ChromaDB temporal index embedding |
| llama_index | -- | RAG pipeline | Historical context retrieval during re-prediction |

### No New Dependencies Required

All libraries needed for Phase 23 are already in the project. d3 ^7.9.0 is installed with `@types/d3` ^7.4.3. ChromaDB, SQLAlchemy, APScheduler, and numpy are all present.

## Architecture Patterns

### Recommended Project Structure

```
src/backtesting/
    __init__.py
    runner.py          # BacktestRunner: window sliding, orchestration, cancellation
    evaluator.py       # Metric computation: Brier, MRR, Hits@k, reliability diagram
    temporal_index.py  # ChromaDB ephemeral temporal index builder
    weight_snapshot.py # CalibrationWeight serialization/restoration per window
    schemas.py         # Internal dataclasses for window configs, results
    export.py          # CSV/JSON export with methodology template

src/scheduler/
    heavy_runner.py    # + run_backtest() module-level function
    job_wrappers.py    # + heavy_backtest() async wrapper

src/api/
    routes/v1/admin.py # + backtest CRUD endpoints
    schemas/admin.py   # + BacktestRunDTO, BacktestResultDTO, BacktestExportDTO
    services/admin_service.py  # + get_backtest_runs(), get_backtest_results(), etc.

src/db/
    models.py          # + BacktestRun, BacktestResult ORM models

frontend/src/admin/
    panels/BacktestingPanel.ts  # Run list + drill-down + d3 charts + export
    admin-types.ts              # + BacktestRun, BacktestResult, BacktestMetrics types
    admin-client.ts             # + backtest API methods
    admin-layout.ts             # + 'backtesting' section registration
    components/AdminSidebar.ts  # + backtesting nav item
```

### Pattern 1: BacktestRunner Orchestration

**What:** Central runner that slides evaluation windows, executes predictions, collects metrics, persists results.
**When to use:** The core orchestration pattern -- called from heavy_runner.py in ProcessPoolExecutor.

```python
# Source: Derived from auto_forecaster.py _build_predictor pattern
class BacktestRunner:
    def __init__(
        self,
        run_config: BacktestRunConfig,  # window_size, slide_step, checkpoints, label
        async_session_factory,
        cancel_event: threading.Event,  # Cancellation signal
    ):
        self._config = run_config
        self._session_factory = async_session_factory
        self._cancel = cancel_event
        self._predictor_cache = None  # Heavy component caching

    def _build_predictor(self, checkpoint_path: str | None = None) -> EnsemblePredictor:
        """Build fresh EnsemblePredictor with optional checkpoint override.

        Follows auto_forecaster.py pattern: cache heavy components (RAG, TKG,
        orchestrator), return fresh EnsemblePredictor each time (mutable state).

        For model comparison: pass checkpoint_path to load specific TKG weights.
        """
        ...

    async def run(self) -> BacktestRunResult:
        """Execute full backtest: slide windows, run predictions, compute metrics.

        For each window [t0, t1]:
          1. Snapshot calibration weights at t1
          2. Build ephemeral ChromaDB index (articles <= t1)
          3. Query predictions in [t1, t2] window
          4. For each prediction: re-predict using temporal RAG + snapshot weights
          5. Compare re-prediction probability against known outcome
          6. Compute window metrics (Brier, MRR, Hits@k)
          7. Persist to backtest_results
          8. Check cancel_event, save partial results if cancelled

        Returns:
            BacktestRunResult with all window metrics + aggregate summary.
        """
        ...
```

### Pattern 2: Temporal ChromaDB Index (Ephemeral)

**What:** Per-window ChromaDB collection built from articles published before the window's prediction date.
**When to use:** Every evaluation window needs its own temporally-bounded article index.

**Critical finding:** The `rss_articles` collection stores `published_at` as ISO 8601 string in metadata per chunk. ChromaDB supports `$lte` metadata filtering on strings. Two viable strategies:

**Strategy A (Recommended): Filtered query, not separate collection.**
Instead of creating N ephemeral collections per run, create a single ephemeral collection with a date ceiling per-window by filtering at query time:

```python
# ChromaDB where filter on published_at metadata
results = collection.query(
    query_texts=[query],
    where={"published_at": {"$lte": window_end_iso}},
    n_results=top_k,
)
```

**BUT the CONTEXT.md explicitly locks "separate temporal ChromaDB index per evaluation window (ephemeral, rebuilt each run)"** because:
- published_at metadata quality varies (GDELT articles may have empty/wrong dates)
- Separate indexes make bias prevention physically impossible to violate

**Strategy B (Locked decision): Ephemeral collection rebuild.**

```python
# Build temporal index for one evaluation window
def build_temporal_index(
    source_client: chromadb.PersistentClient,
    source_collection_name: str,
    cutoff_date: str,  # ISO 8601
    temp_persist_dir: str,  # In-memory or tempdir
) -> chromadb.Collection:
    """
    1. Query all chunks from source collection with published_at <= cutoff_date
    2. Create ephemeral ChromaDB client (in-memory via chromadb.Client())
    3. Create new collection with same embedding function
    4. Bulk-add filtered chunks
    5. Return new collection for RAG pipeline injection
    """
```

**Key implementation detail:** ChromaDB's `get()` with `where` filter supports pagination via `offset` + `limit`. The full `rss_articles` collection must be queried with `where={"published_at": {"$lte": cutoff_iso}}` and all matching chunks copied to the ephemeral collection. Use `chromadb.Client()` (in-memory, no persistence) rather than `PersistentClient` to avoid disk caching.

### Pattern 3: Calibration Weight Snapshotting

**What:** Serialize the full `calibration_weights` table state at each window boundary.
**When to use:** Before each evaluation window's re-prediction phase.

```python
# Source: Derived from weight_loader.py + weight_optimizer.py
async def snapshot_calibration_weights(
    session: AsyncSession,
    as_of: datetime,
) -> dict[str, float]:
    """Load calibration_weights and calibration_weight_history to reconstruct
    the weight state as it existed at `as_of` timestamp.

    Resolution strategy:
      1. Query calibration_weight_history for the latest entry per cameo_code
         where computed_at <= as_of AND auto_applied = True
      2. Fall back to cold-start priors for codes with no history before as_of

    Returns:
        Dict of cameo_code -> alpha, suitable for direct injection into
        a WeightLoader's _weights cache (bypassing DB query).
    """
```

**Critical subtlety:** The `calibration_weight_history` table records every calibration run with `computed_at` timestamps. This is the time-travel mechanism -- we reconstruct what weights *were* at any point by finding the most recent auto_applied entry before the window date. The `calibration_weights` table only holds current state, which is useless for historical snapshots.

### Pattern 4: Heavy Job with Cancellation

**What:** BacktestRunner runs in ProcessPoolExecutor with threading.Event for cancellation.
**When to use:** On-demand backtest triggering from admin panel.

```python
# Source: Derived from job_wrappers.py heavy_daily_pipeline pattern
# In heavy_runner.py:
_backtest_cancel_event: threading.Event | None = None

def run_backtest(config_json: str) -> int:
    """Execute backtest in subprocess worker.

    Uses threading.Event for cancellation check between windows.
    Persists partial results on cancel -- completed windows are kept.
    """
    global _backtest_cancel_event
    _backtest_cancel_event = threading.Event()

    config = BacktestRunConfig.from_json(config_json)
    runner = BacktestRunner(config, cancel_event=_backtest_cancel_event)

    asyncio.run(runner.run())
    return 0

def cancel_backtest() -> bool:
    """Signal cancellation to the running backtest."""
    if _backtest_cancel_event is not None:
        _backtest_cancel_event.set()
        return True
    return False
```

**Cancellation limitation:** Since the backtest runs in a ProcessPoolExecutor worker (separate process), the threading.Event must be within that process. The main process cannot directly signal it. Options:
1. **multiprocessing.Event** instead of threading.Event -- shared between processes via fork
2. **DB-based cancellation flag** -- the runner checks a `cancelled` column on the `backtest_runs` row between windows
3. **Process termination** -- `executor.shutdown(cancel_futures=True)` (Python 3.9+)

**Recommendation:** DB-based cancellation (option 2). The runner already has a DB session. Polling a `status` column between windows is trivial, process-safe, and survives restarts. Set `backtest_runs.status = 'cancelling'` from admin endpoint; runner checks at each window boundary and transitions to `'cancelled'`.

### Pattern 5: Admin Panel with d3 Charts

**What:** BacktestingPanel follows AccuracyPanel pattern with d3 SVG charts.
**When to use:** The drill-down view renders Brier curves, reliability diagrams, comparison tables.

Existing d3 patterns in the codebase:
- `ScenarioExplorer.ts`: `import * as d3 from 'd3'` -- uses d3.hierarchy, d3.tree for layout
- `expandable-card.ts`: `import * as d3 from 'd3'` -- uses d3.scaleLinear, d3.line for sparkline SVGs
- `CountryBriefPage.ts`: `import * as d3 from 'd3'` -- same pattern

For Phase 23 charts:
- **Brier score curves:** `d3.scaleLinear` + `d3.line` + `d3.axisBottom/Left` (time series)
- **Reliability diagram:** `d3.scaleLinear` for probability axes + `d3.line` for calibration curve + diagonal reference line
- **Comparison chart:** Grouped bar chart via `d3.scaleBand` + `d3.scaleLinear`
- **Confidence bands:** `d3.area` for shaded uncertainty regions

All chart types are well within d3 v7's capabilities. No additional d3 sub-modules needed.

### Anti-Patterns to Avoid

- **Importing d3 sub-modules individually:** The project imports `* as d3 from 'd3'`. Follow this pattern; Vite tree-shakes via the chunk config (`d3: ['d3']` in vite.config.ts).
- **Reusing EnsemblePredictor instances:** Memory note: "Fresh EnsemblePredictor per prediction -- mutable _forecast_output state prevents reuse." The `_build_predictor()` pattern MUST return a new instance per prediction.
- **Using calibration_weights table directly for historical snapshots:** This table holds CURRENT state only. Use `calibration_weight_history` for time-travel.
- **Caching ChromaDB temporal indexes on disk:** Locked decision: "No disk caching of temporal indexes." Use `chromadb.Client()` (ephemeral in-memory).
- **Scheduling backtests:** Locked decision: on-demand only. Do NOT register an APScheduler interval job.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Brier score computation | Custom formula | `numpy` mean squared error | One-liner: `np.mean((predictions - outcomes) ** 2)` |
| Reliability diagram bins | Manual bucketing | `numpy.histogram` with custom bin edges | Handles edge cases (empty bins, single-value bins) |
| SVG chart rendering | Manual DOM creation | `d3.scaleLinear`, `d3.line`, `d3.axisBottom` | Axes, ticks, label formatting all handled |
| Time-windowed queries | Raw SQL date arithmetic | SQLAlchemy `func.date_trunc` + between | Type-safe, timezone-aware |
| JSON serialization for export | Manual dict building | Pydantic `.model_dump(mode='json')` | Handles datetime, nested models, None values |
| Calibration weight reconstruction | Custom snapshot table | `calibration_weight_history` table | Already exists, has `computed_at` + `auto_applied` columns |
| Background job mutual exclusion | Custom locking | Existing `_heavy_job_lock` in job_wrappers.py | Already handles FIFO queue for heavy jobs |
| Admin auth | Custom middleware | Existing `verify_admin` Depends | Router-level X-Admin-Key validation |

## Common Pitfalls

### Pitfall 1: ChromaDB Metadata String Comparison vs Date Comparison

**What goes wrong:** ChromaDB stores `published_at` as a string (ISO 8601). String comparison of ISO dates works correctly for `$lte` ONLY if all dates use the same format (YYYY-MM-DDTHH:MM:SS). If some articles have date-only strings ("2026-03-01") and others have full timestamps ("2026-03-01T14:22:33Z"), string comparison breaks.

**Why it happens:** Different RSS feeds produce different date formats. The `article_processor.py` passes through `published_at` from feed metadata without normalization.

**How to avoid:** When building the temporal index, normalize all `published_at` values to a consistent format. Better yet, when querying source chunks for the temporal index, fetch ALL chunks and filter in Python using `datetime.fromisoformat()` rather than relying on ChromaDB's string-based `$lte`.

**Warning signs:** Temporal index includes articles from after the cutoff date.

### Pitfall 2: EnsemblePredictor Mutable State

**What goes wrong:** Reusing an EnsemblePredictor across predictions causes `_forecast_output` from prediction N to leak into prediction N+1's output.

**Why it happens:** `_get_llm_prediction()` stores `self._forecast_output` as instance state. The `predict()` method reads it later to build the final ForecastOutput.

**How to avoid:** Create a fresh `EnsemblePredictor()` for EVERY prediction call. The `_build_predictor()` pattern in `auto_forecaster.py` already does this correctly -- cache the heavy components (orchestrator, TKG, weight_loader), construct new EnsemblePredictor each time.

**Warning signs:** Forecast output contains reasoning/scenarios from a different question.

### Pitfall 3: Gemini API Budget Exhaustion

**What goes wrong:** A full backtest across 5-6 weeks with 14-day overlapping windows could require 30-50+ Gemini API calls. At 3 new + 5 reforecast daily cap, a single backtest run could consume the entire daily budget.

**Why it happens:** Live re-prediction is a locked decision. Each prediction in each window triggers a real Gemini call.

**How to avoid:**
1. The backtest runner MUST respect the existing Gemini budget tracking system.
2. Warn the operator in the admin UI about estimated API call count before starting.
3. Allow the run to be cancelled between windows with partial results preserved.
4. Consider a "dry-run" mode that computes metrics from existing predictions without re-running Gemini.

**Warning signs:** Daily Gemini budget exhausted after a backtest, blocking organic forecast generation.

### Pitfall 4: Look-Ahead Bias in calibration_weight_history

**What goes wrong:** If the `calibration_weight_history` table was populated retroactively (bulk-inserted with estimated timestamps), the "as of" query could return weights that encode future information.

**Why it happens:** The `weight_optimizer.py` writes history rows with `computed_at = datetime.now(tz=utc)` at the time of computation. If calibration was first deployed after predictions were already made, there may be no history rows for early windows.

**How to avoid:** When no `calibration_weight_history` entry exists before a window's end date, fall back to cold-start priors from `priors.py` (the same fallback chain that `WeightLoader` uses). This is the correct behavior -- before calibration was deployed, the system used cold-start priors, so the backtest should too.

**Warning signs:** Early windows show suspiciously good calibration metrics.

### Pitfall 5: ProcessPoolExecutor Pickling of asyncio Objects

**What goes wrong:** Passing an async_session_factory or other asyncio objects to a ProcessPoolExecutor worker fails because they're not pickleable.

**Why it happens:** ProcessPoolExecutor uses pickle to transfer arguments to worker processes. SQLAlchemy async sessions, aiohttp sessions, and asyncio locks are not pickleable.

**How to avoid:** Follow the `run_polymarket_cycle()` pattern in `heavy_runner.py`: the function creates its own DB engine and session factory inside the worker process via `_pg.init_db()`. All heavy arguments must be serializable (pass JSON config strings, not objects).

**Warning signs:** `PicklingError` at job submission time.

### Pitfall 6: RE-GCN vs TiRGN Checkpoint Format Mismatch

**What goes wrong:** RE-GCN checkpoints use `.npz` + `.json` with `regcn_jraph_*` naming. TiRGN checkpoints use `.npz` + `.json` with `tirgn_*` naming. The JSON metadata has different fields.

**Why it happens:** The two model backends were implemented in different phases with different serialization conventions.

**How to avoid:** Checkpoint discovery must understand BOTH naming patterns:
- TiRGN: `tirgn_*.json` files with `"model_type": "tirgn"` in metadata
- RE-GCN: `regcn_jraph_*.json` files (no `model_type` field, or implicitly regcn)
- Both store metrics: `mrr`, `hits_at_1`, `hits_at_3`, `hits_at_10`
- Both have `entity_to_id` and either `relation_to_id` or embedded in config

The backtest runner must set `TKG_BACKEND` (or equivalent) when loading each checkpoint.

## Code Examples

### Database Schema: backtest_runs and backtest_results

```python
# Source: Derived from db/models.py ORM patterns
class BacktestRun(Base):
    """A single backtesting evaluation run."""
    __tablename__ = "backtest_runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_uuid)
    label: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Run configuration (frozen at start)
    window_size_days: Mapped[int] = mapped_column(Integer, nullable=False, default=14)
    slide_step_days: Mapped[int] = mapped_column(Integer, nullable=False, default=7)
    min_predictions_per_window: Mapped[int] = mapped_column(Integer, nullable=False, default=3)
    checkpoints_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    # e.g. {"tirgn": "tirgn_best.npz", "regcn": "regcn_jraph_best.npz"}

    # Run lifecycle
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, default="pending"
    )  # pending | running | completed | cancelled | failed
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    # Progress tracking
    total_windows: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    completed_windows: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    total_predictions: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    # Aggregate summary (computed on completion)
    aggregate_brier: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    aggregate_mrr: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    vs_polymarket_record_json: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    # e.g. {"geopol_wins": 5, "polymarket_wins": 3, "draws": 1}

    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class BacktestResult(Base):
    """Per-window evaluation metrics for a backtest run."""
    __tablename__ = "backtest_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("backtest_runs.id"), nullable=False, index=True
    )

    # Window definition
    window_start: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    window_end: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    prediction_start: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    prediction_end: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    # Model identification
    checkpoint_name: Mapped[str] = mapped_column(String(100), nullable=False)
    # "tirgn_best" or "regcn_jraph_best" etc.

    # Metrics
    num_predictions: Mapped[int] = mapped_column(Integer, nullable=False)
    brier_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    mrr: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    hits_at_1: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    hits_at_10: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Calibration data (for reliability diagrams)
    calibration_bins_json: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    # {"bins": [0.0, 0.1, ..., 1.0], "predicted_avg": [...], "observed_freq": [...], "counts": [...]}

    # Per-prediction details (for drill-down)
    prediction_details_json: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    # [{"prediction_id": "...", "question": "...", "predicted_prob": 0.7, "outcome": 1.0, "brier": 0.09}, ...]

    # Polymarket comparison (for Geopol vs PM head-to-head)
    polymarket_brier: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    geopol_vs_pm_wins: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    pm_vs_geopol_wins: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # Calibration weight state used for this window
    weight_snapshot_json: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    # {"global": 0.58, "super:verbal_conflict": 0.62, "14": 0.71, ...}

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    __table_args__ = (
        Index("ix_backtest_results_run_id_window", "run_id", "window_start"),
    )
```

### AdminClient Extension

```typescript
// Source: Derived from admin-client.ts pattern

// In admin-client.ts -- add to AdminClient class:

/** GET /backtesting/runs -- list all backtest runs. */
async getBacktestRuns(): Promise<BacktestRun[]> {
  return this.request<BacktestRun[]>('/backtesting/runs');
}

/** POST /backtesting/runs -- start a new backtest run. */
async startBacktestRun(config: StartBacktestRequest): Promise<BacktestRun> {
  return this.request<BacktestRun>('/backtesting/runs', {
    method: 'POST',
    body: JSON.stringify(config),
  });
}

/** GET /backtesting/runs/{run_id} -- get run details with all results. */
async getBacktestRun(runId: string): Promise<BacktestRunDetail> {
  return this.request<BacktestRunDetail>(`/backtesting/runs/${runId}`);
}

/** POST /backtesting/runs/{run_id}/cancel -- cancel a running backtest. */
async cancelBacktestRun(runId: string): Promise<{ status: string }> {
  return this.request<{ status: string }>(
    `/backtesting/runs/${runId}/cancel`,
    { method: 'POST' },
  );
}

/** GET /backtesting/runs/{run_id}/export?format=csv|json -- export results. */
async exportBacktestRun(runId: string, format: 'csv' | 'json'): Promise<Blob> {
  const url = `${API_BASE}/backtesting/runs/${runId}/export?format=${format}`;
  const response = await fetch(url, {
    headers: { 'X-Admin-Key': this.key },
  });
  if (!response.ok) throw new AdminApiError(response.status, url);
  return response.blob();
}

/** GET /backtesting/checkpoints -- list available model checkpoints. */
async getCheckpoints(): Promise<CheckpointInfo[]> {
  return this.request<CheckpointInfo[]>('/backtesting/checkpoints');
}
```

### d3 Brier Score Curve Chart

```typescript
// Source: Derived from expandable-card.ts d3 sparkline pattern + d3 v7 docs
function renderBrierCurve(
  container: HTMLElement,
  results: BacktestWindowResult[],
  width: number = 600,
  height: number = 300,
): void {
  const margin = { top: 20, right: 80, bottom: 40, left: 50 };
  const w = width - margin.left - margin.right;
  const h = height - margin.top - margin.bottom;

  const svg = d3.select(container)
    .append('svg')
    .attr('viewBox', `0 0 ${width} ${height}`)
    .attr('class', 'brier-curve-svg');

  const g = svg.append('g')
    .attr('transform', `translate(${margin.left},${margin.top})`);

  // Scales
  const xScale = d3.scaleTime()
    .domain(d3.extent(results, d => new Date(d.window_end)) as [Date, Date])
    .range([0, w]);

  const yScale = d3.scaleLinear()
    .domain([0, d3.max(results, d => d.brier_score) ?? 0.5])
    .nice()
    .range([h, 0]);

  // Axes
  g.append('g').attr('transform', `translate(0,${h})`).call(d3.axisBottom(xScale));
  g.append('g').call(d3.axisLeft(yScale));

  // Perfect calibration reference line at Brier = 0.25 (random baseline)
  g.append('line')
    .attr('x1', 0).attr('x2', w)
    .attr('y1', yScale(0.25)).attr('y2', yScale(0.25))
    .attr('stroke', 'var(--text-secondary)')
    .attr('stroke-dasharray', '4,4')
    .attr('opacity', 0.5);

  // Brier curve line
  const line = d3.line<BacktestWindowResult>()
    .x(d => xScale(new Date(d.window_end)))
    .y(d => yScale(d.brier_score));

  g.append('path')
    .datum(results)
    .attr('fill', 'none')
    .attr('stroke', 'var(--accent)')
    .attr('stroke-width', 2)
    .attr('d', line);

  // Data points
  g.selectAll('.dot')
    .data(results)
    .join('circle')
    .attr('cx', d => xScale(new Date(d.window_end)))
    .attr('cy', d => yScale(d.brier_score))
    .attr('r', 4)
    .attr('fill', 'var(--accent)');
}
```

### Temporal ChromaDB Index Builder

```python
# Source: Derived from article_processor.py ArticleIndex class
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from datetime import datetime

EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
BATCH_SIZE = 500  # ChromaDB add() batch limit

def build_temporal_chromadb_index(
    source_persist_dir: str,
    source_collection_name: str,
    cutoff_date: datetime,
) -> chromadb.Collection:
    """Build an ephemeral in-memory ChromaDB collection with articles <= cutoff_date.

    Args:
        source_persist_dir: Path to the persistent ChromaDB storage.
        source_collection_name: Name of the source collection (e.g. "rss_articles").
        cutoff_date: Only include articles published on or before this date.

    Returns:
        An in-memory ChromaDB Collection ready for RAG queries.
    """
    cutoff_iso = cutoff_date.isoformat()

    # Open source (persistent, read-only)
    source_client = chromadb.PersistentClient(path=source_persist_dir)
    ef = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
    source_col = source_client.get_collection(
        name=source_collection_name,
        embedding_function=ef,
    )

    # Create ephemeral target (in-memory, no disk persistence)
    temp_client = chromadb.Client()
    temp_col = temp_client.create_collection(
        name=f"temporal_{cutoff_date.strftime('%Y%m%d')}",
        embedding_function=ef,
    )

    # Paginated fetch with date filtering
    # ChromaDB get() returns all matching docs; use offset/limit for large collections
    offset = 0
    total_copied = 0
    while True:
        batch = source_col.get(
            where={"published_at": {"$lte": cutoff_iso}},
            limit=BATCH_SIZE,
            offset=offset,
            include=["documents", "metadatas", "embeddings"],
        )
        if not batch["ids"]:
            break

        # Filter out empty published_at (articles without dates -- exclude to be safe)
        valid_indices = [
            i for i, meta in enumerate(batch["metadatas"])
            if meta.get("published_at", "") != ""
        ]

        if valid_indices:
            temp_col.add(
                ids=[batch["ids"][i] for i in valid_indices],
                documents=[batch["documents"][i] for i in valid_indices] if batch["documents"] else None,
                metadatas=[batch["metadatas"][i] for i in valid_indices],
                embeddings=[batch["embeddings"][i] for i in valid_indices] if batch["embeddings"] else None,
            )
            total_copied += len(valid_indices)

        offset += BATCH_SIZE

    return temp_col
```

### Heavy Job Runner Pattern

```python
# Source: Derived from heavy_runner.py run_polymarket_cycle pattern
def run_backtest(config_json: str) -> int:
    """Execute a backtest run in a ProcessPoolExecutor worker.

    Args:
        config_json: JSON-serialized BacktestRunConfig. Passed as string
            because ProcessPoolExecutor pickles arguments.

    Returns:
        0 on success (or clean cancellation), 1 on failure.
    """
    import asyncio as _asyncio
    import json as _json
    import logging as _logging

    _logger = _logging.getLogger(__name__)

    async def _run() -> None:
        from src.backtesting.runner import BacktestRunner, BacktestRunConfig
        from src.db import postgres as _pg

        if _pg.async_session_factory is None:
            _pg.init_db()

        config = BacktestRunConfig.from_json(config_json)
        runner = BacktestRunner(
            run_config=config,
            async_session_factory=_pg.async_session_factory,
        )
        await runner.run()

    _logger.info("Starting backtest run (in-process)")
    try:
        _asyncio.run(_run())
        _logger.info("Backtest run completed successfully")
        return 0
    except Exception:
        _logger.exception("Backtest run failed")
        return 1
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Fixed alpha=0.6 | Per-CAMEO dynamic weights via L-BFGS-B | Phase 13 | calibration_weight_history exists for time-travel |
| RE-GCN only | TiRGN default (3.47x MRR improvement) | Phase 11 | Two checkpoint families to compare |
| systemd timers | APScheduler in-process | Phase 20 | Backtest job uses same scheduler infrastructure |
| Mock fixtures | Real predictions in PostgreSQL | Phase 14 | Actual prediction data to backtest against |
| No outcome tracking | outcome_records table | Phase 9 | Ground truth data for Brier score computation |
| No Polymarket | Polymarket comparisons + accuracy | Phase 22 | Head-to-head Brier data for the killer metric |

## Open Questions

1. **Gemini budget isolation for backtests**
   - What we know: Backtesting burns real Gemini API calls. The existing budget system tracks daily usage.
   - What's unclear: Should backtest calls count against the daily budget cap? Or should they have their own separate budget?
   - Recommendation: Count against the same budget but allow the operator to see estimated call count before starting. Add a warning in the UI: "This run will require ~N Gemini calls."

2. **ChromaDB in-memory collection size limits**
   - What we know: `chromadb.Client()` creates an in-memory store. The rss_articles collection grows over time.
   - What's unclear: For a 14-day window with ~90 days of articles, how much memory does an in-memory ChromaDB collection require?
   - Recommendation: Monitor memory usage during first runs. If excessive, consider filtering to only articles from the prediction window's country/region rather than all articles.

3. **MRR computation without TKG retraining**
   - What we know: Locked decision says "static model weights for v3.0." MRR measures TKG's link prediction ranking quality.
   - What's unclear: MRR computed on static weights across different time windows measures consistency, not improvement trajectory. Is this meaningful for investors?
   - Recommendation: Include MRR as a secondary metric. Brier score is the primary metric because it captures the full ensemble output (LLM + TKG + calibration). MRR is TKG-specific and less meaningful without per-window retraining.

4. **Empty window handling**
   - What we know: Minimum 3 predictions per window for inclusion.
   - What's unclear: What happens for windows where fewer than 3 predictions have resolved outcomes? This is likely for early system history.
   - Recommendation: Skip such windows in the evaluation curve but log them. The UI should show a gap in the curve (not interpolate) with a tooltip explaining "insufficient data."

## Sources

### Primary (HIGH confidence)
- `src/forecasting/ensemble_predictor.py` -- EnsemblePredictor API, mutable state pattern, _forecast_output
- `src/polymarket/auto_forecaster.py` -- `_build_predictor()` pattern, component caching, fresh predictor per call
- `src/calibration/weight_loader.py` -- WeightLoader hierarchical resolution, cache, async session
- `src/calibration/weight_optimizer.py` -- `CalibrationResult`, `calibration_weight_history` persistence
- `src/db/models.py` -- All ORM models, CalibrationWeight, CalibrationWeightHistory, Prediction, OutcomeRecord
- `src/ingest/article_processor.py` -- ArticleIndex class, `rss_articles` collection, `published_at` metadata
- `src/forecasting/rag_pipeline.py` -- RAGPipeline ChromaDB setup, `graph_patterns` collection
- `src/forecasting/tkg_predictor.py` -- TKGPredictor, checkpoint loading, tirgn/regcn backend selection
- `src/scheduler/job_wrappers.py` -- `_heavy_job_lock`, ProcessPoolExecutor pattern
- `src/scheduler/heavy_runner.py` -- Module-level functions, subprocess/in-process patterns
- `src/scheduler/core.py` -- AsyncIOScheduler factory, dual executors
- `src/api/routes/v1/admin.py` -- Admin router pattern, verify_admin dependency
- `src/api/services/admin_service.py` -- AdminService pattern, daemon type mapping
- `frontend/src/admin/panels/AccuracyPanel.ts` -- Admin panel pattern, stat cards, sortable table
- `frontend/src/admin/admin-client.ts` -- AdminClient typed API access pattern
- `frontend/src/admin/admin-types.ts` -- TypeScript interfaces, AdminSection type
- `frontend/src/admin/admin-layout.ts` -- Panel mounting/destruction lifecycle
- `frontend/src/admin/components/AdminSidebar.ts` -- NAV_ITEMS, section registration
- `models/tkg/` -- Checkpoint file layout: `tirgn_best.{npz,json}`, `regcn_jraph_best.{npz,json}`
- `alembic/versions/` -- Migration naming pattern: `YYYYMMDD_NNN_description.py`

### Secondary (MEDIUM confidence)
- [ChromaDB Metadata Filtering](https://docs.trychroma.com/docs/querying-collections/metadata-filtering) -- `$lte` operator for string metadata, where clause syntax
- [ChromaDB Time-based Queries](https://cookbook.chromadb.dev/strategies/time-based-queries/) -- Strategies for temporal filtering

### Tertiary (LOW confidence)
- ChromaDB in-memory collection memory usage -- No benchmarks found for collection sizes typical in this project. Need empirical testing.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- All libraries already in project, verified against package.json and pyproject.toml
- Architecture patterns: HIGH -- All patterns derived from reading actual source code
- Database schema: HIGH -- Schema design follows existing ORM model patterns exactly
- d3 charting: HIGH -- d3 v7 already imported in 3 components, verified chart types feasible
- Bias prevention: HIGH -- calibration_weight_history table exists with computed_at timestamps; ChromaDB metadata includes published_at
- Admin panel integration: HIGH -- AdminPanel interface, AdminClient, AdminSidebar all well-documented in source
- Cancellation mechanism: MEDIUM -- DB-based cancellation recommended but not battle-tested in this codebase
- ChromaDB temporal index memory: LOW -- No empirical data on in-memory collection sizes

**Research date:** 2026-03-07
**Valid until:** 2026-04-07 (stable -- all findings based on current codebase, not external APIs)
