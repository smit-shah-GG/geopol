# Phase 7: Bootstrap Pipeline - Research

**Researched:** 2026-01-30
**Domain:** Pipeline orchestration, idempotency, checkpoint management
**Confidence:** HIGH

## Summary

The bootstrap pipeline must orchestrate three existing subsystems: GDELT event ingestion, knowledge graph construction, and RAG index building. Research of the existing codebase reveals well-structured but disconnected components that need unified orchestration with idempotent stage execution.

The existing codebase already has mature implementations for each stage:
- GDELT ingestion via `GDELTHistoricalCollector` and `GDELTDataProcessor` with daily CSV outputs
- Knowledge graph via `TemporalKnowledgeGraph.add_events_batch()` with SQLite streaming
- RAG indexing via `RAGPipeline.index_graph_patterns()` with ChromaDB persistence

The recommended approach is a simple JSON state file tracking stage completion, combined with file existence checks for idempotent stage detection. This is preferred over more complex orchestration frameworks (Dagster, Prefect) given the project's scope and existing tooling.

**Primary recommendation:** Implement a lightweight JSON checkpoint file (`data/bootstrap_state.json`) that tracks stage completion timestamps and output paths, using file existence as the primary idempotency mechanism.

## Standard Stack

The established libraries/tools for this domain:

### Core (Already in Project)
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| Python stdlib `json` | builtin | Checkpoint state persistence | Zero dependencies, human-readable |
| Python stdlib `pathlib` | builtin | File existence checks | Idiomatic path handling |
| Python stdlib `logging` | builtin | Progress reporting | Already used throughout codebase |
| Python stdlib `argparse` | builtin | CLI interface | Already used in `run_pipeline.py` |

### Supporting (Already in Project)
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `pandas` | >=2.0.0 | Data loading/validation | Checking processed data existence |
| `chromadb` | >=1.4.0 | RAG store validation | Checking index population |
| `networkx` | >=3.0 | Graph validation | Checking graph persistence |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| JSON state file | SQLite state table | SQLite more robust for concurrent access, but JSON simpler for single-process bootstrap |
| File existence | Database flags | Database flags better for distributed systems, file checks simpler for local execution |
| Custom orchestrator | Dagster/Prefect | Full orchestrators overkill for single-script linear pipeline |

**Installation:**
No additional dependencies needed - all required libraries already in `pyproject.toml`.

## Architecture Patterns

### Recommended Project Structure
```
scripts/
  bootstrap.py           # Main bootstrap entry point

src/
  bootstrap/
    __init__.py
    orchestrator.py      # StageOrchestrator class
    stages.py            # Stage definitions (GDELT, Graph, RAG)
    checkpoint.py        # CheckpointManager class
    validation.py        # Stage output validators

data/
  bootstrap_state.json   # Checkpoint state file
  gdelt/
    raw/                 # Daily CSV files (already exists)
    processed/
      events.parquet     # Processed events (already exists)
  graphs/
    knowledge_graph.graphml  # Persisted graph
  chroma_db/             # RAG vector store (already exists)
```

### Pattern 1: Stage State Machine
**What:** Each stage has discrete states: PENDING, RUNNING, COMPLETED, FAILED
**When to use:** Tracking multi-stage pipeline execution with checkpoints
**Example:**
```python
# Source: Codebase pattern from src/pipeline.py
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import json

class StageStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class StageState:
    name: str
    status: StageStatus = StageStatus.PENDING
    started_at: str | None = None
    completed_at: str | None = None
    output_path: str | None = None
    error: str | None = None

@dataclass
class BootstrapState:
    stages: dict[str, StageState] = field(default_factory=dict)
    last_updated: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def save(self, path: Path) -> None:
        data = {
            "stages": {
                name: {
                    "status": stage.status.value,
                    "started_at": stage.started_at,
                    "completed_at": stage.completed_at,
                    "output_path": stage.output_path,
                    "error": stage.error
                }
                for name, stage in self.stages.items()
            },
            "last_updated": datetime.utcnow().isoformat()
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "BootstrapState":
        if not path.exists():
            return cls()
        with open(path) as f:
            data = json.load(f)
        state = cls()
        for name, stage_data in data.get("stages", {}).items():
            state.stages[name] = StageState(
                name=name,
                status=StageStatus(stage_data["status"]),
                started_at=stage_data.get("started_at"),
                completed_at=stage_data.get("completed_at"),
                output_path=stage_data.get("output_path"),
                error=stage_data.get("error")
            )
        return state
```

### Pattern 2: Dual Idempotency Check
**What:** Check both checkpoint state AND output file existence for robust skip detection
**When to use:** Resuming from interruption where state file may be stale
**Example:**
```python
# Source: Best practice from Prefect idempotency patterns
def should_skip_stage(stage_name: str, state: BootstrapState, output_path: Path) -> bool:
    """
    Dual check for idempotency:
    1. State file says completed
    2. Output file actually exists

    Both must be true to skip, preventing false positives.
    """
    stage = state.stages.get(stage_name)
    if stage is None:
        return False

    if stage.status != StageStatus.COMPLETED:
        return False

    # Critical: verify output actually exists
    if not output_path.exists():
        logger.warning(f"Stage {stage_name} marked complete but output missing")
        return False

    return True
```

### Pattern 3: Progress Reporter Protocol
**What:** Unified interface for stage progress reporting
**When to use:** Consistent stdout progress across all stages
**Example:**
```python
# Source: Pattern from src/monitoring.py DataQualityMonitor
import sys
from typing import Protocol

class ProgressReporter(Protocol):
    def stage_start(self, name: str) -> None: ...
    def stage_progress(self, name: str, message: str) -> None: ...
    def stage_complete(self, name: str, duration_sec: float) -> None: ...
    def stage_error(self, name: str, error: str) -> None: ...

class ConsoleReporter:
    def stage_start(self, name: str) -> None:
        print(f"[STAGE] {name}: Starting...", flush=True)

    def stage_progress(self, name: str, message: str) -> None:
        print(f"[STAGE] {name}: {message}", flush=True)

    def stage_complete(self, name: str, duration_sec: float) -> None:
        print(f"[STAGE] {name}: Completed in {duration_sec:.1f}s", flush=True)

    def stage_error(self, name: str, error: str) -> None:
        print(f"[STAGE] {name}: FAILED - {error}", file=sys.stderr, flush=True)
```

### Anti-Patterns to Avoid
- **Global mutable state:** Don't use module-level variables for checkpoint tracking; pass state explicitly
- **Swallowing exceptions:** Don't catch-all; let validation errors propagate with context
- **Implicit dependencies:** Don't assume previous stage ran; always validate inputs exist
- **Relying solely on checkpoint file:** Always verify output artifacts exist

## Don't Hand-Roll

Problems that look simple but have existing solutions:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| GDELT ingestion | Custom HTTP fetcher | `GDELTHistoricalCollector` (existing) | Already handles rate limiting, retries, daily saves |
| Event processing | Custom CSV parser | `GDELTDataProcessor.process_all()` (existing) | Already handles entity extraction, relation mapping |
| Graph building | Custom edge insertion | `TemporalKnowledgeGraph.add_events_batch()` (existing) | Handles batching, normalization, statistics |
| Graph persistence | Custom file format | `GraphPersistence.save()` (existing) | GraphML format with round-trip validation |
| RAG indexing | Custom embedding loop | `RAGPipeline.index_graph_patterns()` (existing) | ChromaDB integration, pattern extraction |
| Progress logging | Custom print statements | Python `logging` module | Configurable handlers, structured output |

**Key insight:** The codebase already has all pipeline components implemented. The bootstrap script is purely an orchestration layer that chains existing functions with state management.

## Common Pitfalls

### Pitfall 1: Incomplete Idempotency
**What goes wrong:** Stage marked complete in checkpoint but output deleted; re-run assumes success
**Why it happens:** Relying solely on checkpoint state without verifying artifacts
**How to avoid:** Always check output file/table existence before skipping stage
**Warning signs:** Bootstrap completes but downstream stages fail on missing inputs

### Pitfall 2: Interrupted State File Write
**What goes wrong:** Script crashes mid-write leaving corrupt JSON; next run fails to parse
**Why it happens:** Non-atomic file writes
**How to avoid:** Write to temp file, then atomic rename (`os.replace()`)
**Warning signs:** JSONDecodeError on bootstrap restart

### Pitfall 3: Stale Graph File Without Events
**What goes wrong:** Empty graph file from previous failed run passes existence check
**Why it happens:** Graph file created but no events added before crash
**How to avoid:** Validate file contents, not just existence (check node/edge count)
**Warning signs:** RAG indexing fails with empty pattern list

### Pitfall 4: Database Lock on Resume
**What goes wrong:** SQLite database locked from previous crashed process
**Why it happens:** Previous run didn't cleanly close connection
**How to avoid:** Use context managers; add explicit connection cleanup in error handlers
**Warning signs:** `sqlite3.OperationalError: database is locked`

### Pitfall 5: Partial RAG Index
**What goes wrong:** ChromaDB collection exists but is incomplete
**Why it happens:** Indexing interrupted mid-batch
**How to avoid:** Check document count in collection matches expected; use `rebuild=True` flag
**Warning signs:** Queries return fewer results than expected

## Code Examples

Verified patterns from existing codebase:

### Existing GDELT Collection Entry Point
```python
# Source: scripts/collect_training_data.py
from src.training.data_collector import GDELTHistoricalCollector

collector = GDELTHistoricalCollector(
    base_delay=2.0,  # Rate limiting
    max_retries=3,
)
df = collector.collect_last_n_days(
    n_days=30,
    quad_classes=[1, 2, 3, 4],  # All event types
)
# Output: data/gdelt/raw/gdelt_YYYY-MM-DD.csv files
```

### Existing Data Processing Entry Point
```python
# Source: src/training/data_processor.py
from src.training.data_processor import GDELTDataProcessor

processor = GDELTDataProcessor()
df = processor.process_all()
# Output: data/gdelt/processed/events.parquet
```

### Existing Graph Building Entry Point
```python
# Source: src/knowledge_graph/graph_builder.py
from src.knowledge_graph.graph_builder import create_graph

graph, stats = create_graph(
    db_path="data/events.db",
    batch_size=1000,
    limit=None  # Process all events
)
# Note: Currently reads from SQLite, may need adapter for parquet
```

### Existing Graph Persistence Entry Point
```python
# Source: src/knowledge_graph/persistence.py
from src.knowledge_graph.persistence import GraphPersistence

persistence = GraphPersistence(graph.graph)
stats = persistence.save("data/graphs/knowledge_graph.graphml", format='graphml')
```

### Existing RAG Indexing Entry Point
```python
# Source: src/forecasting/rag_pipeline.py
from src.forecasting.rag_pipeline import RAGPipeline

rag = RAGPipeline(
    persist_dir="./chroma_db",
    collection_name="graph_patterns",
)
stats = rag.index_graph_patterns(
    graph=temporal_graph,
    time_window_days=30,
    min_pattern_size=3,
    rebuild=True  # Critical for idempotency
)
```

### Atomic State File Write
```python
# Source: Standard Python pattern
import os
import tempfile

def save_state_atomic(state: BootstrapState, path: Path) -> None:
    """Atomic write to prevent corruption on interrupt."""
    path.parent.mkdir(parents=True, exist_ok=True)

    # Write to temp file in same directory (for atomic rename)
    fd, temp_path = tempfile.mkstemp(
        dir=path.parent,
        prefix='.bootstrap_state_',
        suffix='.tmp'
    )
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(state.to_dict(), f, indent=2)
        os.replace(temp_path, path)  # Atomic on POSIX
    except:
        os.unlink(temp_path)  # Clean up temp file
        raise
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| DAG-based orchestrators (Airflow) | Asset-based orchestrators (Dagster) | 2023 | Simpler mental model for data pipelines |
| Manual retry logic | Declarative idempotency keys | 2024 | Automatic deduplication |
| File locks for concurrency | Single-process with atomic writes | N/A | Simpler for local pipelines |

**Deprecated/outdated:**
- Complex orchestration frameworks for single-process pipelines: Overkill for a script that runs sequentially
- Pickle-based checkpointing: JSON preferred for debuggability and human inspection

## Open Questions

Things that couldn't be fully resolved:

1. **Graph builder SQLite vs Parquet source**
   - What we know: `TemporalKnowledgeGraph.add_events_batch()` reads from SQLite
   - What's unclear: Should bootstrap use SQLite (existing `EventStorage`) or read directly from parquet?
   - Recommendation: Add parquet adapter method to graph builder, or ensure events are in SQLite before graph stage

2. **RAG index validation threshold**
   - What we know: `RAGPipeline.get_index_statistics()` returns document count
   - What's unclear: What's the minimum valid document count? Depends on graph size
   - Recommendation: Validate document count > 0; full validation may require domain knowledge

3. **Concurrent bootstrap protection**
   - What we know: Atomic writes protect state file integrity
   - What's unclear: Should bootstrap prevent concurrent runs via PID file?
   - Recommendation: Optional PID file lock for safety, but not critical for typical usage

## Sources

### Primary (HIGH confidence)
- Codebase analysis: `src/pipeline.py`, `src/training/data_collector.py`, `src/training/data_processor.py`
- Codebase analysis: `src/knowledge_graph/graph_builder.py`, `src/knowledge_graph/persistence.py`
- Codebase analysis: `src/forecasting/rag_pipeline.py`
- Codebase analysis: `src/monitoring.py`, `src/training/progress_monitor.py`

### Secondary (MEDIUM confidence)
- [Prefect Blog: Idempotent Data Pipelines](https://www.prefect.io/blog/the-importance-of-idempotent-data-pipelines-for-resilience) - Idempotency patterns
- [Dagster: Checkpointing](https://dagster.io/glossary/checkpointing) - Checkpoint concepts
- [Dagster: Pipeline Orchestration Tools 2026](https://dagster.io/learn/data-pipeline-orchestration-tools) - Industry landscape

### Tertiary (LOW confidence)
- [python-checkpointing GitHub](https://github.com/a-rahimi/python-checkpointing) - Generator-based checkpointing approach (not recommended for this use case)

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - All libraries already in project and working
- Architecture: HIGH - Patterns derived from existing codebase conventions
- Pitfalls: HIGH - Based on analysis of existing code paths and failure modes

**Research date:** 2026-01-30
**Valid until:** 60 days (stable Python patterns, existing codebase)

## Entry Points Summary

For quick reference, here are the existing functions to chain:

| Stage | Module | Function/Class | Input | Output |
|-------|--------|----------------|-------|--------|
| 1. GDELT Collect | `src.training.data_collector` | `GDELTHistoricalCollector.collect_last_n_days()` | None | `data/gdelt/raw/*.csv` |
| 2. Process Events | `src.training.data_processor` | `GDELTDataProcessor.process_all()` | CSV files | `data/gdelt/processed/events.parquet` |
| 3. Build Graph | `src.knowledge_graph.graph_builder` | `TemporalKnowledgeGraph.add_events_batch()` | SQLite/Parquet | In-memory graph |
| 4. Persist Graph | `src.knowledge_graph.persistence` | `GraphPersistence.save()` | Graph object | `data/graphs/*.graphml` |
| 5. Index RAG | `src.forecasting.rag_pipeline` | `RAGPipeline.index_graph_patterns()` | Graph object | `chroma_db/` |

**Note:** Stage 3 currently expects SQLite. May need adapter or ensure events loaded into `EventStorage` first.
