# Tech Debt Audit Report

**Audited:** 2026-01-27
**Auditor:** Claude session (manual code verification)
**Status:** All items verified against source code

---

## Executive Summary

v1.0 MVP shipped 2026-01-23. The system is functional end-to-end but has 4 verified issues requiring attention before production use. The most impactful is the RAG pipeline never being indexed, which disables historical context grounding entirely.

---

## Project State

- **Milestone:** v1.0 complete, v1.1 not started
- **Phases:** 5 phases, 16 plans executed
- **Stats:** 89 files, 37,414 LOC Python
- **Main artifacts:**
  - `forecast.py` - CLI entry point
  - `src/forecasting/` - Ensemble predictor, TKG predictor, LLM integration
  - `src/knowledge_graph/` - Graph construction and querying
  - `src/training/` - Data collection, model training
  - `models/tkg/regcn_jraph_best.npz` - Valid trained checkpoint (15MB, 13 arrays)

---

## Verified Tech Debt

### 1. UAT-005: NetworkX shortest_path API Bug

**Status:** CONFIRMED BUG
**Severity:** Minor (not on critical path)
**Location:** `src/knowledge_graph/temporal_index.py:233`

**The Bug:**
```python
# Line 233 - BROKEN
path = nx.shortest_path(self.graph, source, target, cutoff=max_length)
```

`nx.shortest_path()` does not have a `cutoff` parameter. This will raise:
```
TypeError: shortest_path() got an unexpected keyword argument 'cutoff'
```

**Verification:**
```bash
.venv/bin/python -c "import networkx as nx; help(nx.shortest_path)" | head -20
# Shows: shortest_path(G, source=None, target=None, weight=None, method='dijkstra')
# No 'cutoff' parameter
```

**Fix:** Use `nx.single_source_shortest_path(G, source, cutoff=k)` then filter for target.

**Impact:** The `shortest_path()` method in `TemporalIndex` class is unusable. Currently not called in any critical path.

**Reference:** `.planning/phases/02-knowledge-graph-engine/02-01-ISSUES.md:10-22`

---

### 2. RAG Index Never Populated

**Status:** CONFIRMED BUG
**Severity:** High (feature disabled)
**Location:** `src/forecasting/forecast_engine.py:84-87, 208`

**The Bug:**

`ForecastEngine.__init__()` creates a RAGPipeline but never indexes it:
```python
# Line 84-85
if enable_rag:
    self.rag_pipeline = rag_pipeline or RAGPipeline()  # Creates empty pipeline
```

Later, `_retrieve_historical_context()` checks for index and returns empty:
```python
# Line 208
if not self.rag_pipeline or not self.rag_pipeline.index:
    return context  # Always returns empty - index is never set!
```

**Verification:**
```bash
# Search for any production call to index_graph_patterns
grep -r "index_graph_patterns" --include="*.py" .
# Results: Only in rag_pipeline.py (definition) and tests
# NO production code calls this method
```

**Evidence from user test (2026-01-24):**
```
Retrieved 0 context items  # RAG retrieval returned nothing
```

**Fix Required:**
Someone must call `rag_pipeline.index_graph_patterns(graph)` with a populated `TemporalKnowledgeGraph` before forecasting. This requires the bootstrap script (see #3).

**The method exists at:** `src/forecasting/rag_pipeline.py:106`

---

### 3. No Bootstrap Script (Phase 1 → 2 → 3)

**Status:** CONFIRMED GAP
**Severity:** Medium (manual workaround exists)

**The Problem:**

Three phases must connect in production:
1. **Phase 1:** `run_pipeline.py` → GDELT events → SQLite (`data/events.db`)
2. **Phase 2:** SQLite → `TemporalKnowledgeGraph` (via `add_events_batch()`)
3. **Phase 3:** Graph → RAG index (via `index_graph_patterns()`)

Only Phase 1 has a runnable script. Phases 2-3 require manual Python:

```python
# This code exists but is NOT in any script
from src.knowledge_graph.graph_builder import TemporalKnowledgeGraph
from src.forecasting.rag_pipeline import RAGPipeline

# Phase 1 → Phase 2
graph = TemporalKnowledgeGraph()
graph.add_events_batch("data/events.db")  # Method at graph_builder.py:160

# Phase 2 → Phase 3
rag = RAGPipeline()
rag.index_graph_patterns(graph)  # Method at rag_pipeline.py:106
```

**Verification:**
```bash
# Check what scripts exist
ls *.py scripts/*.py
# run_pipeline.py - Phase 1 only
# forecast.py - Assumes everything is already set up
# scripts/*.py - Training scripts, no bootstrap

# Check if add_events_batch is called anywhere in runnable code
grep -r "add_events_batch" --include="*.py" . | grep -v test | grep -v ".pyc"
# Only in graph_builder.py (definition) and a utility function
```

**Impact:** Users must manually run Python code to populate the graph and RAG index before `forecast.py` will return meaningful historical context.

**Reference:** `.planning/milestones/v1.0-MILESTONE-AUDIT.md:85-101`

---

### 4. Broken Checkpoint Files

**Status:** CONFIRMED (cleanup needed)
**Severity:** Low (artifacts from pre-fix training)

**The Problem:**

Previous training runs saved empty checkpoints due to a bug in `save_checkpoint()` (fixed 2026-01-24 in commit `1510b95`).

**Current state of `models/tkg/`:**
```
regcn_jraph_best.npz    - 15MB, 13 arrays  ✓ VALID (post-fix)
regcn_jraph_epoch_1.npz - 22 bytes, 0 arrays  ✗ BROKEN
regcn_jraph_epoch_2.npz - 22 bytes, 0 arrays  ✗ BROKEN
regcn_jraph_epoch_3.npz - 22 bytes, 0 arrays  ✗ BROKEN
regcn_jraph_final.npz   - 22 bytes, 0 arrays  ✗ BROKEN
regcn_trained.pt        - 9.1MB  (PyTorch format, separate)
```

**Verification:**
```bash
.venv/bin/python -c "import numpy as np; f = np.load('models/tkg/regcn_jraph_best.npz'); print(len(f.keys()))"
# Output: 13

.venv/bin/python -c "import numpy as np; f = np.load('models/tkg/regcn_jraph_final.npz'); print(len(f.keys()))"
# Output: 0
```

**The Fix (already applied):**

Commit `1510b95` fixed `src/training/train_jraph.py`. The bug was:
```python
# OLD (broken) - checked for .value attribute that doesn't exist on JAX arrays
def flatten_state(prefix: str, obj):
    if hasattr(obj, 'value'):  # Never true after tree_flatten_with_path
        flat_state[prefix] = np.array(obj.value)

# NEW (working) - checks for array shape
leaves_with_paths, _ = jax.tree_util.tree_flatten_with_path(state)
for key_path, leaf in leaves_with_paths:
    if hasattr(leaf, 'shape') and len(leaf.shape) > 0:
        # ... save array
```

**Cleanup:** Delete the broken `.npz` files or retrain to replace them.

---

## Documented Future Work (Not Bugs)

### Scalability: Graph Partitioning

**Status:** Acknowledged limitation
**Location:** Documented in `.planning/PROJECT.md:87`, `.planning/STATE.md:93`

No code exists for graph partitioning. This is documented as "future work for >1M events." The current implementation handles the 1.8M GDELT events in-memory but may need partitioning for larger datasets.

### TODOs in Code

Three TODOs exist as enhancement placeholders:

1. `src/forecasting/tkg_predictor.py:544`
   ```python
   # TODO: Implement proper subject prediction in RE-GCN
   ```

2. `src/forecasting/tkg_predictor.py:580`
   ```python
   # TODO: Track pattern recency for more accurate decay
   ```

3. `src/forecasting/graph_validator.py:157`
   ```python
   # TODO: Use NLP to extract full (entity1, relation, entity2) triples.
   ```

These are enhancement opportunities, not bugs.

---

## Training Context

### Last Successful Training (2026-01-24)

```bash
# Configuration that worked on RTX 3060 12GB
.venv/bin/python scripts/train_tkg_jraph.py \
    --data-path data/tkg_training_data.parquet \
    --output-dir models/tkg \
    --max-events 100000 \
    --embedding-dim 128 \
    --hidden-dim 128 \
    --num-layers 1 \      # Key: 1 layer to avoid OOM
    --batch-size 256 \
    --epochs 50 \
    --learning-rate 0.001
```

**Results:**
- Epoch 1: MRR 0.1381, Hits@10: 0.2200
- Checkpoint: `models/tkg/regcn_jraph_best.npz` (15MB, valid)

**OOM notes:**
- 1.8M events OOM'd immediately
- 200k events + 3 layers OOM'd
- 100k events + 1 layer worked

### Data Available

```bash
ls -lh data/tkg_training_data.parquet
# 1.8M events, 591K triple patterns
```

---

## Key File Locations

| Component | File | Key Lines |
|-----------|------|-----------|
| Checkpoint save/load | `src/training/train_jraph.py` | `save_checkpoint()`, `load_checkpoint()` |
| Graph builder | `src/knowledge_graph/graph_builder.py` | `add_events_batch()` at line 160 |
| RAG indexing | `src/forecasting/rag_pipeline.py` | `index_graph_patterns()` at line 106 |
| Forecast engine | `src/forecasting/forecast_engine.py` | `__init__()`, `_retrieve_historical_context()` |
| Shortest path bug | `src/knowledge_graph/temporal_index.py` | Line 233 |
| TKG predictor | `src/forecasting/tkg_predictor.py` | TODOs at 544, 580 |

---

## Recommended Next Actions

1. **Create bootstrap script** (fixes #2 and #3):
   ```python
   # scripts/build_rag_index.py
   from src.knowledge_graph.graph_builder import TemporalKnowledgeGraph
   from src.forecasting.rag_pipeline import RAGPipeline

   graph = TemporalKnowledgeGraph()
   stats = graph.add_events_batch("data/events.db")
   print(f"Loaded {stats['valid_events']} events")

   rag = RAGPipeline()
   rag.index_graph_patterns(graph)
   print("RAG index built")
   ```

2. **Fix UAT-005** (5 min fix):
   ```python
   # In temporal_index.py:232-236
   try:
       paths = nx.single_source_shortest_path(self.graph, source, cutoff=max_length)
       return paths.get(target)
   except nx.NetworkXError:
       return None
   ```

3. **Clean up broken checkpoints**:
   ```bash
   rm models/tkg/regcn_jraph_epoch_*.npz models/tkg/regcn_jraph_final.npz
   ```

4. **Retrain with more data** (optional, user-initiated):
   - Current: 100k events, 1 layer, MRR 0.138
   - Consider: gradient checkpointing or model parallelism for full 1.8M events

---

## Session Continuity

**This audit performed:** 2026-01-27
**Previous session:** Fixed checkpoint save/load bug (commit `1510b95`)
**Config:** `.planning/config.json` has `model_profile: "quality"`

To resume work:
1. Read this file
2. Check `.planning/STATE.md` for project state
3. Run `/gsd:progress` to see current status
