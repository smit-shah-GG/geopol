# Phase 06 Research: NetworkX API Fix

## Bug Analysis

**Root cause:** `nx.shortest_path()` does not accept a `cutoff` parameter. The `cutoff` parameter belongs to `nx.single_source_shortest_path()` and related functions. Current code passes `cutoff=max_length` which is either silently ignored or raises TypeError depending on NetworkX version.

**Location:** `src/knowledge_graph/temporal_index.py:233`

```python
# CURRENT (broken)
path = nx.shortest_path(self.graph, source, target, cutoff=max_length)

# The nx.shortest_path signature is:
# nx.shortest_path(G, source=None, target=None, weight=None, method='dijkstra')
# There is NO cutoff parameter.
```

**Correct API options:**
1. `nx.shortest_path(G, source, target)` — returns shortest path, no length limit
2. `nx.single_source_shortest_path(G, source, cutoff=max_length)` — returns dict of all paths from source up to cutoff length, then filter for target
3. `nx.shortest_path_length(G, source, target)` + check — get length first, compare to max

**Recommendation:** Option 2 (`single_source_shortest_path`) preserves the `max_length` constraint which is semantically important for limiting graph traversal depth.

## Files Affected

| File | Line | Change |
|------|------|--------|
| `src/knowledge_graph/temporal_index.py` | 233 | Fix API call |
| `src/knowledge_graph/test_temporal_index.py` | 154-158 | Update test |

## Exception Handling

Current code catches `nx.NetworkXNoPath`. With `single_source_shortest_path`, a missing path means the target key is absent from the returned dict — no exception raised. Exception handling needs adjustment.

## Risk

Minimal. Single method, well-tested path, no downstream API changes (method signature unchanged).
