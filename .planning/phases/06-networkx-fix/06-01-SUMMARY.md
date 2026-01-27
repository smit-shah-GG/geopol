# Phase 6 Plan 1: Fix NetworkX shortest_path API Misuse Summary

**One-liner:** Replaced broken `nx.shortest_path(cutoff=...)` with `nx.single_source_shortest_path(cutoff=...)` and added regression tests for connected, disconnected, and cutoff-exceeded cases.

## What Was Done

### Task 1: Fix NetworkX API call in TemporalIndex.shortest_path
- Replaced `nx.shortest_path(self.graph, source, target, cutoff=max_length)` which raises `TypeError` (no `cutoff` parameter exists on that function)
- Now uses `nx.single_source_shortest_path(self.graph, source, cutoff=max_length)` which returns a dict of reachable paths bounded by hop count
- Changed exception handling from `nx.NetworkXNoPath` to `nx.NodeNotFound` (correct for the new API)
- Returns `paths.get(target)` -- `None` when target unreachable or beyond cutoff
- **Commit:** `56fe9c7`

### Task 2: Update tests for shortest_path
- `test_shortest_path_connected`: Asserts USA->CHN direct path returns `['USA', 'CHN']`
- `test_shortest_path_disconnected`: Asserts NATO->CHN (no outgoing edges from NATO) returns `None`
- `test_shortest_path_cutoff_exceeded`: Local graph with 2-hop path X->Y->Z, asserts `max_length=1` returns `None` (core BUG-01 regression test)
- Removed old `test_shortest_path` which had no assertions (just `# Path may be None`)
- **Commit:** `ad70abd`

## Verification Results

- 17/17 tests pass in `test_temporal_index.py`
- `nx.shortest_path` as a function call eliminated from codebase (only in docstring comment)
- `nx.single_source_shortest_path` confirmed on line 237 with `cutoff=max_length`

## Deviations from Plan

None -- plan executed exactly as written.

## Decisions Made

| Decision | Rationale |
|----------|-----------|
| Catch `nx.NodeNotFound` instead of `nx.NetworkXNoPath` | `single_source_shortest_path` does not raise `NetworkXNoPath`; it omits unreachable nodes from the returned dict. `NodeNotFound` covers the case where `source` is not in the graph. |
| Local graph in cutoff test | Avoids perturbing shared `sample_graph` fixture that other tests assert edge counts against |

## Files Changed

| File | Action |
|------|--------|
| `src/knowledge_graph/temporal_index.py` | Modified `shortest_path` method (lines 220-240) |
| `src/knowledge_graph/test_temporal_index.py` | Replaced 1 test with 3 focused tests |

## Duration

~2 minutes
