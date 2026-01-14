# UAT Issues: Phase 2 Plan 1

**Tested:** 2026-01-09
**Source:** .planning/phases/02-knowledge-graph-engine/02-01-SUMMARY.md
**Tester:** User via /gsd:verify-work
**Last Updated:** 2026-01-13 (resolved UAT-004, added UAT-005)

## Open Issues

### UAT-005: NetworkX shortest_path API incompatibility

**Discovered:** 2026-01-13
**Phase/Plan:** 02-01
**Severity:** Minor (non-blocking)
**Feature:** Shortest path queries
**Description:** The `shortest_path` method uses `cutoff` parameter which doesn't exist in `nx.shortest_path`
**Location:** `src/knowledge_graph/temporal_index.py:233`
**Expected:** `nx.shortest_path(G, source, target, cutoff=k)` should limit path length
**Actual:** `TypeError: shortest_path() got an unexpected keyword argument 'cutoff'`
**Impact:** `shortest_path` method unusable; not currently used in critical path
**Fix:** Use `nx.single_source_shortest_path(G, source, cutoff=k)` then filter for target
**Status:** Deferred - Non-blocking, can be fixed in future maintenance

## Resolved Issues

### UAT-004: Temporal index bisect operations mix types incorrectly

**Discovered:** 2026-01-09 (during re-verification)
**Phase/Plan:** 02-01
**Severity:** Minor (non-blocking)
**Feature:** Temporal index time-range queries
**Description:** The `edges_in_time_range` method in temporal_index.py uses bisect operations that incorrectly compare strings with integers
**Location:** `src/knowledge_graph/temporal_index.py` lines 90-91
**Expected:** Binary search should work with consistent types for comparison
**Actual:** TypeError: '<' not supported between instances of 'str' and 'int'
**Root Cause:** Type hint claimed `List[Tuple[str, int, int, int]]` but actual data was `List[Tuple[str, str, str, int]]`. Bisect search used integer placeholders `(start_date, 0, 0, 0)` that couldn't compare with string node names when timestamps collided.
**Fix:** Changed bisect sentinels to type-consistent values: `(start_date, '', '', 0)` and `(end_date, '~', '~', 2**31)`. Fixed type hint. Added regression test `test_edges_in_time_range_exact_timestamp_match`.
**Resolved:** 2026-01-13

### UAT-001: Import paths are incorrect throughout the codebase

**Discovered:** 2026-01-09
**Phase/Plan:** 02-01
**Severity:** Blocker
**Feature:** Module imports across all files
**Description:** Imports use bare module names (e.g., `from entity_normalization import`) instead of proper relative or absolute imports
**Expected:** Modules should import correctly using relative imports (`from .entity_normalization import`) or absolute (`from src.knowledge_graph.entity_normalization import`)
**Actual:** ModuleNotFoundError when trying to import modules
**Repro:**
1. Try to import graph_builder: `from src.knowledge_graph.graph_builder import TemporalKnowledgeGraph`
2. Fails with: ModuleNotFoundError: No module named 'entity_normalization'
**Resolved:** 2026-01-09 - Fixed in 02-01-FIX.md
**Commit:** 21c3348

### UAT-002: Missing networkx dependency

**Discovered:** 2026-01-09
**Phase/Plan:** 02-01
**Severity:** Major (resolved during testing)
**Feature:** Graph operations
**Description:** NetworkX library was not installed despite being a critical dependency
**Expected:** NetworkX should be installed as part of the requirements
**Actual:** ModuleNotFoundError: No module named 'networkx'
**Repro:** Import any module that uses networkx
**Note:** User installed networkx manually during testing, but it should be in requirements
**Resolved:** 2026-01-09 - Verified in 02-01-FIX.md (was already in requirements.txt)

### UAT-003: Method signatures don't match documentation

**Discovered:** 2026-01-09
**Phase/Plan:** 02-01
**Severity:** Minor
**Feature:** API consistency
**Description:** Some method names differ from what was described in SUMMARY (e.g., `resolve_actor` vs `normalize`)
**Expected:** Consistent API as documented
**Actual:** Different method names requiring discovery via dir()
**Repro:** Try calling EntityNormalizer.normalize() - doesn't exist, it's resolve_actor()
**Resolved:** 2026-01-09 - Fixed in 02-01-FIX.md (added API documentation)
**Commit:** 89d73c2

---

*Phase: 02-knowledge-graph-engine*
*Plan: 01*
*Tested: 2026-01-09*