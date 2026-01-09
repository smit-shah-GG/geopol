# UAT Issues: Phase 2

**Tested:** 2026-01-09
**Source:** Phase 2 summaries (02-01, 02-02, 02-03)
**Tester:** User via /gsd:verify-work

## Open Issues

[None - all resolved]

## Resolved Issues

### UAT-001: QueryEngine entity resolution fails

**Discovered:** 2026-01-09
**Phase/Plan:** 02-03
**Severity:** Major
**Feature:** Query Engine entity resolution
**Description:** QueryEngine._resolve_entity() raises ValueError even when entities exist in graph
**Expected:** Query should resolve entities that are present as nodes in the graph
**Actual:** ValueError: Unknown entity: USA (even though 'USA' is in graph.nodes())
**Files affected:** src/knowledge_graph/query_engine.py line 433
**Root cause:** QueryEngine expects entity_to_id mappings but doesn't receive them from graph
**Resolution:** Fixed in 02-FIX.md - modified _resolve_entity to check graph.nodes() when mappings unavailable
**Commits:** 2eb2ae1 (fix), b15b07d (test)
**Verified:** Original repro steps now work correctly

---

*Phase: 02-knowledge-graph-engine*
*Plan: All (02-01, 02-02, 02-03)*
*Tested: 2026-01-09*