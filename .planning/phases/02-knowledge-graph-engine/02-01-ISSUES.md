# UAT Issues: Phase 2 Plan 1

**Tested:** 2026-01-09
**Source:** .planning/phases/02-knowledge-graph-engine/02-01-SUMMARY.md
**Tester:** User via /gsd:verify-work

## Open Issues

[All issues resolved - see Resolved Issues section]

## Resolved Issues

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