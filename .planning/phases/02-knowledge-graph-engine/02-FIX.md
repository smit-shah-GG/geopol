---
phase: 02-knowledge-graph-engine
plan: 02-FIX
type: fix
---

<objective>
Fix 1 UAT issue from Phase 02.

Source: 02-ISSUES.md
Priority: 0 critical, 1 major, 0 minor
</objective>

<execution_context>
@~/.claude/get-shit-done/workflows/execute-phase.md
@~/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/STATE.md
@.planning/ROADMAP.md

**Issues being fixed:**
@.planning/phases/02-knowledge-graph-engine/02-ISSUES.md

**Original plan for reference:**
@.planning/phases/02-knowledge-graph-engine/02-03-PLAN.md
</context>

<tasks>
<task type="auto">
  <name>Fix UAT-001: QueryEngine entity resolution fails</name>
  <files>src/knowledge_graph/query_engine.py</files>
  <action>
    Fix the _resolve_entity method to handle cases where entity_to_id mappings are not provided.

    Current issue: When entity_to_id is None, the method falls through to "raise ValueError" even when the entity exists in the graph.

    Solution:
    1. Check if entity exists directly in graph.nodes() when entity_to_id is None
    2. If entity is in graph.nodes(), return the entity name itself (graphs can use string IDs)
    3. Only raise ValueError if entity not in graph AND not in mappings

    Modify _resolve_entity method (around line 425-433):
    - Add check: if self.entity_to_id is None and self.graph and entity_name in self.graph.nodes()
    - Return entity_name directly in this case
    - This allows QueryEngine to work with graphs that use string node IDs
  </action>
  <verify>
    uv run python -c "
from src.knowledge_graph.query_engine import QueryEngine
from src.knowledge_graph.graph_builder import TemporalKnowledgeGraph
from datetime import date

# Create test graph
tkg = TemporalKnowledgeGraph()
event = {'event_date': date(2024, 1, 1), 'actor1_code': 'USA', 'actor2_code': 'CHN',
         'event_code': '190', 'quad_class': 4, 'goldstein_scale': -10.0,
         'tone': -5.0, 'num_mentions': 10}
tkg.add_event_from_db_row(event)

# Test query without entity mappings
engine = QueryEngine(graph=tkg.graph, vector_store=None, embedding_model=None)
query = {'query_type': 'entity_pair', 'entity1': 'USA', 'entity2': 'CHN'}
result = engine.execute_query(query)
print('✓ Query executed without ValueError')
"
  </verify>
  <done>QueryEngine can resolve entities that exist in graph.nodes() even without entity_to_id mappings</done>
</task>

<task type="auto">
  <name>Add test coverage for entity resolution without mappings</name>
  <files>src/knowledge_graph/test_query_engine.py</files>
  <action>
    Add a test case that verifies QueryEngine works correctly when initialized without entity_to_id mappings.

    Test should:
    1. Create a simple graph with string node IDs (e.g., 'USA', 'CHN', 'RUS')
    2. Initialize QueryEngine with graph but no entity_to_id/id_to_entity mappings
    3. Execute an entity_pair query using the string IDs
    4. Verify query succeeds without ValueError
    5. Verify results are returned correctly

    This ensures the fix is covered by tests and prevents regression.
  </action>
  <verify>
    uv run python -m pytest src/knowledge_graph/test_query_engine.py::test_entity_resolution_without_mappings -xvs
  </verify>
  <done>Test passes, confirming entity resolution works without mappings</done>
</task>
</tasks>

<verification>
Before declaring plan complete:
- [ ] UAT-001 fixed: QueryEngine resolves entities from graph.nodes()
- [ ] Original repro steps no longer cause ValueError
- [ ] Test coverage added for the fix
- [ ] All existing tests still pass
</verification>

<success_criteria>
- UAT-001 from 02-ISSUES.md addressed
- Entity resolution works with or without entity_to_id mappings
- Tests pass
- Ready for re-verification with /gsd:verify-work
</success_criteria>

<output>
After completion, create `.planning/phases/02-knowledge-graph-engine/02-FIX-SUMMARY.md`
</output>