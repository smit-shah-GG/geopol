# UAT Issues: Phase 2 Plan 2

**Tested:** 2026-01-09
**Source:** .planning/phases/02-knowledge-graph-engine/02-02-SUMMARY.md
**Tester:** User via /gsd:verify-work

## Open Issues

[None]

## Resolved Issues

### UAT-001: Import paths are incorrect in embedding modules

**Discovered:** 2026-01-09
**Phase/Plan:** 02-02
**Severity:** Blocker
**Feature:** Module imports
**Description:** Import statements in 9 files use bare `knowledge_graph` instead of `src.knowledge_graph`
**Expected:** Imports should use `from src.knowledge_graph.embeddings import ...`
**Actual:** ModuleNotFoundError: No module named 'knowledge_graph'
**Files affected:**
- embedding_trainer.py
- evaluation.py
- temporal_embeddings.py
- vector_store.py
- test_embeddings.py
- test_embedding_trainer.py
- test_evaluation.py
- test_temporal_embeddings.py
- test_vector_store.py
**Repro:**
1. Try to import: `from src.knowledge_graph.embedding_trainer import EmbeddingTrainer`
2. Fails with ModuleNotFoundError
**Resolution:** Fixed in commit 77549cd - changed all imports to use `src.knowledge_graph` prefix
**Verified:** All modules can be imported successfully, test suite collects 14 tests

### UAT-002: Missing required dependencies

**Discovered:** 2026-01-09
**Phase/Plan:** 02-02
**Severity:** Major
**Feature:** Module dependencies
**Description:** PyTorch and typing_extensions were not installed despite being in requirements
**Expected:** All dependencies installed and working
**Actual:** Had to manually install torch (CPU version) and typing_extensions
**Repro:** Fresh environment missing torch and typing_extensions
**Resolution:** Fixed in commit b4bbba7 - added typing_extensions>=4.0.0 and matplotlib>=3.7.0 to requirements.txt with clear CPU-only torch installation instructions
**Verified:** Requirements.txt now documents all dependencies with installation commands

---

*Phase: 02-knowledge-graph-engine*
*Plan: 02*
*Tested: 2026-01-09*