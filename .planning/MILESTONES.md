# Project Milestones: Explainable Geopolitical Forecasting Engine

## v1.1 Tech Debt Remediation (Shipped: 2026-01-30)

**Delivered:** Stabilized v1.0 foundation by fixing NetworkX API bug, adding single-command bootstrap pipeline with checkpoint/resume, and implementing graph partitioning for >1M event scalability.

**Phases completed:** 6-8 (5 plans total)

**Key accomplishments:**

- Fixed NetworkX API misuse (UAT-005) — replaced broken shortest_path with single_source_shortest_path
- Bootstrap pipeline with 5-stage orchestrator for zero-to-operational system initialization
- Dual idempotency (checkpoint + output validation) for robust resume after interruption
- Temporal-first graph partitioning with SQLite index and LRU cache
- Cross-partition query correctness (100%) via merged graph approach for k-hop traversal

**Stats:**

- 20 source files created/modified
- 40,257 lines of Python (up from 37,414)
- 3 phases, 5 plans, ~12 tasks
- 2 days from milestone start to ship (2026-01-28 → 2026-01-30)

**Git range:** `feat(06-01)` → `feat(08-02)`

**What's next:** v1.2 for performance profiling, multi-source data ingestion (ACLED, ICEWS), or web dashboard.

---

## v1.0 MVP (Shipped: 2026-01-23)

**Delivered:** Fully functional explainable geopolitical forecasting engine combining GDELT data ingestion, temporal knowledge graphs, and hybrid LLM+TKG ensemble prediction with probability calibration.

**Phases completed:** 1-5 (16 plans total)

**Key accomplishments:**

- GDELT event ingestion pipeline with intelligent sampling for conflicts/diplomatic events
- Temporal knowledge graph construction with entity resolution and relationship extraction
- RE-GCN implementation trained on 1.8M GDELT events (591K triple patterns)
- Hybrid ensemble forecaster combining LLM reasoning (60%) with TKG patterns (40%)
- Isotonic calibration and temperature scaling for probability calibration
- Weekly automated retraining scheduler with model versioning

**Stats:**

- 89 source files created/modified
- 37,414 lines of Python
- 5 phases, 16 plans, ~80 tasks
- 15 days from project start to ship (2026-01-09 → 2026-01-23)

**Git range:** Phase 01 commits → `feat(05-04)`

**What's next:** Project complete. Consider v1.1 for production deployment scripts, performance optimization, or expanded data sources.

---

*For archived roadmap and requirements, see `.planning/milestones/v1.0-*`*
