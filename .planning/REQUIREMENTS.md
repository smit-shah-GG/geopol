# Requirements: Geopol

**Core Value:** Explainability — every forecast must provide clear, traceable reasoning paths

## Delivered Requirements

### v1.0 MVP (shipped 2026-01-23)

- [x] Event data ingestion from GDELT API with custom enrichment pipeline
- [x] Temporal knowledge graph construction from event streams
- [x] Hybrid model ensemble combining TKG algorithms (RE-GCN/TiRGN) with LLM reasoning
- [x] Explainable reasoning chain generation for each prediction
- [x] Probability calibration system with Brier score optimization
- [x] Evaluation framework against historical events

### v1.1 Tech Debt (shipped 2026-01-30)

- [x] Fix NetworkX shortest_path API to use single_source_shortest_path
- [x] Production bootstrap script connecting data ingestion -> graph build -> RAG indexing
- [x] Graph partitioning for scalability beyond 1M events

## v2.0 Requirements

**Status:** Pending definition. Run `/gsd:new-milestone` to define.

The original v2.0 requirements (23 Llama-TGL deep integration requirements) were cancelled on 2026-02-14. Full cancelled requirements archived in `.planning/archive/v2.0-llama-cancelled.md`.

## Future Requirements (Backlog)

Tracked ideas not yet assigned to a milestone.

### Potential Improvements

- **ADV-03**: Multi-language support (non-English GDELT sources)
- **ADV-04**: Real-time 15-minute update cycles (vs daily batch)

### Out of Scope

Explicitly excluded. Documented to prevent scope creep.

| Feature | Reason |
|---------|--------|
| Web production deployment | Research prototype; CLI sufficient for now |
| Human forecaster integration | Pure AI system per original scope |
| Cross-sector impact modeling | Financial/supply chain out of scope per v1.0 |

---
*Last updated: 2026-02-14 — v2.0 Llama requirements cancelled*
*Original requirements defined: 2026-01-31*
