# Requirements: Explainable Geopolitical Forecasting Engine

**Defined:** 2026-01-28
**Core Value:** Explainability — every forecast must provide clear, traceable reasoning paths

## v1.1 Requirements

Requirements for tech debt remediation. Each maps to roadmap phases.

### Bug Fixes

- [ ] **BUG-01**: Graph query path uses correct NetworkX API (`single_source_shortest_path` instead of `shortest_path`) so entity relationship queries return valid results without API errors

### Infrastructure

- [ ] **INFRA-01**: Production bootstrap script orchestrates full pipeline (GDELT ingestion → knowledge graph construction → RAG index build) in a single invocation with error handling and progress reporting
- [ ] **INFRA-02**: Bootstrap script is idempotent — re-running skips completed stages and resumes from last successful checkpoint

### Scalability

- [ ] **SCALE-01**: Knowledge graph supports >1M events through graph partitioning without degrading query performance
- [ ] **SCALE-02**: Partitioned graph maintains cross-partition entity resolution so queries spanning partitions return correct results

## Future Requirements

Deferred to v1.2+. Not in current roadmap.

- **PERF-01**: Query response time profiling and optimization targets
- **DATA-01**: Multi-source data ingestion beyond GDELT (ACLED, ICEWS)
- **UI-01**: Web dashboard for forecast visualization

## Out of Scope

| Feature | Reason |
|---------|--------|
| New forecasting algorithms | Stabilization milestone — no new ML capabilities |
| Real-time processing | Batch architecture decision unchanged |
| Multi-language NLP | English-only constraint unchanged |
| Web frontend | CLI/API only for research prototype |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| BUG-01 | Phase 6 | Pending |
| INFRA-01 | Phase 7 | Pending |
| INFRA-02 | Phase 7 | Pending |
| SCALE-01 | Phase 8 | Pending |
| SCALE-02 | Phase 8 | Pending |

**Coverage:**
- v1.1 requirements: 5 total
- Mapped to phases: 5
- Unmapped: 0

---
*Requirements defined: 2026-01-28*
*Last updated: 2026-01-28 after roadmap creation*
