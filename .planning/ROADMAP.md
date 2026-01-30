# Roadmap: Explainable Geopolitical Forecasting Engine

## Milestones

- **v1.0 MVP** — Phases 1-5 (shipped 2026-01-23)
- **v1.1 Tech Debt Remediation** — Phases 6-8 (in progress)

## Phases

<details>
<summary>v1.0 MVP (Phases 1-5) - SHIPPED 2026-01-23</summary>

### Phase 1: Data Foundation
**Goal**: GDELT event ingestion pipeline with intelligent sampling
**Plans**: 3 plans

Plans:
- [x] 01-01: GDELT API client and event schema
- [x] 01-02: Event filtering and sampling strategy
- [x] 01-03: Storage layer and data pipeline

### Phase 2: Knowledge Graph Engine
**Goal**: Temporal knowledge graph construction from event streams
**Plans**: 3 plans

Plans:
- [x] 02-01: Entity extraction and resolution
- [x] 02-02: Relationship extraction and graph construction
- [x] 02-03: Graph querying and traversal

### Phase 3: Hybrid Forecasting
**Goal**: Ensemble prediction combining LLM reasoning with TKG patterns
**Plans**: 4 plans

Plans:
- [x] 03-01: RAG indexing pipeline
- [x] 03-02: LLM integration with Gemini
- [x] 03-03: RE-GCN TKG predictor
- [x] 03-04: Ensemble forecaster and CLI

### Phase 4: Calibration & Evaluation
**Goal**: Probability calibration and historical evaluation
**Plans**: 2 plans

Plans:
- [x] 04-01: Isotonic and temperature scaling calibration
- [x] 04-02: Evaluation framework against historical events

### Phase 5: TKG Training
**Goal**: RE-GCN training pipeline on GDELT data
**Plans**: 4 plans

Plans:
- [x] 05-01: GDELT data collection for training
- [x] 05-02: TKG data preprocessing
- [x] 05-03: JAX/jraph RE-GCN training
- [x] 05-04: Retraining scheduler

</details>

### v1.1 Tech Debt Remediation (In Progress)

**Milestone Goal:** Stabilize v1.0 foundation by resolving all known technical debt before adding new features.

#### Phase 6: NetworkX API Fix
**Goal**: Graph entity queries return valid results without API errors
**Depends on**: Phase 5 (v1.0 complete)
**Requirements**: BUG-01
**Success Criteria** (what must be TRUE):
  1. Running entity relationship queries against the knowledge graph returns path results without raising NetworkX API exceptions
  2. `single_source_shortest_path` is used in all graph traversal code paths that previously called `shortest_path` incorrectly
**Plans**: 1 plan

Plans:
- [x] 06-01: Fix NetworkX API call and update tests

#### Phase 7: Bootstrap Pipeline
**Goal**: A single command takes the system from zero data to fully operational (ingested events, built graph, indexed RAG store)
**Depends on**: Phase 6
**Requirements**: INFRA-01, INFRA-02
**Success Criteria** (what must be TRUE):
  1. Running the bootstrap script with no prior data completes the full pipeline (GDELT ingestion, knowledge graph construction, RAG index build) and the system is ready to accept forecast queries
  2. Running the bootstrap script a second time skips already-completed stages and finishes in significantly less time than a fresh run
  3. If the bootstrap script is interrupted mid-execution and re-run, it resumes from the last successful checkpoint without re-processing completed stages
  4. The bootstrap script reports progress for each stage (stage name, status, errors if any) to stdout
**Plans**: 2 plans

Plans:
- [x] 07-01: Bootstrap orchestration module with stage definitions
- [x] 07-02: Checkpoint/resume and dual idempotency

#### Phase 8: Graph Partitioning
**Goal**: Knowledge graph scales beyond 1M events through partitioning while preserving query correctness across partition boundaries
**Depends on**: Phase 7
**Requirements**: SCALE-01, SCALE-02
**Success Criteria** (what must be TRUE):
  1. Loading a knowledge graph with >1M events completes without running out of memory on the target hardware (CPU-only, standard research workstation)
  2. Entity relationship queries spanning multiple graph partitions return the same results as they would on a single unpartitioned graph
  3. Query performance on a partitioned 1M+ event graph is within 2x of query performance on a 100K event unpartitioned graph (no catastrophic degradation)
**Plans**: 2 plans

Plans:
- [ ] 08-01-PLAN.md — Partition index (SQLite) and partition manager (temporal-first partitioning, LRU cache)
- [ ] 08-02-PLAN.md — Boundary resolver, scatter-gather query router, PartitionedTemporalGraph interface

## Progress

**Execution Order:** 6 → 7 → 8

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|----------------|--------|-----------|
| 1. Data Foundation | v1.0 | 3/3 | Complete | 2026-01-11 |
| 2. Knowledge Graph | v1.0 | 3/3 | Complete | 2026-01-13 |
| 3. Hybrid Forecasting | v1.0 | 4/4 | Complete | 2026-01-17 |
| 4. Calibration | v1.0 | 2/2 | Complete | 2026-01-19 |
| 5. TKG Training | v1.0 | 4/4 | Complete | 2026-01-23 |
| 6. NetworkX Fix | v1.1 | 1/1 | Complete | 2026-01-28 |
| 7. Bootstrap Pipeline | v1.1 | 2/2 | Complete | 2026-01-30 |
| 8. Graph Partitioning | v1.1 | 0/2 | Planned | - |
