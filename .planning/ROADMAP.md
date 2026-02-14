# Roadmap: Explainable Geopolitical Forecasting Engine

## Milestones

- **v1.0 MVP** — Phases 1-5 (shipped 2026-01-23)
- **v1.1 Tech Debt Remediation** — Phases 6-8 (shipped 2026-01-30)
- **v2.0 (Direction Pending)** — Llama-TGL plan cancelled 2026-02-14, new direction TBD via `/gsd:new-milestone`

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

<details>
<summary>v1.1 Tech Debt Remediation (Phases 6-8) - SHIPPED 2026-01-30</summary>

### Phase 6: NetworkX API Fix
**Goal**: Graph entity queries return valid results without API errors
**Plans**: 1 plan

Plans:
- [x] 06-01: Fix NetworkX API call and update tests

### Phase 7: Bootstrap Pipeline
**Goal**: A single command takes the system from zero data to fully operational
**Plans**: 2 plans

Plans:
- [x] 07-01: Bootstrap orchestration module with stage definitions
- [x] 07-02: Checkpoint/resume and dual idempotency

### Phase 8: Graph Partitioning
**Goal**: Knowledge graph scales beyond 1M events through partitioning
**Plans**: 2 plans

Plans:
- [x] 08-01: Partition index (SQLite) and partition manager (temporal-first partitioning, LRU cache)
- [x] 08-02: Boundary resolver, scatter-gather query router, PartitionedTemporalGraph interface

</details>

### v2.0 (Direction Pending)

The original v2.0 plan (Llama-TGL deep token-space integration) was cancelled on 2026-02-14.

**Reason:** Frontier-class Gemini reasoning significantly outperforms 4-bit Llama2-7B constrained to RTX 3060 12GB. Engineering effort (6 phases) for uncertain gains on an inferior reasoning engine is not justified.

**Archived:** Full plan, requirements, and phases preserved in `.planning/archive/v2.0-llama-cancelled.md`.

**Next step:** Run `/gsd:new-milestone` to define the new v2.0 direction.

## Progress

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|----------------|--------|-----------|
| 1. Data Foundation | v1.0 | 3/3 | Complete | 2026-01-11 |
| 2. Knowledge Graph | v1.0 | 3/3 | Complete | 2026-01-13 |
| 3. Hybrid Forecasting | v1.0 | 4/4 | Complete | 2026-01-17 |
| 4. Calibration | v1.0 | 2/2 | Complete | 2026-01-19 |
| 5. TKG Training | v1.0 | 4/4 | Complete | 2026-01-23 |
| 6. NetworkX Fix | v1.1 | 1/1 | Complete | 2026-01-28 |
| 7. Bootstrap Pipeline | v1.1 | 2/2 | Complete | 2026-01-30 |
| 8. Graph Partitioning | v1.1 | 2/2 | Complete | 2026-01-30 |
| ~~9-14~~ | ~~v2.0 Llama-TGL~~ | - | Cancelled | 2026-02-14 |

**Total:** 8 phases complete (v1.0 + v1.1), 21 plans delivered. v2.0 direction pending.
