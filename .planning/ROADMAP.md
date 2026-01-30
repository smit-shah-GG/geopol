# Roadmap: Explainable Geopolitical Forecasting Engine

## Milestones

- **v1.0 MVP** — Phases 1-5 (shipped 2026-01-23)
- **v1.1 Tech Debt Remediation** — Phases 6-8 (shipped 2026-01-30)
- **v2.0 Hybrid Architecture** — Phases 9-14 (in progress)

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

### v2.0 Hybrid Architecture (In Progress)

**Milestone Goal:** Replace post-hoc 60/40 ensemble with deep token-space integration where TKG embeddings project directly into Llama2-7B token space, targeting 40-60% accuracy improvement.

#### Phase 9: Environment Setup & Data Preparation
**Goal**: JAX/PyTorch memory coordination works, GDELT data prepared for deep integration training
**Depends on**: Phase 8 (graph infrastructure)
**Requirements**: DEEP-05, MIG-01
**Success Criteria** (what must be TRUE):
  1. JAX and PyTorch coexist in same process without OOM from memory pre-allocation conflict
  2. DLPack zero-copy tensor conversion transfers embeddings JAX->PyTorch without CPU round-trip
  3. CAMEO relation codes map to training-friendly taxonomy (20 root categories)
  4. v1.1 Gemini forecasting path remains fully operational during v2.0 development
**Plans**: TBD

Plans:
- [ ] 09-01: TBD
- [ ] 09-02: TBD

#### Phase 10: TKG Encoder & Adapter Architecture
**Goal**: TKG encoder upgraded to HisMatch, adapter layers project 200-dim embeddings to 4096-dim Llama token space
**Depends on**: Phase 9 (environment ready)
**Requirements**: TKG-01, TKG-03, TKG-04, DEEP-01, DEEP-03, DEEP-04
**Success Criteria** (what must be TRUE):
  1. RE-GCN replaced with HisMatch encoder achieving ~46% MRR on GDELT test set
  2. Entity/relation adapters project (batch, 200) embeddings to (batch, 4096) with correct layer norm
  3. Llama2-7B loads with 4-bit NF4 quantization using <4GB VRAM
  4. LoRA adapters configured on q_proj/v_proj with frozen backbone (verify trainable param count)
  5. Temporal window length adapts based on event density in query region
**Plans**: TBD

Plans:
- [ ] 10-01: TBD
- [ ] 10-02: TBD
- [ ] 10-03: TBD

#### Phase 11: Temporal Tokenizer & Llama Integration
**Goal**: Graph embeddings tokenized across T snapshots and injected as soft prompts into Llama generation
**Depends on**: Phase 10 (adapters ready)
**Requirements**: DEEP-02
**Success Criteria** (what must be TRUE):
  1. Temporal tokenizer sequences T=5-7 graph snapshots as soft tokens concatenated with text query
  2. Graph token positions tracked for embedding injection during LLM forward pass
  3. Llama generates coherent text with injected graph tokens (sanity check: removing graph tokens changes output)
**Plans**: TBD

Plans:
- [ ] 11-01: TBD

#### Phase 12: Two-Stage Training Pipeline
**Goal**: Adapters and LoRA trained with cross-modal alignment, model predicts 42+ CAMEO relations with context-aware confidence
**Depends on**: Phase 11 (integration ready)
**Requirements**: DEEP-06, DEEP-07, DEEP-08
**Success Criteria** (what must be TRUE):
  1. Stage 1 training achieves cross-modal alignment on 100K high-quality samples (adapter output variance >0.1, cosine similarity <0.95)
  2. Stage 2 training generalizes across diverse CAMEO relations with stratified sampling
  3. Model predicts 42+ CAMEO relation types with per-class F1 >0.3 on held-out test
  4. Model outputs confidence scores that vary based on graph signal strength (not fixed confidence)
  5. LLM reasoning references graph structure in explanations (not purely text-based)
**Plans**: TBD

Plans:
- [ ] 12-01: TBD
- [ ] 12-02: TBD

#### Phase 13: Evaluation & Calibration
**Goal**: v2.0 TGL-LLM validated against v1.1 baseline with calibrated probabilities
**Depends on**: Phase 12 (trained model)
**Requirements**: VAL-01, VAL-02, VAL-03, VAL-04, VAL-05, VAL-06
**Success Criteria** (what must be TRUE):
  1. A/B test harness runs v1.1 Gemini and v2.0 TGL-LLM on identical held-out questions
  2. v2.0 shows measurable accuracy improvement over v1.1 (any positive delta validates architecture)
  3. Brier scores computed for both models; v2.0 calibration does not regress (ECE <0.15)
  4. Per-relation accuracy breakdown available for all 42+ CAMEO categories
  5. Inference latency benchmarked end-to-end (expected: 10-30s vs v1.1's ~2s)
**Plans**: TBD

Plans:
- [ ] 13-01: TBD
- [ ] 13-02: TBD

#### Phase 14: Dashboard & Integration
**Goal**: Streamlit dashboard displays comparison metrics, v2.0 integrated as primary forecaster with v1.1 fallback
**Depends on**: Phase 13 (evaluation complete)
**Requirements**: DASH-01, DASH-02, DASH-03, TKG-02, MIG-02
**Success Criteria** (what must be TRUE):
  1. Streamlit dashboard displays accuracy, Brier, and latency metrics with charts
  2. Dashboard shows live side-by-side predictions from v1.1 and v2.0 models
  3. Dashboard updates metrics as new predictions are evaluated
  4. Entity embeddings cached to amortize RE-GCN/HisMatch encoding cost across predictions
  5. Manual fallback to v1.1 available if v2.0 predictions fail or degrade
**Plans**: TBD

Plans:
- [ ] 14-01: TBD
- [ ] 14-02: TBD

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
| 9. Environment Setup | v2.0 | 0/2 | Not started | - |
| 10. TKG Encoder & Adapter | v2.0 | 0/3 | Not started | - |
| 11. Temporal Tokenizer | v2.0 | 0/1 | Not started | - |
| 12. Training Pipeline | v2.0 | 0/2 | Not started | - |
| 13. Evaluation & Calibration | v2.0 | 0/2 | Not started | - |
| 14. Dashboard & Integration | v2.0 | 0/2 | Not started | - |

**Total:** 14 phases, 21 plans complete (v1.0+v1.1), 12 plans pending (v2.0)
