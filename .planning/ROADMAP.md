# Roadmap: Explainable Geopolitical Forecasting Engine

## Overview

Building a hybrid intelligence forecasting system that processes GDELT events through temporal knowledge graphs and ensemble models to produce explainable predictions. The journey progresses from establishing data pipelines through knowledge representation to hybrid prediction and finally calibration, focusing on conflicts and diplomatic events with evaluation on recent (2023-2024) geopolitical developments.

## Domain Expertise

None

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [x] **Phase 1: Data Foundation** - GDELT API integration with sampling strategy for conflicts/diplomatic events
- [x] **Phase 2: Knowledge Graph Engine** - Temporal knowledge graph construction with vector embeddings
- [x] **Phase 3: Hybrid Forecasting** - TKG algorithms (RE-GCN/TiRGN) combined with LLM reasoning
- [x] **Phase 4: Calibration & Evaluation** - Brier score optimization with explainable reasoning chains
- [ ] **Phase 5: TKG Training** - TKG predictor training with RE-GCN implementation and GDELT data pipeline

## Phase Details

### Phase 1: Data Foundation
**Goal**: Establish GDELT data pipeline with intelligent sampling for compute-constrained environment
**Depends on**: Nothing (first phase)
**Research**: Likely (external API)
**Research topics**: GDELT API v2 documentation, rate limits, CAMEO QuadClass filtering for conflicts (4) and diplomatic (1), optimal sampling strategies for 500K-1M daily articles
**Plans**: 3 plans

Plans:
- [ ] 01-01: GDELT API client with rate limiting and error handling
- [ ] 01-02: Event storage schema optimized for TKG construction
- [ ] 01-03: Intelligent sampling strategy for conflicts/diplomatic events

### Phase 2: Knowledge Graph Engine
**Goal**: Build temporal knowledge graph from event streams with efficient vector representations
**Depends on**: Phase 1
**Research**: Likely (new system, technology choice)
**Research topics**: CPU-friendly vector databases (Milvus vs Qdrant vs FAISS), NetworkX for graph operations, embedding models (TransE/RotatE/ComplEx), temporal extensions (DE-SimplE/HyTE)
**Plans**: 3 plans

Plans:
- [x] 02-01: TKG construction from GDELT events with entity/relation extraction (COMPLETE 2026-01-09)
- [x] 02-02: Vector embedding system for entities and temporal relations (COMPLETE 2026-01-09)
- [x] 02-03: Graph query interface for pattern retrieval (COMPLETE 2026-01-09)

### Phase 3: Hybrid Forecasting ✅
**Goal**: Implement ensemble prediction engine combining graph algorithms with LLM reasoning
**Depends on**: Phase 2
**Research**: Completed
**Status**: COMPLETE (4/4 plans executed)
**Accomplishments**: Gemini API integration, RAG pipeline, RE-GCN/TKG predictor, ensemble with CLI
**Plans**: 4 plans (all complete)

Plans:
- [x] 03-01: Gemini API integration with multi-step reasoning (COMPLETE 2026-01-10)
- [x] 03-02: RAG pipeline for historical grounding (COMPLETE 2026-01-10)
- [x] 03-03: TKG algorithms (RE-GCN) integration (COMPLETE 2026-01-10)
- [x] 03-04: Ensemble layer and CLI interface (COMPLETE 2026-01-10)

### Phase 4: Calibration & Evaluation ✅
**Goal**: Transform raw predictions into calibrated probabilities with explainable reasoning paths
**Depends on**: Phase 3
**Research**: Completed
**Status**: COMPLETE (2/2 plans executed)
**Accomplishments**: Isotonic calibration, temperature scaling, Brier scoring, human baseline comparison
**Plans**: 2 plans (all complete)

Plans:
- [x] 04-01: Probability calibration with Brier score optimization (COMPLETE 2026-01-13)
- [x] 04-02: Evaluation framework using 2023-2024 events with explainability metrics (COMPLETE 2026-01-13)

### Phase 5: TKG Training
**Goal**: Train the Temporal Knowledge Graph predictor with real GDELT data and implement RE-GCN for production use
**Depends on**: Phase 4
**Research**: Completed
**Status**: In progress (2/4 plans executed)
**Plans**: 4 plans

Plans:
- [x] 05-01: GDELT data collection pipeline (COMPLETE 2026-01-13)
- [x] 05-02: RE-GCN implementation (COMPLETE 2026-01-13)
- [ ] 05-03: Training pipeline
- [ ] 05-04: Integration and evaluation

**Details:**
The TKG predictor is currently not trained, limiting the system to LLM-only predictions. This phase will:
- Collect historical GDELT event data (30-90 days)
- Build temporal knowledge graphs from event sequences
- Implement RE-GCN or frequency-based predictor
- Train on historical patterns with evaluation metrics
- Integrate trained model into ensemble predictions

## Progress

**Execution Order:**
Phases execute in numeric order: 1 → 2 → 3 → 4 → 5

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Data Foundation | 3/3 | Complete | 2026-01-09 |
| 2. Knowledge Graph Engine | 3/3 | Complete | 2026-01-09 |
| 3. Hybrid Forecasting | 4/4 | Complete | 2026-01-10 |
| 4. Calibration & Evaluation | 2/2 | Complete | 2026-01-13 |
| 5. TKG Training | 2/4 | In progress | - |

## Technical Context

**Language**: Python (scientific computing ecosystem)
**Event Focus**: Conflicts (QuadClass 4) and Diplomatic (QuadClass 1) events
**Evaluation**: Recent events (2023-2024) for contemporary relevance
**Compute**: CPU-optimized algorithms due to limited GPU resources