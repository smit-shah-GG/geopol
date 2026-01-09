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

- [ ] **Phase 1: Data Foundation** - GDELT API integration with sampling strategy for conflicts/diplomatic events
- [ ] **Phase 2: Knowledge Graph Engine** - Temporal knowledge graph construction with vector embeddings
- [ ] **Phase 3: Hybrid Forecasting** - TKG algorithms (RE-GCN/TiRGN) combined with LLM reasoning
- [ ] **Phase 4: Calibration & Evaluation** - Brier score optimization with explainable reasoning chains

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
- [ ] 02-02: Vector embedding system for entities and temporal relations
- [ ] 02-03: Graph query interface for pattern retrieval

### Phase 3: Hybrid Forecasting
**Goal**: Implement ensemble prediction engine combining graph algorithms with LLM reasoning
**Depends on**: Phase 2
**Research**: Likely (algorithm selection)
**Research topics**: CPU-optimized implementations of RE-GCN/TiRGN, smaller LLMs (7B max) for reasoning, ensemble combination strategies, prompt engineering for geopolitical reasoning
**Plans**: 3 plans

Plans:
- [ ] 03-01: TKG prediction algorithms (RE-GCN/TiRGN) implementation
- [ ] 03-02: LLM reasoning component with RAG for scenario generation
- [ ] 03-03: Ensemble layer combining TKG and LLM predictions

### Phase 4: Calibration & Evaluation
**Goal**: Transform raw predictions into calibrated probabilities with explainable reasoning paths
**Depends on**: Phase 3
**Research**: Unlikely (established patterns from geopol.md reference)
**Plans**: 2 plans

Plans:
- [ ] 04-01: Probability calibration with Brier score optimization
- [ ] 04-02: Evaluation framework using 2023-2024 events with explainability metrics

## Progress

**Execution Order:**
Phases execute in numeric order: 1 → 2 → 3 → 4

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Data Foundation | 3/3 | Complete | 2026-01-09 |
| 2. Knowledge Graph Engine | 1/3 | In progress | - |
| 3. Hybrid Forecasting | 0/3 | Not started | - |
| 4. Calibration & Evaluation | 0/2 | Not started | - |

## Technical Context

**Language**: Python (scientific computing ecosystem)
**Event Focus**: Conflicts (QuadClass 4) and Diplomatic (QuadClass 1) events
**Evaluation**: Recent events (2023-2024) for contemporary relevance
**Compute**: CPU-optimized algorithms due to limited GPU resources