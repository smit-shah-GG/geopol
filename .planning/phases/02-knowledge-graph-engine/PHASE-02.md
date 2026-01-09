# Phase 2: Knowledge Graph Engine

## Overview
Build temporal knowledge graph from GDELT event streams with efficient vector representations for geopolitical forecasting.

## Research Summary

### Technology Selection
After extensive analysis of CPU-friendly architectures:

**Vector Database**: Qdrant (self-hosted)
- HNSW index with native payload filtering
- 30-50ms search latency for 1M vectors
- Temporal and categorical filtering support
- 2.8GB memory for 1M 384-dim vectors

**Graph Library**: NetworkX + Custom Temporal Indexing
- In-memory MultiDiGraph (~200MB for 1M facts)
- O(1) node lookup, O(k) edge enumeration
- Custom temporal indexing for time-range queries
- Avoids heavyweight GNN frameworks (DGL/PyG)

**Embedding Model**: RotatE with HyTE temporal extension
- 256 dimensions for CPU efficiency
- Complex-space rotations handle asymmetric relations
- 0.35 MRR on ICEWS benchmark
- 5-minute training for 100K facts on 8-core CPU

**Temporal Reasoning**: HyTE + TA-DistMult ensemble
- HyTE: Fast orthogonal projections (0.1ms/fact)
- TA-DistMult: LSTM-based for extrapolation
- Ensemble via log-sum-exp combination

### Architecture Decisions

1. **Entity normalization**: CAMEO codes → canonical IDs with Wikidata mapping
2. **Relation classification**: Deterministic mapping from (event_code, quad_class)
3. **Confidence aggregation**: Bayesian log-odds combination
4. **Temporal bucketing**: Weekly (52 hyperplanes) for diplomatic cycles
5. **Memory strategy**: In-memory during training, disk serialization for inference

## Plans

### Plan 02-01: TKG Construction from GDELT Events
Transform SQLite events into NetworkX temporal MultiDiGraph with entity/relation extraction.
- Entity normalization from CAMEO codes
- Relation classification with confidence scoring
- Temporal indexing for efficient queries
- Target: 1000 events/second throughput

### Plan 02-02: Vector Embedding System
Train RotatE embeddings with HyTE temporal projections, store in Qdrant.
- 256-dim RotatE implementation
- CPU-optimized training pipeline
- HyTE weekly projections
- Qdrant indexing with metadata

### Plan 02-03: Graph Query Interface
Unified API for graph traversal, semantic search, and temporal filtering.
- Query parser with validation
- K-hop traversal with time bounds
- Semantic similarity via Qdrant
- Explanation generation for results

## Performance Targets

| Component | Metric | Target | Rationale |
|-----------|--------|--------|-----------|
| Entity extraction | Throughput | 100 articles/sec | Flair CPU optimization |
| Graph construction | Throughput | 1000 triples/sec | NetworkX hash operations |
| RotatE training | Time/100K facts | < 10 minutes | 256-dim, batch=256 |
| Vector indexing | Throughput | 50K vectors/sec | Qdrant batch upload |
| Query latency | P95 | < 10ms | HNSW + payload filtering |
| Memory usage | 1M facts | < 300MB graph + 500MB vectors | In-memory efficiency |

## Success Metrics

- Graph construction from 100K events in < 2 minutes
- RotatE achieves > 0.30 MRR on link prediction
- Semantic search improves recall by > 30%
- All queries return with explanation paths
- Total memory footprint < 1GB for 1M facts

## Risk Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| Incomplete actor coverage | Missing entities | Robust fallback ID generation |
| Training instability | Poor embeddings | Gradient clipping, careful initialization |
| Query complexity explosion | Slow queries | Query timeout, result limits |
| Memory overflow | System crash | Batch processing, disk spillover |

## Dependencies from Phase 1
- SQLite database with deduplicated events
- Event schema with QuadClass and confidence fields
- Data pipeline for continuous updates

## Outputs for Phase 3
- NetworkX temporal knowledge graph
- Trained RotatE embeddings in Qdrant
- Query API returning facts with explanations
- Performance benchmarks and quality metrics