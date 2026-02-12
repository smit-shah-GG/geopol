# Geopolitical Forecasting Engine

An AI-powered system for processing GDELT events and generating explainable geopolitical forecasts using Temporal Knowledge Graphs and hybrid models.

## Overview

This project implements a geopolitical forecasting engine that:
- Ingests conflict and diplomatic events from GDELT
- Constructs temporal knowledge graphs
- Generates probabilistic forecasts with clear reasoning paths
- Calibrates predictions using Brier scores

## Setup

1. Install dependencies (requires [uv](https://docs.astral.sh/uv/)):
```bash
uv sync
```

2. Configure environment (optional):
```bash
cp .env.example .env
# Edit .env with your settings
```

3. Bootstrap the system (single command, zero-to-operational):
```bash
uv run python scripts/bootstrap.py
```

The bootstrap pipeline executes 5 stages in sequence:
1. **collect** - Fetch GDELT events (30 days by default)
2. **process** - Transform to TKG format + load into SQLite
3. **graph** - Build temporal knowledge graph
4. **persist** - Save graph to GraphML
5. **index** - Index patterns in RAG store

Bootstrap supports checkpoint/resume - if interrupted, re-running skips completed stages:
```bash
# Check what would run without executing
uv run python scripts/bootstrap.py --dry-run

# Force re-run of a specific stage
uv run python scripts/bootstrap.py --force-stage collect

# Customize collection period
uv run python scripts/bootstrap.py --n-days 60
```

## Quick Start

1. Verify system readiness:
```bash
uv run python scripts/preflight.py
```

2. Run a forecast:
```bash
uv run python scripts/forecast.py --question "Will NATO expand to include new members in the next 6 months?"
```

3. For detailed output:
```bash
uv run python scripts/forecast.py --question "..." --verbose
```

## Architecture

### Phase 1: Data Foundation
- GDELT API client with rate limiting
- Event storage with deduplication
- Intelligent sampling for compute constraints

### Phase 2: Knowledge Graph Engine
- Temporal knowledge graph construction
- Entity extraction and linking
- Relationship modeling

### Phase 3: Hybrid Forecasting
- TKG-based prediction algorithms
- LLM reasoning chains (Gemini integration)
- RAG-augmented probability generation

### Phase 4: Calibration & Evaluation
- Isotonic and temperature scaling calibration
- Brier score evaluation framework
- Benchmark against historical events

### Phase 5: TKG Training
- GDELT data collection and preprocessing for training
- JAX/jraph RE-GCN model training
- Automated retraining scheduler

### Phase 6-8: Infrastructure Hardening (v1.1)
- NetworkX API compatibility fixes
- Atomic checkpoint/resume bootstrap pipeline
- **Graph partitioning** for >1M event scalability:
  - Temporal-first partitioning (30-day windows)
  - SQLite partition index for O(1) entity lookups
  - LRU cache with memory-aware eviction
  - Scatter-gather cross-partition queries

## Core Principles

**Explainability**: Every forecast provides traceable reasoning paths from evidence to prediction.

## Data Sources

- GDELT 2.0: Real-time global event data
- Focus: QuadClass 1 (verbal cooperation) and 4 (material conflict)
- Quality: GDELT100 filtering (100+ mentions)

## Technical Stack

- **Language**: Python 3.10+
- **Data Access**: gdeltdoc, gdeltPyR
- **Processing**: pandas, dask, pyarrow
- **Knowledge Graphs**: NetworkX (graph operations), GraphML (persistence)
- **TKG Training**: JAX, Flax, jraph (RE-GCN implementation)
- **LLM Integration**: Google GenAI (Gemini)
- **RAG Pipeline**: LlamaIndex, ChromaDB (vector store)
- **Calibration**: scikit-learn (isotonic regression), netcal
- **Storage**: SQLite (events, partition index), GraphML (knowledge graph, partitions), ChromaDB (RAG index)
- **Infrastructure**: Atomic checkpoint/resume bootstrap, temporal graph partitioning with LRU cache