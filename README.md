# Geopolitical Forecasting Engine

An AI-powered system for processing GDELT events and generating explainable geopolitical forecasts using Temporal Knowledge Graphs and hybrid models.

## Overview

This project implements a geopolitical forecasting engine that:
- Ingests conflict and diplomatic events from GDELT
- Constructs temporal knowledge graphs
- Generates probabilistic forecasts with clear reasoning paths
- Calibrates predictions using Brier scores

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure environment (optional):
```bash
cp .env.example .env
# Edit .env with your settings
```

3. Run the data pipeline:
```bash
python run_pipeline.py --date 2026-01-09
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
- LLM reasoning chains
- Probability generation

### Phase 4: Calibration & Evaluation
- Brier score calibration
- Benchmark evaluation
- Performance metrics

## Core Principles

**Explainability**: Every forecast provides traceable reasoning paths from evidence to prediction.

## Data Sources

- GDELT 2.0: Real-time global event data
- Focus: QuadClass 1 (verbal cooperation) and 4 (material conflict)
- Quality: GDELT100 filtering (100+ mentions)

## Technical Stack

- **Language**: Python 3.10+
- **Data Access**: gdeltdoc, gdeltPyR
- **Processing**: pandas, dask
- **Storage**: SQLite (development), PostgreSQL (production-ready)
- **Scheduling**: schedule (batch processing)