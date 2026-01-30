# Explainable Geopolitical Forecasting Engine

## What This Is

An AI-powered geopolitical forecasting engine that combines multiple models to predict political events with clear reasoning paths. The system ingests event data from public APIs and custom sources, processes it through hybrid intelligence algorithms, and produces calibrated probability estimates with explainable reasoning chains. Inspired by systems like IARPA SAGE but optimized for transparency over raw performance.

## Core Value

Explainability — every forecast must provide clear, traceable reasoning paths showing why specific predictions were made.

## Requirements

### Validated

- Event data ingestion from GDELT API with custom enrichment pipeline — v1.0
- Temporal knowledge graph construction from event streams — v1.0
- Hybrid model ensemble combining TKG algorithms (RE-GCN/TiRGN) with LLM reasoning — v1.0
- Explainable reasoning chain generation for each prediction — v1.0
- Probability calibration system with Brier score optimization — v1.0
- Evaluation framework against historical events — v1.0
- Fix NetworkX shortest_path API to use single_source_shortest_path — v1.1
- Production bootstrap script connecting data ingestion → graph build → RAG indexing — v1.1
- Graph partitioning for scalability beyond 1M events — v1.1

### Active

- Deep token-space integration projecting TKG embeddings into LLM token space (TGL-LLM architecture)
- Adaptive ensemble weighting based on context quality assessment
- TKG algorithm upgrade from RE-GCN to state-of-art (HisMatch or similar)
- Gradual migration path with parallel Gemini/Llama systems for validation
- Benchmark framework comparing v2.0 against v1.1 baseline

## Current Milestone: v2.0 Hybrid Architecture

**Goal:** Replace post-hoc weighted voting with deep graph-language integration where TKG embeddings are projected directly into the LLM's token space, achieving 40-60% accuracy improvement over v1.1.

**Target features:**
- TGL-LLM style adapter architecture (RGCN → GRU → learned adapter → Llama2-7B)
- Self-hosted Llama2-7B inference (4-bit quantized for RTX 3060 12GB)
- Context-quality-aware adaptive weighting (when deep integration unavailable)
- TKG algorithm upgrade (RE-GCN → HisMatch for ~6% MRR gain)
- A/B comparison framework (v1.1 Gemini vs v2.0 Llama-TGL)

**Constraints:**
- RTX 3060 12GB VRAM — adapter training feasible, full fine-tuning not feasible
- Frozen Llama backbone — only adapter layers trained
- Must validate gains before deprecating Gemini path

**Success criteria:** Any measurable accuracy improvement over v1.1 baseline.

### Out of Scope

- Real-time processing — daily batch updates only, not 15-minute cycles
- Human crowdsourcing — pure AI system without human forecaster integration
- Multi-language support — English sources only for v1
- Financial market modeling — no cross-sector impact propagation
- Production deployment — research prototype focused on algorithm development
- User interface — API/CLI only, no web frontend

## Context

Drawing from the comprehensive technical reference in geopol.md, this project implements the forecast engine component of a geopolitical forecasting system. The architecture prioritizes interpretability and CPU-efficiency over state-of-the-art accuracy, making it suitable for research and development on limited compute resources.

Key technical inspirations:
- IARPA SAGE's hybrid architecture (10% improvement over human baselines)
- TKG algorithms like RE-GCN (40.4% MRR) and TiRGN (44.0% MRR)
- Explainable approaches like xERTE with reasoning paths
- Calibration methods from superforecaster research

This is a greenfield implementation starting from first principles rather than extending existing codebases.

## Constraints

- **Compute**: Limited GPU resources — must optimize for CPU-friendly models and avoid large transformer architectures
- **Data volume**: Cannot process full GDELT firehose (500K-1M articles/day) — must use selective sampling
- **Model size**: Prefer smaller models (7B parameters max) over frontier LLMs for cost and latency

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Hybrid intelligence over pure ML | Combines strengths of multiple approaches for robustness | Good |
| Mixed data approach (APIs + custom) | Balances development speed with unique value creation | Good |
| Explainability as core value | Trust and interpretability matter more than raw accuracy | Good |
| Batch processing over real-time | Reduces complexity and compute requirements significantly | Good |
| Python for implementation | Scientific computing ecosystem | Good |
| RE-GCN over TiRGN | More mature implementation available | Good |
| 60/40 LLM/TKG ensemble weighting | Balance reasoning with pattern matching | ⚠️ Revisit — v2.0 targets deep integration |
| JAX/jraph for TKG training | Memory efficiency on CPU | Good |
| Weekly retraining schedule | Captures evolving geopolitical patterns | Good |

## Current State

**Version:** v1.1 (shipped 2026-01-30)

**Tech Stack:**
- Python 3.11+ with uv package management
- PyTorch (CPU-only) for inference
- JAX/jraph for training
- NetworkX for graph operations
- SQLite for event, prediction, and partition index storage
- Gemini API via google-genai SDK

**Codebase:**
- ~100 source files
- 40,257 lines of Python
- 8 phases, 21 plans delivered across 2 milestones

**Known Issues:**
- datetime.utcnow() deprecated in Python 3.12+ (minor, in bootstrap code)

---
*Last updated: 2026-01-31 after v2.0 milestone started*