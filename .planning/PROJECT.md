# Explainable Geopolitical Forecasting Engine

## What This Is

An AI-powered geopolitical forecasting engine that combines multiple models to predict political events with clear reasoning paths. The system ingests event data from public APIs and custom sources, processes it through hybrid intelligence algorithms, and produces calibrated probability estimates with explainable reasoning chains. Inspired by systems like IARPA SAGE but optimized for transparency over raw performance.

## Core Value

Explainability — every forecast must provide clear, traceable reasoning paths showing why specific predictions were made.

## Requirements

### Validated

(None yet — ship to validate)

### Active

- [ ] Event data ingestion from GDELT API with custom enrichment pipeline
- [ ] Temporal knowledge graph construction from event streams
- [ ] Hybrid model ensemble combining TKG algorithms (RE-GCN/TiRGN) with LLM reasoning
- [ ] Explainable reasoning chain generation for each prediction
- [ ] Probability calibration system with Brier score optimization
- [ ] Evaluation framework against historical events

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
| Hybrid intelligence over pure ML | Combines strengths of multiple approaches for robustness | — Pending |
| Mixed data approach (APIs + custom) | Balances development speed with unique value creation | — Pending |
| Explainability as core value | Trust and interpretability matter more than raw accuracy | — Pending |
| Batch processing over real-time | Reduces complexity and compute requirements significantly | — Pending |

---
*Last updated: 2026-01-09 after initialization*