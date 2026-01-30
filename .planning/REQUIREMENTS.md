# Requirements: Geopol v2.0 Hybrid Architecture

**Defined:** 2026-01-31
**Core Value:** Explainability — every forecast must provide clear, traceable reasoning paths

## v2.0 Requirements

Requirements for deep token-space integration milestone. Each maps to roadmap phases.

### Deep Integration Core

- [ ] **DEEP-01**: System projects TKG embeddings (200-dim) into Llama2-7B token space (4096-dim) via learned adapter layers
- [ ] **DEEP-02**: System performs temporal tokenization across T=5-7 historical snapshots for temporal context
- [ ] **DEEP-03**: System runs Llama2-7B inference with 4-bit NF4 quantization fitting RTX 3060 12GB VRAM
- [ ] **DEEP-04**: System trains LoRA adapters on frozen Llama2-7B backbone for domain-specific fine-tuning
- [ ] **DEEP-05**: System bridges JAX RE-GCN embeddings to PyTorch Llama via zero-copy transfer (DLPack or NumPy)
- [ ] **DEEP-06**: LLM reasons jointly over graph structure and text, not just post-hoc combination
- [ ] **DEEP-07**: System predicts 42+ CAMEO relation types (multi-class), not just binary outcomes
- [ ] **DEEP-08**: System assesses context quality and adjusts confidence based on graph signal strength

### TKG Encoder

- [ ] **TKG-01**: RE-GCN encoder extracts temporal embedding sequences (T snapshots) for adapter input
- [ ] **TKG-02**: System caches entity embeddings to amortize RE-GCN encoding cost across predictions
- [ ] **TKG-03**: System upgrades TKG algorithm from RE-GCN (40.4% MRR) to HisMatch (~46.4% MRR)
- [ ] **TKG-04**: System adapts temporal window length based on event density in query region

### Validation & Comparison

- [ ] **VAL-01**: System runs v1.1 Gemini and v2.0 Llama-TGL in parallel on same test questions
- [ ] **VAL-02**: System computes accuracy metrics (precision, recall, F1) on held-out geopolitical events
- [ ] **VAL-03**: System computes Brier scores for both v1.1 and v2.0 predictions
- [ ] **VAL-04**: System reports per-relation accuracy breakdown across CAMEO taxonomy
- [ ] **VAL-05**: System generates calibration analysis (ECE, reliability diagrams) for both models
- [ ] **VAL-06**: System benchmarks inference latency (end-to-end prediction time) for both models

### Dashboard

- [ ] **DASH-01**: Streamlit dashboard displays comparison metrics (accuracy, Brier, latency) with charts
- [ ] **DASH-02**: Streamlit dashboard shows live side-by-side predictions from v1.1 and v2.0 models
- [ ] **DASH-03**: Dashboard updates metrics as new predictions are evaluated

### Migration

- [ ] **MIG-01**: Gemini-based v1.1 path remains operational throughout v2.0 development
- [ ] **MIG-02**: System supports manual fallback to v1.1 if v2.0 predictions fail or degrade

## Future Requirements (v2.1+)

Deferred to future milestone. Tracked but not in current roadmap.

### Advanced Features

- **ADV-01**: Automatic failover between models based on confidence threshold
- **ADV-02**: Hybrid mode selecting higher-confidence model per-query
- **ADV-03**: Multi-language support (non-English GDELT sources)
- **ADV-04**: Real-time 15-minute update cycles (vs daily batch)

### Alternative Models

- **ALT-01**: Support for Llama3-8B as alternative backbone
- **ALT-02**: Support for smaller models (Phi-3, TinyLlama) for faster iteration

## Out of Scope

Explicitly excluded. Documented to prevent scope creep.

| Feature | Reason |
|---------|--------|
| Full Llama fine-tuning | Exceeds RTX 3060 12GB VRAM; LoRA sufficient |
| Web production deployment | Research prototype; Streamlit dashboard sufficient |
| Human forecaster integration | Pure AI system per original scope |
| Cross-sector impact modeling | Financial/supply chain out of scope per v1.0 |
| Llama3/larger models | Llama2-7B first; upgrade path is v2.1 |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| DEEP-01 | TBD | Pending |
| DEEP-02 | TBD | Pending |
| DEEP-03 | TBD | Pending |
| DEEP-04 | TBD | Pending |
| DEEP-05 | TBD | Pending |
| DEEP-06 | TBD | Pending |
| DEEP-07 | TBD | Pending |
| DEEP-08 | TBD | Pending |
| TKG-01 | TBD | Pending |
| TKG-02 | TBD | Pending |
| TKG-03 | TBD | Pending |
| TKG-04 | TBD | Pending |
| VAL-01 | TBD | Pending |
| VAL-02 | TBD | Pending |
| VAL-03 | TBD | Pending |
| VAL-04 | TBD | Pending |
| VAL-05 | TBD | Pending |
| VAL-06 | TBD | Pending |
| DASH-01 | TBD | Pending |
| DASH-02 | TBD | Pending |
| DASH-03 | TBD | Pending |
| MIG-01 | TBD | Pending |
| MIG-02 | TBD | Pending |

**Coverage:**
- v2.0 requirements: 23 total
- Mapped to phases: 0 (pending roadmap)
- Unmapped: 23

---
*Requirements defined: 2026-01-31*
*Last updated: 2026-01-31 after initial definition*
