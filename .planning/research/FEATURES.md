# Features Research: Deep Integration Capabilities

**Domain:** TGL-LLM Deep Token-Space Integration for Geopolitical Forecasting
**Researched:** 2026-01-31
**Overall confidence:** MEDIUM (TGL-LLM paper verified, hardware constraints extrapolated)

## Table Stakes

Features that MUST exist for deep integration to function. Without these, you have an ensemble, not deep integration.

| Feature | Why Required | Complexity | Dependencies |
|---------|--------------|------------|--------------|
| **Temporal Graph Adapter** | Projects graph embeddings into LLM token space; without this, graph and language remain separate modalities | High | RE-GCN embeddings (exists), adapter architecture design |
| **Hybrid Graph Tokenization** | Concatenates graph tokens in temporal order as entity descriptions; enables LLM to explore temporal patterns | High | Adapter, vocabulary alignment |
| **Two-Stage Training Pipeline** | Stage 1: Cross-modal alignment. Stage 2: Task-specific fine-tuning. Single-stage fails on alignment | Medium | LoRA infrastructure, training data curation |
| **Frozen LLM Backbone** | Required for 12GB VRAM constraint; full fine-tuning needs ~56GB for Llama2-7B | N/A (constraint) | LoRA or similar PEFT method |
| **Entity/Relation Adapter Layers** | Two-layer perceptrons (EA/RA) that map 200-dim graph embeddings to LLM token space | Medium | Graph embedding dimension match |
| **GRU Temporal Evolution Capture** | Captures evolution between consecutive graph states; RGCN alone is static | Medium | Temporal graph snapshots |

### Table Stakes Rationale

The TGL-LLM paper explicitly identifies that "existing LLM-based methods are limited by insufficient modeling of temporal patterns and ineffective cross-modal alignment between graph and language." The table stakes above directly address these two failure modes. Without the adapter, there's no cross-modal alignment. Without the GRU and hybrid tokenization, temporal patterns are lost.

## Differentiators

Capabilities that deep integration enables **beyond** what v1.1 ensemble can do.

| Feature | Value Proposition | Why v1.1 Cannot Do This | Complexity |
|---------|-------------------|-------------------------|------------|
| **Joint Graph-Language Reasoning** | LLM reasons over graph structure and text simultaneously; discovers patterns neither modality reveals alone | v1.1 makes independent predictions, then averages; no cross-modal reasoning | High |
| **LLM Inspecting Graph Structure** | LLM can attend to specific graph neighborhoods, temporal sequences, and relational patterns | v1.1 LLM only sees text; graph patterns are opaque | High |
| **Temporal Pattern Discovery** | LLM learns to identify recurring temporal motifs in entity interactions | v1.1 TKG captures patterns but cannot communicate them to LLM | High |
| **Explanation Grounding** | Reasoning chains can reference specific graph structures ("based on 5 recent CONFLICT events...") | v1.1 explanations are purely text-based | Medium |
| **Adaptive Confidence Based on Graph Quality** | LLM can recognize when graph signal is weak and adjust confidence | v1.1 uses fixed 60/40 weighting regardless of context quality | Medium |
| **Multi-Hop Relational Reasoning** | LLM can trace chains of relationships across graph structure | v1.1 TKG only scores direct triples; no path reasoning | High |

### Critical Differentiator: Joint Reasoning

The fundamental limitation of v1.1's ensemble approach is that the LLM and TKG operate as **black boxes to each other**:

```
v1.1 Architecture (Late Fusion):
LLM(question) -> P_llm = 0.65
TKG(entity1, relation, entity2) -> P_tkg = 0.72
Ensemble = 0.6 * 0.65 + 0.4 * 0.72 = 0.678
```

The LLM cannot know WHY the TKG produced 0.72. The TKG cannot incorporate LLM reasoning about context. They are information-theoretically isolated.

```
v2.0 Architecture (Early Fusion via TGL-LLM):
graph_tokens = Adapter(RGCN(graph), GRU(temporal_sequence))
LLM(question + graph_tokens) -> P = 0.78 + reasoning_chain
```

The LLM can now attend to graph structure, query specific temporal patterns, and produce reasoning that references both text and graph.

## Expected Performance

Based on TGL-LLM paper and HTKGH benchmarks.

### Quantitative Improvements (from TGL-LLM, arXiv 2501.11911)

| Metric | Pure TKG (RE-GCN) | Pure LLM (CoH) | TGL-LLM | Improvement |
|--------|-------------------|----------------|---------|-------------|
| Acc@4 (POLECAT-IR) | 0.4358 | ~0.50 | **0.8514** | +95% over TKG |
| Acc@10 (POLECAT-IR) | baseline | ~0.55 | **0.7407** | +70% over TKG |

**Caveats:**
- POLECAT datasets are multi-class relation prediction (42 classes), not binary
- These gains are on clean benchmark data with optimal context
- Production gains will be lower due to noise and context quality variance

### Context Quality Impact (from HTKGH, arXiv 2601.00430)

| Context Filtering | LLM Performance vs GNN |
|-------------------|------------------------|
| No filtering | LLMs underperform GNNs |
| Moderate filtering | Roughly equivalent |
| Tight filtering | **LLMs beat GNNs by up to 21%** |

**Implication:** Deep integration amplifies the importance of context quality. With poor context, deep integration may underperform v1.1's ensemble. With good context, it dramatically outperforms.

### Projected v2.0 vs v1.1 Performance

| Scenario | Expected Improvement | Confidence |
|----------|---------------------|------------|
| **Best case** (clean data, optimal context) | +40-60% accuracy | LOW (optimistic) |
| **Realistic case** (production data, moderate context) | +15-25% accuracy | MEDIUM |
| **Degraded case** (noisy data, poor context) | +0-10% or negative | MEDIUM |

**Note:** v1.1's fixed 60/40 weighting is actually robust to noise. Deep integration's performance is more variable. This is why adaptive weighting based on context quality assessment is a critical differentiator.

## Anti-Features

Things to deliberately **NOT** build. These are common mistakes or over-engineering traps.

| Anti-Feature | Why Avoid | What to Do Instead |
|--------------|-----------|-------------------|
| **Full LLM Fine-Tuning** | 56+ GB VRAM required; exceeds RTX 3060 12GB by 4x | Use LoRA/QLoRA with frozen backbone |
| **Real-Time Inference** | 11+ hours training on A40; inference latency adds up | Batch processing with cached embeddings |
| **Maximum Context Window** | Research shows tight filtering beats maximum context; "attention sink" issues | Quality-filtered context (top-k most relevant) |
| **Complex Multi-Stage Pipelines** | TGL-LLM's two-stage is already complex; more stages = more failure modes | Stick to two-stage: alignment then task-specific |
| **End-to-End Differentiable Training** | Memory explosion when backpropagating through graph + LLM | Frozen backbone + adapter-only training |
| **Replacing Gemini with Llama Immediately** | Loss of working system before v2.0 validated | Parallel systems until v2.0 proves gains |
| **Ignoring Text Modality** | Some papers show LLMs with only node text perform well | Maintain text context; graph is additive, not replacement |

### Anti-Feature: Structure as Replacement for Semantics

Research (arXiv 2511.16767) shows:
> "LLMs leveraging only node textual descriptions already achieve strong performance across tasks; and most structural encoding strategies offer marginal or even negative gains."

This is counterintuitive but critical: graph structure enhances semantics, it does not replace it. Deep integration should fuse structure WITH text, not substitute structure FOR text.

## Comparison: v1.1 Ensemble vs v2.0 Deep Integration

### Architecture Comparison

| Aspect | v1.1 Ensemble | v2.0 Deep Integration |
|--------|---------------|----------------------|
| **Fusion Type** | Late fusion (post-hoc combination) | Early fusion (joint input space) |
| **Information Flow** | Unidirectional (models -> ensemble) | Bidirectional (graph tokens <-> LLM attention) |
| **Reasoning Visibility** | LLM/TKG are black boxes to each other | LLM can inspect graph structure |
| **Weighting** | Fixed 60/40 | Context-adaptive |
| **Explainability** | Separate explanations from each model | Unified reasoning chain referencing both |
| **Failure Mode** | Graceful (one fails, other continues) | Coupled (adapter failure blocks both) |

### Capability Matrix

| Capability | v1.1 | v2.0 | Notes |
|------------|------|------|-------|
| Binary prediction | Good | Better | Both work; v2.0 marginal gain |
| Multi-class prediction | Poor | Good | v1.1 averaging destroys class boundaries |
| Temporal pattern detection | TKG only | Joint | LLM can reason about patterns |
| Context-sensitive confidence | No | Yes | v2.0 adapts to context quality |
| Graph structure inspection | No | Yes | LLM attends to graph tokens |
| Explanation grounding | Text only | Text + Graph | v2.0 cites graph evidence |
| Robustness to noise | High | Variable | v1.1's fixed weighting is stable |
| Hardware requirements | Lower | Higher | v2.0 needs adapter training infrastructure |

### When to Prefer Each Approach

**Prefer v1.1 Ensemble When:**
- Data quality is inconsistent
- Explanation requirements are low
- Hardware is limited (inference-only)
- Rapid iteration is needed
- Binary classification is sufficient

**Prefer v2.0 Deep Integration When:**
- Multi-class prediction is required (42+ relation types)
- Explainability must reference graph structure
- Context quality can be controlled
- Hardware supports adapter training (~23GB VRAM)
- Accuracy is the primary objective

## Dependencies on Existing Features

v2.0 deep integration builds on v1.1 infrastructure.

| Existing Feature | How v2.0 Uses It | Modification Needed |
|------------------|------------------|---------------------|
| RE-GCN TKG embeddings | Input to adapter layers | None (use as-is) |
| Temporal graph snapshots | GRU temporal input | May need finer granularity |
| GDELT ingestion pipeline | Source data | None |
| Entity/relation vocabulary | Adapter vocabulary alignment | Export entity2id, relation2id |
| Gemini LLM path | Parallel validation during development | Keep operational until v2.0 validated |
| Probability calibration | Post-hoc calibration of v2.0 outputs | Adapt to single-model output |
| RAG pipeline | Historical context retrieval | Quality filtering critical for v2.0 |

## Hardware Constraints Analysis

### TGL-LLM Reference (RTX A40)
- **VRAM Used:** 23.08 GB
- **Training Time:** 11.26 hours
- **Model:** Llama2-7B-chat

### RTX 3060 12GB Feasibility

| Component | Memory | Feasible? |
|-----------|--------|-----------|
| Llama2-7B (INT4 quantized) | ~3.5 GB | Yes |
| KV Cache (2K context) | ~0.5 GB | Yes |
| RE-GCN graph embeddings | ~0.5 GB | Yes |
| Adapter layers (2-layer MLPs) | ~0.1 GB | Yes |
| Training activations (LoRA) | ~4-5 GB | Yes |
| **Total** | ~9-10 GB | **Yes, with margin** |

**Critical Constraint:** The 23GB reported by TGL-LLM assumes FP16 and larger context. With INT4 quantization and constrained context, 12GB is achievable for training. Inference is comfortable.

## Sources

### Primary References (HIGH confidence)
- [TGL-LLM: Integrate Temporal Graph Learning into LLM-based Temporal Knowledge Graph Model](https://arxiv.org/abs/2501.11911) - Core architecture, performance metrics
- [HTKGH: Toward Better Temporal Structures for Geopolitical Events Forecasting](https://arxiv.org/abs/2601.00430) - Context filtering impact, LLM benchmarks

### Supporting References (MEDIUM confidence)
- [Frontiers: Practices, opportunities and challenges in the fusion of knowledge graphs and large language models](https://www.frontiersin.org/journals/computer-science/articles/10.3389/fcomp.2025.1590632/full) - Fusion strategies taxonomy
- [Attention Mechanisms Perspective: LLM Processing of Graph-Structured Data](https://arxiv.org/html/2505.02130v1) - Attention sink issues
- [When Structure Doesn't Help: LLMs Do Not Read Text-Attributed Graphs as Effectively as We Expected](https://arxiv.org/html/2511.16767) - Structure vs semantics findings

### VRAM/Hardware References (MEDIUM confidence)
- [LLM GPU VRAM Requirements Explained 2025](https://www.propelrc.com/llm-gpu-vram-requirements-explained/) - Memory calculations
- [A Guide to vRAM requirements for fine-tuning LLM & AI models](https://dataoorts.com/vram-requirements-for-fine-tuning-llms-ai-models/) - LoRA training estimates

## Gaps and Open Questions

1. **Context Quality Assessment:** How to automatically assess context quality to trigger adaptive weighting? No clear methodology found.

2. **Attention Sink Mitigation:** Research shows LLMs struggle with graph structure attention. How does TGL-LLM's hybrid tokenization address this?

3. **Negative Transfer Risk:** If graph signal is noisy, deep integration may perform worse than v1.1. What's the fallback strategy?

4. **Calibration Impact:** v1.1's temperature scaling is tuned for ensemble outputs. v2.0 produces different confidence distributions. Recalibration needed.

5. **INT4 vs FP16 Quality:** TGL-LLM used FP16. Does INT4 quantization degrade adapter alignment quality?
