# Architecture Research: TGL-LLM Integration

**Domain:** Temporal Knowledge Graph + LLM Deep Integration
**Researched:** 2026-01-31
**Mode:** Architecture dimension of project research
**Overall Confidence:** MEDIUM-HIGH (verified against TGL-LLM paper and existing codebase)

## Executive Summary

The TGL-LLM integration fundamentally transforms the geopol forecasting architecture from a **late-fusion ensemble** (60/40 weighted voting between independent LLM and TKG predictions) to a **deep token-space fusion** where graph embeddings become native input tokens to the LLM decoder.

The existing RE-GCN pipeline (JAX/jraph implementation in `src/training/models/regcn_jraph.py`) produces entity embeddings compatible with the TGL-LLM pattern. The core additions are:
1. **Adapter layers** projecting RE-GCN embeddings (dim=200) to Llama2-7B token space (dim=4096)
2. **Temporal token sequencing** concatenating recent graph states as soft prompts
3. **LoRA fine-tuning** for parameter-efficient LLM adaptation
4. **LLM replacement**: Gemini API -> local Llama2-7B

The RAG pipeline becomes **optional/redundant** for core forecasting (graph embeddings carry the semantic signal), but may be retained for explainability or edge cases.

---

## Component Changes Matrix

| Component | Current State | v2.0 State | Change Type |
|-----------|---------------|------------|-------------|
| `src/forecasting/ensemble_predictor.py` | 60/40 weighted voting | **DEPRECATED** | Remove |
| `src/forecasting/tkg_predictor.py` | RE-GCN wrapper, link prediction | **Adapter input source** | Modify |
| `src/forecasting/reasoning_orchestrator.py` | Multi-step LLM reasoning with Gemini | **Refactor for Llama2** | Major Modify |
| `src/forecasting/rag_pipeline.py` | LlamaIndex + ChromaDB retrieval | **Optional/Explainability** | Demote |
| `src/forecasting/gemini_client.py` | Gemini 3.0 Pro API client | **DEPRECATED** | Remove |
| `src/training/models/regcn_jraph.py` | JAX/jraph RE-GCN | **Encoder backbone** | Minor Modify |
| `src/forecasting/tkg_models/regcn_wrapper.py` | PyTorch RE-GCN wrapper | **Keep for fallback** | No Change |
| `src/forecasting/tkg_models/data_adapter.py` | NetworkX -> quadruples | **Unchanged** | No Change |
| `src/calibration/temperature_scaler.py` | Category-specific calibration | **Unchanged** | No Change |
| **NEW**: `src/forecasting/adapters/entity_adapter.py` | - | 2-layer MLP projection | Create |
| **NEW**: `src/forecasting/adapters/relation_adapter.py` | - | 2-layer MLP projection | Create |
| **NEW**: `src/forecasting/adapters/temporal_tokenizer.py` | - | Temporal token sequencing | Create |
| **NEW**: `src/forecasting/llama_decoder.py` | - | Llama2-7B with LoRA | Create |
| **NEW**: `src/forecasting/tgl_llm_predictor.py` | - | End-to-end TGL-LLM pipeline | Create |
| **NEW**: `src/training/train_adapter.py` | - | Two-stage adapter training | Create |

---

## Data Flow Architecture

### Current v1.0 Architecture (Late Fusion)

```
GDELT Events
    |
    v
+-------------------+
| Knowledge Graph   | (NetworkX MultiDiGraph)
| graph_builder.py  |
+-------------------+
    |
    +------------------+------------------+
    |                                     |
    v                                     v
+-------------------+           +-------------------+
| TKG Predictor     |           | RAG Pipeline      |
| (RE-GCN wrapper)  |           | (LlamaIndex)      |
| link prediction   |           | retrieval         |
+-------------------+           +-------------------+
    |                                     |
    | TKG probability                     | Context strings
    |                                     |
    +----------+   +----------------------+
               |   |
               v   v
         +-------------------+
         | Gemini Client     |
         | (API call)        |
         +-------------------+
               |
               v
         +-------------------+
         | Reasoning         |
         | Orchestrator      |
         | (scenario gen)    |
         +-------------------+
               |
               | LLM probability
               |
    +----------+----------+
    |                     |
    v                     v
+-------------------+
| Ensemble Predictor|
| P = 0.6*LLM +     |
|     0.4*TKG       |
+-------------------+
    |
    v
Final Prediction
```

### Proposed v2.0 Architecture (Deep Fusion)

```
GDELT Events
    |
    v
+-------------------+
| Knowledge Graph   | (NetworkX MultiDiGraph)
| graph_builder.py  |
+-------------------+
    |
    v
+-------------------+
| Data Adapter      | NetworkX -> quadruples (s,r,o,t)
| data_adapter.py   |
+-------------------+
    |
    v
+-----------------------------------+
| RE-GCN Encoder (JAX/jraph)        |
| regcn_jraph.py                    |
|                                   |
|  1. RGCN spatial aggregation      |
|  2. GRU temporal evolution        |
|  3. Output: entity embeddings     |
|     Shape: (num_entities, 200)    |
+-----------------------------------+
    |
    | Recent T timesteps of embeddings
    | {E_{t-T}, ..., E_t}
    |
    v
+-----------------------------------+
| JAX -> NumPy Bridge               | <-- NEW
| Zero-copy via DLPack if possible  |
+-----------------------------------+
    |
    v
+-----------------------------------+
| Adapter Layers (PyTorch)          | <-- NEW
| entity_adapter.py                 |
| relation_adapter.py               |
|                                   |
|  - 2-layer MLP per adapter        |
|  - Projects 200 -> 4096 dim       |
|  - Trainable parameters           |
+-----------------------------------+
    |
    v
+-----------------------------------+
| Temporal Tokenizer                | <-- NEW
| temporal_tokenizer.py             |
|                                   |
|  - Concatenates T graph tokens    |
|  - Interleaves with text tokens   |
|  - Produces hybrid prompt         |
+-----------------------------------+
    |
    | Hybrid prompt: [graph_tokens | text_tokens]
    |
    v
+-----------------------------------+
| Llama2-7B Decoder                 | <-- NEW
| llama_decoder.py                  |
|                                   |
|  - LoRA fine-tuning (frozen base) |
|  - Causal generation              |
|  - Structured output parsing      |
+-----------------------------------+
    |
    v
+-------------------+
| Output Parser     |
| (scenario/prob)   |
+-------------------+
    |
    v
+-------------------+
| Temperature       |
| Calibration       |
| (existing)        |
+-------------------+
    |
    v
Final Prediction
```

---

## New Components Required

### 1. Entity Adapter (`src/forecasting/adapters/entity_adapter.py`)

**Purpose:** Project entity embeddings from graph space (dim=200) to LLM token space (dim=4096).

**Architecture:**
```python
class EntityAdapter(nn.Module):
    """
    Two-layer MLP projecting graph entity embeddings to LLM token space.

    Input:  (batch, 200) - RE-GCN entity embeddings
    Output: (batch, 4096) - Llama2-7B token embeddings
    """
    def __init__(self, graph_dim=200, llm_dim=4096, hidden_dim=1024):
        self.fc1 = nn.Linear(graph_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, llm_dim)
        self.activation = nn.GELU()
        self.layer_norm = nn.LayerNorm(llm_dim)

    def forward(self, entity_emb):
        x = self.activation(self.fc1(entity_emb))
        x = self.fc2(x)
        return self.layer_norm(x)
```

**Critical Notes:**
- Layer normalization essential for stable training
- Hidden dimension (1024) is empirical; tune during training
- Output dimension MUST match Llama2-7B embedding dimension exactly

### 2. Relation Adapter (`src/forecasting/adapters/relation_adapter.py`)

**Purpose:** Project relation embeddings to LLM token space.

**Architecture:** Identical to EntityAdapter. Separate class for clarity and potential future divergence.

### 3. Temporal Tokenizer (`src/forecasting/adapters/temporal_tokenizer.py`)

**Purpose:** Sequence multiple timesteps of graph tokens with text tokens for the hybrid prompt.

**Architecture:**
```python
class TemporalTokenizer:
    """
    Combines graph tokens from T recent timesteps with text tokens.

    Produces prompts of form:
    [<graph_t-T>] [<graph_t-T+1>] ... [<graph_t>] <text_query>
    """
    def __init__(self, entity_adapter, relation_adapter, num_timesteps=5):
        self.entity_adapter = entity_adapter
        self.relation_adapter = relation_adapter
        self.num_timesteps = num_timesteps

    def tokenize(self, entity_embeddings_sequence, query_text, tokenizer):
        """
        Args:
            entity_embeddings_sequence: List of (num_entities, 200) arrays
            query_text: String forecasting question
            tokenizer: Llama tokenizer

        Returns:
            input_ids: Token IDs with graph tokens as special embeddings
            graph_positions: Indices where graph tokens are inserted
            graph_embeddings: The actual graph token embeddings (4096 dim)
        """
        ...
```

**Critical Notes:**
- Must handle variable-length text queries
- Graph token positions tracked for embedding injection during forward pass
- Consider attention masking for temporal ordering

### 4. JAX-PyTorch Bridge (`src/forecasting/adapters/jax_bridge.py`)

**Purpose:** Convert JAX arrays from RE-GCN encoder to PyTorch tensors for adapter layers.

**Architecture:**
```python
import jax
import numpy as np
import torch

def jax_to_torch(jax_array, device='cuda'):
    """
    Convert JAX array to PyTorch tensor via NumPy.

    Zero-copy not guaranteed due to different memory layouts.
    For production: batch conversions, avoid per-inference copies.
    """
    # JAX -> NumPy (may copy)
    np_array = np.asarray(jax_array)
    # NumPy -> PyTorch (shares memory if contiguous)
    return torch.from_numpy(np_array).to(device)
```

**Critical Notes:**
- DLPack zero-copy works but has edge cases with non-contiguous memory
- For production: pre-compute embeddings, store as PyTorch tensors
- Consider caching strategy to avoid repeated conversions

### 5. Llama Decoder (`src/forecasting/llama_decoder.py`)

**Purpose:** Llama2-7B with LoRA fine-tuning for forecasting.

**Architecture:**
```python
from transformers import LlamaForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

class LlamaDecoder:
    def __init__(self, model_path="meta-llama/Llama-2-7b-chat-hf"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = LlamaForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )

        # LoRA configuration
        lora_config = LoraConfig(
            r=16,  # LoRA rank
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        self.model = get_peft_model(self.model, lora_config)

    def generate_with_graph_tokens(self, input_ids, graph_embeddings, graph_positions):
        """
        Generate text with injected graph token embeddings.

        Modifies input embeddings at graph_positions before forward pass.
        """
        ...
```

**Critical Notes:**
- VRAM requirement: ~14GB for Llama2-7B in fp16
- Consider 4-bit quantization (bitsandbytes) for <10GB VRAM
- LoRA rank 16 is conservative; paper suggests it works well

### 6. TGL-LLM Predictor (`src/forecasting/tgl_llm_predictor.py`)

**Purpose:** End-to-end pipeline orchestrating all components.

**Architecture:**
```python
class TGLLLMPredictor:
    """
    End-to-end TGL-LLM forecasting pipeline.

    Combines:
    - RE-GCN encoder (from existing training)
    - Adapter layers (trained)
    - Temporal tokenizer
    - Llama2-7B decoder (LoRA fine-tuned)
    """
    def __init__(self, regcn_checkpoint, adapter_checkpoint, llama_path):
        # Load components
        self.encoder = load_regcn(regcn_checkpoint)  # JAX
        self.entity_adapter = EntityAdapter.load(adapter_checkpoint)
        self.relation_adapter = RelationAdapter.load(adapter_checkpoint)
        self.temporal_tokenizer = TemporalTokenizer(...)
        self.decoder = LlamaDecoder(llama_path)

    def forecast(self, question, graph, num_timesteps=5):
        """
        Generate forecast for question given temporal knowledge graph.
        """
        # 1. Encode graph through RE-GCN (JAX)
        embeddings = self.encoder.evolve_embeddings(graph_snapshots)

        # 2. Convert to PyTorch
        embeddings_pt = jax_to_torch(embeddings[-num_timesteps:])

        # 3. Project through adapters
        graph_tokens = self.entity_adapter(embeddings_pt)

        # 4. Build hybrid prompt
        input_ids, positions, tokens = self.temporal_tokenizer.tokenize(
            graph_tokens, question, self.decoder.tokenizer
        )

        # 5. Generate with Llama
        output = self.decoder.generate_with_graph_tokens(
            input_ids, tokens, positions
        )

        # 6. Parse structured output
        return self.parse_forecast(output)
```

---

## Modified Components

### 1. `src/forecasting/tkg_predictor.py`

**Current Role:** High-level TKG prediction interface.

**v2.0 Role:** Provide embeddings for adapter input.

**Changes:**
```diff
class TKGPredictor:
+   def get_temporal_embeddings(self, graph, num_timesteps=5):
+       """
+       Get entity embeddings for recent timesteps.
+
+       Returns:
+           List of (num_entities, embedding_dim) arrays
+       """
+       # Filter to recent timesteps
+       # Run RE-GCN evolve_embeddings
+       # Return sequence of embedding snapshots
+       ...

    # Existing methods remain for fallback/comparison
    def predict_future_events(self, ...):
        ...
```

### 2. `src/training/models/regcn_jraph.py`

**Current Role:** JAX/jraph RE-GCN training.

**v2.0 Role:** Encoder backbone (frozen during adapter training).

**Changes:**
```diff
class REGCNJraph(nnx.Module):
    def evolve_embeddings(self, graphs, training=True, rng_key=None):
        # Existing implementation
        ...
+       # Return intermediate embeddings per timestep if needed
+       return h, intermediate_embeddings

+   def get_embeddings_sequence(self, graphs, num_timesteps=5):
+       """
+       Get embedding sequence for TGL-LLM adapter.
+       """
+       embeddings = []
+       h = self.entity_emb.value
+
+       for t, graph in enumerate(graphs[-num_timesteps:]):
+           x = self.encode_snapshot(h, graph, training=False)
+           h = self.gru(h, x)
+           embeddings.append(h.copy())
+
+       return embeddings
```

### 3. `src/forecasting/reasoning_orchestrator.py`

**Current Role:** Multi-step LLM reasoning with Gemini.

**v2.0 Role:** **Deprecated for core path.** May be retained for:
- A/B testing against TGL-LLM
- Fallback when Llama unavailable
- Explainability layer

**Changes:** Mark as deprecated, maintain for compatibility.

---

## Deprecated/Optional Components

### Deprecated (Remove or Archive)

| Component | Reason | Action |
|-----------|--------|--------|
| `ensemble_predictor.py` | Late fusion replaced by deep fusion | Archive to `src/forecasting/_deprecated/` |
| `gemini_client.py` | Gemini replaced by local Llama2 | Archive |

### Optional (Retain for Edge Cases)

| Component | New Role | Notes |
|-----------|----------|-------|
| `rag_pipeline.py` | Explainability, edge case retrieval | Keep ChromaDB index, use for "why" explanations |
| `reasoning_orchestrator.py` | A/B testing, fallback | Refactor to accept Llama client as dependency injection |
| `graph_validator.py` | Scenario validation | Still useful for explicit validation checks |

---

## Build Order (Phase Sequence)

Based on component dependencies:

### Phase 1: Foundation (No LLM changes)
1. **JAX-PyTorch bridge** - No dependencies, enables testing
2. **EntityAdapter** - Depends on bridge
3. **RelationAdapter** - Same pattern as EntityAdapter
4. **Unit tests for adapters** - Verify dimension matching

### Phase 2: Encoder Modifications
1. **Modify `regcn_jraph.py`** - Add `get_embeddings_sequence()`
2. **Modify `tkg_predictor.py`** - Add `get_temporal_embeddings()`
3. **Integration test** - Verify embeddings flow through adapters

### Phase 3: Llama Integration
1. **LlamaDecoder** - Standalone Llama2-7B wrapper with LoRA
2. **TemporalTokenizer** - Depends on adapters and LlamaDecoder
3. **Test graph token injection** - Verify embeddings replace placeholder tokens

### Phase 4: End-to-End Pipeline
1. **TGLLLMPredictor** - Orchestrates all components
2. **Output parser** - Extract structured forecasts from Llama output
3. **Integration with existing calibration** - Temperature scaling on outputs

### Phase 5: Training Infrastructure
1. **Dataset preparation** - Format training data for two-stage training
2. **Training script** - Stage 1: high-quality subset, Stage 2: diversity
3. **Evaluation harness** - Compare against v1.0 baseline

### Phase 6: Deprecation and Cleanup
1. **Archive deprecated components**
2. **Update forecasting API** - New default predictor
3. **Documentation update**

---

## Critical Integration Points

### 1. JAX-PyTorch Boundary

**Location:** Between `regcn_jraph.py` (JAX) and adapters (PyTorch)

**Challenge:** Memory copies, device placement, gradient flow

**Solution:**
- For inference: NumPy bridge (no gradients needed)
- For training: Either train adapters only (freeze encoder) or use torch2jax for full backprop
- **Recommendation:** Freeze RE-GCN encoder, train only adapters + LoRA (matches TGL-LLM paper)

### 2. Embedding Dimension Matching

**Graph embeddings:** 200 (current RE-GCN)
**Llama2-7B tokens:** 4096

**Solution:** Adapter layers project 200 -> 4096 with intermediate hidden layer (1024)

### 3. Temporal Alignment

**Challenge:** Graph snapshots may not align with query timestamps

**Solution:**
- Use T most recent snapshots (T=5-7 optimal per TGL-LLM paper)
- No need for exact timestamp matching; temporal ordering sufficient

### 4. Memory Requirements

**Llama2-7B fp16:** ~14GB VRAM
**RE-GCN embeddings:** ~800KB per snapshot (100K entities * 200 dim * 4 bytes)
**Adapters:** ~8MB each

**Recommendation:** 16GB+ GPU for training, 8GB+ for inference with 4-bit quantization

---

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| JAX-PyTorch copy overhead | Inference latency | Pre-compute and cache embeddings |
| LoRA undertrained | Poor forecasting | Two-stage training, adequate epochs |
| Llama2 license constraints | Distribution issues | Use Llama2-7B-chat (permissive) or switch to Llama3 |
| VRAM exhaustion | Cannot run locally | Quantization (4-bit), gradient checkpointing |
| Temporal window too short | Missing patterns | Hyperparameter search on T |

---

## Verification Checklist

Before implementation:
- [ ] Confirm Llama2-7B embedding dimension (4096 expected)
- [ ] Verify RE-GCN output dimension (200 per codebase)
- [ ] Test JAX-NumPy-PyTorch round-trip on sample embeddings
- [ ] Confirm LoRA works with embedding injection

During implementation:
- [ ] Adapter output norm matches Llama token embedding norm
- [ ] No NaN/Inf during initial forward passes
- [ ] Generation quality on sanity check prompts

Post-implementation:
- [ ] A/B test against v1.0 ensemble
- [ ] Calibration curve comparison
- [ ] Latency benchmarks

---

## Sources

- [TGL-LLM Paper](https://arxiv.org/html/2501.11911v1) - Core architecture reference (HIGH confidence)
- [TEA-GLM / Zero-shot Graph LLM](https://arxiv.org/html/2408.14512) - Token alignment techniques (MEDIUM confidence)
- [torch_jax_interop](https://github.com/lebrice/torch_jax_interop) - JAX-PyTorch bridge (HIGH confidence)
- [PyTorch/XLA JAX Bridge](https://pytorch.org/blog/pytorch-xla-2-7-release-usability-vllm-boosts-jax-bridge-gpu-build/) - Experimental XLA integration (LOW confidence - experimental)
- Existing codebase analysis (HIGH confidence)
