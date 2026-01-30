# Stack Research: TGL-LLM Integration

**Project:** Geopolitical Forecasting Engine - Deep Token-Space Integration
**Researched:** 2026-01-31
**Constraint:** RTX 3060 12GB VRAM
**Overall Confidence:** HIGH (verified via official sources and PyPI)

---

## Executive Summary

Replacing the current 60/40 post-hoc ensemble with deep token-space integration requires:
1. **Self-hosted Llama2-7B** via `transformers` + `bitsandbytes` (4-bit NF4)
2. **Projection layer** mapping 200-dim TKG embeddings to 4096-dim LLM hidden space
3. **PEFT/LoRA** for adapter training without touching base LLM weights
4. **DLPack bridge** for zero-copy JAX->PyTorch tensor conversion

The RTX 3060's 12GB VRAM is sufficient for 4-bit Llama2-7B (~4-5GB weights + ~3GB KV cache + ~2GB projection layers + ~2GB headroom).

---

## Additions Required

### Core LLM Inference

| Package | Version | Purpose | Rationale |
|---------|---------|---------|-----------|
| `transformers` | `>=5.0.0` | Llama2 model loading + inference | v5 is current stable; consolidates tokenizers, weekly releases, 400+ architectures. Native `BitsAndBytesConfig` support. |
| `bitsandbytes` | `>=0.49.0` | 4-bit NF4 quantization | Latest stable (Jan 2026). CUDA 12/13 support, sm86 (RTX 3060/Ampere) verified. |
| `accelerate` | `>=1.12.0` | `device_map="auto"`, big model inference | Required by transformers for quantized loading. Handles layer placement automatically. |

### Adapter Training

| Package | Version | Purpose | Rationale |
|---------|---------|---------|-----------|
| `peft` | `>=0.18.1` | LoRA adapter training | Latest stable (Jan 2026). Python 3.10+ required. Supports hotswapping, trainable token indices. |

### JAX-PyTorch Bridge

| Package | Version | Purpose | Rationale |
|---------|---------|---------|-----------|
| `jax2torch` | `>=0.1.0` | Zero-copy JAX->PyTorch conversion | Uses DLPack under the hood. Avoids CPU round-trip for GPU tensors. |

**Alternative (no extra dependency):** Direct DLPack conversion:
```python
import torch
import jax

def jax_to_torch(x: jax.Array) -> torch.Tensor:
    """Zero-copy JAX array to PyTorch tensor."""
    return torch.utils.dlpack.from_dlpack(jax.dlpack.to_dlpack(x))
```

This avoids adding `jax2torch` if you want minimal dependencies. Both approaches are equivalent.

---

## Quantization Approach

**Recommendation:** NF4 (4-bit NormalFloat) with bfloat16 compute dtype

### VRAM Budget (RTX 3060 12GB)

| Component | VRAM | Notes |
|-----------|------|-------|
| Llama2-7B weights (NF4) | ~3.5-4.0 GB | 7B params @ 4 bits |
| KV cache (2048 context) | ~2.0-2.5 GB | Scales with context length |
| Projection layer | ~0.4 GB | 200 -> 4096, float16 |
| LoRA adapters | ~0.1 GB | r=16, alpha=32 typical |
| Activations/buffers | ~1.5 GB | Generation overhead |
| **Total** | **~8-9 GB** | ~3GB headroom |

### Configuration

```python
from transformers import BitsAndBytesConfig
import torch

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",           # NormalFloat4 - better than FP4
    bnb_4bit_compute_dtype=torch.bfloat16,  # RTX 3060 supports bf16
    bnb_4bit_use_double_quant=True,      # Quantize the quantization constants
)
```

**Why NF4 over GPTQ/AWQ:**
- NF4 is integrated natively into transformers (no separate quantization step)
- GPTQ/AWQ require pre-quantized model files
- For training (QLoRA), NF4 is the standard approach
- GPTQ is slightly better for pure inference, but the difference is marginal

**Confidence:** HIGH - verified via [Hugging Face bitsandbytes docs](https://huggingface.co/docs/transformers/main/en/quantization/bitsandbytes), [VRAM calculators](https://apxml.com/tools/vram-calculator), and [RTX 3060 compatibility reports](https://localllm.in/blog/ollama-vram-requirements-for-local-llms).

---

## Adapter Framework

**Recommendation:** PEFT with LoRA for projection layer training

### Architecture Design (LLaGA-inspired)

The approach follows [LLaGA: Large Language and Graph Assistant](https://arxiv.org/abs/2402.08170) pattern:

1. **TKG Encoder** (existing JAX/jraph RE-GCN) produces entity embeddings: `(num_entities, 200)`
2. **Projection MLP** maps TKG embeddings to LLM token space: `200 -> 4096`
3. **Soft tokens** are injected into the LLM's input embedding sequence
4. **LoRA adapters** on attention layers allow the LLM to learn to use graph context

```
TKG Embeddings (200-dim)
         |
    [Projection MLP]  <-- Trainable
         |
    Soft Tokens (4096-dim, N tokens)
         |
    [Concat with text tokens]
         |
    Llama2-7B (frozen, 4-bit)
         + LoRA adapters (trainable)
         |
    Prediction Output
```

### Projection Layer Design

```python
import torch
import torch.nn as nn

class GraphProjection(nn.Module):
    """Project TKG embeddings into LLM token space."""

    def __init__(
        self,
        tkg_dim: int = 200,      # RE-GCN embedding dimension
        llm_dim: int = 4096,      # Llama2-7B hidden dimension
        num_virtual_tokens: int = 8,  # How many "graph tokens" to inject
    ):
        super().__init__()
        self.num_virtual_tokens = num_virtual_tokens

        # Two-layer MLP following LLaGA pattern
        self.projection = nn.Sequential(
            nn.Linear(tkg_dim, llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim * num_virtual_tokens),
        )

    def forward(self, entity_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            entity_embeddings: (batch, num_entities, tkg_dim)

        Returns:
            virtual_tokens: (batch, num_virtual_tokens, llm_dim)
        """
        # Aggregate entity embeddings (mean pooling)
        aggregated = entity_embeddings.mean(dim=1)  # (batch, tkg_dim)

        # Project to LLM space
        projected = self.projection(aggregated)  # (batch, llm_dim * num_tokens)

        # Reshape to virtual token sequence
        return projected.view(-1, self.num_virtual_tokens, 4096)
```

### LoRA Configuration

```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=16,                          # Low-rank dimension
    lora_alpha=32,                 # Scaling factor
    target_modules=["q_proj", "v_proj"],  # Attention projection layers
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# Apply LoRA to quantized model
model = get_peft_model(quantized_model, lora_config)
```

**Why LoRA targets only q_proj/v_proj:**
- k_proj has minimal impact on downstream performance
- Reduces trainable parameters by 33%
- Standard practice from QLoRA paper

**Confidence:** HIGH - architecture verified via [LLaGA paper](https://arxiv.org/abs/2402.08170), [PEFT docs](https://huggingface.co/docs/peft/en/index), and [transformers integration](https://huggingface.co/docs/transformers/en/peft).

---

## JAX<->PyTorch Bridge

### The Problem

- TKG encoder: JAX/jraph (RE-GCN) produces `jax.Array` on GPU
- LLM: PyTorch (Llama2) consumes `torch.Tensor` on GPU
- Naive: `np.asarray()` -> CPU -> GPU round-trip = slow

### The Solution: DLPack

DLPack is a tensor interchange format. Both JAX and PyTorch support it natively.

```python
import jax
import torch

def jax_to_torch_gpu(x: jax.Array) -> torch.Tensor:
    """Zero-copy conversion for GPU arrays."""
    # Force device synchronization (JAX is async)
    x = jax.device_get(x) if x.device().platform == 'cpu' else x

    # DLPack transfer - no copy if same device
    return torch.utils.dlpack.from_dlpack(jax.dlpack.to_dlpack(x))


def torch_to_jax_gpu(x: torch.Tensor) -> jax.Array:
    """Zero-copy conversion for GPU tensors."""
    return jax.dlpack.from_dlpack(torch.utils.dlpack.to_dlpack(x))
```

### Integration Point

In your pipeline:

```python
# 1. Run TKG encoder (JAX)
entity_embeddings_jax = regcn_model.evolve_embeddings(graphs, training=False)
# Shape: (num_entities, 200), dtype: jax.Array

# 2. Convert to PyTorch (zero-copy)
entity_embeddings_torch = jax_to_torch_gpu(entity_embeddings_jax)
# Shape: (num_entities, 200), dtype: torch.Tensor

# 3. Project to LLM space
virtual_tokens = projection_layer(entity_embeddings_torch.unsqueeze(0))
# Shape: (1, num_virtual_tokens, 4096)

# 4. Inject into Llama2 and generate
...
```

### Caveat: Device Mismatch

If JAX uses one GPU and PyTorch uses another (or CPU), DLPack transfer will fail. Ensure both frameworks target the same CUDA device:

```python
# Force JAX to use same device as PyTorch
jax.config.update("jax_default_device", jax.devices("cuda")[0])

# Force PyTorch to use CUDA:0
torch.cuda.set_device(0)
```

**Confidence:** HIGH - verified via [JAX DLPack docs](https://github.com/jax-ml/jax/discussions/18765), [torch_jax_interop library](https://github.com/lebrice/torch_jax_interop), and [PyTorch forums](https://discuss.pytorch.org/t/convert-jax-array-to-torch-tensor/64079).

---

## Not Recommended

### 1. vLLM

**What it is:** High-performance LLM inference server with PagedAttention.

**Why not:**
- Designed for server deployments with many concurrent users
- Overhead is not justified for single-request forecasting pipeline
- Adds operational complexity (separate server process)
- Your use case: single inference at a time, not throughput-optimized serving

**When to reconsider:** If you need to serve multiple concurrent forecast requests.

### 2. llama.cpp / llama-cpp-python

**What it is:** C++ inference engine with Python bindings. Excellent for CPU inference.

**Why not:**
- You need to train adapter layers (projection + LoRA)
- llama.cpp doesn't integrate with PEFT/LoRA training
- Your existing stack is Python-native (transformers ecosystem)
- The VRAM constraint is solvable with bitsandbytes (no need for GGUF/GGML)

**When to reconsider:** If you need to deploy to edge devices without GPU.

### 3. AWQ/GPTQ Pre-Quantized Models

**What it is:** Weights quantized offline using calibration data.

**Why not:**
- Requires separate quantization step and specific model files
- Can't easily combine with QLoRA training
- NF4 (bitsandbytes) achieves similar quality with simpler workflow
- GPTQ/AWQ is better for pure inference, but you need training capability

**When to reconsider:** If you freeze the architecture and only do inference.

### 4. Custom Fusion Layers in JAX

**What it is:** Rewrite Llama2 in JAX to avoid framework bridge.

**Why not:**
- Massive engineering effort
- Lose access to transformers ecosystem (tokenizers, configs, PEFT)
- bitsandbytes only works with PyTorch
- DLPack bridge is zero-copy and trivial to implement

**When to reconsider:** If you're building a research paper, not a product.

### 5. DeepSpeed ZeRO / FSDP

**What it is:** Distributed training frameworks for sharding model weights.

**Why not:**
- You have 1 GPU, not a cluster
- ZeRO-Offload (CPU offloading) is slower than 4-bit quantization
- Adds significant complexity for no benefit at your scale

**When to reconsider:** If you scale to multi-GPU training.

---

## Integration with Existing Stack

### pyproject.toml Additions

```toml
dependencies = [
    # ... existing dependencies ...

    # TGL-LLM Integration (NEW)
    "transformers>=5.0.0",      # Was not in deps, llama-index uses it internally
    "bitsandbytes>=0.49.0",     # 4-bit quantization
    "accelerate>=1.12.0",       # device_map support
    "peft>=0.18.1",             # LoRA adapter training
]
```

**Note:** `transformers` may already be a transitive dependency via `llama-index`. Pin explicitly to ensure v5.x.

### Conflicts to Watch

| Existing Dep | Potential Conflict | Resolution |
|--------------|-------------------|------------|
| `jax[cuda12]>=0.6.2` | CUDA driver version | Both require CUDA 12.x - no conflict if driver is 535+ |
| `torch>=2.0.0` | CUDA compatibility | Update to `torch>=2.5.0` for CUDA 12.4+ and bf16 support |
| `google-genai>=1.0` | Will be replaced | Keep for fallback, remove when Llama2 integration is stable |

### Architecture Change

**Before (current ensemble_predictor.py):**
```
LLM (Gemini API) --> probability_llm --|
                                       |--> weighted average --> output
TKG (RE-GCN)     --> probability_tkg --|
```

**After (deep integration):**
```
TKG (RE-GCN) --> embeddings --> projection --> soft tokens --|
                                                             |--> Llama2 --> output
Question    --> tokenize ----------------------------------- |
```

The TKG doesn't produce a separate probability - it provides context that the LLM uses for reasoning.

---

## Installation Commands

```bash
# Update torch to 2.5+ with CUDA 12.4
uv add "torch>=2.5.0"

# Add TGL-LLM dependencies
uv add "transformers>=5.0.0" "bitsandbytes>=0.49.0" "accelerate>=1.12.0" "peft>=0.18.1"

# Verify installation
uv run python -c "
import torch
import bitsandbytes as bnb
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'bitsandbytes: {bnb.__version__}')
print(f'CUDA setup: {bnb.cuda_setup.main()}')
"
```

---

## Sources

### Official Documentation (HIGH confidence)
- [Hugging Face Transformers v5](https://huggingface.co/docs/transformers/main/en/index)
- [bitsandbytes Documentation](https://huggingface.co/docs/bitsandbytes/main/en/installation)
- [PEFT Documentation](https://huggingface.co/docs/peft/en/index)
- [Accelerate Big Model Inference](https://huggingface.co/docs/accelerate/concept_guides/big_model_inference)

### PyPI Versions (HIGH confidence)
- [bitsandbytes 0.49.0](https://pypi.org/project/bitsandbytes/) - Jan 8, 2026
- [peft 0.18.1](https://pypi.org/project/peft/) - Jan 9, 2026
- [accelerate 1.12.0](https://pypi.org/project/accelerate/)
- [transformers 5.0.0](https://pypi.org/project/transformers/)

### Architecture References (HIGH confidence)
- [LLaGA: Large Language and Graph Assistant](https://arxiv.org/abs/2402.08170) - ICML 2024
- [Llama 2 Architecture](https://huggingface.co/docs/transformers/model_doc/llama2) - 4096 hidden dim
- [JAX DLPack Interop](https://github.com/jax-ml/jax/discussions/18765)

### VRAM Calculations (MEDIUM confidence - multiple sources agree)
- [VRAM Calculator](https://apxml.com/tools/vram-calculator)
- [Ollama VRAM Requirements](https://localllm.in/blog/ollama-vram-requirements-for-local-llms)
- [RTX 3060 LLM Guide](https://www.bacloud.com/en/blog/163/guide-to-gpu-requirements-for-running-ai-models.html)
