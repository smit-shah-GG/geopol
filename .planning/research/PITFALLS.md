# Pitfalls Research: TGL-LLM Integration

**Domain:** Deep LLM-TKG integration for geopolitical forecasting
**Researched:** 2026-01-31
**Confidence:** MEDIUM (verified against TGL-LLM paper, JAX/PyTorch docs, adapter training literature)

## Executive Summary

Integrating TGL-LLM style deep token-space alignment into the existing geopol system faces five categories of pitfalls:

1. **Critical blockers** that will halt progress entirely (VRAM exhaustion, framework memory conflicts)
2. **Quality degraders** that produce a working but inferior system (cross-modal alignment collapse, calibration regression)
3. **Integration traps** at system boundaries (JAX-PyTorch interop, GDELT-POLECAT domain shift)
4. **Resource constraints** specific to RTX 3060 12GB (batch size limits, quantization quality loss)
5. **Adapter training pitfalls** that cause unstable or overfit models

The most dangerous pitfalls are those that appear to work but silently degrade forecasting quality, particularly cross-modal alignment failures and calibration regression.

---

## Critical Pitfalls (Block Progress)

### CP-1: JAX/PyTorch GPU Memory Pre-allocation Conflict

**What goes wrong:** JAX pre-allocates 75% of GPU memory on first operation. PyTorch uses lazy caching allocation. When both frameworks run in the same process, they fight for VRAM and cause OOM even when total model size fits.

**Why it happens:** Your existing system uses JAX/jraph for TKG training (`regcn_jraph.py`) and PyTorch for inference (`tkg_predictor.py` loads via `torch.load`). The TGL-LLM integration requires PyTorch transformers for the Llama backbone. Running both in the same process without memory coordination guarantees failure on 12GB VRAM.

**Consequences:**
- OOM errors that appear random (depend on operation order)
- Silent memory corruption leading to NaN losses
- Inability to run end-to-end pipeline

**Prevention:**
1. Set `XLA_PYTHON_CLIENT_PREALLOCATE='false'` and `XLA_PYTHON_CLIENT_MEM_FRACTION='0.5'` before any JAX import
2. **Process isolation**: Run JAX graph encoder and PyTorch LLM in separate processes with explicit tensor serialization
3. Use `jax_to_torch` from `torch_jax_interop` with DLPack zero-copy when possible, but verify tensor layouts

**Warning signs:**
- `nvidia-smi` shows 10GB+ allocated before model loading
- OOM errors that only occur in certain import orders
- Training works in isolation but fails in pipeline

**Detection:** Monitor GPU memory allocation per-framework before model loading begins.

**Phase:** Must address in Phase 1 (Environment Setup) before any model integration work.

**Severity:** BLOCKS PROGRESS

**Sources:**
- [JAX GPU Memory Allocation Docs](https://docs.jax.dev/en/latest/gpu_memory_allocation.html)
- [torch_jax_interop GitHub](https://github.com/lebrice/torch_jax_interop)
- [TorchAX Memory Issues](https://github.com/google/torchax/issues/16)

---

### CP-2: VRAM Exhaustion During Two-Stage Training

**What goes wrong:** TGL-LLM's two-stage training requires ~23GB VRAM on A40 (from paper). RTX 3060 has 12GB. Naive implementation will OOM during Stage 1 fine-tuning.

**Why it happens:** Stage 1 fine-tunes LLM with LoRA on 100K high-quality samples. Even with LoRA, Llama2-7B in FP16 requires ~14GB for weights alone. Activations, optimizer states, and graph adapter embeddings push this higher.

**Consequences:**
- Training cannot start or crashes mid-batch
- Forced to reduce batch size to 1, destroying training dynamics
- Model cannot learn cross-modal alignment with insufficient batching

**Prevention:**
1. **Aggressive quantization**: QLoRA with 4-bit base model reduces to ~3.5GB for weights
2. **Gradient checkpointing**: Trade 20% speed for 40% memory reduction
3. **Gradient accumulation**: Simulate batch size 128 with micro-batches of 8
4. **Model choice**: Consider Llama2-7B-chat (matches TGL-LLM paper) or smaller Phi-3/Qwen2 models
5. **Offloading**: Use `accelerate` library for CPU offloading during backward pass

**Required configuration:**
```python
# Minimum viable config for 12GB VRAM
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)
# Enable gradient checkpointing
model.gradient_checkpointing_enable()
# Accumulation to simulate batch 128
gradient_accumulation_steps = 16  # with micro_batch=8
```

**Warning signs:**
- OOM during first backward pass
- Loss goes to NaN after a few steps (gradient explosion from tiny batches)
- Training completes but validation accuracy is random

**Detection:** Profile memory with `torch.cuda.memory_summary()` before and after each stage.

**Phase:** Must solve in Phase 2 (Adapter Architecture) before training begins.

**Severity:** BLOCKS PROGRESS

**Sources:**
- [TGL-LLM Paper - Hardware Requirements](https://arxiv.org/html/2501.11911v1)
- [Gradient Checkpointing Guide](https://medium.com/mlworks/gradient-checkpointing-the-unsung-hero-of-llm-training-ac2bbe5d4396)
- [QLoRA Memory Guide](https://alain-airom.medium.com/run-big-llms-on-small-gpus-a-hands-on-guide-to-4-bit-quantization-and-qlora-40e9e2c95054)
- [RTX 3060 LLM Benchmarks](https://www.databasemart.com/blog/ollama-gpu-benchmark-rtx3060ti)

---

### CP-3: Graph Adapter Projection Dimension Mismatch

**What goes wrong:** TGL-LLM uses 200-dim graph embeddings projected to Llama's 4096-dim token space via 2-layer MLP. Incorrect projection dimensions cause shape errors or silent performance degradation.

**Why it happens:** Your existing RE-GCN uses `embedding_dim=200` (matches TGL-LLM). But the adapter MLP must project 200 -> 4096 for Llama2-7B. If you use a different backbone (Phi-3: 3072, Qwen2-7B: 3584), dimensions change.

**Consequences:**
- Runtime shape mismatch errors
- If dimensions "happen to work" through broadcasting, catastrophically wrong representations
- Adapter trains but graph tokens are meaningless to LLM

**Prevention:**
1. **Explicit dimension config**: Create dataclass with all dimension parameters locked
2. **Shape assertions**: Add `assert emb.shape[-1] == self.llm_hidden_dim` at every projection boundary
3. **Test with dummy data**: Verify tensor shapes through full forward pass before training

```python
@dataclass
class AdapterConfig:
    graph_emb_dim: int = 200  # From RE-GCN
    llm_hidden_dim: int = 4096  # Llama2-7B
    adapter_hidden_dim: int = 512  # Intermediate
    num_adapter_layers: int = 2

    def __post_init__(self):
        # Validate against model
        assert self.llm_hidden_dim in [3072, 3584, 4096], "Unknown LLM hidden dim"
```

**Warning signs:**
- `RuntimeError: mat1 and mat2 shapes cannot be multiplied`
- Loss starts very high and never decreases
- Adapter weights converge to near-zero

**Detection:** Print tensor shapes at every layer boundary during first forward pass.

**Phase:** Phase 2 (Adapter Architecture) - must lock before training.

**Severity:** BLOCKS PROGRESS (or silent quality degradation if dimensions accidentally broadcast)

**Sources:**
- [TGL-LLM Architecture Details](https://arxiv.org/html/2501.11911v1)
- [GraphAdapter Paper](https://arxiv.org/html/2402.12984v1)

---

## Quality Pitfalls (Degrade Results)

### QP-1: Cross-Modal Alignment Collapse

**What goes wrong:** The graph adapter learns to project all graph embeddings to a narrow region of token space, causing "mode collapse" where LLM treats all graph tokens as semantically identical.

**Why it happens:** Two-stage training is designed to prevent this, but:
- If Stage 1 quality subset is poorly selected, adapter learns spurious correlations
- If influence function approximation is inaccurate, "high quality" samples are actually low quality
- Imbalanced relation distribution (your GDELT data is heavily skewed toward `10_Q1` - Make Statement) causes adapter to overfit to common patterns

**Consequences:**
- LLM ignores graph tokens entirely (they add noise, not signal)
- Predictions collapse to LLM's text-only reasoning
- Accuracy drops below non-TGL baseline (your current Gemini API system)

**Prevention:**
1. **Monitor adapter output variance**: Track std of projected graph tokens during training
2. **Diversity sampling in Stage 2**: TGL-LLM uses stratified sampling - implement this carefully
3. **Relation balancing**: Undersample frequent relations (`10_Q1`, `42_Q1`) in training data
4. **Intermediate evaluation**: Check if model performance on graph-heavy queries improves

**Warning signs:**
- Projected graph token embeddings cluster tightly (cosine similarity > 0.95)
- Validation accuracy improves then plateaus very early
- Removing graph tokens doesn't change predictions

**Detection:**
```python
def check_alignment_collapse(adapter, graph_embeddings):
    projected = adapter(graph_embeddings)
    cos_sim = F.cosine_similarity(projected.unsqueeze(1), projected.unsqueeze(0), dim=-1)
    if cos_sim.mean() > 0.9:
        logging.warning("ALIGNMENT COLLAPSE DETECTED")
```

**Phase:** Phase 3 (Training Pipeline) - monitor throughout training.

**Severity:** DEGRADES QUALITY (system works but worse than baseline)

**Sources:**
- [TGL-LLM Training Details](https://arxiv.org/html/2501.11911v1)
- [Cross-Modal Alignment in VLMs](https://openreview.net/forum?id=uQEsLZU15E)

---

### QP-2: Calibration Regression from Deep Integration

**What goes wrong:** Your v1.1 system has calibrated probability outputs (isotonic calibration, temperature scaling). Deep LLM integration produces logits/embeddings, not calibrated probabilities. Integration destroys existing calibration.

**Why it happens:**
- TGL-LLM outputs MCQ accuracy, not probabilistic forecasts
- LLM confidence scores are notoriously miscalibrated
- Your existing `IsotonicCalibrator` and `TemperatureScaler` expect specific input distributions

**Consequences:**
- Brier scores get worse even if accuracy improves
- Users receive overconfident predictions
- Downstream decision systems (if any) make poor choices

**Prevention:**
1. **Separate calibration stage**: Re-calibrate after TGL-LLM integration, don't assume transfer
2. **Track ECE/Brier during development**: Don't just track accuracy
3. **Preserve baseline comparison**: Always benchmark against current Gemini system
4. **Design for probability output**: Add temperature scaling layer after TGL-LLM

```python
# After TGL-LLM logits
raw_logits = tgl_llm(query)
calibrated_probs = temperature_scaler(raw_logits)  # Re-trained post-integration
```

**Warning signs:**
- Accuracy improves but Brier score worsens
- Confidence histogram shows extreme U-shape (all predictions near 0 or 1)
- ECE > 0.15 on validation set

**Detection:** Plot reliability diagram after each training checkpoint.

**Phase:** Phase 4 (Evaluation & Calibration) - dedicated phase required.

**Severity:** DEGRADES QUALITY (predictions work but untrustworthy)

**Sources:**
- [TGL-LLM Evaluation Section](https://arxiv.org/html/2501.11911v1) (uses Acc@K, not Brier)
- [LLM Calibration Research](https://arxiv.org/html/2505.18697)

---

### QP-3: 4-Bit Quantization Quality Degradation on Reasoning Tasks

**What goes wrong:** 4-bit quantization required for 12GB VRAM disproportionately degrades reasoning and chain-of-thought performance compared to 8-bit or FP16.

**Why it happens:** Geopolitical forecasting requires multi-hop reasoning (entity -> relation -> consequence chains). Research shows 4-bit quantization has "noticeable degradation" on Chain-of-Thought tasks compared to 8-bit.

**Consequences:**
- Model appears to work but makes subtle logical errors
- Complex multi-step predictions fail more often
- Temporal reasoning (your core use case) particularly affected

**Prevention:**
1. **Use 8-bit for inference**: Only use 4-bit during training (QLoRA), merge and quantize to 8-bit for inference
2. **Test reasoning explicitly**: Create test set of multi-hop reasoning queries
3. **Consider smaller FP16 models**: Phi-3-mini (3.8B) in FP16 may outperform Llama2-7B in 4-bit
4. **Hybrid precision**: Keep attention layers in higher precision

**Warning signs:**
- High accuracy on simple queries, low on complex ones
- Model gives correct intermediate steps but wrong final answer
- Longer context queries have disproportionately higher error

**Detection:** Create stratified test set by reasoning depth (1-hop, 2-hop, 3-hop).

**Phase:** Phase 2 (Adapter Architecture) for quantization choice, Phase 4 for evaluation.

**Severity:** DEGRADES QUALITY (subtle, hard to detect)

**Sources:**
- [LLaMA3 Quantization Study](https://link.springer.com/article/10.1007/s44267-024-00070-x)
- [Quantization Quality Comparison](https://www.lesswrong.com/posts/qmPXQbyYA66DuJbht/comparing-quantized-performance-in-llama-models)

---

### QP-4: Historical Window Length Mismatch

**What goes wrong:** TGL-LLM uses T=5-7 historical timesteps. Your system uses 30-day history (`history_length=30` in `TKGPredictor`). Mismatch causes either information loss or noise injection.

**Why it happens:** TGL-LLM's optimal T was tuned for POLECAT's temporal density. GDELT has different event density per time unit. Blindly using T=5 may capture too little context; T=30 introduces "noisy information" (per TGL-LLM ablation).

**Consequences:**
- Model either misses important historical context or gets confused by irrelevant old events
- Temporal attention patterns learn wrong dependencies
- Worse than single-snapshot baseline

**Prevention:**
1. **Analyze GDELT temporal density**: Count events per day in your data
2. **Ablation study on T**: Test T in [3, 5, 7, 10, 14] before full training
3. **Adaptive windowing**: Use event count, not fixed days
4. **Temporal decay in embeddings**: Weight older events lower (you already have `decay_rate=0.95`)

**Warning signs:**
- Performance drops when increasing T beyond certain point
- Attention weights on old timesteps are near-zero
- Better accuracy with shorter history

**Detection:** Log attention weights per timestep, plot average attention vs. time.

**Phase:** Phase 3 (Training Pipeline) - hyperparameter search required.

**Severity:** DEGRADES QUALITY

**Sources:**
- [TGL-LLM Ablation on T](https://arxiv.org/html/2501.11911v1) - "deeper graph representations...are challenging for LLMs"

---

## Integration Pitfalls (JAX-PyTorch, GDELT-POLECAT)

### IP-1: GDELT to POLECAT Domain Shift

**What goes wrong:** TGL-LLM was trained/evaluated on POLECAT (80 relations, 25K-34K entities per country subset). GDELT uses CAMEO codes (20 root codes, 300+ subcodes) with different entity normalization. Direct transfer fails.

**Why it happens:**
- POLECAT is ICEWS successor with LLM-based entity extraction
- GDELT uses automated TABARI/PETRARCH extraction with different entity resolution
- Relation taxonomies don't map 1:1
- Entity naming conventions differ (POLECAT: normalized, GDELT: as-extracted from news)

**Consequences:**
- Pretrained TGL-LLM adapters don't transfer to GDELT
- Entity vocabulary has massive OOV rate
- Relation distribution shift causes adapter confusion

**Prevention:**
1. **Relation mapping layer**: Create CAMEO -> POLECAT relation mapping (or vice versa)
2. **Entity normalization**: Your existing `ENTITY_ALIASES` dict needs expansion
3. **Domain adaptation fine-tuning**: Treat GDELT as target domain, fine-tune adapter
4. **Don't use pretrained TGL-LLM directly**: Train from scratch on GDELT with TGL-LLM architecture

```python
# CAMEO to semantic category mapping (already partially in your code)
CAMEO_TO_POLECAT_CATEGORY = {
    "10": "STATEMENT",  # Make public statement
    "11": "STATEMENT",  # Decline comment
    ...
    "19": "CONFLICT",   # Use military force
    "20": "DIPLOMACY",  # Appeal
}
```

**Warning signs:**
- High OOV rate for entities (>30%)
- Relation distribution in training very different from POLECAT
- Model performs well on POLECAT-like test cases, poorly on GDELT-specific ones

**Detection:** Compute entity/relation overlap statistics between your data and POLECAT.

**Phase:** Phase 1 (Data Preparation) - must resolve before training.

**Severity:** DEGRADES QUALITY or BLOCKS PROGRESS (if OOV rate too high)

**Sources:**
- [POLECAT Dataset Description](https://arxiv.org/html/2501.11911v1)
- [DAEA: Entity Alignment with Domain Adaptation](https://aclanthology.org/2025.coling-main.393.pdf)
- [Lifelong KG Embedding for OOV](https://arxiv.org/abs/2211.15845)

---

### IP-2: JAX→PyTorch Tensor Layout Incompatibility

**What goes wrong:** JAX arrays and PyTorch tensors can have different memory layouts. DLPack zero-copy fails silently on non-contiguous tensors, causing either errors or wrong values.

**Why it happens:**
- JAX: row-major by default, but XLA may optimize to different layouts
- PyTorch: row-major but with potential striding from views
- Channels-first image tensors (not your case, but illustrative) cause JAX to refuse DLPack

**Consequences:**
- Graph embeddings from JAX RE-GCN arrive at PyTorch adapter with wrong values
- Gradients don't flow correctly through interop boundary
- Training appears to work but adapter learns garbage

**Prevention:**
1. **Always call `.contiguous()` on PyTorch side after conversion**
2. **Verify values after conversion**: Compare sum/mean/max of original and converted
3. **Explicit dtype matching**: Ensure both frameworks use same precision
4. **Serialize through numpy for debugging**: Slow but correct

```python
# Safe conversion pattern
def jax_to_torch_safe(jax_array):
    # Ensure contiguous in JAX
    jax_array = jnp.ascontiguousarray(jax_array)
    # Convert via DLPack
    torch_tensor = torch.from_dlpack(jax.dlpack.to_dlpack(jax_array))
    # Ensure contiguous in PyTorch
    return torch_tensor.contiguous()
```

**Warning signs:**
- NaN or Inf appearing after framework boundary
- Tensor values differ from expected (check with print)
- Training loss oscillates wildly

**Detection:** Add assertion `assert torch.isfinite(tensor).all()` after every conversion.

**Phase:** Phase 2 (Adapter Architecture) - must verify before training.

**Severity:** BLOCKS PROGRESS (if values wrong) or DEGRADES QUALITY (if subtle corruption)

**Sources:**
- [NVIDIA Interoperability Blog](https://developer.nvidia.com/blog/machine-learning-frameworks-interoperability-part-1-memory-layouts-and-memory-pools/)
- [torch_jax_interop Layout Issues](https://github.com/lebrice/torch_jax_interop)

---

### IP-3: Frozen Backbone Gradient Leakage

**What goes wrong:** Intending to freeze LLM backbone but gradients still flow through it, either updating weights unintentionally or causing OOM from storing full activation graph.

**Why it happens:**
- Forgetting to call `requires_grad_(False)` on backbone parameters
- Using `model.eval()` (affects dropout/batchnorm, NOT gradient computation)
- Adapter attached in a way that creates gradient path through frozen layers

**Consequences:**
- OOM from storing activations for backbone backward pass
- Backbone weights shift, causing catastrophic forgetting
- Training is 10x slower than expected

**Prevention:**
1. **Explicit parameter freezing**:
```python
for param in llm.parameters():
    param.requires_grad = False
# Verify
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
```
2. **Use `torch.no_grad()` context for backbone forward when possible**
3. **Check gradient checkpointing compatibility**: Some implementations break with frozen layers

**Warning signs:**
- Memory usage much higher than expected during backward
- Training much slower than LoRA benchmarks suggest
- Backbone weights change between epochs

**Detection:** Save backbone weights before training, compare after first epoch.

**Phase:** Phase 2 (Adapter Architecture) - verify freeze before training.

**Severity:** BLOCKS PROGRESS (OOM) or DEGRADES QUALITY (forgetting)

**Sources:**
- [Practical LoRA Tips](https://magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms)
- [Frozen Backbone Best Practices](https://medium.com/data-and-beyond/dont-freeze-your-pretrained-backbone-follow-these-3-steps-instead-93d3d16ceadb)

---

## Resource Pitfalls (VRAM, Training Time)

### RP-1: Training Time Explosion on RTX 3060

**What goes wrong:** TGL-LLM reports ~10h training on A40 (40GB, 86 TFLOPS). RTX 3060 has 12GB and 13 TFLOPS. Naive scaling suggests 60-80 hours. Actual time may be worse due to memory bottlenecks.

**Why it happens:**
- A40 can run larger batches, better GPU utilization
- RTX 3060 has lower memory bandwidth (360 GB/s vs 696 GB/s)
- Gradient accumulation adds overhead
- Gradient checkpointing adds 20% compute

**Consequences:**
- Development iteration cycle becomes days instead of hours
- Hyperparameter search infeasible
- Deadline slip

**Prevention:**
1. **Smaller model**: Consider Phi-3-mini or TinyLlama for rapid iteration
2. **Dataset subsampling**: Train on 10% data first for architecture validation
3. **Cloud burst for final training**: Use RunPod/Lambda for final model with full data
4. **Mixed precision**: Ensure `torch.cuda.amp` is enabled

**Estimated times (RTX 3060):**
| Configuration | Estimated Time |
|--------------|---------------|
| Llama2-7B, full GDELT, QLoRA | 80-100 hours |
| Llama2-7B, 10% data, QLoRA | 8-10 hours |
| Phi-3-mini, full data, LoRA | 20-30 hours |
| TinyLlama, full data, LoRA | 10-15 hours |

**Warning signs:**
- First epoch takes >2 hours
- GPU utilization <50% (memory bottleneck)
- Frequent CUDA synchronization stalls

**Detection:** Profile with `torch.profiler` or `nvidia-nsys`.

**Phase:** Phase 1 (Environment Setup) - set realistic expectations.

**Severity:** BLOCKS PROGRESS (if timeline unrealistic)

**Sources:**
- [TGL-LLM Training Time](https://arxiv.org/html/2501.11911v1)
- [RTX 3060 LLM Guide](https://www.thinkmasters.com/blog/item/62-proxmox-based-vm-with-nvidia-rtx-3060-for-llm-inference-benchmarking)

---

### RP-2: Inference Latency Regression

**What goes wrong:** Current Gemini API gives ~2s response time. Self-hosted Llama2-7B on RTX 3060 gives 7-10 tokens/second. Graph encoding adds more latency. Total inference becomes 10-30s per query.

**Why it happens:**
- Llama2-7B in 4-bit still slower than API calls to cloud TPUs
- Graph encoder (RE-GCN) forward pass adds ~100ms
- No batching in single-query use case

**Consequences:**
- User experience degradation (if interactive)
- Throughput drops for batch forecasting
- Pipeline bottleneck shifts to inference

**Prevention:**
1. **Accept latency trade-off**: Document expected latency, set user expectations
2. **Speculative decoding**: If using vLLM, enable for ~2x speedup
3. **Cache graph embeddings**: Pre-compute and cache for common entities
4. **Batch queries**: Process multiple queries together when possible

**Warning signs:**
- P95 latency >30s
- GPU utilization spikes then drops (waiting for memory)
- Throughput <10 queries/minute

**Detection:** Benchmark end-to-end latency before and after integration.

**Phase:** Phase 5 (Production Optimization) - addressed after correctness verified.

**Severity:** DEGRADES QUALITY (of user experience, not predictions)

**Sources:**
- [RTX 3060 Token Speeds](https://www.databasemart.com/blog/ollama-gpu-benchmark-rtx3060ti)

---

## Adapter Training Pitfalls

### AT-1: LoRA Rank Selection Failure

**What goes wrong:** LoRA rank too low = adapter can't learn complex cross-modal alignment. Rank too high = overfitting, loses parameter efficiency benefit.

**Why it happens:** TGL-LLM paper doesn't specify LoRA rank. Default rank=8 may be insufficient for graph-to-language projection. Rank=64+ approaches full fine-tuning cost.

**Consequences:**
- Low rank: adapter converges but performance plateaus early
- High rank: adapter overfits, validation loss increases after few epochs

**Prevention:**
1. **Start with rank=16**, common for cross-modal alignment
2. **Apply to all linear layers**: Not just Q/V matrices (per 2025 research)
3. **Use alpha = 2 * rank** as scaling factor
4. **Monitor validation loss**: If plateauing with low rank, increase

```python
peft_config = LoraConfig(
    r=16,  # Start here
    lora_alpha=32,  # 2x rank
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
)
```

**Warning signs:**
- Training loss decreases but validation stagnates (too low rank)
- Validation loss increases after epoch 2-3 (too high rank / overfitting)
- Trainable parameter count >1% of model (rank too high)

**Detection:** Run rank sweep [4, 8, 16, 32] on small data subset.

**Phase:** Phase 3 (Training Pipeline) - hyperparameter search.

**Severity:** DEGRADES QUALITY

**Sources:**
- [Practical LoRA Tips](https://magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms)
- [LoRA Rank Selection](https://www.databricks.com/blog/efficient-fine-tuning-lora-guide-llms)

---

### AT-2: Over-Memorization on Small High-Quality Subset

**What goes wrong:** Stage 1 trains on 100K high-quality samples (per TGL-LLM). If your GDELT filtered dataset is smaller, or quality selection is poor, adapter memorizes rather than generalizes.

**Why it happens:**
- "Over-memorization" differs from overfitting: model memorizes training data verbatim
- With <1K samples, memorization is guaranteed
- 2025 research shows this causes "high test perplexity while maintaining good test accuracy" (misleading metrics)

**Consequences:**
- Model reproduces training examples exactly but fails on novel queries
- Appears to work in development, fails in production
- Memorization masks cross-modal alignment failure

**Prevention:**
1. **Minimum 1K samples per task**: TGL-LLM uses 100K, scale down proportionally
2. **Data augmentation**: Paraphrase entity names, shuffle temporal order
3. **Regularization**: Dropout in adapter layers, weight decay
4. **Early stopping on novel validation set**: Don't use training distribution for validation

```python
# Augmentation example
def augment_quadruple(quad):
    s, r, o, t = quad
    # Swap subject/object for symmetric relations
    if relation_is_symmetric(r):
        if random.random() < 0.5:
            s, o = o, s
    # Add temporal noise
    t = t + random.randint(-1, 1)
    return (s, r, o, t)
```

**Warning signs:**
- Training accuracy >99%, validation accuracy <80%
- Model outputs exact training example text verbatim
- Novel entity combinations always fail

**Detection:** Inject synthetic test cases not in training data, check failure rate.

**Phase:** Phase 3 (Training Pipeline) - data preparation and monitoring.

**Severity:** DEGRADES QUALITY (appears to work, fails in production)

**Sources:**
- [Fine-Tuning LLMs on Small Data](https://dialzara.com/blog/fine-tuning-llms-with-small-data-guide)
- [Over-Memorization Research](https://arxiv.org/html/2412.13337v1)

---

### AT-3: Influence Function Approximation Error

**What goes wrong:** TGL-LLM uses influence functions with Hessian-vector products (HVP) to select high-quality samples. Approximation errors cause wrong samples to be selected, degrading Stage 1 training.

**Why it happens:**
- Influence functions require Hessian computation, which is approximated
- Approximation quality depends on checkpoint used
- Small errors in influence scores compound across 100K sample selection

**Consequences:**
- "High quality" subset actually contains noisy/mislabeled samples
- Stage 1 trains on garbage, Stage 2 cannot recover
- Model learns wrong patterns from start

**Prevention:**
1. **Simpler selection heuristic**: Use loss-based selection instead of influence functions
2. **Manual validation**: Spot-check selected samples before training
3. **Ensemble influence estimates**: Use multiple checkpoints, average scores
4. **Skip influence functions initially**: Use random sampling for first iteration

```python
# Simpler heuristic: select samples with moderate loss (not too easy, not too hard)
losses = compute_loss_per_sample(model, data)
percentile_25 = np.percentile(losses, 25)
percentile_75 = np.percentile(losses, 75)
quality_mask = (losses > percentile_25) & (losses < percentile_75)
quality_subset = data[quality_mask]
```

**Warning signs:**
- Selected "high quality" samples look obviously wrong on inspection
- Stage 1 loss doesn't decrease as expected
- Model performance worse than random sample selection

**Detection:** Compare random vs. influence-selected subsets on small validation.

**Phase:** Phase 3 (Training Pipeline) - can skip for v1, add in optimization phase.

**Severity:** DEGRADES QUALITY

**Sources:**
- [TGL-LLM Influence Function Section](https://arxiv.org/html/2501.11911v1)

---

## Mitigation Matrix

| Pitfall | Warning Signs | Prevention | Phase | Severity |
|---------|--------------|------------|-------|----------|
| **CP-1**: JAX/PyTorch Memory Conflict | nvidia-smi >10GB before loading | `XLA_PYTHON_CLIENT_PREALLOCATE=false`, process isolation | 1 | BLOCKS |
| **CP-2**: VRAM Exhaustion | OOM on first backward | QLoRA 4-bit, gradient checkpointing, accumulation | 2 | BLOCKS |
| **CP-3**: Projection Dimension Mismatch | Shape errors or very high loss | Explicit config dataclass, shape assertions | 2 | BLOCKS |
| **QP-1**: Alignment Collapse | Graph token cosine sim >0.95 | Monitor variance, stratified sampling, relation balancing | 3 | DEGRADES |
| **QP-2**: Calibration Regression | ECE >0.15, Brier worse than baseline | Separate calibration stage, track ECE throughout | 4 | DEGRADES |
| **QP-3**: 4-Bit Reasoning Degradation | Low accuracy on multi-hop queries | 8-bit inference, test reasoning explicitly | 2,4 | DEGRADES |
| **QP-4**: Historical Window Mismatch | Attention on old steps near-zero | Ablation on T, adaptive windowing | 3 | DEGRADES |
| **IP-1**: GDELT-POLECAT Domain Shift | >30% entity OOV rate | Relation mapping, entity normalization, train from scratch | 1 | DEGRADES/BLOCKS |
| **IP-2**: Tensor Layout Incompatibility | NaN after conversion | `.contiguous()`, verify values, explicit dtypes | 2 | BLOCKS/DEGRADES |
| **IP-3**: Frozen Backbone Gradient Leak | Memory >expected, weights change | `requires_grad_(False)`, verify trainable count | 2 | BLOCKS/DEGRADES |
| **RP-1**: Training Time Explosion | First epoch >2h | Smaller model, data subset, cloud burst | 1 | BLOCKS |
| **RP-2**: Inference Latency Regression | P95 >30s | Cache embeddings, batch queries, speculative decoding | 5 | DEGRADES |
| **AT-1**: LoRA Rank Selection | Plateau early or overfit | Start rank=16, alpha=2*r, sweep on subset | 3 | DEGRADES |
| **AT-2**: Over-Memorization | >99% train acc, <80% val | Min 1K samples, augmentation, early stopping | 3 | DEGRADES |
| **AT-3**: Influence Function Error | Selected samples look wrong | Use loss-based selection, manual validation | 3 | DEGRADES |

---

## Phase-Specific Prioritization

### Phase 1: Environment & Data
**Must address:**
- CP-1 (JAX/PyTorch memory) - will block everything
- IP-1 (GDELT-POLECAT mapping) - required for training data
- RP-1 (training time estimation) - set realistic expectations

### Phase 2: Adapter Architecture
**Must address:**
- CP-2 (VRAM exhaustion) - quantization config
- CP-3 (dimension mismatch) - architecture definition
- IP-2 (tensor layout) - interop verification
- IP-3 (gradient freeze) - backbone handling
- QP-3 (quantization quality) - precision choices

### Phase 3: Training Pipeline
**Must address:**
- QP-1 (alignment collapse) - monitoring + data balance
- QP-4 (window length) - hyperparameter search
- AT-1 (LoRA rank) - hyperparameter search
- AT-2 (over-memorization) - data handling
- AT-3 (influence functions) - can defer to v2

### Phase 4: Evaluation & Calibration
**Must address:**
- QP-2 (calibration regression) - re-calibration stage
- QP-3 (reasoning degradation) - test suite

### Phase 5: Production Optimization
**Must address:**
- RP-2 (inference latency) - caching, batching

---

## Confidence Assessment

| Area | Confidence | Rationale |
|------|------------|-----------|
| JAX/PyTorch interop pitfalls | HIGH | Official docs + GitHub issues confirm |
| TGL-LLM architecture pitfalls | MEDIUM | Paper provides detail, but your setup differs |
| VRAM constraints | HIGH | Hardware specs well documented |
| GDELT-POLECAT transfer | LOW | No direct research found on this specific transfer |
| Adapter training pitfalls | MEDIUM | General LoRA research applies, TGL-LLM specific details sparse |
| Calibration impact | MEDIUM | Known issue with LLM integration, your specific setup untested |

---

## Sources Summary

**Primary (HIGH confidence):**
- [TGL-LLM Paper](https://arxiv.org/html/2501.11911v1)
- [JAX GPU Memory Docs](https://docs.jax.dev/en/latest/gpu_memory_allocation.html)
- [Practical LoRA Tips - Sebastian Raschka](https://magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms)
- [Gradient Checkpointing Guide](https://medium.com/mlworks/gradient-checkpointing-the-unsung-hero-of-llm-training-ac2bbe5d4396)

**Secondary (MEDIUM confidence):**
- [GraphAdapter Paper](https://arxiv.org/html/2402.12984v1)
- [torch_jax_interop GitHub](https://github.com/lebrice/torch_jax_interop)
- [RTX 3060 LLM Benchmarks](https://www.databasemart.com/blog/ollama-gpu-benchmark-rtx3060ti)
- [QLoRA Guide](https://alain-airom.medium.com/run-big-llms-on-small-gpus-a-hands-on-guide-to-4-bit-quantization-and-qlora-40e9e2c95054)

**Supplementary (LOW confidence - needs validation):**
- [Cross-Modal Alignment in VLMs](https://openreview.net/forum?id=uQEsLZU15E)
- [LLaMA Quantization Studies](https://link.springer.com/article/10.1007/s44267-024-00070-x)
