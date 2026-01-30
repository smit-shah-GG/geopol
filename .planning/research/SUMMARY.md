# Project Research Summary

**Project:** v2.0 Hybrid Architecture - Deep Token-Space Integration
**Domain:** TGL-LLM Integration for Geopolitical Forecasting
**Researched:** 2026-01-31
**Confidence:** MEDIUM-HIGH

## Executive Summary

The v2.0 deep integration replaces the current post-hoc 60/40 ensemble with TGL-LLM style token-space fusion where temporal knowledge graph embeddings become native input tokens to an LLM decoder. The existing JAX/jraph RE-GCN encoder (200-dim embeddings) remains the backbone; new components include PyTorch adapter layers (200->4096 projection), temporal tokenization across T=5-7 snapshots, and self-hosted Llama2-7B with LoRA fine-tuning. The RTX 3060 12GB constraint necessitates 4-bit quantization and gradient accumulation to achieve the ~23GB training footprint reported in the TGL-LLM paper.

The core architectural shift moves from late fusion (independent LLM and TKG predictions averaged) to early fusion (graph embeddings injected as soft prompts before LLM reasoning). This enables joint graph-language reasoning where the LLM can attend to specific temporal patterns and multi-hop relationships, with TGL-LLM benchmarks showing +70-95% accuracy gains on POLECAT datasets. However, the GDELT-to-POLECAT domain shift (different entity extraction, relation taxonomy, and event density) requires training from scratch rather than transfer learning.

Critical risks center on framework interoperability (JAX/PyTorch GPU memory conflicts), cross-modal alignment collapse during two-stage training, and calibration regression from moving to LLM-generated probabilities. The JAX memory pre-allocation conflict is a showstopper that must be resolved before any model work begins. The VRAM constraint is solvable via QLoRA but will increase training time from ~10h (A40) to 80-100h (RTX 3060) for full-dataset training.

## Key Findings

### Recommended Stack

The deep integration requires four new dependencies atop the existing JAX/jraph stack: `transformers>=5.0.0` for Llama2-7B loading, `bitsandbytes>=0.49.0` for 4-bit NF4 quantization, `accelerate>=1.12.0` for device placement with quantization support, and `peft>=0.18.1` for LoRA adapter training. The existing RE-GCN encoder remains JAX-based; only the LLM decoder and adapter layers migrate to PyTorch.

**Core technologies:**
- **Llama2-7B-chat (4-bit NF4)**: Self-hosted LLM backbone with 4096-dim hidden space; replaces Gemini API for controllable token injection. NF4 quantization reduces VRAM from 14GB to ~3.5GB without requiring pre-quantized model files.
- **PEFT LoRA (r=16, alpha=32)**: Parameter-efficient fine-tuning targeting attention projections (q_proj, v_proj). Keeps backbone frozen to avoid 56GB full fine-tuning requirement. Rank 16 is conservative but sufficient for cross-modal alignment per LoRA research.
- **DLPack bridge (JAX->PyTorch)**: Zero-copy tensor conversion via `torch.utils.dlpack.from_dlpack()` and `jax.dlpack.to_dlpack()`. Avoids CPU round-trip for GPU tensors but requires explicit `.contiguous()` calls to prevent layout incompatibility.
- **Two-layer MLP adapters**: Project 200-dim graph embeddings to 4096-dim LLM token space. Hidden dimension 1024 is empirical; layer normalization essential for stable training. Entity and relation adapters share architecture but have separate parameters.

**VRAM budget (RTX 3060 12GB):**
- Llama2-7B NF4: ~3.5-4.0 GB
- KV cache (2K context): ~2.0-2.5 GB
- Projection layers: ~0.4 GB
- LoRA adapters: ~0.1 GB
- Activations/buffers: ~1.5 GB
- **Total: ~8-9 GB** with ~3GB headroom

**Not recommended:**
- vLLM (server overhead unjustified for single-request forecasting)
- llama.cpp (no PEFT/LoRA training integration)
- AWQ/GPTQ (complicates training workflow; NF4 achieves similar quality)
- DeepSpeed/FSDP (single GPU, no sharding benefit)

### Expected Features

**Must have (table stakes):**
- **Temporal Graph Adapter**: Projects 200-dim RE-GCN embeddings to 4096-dim Llama token space via 2-layer MLP. Without this, graph and language remain separate modalities.
- **Hybrid Graph Tokenization**: Concatenates T=5-7 graph snapshots as soft tokens with text query. Enables LLM to explore temporal patterns that v1.1 ensemble cannot communicate.
- **Two-Stage Training Pipeline**: Stage 1 fine-tunes on 100K high-quality samples for cross-modal alignment; Stage 2 adds diverse samples for generalization. Single-stage fails on alignment per TGL-LLM ablations.
- **Frozen LLM Backbone**: Required for 12GB VRAM; full fine-tuning needs ~56GB. LoRA adapters on attention layers allow task learning without backbone updates.
- **GRU Temporal Evolution**: Captures dynamics between consecutive graph states. RE-GCN alone is static; GRU essential for temporal pattern learning.

**Should have (competitive differentiators):**
- **Joint Graph-Language Reasoning**: LLM reasons over graph structure and text simultaneously, discovering patterns neither modality reveals alone. v1.1's fixed 60/40 weighting is information-theoretically isolated.
- **Multi-Hop Relational Reasoning**: LLM traces chains of relationships across graph structure. v1.1 TKG only scores direct triples without path reasoning.
- **Adaptive Confidence Based on Context Quality**: LLM recognizes when graph signal is weak and adjusts confidence. v1.1 uses fixed weighting regardless of context quality.
- **Explanation Grounding**: Reasoning chains reference specific graph structures ("based on 5 recent CONFLICT events..."). v1.1 explanations are purely text-based.

**Defer (v2+):**
- Real-time inference (11+ hours training; inference latency increases from ~2s to 10-30s)
- Maximum context window (research shows tight filtering beats maximum context due to attention sink)
- End-to-end differentiable training (memory explosion; frozen backbone + adapter-only training is mandatory)

**Anti-features (do not build):**
- Replacing Gemini immediately (parallel systems until v2.0 validates gains)
- Using structure as replacement for text (graph enhances semantics, does not replace them per arXiv 2511.16767)
- Full LLM fine-tuning (exceeds hardware constraint by 4x)

**Expected performance:**
- Best case (clean data, optimal context): +40-60% accuracy vs v1.1
- Realistic case (production data, moderate context): +15-25% accuracy
- Degraded case (noisy data, poor context): +0-10% or negative

The variable performance profile contrasts with v1.1's robust fixed weighting. Context quality assessment becomes critical.

### Architecture Approach

The v2.0 architecture transforms the data flow from late fusion (independent models -> weighted average) to early fusion (graph tokens -> LLM input embedding sequence). The existing RE-GCN encoder remains the backbone; new components include JAX-PyTorch bridge for zero-copy tensor conversion, adapter layers projecting graph embeddings to LLM token space, temporal tokenizer sequencing T snapshots as soft prompts, Llama2-7B decoder with LoRA, and output parser extracting structured forecasts.

**Major components:**
1. **Entity/Relation Adapters** (PyTorch, trainable) — Two-layer MLPs (200 -> 1024 -> 4096) with GELU activation and layer normalization. Project graph embeddings into Llama's token space. Separate adapters for entities and relations enable specialized projection strategies.
2. **Temporal Tokenizer** (PyTorch) — Sequences T=5-7 recent graph snapshots as soft tokens, concatenates with text query tokens. Tracks graph token positions for embedding injection during LLM forward pass. Handles variable-length text queries and temporal ordering.
3. **JAX-PyTorch Bridge** (interop layer) — Converts JAX arrays from RE-GCN to PyTorch tensors via DLPack zero-copy. Must enforce `.contiguous()` and same dtype to avoid layout incompatibility. For inference only (no gradient flow through bridge).
4. **Llama Decoder** (PyTorch, frozen with LoRA) — Llama2-7B-chat quantized to 4-bit NF4 with LoRA adapters (r=16) on attention projections. Generates forecasts conditioned on hybrid prompt. LoRA targets q_proj/v_proj to reduce trainable parameters by 33%.
5. **TGL-LLM Predictor** (orchestrator) — End-to-end pipeline: RE-GCN encoding (JAX) -> bridge -> adapter projection (PyTorch) -> temporal tokenization -> Llama generation -> output parsing. Replaces `ensemble_predictor.py` as primary forecasting interface.

**Modified components:**
- `tkg_predictor.py`: Add `get_temporal_embeddings()` method returning sequence of (num_entities, 200) arrays for T recent snapshots.
- `regcn_jraph.py`: Add `get_embeddings_sequence()` method for temporal snapshot extraction. Existing `evolve_embeddings()` remains for training.
- `reasoning_orchestrator.py`: Deprecate for core path; retain for A/B testing and fallback.

**Deprecated components:**
- `ensemble_predictor.py` — Late fusion replaced by deep fusion. Archive to `_deprecated/`.
- `gemini_client.py` — Gemini API replaced by local Llama2. Archive but keep Gemini path operational for parallel validation.

**Optional components:**
- `rag_pipeline.py` — Retain for explainability and edge cases. Core forecasting no longer depends on RAG; graph embeddings carry semantic signal.

**Critical integration points:**
1. **JAX-PyTorch boundary**: Memory copies, device placement, no gradient flow. For training, freeze RE-GCN encoder and train only adapters + LoRA (matches TGL-LLM paper). For production, pre-compute embeddings and cache as PyTorch tensors.
2. **Embedding dimension matching**: RE-GCN produces 200-dim, Llama2-7B expects 4096-dim. Adapter hidden layer (1024) is tunable; output dimension must match exactly.
3. **Temporal alignment**: Graph snapshots may not align with query timestamps. Use T most recent snapshots; exact timestamp matching unnecessary (temporal ordering sufficient per TGL-LLM).
4. **Memory requirements**: Llama2-7B fp16 ~14GB, RE-GCN embeddings ~800KB/snapshot, adapters ~8MB each. 16GB+ GPU for training, 8GB+ for inference with 4-bit quantization.

### Critical Pitfalls

**1. JAX/PyTorch GPU Memory Pre-allocation Conflict (BLOCKS PROGRESS)**
JAX pre-allocates 75% of GPU memory on first operation; PyTorch uses lazy caching. When both run in the same process, they fight for VRAM and cause OOM even when total model size fits. Set `XLA_PYTHON_CLIENT_PREALLOCATE='false'` and `XLA_PYTHON_CLIENT_MEM_FRACTION='0.5'` before any JAX import. Process isolation (separate processes for JAX encoder and PyTorch LLM with explicit tensor serialization) is the robust solution. Warning signs: `nvidia-smi` shows >10GB allocated before model loading, OOM errors dependent on import order.

**2. VRAM Exhaustion During Two-Stage Training (BLOCKS PROGRESS)**
TGL-LLM requires ~23GB on A40; RTX 3060 has 12GB. Naive implementation OOMs during Stage 1 fine-tuning. Use QLoRA with 4-bit NF4 base model (~3.5GB weights), gradient checkpointing (trades 20% speed for 40% memory reduction), and gradient accumulation to simulate batch 128 with micro-batches of 8. Expected training time: 80-100 hours on RTX 3060 (vs 10h on A40). Warning signs: OOM during first backward pass, loss goes NaN from tiny batches.

**3. Cross-Modal Alignment Collapse (DEGRADES QUALITY)**
Graph adapter learns to project all embeddings to narrow token space region, causing mode collapse where LLM treats all graph tokens as identical. Occurs when Stage 1 quality subset is poorly selected or relation distribution is imbalanced (GDELT heavily skewed toward `10_Q1` Make Statement). Monitor adapter output variance during training; cosine similarity >0.95 indicates collapse. Prevention: stratified sampling in Stage 2, undersample frequent relations, check if removing graph tokens changes predictions (if not, adapter failed).

**4. Calibration Regression from Deep Integration (DEGRADES QUALITY)**
v1.1 has calibrated probability outputs (isotonic calibration, temperature scaling). Deep integration produces LLM logits that are notoriously miscalibrated. Brier scores may worsen even if accuracy improves. Add dedicated calibration stage after TGL-LLM integration; track ECE/Brier during development, not just accuracy. Warning signs: accuracy improves but Brier score worsens, confidence histogram shows extreme U-shape, ECE >0.15 on validation.

**5. GDELT-to-POLECAT Domain Shift (DEGRADES/BLOCKS)**
TGL-LLM trained on POLECAT (80 relations, 25K-34K entities, LLM-based extraction). GDELT uses CAMEO codes (300+ subcodes) with different entity normalization. Relation taxonomies don't map 1:1; entity vocabulary has high OOV rate. Cannot use pretrained TGL-LLM directly. Create CAMEO->POLECAT relation mapping, expand `ENTITY_ALIASES` dict, train from scratch on GDELT with TGL-LLM architecture. Warning signs: >30% entity OOV rate, relation distribution differs from POLECAT, model performs well on POLECAT test cases but poorly on GDELT-specific queries.

**6. 4-Bit Quantization Quality Degradation on Reasoning (DEGRADES QUALITY, SUBTLE)**
4-bit quantization required for 12GB VRAM disproportionately degrades reasoning/chain-of-thought performance compared to 8-bit or FP16. Geopolitical forecasting requires multi-hop reasoning (entity -> relation -> consequence chains). Model appears to work but makes subtle logical errors on complex queries. Use 8-bit for inference (only 4-bit during QLoRA training), test reasoning explicitly with stratified test set by depth (1-hop, 2-hop, 3-hop). Consider smaller FP16 models (Phi-3-mini 3.8B) may outperform Llama2-7B in 4-bit.

**7. Historical Window Length Mismatch (DEGRADES QUALITY)**
TGL-LLM uses T=5-7 timesteps tuned for POLECAT density. Your system uses 30-day history. GDELT has different event density per time unit. Blindly using T=5 may miss context; T=30 introduces noise. Analyze GDELT temporal density (events/day), run ablation study on T in [3, 5, 7, 10, 14] before full training. Use event count, not fixed days; weight older events lower (existing `decay_rate=0.95`). Warning signs: performance drops when increasing T, attention weights on old timesteps near-zero.

**8. Training Time Explosion on RTX 3060 (BLOCKS PROGRESS if timeline unrealistic)**
TGL-LLM reports ~10h on A40 (40GB, 86 TFLOPS). RTX 3060 has 12GB and 13 TFLOPS. Gradient accumulation and checkpointing add overhead. Expected: 80-100h for Llama2-7B full GDELT with QLoRA; 8-10h for 10% data subset; 20-30h for Phi-3-mini; 10-15h for TinyLlama. Use data subsampling (10%) for architecture validation; cloud burst (RunPod/Lambda) for final training with full data.

## Implications for Roadmap

Based on research, v2.0 requires 6 phases with distinct failure modes per phase. The critical path prioritizes environment setup (memory conflicts are showstoppers) before any model work, followed by adapter architecture (dimension/quantization decisions lock in constraints), then training infrastructure (where quality degradation risks emerge).

### Phase 1: Environment Setup & Data Preparation
**Rationale:** JAX/PyTorch memory conflict is a showstopper that blocks all subsequent work. GDELT-to-POLECAT domain shift requires relation mapping and entity normalization before training data can be prepared. Training time estimation sets realistic timeline expectations.

**Delivers:**
- JAX/PyTorch memory coordination (environment variables, process isolation strategy)
- CAMEO->POLECAT relation mapping layer (20 root codes to semantic categories)
- Entity normalization expansion (augment existing `ENTITY_ALIASES`)
- Training time benchmarks (profile QLoRA training on data subset)
- Dataset preparation for two-stage training (quality subset selection heuristic, not influence functions for v1)

**Addresses:**
- CP-1 (JAX/PyTorch memory conflict) — must solve before any model loading
- IP-1 (GDELT-POLECAT domain shift) — required for training data
- RP-1 (training time estimation) — sets expectations, determines whether cloud burst needed

**Avoids:**
- OOM errors from framework memory conflict
- High entity OOV rate destroying adapter training
- Unrealistic timeline expectations (80-100h on RTX 3060 for full training)

**Research flags:** Standard environment configuration; skip research-phase.

### Phase 2: Adapter Architecture & Quantization
**Rationale:** Architecture decisions (dimensions, quantization precision, LoRA rank) lock in constraints for all subsequent phases. Dimension mismatches cause shape errors; wrong quantization choice degrades reasoning quality; incorrect LoRA configuration causes undertrained or overfit adapters.

**Delivers:**
- Entity/Relation adapter implementations (2-layer MLP, 200->1024->4096)
- JAX-PyTorch bridge with `.contiguous()` enforcement and dtype verification
- Dimension config dataclass with shape assertions at every projection boundary
- Quantization config (4-bit NF4 for training, evaluate 8-bit for inference)
- LoRA configuration (r=16, alpha=32, targets q_proj/v_proj)
- Frozen backbone verification (check trainable parameter count)
- Unit tests: dimension matching, tensor layout round-trip, adapter output norm matching LLM token embedding norm

**Addresses:**
- CP-2 (VRAM exhaustion) — QLoRA config with gradient checkpointing
- CP-3 (dimension mismatch) — explicit config, shape assertions
- IP-2 (tensor layout incompatibility) — `.contiguous()`, value verification
- IP-3 (frozen backbone gradient leak) — `requires_grad_(False)`, verify trainable count
- QP-3 (4-bit reasoning degradation) — evaluate 8-bit for inference, create reasoning test set

**Avoids:**
- Shape errors or silent broadcasting issues
- NaN/Inf from tensor layout incompatibility
- OOM from gradient flow through frozen backbone
- Subtle reasoning errors from excessive quantization

**Research flags:** Standard adapter pattern from LLaGA/TGL-LLM; skip research-phase. Profile memory before training.

### Phase 3: Temporal Tokenizer & Llama Integration
**Rationale:** With adapters and bridge validated, integrate Llama2-7B decoder and temporal token sequencing. This phase verifies graph token injection works before expensive training begins.

**Delivers:**
- Temporal tokenizer (sequences T snapshots as soft tokens, concatenates with text query)
- Llama decoder wrapper (Llama2-7B-chat with LoRA, quantization config)
- Graph token injection mechanism (replaces placeholder tokens with projected embeddings)
- Modified `tkg_predictor.py` with `get_temporal_embeddings()` method
- Modified `regcn_jraph.py` with `get_embeddings_sequence()` method
- Integration test: RE-GCN embeddings -> adapter -> temporal tokenizer -> Llama generation
- Sanity check prompts: verify Llama generates reasonable text with graph tokens

**Addresses:**
- Entity/relation adapter integration with temporal sequencing
- Llama2-7B loading with quantization
- Graph token position tracking for embedding injection

**Avoids:**
- Token sequence misalignment between graph and text
- Placeholder token replacement errors
- Generation quality issues before training (if sanity checks fail, architecture wrong)

**Research flags:** Standard pattern; skip research-phase. Test thoroughly before training.

### Phase 4: Two-Stage Training Pipeline
**Rationale:** TGL-LLM's two-stage training (Stage 1: alignment on high-quality subset, Stage 2: diversity) is essential for cross-modal alignment. This phase implements training infrastructure and monitoring for alignment collapse, the primary quality degradation risk.

**Delivers:**
- Training script with two-stage structure (Stage 1: 100K quality samples, Stage 2: diverse stratified sampling)
- Quality subset selection (loss-based heuristic, defer influence functions to v2)
- Stratified sampling for Stage 2 (undersample frequent GDELT relations like `10_Q1`)
- Alignment collapse detection (monitor adapter output variance, cosine similarity)
- Historical window hyperparameter search (T in [3, 5, 7, 10, 14])
- LoRA rank validation (start r=16, sweep [8, 16, 32] on subset)
- Data augmentation (entity name paraphrasing, temporal shuffling) to prevent over-memorization
- Gradient accumulation to simulate batch 128 (micro-batch 8, accumulation steps 16)
- Checkpointing and early stopping on novel validation set

**Addresses:**
- QP-1 (alignment collapse) — monitoring, stratified sampling, relation balancing
- QP-4 (window length mismatch) — ablation study on T
- AT-1 (LoRA rank selection) — hyperparameter search
- AT-2 (over-memorization) — augmentation, minimum 1K samples per task, early stopping
- AT-3 (influence function approximation) — defer to v2, use loss-based selection

**Avoids:**
- Mode collapse where graph tokens become meaningless
- Temporal window too short (missing context) or too long (noise injection)
- Undertrained adapters (rank too low) or overfit adapters (rank too high)
- Memorization masking alignment failure

**Research flags:** NEEDS RESEARCH-PHASE for hyperparameter search strategy and ablation study design. Two-stage training requires careful data curation.

### Phase 5: Evaluation & Calibration
**Rationale:** Accuracy alone is insufficient; calibrated probabilities are essential for trustworthy forecasts. Deep integration destroys existing calibration. Dedicated phase ensures Brier scores don't regress despite accuracy improvements.

**Delivers:**
- A/B test harness (v2.0 TGL-LLM vs v1.1 ensemble on held-out GDELT)
- Stratified test set by reasoning depth (1-hop, 2-hop, 3-hop) to detect quantization degradation
- Calibration analysis (reliability diagrams, ECE, Brier scores per category)
- Temperature scaling layer for TGL-LLM outputs (re-trained, not transferred from v1.1)
- Isotonic calibration re-fitting if temperature scaling insufficient
- Latency benchmarks (end-to-end inference time including RE-GCN encoding)
- Context quality impact analysis (compare performance on clean vs noisy data subsets)

**Addresses:**
- QP-2 (calibration regression) — dedicated calibration stage, track ECE/Brier
- QP-3 (4-bit reasoning degradation) — stratified test by reasoning depth

**Avoids:**
- Deploying overconfident predictions (ECE >0.15)
- Accuracy improvements masking probability calibration failures
- Reasoning degradation from quantization going undetected

**Research flags:** Standard calibration methods; skip research-phase. Focus on comprehensive evaluation.

### Phase 6: Integration & Deprecation
**Rationale:** With v2.0 validated, integrate into production forecasting API, deprecate v1.1 components, and establish operational procedures. Keep Gemini path operational until v2.0 proves production-ready.

**Delivers:**
- TGLLLMPredictor as default forecasting interface (replaces ensemble_predictor.py)
- Output parser for structured forecast extraction from Llama generation
- Embedding cache layer (pre-compute and cache RE-GCN embeddings for common entities)
- Batch query processing (amortize RE-GCN encoding across multiple queries)
- Archive deprecated components (ensemble_predictor.py, gemini_client.py to _deprecated/)
- Documentation update (API changes, expected latency, calibration characteristics)
- Parallel validation: Gemini path remains operational for comparison

**Addresses:**
- RP-2 (inference latency regression) — caching, batching strategies
- Production deployment with fallback to v1.1

**Avoids:**
- Losing working v1.1 system before v2.0 proves gains
- Inference latency surprises (document 10-30s expected vs 2s Gemini)
- Single point of failure (parallel systems during transition)

**Research flags:** Standard integration patterns; skip research-phase.

### Phase Ordering Rationale

**Phase 1 before Phase 2:** Memory conflicts and domain shift must be resolved before any model architecture work. JAX/PyTorch OOM blocks all progress; high entity OOV rate destroys training.

**Phase 2 before Phase 3:** Adapter architecture and quantization decisions lock in constraints. Dimension mismatches cause shape errors; wrong quantization degrades quality. Must be correct before Llama integration.

**Phase 3 before Phase 4:** Graph token injection must be verified with sanity checks before expensive training. If injection mechanism is broken, training wastes 80-100 hours.

**Phase 4 before Phase 5:** Training must complete before evaluation. Two-stage training takes majority of timeline (80-100h on RTX 3060).

**Phase 5 before Phase 6:** Calibration and A/B testing validate v2.0 beats v1.1 before deprecation. Premature deprecation loses working system.

**Critical path:** Phase 1 (setup) -> Phase 2 (architecture) -> Phase 4 (training). Phase 3 (temporal tokenizer) and Phase 5 (evaluation) can partially overlap with training if needed.

### Research Flags

**Phases needing research-phase:**
- **Phase 4 (Training):** Hyperparameter search strategy for T (window length), LoRA rank, and learning rate requires domain-specific tuning. Two-stage training data curation (quality subset selection, stratified sampling) needs careful design to avoid alignment collapse.

**Phases with standard patterns (skip research-phase):**
- **Phase 1 (Environment):** JAX/PyTorch memory coordination is documented; GDELT-POLECAT relation mapping is data engineering.
- **Phase 2 (Adapters):** LLaGA/TGL-LLM adapter pattern is well-established; dimension matching and quantization config are straightforward.
- **Phase 3 (Temporal Tokenizer):** Standard token concatenation; Llama integration follows PEFT documentation.
- **Phase 5 (Evaluation):** Temperature scaling and isotonic calibration are standard probability calibration methods.
- **Phase 6 (Integration):** Standard API refactoring and deprecation.

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | Official docs for transformers/bitsandbytes/peft verified; VRAM calculations confirmed via multiple sources; JAX-PyTorch DLPack interop documented in both frameworks |
| Features | MEDIUM | TGL-LLM paper verified for architecture and benchmarks (POLECAT Acc@4: 0.8514); GDELT performance extrapolation unverified; context quality impact demonstrated in HTKGH paper but not for this specific setup |
| Architecture | MEDIUM-HIGH | TGL-LLM architecture pattern verified; adapter design from LLaGA paper confirmed; JAX/jraph RE-GCN analysis from existing codebase; PyTorch integration standard but JAX-PyTorch boundary introduces complexity |
| Pitfalls | MEDIUM | JAX/PyTorch memory conflict documented in official docs and GitHub issues; VRAM constraints verified via hardware specs; TGL-LLM training pitfalls inferred from paper (doesn't detail failure modes); GDELT-POLECAT domain shift speculative (no direct research found) |

**Overall confidence:** MEDIUM-HIGH

Research is grounded in verified sources (TGL-LLM paper, official framework docs, hardware specs) but extrapolation to GDELT domain and RTX 3060 hardware introduces uncertainty. The TGL-LLM paper reports results on POLECAT with A40 hardware; direct transferability to GDELT on RTX 3060 is unverified.

### Gaps to Address

**1. GDELT-to-POLECAT transfer quality:**
No research found on cross-dataset transfer for TGL-LLM architecture. POLECAT uses LLM-based entity extraction; GDELT uses TABARI/PETRARCH. Entity vocabulary overlap unknown. **Mitigation:** Compute entity/relation overlap statistics early in Phase 1; if OOV >30%, may need more aggressive entity normalization or consider training on ICEWS as intermediate domain.

**2. RTX 3060 training time with QLoRA:**
Scaling from A40 (10h, 23GB) to RTX 3060 (80-100h estimated, 12GB) assumes linear scaling adjusted for TFLOPS and memory bandwidth. Actual time may be worse due to memory bottleneck effects. **Mitigation:** Profile training on 10% data subset in Phase 4; if epoch time >2h, revise timeline or use cloud burst for final training.

**3. 4-bit quantization impact on geopolitical reasoning:**
Research shows 4-bit degrades chain-of-thought, but geopolitical forecasting may be more or less sensitive than generic reasoning benchmarks. **Mitigation:** Create geopolitical reasoning test set (multi-hop relation chains, temporal reasoning) early in Phase 5; compare 4-bit vs 8-bit inference quality.

**4. Cross-modal alignment collapse detection threshold:**
Cosine similarity >0.95 is heuristic, not verified threshold. May need domain-specific tuning. **Mitigation:** Log alignment metrics throughout training in Phase 4; establish baseline on known-good checkpoint.

**5. Optimal historical window length for GDELT:**
TGL-LLM uses T=5-7 for POLECAT; GDELT event density differs. Optimal T unknown. **Mitigation:** Ablation study in Phase 4 is mandatory, not optional.

## Sources

### Primary (HIGH confidence)
- [TGL-LLM: Integrate Temporal Graph Learning into LLM-based Temporal Knowledge Graph Model](https://arxiv.org/abs/2501.11911) — Core architecture (entity/relation adapters, two-stage training, GRU temporal evolution), POLECAT benchmarks (Acc@4: 0.8514), hardware requirements (23.08GB VRAM, 11.26h on A40)
- [LLaGA: Large Language and Graph Assistant](https://arxiv.org/abs/2402.08170) — Graph projection to LLM token space pattern, two-layer MLP design (200->4096), soft token injection
- [Hugging Face Transformers v5 Documentation](https://huggingface.co/docs/transformers/main/en/index) — Llama2-7B architecture (4096 hidden dim), quantization integration, device placement
- [bitsandbytes Documentation](https://huggingface.co/docs/bitsandbytes/main/en/installation) — NF4 quantization (4-bit NormalFloat), BitsAndBytesConfig parameters, CUDA 12 compatibility
- [PEFT Documentation](https://huggingface.co/docs/peft/en/index) — LoRA configuration (r, alpha, target_modules), QLoRA training, adapter hotswapping
- [JAX GPU Memory Allocation](https://docs.jax.dev/en/latest/gpu_memory_allocation.html) — XLA_PYTHON_CLIENT_PREALLOCATE and XLA_PYTHON_CLIENT_MEM_FRACTION environment variables

### Secondary (MEDIUM confidence)
- [HTKGH: Toward Better Temporal Structures for Geopolitical Events Forecasting](https://arxiv.org/abs/2601.00430) — Context quality impact (LLMs beat GNNs by 21% with tight filtering), geopolitical forecasting benchmarks
- [torch_jax_interop GitHub](https://github.com/lebrice/torch_jax_interop) — DLPack zero-copy conversion, tensor layout compatibility, device placement coordination
- [Practical Tips for Finetuning LLMs (Sebastian Raschka)](https://magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms) — LoRA rank selection (r=16 standard), alpha=2*r scaling, target_modules best practices
- [Gradient Checkpointing Guide](https://medium.com/mlworks/gradient-checkpointing-the-unsung-hero-of-llm-training-ac2bbe5d4396) — Memory reduction (40%) vs speed tradeoff (20%), compatibility with frozen backbone
- [QLoRA Guide](https://alain-airom.medium.com/run-big-llms-on-small-gpus-a-hands-on-guide-to-4-bit-quantization-and-qlora-40e9e2c95054) — 4-bit NF4 vs GPTQ/AWQ comparison, gradient accumulation patterns

### Tertiary (LOW confidence, needs validation)
- [RTX 3060 LLM Benchmarks](https://www.databasemart.com/blog/ollama-gpu-benchmark-rtx3060ti) — Token generation speed (~7-10 tokens/s for 7B models), inference latency estimates
- [LLaMA Quantization Study](https://link.springer.com/article/10.1007/s44267-024-00070-x) — 4-bit degradation on chain-of-thought tasks (not geopolitical reasoning specifically)
- [When Structure Doesn't Help](https://arxiv.org/html/2511.16767) — LLM performance with node text vs graph structure (marginal or negative gains from structure alone)
- [DAEA: Entity Alignment with Domain Adaptation](https://aclanthology.org/2025.coling-main.393.pdf) — Cross-domain entity alignment methods (GDELT-POLECAT entity vocabulary overlap unknown)

### Existing Codebase (HIGH confidence)
- `src/training/models/regcn_jraph.py` — RE-GCN implementation details (embedding_dim=200, GRU temporal evolution, JAX/jraph)
- `src/forecasting/tkg_predictor.py` — TKG prediction interface (history_length=30, decay_rate=0.95)
- `src/forecasting/ensemble_predictor.py` — Current 60/40 weighted voting architecture
- `src/calibration/temperature_scaler.py` — Existing calibration infrastructure (isotonic + temperature scaling)
- `src/database/models.py` — CAMEO code usage, ENTITY_ALIASES dict for normalization

---
*Research completed: 2026-01-31*
*Ready for roadmap: yes*
