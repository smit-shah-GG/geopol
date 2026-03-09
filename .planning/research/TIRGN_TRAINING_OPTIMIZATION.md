# TiRGN Training: Evolve-Once-Per-Epoch Optimization

**Date:** 2026-03-09
**Branch:** `train-evolve-once-per-epoch`
**Constraint:** RTX 3060 12GB VRAM, 32GB RAM
**Status:** Implemented, first training run complete (best MRR 0.2881 at epoch 15, early stopped at 30). GRU bias fix verified. Early stopping should be switched from MRR to validation NLL (see "Evaluation Metrics" section). `master` retains per-batch evolution for cloud/high-VRAM training.

---

## Problem

TiRGN training OOMs on 12GB VRAM when run without `--max-events` cap.

The root cause is **not batch size** — it's the R-GCN einsum intermediate tensor in `RelationalGraphConv.__call__` (`regcn_jax.py:137`):

```python
basis_msgs = jnp.einsum('ei,bio->ebo', src_feats, self.basis.value)
```

Output shape: `(max_edges, num_bases, embedding_dim)` × float32.

With 30 days of GDELT data, `PaddedSnapshots` pads all snapshots to the max-edge day. Current data profile:

| Metric | Value |
|--------|-------|
| Total events (30-day window) | 2,093,895 |
| Max raw edges/day | 96,246 |
| Max edges with inverses | 192,492 |
| Median raw edges/day | 73,346 |
| Einsum tensor at max | `192492 × 30 × 200 × 4 = 4.30 GiB` |
| Peak VRAM (with checkpoint recompute + XLA) | ~12.4 GiB |

The 12GB card cannot fit this. The previous successful run used `--max-events 200000`, which randomly sampled events down to 200K, capping the heaviest snapshot at ~9K edges (18K with inverses, 0.41 GiB einsum).

### Why PaddedSnapshots causes uniform padding

JAX's `lax.scan` requires uniform tensor shapes across all iterations. Every snapshot is padded to `max_edges` — the maximum across all 30 days. Even though median-day snapshots have far fewer edges, the pad ceiling is set by the single densest day. The `edge_mask` zeros out padding during computation, but the memory is still allocated.

### Why batch size doesn't help

The R-GCN message passing operates on the **full graph snapshot** regardless of batch size. `evolve_embeddings` encodes all entities through the graph to produce embeddings *before* any batch-level scoring happens. Batch size only controls how many triples are scored in the NLL loss — a negligible fraction of total compute.

---

## The per-batch evolution bottleneck

In the original training loop, **every batch** called `model.compute_loss()`, which internally called `model.evolve_embeddings()` — the full 30-snapshot R-GCN+GRU scan. With 1M events at batch_size=1024, that's 781 full scans per epoch:

```
Epoch cost = 781 batches × (scan + loss)
           = 781 × (expensive + cheap)
           ≈ 126 min/epoch
```

The entity embeddings produced by `evolve_embeddings` depend only on model parameters + graph structure — **not** on which triples are being scored. Re-evolving per batch gives marginally fresher gradients (parameters are updated between batches), but at 781× the compute cost.

---

## Solution: evolve once per epoch

**Branch:** `train-evolve-once-per-epoch`

### Changes

**`src/training/models/tirgn_jax.py`:**
- Added `compute_loss_from_embeddings()`: decoder-only loss that takes pre-evolved embeddings as input. Scores triples via the fused copy-generation distribution (TimeConvTransE + GlobalHistoryEncoder), computes label-smoothed NLL.
- `compute_loss()` refactored to delegate to `compute_loss_from_embeddings()` internally. **API unchanged** — existing callers (tests, protocol, regcn_wrapper) are unaffected.

**`src/training/train_tirgn.py`:**
- Split `train_step` into two JIT-compiled functions:
  - `evolve_step(model, rng_key)` → runs the R-GCN+GRU scan once per epoch
  - `train_step(model, optimizer, entity_emb, pos_jax, history_mask)` → decoder-only loss + optimizer update per batch
- Entity embeddings flow as concrete arrays between JIT boundaries (no `stop_gradient` — see "Why stop_gradient is NOT used" below).

### Cost structure after optimization

```
Epoch cost = 1 × scan + 781 × loss
           = 1 × expensive + 781 × cheap
           ≈ 10 min/epoch
```

~12× speedup.

### Why `stop_gradient` is NOT used

Initial implementation wrapped `evolve_step` output in `jax.lax.stop_gradient()`. This killed training — loss and MRR were completely flat across epochs. The problem: the decoder scores triples by indexing into `entity_emb` (e.g., `entity_emb[subjects]`). With `stop_gradient`, these looked-up values were treated as constants by autodiff, so the decoder weights received zero useful gradient signal — the loss landscape was flat.

The fix: `evolve_step` is called **outside** the `train_step` JIT boundary. JAX's autodiff only traces within a single JIT compilation — it cannot backpropagate through a separate JIT'd function call that already returned a concrete array. So the scan is naturally excluded from backward passes without any explicit gradient manipulation. The entity embedding values remain live tensors that the decoder can differentiate against.

### GRU update gate bias initialization (critical)

Even after removing `stop_gradient`, training was still completely flat (loss and MRR byte-identical across epochs). Root cause: **exponential embedding collapse in the GRU scan**.

With `b_z = 0` (zero-initialized update gate bias):
- `z = sigmoid(0) = 0.5` at every snapshot step
- `h_new = (1 - z) * h + z * h_tilde ≈ 0.5 * h + 0.5 * ~0 ≈ 0.5 * h`
- After 30 snapshots: `19.52 × 0.5^30 ≈ 1e-8` — entity embeddings are dead
- Decoder scores `output @ entity_emb.T ≈ 0` → uniform softmax → zero learning signal

In per-batch evolution, backprop through the scan teaches the GRU to set its own gate biases for state preservation. Without those gradients (evolve-once), the GRU stays in its random "forget 50% per step" regime forever.

**Fix:** Initialize `b_z = -3.0` (in `GRUCell.__init__`, `regcn_jax.py`):
- `z = sigmoid(-3) ≈ 0.047`
- `h_new ≈ 0.953 * h + 0.047 * h_tilde` — preserves 95% of hidden state per step
- After 30 snapshots: `19.52 × 0.95^30 ≈ 4.36` — embeddings retain meaningful signal
- Decoder produces non-uniform scores → gradients flow → training progresses

This is the GRU analogue of the Jozefowicz et al. (2015) LSTM "forget gate bias = 1" initialization trick.

| Metric | b_z=0 | b_z=-3.0 |
|--------|-------|----------|
| Per-step norm ratio | 0.50 | 0.95 |
| Evolved emb norm (30 snaps) | 3.6e-8 | 4.36 |
| Batch 1→3 loss delta | +0.043 | -0.074 |
| Decoder param change per batch | 5.9e-6 | 0.276 |

---

## Quality impact

The per-batch evolution approach gives each batch gradients through the full model (scan + decoder). The evolve-once approach only gives decoder gradients per batch.

**Expected MRR impact: moderate.** The scan parameters (R-GCN weights, entity GRU, initial entity embeddings) do NOT receive gradient signal in evolve-once mode. They only change via weight decay. The entity embeddings are approximately the initial xavier random values, and the decoder must learn to score within this fixed embedding space. This is suboptimal compared to jointly-optimized embeddings.

The eval function (`_evaluate_tirgn`) already evolves once and batches over scoring — the eval path is unchanged.

The per-batch approach is preserved in `model.compute_loss()` for callers with sufficient GPU budget (e.g., cloud A10G/V100 instances).

---

## Evaluation Metrics: Why MRR Is Wrong for This System

### The metric misalignment

MRR measures **ranking quality**: "how highly does the model rank the correct entity among all candidates?" This is the standard TKG academic benchmark metric because research frames TKG as link prediction.

Geopol doesn't do link prediction. It does **probability estimation**:

```
TKGPredictor: sigmoid(model_score) → P_tkg
EnsemblePredictor: α(cameo) * P_llm + (1-α) * P_tkg → P_final
CalibrationOptimizer: minimize Brier(P_final, outcome)
```

The ensemble needs a **well-calibrated probability** from the TKG — when TiRGN says confidence=0.73, that event should happen ~73% of the time. Whether the correct entity is ranked #1 or #3 is secondary. What matters is whether `sigmoid(score)` closely tracks actual outcome probability.

### Why loss-MRR divergence happens

NLL loss can improve (better probability estimates) while MRR stagnates. This occurs when the model gets better at distinguishing "correct entity" from "random entity" (NLL improves) but not from "semantically similar entity" (MRR stalls). For probability estimation, the first improvement is exactly what matters.

Observed in training run (evolve-once, --max-events 200000):
- Loss: 4.82 → 4.52 (steady 6.4% reduction, still improving at epoch 30)
- MRR: 0.26 → 0.29 best (noisy, oscillating ±0.03 per epoch)
- H@10: 0.51 → 0.56 (stable upward trend)

Early stopping on MRR killed a productive run at epoch 30 while loss was clearly still improving.

### Recommended metric hierarchy

| Priority | Metric | Role | Why |
|----------|--------|------|-----|
| 1 (train) | **Validation NLL** | Early stopping criterion | Direct proxy for calibration quality; aligned with downstream Brier optimization |
| 2 (monitor) | **H@10** | Operational relevance | "Is the correct answer in the candidate set?" — determines whether TKGPredictor returns useful results |
| 3 (monitor) | **H@1** | Quality signal | "Is the model's top prediction correct?" — measures discrimination power |
| 4 (log only) | **MRR** | Benchmark comparability | Standard TKG metric for papers and external communication |
| 5 (system) | **Brier score** | Ultimate system metric | Computed at ensemble level with resolved outcomes, not during TKG training |

### Target values

**Validation NLL (primary training metric):**

| Level | NLL | Perplexity | Interpretation |
|-------|-----|------------|----------------|
| Random baseline | log(N) ≈ 8.3 | N (≈4000) | Guessing uniformly across all entities |
| Current evolve-once | ~4.5 | ~90 | Narrowed to ~90 candidates on average |
| Target (evolve-once) | < 3.5 | < 33 | Narrowed to ~33 candidates |
| Target (per-batch, master) | < 3.0 | < 20 | Narrowed to ~20 candidates |
| Theoretical floor | 0 | 1 | Perfect prediction every time |

Perplexity = exp(NLL). Intuitive interpretation: "the model narrows prediction from N entities down to perplexity entities on average." A perplexity of 20 on a 4,000-entity graph = 99.5% uncertainty reduction.

**Hits@K (operational monitoring):**

| Metric | Random | Current (evolve-once) | Good | Strong | SOTA (ICEWS14) |
|--------|--------|----------------------|------|--------|----------------|
| H@1 | 0.025% | ~10% | > 20% | > 30% | 30-38% |
| H@3 | 0.075% | ~31% | > 35% | > 45% | 45-55% |
| H@10 | 0.25% | ~56% | > 60% | > 70% | 65-72% |

Note: ICEWS14 benchmarks are on cleaner, smaller data. GDELT is noisier (GDELT auto-coded, ICEWS human-curated) and has more entities, so equivalent architectural quality produces lower absolute numbers on GDELT.

**MRR (external communication only):**

| Level | MRR | Context |
|-------|-----|---------|
| Baseline (RE-GCN) | 0.14 | Geopol v1.0 |
| Previous per-batch TiRGN | 0.49 | Geopol v2.0, full gradient flow |
| Current evolve-once | 0.29 | Decoder-only optimization, 12× faster training |
| Published SOTA (ICEWS14) | 0.44-0.46 | HisMatch, TiRGN, DNCL |
| Industry "good" range | 0.40-0.60 | Depends on dataset and entity count |

**System-level Brier score (ultimate metric):**

| Benchmark | Brier Score | Source |
|-----------|-------------|--------|
| Superforecasters | 0.081 | ForecastBench 2025 |
| Best LLM (GPT-4.5) | 0.101 | ForecastBench 2025 |
| Human crowd aggregated | 0.149 | IARPA ACE |
| Uninformed baseline | 0.250 | Always predicting 50% |

Brier is a system-level metric — it measures the full ensemble (LLM + TKG + calibration), not the TKG alone. The TKG's contribution to Brier is mediated by the per-CAMEO α weight: if α → 1.0 (LLM dominant), TKG improvements have negligible Brier impact.

### Loss logged in training IS NLL

The `Loss: X.XXXX` in training logs is label-smoothed NLL, computed in `compute_loss_from_embeddings()` (`tirgn_jax.py:423`):

```python
loss = (1 - ε) * NLL_hard + ε * NLL_uniform   # ε = 0.1
```

The label smoothing inflates absolute values ~5-10% vs pure NLL (the `ε * NLL_uniform` term regularizes against overconfidence). This does not affect relative improvement tracking or comparisons to the `log(N) ≈ 8.3` baseline. When reporting NLL targets, the smoothed values are directly usable.

### Why NLL has no universal "good" range

Unlike MRR (normalized to [0, 1]), NLL is domain-specific — it depends on entity count N. An NLL of 3.0 on a 4,000-entity GDELT graph is phenomenal; the same value on a 50-entity toy dataset is garbage. The theoretical baseline is `log(N)`.

NLL is the correct *training* metric but the wrong *client-facing* metric. Translation via perplexity or H@K is required.

### Metric by audience

**Internal (engineering):**
- **Validation NLL** — early stopping criterion, lower is better. Target: < 3.5 (evolve-once), < 3.0 (per-batch cloud).

**Technical due diligence:**
- **Perplexity** = exp(NLL). "Narrows prediction from 4,000+ entities to ~20." That's 99.5% uncertainty reduction.
- Perplexity < 33 (evolve-once target), < 20 (cloud target).

**Business clients / sales:**
- **H@10** — "Correct actor in top 10 predictions X% of the time, across thousands of entities." Current: ~56%. Target: > 65%.
- **H@1** for "wow factor" — > 20% means 1-in-5 perfect predictions across 4,000 entities (random: 0.025%).

**System-level (ultimate metric):**
- **Brier score** — measures full ensemble, not TKG alone. Superforecasters: 0.081. Best LLM: 0.101.

### Tiered client pitch by achievement level

| Tier | NLL | Perplexity | H@10 | Client pitch |
|------|-----|------------|------|-------------|
| Current | 4.5 | 90 | 56% | "Narrows 4,000 entities to ~90 candidates" |
| Good | 3.5 | 33 | 65% | "99.2% uncertainty reduction, correct actor in top 10 two-thirds of the time" |
| Strong | 3.0 | 20 | 72% | "99.5% uncertainty reduction across 4,000+ geopolitical actors" |
| SOTA-equiv | 2.5 | 12 | 78% | "Narrows to 12 candidates — nearly order-of-magnitude better than academic benchmarks on noisier data" |

Note: ICEWS14 published SOTA is H@10 65-72%, but ICEWS is human-curated with fewer entities. Matching those numbers on auto-coded GDELT (500K-1M articles/day) is significantly harder — equivalent quality produces lower absolute numbers. Frame this: "We operate on the noisiest, highest-volume geopolitical event stream, not curated academic datasets."

### Implementation status: COMPLETE

1. ~~Add validation NLL computation to `_evaluate_tirgn()`~~ — Done. Vectorized NLL via `-log(fused_probs[target_entities])`, returned as `val_loss`.
2. ~~Switch early stopping from `best_mrr` to `best_val_loss`~~ — Done. Comparison flipped to `<` (lower is better). `best_mrr` tracked for monitoring only.
3. ~~Continue logging MRR/H@K for monitoring~~ — Done. Log line now: `Loss | Val NLL | MRR | H@10 | Time`. Early stopping message reports `best val NLL`.
4. Return dict includes both `best_val_loss` (primary) and `best_mrr` (backward compat).
5. Tests updated: early stopping tests use val_loss (lower-is-better) semantics. All 8 tests pass.

---

## `--max-events` scaling reference

All measurements assume 30-day GDELT window, `embedding_dim=200`, `num_bases=30`, `batch_size=1024`.

| max_events | Data coverage | Entities | Max edges (inv) | Einsum peak | Est. VRAM | Epoch (old) | Epoch (new) |
|------------|--------------|----------|-----------------|-------------|-----------|-------------|-------------|
| 200,000 | 9.5% | 4,003 | 18,386 | 0.41 GiB | ~3.0 GiB | 5.0 min | ~0.7 min |
| 600,000 | 29% | 5,040 | 55,096 | 1.23 GiB | ~5.1 GiB | 45 min | ~4 min |
| 1,000,000 | 48% | 5,459 | 91,956 | 2.06 GiB | ~7.1 GiB | 126 min | ~10 min |
| 1,400,000 | 67% | 5,692 | 128,676 | 2.88 GiB | ~9.2 GiB | 246 min | ~20 min |
| 2,093,895 | 100% | 5,931 | 192,492 | 4.30 GiB | ~12.4 GiB | OOM | OOM |

**Recommended for RTX 3060:** `--max-events 1400000` (67% coverage, 96% entity coverage, ~2.8 GiB VRAM headroom).

Full dataset (unlimited) requires ≥16GB VRAM (T4/V100) regardless of evolve strategy — the einsum peak is a forward-pass constraint, not a gradient one.

---

## AWS EC2 GPU sizing (if provisioned)

Peak VRAM is dominated by the R-GCN einsum, which depends on `max_edges` per snapshot — NOT on snapshot count. 90-day training windows have similar peak VRAM to 30-day windows (same max daily edge count, more scan iterations = more wall-clock time only).

| Instance | GPU | VRAM | Cost/hr | Fits 30d full | Fits 90d full |
|----------|-----|------|---------|---------------|---------------|
| g4dn.xlarge | T4 | 16 GB | $0.526 | Yes (~3.5 GiB headroom) | Yes |
| g5.xlarge | A10G | 24 GB | $1.006 | Yes (comfortable) | Yes |
| p3.2xlarge | V100 | 16 GB | $3.06 | Yes | Yes |

The per-batch evolve strategy (`master` branch) is viable on these instances with full data and would give optimal gradient quality.

---

## Edge distribution context

The per-snapshot edge distribution is **bimodal**, not outlier-dominated:

- 88/118 days have ≤10K edges (older sparse historical dates)
- 30/118 days have >25K edges (recent steady-state GDELT volume)
- Top 10 days cluster at 79K–96K — all Feb–Mar 2026

The "max" of 96,246 is only 19% above P75 of the recent 30-day window. Capping edges per snapshot would discard the majority of real signal. The `--max-events` random sampling approach is superior: it uniformly thins all snapshots proportionally, preserving temporal distribution while reducing max density.
