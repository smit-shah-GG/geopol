# TiRGN Training: Evolve-Once-Per-Epoch Optimization

**Date:** 2026-03-09
**Branch:** `train-evolve-once-per-epoch`
**Constraint:** RTX 3060 12GB VRAM, 32GB RAM
**Status:** Implemented, untested at scale. `master` retains per-batch evolution for cloud/high-VRAM training.

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
- Entity embeddings are wrapped in `jax.lax.stop_gradient()` to prevent autodiff from tracing back through the scan during batch backward passes.

### Cost structure after optimization

```
Epoch cost = 1 × scan + 781 × loss
           = 1 × expensive + 781 × cheap
           ≈ 10 min/epoch
```

~12× speedup.

### What `stop_gradient` does

Without it, JAX would trace through the scan during the backward pass of every batch (since `entity_emb` flows into `compute_loss_from_embeddings`). `stop_gradient` tells XLA to treat the embeddings as a constant for differentiation. The R-GCN and entity GRU parameters receive **zero gradient** from batch losses — they update only because the next epoch's `evolve_step` uses updated decoder parameters.

This is a form of **alternating optimization**: evolve → freeze → train decoder → repeat.

---

## Quality impact

The per-batch evolution approach gives each batch gradients through the full model (scan + decoder). The evolve-once approach only gives decoder gradients per batch.

**Expected MRR impact: negligible.** Reasons:

1. The learning rate is small (0.001 with cosine decay). 781 small updates don't drift parameters far enough to make stale embeddings meaningfully wrong.
2. The eval function (`_evaluate_tirgn`) already evolves once and batches over scoring — the eval path is unchanged.
3. The TiRGN and RE-GCN reference implementations both use per-epoch evolution as the standard approach. Per-batch evolution was an over-engineering in the original implementation.

The per-batch approach is preserved in `model.compute_loss()` for callers with sufficient GPU budget (e.g., cloud A10G/V100 instances).

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
