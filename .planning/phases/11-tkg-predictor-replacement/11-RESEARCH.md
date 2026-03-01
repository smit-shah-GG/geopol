# Phase 11: TKG Predictor Replacement - Research

**Researched:** 2026-03-01
**Domain:** Temporal Knowledge Graph Reasoning -- TiRGN JAX/Flax Port
**Confidence:** MEDIUM (no published JAX implementation exists; architecture understood from paper + PyTorch reference)

## Summary

TiRGN (Time-Guided Recurrent Graph Network, Li et al., IJCAI 2022) extends RE-GCN with two key innovations: (1) a global history encoder that captures repeated facts across all prior timestamps via a sparse binary vocabulary matrix, and (2) a Time-ConvTransE decoder that integrates learned periodic and non-periodic time embeddings into the scoring function. The local and global predictions are fused via a scalar `history_rate` parameter (alpha) that linearly interpolates between the two distributions.

The reference PyTorch implementation (github.com/Liyyy2122/TiRGN) uses DGL for graph operations and PyTorch for the neural components. No JAX implementation exists anywhere. The port requires reimplementing: (a) the R-GCN local encoder (already done in our codebase as `REGCN` in Flax NNX), (b) the global history encoder (new -- sparse binary matrix construction + constrained decoding), (c) Time-ConvTransE decoder (extends our existing `ConvTransEDecoder` with time channels), and (d) the copy-generation fusion mechanism (sigmoid-weighted interpolation between raw and history-based softmax distributions).

**Primary recommendation:** Implement TiRGN as a new `nnx.Module` class (`TiRGN`) that reuses the structural patterns from our existing `REGCN` implementation (RelationalGraphConv, GRUCell) but adds the global history encoder, Time-ConvTransE, and copy-generation mechanism as new components. The existing `TKGModelProtocol` interface is sufficient -- no protocol changes needed.

## Standard Stack

The established libraries/tools for this domain:

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| JAX | 0.9.0 (installed) | Autodiff, JIT, GPU compute | Already in use, mandatory |
| Flax | 0.12.2 (installed) | Neural network modules (NNX API) | Already in use, nnx.Module patterns |
| optax | 0.2.6 (installed) | Optimizer (adam + clip_by_global_norm) | Already in use for RE-GCN training |
| NumPy | >=1.24 (installed) | Data preprocessing, sparse history construction | Already in use |
| SciPy | >=1.11 (installed) | CSR sparse matrices for global history | Already in use |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| tensorboardX | >=2.6 | TensorBoard metric logging from JAX | Always -- local training dashboard |
| wandb | >=0.16 | Cloud experiment tracking | Optional -- when WANDB_API_KEY set |
| jax-smi | >=1.0 | GPU memory monitoring in JAX | During training for VRAM logging |

### Not Needed
| Library | Why Not |
|---------|---------|
| DGL | Reference impl uses DGL; our codebase uses raw JAX scatter-add (already working in REGCN) |
| jmp | DeepMind's mixed precision lib is unmaintained; use manual dtype casting (bfloat16 on Ampere is trivial) |
| MPX | Equinox-based; we use Flax NNX, not Equinox |
| torch | TiRGN reference is PyTorch; we port to JAX, no torch dependency |

**Installation:**
```bash
uv add tensorboardX wandb jax-smi
```

## Architecture Patterns

### TiRGN Component Map (Reference PyTorch -> JAX Port)

```
Reference (PyTorch/DGL)              Port (JAX/Flax NNX)
========================             =====================
RecurrentRGCN                   -->  TiRGN(nnx.Module)
  RGCNCell (UnionRGCNLayer)     -->  RelationalGraphConv (reuse existing, minor tweaks)
  nn.GRUCell (entity)           -->  GRUCell (reuse existing)
  nn.GRUCell (relation)         -->  GRUCell (new instance, h_dim*2 -> h_dim)
  TimeConvTransE (raw decoder)  -->  TimeConvTransEDecoder (new, extends ConvTransEDecoder)
  TimeConvTransE (hist decoder) -->  TimeConvTransEDecoder (second instance for history mode)
  TimeConvTransR (rel decoder)  -->  NOT NEEDED (we only predict entities, not relations)
  history_vocabulary (sparse)   -->  build_global_history() -> jnp bool mask
  history_rate (alpha)          -->  history_rate: float config param
```

### Recommended Project Structure
```
src/
├── training/
│   ├── models/
│   │   ├── regcn_jax.py          # Existing RE-GCN (untouched)
│   │   ├── tirgn_jax.py          # NEW: TiRGN model module
│   │   ├── components/
│   │   │   ├── rgcn_layers.py    # Extract RelationalGraphConv, GRUCell (shared)
│   │   │   ├── time_conv_transe.py  # NEW: Time-ConvTransE decoder
│   │   │   └── global_history.py    # NEW: Global history encoder
│   │   └── __init__.py
│   ├── train_jax.py              # Existing training loop
│   ├── train_tirgn.py            # NEW: TiRGN-specific training entrypoint
│   ├── training_logger.py        # NEW: TensorBoard + W&B abstraction
│   └── scheduler.py              # Modified: model-agnostic retraining
├── protocols/
│   └── tkg.py                    # Existing TKGModelProtocol (unchanged)
└── forecasting/
    └── tkg_predictor.py          # Modified: backend dispatch via TKG_BACKEND envvar
```

### Pattern 1: TiRGN Forward Pass (Local + Global Fusion)

**What:** The TiRGN forward pass evolves entity embeddings locally (R-GCN + GRU per snapshot), then scores predictions using both raw (open-vocabulary) and history-constrained decoders, fusing with `history_rate`.

**When to use:** Every forward pass during training and inference.

**Pseudocode:**
```python
class TiRGN(nnx.Module):
    """TiRGN: Time-Guided Recurrent Graph Network."""

    def __init__(self, num_entities, num_relations, embedding_dim=200,
                 num_layers=1, num_bases=30, history_rate=0.5,
                 history_window=50, *, rngs):
        # Entity embeddings (learnable)
        self.entity_emb = nnx.Param(...)  # (num_entities, embedding_dim)
        self.rel_emb = nnx.Param(...)     # (num_relations*2, embedding_dim)

        # Local encoder: R-GCN layers + entity GRU + relation GRU
        self.rgcn_layers = nnx.List([RelationalGraphConv(...) for _ in range(num_layers)])
        self.entity_gru = GRUCell(embedding_dim, rngs=rngs)
        self.relation_gru = GRUCell(embedding_dim, input_dim=embedding_dim*2, rngs=rngs)

        # Time encoding (learnable periodic + non-periodic)
        self.weight_t1 = nnx.Param(...)  # non-periodic weight
        self.bias_t1 = nnx.Param(...)
        self.weight_t2 = nnx.Param(...)  # periodic (sin) weight
        self.bias_t2 = nnx.Param(...)

        # Raw decoder (open vocabulary)
        self.decoder_raw = TimeConvTransEDecoder(num_entities, embedding_dim, rngs=rngs)
        # History decoder (constrained to historical vocabulary)
        self.decoder_hist = TimeConvTransEDecoder(num_entities, embedding_dim, rngs=rngs)

        self.history_rate = history_rate  # alpha blending factor

    def evolve_embeddings(self, snapshots, training=True):
        """Local encoder: R-GCN + GRU per snapshot."""
        h = self.entity_emb.value
        r = self.rel_emb.value

        for snapshot in snapshots:
            # 1. Aggregate relation context from current edges
            rel_context = aggregate_relation_context(snapshot, r)
            # 2. Evolve relation embeddings via relation GRU
            r = self.relation_gru(r, rel_context)
            # 3. R-GCN message passing
            h_rgcn = self.encode_snapshot(h, snapshot, training)
            # 4. Evolve entity embeddings via entity GRU
            h = self.entity_gru(h, h_rgcn)

        return h  # (num_entities, embedding_dim)

    def compute_scores(self, entity_emb, triples, time_idx=None,
                       global_history_mask=None):
        """Fused local+global scoring."""
        t1, t2 = self.get_time_encoding(time_idx)

        # Raw mode: score all entities
        raw_scores = self.decoder_raw(entity_emb, triples, t1, t2)
        raw_probs = jax.nn.softmax(raw_scores, axis=-1)

        if global_history_mask is not None:
            # History mode: score only entities in history
            hist_scores = self.decoder_hist(entity_emb, triples, t1, t2,
                                            mask=global_history_mask)
            hist_probs = jax.nn.softmax(hist_scores, axis=-1)

            # Copy-generation fusion
            final_probs = (self.history_rate * hist_probs +
                          (1 - self.history_rate) * raw_probs)
        else:
            final_probs = raw_probs

        return jnp.log(final_probs + 1e-10)  # log-probs for NLL loss
```

### Pattern 2: Global History Construction

**What:** Build a binary mask indicating which entities appeared as objects for each (subject, relation) pair across all historical snapshots up to current timestamp.

**When to use:** Before each training step and at inference time.

```python
def build_global_history(historical_triples, num_entities, num_relations):
    """Build sparse global history vocabulary.

    Args:
        historical_triples: All triples from timestamps [0, t-1].
            Shape: (N, 3) where columns are [subject, relation, object].
        num_entities: Total entity count.
        num_relations: Total relation count (including inverse).

    Returns:
        history_mask: (num_entities * num_relations, num_entities) boolean mask.
            Row index = subject * num_relations + relation.
            Column index = object entity.
            True means this (s, r, o) triple appeared in history.
    """
    # Use scipy.sparse CSR for construction, convert to dense JAX for scoring
    from scipy import sparse

    rows = historical_triples[:, 0] * num_relations + historical_triples[:, 1]
    cols = historical_triples[:, 2]
    data = np.ones(len(rows), dtype=np.float32)

    history_sparse = sparse.csr_matrix(
        (data, (rows, cols)),
        shape=(num_entities * num_relations, num_entities)
    )

    # Convert to dense boolean for JAX (or use sparse ops if memory-constrained)
    return jnp.array(history_sparse.toarray() > 0)
```

**Critical note on memory:** The full dense history mask for GDELT would be `7700 * 480 * 7700 = ~28 billion booleans = 28GB`. This is obviously impossible. The reference implementation handles this by building the history mask **per-batch**, not globally. For each batch of query triples `(s, r, ?)`, look up only the row `s * num_relations + r` and get the entity mask for that specific (s,r) pair. This is a sparse lookup, not a dense matrix.

```python
def get_batch_history_mask(historical_triples, batch_triples, num_entities, num_relations):
    """Build history mask for a specific batch of query triples.

    For each (s, r) pair in the batch, find all entities that appeared as
    objects of (s, r, ?) in the historical data.

    Returns:
        mask: (batch_size, num_entities) boolean mask
    """
    batch_size = batch_triples.shape[0]
    mask = jnp.zeros((batch_size, num_entities), dtype=jnp.bool_)

    for i in range(batch_size):
        s, r = batch_triples[i, 0], batch_triples[i, 1]
        # Find all objects for this (s, r) pair in history
        match = (historical_triples[:, 0] == s) & (historical_triples[:, 1] == r)
        objects = historical_triples[match, 2]
        mask = mask.at[i, objects].set(True)

    return mask
```

In practice, the reference implementation pre-computes and saves these as per-timestamp sparse `.npz` files via `get_history.py`, then loads them at training time. Our implementation should follow the same pattern.

### Pattern 3: Time-ConvTransE Decoder

**What:** Extends ConvTransE with two additional time channels (periodic + non-periodic).

```python
class TimeConvTransEDecoder(nnx.Module):
    """ConvTransE with temporal encoding channels.

    Input is 4-channel: [subject_emb, relation_emb, time_periodic, time_nonperiodic]
    Each channel has shape (embedding_dim,), stacked to (4, embedding_dim).
    1D convolution operates across the 4-channel stack.
    """

    def __init__(self, num_entities, embedding_dim, num_filters=50,
                 kernel_size=3, *, rngs):
        # BatchNorm for 4-channel input
        self.bn0 = nnx.BatchNorm(4, rngs=rngs)
        # Conv1d: 4 input channels -> num_filters output channels
        self.conv = nnx.Conv(
            in_features=4, out_features=num_filters,
            kernel_size=(kernel_size,), rngs=rngs
        )
        self.bn1 = nnx.BatchNorm(num_filters, rngs=rngs)
        # FC projection
        conv_out_dim = num_filters * (embedding_dim - kernel_size + 1)
        self.fc = nnx.Linear(conv_out_dim, embedding_dim, rngs=rngs)
        self.bn2 = nnx.BatchNorm(embedding_dim, rngs=rngs)

        # All-entity embedding matrix for final scoring
        self.entity_emb_weight = nnx.Param(...)  # (num_entities, embedding_dim)

    def __call__(self, subj_emb, rel_emb, time_periodic, time_nonperiodic,
                 mask=None):
        """Score all entities for given (subject, relation, time).

        Returns: (batch_size, num_entities) scores
        """
        # Stack 4 channels: (batch, 4, embedding_dim)
        x = jnp.stack([subj_emb, rel_emb, time_periodic, time_nonperiodic], axis=1)
        x = self.bn0(x)
        # Conv1d -> ReLU -> flatten -> FC
        x = self.conv(x)
        x = jax.nn.relu(self.bn1(x))
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        x = self.bn2(x)
        # Score against all entities
        scores = x @ self.entity_emb_weight.value.T  # (batch, num_entities)

        if mask is not None:
            # Mask out entities not in history (set to -inf before softmax)
            scores = jnp.where(mask, scores, -1e9)

        return scores
```

### Pattern 4: Training Loop with Observability

```python
def train_tirgn(config, model, snapshots, train_triples, val_triples,
                entity_to_id, relation_to_id, logger):
    """Training loop with TensorBoard + optional W&B logging."""

    # Setup logging
    tb_writer = SummaryWriter(log_dir=f"runs/tirgn_{datetime.now():%Y%m%d_%H%M%S}")
    use_wandb = os.environ.get("WANDB_API_KEY") is not None
    if use_wandb:
        import wandb
        wandb.init(project="geopol-tkg", config=dataclasses.asdict(config))

    optimizer = nnx.Optimizer(model, optax.chain(
        optax.clip_by_global_norm(config.grad_clip),
        optax.adam(config.learning_rate),
    ), wrt=nnx.Param)

    best_mrr = 0.0
    patience_counter = 0

    for epoch in range(1, config.epochs + 1):
        epoch_start = time.time()
        # ... training step (loss computation, gradient, update) ...

        # Log metrics
        metrics = {
            "loss": avg_loss,
            "lr": config.learning_rate,
            "epoch_time": time.time() - epoch_start,
        }

        # VRAM monitoring
        device = jax.devices()[0]
        mem_stats = device.memory_stats()
        metrics["vram_used_mb"] = mem_stats["peak_bytes_in_use"] / 1e6

        # Periodic evaluation
        if epoch % config.eval_interval == 0:
            eval_metrics = compute_mrr(model, snapshots, val_triples, ...)
            metrics.update(eval_metrics)

            # Early stopping
            if eval_metrics["mrr"] > best_mrr:
                best_mrr = eval_metrics["mrr"]
                patience_counter = 0
                save_checkpoint(...)
            else:
                patience_counter += 1
                if patience_counter >= config.patience:
                    break

        # TensorBoard
        for k, v in metrics.items():
            tb_writer.add_scalar(k, v, epoch)

        # W&B (optional)
        if use_wandb:
            wandb.log(metrics, step=epoch)

    tb_writer.close()
    if use_wandb:
        wandb.finish()
```

### Anti-Patterns to Avoid

- **Dense global history matrix:** Do NOT try to materialize a `(num_entities * num_relations, num_entities)` dense matrix. For GDELT this is ~28GB. Use per-batch sparse lookups or pre-computed per-timestamp sparse files.
- **Wrapping existing REGCN:** Do NOT try to wrap or inherit from the existing `REGCN` class. TiRGN has different GRU topology (relation GRU + entity GRU), different decoder, and time encoding. Clean reimplementation using shared components (RelationalGraphConv, GRUCell) is the correct approach.
- **Multi-task relation prediction:** The reference implementation does entity + relation prediction jointly. Our use case only needs entity prediction (link prediction for `(s, r, ?)`). Skip the `TimeConvTransR` decoder and relation prediction loss entirely. This simplifies the model and reduces parameters.
- **Using `jax.lax.scan` for snapshot loop:** The snapshot loop cannot use `jax.lax.scan` because each snapshot has different shapes (different number of edges). Use Python `for` loop with `jax.checkpoint` per snapshot, exactly as the existing REGCN does.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Multi-head attention | Custom attention | `flax.nnx.MultiHeadAttention` | Built-in, tested, supports mixed precision via `dtype` param |
| 1D Convolution | Manual conv loop (like existing ConvTransEDecoder) | `flax.nnx.Conv` | The existing ConvTransEDecoder has a manual conv loop -- the Time-ConvTransE should use `nnx.Conv` instead |
| Batch normalization | Manual mean/var tracking | `flax.nnx.BatchNorm` | Reference uses BatchNorm1d extensively; NNX has built-in |
| Optimizer state | Manual adam implementation | `optax.adam` + `nnx.Optimizer` | Already in use, handles state correctly |
| Gradient checkpointing | Manual activation caching | `jax.checkpoint` / `jax.remat` | Already used in REGCN, proven pattern |
| Sparse matrix ops | Dense numpy masks | `scipy.sparse.csr_matrix` | Global history is inherently sparse; dense = OOM |
| VRAM monitoring | nvidia-smi parsing | `jax.devices()[0].memory_stats()` | JAX preallocates GPU memory; nvidia-smi always shows ~90% usage regardless of actual use |
| TensorBoard logging | Custom file writing | `tensorboardX.SummaryWriter` | Standard, works with JAX tensors directly |

**Key insight:** The reference TiRGN implementation hand-rolls DGL graph operations and manual convolution. Our JAX port should leverage Flax NNX's built-in modules (`nnx.Conv`, `nnx.BatchNorm`, `nnx.MultiHeadAttention` if needed) instead of reimplementing these from scratch.

## Common Pitfalls

### Pitfall 1: Global History Memory Explosion
**What goes wrong:** Attempting to build a dense `(E*R, E)` history matrix where E=7700, R=480 produces a 28GB tensor.
**Why it happens:** The naive implementation from the paper description suggests a single matrix. The actual reference implementation uses per-timestamp sparse files.
**How to avoid:** Pre-compute global history as per-timestamp sparse `.npz` files (same as reference `get_history.py`). At training time, load only the sparse matrix for the current timestamp. Convert to dense per-batch masks only for the query triples in the current batch.
**Warning signs:** OOM during data preparation, not during model forward pass.

### Pitfall 2: History Mode Softmax Over Masked Entities
**What goes wrong:** Applying softmax over all entities and then masking produces incorrect probability distributions. The masked entries contribute to the denominator.
**How to avoid:** Set non-history entities to `-inf` BEFORE softmax, not after. `scores = jnp.where(mask, scores, -jnp.inf)` then `softmax(scores)`. This ensures the probability mass is distributed only over historical entities.
**Warning signs:** History mode probabilities are uniformly tiny (close to 1/num_entities), no difference between history and raw mode.

### Pitfall 3: Time Encoding Dimensional Mismatch
**What goes wrong:** Time-ConvTransE expects 4 channels `(subj, rel, t1, t2)` each of `embedding_dim` length. Time indices are scalars, not embedding-dim vectors.
**Why it happens:** Confusion between time index (scalar integer) and time embedding (vector).
**How to avoid:** Time encoding produces two vectors of `embedding_dim`: `t1 = W_t1 * time_idx + b_t1` (element-wise, both W and b are `(embedding_dim,)` shaped), `t2 = sin(W_t2 * time_idx + b_t2)`. These are full embedding-dim vectors, not scalars.
**Warning signs:** Shape errors in Conv1d input, or model ignoring temporal signal.

### Pitfall 4: Flax NNX BatchNorm in Eval Mode
**What goes wrong:** `nnx.BatchNorm` updates running statistics during training. If not switched to eval mode during validation/inference, running stats get corrupted.
**Why it happens:** NNX BatchNorm uses `use_running_average` flag.
**How to avoid:** Pass `use_running_average=True` during evaluation. Structure the model's `__call__` to accept a `training` flag and propagate it to all BatchNorm layers.
**Warning signs:** Validation metrics are noisy/unstable, or inference results differ between runs.

### Pitfall 5: Loss Function Mismatch
**What goes wrong:** Using margin ranking loss (as in existing RE-GCN) instead of NLL loss (as TiRGN requires).
**Why it happens:** Copy-paste from existing `REGCN.compute_loss()`.
**How to avoid:** TiRGN produces a probability distribution over all entities (via softmax fusion of raw + history modes). The correct loss is `NLLLoss(log(P_fused), target_entity)`, NOT margin ranking loss. The protocol's `compute_loss` method must return this NLL loss.
**Warning signs:** Training diverges, or model converges but MRR is much worse than expected.

### Pitfall 6: JAX Pre-allocation Hiding Real OOM
**What goes wrong:** JAX preallocates 75% of GPU memory at startup. `nvidia-smi` always shows high usage. Actual OOM happens later when the preallocated pool is exhausted.
**Why it happens:** JAX memory management is fundamentally different from PyTorch.
**How to avoid:** Use `jax.devices()[0].memory_stats()["peak_bytes_in_use"]` for real usage. Set `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9` to use more of the 12GB. Monitor with `jax-smi`.
**Warning signs:** `jax.errors.OutOfMemoryError` despite nvidia-smi showing memory available.

### Pitfall 7: Protocol Compliance with Different Loss Signature
**What goes wrong:** `TKGModelProtocol.compute_loss()` expects `(snapshots, pos_triples, neg_triples, margin)` but TiRGN uses NLL loss without negative triples or margin.
**Why it happens:** The protocol was designed around RE-GCN's margin ranking loss.
**How to avoid:** TiRGN's `compute_loss()` should accept but ignore `neg_triples` and `margin`. Internally, it computes the NLL loss over `pos_triples` using the softmax-fused distribution. The interface signature stays compatible, but the semantics change. Document this clearly.
**Warning signs:** TypeError when swapping models, or silent correctness bugs if negative triples are accidentally used.

## Code Examples

### Time Encoding (Verified from Reference Implementation)

```python
# Source: github.com/Liyyy2122/TiRGN/blob/main/src/rrgcn.py
def get_time_encoding(self, time_indices):
    """Compute periodic and non-periodic time embeddings.

    Args:
        time_indices: Integer time step indices (batch,)

    Returns:
        t_nonperiodic: (batch, embedding_dim) non-periodic component
        t_periodic: (batch, embedding_dim) periodic (sinusoidal) component
    """
    # time_indices: (batch,) -> (batch, 1) for broadcasting
    T = time_indices[:, None].astype(jnp.float32)

    # Non-periodic: linear function of time
    t1 = self.weight_t1.value * T + self.bias_t1.value  # (batch, embedding_dim)

    # Periodic: sinusoidal function of time
    t2 = jnp.sin(self.weight_t2.value * T + self.bias_t2.value)  # (batch, embedding_dim)

    return t1, t2
```

### Copy-Generation Fusion (Verified from Reference Implementation)

```python
# Source: github.com/Liyyy2122/TiRGN/blob/main/src/rrgcn.py
def fused_scoring(self, entity_emb, rel_emb, t1, t2, triples,
                  history_mask, history_rate):
    """Compute fused copy-generation scores.

    Args:
        entity_emb: (num_entities, dim) evolved entity embeddings
        rel_emb: (num_relations*2, dim) relation embeddings
        t1, t2: (batch, dim) time encodings
        triples: (batch, 3) query triples [s, r, o]
        history_mask: (batch, num_entities) bool -- True for historical entities
        history_rate: float in [0, 1] -- alpha blending factor

    Returns:
        log_probs: (batch, num_entities) log-probabilities
    """
    subj_emb = entity_emb[triples[:, 0]]
    r_emb = rel_emb[triples[:, 1]]

    # Raw mode: score all entities (generation)
    raw_scores = self.decoder_raw(subj_emb, r_emb, t1, t2)
    raw_probs = jax.nn.softmax(raw_scores, axis=-1)

    # History mode: score only historical entities (copy)
    hist_scores = self.decoder_hist(subj_emb, r_emb, t1, t2)
    # Mask non-historical entities BEFORE softmax
    hist_scores = jnp.where(history_mask, hist_scores, -1e9)
    hist_probs = jax.nn.softmax(hist_scores, axis=-1)

    # Linear interpolation (copy-generation fusion)
    fused_probs = history_rate * hist_probs + (1.0 - history_rate) * raw_probs

    return jnp.log(fused_probs + 1e-10)
```

### Mixed Precision Training Pattern

```python
# Manual bfloat16 for Flax NNX (no jmp dependency needed)
def train_step_mixed_precision(model, optimizer, snapshots, triples,
                               history_mask, config):
    """Single training step with bfloat16 forward/backward."""

    def loss_fn(model):
        # Cast inputs to bfloat16 for forward pass
        # Note: nnx.Param values stay float32; intermediates are bfloat16
        log_probs = model.compute_scores(
            model.evolve_embeddings(snapshots, training=True),
            triples,
            time_idx=triples[:, 3] if triples.shape[1] > 3 else None,
            global_history_mask=history_mask,
        )
        # Loss computation in float32 for numerical stability
        loss = -jnp.mean(log_probs[jnp.arange(len(triples)), triples[:, 2]])
        return loss

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)
    return loss
```

For bfloat16 in specific layers, use the `dtype` parameter on `nnx.Conv`, `nnx.Linear`, and `nnx.MultiHeadAttention`:
```python
self.conv = nnx.Conv(in_features=4, out_features=50, kernel_size=(3,),
                     dtype=jnp.bfloat16, param_dtype=jnp.float32, rngs=rngs)
```
This computes in bfloat16 but stores parameters in float32 (master weights pattern).

### VRAM Monitoring

```python
def log_vram_usage():
    """Get current and peak GPU memory usage from JAX."""
    device = jax.devices("gpu")[0]
    stats = device.memory_stats()
    return {
        "vram_current_mb": stats["bytes_in_use"] / 1e6,
        "vram_peak_mb": stats["peak_bytes_in_use"] / 1e6,
        "vram_limit_mb": stats["bytes_limit"] / 1e6,
    }
```

### Backend Dispatch via Environment Variable

```python
# In tkg_predictor.py or a factory module
import os

def create_tkg_model(num_entities, num_relations, embedding_dim=200,
                     seed=0, **kwargs):
    """Factory function respecting TKG_BACKEND envvar."""
    backend = os.environ.get("TKG_BACKEND", "tirgn").lower()

    if backend == "tirgn":
        from src.training.models.tirgn_jax import TiRGN, create_tirgn_model
        return create_tirgn_model(num_entities, num_relations,
                                  embedding_dim, seed=seed, **kwargs)
    elif backend == "regcn":
        from src.training.models.regcn_jax import create_model
        return create_model(num_entities, num_relations,
                           embedding_dim, seed=seed)
    else:
        raise ValueError(f"Unknown TKG_BACKEND: {backend}. Use 'tirgn' or 'regcn'.")
```

## VRAM Budget Estimation

**Target:** RTX 3060 12GB, GDELT dataset (~7,700 entities, 240 relations, 200-dim embeddings, 50-timestamp window)

### Parameter Memory (float32)

| Component | Size | Memory |
|-----------|------|--------|
| Entity embeddings | 7,700 * 200 | 6.2 MB |
| Relation embeddings | 480 * 200 | 0.4 MB |
| R-GCN basis (1 layer) | 30 * 200 * 200 | 4.8 MB |
| R-GCN coefficients | 480 * 30 | 0.06 MB |
| R-GCN self-loop weight | 200 * 200 | 0.16 MB |
| Entity GRU (3 gates) | 3 * (200*200 + 200*200 + 200) | 1.0 MB |
| Relation GRU (3 gates) | 3 * (400*200 + 200*200 + 200) | 1.4 MB |
| Time-ConvTransE decoder x2 | 2 * (50*3*4 + 50*198*200 + 200) | ~15 MB |
| Time encoding params | 4 * 200 | 0.003 MB |
| **Total parameters** | ~7.5M params | **~30 MB** |

### Optimizer State (adam = 2x params)
| Component | Memory |
|-----------|--------|
| First moment (m) | 30 MB |
| Second moment (v) | 30 MB |
| **Total optimizer** | **60 MB** |

### Activations (forward pass, no checkpointing)

| Component | Per-snapshot | 50 snapshots |
|-----------|-------------|--------------|
| Entity embeddings (float32) | 7,700 * 200 = 6.2 MB | 310 MB |
| R-GCN intermediates | ~12 MB | 600 MB |
| GRU intermediates | ~6 MB | 300 MB |
| **Total activations** | | **~1.2 GB** |

### With Gradient Checkpointing + bfloat16

| Component | Memory |
|-----------|--------|
| Parameters (float32 master) | 30 MB |
| Optimizer state | 60 MB |
| Activations (checkpointed, only 1 snapshot at a time in bfloat16) | ~12 MB |
| Gradients (float32) | 30 MB |
| Batch history mask (batch_size * 7700 booleans) | ~8 MB (for batch=1024) |
| Decoder intermediates | ~50 MB |
| **Total estimated** | **~200 MB** |

**Verdict:** 200MB is well within the 12GB envelope. Even without gradient checkpointing (~1.4 GB total), this fits easily. The RTX 3060 constraint is not a concern for GDELT-scale data with these entity/relation counts. Gradient checkpointing is insurance, not necessity.

**When VRAM becomes a concern:** If entity count exceeds ~100K or embedding dimension exceeds ~500, the entity embedding matrix and decoder scoring (`batch * num_entities` matrix) become significant. At GDELT scale, this is not an issue.

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| RE-GCN only (single local encoder) | TiRGN (local + global + time encoding) | 2022 (IJCAI) | +2-4% MRR on ICEWS14/GDELT |
| Margin ranking loss | NLL loss with softmax distribution | CyGNet 2021 / TiRGN 2022 | Enables copy-generation probability fusion |
| DGL for graph operations | Raw JAX scatter-add | Our codebase (Phase 9) | Eliminates jraph/DGL dependency |
| jmp for mixed precision | Manual dtype params on Flax modules | jmp unmaintained, Flax 0.12 has dtype support | Simpler, no extra dependency |
| Flax Linen (functional) | Flax NNX (imperative, mutable) | Flax 0.8+ | Our codebase uses NNX exclusively |

**TiRGN vs RE-GCN Performance (from published results):**

| Dataset | RE-GCN MRR | TiRGN MRR | Improvement |
|---------|-----------|-----------|-------------|
| ICEWS14 | 42.00 | 44.04 | +4.9% |
| ICEWS18 | 30.58 | 33.66 | +10.1% |
| GDELT | 19.69 | 21.67 | +10.1% |

The improvement on GDELT is the most relevant benchmark for this project. A 10% relative MRR improvement (19.69 -> 21.67) is substantial and well above the 5% acceptance threshold defined in CONTEXT.md.

**Post-TiRGN models (context, not targets):**
- HisMatch (2023): 46.4% MRR on ICEWS14 -- pattern matching approach, fundamentally different architecture
- TRCL (2025): +1% over TiRGN via contrastive learning -- marginal gain, much more complex
- LGCL (2024): Adds contrastive loss to TiRGN-like architecture -- again, marginal improvement

TiRGN remains the best ROI for this project: significant improvement over RE-GCN baseline, manageable implementation complexity, and well-documented reference code.

## Open Questions

### 1. Relation GRU Input Dimension
**What we know:** The reference implementation uses `nn.GRUCell(h_dim*2, h_dim)` for the relation GRU, meaning it concatenates relation embeddings with aggregated context before the GRU. Our existing `GRUCell` only supports `(h_dim, h_dim)`.
**What's unclear:** Whether to modify the existing `GRUCell` to support different input/hidden dimensions, or create a new class.
**Recommendation:** Create a `GRUCell` variant that accepts `(input_dim, hidden_dim)` rather than assuming `input_dim == hidden_dim`. This is a ~5-line change.

### 2. Exact `history_rate` Tuning
**What we know:** The reference implementation uses different `history_rate` values per dataset (0.0 to 1.0). For GDELT it appears to be around 0.3-0.5.
**What's unclear:** Optimal value for our specific GDELT subset (30-day window vs full dataset).
**Recommendation:** Default to 0.3, make it a config parameter. Tune empirically after baseline is established.

### 3. Relation Prediction (Multi-Task)
**What we know:** The reference implementation does joint entity + relation prediction with `task_weight` parameter (default 0.7 for entity, 0.3 for relation).
**What's unclear:** Whether relation prediction contributes to entity prediction quality (regularization effect).
**Recommendation:** Skip relation prediction in v1. If entity MRR is below expectations, add relation prediction as a potential improvement (adds ~30% more parameters via TimeConvTransR decoder).

### 4. Static Graph Component
**What we know:** TiRGN optionally uses a static graph (word embeddings + static relations) for additional context. The reference implementation supports this with `--add-static-graph` flag.
**What's unclear:** Whether this is needed for GDELT (which has no static graph).
**Recommendation:** Skip entirely. Our GDELT data is purely temporal. This is confirmed by the reference implementation using it only for ICEWS datasets with word features.

### 5. BatchNorm vs LayerNorm in NNX
**What we know:** The reference uses `BatchNorm1d` in the decoder. Our existing RE-GCN does not use any normalization in the decoder.
**What's unclear:** Whether Flax NNX's `nnx.BatchNorm` handles the (batch, channels, length) layout identically to PyTorch's `BatchNorm1d`.
**Recommendation:** Verify `nnx.BatchNorm` axis configuration during implementation. The feature axis must be specified correctly (Flax uses `feature_axes` parameter vs PyTorch's implicit channel dimension).

## Sources

### Primary (HIGH confidence)
- [TiRGN GitHub Repository](https://github.com/Liyyy2122/TiRGN) -- Complete reference implementation in PyTorch/DGL. Source code for `rrgcn.py`, `decoder.py`, `get_history.py` examined.
- [IJCAI 2022 Proceedings](https://www.ijcai.org/proceedings/2022/299) -- Official paper publication page
- [Flax NNX API Reference: MultiHeadAttention](https://flax.readthedocs.io/en/stable/api_reference/flax.nnx/nn/attention.html) -- Constructor params, dtype support
- [Flax NNX Module System](https://flax.readthedocs.io/en/latest/nnx_basics.html) -- Module patterns, nnx.Param, nnx.Optimizer, nnx.List
- [JAX gradient checkpointing docs](https://docs.jax.dev/en/latest/gradient-checkpointing.html) -- jax.checkpoint / jax.remat usage
- [JAX device memory profiling](https://docs.jax.dev/en/latest/device_memory_profiling.html) -- memory_stats() API
- Installed versions verified: Flax 0.12.2, JAX 0.9.0, optax 0.2.6

### Secondary (MEDIUM confidence)
- [PMC Article on TKG Reasoning](https://pmc.ncbi.nlm.nih.gov/articles/PMC11784877/) -- TiRGN vs RE-GCN performance numbers on GDELT (ICEWS14 MRR: TiRGN 44.04 vs RE-GCN 42.00; GDELT MRR: TiRGN 21.67 vs RE-GCN 19.69)
- [TiRGN Architecture Blog Post](https://suojifeng.xyz/2023/09/10/tirgn/) -- Architecture equations and component descriptions
- [JAX Stack TensorBoard Guide](https://docs.jaxstack.ai/en/latest/JAX_visualizing_models_metrics.html) -- tf.summary logging pattern for JAX
- [W&B JAX/Flax Integration](https://wandb.ai/jax-series/simple-training-loop/reports/Writing-a-Training-Loop-in-JAX-and-Flax--VmlldzoyMzA4ODEy) -- wandb.log() integration pattern
- [JMP library (unmaintained)](https://github.com/google-deepmind/jmp) -- Confirmed unmaintained; manual dtype approach preferred

### Tertiary (LOW confidence)
- [UvA DL Notebooks: Single GPU Training](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/scaling/JAX/single_gpu_techniques.html) -- Mixed precision + gradient checkpointing combined patterns
- [CyGNet paper (arXiv)](https://arxiv.org/abs/2012.08492) -- Original copy-generation mechanism concept
- VRAM estimation: Calculated from model parameter counts, not empirically measured. Actual usage may vary with JIT compilation overhead and XLA buffer management.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- all libraries already installed and in use, no new core dependencies
- Architecture: MEDIUM -- Paper architecture well-understood, reference code examined, but JAX port is novel work with potential surprises in NNX module patterns
- VRAM estimation: MEDIUM -- Calculated from parameter counts, but JAX/XLA memory overhead can be non-trivial (JIT compilation buffers, memory fragmentation). Conservative estimate suggests 10x headroom.
- Performance expectations: MEDIUM -- Published benchmarks on full GDELT (2,975 timestamps, 1.7M triples). Our subset (30 days, ~50K-200K triples) may show different characteristics.
- Pitfalls: HIGH -- Derived from examining both reference implementation and our existing codebase

**Research date:** 2026-03-01
**Valid until:** 2026-04-01 (stable domain, TiRGN paper is from 2022, architecture is well-established)
