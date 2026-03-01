# Phase 11: TKG Predictor Replacement - Context

**Gathered:** 2026-03-01
**Status:** Ready for planning

<domain>
## Phase Boundary

Port TiRGN algorithm from PyTorch to JAX using Flax/Linen, replacing RE-GCN as the temporal knowledge graph backend. Must fit within RTX 3060 12GB VRAM and 24-hour training envelope. RE-GCN remains as a config-selectable fallback. Weekly automated retraining via existing scheduler.

No new data sources, no new API endpoints, no frontend changes. This phase touches the TKG model layer only, behind the existing TKGModelProtocol interface.

</domain>

<decisions>
## Implementation Decisions

### Port Fidelity
- Full reproduction of the TiRGN paper (Li et al., 2022) — all components ported
- Global history encoder, copy-generation mechanism, and attention layers all included
- Copy-generation uses faithful sigmoid gate as described in the paper (not Gumbel-Softmax) — establish baseline first, optimize later
- Flax/Linen throughout — re-implement local encoder in Flax alongside global encoder rather than wrapping existing RE-GCN GraphsTuple code
- Sliding window for global history encoder (not full history) — window size is a tunable hyperparameter, start at 50 timestamps

### Resource Envelope
- Mixed precision (bfloat16 forward/backward, float32 parameter updates) — standard for Ampere GPUs, ~40-50% VRAM reduction
- OOM fallback escalation: gradient checkpointing first (no accuracy cost), dataset subsampling second (temporal subset or per-timestamp sampling)
- Early stopping on validation MRR with configurable patience — may finish well under 24h
- History window size tunable, default 50, determined empirically based on VRAM headroom

### Training Observability
- TensorBoard as always-available local dashboard (tensorboard --logdir runs/)
- Weights & Biases as optional cloud dashboard when W&B API key is configured
- Both log per-epoch metrics: loss, MRR, Hits@1/3/10, learning rate, VRAM usage, epoch duration
- Training can be monitored from any device without the original terminal

### Failure Strategy
- Ship TiRGN if within 5% MRR of RE-GCN — the architecture (copy-generation, global history) provides qualitative value even at slight accuracy parity
- If TiRGN misses the 5% threshold: one round of hyperparameter tuning (2-3 runs varying learning rate and history window)
- If still no improvement after tuning: abort, keep RE-GCN, mark phase as cancelled
- No sunk-cost fallacy — the research flag on this phase acknowledges failure is a valid outcome

### Migration & Coexistence
- RE-GCN remains as permanent config-selectable fallback (envvar TKG_BACKEND=tirgn|regcn)
- Config-only swap — requires process restart, no per-request model selection
- Single model loaded in memory at a time — no dual-model VRAM footprint
- No model_id tracking column — the active model is a system-wide setting, not per-prediction metadata

### Claude's Discretion
- JAX/Flax implementation specifics (scan vs unroll for RNN components, parameter initialization)
- Specific batch sizes, learning rates, embedding dimensions
- Test/train/validation split methodology for GDELT
- Code organization within src/training/ or new src/tkg/ module
- TensorBoard log directory structure and metric naming
- W&B project/run naming conventions
- Gumbel-Softmax as a future optimization (noted in backlog, not this phase)

</decisions>

<specifics>
## Specific Ideas

- The existing RE-GCN local encoder is already in JAX with custom GraphsTuple NamedTuples (jraph eliminated in Phase 9). TiRGN's Flax reimplementation of the local encoder should NOT wrap this — it's a clean rewrite in Flax that the old code can coexist alongside.
- TKGModelProtocol (@runtime_checkable, defined in Phase 9) is the swap interface. Both REGCNPredictor and TiRGNPredictor must satisfy it. EnsemblePredictor consumes the protocol — zero downstream changes.
- The weekly retraining scheduler (scripts/retrain_tkg.py from Phase 5) must work with TiRGN. Training observability (TensorBoard + W&B) should be active during automated retraining runs, not just manual training.

</specifics>

<deferred>
## Deferred Ideas

- Gumbel-Softmax copy-generation gate — optimization over sigmoid gate, try after baseline is established
- Per-request model selection / A/B comparison at API level — considered and rejected for simplicity; revisit if calibration data in Phase 13 suggests model-specific strengths per CAMEO category
- Hybrid ensemble (running both TiRGN and RE-GCN simultaneously, weighted by the ensemble) — interesting but doubles compute per prediction; revisit if the two models have complementary failure modes

</deferred>

---

*Phase: 11-tkg-predictor-replacement*
*Context gathered: 2026-03-01*
