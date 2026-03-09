# Architecture Assessment: TKG Model & Ensemble Strategy

**Date:** 2026-03-09
**Context:** Post-v3.0 ship, exploring whether fundamental architecture changes are warranted.
**Verdict:** No changes recommended. Both TiRGN and the linear ensemble are the right choices.

---

## Question 1: Should TiRGN Be Replaced?

### Candidates Evaluated

| Model | ICEWS14 MRR (pub.) | Architecture | Verdict |
|-------|-------------------|--------------|---------|
| RE-GCN | 40.4% | R-GCN + GRU + ConvTransE | Retired (geopol v1.0) |
| xERTE | 40.0% | Iterative subgraph + attention | Rejected: 9% MRR below our TiRGN, explainability served by LLM |
| TiRGN | 44.0% | R-GCN + GRU + Time-ConvTransE + copy-gen | **Current — keep** |
| HisMatch | 46.4% | Historical structure matching | Rejected: see below |
| TRCL (2025) | ~45.0% | TiRGN + contrastive learning | Rejected: +1% over published TiRGN, marginal |
| DNCL (2025) | ~46.2%* | Dual-gate + noise-aware contrastive | Rejected: complexity, noise handling overlaps with ensemble |
| TRANSFIR (ICLR 2026) | N/A (inductive) | Inductive reasoning for emerging entities | Watch list: too new, non-standard eval split |

*DNCL claims 4.73% over "second best" — baseline unclear.

### Why TiRGN Wins

1. **Our optimized TiRGN already outperforms published SOTA.** Geopol TiRGN achieves 0.4944 MRR (per-batch, master branch) via training quality improvements (AdamW, cosine LR, stochastic dropout, label smoothing). Published HisMatch is 0.464. Our optimizations matter more than architecture.

2. **TKG is 40% of the ensemble.** Even a 5% MRR improvement translates to ~2% improvement in final blended probability. Per-CAMEO calibration further dampens impact by shifting α away from TKG where it underperforms.

3. **Operational constraints dominate.** Weekly retraining on RTX 3060 12GB. JAX/Flax NNX ecosystem. All candidates are PyTorch-only — JAX ports are substantial engineering.

4. **Biggest wins are elsewhere.** Calibration quality, LLM prompt engineering, data pipeline denoising all have higher ROI than swapping the TKG backbone.

### Per-Candidate Rejection Rationale

**HisMatch (46.4%):**
- PyTorch only; JAX port is non-trivial (different architectural primitive — structure matching vs scan-based evolution)
- Requires heavy offline preprocessing (two scripts to precompute historical structures for all entities)
- Fragile on novel geopolitical configurations (no historical precedent to match)
- 2.4% published improvement vanishes against our optimized 49.44% TiRGN

**TRCL (2025):**
- Only +1.03% over published TiRGN — our training optimizations already yield +5.44%
- Adds contrastive learning complexity for negligible gain

**DNCL (2025):**
- Noise handling overlaps with what the LLM ensemble already provides
- Adversarial training adds instability on a weekly retraining cadence

**TRANSFIR (ICLR 2026):**
- 24.6% MRR improvement specifically on emerging entities (useful for new geopolitical actors)
- Too new (single reference implementation), non-standard eval split (5:2:3 vs 8:1:1)
- Worth watching; the inductive capability could be added as a lightweight module without replacing TiRGN

**xERTE (40.0%):**
- 9.4 MRR points below our TiRGN
- Iterative subgraph expansion is inherently sequential (bad GPU throughput)
- LLM already provides narrative explanations — structural reasoning paths add marginal value at severe accuracy cost

### Recommended TiRGN-Specific Improvements (Not Architecture Swaps)

1. **Temporal decay in GlobalHistoryEncoder (~50 lines):** TiRGN's most cited weakness is equal-weighting all historical entities regardless of recency. Apply exponential decay (e.g., 0.95^distance) to the history mask before scoring. This is what TRCL and LGevo both add, without their contrastive learning overhead.

2. **Per-snapshot stochastic edge sampling (~100 lines in create_padded_snapshots):** Instead of global --max-events sampling, sample edges per snapshot to a target density. Dense days get thinned more, sparse days keep all edges. Better temporal distribution preservation.

3. **Lightweight inductive entity fallback (~200 lines, inference-time only, TRANSFIR-inspired):** When TKGPredictor encounters an entity absent from entity_to_id, generate an embedding from LLM contextual understanding and project it into TiRGN's entity space via a learned linear layer.

### Sources

- [TRCL — Recurrent Encoding + Contrastive Learning (2025)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11784877/)
- [DNCL — Dual-gate Noise-aware Framework (2025)](https://www.nature.com/articles/s41598-025-00314-w)
- [TRANSFIR — Inductive TKG Reasoning (ICLR 2026)](https://openreview.net/pdf/02fbc09156d1d73d619cc230cb24205df328993d.pdf)
- [HiSMatch — Official GitHub](https://github.com/Lee-zix/HiSMatch)
- [LGevo — Local-Global Evolutionary Patterns (2025)](https://www.sciencedirect.com/science/article/abs/pii/S0031320325013068)
- [TiRGN — Original IJCAI 2022 Paper](https://www.ijcai.org/proceedings/2022/299)

---

## Question 2: Should the Ensemble Approach Be Replaced?

### Current Architecture

```
P_final = α(cameo) * P_llm + (1 - α(cameo)) * P_tkg
```

Two independent components, linearly blended, with per-CAMEO α learned via Brier score minimization. Hierarchical fallback: CAMEO root code → super-category → global → cold-start prior.

### Alternatives Evaluated

**1. Graph-Augmented LLM Prompting (TKG → LLM context, single output)**

Run TKG first, inject structural analysis into Gemini prompt, let LLM produce one graph-informed probability.

Rejected because:
- Destroys error decorrelation — the statistical basis of ensemble gain. When P_llm and P_tkg are computed independently, their errors are uncorrelated. Feeding TKG output into the LLM prompt causes anchoring bias, correlating errors.
- Replaces a calibrated, empirically-validated blend with "hope the LLM weighs evidence correctly." LLMs are poorly calibrated on probabilities — that's why the TKG exists.

**2. Stacking / Meta-Learner**

Train a small model on (P_llm, P_tkg, cameo_code, features) → P_final.

Rejected because:
- Sample size insufficient. ~8 predictions/day, weekly retraining → ~720 resolved predictions after 3 months, maybe 30-50 per CAMEO super-category. Non-linear meta-learners overfit catastrophically at this scale.
- Linear blend with calibrated α IS the correct complexity for this data regime. It's not a simplification — it's the maximum model complexity the data can support.

**3. LLM as Router / Judge**

Use LLM to decide whether to incorporate TKG prediction based on question characteristics.

Rejected because:
- Per-CAMEO α already does this empirically. When α → 0.65 for verbal cooperation, the system is already saying "mostly ignore TKG here." An LLM router reimplements the calibration layer with less rigor and more latency.

**4. Drop TKG Entirely (Pure LLM + Structured RAG)**

Extract graph features via SQL (degree trends, Goldstein averages, escalation indices) and inject as text into LLM prompt. No TiRGN training needed.

Rejected because:
- Loses learned temporal dynamics (GRU evolution, copy-generation patterns)
- Cold-start priors show material conflict α = 0.50, meaning TKG is equally weighted with LLM for the most consequential predictions (military action, mass violence). Dropping TKG removes a 50% contributor where it matters most.
- Frontier LLM Brier results (0.101-0.135) are aggregated across question types — individual model performance on low-base-rate geopolitical events is worse.

**5. TKG as Evidence Retriever (not probability source) — FUTURE CONSIDERATION**

Instead of producing P_tkg (scalar), TKG retrieves structurally similar historical patterns with outcomes. LLM reasons over patterns to produce P_final.

This is the most architecturally interesting alternative. It plays to each component's strength: TKG for structural pattern recognition, LLM for contextual reasoning. Addresses the lossy triple-mapping problem (natural language question → (s, r, o) triple).

Not recommended now because:
- Requires fundamentally different TKG inference pipeline (retrieve subgraphs, not score triples)
- Breaks the clean calibration framework (no separate P_tkg → can't optimize α)
- Needs entirely new calibration methodology

Flagged for v4.0+ consideration as resolved prediction corpus grows from hundreds to thousands.

### Why the Ensemble Is Hard to Beat

Three structural reasons:

1. **Error decorrelation.** Independent computation ensures LLM and TKG errors are uncorrelated, maximizing ensemble diversity. Any approach that feeds one component's output into the other reduces this.

2. **Right complexity for the data.** With ~30-50 resolved predictions per CAMEO category, linear blending with one parameter (α) is the maximum model complexity the data can support without overfitting.

3. **Operational transparency.** Every prediction stores `ensemble_info_json` with both component probabilities. The calibration optimizer can diagnose exactly where the system is miscalibrated. Architectures that obscure component contributions (like LLM reasoning over graph evidence) make this impossible.

### System-Level Benchmarks (Context)

| Benchmark | Brier Score | Source |
|-----------|-------------|--------|
| Superforecasters | 0.081 | ForecastBench 2025 |
| Best LLM (GPT-4.5) | 0.101 | ForecastBench 2025 |
| Human crowd aggregated | 0.149 | IARPA ACE |
| Uninformed baseline | 0.250 | Always predicting 50% |
| LLM projected parity with supers | Nov 2026 | ~0.016 Brier improvement/year |

Sources:
- [ForecastBench LLM Evaluation](https://forecastingresearch.substack.com/p/ai-llm-forecasting-model-forecastbench-benchmark)
- [LLM vs Superforecaster Convergence](https://markets.financialcontent.com/stocks/article/predictstreet-2026-1-18-the-great-forecast-convergence-ai-closing-the-20-gap-on-human-superforecasters)
- [LEAP Wave 5 Report](https://leap.forecastingresearch.org/reports/wave5)
- [arXiv 2507.04562 — LLM Forecasting Evaluation](https://arxiv.org/html/2507.04562v3)
