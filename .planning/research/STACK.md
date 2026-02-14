# Technology Stack: v2.0 Operationalization & Forecast Quality

**Project:** Geopolitical Forecasting Engine v2.0
**Researched:** 2026-02-14
**Constraint:** RTX 3060 12GB, Python 3.11+, JAX/jraph for TKG, Gemini API for LLM
**Overall Confidence:** MEDIUM-HIGH (TKG algorithms LOW-MEDIUM, supporting libraries HIGH)

---

## Executive Summary

v2.0 adds five capability dimensions to the existing v1.1 stack: (1) a Streamlit web frontend, (2) scheduled automation via systemd, (3) a superior TKG algorithm, (4) dynamic per-CAMEO calibration, and (5) micro-batch GDELT ingest. The most consequential finding from this research is twofold:

**First:** jraph (the JAX graph neural network library from Google DeepMind) was **archived on 2025-05-21** and is now read-only. The project currently depends on jraph v0.0.6.dev0. While the dependency is shallow (only `GraphsTuple` NamedTuple and `segment_sum` wrapper are used, both trivially replaceable with native JAX), this must be addressed in v2.0 regardless of TKG algorithm choice.

**Second:** No TKG algorithm superior to RE-GCN has a JAX implementation. All candidates (HiSMatch, TiRGN, TRCL, CENET) are PyTorch+DGL. Porting any of them to JAX is a 2-4 week engineering effort. The recommendation is to port TiRGN (best effort/reward ratio) or, if resources are tight, to keep RE-GCN and focus the TKG investment on eliminating the jraph dependency and optimizing the existing encoder.

The remaining dimensions (Streamlit, scheduling, calibration, ingest) are straightforward library additions with well-established solutions.

---

## CRITICAL: jraph Archived — Mandatory Migration

**Status:** jraph was archived by Google DeepMind on 2025-05-21. No further updates, bug fixes, or security patches.

**Impact on this project:** The current `regcn_jraph.py` imports:
- `jraph.GraphsTuple` — a NamedTuple with 7 fields (nodes, edges, senders, receivers, n_node, n_edge, globals). Trivially replaceable with a local NamedTuple definition.
- `jraph.segment_sum` — a thin wrapper around `jax.ops.segment_sum`, which is a JAX built-in.

**Migration effort:** ~2 hours. Define a local `GraphsTuple` NamedTuple and replace `jraph.segment_sum` with `jax.ops.segment_sum`. No algorithmic changes. This should be done in the first v2.0 phase regardless of other decisions.

**Alternative considered:** JraphX (v0.0.4) — an unofficial community successor with PyG-inspired API. Not recommended. It is experimental (version 0.0.4), community-maintained with no guarantees, and the project's actual jraph usage is trivial enough that bringing in a new dependency is overkill.

**Sources:**
- [jraph GitHub (archived)](https://github.com/google-deepmind/jraph) — HIGH confidence
- [jax.ops.segment_sum docs](https://docs.jax.dev/en/latest/_autosummary/jax.ops.segment_sum.html) — HIGH confidence

---

## 1. TKG Algorithm Candidates (CRITICAL)

### Current Baseline: RE-GCN

- **ICEWS14 MRR:** 42.00% (from TRCL benchmark paper; geopol.md states 40.4% — the discrepancy likely reflects different evaluation protocols or hyperparameter tuning)
- **Architecture:** R-GCN spatial encoder + GRU temporal evolution + MLP decoder
- **Framework:** JAX/jraph/Flax NNX (already implemented)
- **Training characteristics:** ~50 epochs on 30 days of GDELT data, fits RTX 3060 12GB

### Candidate Comparison

| Algorithm | ICEWS14 MRR | ICEWS18 MRR | H@1 (14) | H@10 (14) | Framework | JAX Port? | Porting Effort |
|-----------|------------|------------|----------|-----------|-----------|-----------|----------------|
| **TRCL** | **45.07%** | **33.78%** | 34.71% | 65.37% | PyTorch | No | HIGH (3-4 weeks) |
| **TiRGN** | **44.04%** | **33.66%** | 33.83% | 63.84% | PyTorch | No | MEDIUM (2-3 weeks) |
| **HiSMatch** | ~46.4%* | — | — | — | PyTorch/DGL | No | HIGH (3-4 weeks) |
| **CEN** | 42.20% | 31.50% | 32.08% | 61.31% | PyTorch | No | MEDIUM (2-3 weeks) |
| **RE-GCN** | 42.00% | 30.58% | 31.63% | 61.65% | **JAX** | **Yes (current)** | N/A |
| **CENET** | 32.42% | 26.40% | 24.56% | 48.13% | PyTorch | No | MEDIUM |

\*HiSMatch's 46.4% MRR is from geopol.md; the TRCL benchmark paper excludes HiSMatch from its comparison table. This number could not be independently verified against the exact same evaluation protocol used for the other models. Confidence: LOW.

**Sources:**
- TRCL benchmark table: [PeerJ Computer Science e2595](https://peerj.com/articles/cs-2595/) — HIGH confidence (peer-reviewed, 2025)
- HiSMatch paper: [EMNLP 2022 Findings](https://aclanthology.org/2022.findings-emnlp.542.pdf) — MEDIUM confidence (self-reported, different eval protocol may apply)
- TiRGN: [IJCAI 2022](https://www.ijcai.org/proceedings/2022/299) — HIGH confidence
- RE-GCN: [SIGIR 2021](https://github.com/Lee-zix/RE-GCN) — HIGH confidence

### Detailed Analysis Per Candidate

#### HiSMatch (claimed 46.4% MRR on ICEWS14)

**Architecture:** Three structure encoders — query history encoder, candidate history encoder, background knowledge encoder. Uses CompGCN (Composition-based GCN) for graph encoding. Frames TKG reasoning as a **matching task** between query and candidate historical subgraphs.

**Framework:** PyTorch + DGL (Deep Graph Library). Same author as RE-GCN (Zixuan Li / Lee-zix). Codebase shares data loading infrastructure with RE-GCN.

**JAX Porting Assessment:**
- **Difficulty: HIGH.** Three separate graph encoders means 3x the porting surface compared to RE-GCN. The matching architecture requires pairwise similarity computation between subgraph representations, which is more complex than RE-GCN's straightforward encode-then-decode pattern.
- **DGL dependency:** DGL's message passing API is more opinionated than jraph. Porting requires reimplementing DGL's `update_all`, `apply_edges`, and `apply_nodes` primitives in terms of `jax.ops.segment_sum` and scatter operations.
- **Historical structure extraction:** HiSMatch requires pre-extracting per-entity historical subgraphs via `get_repetitive_history.py` and `get_history_dict.py`. This preprocessing is Python/NumPy, trivially portable.

**Memory estimate:** Three encoders + matching layer = ~1.5-2x RE-GCN parameter count at same embedding dim. With 200-dim embeddings and 7K entities (ICEWS14 scale), fits in 12GB. With GDELT-scale entities (7.7K entities, 240 relations, 2.2M+ triples), unknown — needs profiling.

**Training time:** No published training time data. Given 3 encoders, expect 2-3x RE-GCN training time per epoch.

**Confidence in 46.4% MRR:** LOW. The number from the original paper may use a different evaluation protocol (e.g., time-aware filtering vs standard filtering). The TRCL paper (2025) — which tested 10 baselines — does not include HiSMatch in its comparison, which is suspicious for a model claiming SOTA.

#### TiRGN (44.04% MRR on ICEWS14)

**Architecture:** Dual-path encoder — local recurrent encoder (R-GCN + GRU, nearly identical to RE-GCN) PLUS global history encoder that captures repeated historical facts. Combines via learned alpha: `P = alpha * P_local + (1-alpha) * P_global`.

**Framework:** PyTorch. Official code at [Liyyy2122/TiRGN](https://github.com/Liyyy2122/TiRGN).

**JAX Porting Assessment:**
- **Difficulty: MEDIUM.** The local encoder is structurally identical to RE-GCN (already implemented in JAX). The global history encoder is a lookup + aggregation over historical fact vocabulary — implementable with `jax.numpy` operations and embedding lookups. The 1D convolution decoder (`TimeConvTransE`) replaces the MLP decoder but is straightforward in Flax NNX.
- **Incremental path:** Implement global history encoder as an add-on to existing RE-GCN code. ~60% of the local encoder code can be reused directly.
- **History snapshot lengths:** Optimal for GDELT is 10 snapshots (training), 11 (testing). This is manageable for memory.

**Memory estimate:** ~1.3-1.5x RE-GCN due to global history matrix. The global history encoder stores a historical vocabulary matrix of size `(num_entities, num_relations)` for each historical timestamp. For GDELT (7.7K entities, 240 relations), this is ~1.85M entries per snapshot * 10 snapshots = ~18.5M entries. At float32, ~74MB. Fits easily in 12GB.

**Training time:** Approximately 1.2-1.5x RE-GCN per epoch (global encoder adds a forward pass but no graph convolution). With the GDELT benchmark (2.2M triples), published results show models of this class train one epoch in ~30 minutes on a single GPU.

**Verdict:** Best candidate for porting. The local encoder is RE-GCN (already done), and the global encoder is a self-contained module.

#### TRCL (45.07% MRR on ICEWS14)

**Architecture:** TiRGN-like dual encoder + contrastive learning loss. The contrastive objective distinguishes historical dependencies from non-historical interference.

**Framework:** PyTorch. **No public code repository found.** The PeerJ paper does not link to a GitHub repo. This is a showstopper.

**JAX Porting Assessment:** Cannot port without source code. Even if obtained, the contrastive learning loss adds training complexity beyond TiRGN.

**Verdict:** Eliminated. No public implementation. Even if code becomes available, marginal gain over TiRGN (+1.03% MRR) does not justify the additional contrastive learning complexity.

### TKG Algorithm Recommendation

**Primary recommendation: Port TiRGN to JAX/Flax NNX.**

Rationale:
1. **+2.04% MRR over RE-GCN** on ICEWS14 (44.04% vs 42.00%) — meaningful improvement.
2. **Reuses ~60% of existing RE-GCN code** — the local encoder IS RE-GCN.
3. **Global history encoder is self-contained** — can be developed and tested independently.
4. **No DGL dependency** — TiRGN's PyTorch code uses standard PyTorch, not DGL's graph API.
5. **Memory-feasible** — estimated 1.3-1.5x RE-GCN, well within 12GB.
6. **Proven on GDELT** — published results on the GDELT benchmark dataset.

**Fallback recommendation: Keep RE-GCN, invest in optimization.**

If porting resources are insufficient, keep RE-GCN but:
1. Eliminate jraph dependency (mandatory regardless)
2. Add ConvTransE decoder (replaces MLP, typically +1-2% MRR)
3. Tune hyperparameters more aggressively with the freed engineering time

**Do NOT attempt HiSMatch porting.** The 3-encoder architecture is too complex for the marginal gain over TiRGN, and the claimed 46.4% MRR is unverified against the same evaluation protocol.

---

## 2. Streamlit Web Frontend

### Core Package

| Package | Version | Purpose | Rationale |
|---------|---------|---------|-----------|
| `streamlit` | `>=1.54.0` | Web frontend framework | Latest stable (2026-02-04). Python 3.10-3.14. `@st.fragment(run_every=...)` for real-time updates. Experimental `st.App` ASGI entry point for middleware. |

### Key Streamlit Features for v2.0

**Real-time chart updates:**
- `@st.fragment(run_every="30s")` decorator on calibration/Brier score plots — reruns only the fragment, not the full page
- `st.session_state` persists user selections across auto-refreshes
- Built-in `st.line_chart`, `st.area_chart` with improved hover performance (v1.54)

**Interactive query input:**
- `st.text_input` + `st.button` for forecast queries
- `st.spinner` / `st.status` for loading states during Gemini API calls
- `st.expander` for reasoning chain display

**Session management:**
- `st.session_state` handles per-user state natively
- No separate session backend needed for demo-scale traffic

### Rate Limiting

Streamlit's new `st.App` (experimental, v1.53+) enables ASGI middleware integration. Use `slowapi` for rate limiting:

| Package | Version | Purpose | Rationale |
|---------|---------|---------|-----------|
| `slowapi` | `>=0.1.9` | Rate limiting via ASGI middleware | Starlette/FastAPI compatible. `SlowAPIASGIMiddleware` for async. Uses in-memory store by default (sufficient for single-server). |

**Architecture for rate limiting with st.App:**
```python
import streamlit as st
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.middleware import SlowAPIASGIMiddleware

limiter = Limiter(key_func=get_remote_address)
app = st.App(middleware=[SlowAPIASGIMiddleware])
app.state.limiter = limiter
```

**Caveat:** `st.App` is experimental as of v1.54. If it proves unstable, the fallback is application-level rate limiting via `st.session_state` counters — cruder but functional. This is a LOW confidence recommendation pending `st.App` stabilization.

### Extensions

| Package | Version | Purpose | When |
|---------|---------|---------|------|
| `streamlit-autorefresh` | `>=1.0.1` | Timer-based auto-refresh for dashboard | Fallback if `@st.fragment(run_every=)` is insufficient. Not needed initially. |
| `plotly` | `>=6.0.0` | Advanced interactive charts | If built-in Streamlit charts lack calibration plot customization. Already a Streamlit optional dep. |

**Sources:**
- [Streamlit 2026 release notes](https://docs.streamlit.io/develop/quick-reference/release-notes/2026) — HIGH confidence
- [Streamlit fragments docs](https://docs.streamlit.io/develop/concepts/architecture/fragments) — HIGH confidence
- [slowapi GitHub](https://github.com/laurentS/slowapi) — HIGH confidence
- [Streamlit PyPI](https://pypi.org/project/streamlit/) — HIGH confidence

---

## 3. Micro-batch GDELT Processing

### Scheduling Approach

**Recommendation: systemd timers, not APScheduler.**

| Approach | Verdict | Rationale |
|----------|---------|-----------|
| **systemd timers** | **RECOMMENDED** | OS-level, survives process crashes, zero Python dependency, native journald logging. The ingest process is a short-lived script (fetch + process + insert), not a long-running daemon. systemd timers are the correct abstraction for "run this script every 15 minutes." |
| APScheduler 3.11 | Not recommended | Requires a continuously running Python process. If the process crashes, scheduling stops. APScheduler 4.0 is still alpha (v4.0.0a6). Adds complexity without benefit for a periodic batch job. |
| `schedule` library | Not recommended | Already in pyproject.toml but same problem as APScheduler — requires persistent process. Single-threaded by default; long-running jobs block the scheduler. |
| cron | Acceptable fallback | Works, but systemd timers provide better logging integration, dependency management, and failure handling. |

**Implementation:**
```ini
# /etc/systemd/system/geopol-ingest.service
[Unit]
Description=GDELT micro-batch ingest
After=network-online.target

[Service]
Type=oneshot
WorkingDirectory=/home/kondraki/personal/geopol
ExecStart=/home/kondraki/personal/geopol/.venv/bin/python scripts/ingest_gdelt.py
User=kondraki
StandardOutput=journal
StandardError=journal
SyslogIdentifier=geopol-ingest

# /etc/systemd/system/geopol-ingest.timer
[Unit]
Description=Run GDELT ingest every 15 minutes

[Timer]
OnCalendar=*:0/15
Persistent=true

[Install]
WantedBy=timers.target
```

### Incremental Graph Updates

No new library needed. The existing `NetworkX` graph supports incremental `add_edges_from()`. The ingest script should:
1. Fetch new GDELT 15-minute slice
2. Process through existing `data_processor.py` pipeline
3. Insert new events into SQLite
4. Add new triples to the in-memory graph (or rebuild from SQLite if graph is stale)

For SQLite concurrent access (ingest writes while Streamlit reads):
- SQLite WAL mode (Write-Ahead Logging) — already sufficient for single-writer/multiple-reader
- Set via `PRAGMA journal_mode=WAL;` on connection

**No new packages required for this dimension.**

---

## 4. Dynamic Per-CAMEO Calibration

### Existing Libraries (Already in Stack)

| Package | Current Version | v2.0 Usage |
|---------|----------------|------------|
| `netcal` | `>=1.3.5` (in pyproject.toml) | ECE computation, reliability diagrams. Already integrated in `src/evaluation/calibration_metrics.py`. |
| `scipy` | `>=1.11.0` (in pyproject.toml) | `scipy.optimize.minimize` for per-CAMEO weight optimization. Bounded L-BFGS-B with box constraints `alpha_i in [0.0, 1.0]`. |
| `scikit-learn` | `>=1.3.0` (in pyproject.toml) | `IsotonicRegression` for non-parametric calibration curves. Already available. |
| `numpy` | `>=1.24.0` (in pyproject.toml) | Array operations for Brier score decomposition. |

### New Libraries: None

The dynamic calibration system does not require new packages. The per-CAMEO alpha optimization is:

```python
from scipy.optimize import minimize

def optimize_alpha_for_category(
    llm_probs: np.ndarray,
    tkg_probs: np.ndarray,
    outcomes: np.ndarray,  # 0 or 1
) -> float:
    """Find optimal alpha for one CAMEO root category."""
    def brier_score(alpha):
        ensemble = alpha * llm_probs + (1 - alpha) * tkg_probs
        return np.mean((ensemble - outcomes) ** 2)

    result = minimize(
        brier_score,
        x0=0.6,  # Current default
        bounds=[(0.0, 1.0)],
        method='L-BFGS-B',
    )
    return result.x[0]
```

This runs in milliseconds for any reasonable number of historical predictions. No online learning framework needed — the optimization is batch (run after each day's outcomes are resolved).

**For reliability diagrams in Streamlit:**
- `netcal` provides `ReliabilityDiagram` visualization
- Alternatively, compute bin-level stats with `netcal.metrics.ECE` and plot with `matplotlib` (already in stack) or Streamlit's native charts

**Sources:**
- [netcal PyPI](https://pypi.org/project/netcal/) — HIGH confidence (v1.3.6, Aug 2024)
- [scipy.optimize.minimize docs](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html) — HIGH confidence

---

## 5. Scheduling & Automation

### Daily Forecast Automation

**Recommendation: systemd timer (same pattern as GDELT ingest)**

```ini
# /etc/systemd/system/geopol-forecast.service
[Unit]
Description=Daily geopolitical forecast generation
After=network-online.target

[Service]
Type=oneshot
WorkingDirectory=/home/kondraki/personal/geopol
ExecStart=/home/kondraki/personal/geopol/.venv/bin/python scripts/daily_forecast.py
User=kondraki
StandardOutput=journal
StandardError=journal
SyslogIdentifier=geopol-forecast
TimeoutStartSec=1800  # 30 minutes max

# /etc/systemd/system/geopol-forecast.timer
[Timer]
OnCalendar=*-*-* 06:00:00
Persistent=true

[Install]
WantedBy=timers.target
```

### Health Monitoring & Alerting

**Recommendation: systemd journal + simple Python health check script + email alerts**

For a single-server deployment, heavyweight monitoring (Prometheus, Grafana, Datadog) is overkill. The appropriate stack:

| Component | Tool | Rationale |
|-----------|------|-----------|
| Log aggregation | **systemd journald** (built-in) | All services log via `StandardOutput=journal`. Query with `journalctl -u geopol-*`. Structured logging via Python's `logging` module with `systemd.journal.JournalHandler`. |
| Health checks | **Custom Python script** on systemd timer | Check: (1) SQLite DB freshness (last event timestamp), (2) last forecast timestamp, (3) Streamlit process alive, (4) disk space. |
| Alerting | **Python `smtplib`** or `apprise` | Send email/notification on health check failure. `apprise` supports 90+ notification services (Slack, Telegram, email) with a single API. |
| Process management | **systemd** | Streamlit runs as a systemd service with `Restart=on-failure`. |

| Package | Version | Purpose | When |
|---------|---------|---------|------|
| `apprise` | `>=1.9.0` | Multi-channel alerting (email, Slack, Telegram) | Only if email-only alerting is insufficient. Lightweight, zero config for basic channels. |

**For systemd journal integration:**

| Package | Version | Purpose | Rationale |
|---------|---------|---------|-----------|
| `systemd-python` | `>=235` | `JournalHandler` for structured logging to journald | Optional. Python's standard `logging` to stdout already goes to journald via systemd. Only needed if structured metadata (key-value pairs) in journal entries is desired. |

**Sources:**
- [systemd journal Python logging](https://lincolnloop.com/blog/logging-systemds-journal-python/) — HIGH confidence
- [Centralized logging with journald](https://www.andrewkroh.com/linux/2025/12/19/centralized-logging-with-journald.html) — HIGH confidence
- [APScheduler PyPI](https://pypi.org/project/APScheduler/) — HIGH confidence (3.11.2 stable, 4.0 alpha)

---

## Recommended Stack Additions Summary

### Must Add (New Dependencies)

| Package | Version | Purpose | Phase |
|---------|---------|---------|-------|
| `streamlit` | `>=1.54.0` | Web frontend | Phase 1 (Frontend) |
| `slowapi` | `>=0.1.9` | Rate limiting middleware | Phase 1 (Frontend) |

### May Add (Conditional)

| Package | Version | Purpose | Condition |
|---------|---------|---------|-----------|
| `apprise` | `>=1.9.0` | Multi-channel alerting | If email-only alerting is insufficient |
| `systemd-python` | `>=235` | Structured journal logging | If structured metadata in journal entries is needed |
| `plotly` | `>=6.0.0` | Advanced interactive charts | If built-in Streamlit charts are insufficient |

### Must Remove / Replace

| Package | Action | Rationale |
|---------|--------|-----------|
| `jraph>=0.0.6.dev0` | **Remove** — replace with local NamedTuple + `jax.ops.segment_sum` | Archived library, no future updates. Dependency is shallow (2 API calls). |
| `schedule>=1.2.0` | **Remove** | Replaced by systemd timers. Not used in any current source code. |

### Must Update

| Package | Current | Target | Rationale |
|---------|---------|--------|-----------|
| (none) | — | — | Existing pinnings are adequate. |

### Explicitly NOT Adding

| Package | Reason |
|---------|--------|
| `APScheduler` | systemd timers are superior for periodic batch jobs on a single server |
| `celery` / `dramatiq` | Task queue is overkill for 2 cron-like jobs |
| `prometheus-client` / `grafana` | Heavyweight monitoring for a single-server demo |
| `DGL` (Deep Graph Library) | Would only be needed for HiSMatch port, which is not recommended |
| `transformers` / `bitsandbytes` / `peft` | Llama integration cancelled per commit c7786d4 |
| `streamlit-autorefresh` | `@st.fragment(run_every=)` covers the use case natively |
| `JraphX` | Experimental (v0.0.4), community-maintained; project's jraph usage is trivial enough to inline |
| `river` / `vowpalwabbit` | Online learning frameworks — the calibration optimization is batch, not streaming |

---

## Installation Commands

```bash
# Add new dependencies
uv add "streamlit>=1.54.0" "slowapi>=0.1.9"

# Remove deprecated dependencies
uv remove schedule jraph

# Verify
uv run python -c "
import streamlit
print(f'Streamlit: {streamlit.__version__}')
import jax
print(f'JAX: {jax.__version__}')
print(f'jax.ops.segment_sum available: {hasattr(jax.ops, \"segment_sum\")}')
"
```

**Note:** After removing jraph, the import in `src/training/models/regcn_jraph.py` will break. The jraph elimination must be done atomically with the code changes to replace `jraph.GraphsTuple` and `jraph.segment_sum`.

---

## Integration Points with Existing Stack

### JAX/Flax NNX (TKG)

- TiRGN port builds on existing `regcn_jraph.py` architecture
- Same `nnx.Module` patterns, same `optax` optimizer, same checkpoint format
- Global history encoder is a new `nnx.Module` that composes with existing encoder
- `jax.ops.segment_sum` replaces `jraph.segment_sum` (identical API, different import)

### SQLite (Storage)

- Micro-batch ingest writes new events to existing SQLite schema
- WAL mode enables concurrent reads (Streamlit) and writes (ingest)
- Prediction store (`src/calibration/prediction_store.py`) already exists for outcome tracking
- Per-CAMEO alpha values stored in a new SQLite table (20 rows, trivial)

### Gemini API (LLM)

- Streamlit interactive queries call existing `gemini_client.py`
- Rate limiting at Streamlit layer protects against API cost abuse
- No changes to Gemini integration itself

### NetworkX (Graph)

- Micro-batch ingest adds edges incrementally via `add_edges_from()`
- Graph reconstruction from SQLite as fallback if in-memory graph drifts
- No new graph library needed

---

## Confidence Assessment

| Area | Confidence | Rationale |
|------|------------|-----------|
| jraph archival | HIGH | Verified directly on GitHub — archived 2025-05-21, read-only |
| TKG benchmarks (TRCL paper) | HIGH | Peer-reviewed 2025 paper with reproducible comparison table |
| HiSMatch MRR claim (46.4%) | LOW | Self-reported, not reproduced in subsequent benchmark papers, different eval protocol possible |
| TiRGN porting feasibility | MEDIUM | Architecture is structurally similar to RE-GCN, but no one has published a JAX port — estimation based on code structure analysis |
| TiRGN memory on GDELT-scale | LOW | Theoretical estimate only; needs profiling on actual GDELT data |
| Streamlit features (fragments, st.App) | HIGH | Verified in official 2025-2026 release notes |
| st.App rate limiting via slowapi | LOW | st.App is experimental; integration pattern not battle-tested |
| systemd timers | HIGH | Standard Linux infrastructure, well-documented |
| Dynamic calibration (scipy) | HIGH | scipy.optimize is the gold standard for bounded optimization |
| netcal ECE/reliability | HIGH | Already integrated and tested in codebase |
| GDELT training time estimates | LOW | Extrapolated from TGL paper and general benchmarks, not measured on this codebase |

---

## Sources

### Peer-Reviewed Papers (HIGH confidence)
- [TRCL: Recurrent encoding + contrastive learning for TKG](https://peerj.com/articles/cs-2595/) — PeerJ Computer Science, 2025
- [TiRGN: IJCAI 2022](https://www.ijcai.org/proceedings/2022/299)
- [HiSMatch: EMNLP 2022 Findings](https://aclanthology.org/2022.findings-emnlp.542.pdf)
- [RE-GCN: SIGIR 2021](https://github.com/Lee-zix/RE-GCN)

### Official Documentation (HIGH confidence)
- [Streamlit 2026 release notes](https://docs.streamlit.io/develop/quick-reference/release-notes/2026)
- [Streamlit fragments](https://docs.streamlit.io/develop/concepts/architecture/fragments)
- [Streamlit PyPI](https://pypi.org/project/streamlit/) — v1.54.0, Python >=3.10
- [APScheduler PyPI](https://pypi.org/project/APScheduler/) — v3.11.2 stable, v4.0.0a6 alpha
- [netcal PyPI](https://pypi.org/project/netcal/) — v1.3.6
- [scipy.optimize.minimize](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html)
- [jax.ops.segment_sum](https://docs.jax.dev/en/latest/_autosummary/jax.ops.segment_sum.html)

### GitHub Repositories (HIGH confidence)
- [jraph (archived)](https://github.com/google-deepmind/jraph) — archived 2025-05-21
- [TiRGN official code](https://github.com/Liyyy2122/TiRGN)
- [HiSMatch official code](https://github.com/Lee-zix/HiSMatch)
- [CENET official code](https://github.com/xyjigsaw/CENET)
- [slowapi](https://github.com/laurentS/slowapi)
- [JraphX](https://dirt.design/jraphx/) — v0.0.4, experimental

### Web Sources (MEDIUM confidence)
- [Streamlit st.App ASGI feedback](https://github.com/streamlit/streamlit/issues/13600)
- [systemd journal Python logging guide](https://lincolnloop.com/blog/logging-systemds-journal-python/)
- [JAX vs PyTorch porting guide](https://cloud.google.com/blog/products/ai-machine-learning/guide-to-jax-for-pytorch-developers)
