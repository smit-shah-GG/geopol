# Feature Research: v2.0 Operationalization & Forecast Quality

**Domain:** AI-powered geopolitical forecasting engine (public demo + operational automation)
**Researched:** 2026-02-14
**Confidence:** MEDIUM-HIGH (benchmark data verified from published papers; operational patterns verified from official documentation; dashboard patterns synthesized from competitor analysis)

---

## 1. TKG Algorithm Landscape

### State of the Art (2024-2026)

Verified benchmark results on ICEWS14 (the standard TKG forecasting benchmark):

| Algorithm | ICEWS14 MRR | ICEWS05-15 MRR | GDELT MRR | Architecture | Year |
|-----------|-------------|----------------|-----------|--------------|------|
| **ACDm** | ~45.8 (est) | -- | ~22.6 (est) | Autoregressive-conditioned diffusion | 2025 |
| **TRCL** | **45.07** | **50.12** | **21.85** | Recurrent encoding + contrastive learning | 2025 |
| **HisMatch** | **46.42** | **52.85** | **22.01** | Historical structure matching + dual encoder | 2022 |
| **TiRGN** | 44.04 | 50.04 | 21.67 | Local recurrent + global history encoders | 2022 |
| **RE-GCN** | 42.00 | 48.03 | 19.69 | R-GCN spatial + GRU temporal | 2021 |
| **xERTE** | ~40.0 | -- | -- | Iterative subgraph expansion w/ attention | 2021 |
| **CyGNet** | 35.05 | 36.81 | 18.48 | Copy-generation network | 2021 |

**Confidence:** HIGH for RE-GCN, TiRGN, HisMatch, TRCL (verified from Table 1 of the Negative-Aware Diffusion paper, arxiv 2602.08815, and TRCL paper PeerJ cs-2595). MEDIUM for ACDm (abstract claims 1.8% over TiRGN but exact table not extracted). LOW for xERTE exact number (training-data-era knowledge, not re-verified from current sources).

**Sources:**
- [TRCL paper (PeerJ)](https://peerj.com/articles/cs-2595/) -- Table 3-6 verified
- [Negative-Aware Diffusion paper (arxiv)](https://arxiv.org/html/2602.08815) -- Table 1 verified
- [HisMatch (EMNLP 2022)](https://aclanthology.org/2022.findings-emnlp.542.pdf) -- cited MRR confirmed
- [TiRGN (IJCAI 2022)](https://www.ijcai.org/proceedings/2022/299)
- [CyGNet (AAAI 2021)](https://github.com/CunchaoZ/CyGNet)

### Algorithm Replacement Analysis for v2.0

**Current:** RE-GCN (JAX/jraph, 200-dim, 42.00% MRR on ICEWS14)

**Candidates ranked by accuracy:**

| Candidate | MRR Gain | Framework | JAX Port Effort | RTX 3060 Fit | Weekly Retrain | Recommendation |
|-----------|----------|-----------|-----------------|--------------|----------------|----------------|
| HisMatch | +4.42 MRR (10.6%) | PyTorch | HIGH -- dual structure encoder, preprocessing generates per-entity history dicts | RISK -- preprocesses per-entity historical subgraphs, memory scales with entity count | UNCLEAR -- no incremental training, preprocessing is expensive | BEST accuracy, WORST portability |
| TiRGN | +2.04 MRR (4.9%) | PyTorch | MEDIUM -- local+global recurrent is structurally similar to RE-GCN | GOOD -- 200-dim hidden, GCN+GRU similar to current model | UNCLEAR -- no incremental training documented | BEST accuracy/effort tradeoff |
| TRCL | +3.07 MRR (7.3%) | PyTorch | MEDIUM-HIGH -- TiRGN base + contrastive learning head | GOOD -- similar memory profile to TiRGN | No incremental | Good accuracy but harder to port than base TiRGN |
| xERTE | ~-2.0 MRR | PyTorch | HIGH -- completely different architecture (subgraph expansion) | GOOD -- inference is lightweight | No incremental | BEST explainability but WORSE accuracy |
| CyGNet | -6.95 MRR | PyTorch | LOW -- simple copy-generation | GOOD -- lightweight | No incremental | REJECT -- worse than current RE-GCN |
| ACDm | ~+3.8 MRR (est) | Unknown | VERY HIGH -- diffusion model architecture | RISK -- diffusion models are memory-intensive | No incremental | Too novel, implementation risk too high |

**Recommendation: TiRGN.**

Rationale:
1. TiRGN's local recurrent encoder (GCN + GRU per timestep) is architecturally similar to RE-GCN (R-GCN + GRU), making JAX/jraph porting tractable. The global history encoder adds a repetition-pattern vocabulary (conceptually similar to CyGNet's copy mechanism) that captures periodic geopolitical events -- this is the primary innovation.
2. The +2.04 MRR gain (4.9% relative) is meaningful and validated across all benchmarks (ICEWS14, ICEWS05-15, GDELT).
3. Hidden dimension 200 matches current RE-GCN configuration -- no embedding dimension changes needed in the data pipeline.
4. All reference implementations are PyTorch. No TKG algorithm has a JAX implementation. This is unavoidable; every candidate requires porting.
5. HisMatch offers +4.42 MRR but the preprocessing step (generating per-entity historical structure dictionaries) adds significant complexity and memory overhead that may not fit RTX 3060 12GB for large GDELT graphs.

**Operational impact of TKG replacement:**
- **Data format:** No change. TiRGN uses the same (s, r, o, t) quadruple format as RE-GCN. The existing `DataAdapter` converts NetworkX graphs to this format already.
- **Retraining pipeline:** Needs new `tirgn_jraph.py` model module + updated `train_jraph.py`. Additionally needs `get_history.py`-equivalent preprocessing to build repetition vocabularies (the global encoder component).
- **Embedding dimensions:** No change. TiRGN uses 200-dim hidden by default, matching current RE-GCN config.
- **Inference changes:** The `TKGPredictor` wrapper needs to call TiRGN instead of RE-GCN, but the API surface (predict top-k links for query) is identical.
- **Incremental retraining:** None of the candidates support it. All require full retrain from scratch. Weekly full retrain on large GDELT data must fit within compute budget.

### What "No Incremental Retraining" Means Operationally

Every TKG algorithm in the current landscape requires retraining from scratch on the full temporal graph. There is no published work on continual/incremental TKG model updates. This means:
- Weekly retraining reprocesses the entire event history
- Training time scales with graph size (entities x relations x timesteps)
- The RTX 3060 12GB constraint caps the graph size that can be trained in a reasonable window
- Micro-batch ingest grows the graph continuously, but the model only "sees" new data after the next retrain cycle

---

## 2. ML Demo Dashboard

### Competitor Analysis

| Platform | Type | Visualization | Interactivity | Credibility Signal | Access |
|----------|------|---------------|---------------|-------------------|--------|
| **Metaculus** | Prediction aggregation | Calibration plots, probability distributions, time-series of forecast updates, Brier score displays | Question browse/filter, forecaster input, track record pages | Leaderboards, calibration curves, peer scores, named forecasters | Public |
| **ACLED CAST** | Conflict forecast | Map view + table view, line graphs with historical overlay, bar charts of driving factors, forecast trends | Country filter, event type disaggregation, time range slider, customizable moving average horizon | Methodology page, named researchers, institutional backing, downloadable data | Registered (free) |
| **Good Judgment Open** | Prediction market | Crowd probability, historical forecast movement, question resolution | Browse questions, submit forecasts, view track record | Superforecaster methodology, IARPA heritage, leaderboards | Public |
| **New Lines Forecast Monitor** | Expert analysis | Thematic categories (Russia/Ukraine, Trade, Middle East, etc.), weekly reports | Filter by date/author/topic, subscribe | Named analysts, institutional affiliation, publication consistency | Public + trial |
| **GeoQuant** | Commercial risk | 40+ indicators across 127 countries, hourly updates, heatmaps | Country selection, indicator drilling | Claimed 76% accuracy on major events, dual-stream methodology | Enterprise (paid) |

**Sources:**
- [Metaculus FAQ](https://www.metaculus.com/faq/)
- [Metaculus Design Language (Medium)](https://metaculus.medium.com/a-new-design-language-for-metaculus-c47c9133fca4)
- [ACLED CAST](https://acleddata.com/platform/cast-conflict-alert-system)
- [ACLED CAST Methodology](https://acleddata.com/methodology/cast-methodology)
- [New Lines Forecast Monitor](https://newlinesinstitute.org/forecast-monitor/)
- [Good Judgment Open](https://www.gjopen.com/)

### What Visitors Expect from a Credible Forecast Dashboard

Based on competitor analysis, visitors to a public forecast demo expect:

**Mandatory elements (table stakes):**
1. **Live forecasts with probabilities** -- not just "likely/unlikely" but numeric probabilities with clear uncertainty bounds
2. **Reasoning transparency** -- why the system predicts what it predicts (this is already the core value)
3. **Historical accuracy record** -- calibration plot showing predicted vs actual frequencies, Brier score display
4. **Methodology explanation** -- what models are used, what data sources, what the limitations are
5. **Question/topic organization** -- categorized by region, topic, or event type

**Credibility signals (what separates serious from gimmicky):**
- Calibration plots showing honest performance (including failures)
- Track record over time, not cherry-picked successes
- Clear methodology disclosure
- Acknowledgment of limitations and uncertainty
- Resolution criteria for each forecast
- Data provenance (GDELT attribution, model version)

**What makes a dashboard gimmicky:**
- Probabilities without confidence intervals
- No historical validation
- Vague methodology ("AI-powered")
- No track record
- Cherry-picked examples
- Real-time gimmicks without substance

---

## 3. Micro-Batch GDELT Ingestion

### GDELT 2.0 15-Minute Update Feed

**Confidence:** HIGH (verified from official GDELT documentation and data endpoints)

**Technical details:**
- **Update URL:** `http://data.gdeltproject.org/gdeltv2/lastupdate.txt` -- updated every 15 minutes, lists the 3 most recent files (Events, Mentions, GKG)
- **Master file list:** `http://data.gdeltproject.org/gdeltv2/masterfilelist.txt` -- complete archive index
- **File format:** Tab-delimited CSV (`.csv` extension but tab-separated)
- **Naming convention:** `YYYYMMDDHHMMSS.export.CSV.zip` (Events), `YYYYMMDDHHMMSS.mentions.CSV.zip` (Mentions), `YYYYMMDDHHMMSS.gkg.csv.zip` (GKG)
- **Volume per cycle:** Varies. Peak hours produce thousands of events per 15-minute window. GKG is the heaviest stream (2.5TB+ annually, so ~5GB/day or ~50MB per 15-min cycle compressed).
- **Access:** 100% free and open. No API keys. No rate limiting documented. Pull-based (you poll the lastupdate file).
- **Reliability:** Generally reliable but no SLA. The file simply appears or doesn't.

**GDELT date resolution caveat:** Even in v2.0, event date fields (`SQLDATE`, `MonthYear`, `Year`, `FractionDate`) record daily resolution only. To track events at 15-minute resolution, use the `DATEADDED` field which records the 15-minute update batch timestamp.

**Sources:**
- [GDELT Data Access](https://www.gdeltproject.org/data.html)
- [GDELT 2.0 Announcement](https://blog.gdeltproject.org/gdelt-2-0-our-global-world-in-realtime/)
- [GDELT Event Codebook V2.0](http://data.gdeltproject.org/documentation/GDELT-Event_Codebook-V2.0.pdf)

### Micro-Batch Ingestion Architecture

**Standard pattern for 15-minute event ingestion into knowledge graphs:**

1. **Poll-based ingestion:** Cron or scheduler polls `lastupdate.txt` every 15 minutes. Downloads new files if available. Idempotent -- if the file was already processed (based on filename), skip.
2. **Parse and filter:** Apply the same sampling/filtering logic as the daily batch (intelligent sampling for conflicts/diplomatic events). Most GDELT events are noise -- same filtering pipeline applies.
3. **Deduplication:** Critical across micro-batches. GDELT assigns `GlobalEventID` to each event. Use this as the dedup key. Additionally, the Mentions table tracks when the same event is re-mentioned -- do not double-count. Between micro-batches and daily dumps, dedup on `GlobalEventID` ensures no duplicates.
4. **Graph update:** Append new events to the existing graph. Entity resolution must handle new entities appearing in micro-batches (new actors, new locations). The graph grows incrementally but the TKG model is not retrained until the weekly cycle.
5. **Expected latency:** Event-to-graph-inclusion should be under 5 minutes (parse + filter + entity resolution + graph append). The 15-minute GDELT cycle is the bottleneck, not your processing.

**Deduplication strategy across micro-batches and daily dumps:**

| Dedup Layer | Key | Strategy |
|-------------|-----|----------|
| Within micro-batch | `GlobalEventID` | Hash set in memory during batch processing |
| Across micro-batches | `GlobalEventID` + processed-file ledger | SQLite table of processed event IDs, checked before insert |
| Micro-batch vs daily dump | `GlobalEventID` | Same table -- daily dump events already ingested via micro-batch are skipped |
| Semantic dedup | Entity pair + event code + date | The existing `deduplication.py` handles near-duplicate events referring to the same real-world occurrence from different sources |

**Sources:**
- [Event Deduplication in Stream Processing (RisingWave)](https://risingwave.com/blog/effective-deduplication-of-events-in-batch-and-stream-processing/)
- [Netflix Real-Time Graph Ingestion](https://netflixtechblog.com/how-and-why-netflix-built-a-real-time-distributed-graph-part-1-ingesting-and-processing-data-80113e124acc)

---

## 4. Dynamic Ensemble Calibration

### How Production Forecast Systems Handle Per-Category Calibration

**Confidence:** MEDIUM (synthesized from research papers and training-data knowledge; no single authoritative source for this specific niche)

**Approaches that work:**

| Approach | Mechanism | Data Requirement | Complexity | Fit for v2.0 |
|----------|-----------|------------------|------------|---------------|
| **Sliding window MLE** | Fit per-category alpha weights on most recent N resolved predictions using maximum likelihood | 50+ resolved predictions per category | LOW | BEST FIT |
| **Bayesian optimization** | Use BODE-style Bayesian optimization to find optimal alpha per category | 30+ resolved predictions per category | MEDIUM | Good but overkill |
| **Online learning** | Update weights incrementally as each prediction resolves (e.g., exponential moving average on Brier residuals) | Starts immediately, improves over time | LOW | Good for warm-start |
| **Hierarchical Bayesian** | Pool information across categories with a hierarchical prior, so sparse categories borrow strength from dense ones | 10+ per category (pooled) | HIGH | Theoretically ideal but complex |

**Recommendation: Sliding window MLE with hierarchical fallback.**

The v2.0 calibration system should:
1. Store all predictions with CAMEO root category labels in the existing `prediction_store.py` (already has `category` column)
2. When a category has 50+ resolved predictions, fit per-category alpha via MLE on the sliding window (last 90 days of resolved predictions)
3. When a category has <50 resolved predictions, fall back to the global alpha (pooled across all categories)
4. Transition is gradual: weight the per-category estimate by `min(n_resolved / 50, 1.0)` and the global estimate by the complement

**How much outcome data is needed:**
- **Minimum for per-category weights:** 50 resolved predictions per CAMEO root category. Below this, the variance in estimated alpha is too high to be useful.
- **Comfortable:** 100+ per category. At this point, per-category weights are statistically stable.
- **Accumulation rate:** With daily forecasts producing ~10-20 predictions across 20 CAMEO categories, and assuming ~1 week resolution lag, it takes approximately 2-3 months before the densest categories (Material Conflict, Verbal Cooperation) reach 50 resolved predictions. Sparse categories (e.g., CAMEO 16: Reduce Relations) may never accumulate enough data for per-category calibration.

**Handling sparse categories:**
- The 20 CAMEO root categories have highly uneven frequency distributions. Categories like 01 (Make Public Statement), 04 (Consult), 14 (Protest), 19 (Fight) will accumulate data quickly. Categories like 16 (Reduce Relations), 09 (Investigate) will be perpetually sparse.
- **Solution:** Group sparse categories into super-categories: (a) Verbal Cooperation (01-05), (b) Material Cooperation (06-09), (c) Verbal Conflict (10-13), (d) Material Conflict (14-20). This gives 4 calibration groups instead of 20, with much faster data accumulation per group.
- The system can start with 4 super-categories, then specialize to 20 root categories as data accumulates.

**Sources:**
- [Bayesian Optimization Dynamic Ensemble (ScienceDirect)](https://www.sciencedirect.com/science/article/abs/pii/S0020025522000135)
- [Calibrating Ensembles with Sparse Data (ScienceDirect)](https://www.sciencedirect.com/science/article/abs/pii/S0169207014001010)
- [Calibration and Sharpness -- Gneiting et al.](https://sites.stat.washington.edu/raftery/Research/PDF/Gneiting2007jrssb.pdf)

---

## 5. Forecast Automation Pipeline

### Daily Pipeline Architecture

**Confidence:** MEDIUM-HIGH (standard MLOps patterns, verified from multiple sources)

A robust daily forecast pipeline for this system should include:

```
06:00 UTC  [Ingest Check]     Verify 15-min micro-batch ingest ran successfully overnight
           |                   Check: graph node/edge count, last event timestamp
           v
06:15 UTC  [Data Validation]   Validate graph freshness: events from last 24h present?
           |                   Check: minimum event count threshold, entity coverage
           v
06:30 UTC  [Forecast Generation] Run Gemini + TKG ensemble on configured question set
           |                     For each active question: query both models, combine
           v
08:00 UTC  [Calibration]       Apply per-category calibration to raw ensemble outputs
           |                   Store predictions in prediction_store
           v
08:15 UTC  [Resolution Check]  Check if any past predictions can now be resolved
           |                   Query GDELT for ground truth on open predictions
           v
08:30 UTC  [Dashboard Update]  Push new forecasts to Streamlit dashboard cache
           |                   Update track record if resolutions occurred
           v
08:45 UTC  [Health Report]     Log pipeline metrics, alert on anomalies
```

### Failure Handling

| Failure Mode | Detection | Response | Retry Policy |
|-------------|-----------|----------|--------------|
| GDELT feed unavailable | Empty lastupdate.txt or download failure | Use stale graph (log warning), skip ingest but continue forecast | 3 retries with exponential backoff, then proceed with stale data |
| Gemini API failure | HTTP error or timeout | Fall back to TKG-only prediction (alpha=0.0) | 3 retries, 30s/60s/120s backoff. If all fail, TKG-only mode |
| TKG model failure | Exception during inference | Fall back to LLM-only prediction (alpha=1.0) | 1 retry (model errors are usually deterministic) |
| Both models fail | Both components error | Skip daily forecast, alert operator | No retry -- manual intervention needed |
| Resolution check fails | GDELT query error | Skip resolution, try again next day | Resolutions are not time-critical -- accumulate |
| Disk full | Write failures | Alert immediately, halt pipeline | No retry -- operator must free space |

### Scheduling Implementation

**Recommendation: systemd timer + Python script.**

Not APScheduler. The daily forecast pipeline is a batch process, not a long-running daemon. systemd timers are:
- More reliable than in-process schedulers (survive process crashes)
- Natively integrated with Linux logging (journald)
- Support OnFailure directives for alerting
- Trivial to inspect (`systemctl status geopol-daily-forecast.timer`)

APScheduler is appropriate for the 15-minute micro-batch ingest (which benefits from being a long-running process with in-memory state). But the daily forecast should be a standalone script invoked by systemd.

### Monitoring

**Lightweight stack for single-server deployment:**

For a single-server system, full Prometheus/Grafana is overkill. The monitoring needs are:
1. **Data freshness:** Timestamp of latest event in graph vs current time. Alert if gap > 2 hours.
2. **Calibration drift:** ECE over rolling window. Alert if ECE > 0.15 (already implemented in `drift_detector.py`).
3. **Pipeline health:** Did today's pipeline complete? How long did each stage take?
4. **Model performance:** Brier score trend over resolved predictions.

**Implementation:** The existing `monitoring.py` (DataQualityMonitor) and `drift_detector.py` provide the foundation. For v2.0:
- Extend `DataQualityMonitor` to emit metrics to a JSON endpoint that the Streamlit dashboard can display
- Add a `/health` page in Streamlit showing pipeline status, last run time, and data freshness
- For alerting: a simple Python script that checks metrics and sends Discord/email notifications via a cron job

If the system outgrows this, Prometheus + Grafana can be layered on later. Do not over-engineer monitoring for a single-server demo.

**Sources:**
- [APScheduler Documentation](https://apscheduler.readthedocs.io/en/3.x/userguide.html)
- [ML Model Monitoring with Prometheus & Grafana (Markaicode)](https://markaicode.com/mlops-data-drift-detection-prometheus-grafana/)
- [ML Pipeline Orchestration (Domo)](https://www.domo.com/glossary/ml-pipeline-orchestration)
- [Evidently ML Monitoring Dashboard Tutorial](https://www.evidentlyai.com/blog/ml-model-monitoring-dashboard-tutorial)

---

## Feature Landscape

### Table Stakes (Users Expect These)

Features users assume exist. Missing these = product feels incomplete or amateurish for a public demo.

| Feature | Why Expected | Complexity | Dependencies | Notes |
|---------|--------------|------------|--------------|-------|
| **Live forecasts with probabilities** | Core product -- visitors come to see predictions | MEDIUM | Forecast engine (exists), daily automation (new) | Already have the engine; need scheduling + display |
| **Reasoning chains per forecast** | Core value prop -- explainability | LOW | Explainer (exists), output_formatter (exists) | Already built. Wire to dashboard. |
| **Historical accuracy display** | Credibility -- every serious platform shows track record | MEDIUM | prediction_store (exists), Brier scorer (exists) | Need calibration plot visualization + Brier trend display |
| **Methodology page** | Credibility -- visitors need to know what they're looking at | LOW | None (static content) | Describe GDELT + TKG + Gemini pipeline, cite algorithms |
| **Topic/region organization** | Navigability -- visitors need to find relevant forecasts | LOW | CAMEO categorization (exists) | Group by CAMEO super-category + region |
| **Daily automated forecasts** | Freshness -- stale predictions destroy credibility | MEDIUM | systemd timer, daily pipeline script (new) | Cannot be a manual process if public-facing |
| **Data freshness indicator** | Trust -- visitors need to know data is current | LOW | monitoring.py (exists) | Show "last updated: X minutes ago" prominently |
| **Rate-limited public access** | Security -- prevent abuse of Gemini API | MEDIUM | Streamlit session management (new) | Essential for interactive query feature |
| **Input validation/sanitization** | Security -- public-facing endpoint | LOW | None | Never pass raw user input to Gemini or graph queries |
| **Error-safe public display** | Security -- no leaked internals | LOW | None | Catch all exceptions, show generic error messages |

### Differentiators (Competitive Advantage)

Features that set this apart from Metaculus, ACLED CAST, and other forecast dashboards.

| Feature | Value Proposition | Complexity | Dependencies | Notes |
|---------|-------------------|------------|--------------|-------|
| **Interactive on-demand queries** | Users can ask their own geopolitical questions and get live AI forecasts with reasoning -- no other public platform does this with a hybrid TKG+LLM system | HIGH | Gemini client (exists), TKG predictor (exists), rate limiting (new) | This is the killer feature. Metaculus requires crowd forecasters; ACLED CAST has fixed questions. On-demand queries are genuinely novel. |
| **TKG + LLM hybrid reasoning display** | Show both the graph patterns and the LLM reasoning side-by-side, so visitors can see how structured data and language models disagree/agree | MEDIUM | ensemble_predictor (exists), explainer (exists) | No competitor shows the dual-model decomposition transparently |
| **Per-category dynamic calibration** | Self-improving system that gets better over time -- publishable and technically impressive | HIGH | prediction_store (exists), isotonic_calibrator (exists), feedback loop (new) | Differentiates from static models; demonstrates ML sophistication |
| **Knowledge graph visualization** | Interactive graph showing entity relationships and event flows -- visually impressive and educational | MEDIUM | knowledge_graph (exists), Streamlit graph component (new) | Could use pyvis or streamlit-agraph for interactive network display |
| **Micro-batch freshness (15-min GDELT)** | Graph updated every 15 minutes vs competitors' daily/weekly -- faster response to breaking events | MEDIUM | GDELT client (exists), micro-batch pipeline (new) | ACLED CAST updates weekly. GDELT's 15-min feed is a genuine advantage. |
| **Calibration plot transparency** | Show calibration curves honestly, including where the system is poorly calibrated -- radical transparency builds trust | LOW | calibration_metrics (exists), Brier scorer (exists) | Metaculus does this well. Match their standard. |

### Anti-Features (Deliberately NOT Building)

| Anti-Feature | Why Tempting | Why Problematic | Alternative |
|--------------|-------------|-----------------|-------------|
| **Real-time prediction updates** | GDELT updates every 15 min, so why not predict every 15 min? | Gemini API cost explodes at 96 calls/day per question. TKG inference is cheap but meaningless without updated embeddings (weekly retrain). Most geopolitical forecasts don't change on 15-min timescales. Creates false impression of precision. | Micro-batch ingest (keep graph fresh) + daily predict (cost-effective). Display "data updated X min ago, forecast updated Y hours ago" separately. |
| **User accounts and saved forecasts** | "Engagement" -- let users track their interests | Transforms the system from a demo into a SaaS product. Requires auth, database schema changes, GDPR considerations, password reset flows, etc. Enormous scope creep for negligible demo value. | Public dashboard with no auth required. Rate limit by IP + session, not by account. |
| **Prediction market / crowd forecasting** | IARPA showed hybrid human-AI beats AI alone by 10% | Fundamentally different product category. Requires incentive structures, forecaster recruiting, question design workflow, leaderboard systems, moderation. Months of work for a different product. | Pure AI system. Reference crowd forecasting research in methodology page. Consider v3.0+ if the demo gains traction. |
| **Multi-language support** | GDELT covers 65+ languages | GDELT's non-English coverage uses machine translation with variable quality. Entity resolution across languages introduces ambiguity. English sources cover the major geopolitical events this system targets. | English only. Acknowledge limitation in methodology page. |
| **Financial market integration** | "Geopolitical risk affects markets" -- sounds impressive | Completely different data pipeline (market feeds, SEC filings, supply chain graphs). Domain-specific causal models. Dilutes the core geopolitical forecasting focus. | Out of scope. The system predicts events, not market moves. |
| **Complex interactive map** | Maps look impressive on dashboards | Geopolitical events are not well-represented by pin-on-map visualizations. Many events (diplomatic statements, policy changes) don't have meaningful geographic coordinates. ACLED CAST uses maps because they forecast localized conflict counts -- a different problem. For this system, a map would be decorative, not functional. | Use region/country tags for organization. If a map is added, make it a lightweight filter, not the primary interface. |
| **HisMatch TKG algorithm** | Best MRR (46.42) | Preprocessing generates per-entity historical structure dictionaries -- O(entities x timestamps) memory. On large GDELT graphs, this may exceed RTX 3060 12GB. The dual-encoder architecture is architecturally different from RE-GCN, making the JAX port significantly harder than TiRGN. The +2.38 MRR gain over TiRGN does not justify the 2-3x implementation complexity. | TiRGN: +2.04 MRR over RE-GCN with tractable porting effort. Revisit HisMatch for v3.0 if hardware improves. |
| **Full Prometheus/Grafana monitoring stack** | "Production monitoring" | Overkill for single-server deployment. Prometheus scraping + Grafana dashboards add operational overhead (database, config, maintenance) that exceeds the monitoring value for one server. | JSON metrics + Streamlit health page + cron-based alerting scripts. Layer Prometheus later if justified. |

---

## Feature Dependencies

```
[Daily Automation Pipeline]
    |--requires--> [Micro-Batch GDELT Ingest] (graph must be fresh for daily predictions)
    |--requires--> [Forecast Engine] (exists)
    |--enables---> [Historical Accuracy Display] (predictions accumulate over time)
    |--enables---> [Dynamic Calibration] (outcome data accumulates)

[Streamlit Dashboard]
    |--requires--> [Daily Automation Pipeline] (needs predictions to display)
    |--requires--> [Rate Limiting] (public access protection)
    |--requires--> [Input Validation] (public access security)
    |--enhances--> [Interactive Queries] (frontend for on-demand forecasts)

[Dynamic Per-Category Calibration]
    |--requires--> [Prediction Store] (exists -- has category column)
    |--requires--> [Ground Truth Resolution] (new -- compares predictions to GDELT outcomes)
    |--requires--> [Sufficient Resolved Predictions] (~50 per super-category, ~2-3 months of operation)
    |--enhances--> [Ensemble Predictor] (replaces fixed alpha with learned weights)

[TKG Algorithm Replacement (TiRGN)]
    |--requires--> [TiRGN JAX/jraph Implementation] (new model module)
    |--requires--> [History Preprocessing] (global repetition vocabulary)
    |--modifies--> [TKGPredictor] (swap regcn_wrapper for tirgn_wrapper)
    |--modifies--> [Training Pipeline] (new training script)
    |--independent of--> [Dashboard] (can be done before or after)

[Micro-Batch GDELT Ingest]
    |--requires--> [GDELT Client] (exists -- needs 15-min polling mode)
    |--requires--> [Deduplication] (exists -- needs cross-batch dedup via GlobalEventID)
    |--requires--> [Graph Builder] (exists -- needs incremental append mode)
    |--enhances--> [Daily Automation] (fresher graph = better forecasts)

[Interactive Queries]
    |--requires--> [Streamlit Dashboard] (frontend)
    |--requires--> [Rate Limiting] (cost protection)
    |--requires--> [Gemini Client] (exists)
    |--requires--> [TKG Predictor] (exists)
    |--conflicts with--> [Budget Control] (each query costs Gemini API credits)
```

### Dependency Notes

- **Daily Automation before Dashboard:** The dashboard needs predictions to display. Without automation, the demo shows stale or no data.
- **Micro-Batch before Daily Automation:** Technically the daily pipeline can work with daily-batch ingest, but micro-batch makes forecasts meaningfully fresher. Implement micro-batch first so the daily pipeline benefits from it immediately.
- **Dynamic Calibration is time-delayed:** Even after implementation, per-category weights are useless until ~50 predictions per super-category resolve (~2-3 months). Build the infrastructure in v2.0, but expect the system to run on global fallback weights initially.
- **TKG Replacement is independent:** The TiRGN port can happen before or after the dashboard work. It does not block or depend on any other v2.0 feature. Recommend doing it early so the improved model is available for all subsequent forecasts.

---

## v2.0 Feature Prioritization

### Launch With (v2.0 Core)

- [x] **Daily automation pipeline** -- without this, the demo shows stale data. P1.
- [x] **Streamlit dashboard with live forecasts** -- the entire point of v2.0. P1.
- [x] **Historical accuracy display** -- credibility is non-negotiable. P1.
- [x] **Methodology page** -- visitors need to know what they're looking at. P1.
- [x] **Rate limiting + input sanitization** -- mandatory for public access. P1.
- [x] **Micro-batch GDELT ingest** -- 15-min freshness is a real differentiator. P1.
- [x] **TiRGN algorithm replacement** -- meaningful accuracy gain, do it early. P1.

### Add After Core Stabilizes (v2.0 Enhancement)

- [ ] **Interactive on-demand queries** -- the killer feature, but requires the core dashboard to work first. Add when the automated forecast display is solid. P2.
- [ ] **Knowledge graph visualization** -- visually impressive but not essential for launch. Add when core is stable. P2.
- [ ] **Dynamic per-category calibration infrastructure** -- build the feedback loop, but it won't produce useful results for 2-3 months. P2 for infrastructure, P3 for per-category specialization.
- [ ] **Dual-model reasoning display** -- show LLM vs TKG side-by-side. Enhances explainability but not blocking. P2.

### Future Consideration (v2.x / v3.0)

- [ ] **HisMatch TKG upgrade** -- revisit when hardware permits or when TiRGN accuracy ceiling is hit. P3.
- [ ] **Prometheus/Grafana monitoring** -- if the system grows beyond single-server. P3.
- [ ] **Multi-source data (ACLED, ICEWS)** -- after micro-batch architecture is proven. P3.
- [ ] **Human forecaster integration** -- only if the project pivots to a prediction platform. P3.

---

## Feature Prioritization Matrix

| Feature | User Value | Implementation Cost | Priority | Phase Suggestion |
|---------|------------|---------------------|----------|------------------|
| Daily automation pipeline | HIGH | MEDIUM | P1 | Phase 1 |
| Streamlit dashboard (core) | HIGH | MEDIUM | P1 | Phase 2 |
| Micro-batch GDELT ingest | HIGH | MEDIUM | P1 | Phase 1 |
| TiRGN algorithm replacement | HIGH | HIGH | P1 | Phase 1 (parallel) |
| Historical accuracy display | HIGH | LOW | P1 | Phase 2 |
| Methodology page | MEDIUM | LOW | P1 | Phase 2 |
| Rate limiting + security | HIGH | LOW | P1 | Phase 2 |
| Interactive on-demand queries | HIGH | HIGH | P2 | Phase 3 |
| Knowledge graph visualization | MEDIUM | MEDIUM | P2 | Phase 3 |
| Dynamic calibration (infra) | MEDIUM | HIGH | P2 | Phase 2-3 |
| Dual-model reasoning display | MEDIUM | LOW | P2 | Phase 3 |
| System health monitoring page | MEDIUM | LOW | P2 | Phase 2 |

---

## Competitor Feature Analysis

| Feature | Metaculus | ACLED CAST | GJ Open | Our Approach |
|---------|----------|------------|---------|--------------|
| Live forecasts | Crowd-aggregated | Model-generated (6-month) | Crowd-aggregated | AI-generated (daily) |
| Reasoning display | Community discussion | Driving factors breakdown | Comment threads | LLM + TKG reasoning chains |
| Calibration | User calibration plots | Not public | Tournament scores | System-wide calibration plots |
| Interactive queries | No (fixed questions) | No (fixed regions) | No (fixed questions) | **Yes -- ask any geopolitical question** |
| Data freshness | Real-time crowd updates | Weekly | Real-time crowd updates | 15-minute GDELT + daily predict |
| Model transparency | Aggregation algorithm disclosed | Methodology page | Aggregation method disclosed | Full model architecture + code |
| Free access | Yes | Registration required | Yes | Yes (rate-limited) |

---

## Sources

### TKG Algorithms
- [TRCL Paper -- PeerJ cs-2595](https://peerj.com/articles/cs-2595/)
- [Negative-Aware Diffusion -- arxiv 2602.08815](https://arxiv.org/html/2602.08815)
- [HisMatch -- EMNLP 2022](https://aclanthology.org/2022.findings-emnlp.542.pdf)
- [HisMatch GitHub](https://github.com/Lee-zix/HiSMatch)
- [TiRGN -- IJCAI 2022](https://www.ijcai.org/proceedings/2022/299)
- [TiRGN GitHub](https://github.com/Liyyy2122/TiRGN)
- [CyGNet GitHub](https://github.com/CunchaoZ/CyGNet)
- [xERTE -- ICLR 2021](https://arxiv.org/abs/2012.15537)
- [ACDm -- Expert Systems with Applications 2025](https://www.sciencedirect.com/science/article/abs/pii/S0957417425040205)

### Dashboard & Competitor Analysis
- [Metaculus FAQ](https://www.metaculus.com/faq/)
- [Metaculus Design Language](https://metaculus.medium.com/a-new-design-language-for-metaculus-c47c9133fca4)
- [Metaculus Scoring System](https://forum.effectivealtruism.org/posts/FodvZaiKftDCHPTub/metaculus-introduces-new-forecast-scores-new-leaderboard-and)
- [ACLED CAST Platform](https://acleddata.com/platform/cast-conflict-alert-system)
- [ACLED CAST Methodology](https://acleddata.com/methodology/cast-methodology)
- [New Lines Forecast Monitor](https://newlinesinstitute.org/forecast-monitor/)
- [Good Judgment Open](https://www.gjopen.com/)

### GDELT Micro-Batch
- [GDELT Data Access](https://www.gdeltproject.org/data.html)
- [GDELT 2.0 Announcement](https://blog.gdeltproject.org/gdelt-2-0-our-global-world-in-realtime/)

### Calibration & Ensemble
- [BODE Dynamic Ensemble -- Information Sciences](https://www.sciencedirect.com/science/article/abs/pii/S0020025522000135)
- [Calibrating Ensembles with Sparse Data](https://www.sciencedirect.com/science/article/abs/pii/S0169207014001010)
- [Calibration and Sharpness -- Gneiting et al.](https://sites.stat.washington.edu/raftery/Research/PDF/Gneiting2007jrssb.pdf)

### Pipeline & Monitoring
- [APScheduler Documentation](https://apscheduler.readthedocs.io/en/3.x/userguide.html)
- [Evidently ML Monitoring](https://www.evidentlyai.com/blog/ml-model-monitoring-dashboard-tutorial)
- [Event Deduplication Patterns](https://risingwave.com/blog/effective-deduplication-of-events-in-batch-and-stream-processing/)
- [Streamlit Production Deployment](https://markaicode.com/deploy-streamlit-apps-dashboard-guide/)

---
*Feature research for: v2.0 Operationalization & Forecast Quality*
*Researched: 2026-02-14*
