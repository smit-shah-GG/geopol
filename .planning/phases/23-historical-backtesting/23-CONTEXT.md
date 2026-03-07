# Phase 23: Historical Backtesting - Context

**Gathered:** 2026-03-07
**Status:** Ready for planning

<domain>
## Phase Boundary

Walk-forward evaluation harness for historical prediction accuracy, model comparison (TiRGN vs RE-GCN), calibration audit with reliability diagrams, and look-ahead bias prevention. Results persist in PostgreSQL, surface in an admin BacktestingPanel, and export as CSV/JSON with methodology documentation. Internal reporting system accessible from the admin dashboard.

**Critical framing:** This is not just internal diagnostics. Backtesting output is the **verifiable track record** presented to institutional investors and fund managers. Every design decision optimizes for: (a) maximum evaluation coverage, (b) quant-grade rigor, and (c) presentable, exportable results that survive scrutiny from fund due diligence teams.

</domain>

<decisions>
## Implementation Decisions

### Evaluation Window Design

- **Window size: 14 days** (train on 14d, predict next 14d)
  - *Rationale:* The system has been generating predictions since late January (~5-6 weeks of data). 7-day windows would produce only 3-5 predictions per window — too noisy for meaningful Brier scores. 14 days balances resolution (enough windows to show a trend) with statistical significance (enough predictions per window to produce trustworthy metrics). As prediction history grows, 14d windows become increasingly valuable.

- **Sliding windows with 50% overlap** (7-day slide step on 14-day window)
  - *Rationale:* Non-overlapping tiles would produce only 2-3 data points across the current 5-6 week history — too few to plot a meaningful curve. Overlapping windows double the number of evaluation points, producing smoother Brier score and calibration curves. The trade-off (predictions scored in multiple windows) is acceptable because the purpose is trend visualization, not independent statistical tests. Funds want to see trajectory, not point estimates.

- **Minimum 3 predictions per window** for inclusion in evaluation curve
  - *Rationale:* The system is young. A higher threshold (5+) would create gaps in the curve during early history. 3 is the absolute minimum for a non-degenerate Brier score. Noisier windows are preferable to missing windows — investors want to see the full timeline of system operation, warts and all. Transparency builds trust more than cherry-picked intervals.

- **Named checkpoint comparison** — admin selects specific model checkpoint files
  - *Rationale:* "Latest only" comparison answers one question ("which model is better now?") but named checkpoints answer the harder question: "how has model accuracy evolved as training data grew?" This is a powerful narrative for investors — it shows the system improves with more data, which is the fundamental value proposition of a learning system. Requires checkpoint files to be discoverable (scan data directory for `.pt`/`.pkl` files with timestamps).

- **All predictions backtested** regardless of provenance (organic + polymarket_driven)
  - *Rationale:* Maximizes sample size — every resolved prediction is precious data at this stage. Polymarket-driven predictions have the bonus of built-in market benchmarks, enabling the head-to-head comparison that is the single most compelling investor metric. No reason to exclude any data from the evaluation surface.

### Run Lifecycle

- **On-demand triggering only** (admin button, no scheduled runs)
  - *Rationale:* Backtesting is investigative, not operational. A weekly scheduled backtest adds scheduler complexity, burns Gemini budget unpredictably, and produces reports nobody checks at 3am. The operator decides when an evaluation is meaningful — typically before investor meetings, after model updates, or after calibration changes. Scheduling can always be added later if the use case emerges.

- **Live re-prediction** (actual EnsemblePredictor runs, not historical replay)
  - *Rationale:* This is the bold but correct choice. Replay-only backtesting can only evaluate "how good were the predictions we already made?" — it cannot answer "what would RE-GCN have predicted here?" or "how does the new calibration affect old questions?" Live re-prediction enables true counterfactual model comparison, which is the core purpose of backtesting. The Gemini budget cost is justified because the output is investor-grade evidence, not a developer convenience tool.

- **Cancellable background job** via ProcessPoolExecutor
  - *Rationale:* A full backtest across 5-6 weeks with live re-prediction could take 30-60+ minutes and burn significant Gemini budget. Without cancellation, an operator who spots a configuration error must either wait for completion or kill the server. Cancellation with partial result persistence means: (a) no wasted compute — completed windows are saved, (b) the operator retains control over budget spend, (c) partial results are still useful for trend analysis. Fits existing heavy job pattern (asyncio.Lock, ProcessPoolExecutor) from Phase 20.

- **Append-only run history** (keep all runs)
  - *Rationale:* Runs are tiny data (a few hundred rows of metrics per run). There is zero storage pressure to prune. Keeping all runs enables: (a) comparing accuracy before/after model updates, (b) demonstrating improvement trajectory to investors over months, (c) reproducing any historical evaluation. Deleting runs would destroy the audit trail that institutional buyers demand.

- **Named runs** with operator-provided labels + descriptions
  - *Rationale:* "Run 2026-03-07T14:22:33Z" is meaningless in a pitch meeting. "Q1 2026 Full Evaluation" or "Post-TiRGN-Retrain Comparison" is immediately referenceable. Named runs become the vocabulary for discussing system performance with non-technical stakeholders. Low implementation cost (one text field), high communication value.

### Admin Panel Display

- **Run list with drill-down** as primary view
  - *Rationale:* The panel serves two workflows: (1) "show me all evaluations" (run list) and (2) "deep-dive into this specific evaluation" (drill-down). A latest-run dashboard optimizes for (2) but makes (1) harder. A side-by-side comparison optimizes for a specific task but is the wrong default. Run list → drill-down is the natural information architecture and matches the existing admin panel patterns (e.g., SourceManager shows feed list → per-feed detail).

- **Summary cards with progressive disclosure** inside drill-down
  - *Rationale:* 4 headline stat cards (Brier score, calibration deviation, hit rate, vs-Polymarket record) give instant read on run quality. Clicking any card expands the full chart/table below. This matches the proven pattern from forecast cards (click-expand inline → full analysis) and optimizes for the screenshot use case: a single screenshot of the 4 summary cards tells the story; drill-down provides the evidence. Fund quants can go deep; executives get the headline.

- **d3 (SVG) for all charts** (Brier curves, reliability diagrams, comparison charts)
  - *Rationale:* d3-hierarchy is already in the project (ScenarioExplorer). SVG gives pixel-perfect control over chart rendering — critical for investor-facing visuals where a misaligned axis label or ugly default color scheme destroys credibility. d3 also produces resolution-independent output (SVG scales cleanly for screenshots at any DPI). Chart.js would be faster to implement but the existing d3 dependency and the quality bar for this phase make SVG the right choice.

- **Four metric sections in drill-down:**
  1. **Brier score curves** — cumulative and rolling, per-window, with confidence bands
  2. **Calibration reliability diagram** — predicted probability vs actual frequency buckets
  3. **Hit rate / accuracy** — binary outcome win/loss with category breakdown
  4. **Geopol vs Polymarket** — head-to-head Brier comparison on co-evaluated questions

- **CSV/JSON export** with per-run and multi-run modes
  - *Rationale:* Per-run export covers the basic use case (download one evaluation). Multi-run export enables quant analysis: import into Python/R/Excel and overlay accuracy curves from different model versions or time periods. The multi-run export merges selected runs into a single file with run labels as column identifiers. Low implementation complexity (JSON serialization of existing data structures), high credibility signal ("we give you the raw data, not just pretty charts").

- **Methodology section in exports**
  - *Rationale:* A CSV of Brier scores without methodology documentation is unverifiable. Fund due diligence teams need to know: window parameters, bias prevention measures, data sources used, model versions evaluated, what was included/excluded and why. Including methodology in the export means every exported file is self-documenting — it doesn't depend on a separate document or verbal explanation that may not accompany the data. This is the difference between "here are some numbers" and "here is a rigorous evaluation."

### Bias Prevention

- **Full temporal calibration weight snapshots** per evaluation window
  - *Rationale:* Per-CAMEO calibration weights are optimized on accumulated outcome data via L-BFGS-B. Using current weights to evaluate historical windows means the weights have seen the outcomes they're being tested against — this is textbook look-ahead bias. Snapshotting weights at each window boundary and restoring them during re-prediction eliminates this vector completely. The implementation cost is moderate (serialize calibration_weights table state per window start date) but the integrity cost of NOT doing it is unacceptable for investor-facing metrics. A fund quant who discovers un-snapshotted weights would dismiss the entire evaluation.

- **Separate temporal ChromaDB index per evaluation window** (ephemeral, rebuilt each run)
  - *Rationale:* The RAG pipeline retrieves articles from ChromaDB to ground LLM reasoning. If the index contains articles published after the prediction date, the LLM has access to information that wouldn't have existed at prediction time — another textbook look-ahead bias. Date-filtered queries (adding a date ceiling) would be cheaper but depend on reliable publish timestamps on every indexed article, which is not guaranteed (GDELT metadata quality varies). A separate temporal index built from articles with `published_date <= window_end` is the rigorous approach: it makes the bias physically impossible rather than relying on query-time filtering. Ephemeral (not cached) ensures no stale index data across runs — each run starts clean. The cost is ~6-8 index rebuilds per run (one per slide step), which is acceptable for an on-demand investigative tool.

- **No disk caching of temporal indexes**
  - *Rationale:* Caching indexes on disk introduces a stale cache risk: if articles are re-indexed, corrected, or the source list changes, cached temporal indexes become silently invalid. For an investor-grade evaluation tool, "silently invalid" is unacceptable. Rebuilding per run is slower but guarantees correctness. Since backtesting is on-demand (not scheduled), the rebuild cost is paid only when an operator explicitly requests an evaluation — this is fine.

### Claude's Discretion

- Exact d3 chart styling (colors, fonts, axis formatting) — should be clean and professional
- Progress reporting granularity during background execution (per-window vs per-prediction)
- Database schema details for backtest_runs and backtest_results tables
- Cancellation mechanism implementation (threading.Event vs shared state)
- ChromaDB temporal index build strategy (full rebuild vs incremental filtered copy)
- Methodology text template content and formatting

</decisions>

<specifics>
## Specific Ideas

- **Investor narrative:** The backtesting system exists to produce a defensible, exportable track record. Every metric, chart, and export should be designed as if a quant analyst at a hedge fund will scrutinize it. "Can they poke holes in this?" is the design filter.
- **Geopol vs Polymarket is the killer metric** — head-to-head Brier comparison against live prediction markets is the single most compelling data point for institutional buyers. This should be the most prominent metric in both the panel and exports.
- **Progressive disclosure pattern** is already proven in the codebase (forecast cards, country drill-down) and should be the default interaction model for the BacktestingPanel.
- **Named runs as vocabulary** — run names become the way the team references evaluations in investor conversations. "The Q1 evaluation showed..." is more powerful than "the March 7th run showed..."

</specifics>

<deferred>
## Deferred Ideas

- **Scheduled backtesting** — if demand emerges for automated periodic evaluations, add an APScheduler job in a future phase. Not justified now given on-demand triggering covers all current use cases.
- **PDF report generation** — formatted PDF with charts and methodology for email distribution to investors. Higher production value than CSV/JSON export but significant implementation cost (PDF rendering library, layout engine). Candidate for v3.1+ or v4.0.
- **Public-facing accuracy dashboard** — exposing backtesting results (or a curated subset) on the public `/dashboard` rather than admin-only. Would require editorial control over which runs/metrics are visible. Significant product decision — its own phase.
- **Per-category backtesting** — breaking down accuracy by CAMEO category, region, or event type. Partially covered by the calibration reliability diagram but a full category-level drill-down could be its own feature.

</deferred>

---

*Phase: 23-historical-backtesting*
*Context gathered: 2026-03-07*
