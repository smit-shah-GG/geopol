# Feature Landscape: v3.0 Operational Command & Verification

**Domain:** Admin/ops layer for AI geopolitical forecasting engine
**Researched:** 2026-03-04
**Overall confidence:** MEDIUM-HIGH

Confidence breakdown:
- Admin dashboard patterns: HIGH (MLflow, Grafana, WM codebase examined directly)
- Feed/source management: HIGH (WM codebase extracted, OSINT platform patterns surveyed)
- Backtesting/calibration: MEDIUM (Metaculus/Polymarket public docs, Brier.fyi examined)
- Global risk seeding: MEDIUM (FSI methodology published, Maplecroft high-level only)
- Polymarket accuracy tracking: HIGH (Brier.fyi architecture documented, Polymarket accuracy page surveyed)

---

## 1. Admin Dashboard

### Table Stakes

| Feature | Why Expected | Complexity | Depends On | Notes |
|---------|--------------|------------|------------|-------|
| Daemon status overview | Operator needs at-a-glance system health | Low | Existing source health endpoint | Green/yellow/red per daemon (GDELT, RSS, ACLED, advisory, Polymarket) |
| Manual trigger buttons | "Run GDELT poll now" without SSH | Low | APScheduler integration | `pause_job`/`resume_job`/`trigger_job` exposed as API endpoints |
| Pause/resume individual daemons | Maintenance windows, debugging | Low | APScheduler | Toggle per-daemon, persist state across restarts |
| Ingest run history table | "What happened in the last 24h?" | Low | Existing `ingest_runs` table | Paginated, sortable by daemon_type/status/timestamp |
| System config display | Show current settings without `.env` access | Low | Settings model | Read-only display of non-secret config values |
| Error log viewer | Recent errors without log file access | Med | Structured logging | Last N errors per daemon, filterable. Ring buffer or DB-backed. |
| Daily forecast pipeline status | Did today's forecast cycle complete? | Low | Existing pipeline | Last run time, success/fail, questions processed count |

### Differentiators

| Feature | Value Proposition | Complexity | Depends On | Notes |
|---------|-------------------|------------|------------|-------|
| Manual reforecast trigger | Re-run specific question without waiting for schedule | Med | Existing forecast pipeline | Admin selects question ID, triggers fresh ensemble prediction |
| TKG training status + manual retrain | Visibility into model freshness + on-demand retrain | Med | `RetrainingScheduler` | Show last trained date, epoch count, MRR. Button to trigger retrain. |
| Config editing (non-secret) | Adjust poll intervals, cap limits without restart | Med | APScheduler + Settings | Hot-reload for safe params (intervals, caps). Restart-required for others. |
| Gemini budget tracker | Remaining daily API calls vs. cap | Low | Existing cap tracking | Show 3/3 new, 2/5 reforecast used today |
| Audit trail | Who triggered what, when | Med | New audit table | Log admin actions: manual triggers, config changes, pause/resume |

### Anti-Features

| Anti-Feature | Why Avoid | What to Do Instead |
|--------------|-----------|-------------------|
| Full CRUD for feed configuration in admin UI | Feed URLs change rarely; UI CRUD adds attack surface and complexity for marginal value. WM hardcodes feeds in `feeds.ts` for good reason. | Add/remove feeds via config file or DB migration. Admin UI shows health only. |
| Real-time log streaming (WebSocket tailing) | Enormous complexity for minimal operational value. Operators needing real-time logs should SSH. | Paginated recent-errors endpoint. Link to Grafana/journalctl for deep dive. |
| Multi-user RBAC | Single operator system. Auth gating at route level is sufficient. | Simple API-key or session-based admin auth. No roles. |
| Dashboard customization (drag-drop panels) | Engineering cost vs. value is terrible for a single-operator tool | Fixed layout, well-designed defaults |
| Notifications/alerts in admin UI | Email alerts already exist via AlertManager | Keep email alerts. Admin UI is pull-based inspection, not push notifications. |

### WM Patterns to Reuse

WM's `UnifiedSettings` (527 lines) demonstrates a proven three-tab modal architecture:
1. **General** -- toggle rows with label/description/switch pattern
2. **Sources** -- grid of toggleable items with region pills, search filter, select-all/none, counter
3. **Status** -- feed status rows (dot + name + item count + last update time)

Key UX patterns from WM worth adopting:
- **Status dot coloring**: `ok` (green) / `warning` (yellow) / `error` (red) / `disabled` (gray) -- already used in geopol's source health endpoint
- **`formatTime` relative timestamps**: "just now", "5m ago", "2:30 PM" -- compact, scannable
- **Region pills for filtering**: Horizontal pill bar to filter by category (wire/mainstream/defense/thinktank etc.)
- **Source counter**: "42/56 sources enabled" footer pattern

WM's `ServiceStatusPanel` adds:
- Category filter buttons (cloud/dev/comm/ai/saas)
- Summary row with counts per status level (N operational, N degraded, N outage)
- Desktop readiness checklist pattern (acceptance checks with pass/fail)

WM's `RuntimeConfigPanel` demonstrates:
- Feature toggle with secret management (staged/validated/committed lifecycle)
- Per-feature status pills: "Ready" / "Staged" / "Needs Keys"
- Inline validation with error hints

**Recommendation for geopol admin**: Use WM's source toggle grid for feed health display. Use the status dot + relative time pattern universally. Skip the full RuntimeConfigPanel complexity -- geopol has no per-user secrets to manage.

---

## 2. Feed/Source Management

### Table Stakes

| Feature | Why Expected | Complexity | Depends On | Notes |
|---------|--------------|------------|------------|-------|
| Source health dashboard | Per-source last-poll time, success rate, item count | Low | Existing `ingest_runs` + source health endpoint | Already partially built. Needs UI surface in admin. |
| Per-source staleness detection | Flag sources that haven't returned data in >N hours | Low | Existing `FeedMonitor` pattern | Extend beyond GDELT to all daemon types |
| Feed category display | Group by wire/mainstream/thinktank/government/intl_org | Low | Existing `FeedCategory` enum in `feed_config.py` | WM's region-pill pattern works here |
| Tier display | Show which feeds are Tier 1 (15min) vs Tier 2 (60min) | Low | Existing `FeedTier` enum | Badge or icon per feed |
| Source error details | Last error message per source | Low | `IngestRun.status` field | Expandable row showing error text |

### Differentiators

| Feature | Value Proposition | Complexity | Depends On | Notes |
|---------|-------------------|------------|------------|-------|
| Add/remove RSS feeds via admin | Expand source coverage without code deploy | Med | DB-backed feed config (currently hardcoded) | Migrate `feed_config.py` constants to DB table. Keep hardcoded as defaults/seed. |
| Feed categorization controls | Assign tier + category when adding new feed | Low | DB-backed feed config | Dropdown for tier (1/2), dropdown for category |
| Per-source article count trend | Sparkline showing articles/day over last 7 days | Med | Time-series query on events table | Detect source death (output drops to 0) |
| ICEWS/UCDP source health | Monitor new data sources alongside RSS/GDELT | Low | New poller modules | Same health pattern, new daemon_type values |
| Feed test/preview | "Fetch this URL now and show what we'd get" | Med | RSS parsing logic | Validates URL works before adding to rotation |

### Anti-Features

| Anti-Feature | Why Avoid | What to Do Instead |
|--------------|-----------|-------------------|
| Feed discovery/recommendation engine | Massive scope creep. Feed curation is a human editorial task. | Manual add with URL + category + tier. |
| Content preview in admin | Showing article text in admin UI is irrelevant to operational concerns | Show metadata only: title, URL, timestamp, extraction success |
| Feed deduplication management | Cross-source dedup is an ingestion pipeline concern, not an admin UI concern | Handle in `ArticleIndexer` with URL-hash dedup (already exists) |
| Polling frequency per-feed customization | Tier system (15min/60min) is sufficient granularity. Per-feed intervals create operational complexity. | Two tiers. Reassign tier if needed. |

### Existing Geopol Assets

The `feed_config.py` already defines the correct data model:
```python
@dataclass(frozen=True, slots=True)
class FeedSource:
    name: str
    url: str
    tier: FeedTier      # TIER_1 (15min) or TIER_2 (60min)
    category: FeedCategory  # wire, mainstream, government, etc.
    lang: str = "en"
```

The `sources.py` API endpoint already auto-discovers daemon types from `ingest_runs`. The admin UI is primarily a rendering concern, not a new backend capability.

---

## 3. Forecast Backtesting

### Table Stakes

| Feature | Why Expected | Complexity | Depends On | Notes |
|---------|--------------|------------|------------|-------|
| Walk-forward evaluation | Standard ML backtesting: train on T, predict T+1, slide window | High | TKG training pipeline, historical data | Core methodology. Requires re-training at each window step. |
| Brier score over time | Cumulative accuracy curve showing improvement/degradation | Med | Resolved predictions in PostgreSQL | Already have `BrierScorer` in `evaluation/`. Need time-series aggregation. |
| Calibration curve (reliability diagram) | Predicted probability bins vs. actual outcome frequency | Med | Existing `CalibrationMetrics` (ECE, MCE, ACE) | Already computed. Need visualization + time windowing. |
| Model comparison (TiRGN vs RE-GCN) | Justify model choice with data | High | Walk-forward eval for both models | Run same eval harness with different TKG backends |
| Resolution tracking | Track which predictions resolved YES/NO/AMBIGUOUS | Low | PostgreSQL predictions table | Need resolution status field + Polymarket outcome cross-ref |

### Differentiators

| Feature | Value Proposition | Complexity | Depends On | Notes |
|---------|-------------------|------------|------------|-------|
| Per-CAMEO calibration audit | Show calibration quality per event type over time | Med | Existing per-CAMEO calibration | Identify which event types the system is well/poorly calibrated on |
| Ensemble component contribution | Which component (TKG, LLM, calibrator) contributed most to accuracy? | High | Decomposable ensemble architecture | Requires storing component-level predictions |
| Temporal accuracy analysis | Accuracy at 1-day, 7-day, 30-day horizons | Med | Prediction timestamp + resolution timestamp | Metaculus pattern: accuracy degrades with horizon length |
| Confidence interval analysis | Are 80% CI intervals actually containing 80% of outcomes? | Med | Continuous prediction support | Standard for probabilistic forecasting evaluation |

### Anti-Features

| Anti-Feature | Why Avoid | What to Do Instead |
|--------------|-----------|-------------------|
| Public-facing accuracy page | v3.0 backtesting is internal operational tooling, not a marketing page | Internal admin report. Public accuracy page is a v4+ consideration. |
| Real-time backtesting | Walk-forward eval is computationally expensive (TKG retraining per window) | Batch job triggered from admin, results stored and displayed |
| A/B testing framework | Overkill for single-operator system with one production model | Model comparison via walk-forward eval is sufficient |
| Automated model selection | "Pick the best model automatically" adds fragility | Human reviews backtest results, decides model swap manually |

### Metaculus/Good Judgment Patterns to Adopt

Metaculus track record pages show:
1. **Calibration curve**: X-axis = predicted probability bucket (0-10%, 10-20%, ..., 90-100%), Y-axis = actual resolution frequency. Perfect calibration = diagonal line.
2. **Brier score summary**: Single number with context ("0.12 -- better than 80% of forecasters")
3. **Question count by category**: How many resolved questions per domain
4. **Time horizon analysis**: Accuracy varies by prediction horizon (1 week vs 6 months)

Brier.fyi adds cross-platform comparison:
- Absolute scores (Brier, Log, Spherical)
- Relative scores vs. median peer performance
- Letter grades (A-F) for intuitive assessment
- Matched-question methodology for fair comparison

**Recommendation**: For geopol backtesting, implement:
1. Reliability diagram (calibration curve) -- the single most informative visualization
2. Cumulative Brier score over time -- shows system improvement/degradation trend
3. Model comparison table (TiRGN vs RE-GCN: MRR, Hits@K, Brier on resolved predictions)
4. Per-CAMEO heatmap -- which event types are well/poorly calibrated

---

## 4. Global Risk Seeding

### Table Stakes

| Feature | Why Expected | Complexity | Depends On | Notes |
|---------|--------------|------------|------------|-------|
| Baseline risk for all ~195 countries | Globe choropleth needs non-zero data for all countries | Med | GDELT event density + external indices | Current system only has risk for countries with active forecasts |
| Composite score methodology | Transparent, reproducible scoring | Med | Data source integration | Must combine multiple signals into single 0-100 score |
| Data source diversity | No single-source dependency for baseline | Med | Multiple ingest pipelines | At minimum: GDELT + one conflict dataset + one governance index |
| Automatic refresh | Baseline scores update without manual intervention | Low | Scheduled job | Daily/weekly update from underlying data sources |

### Differentiators

| Feature | Value Proposition | Complexity | Depends On | Notes |
|---------|-------------------|------------|------------|-------|
| Active forecast override | Countries with live predictions use prediction-derived risk instead of baseline | Low | Existing country risk SQL | Coalesce: active predictions > baseline seed |
| Risk trend arrows | Rising/falling/stable indicators per country | Low | Historical baseline scores | 7-day or 30-day delta comparison |
| Risk decomposition | Show which factors contribute to a country's score | Med | Component scores stored separately | "High conflict + moderate governance + low economic risk" |
| Data source attribution | "Score includes: GDELT (45%), ACLED (30%), FSI (25%)" | Low | Weight tracking | Transparency for operators |

### Anti-Features

| Anti-Feature | Why Avoid | What to Do Instead |
|--------------|-----------|-------------------|
| Custom index weighting UI | Single operator does not need a GUI slider for index weights | Hardcode sensible defaults. Adjust via config if needed. |
| Sub-national risk scores | Enormous data + complexity increase for marginal value at v3.0 | Country-level only. Sub-national is v5+ if ever. |
| Real-time risk updates | Baseline is inherently slow-moving (conflict, governance). Real-time is noise. | Daily batch update is sufficient. Active forecasts provide real-time signal. |
| Proprietary index integration (EIU, Maplecroft) | Expensive subscriptions, licensing complexity, API reliability | Use open data: GDELT (free), ACLED (free for research), UCDP (free), FSI (published annually) |

### FSI Methodology Reference (Fund for Peace)

The Fragile States Index provides the gold-standard methodology for composite country risk:

**12 indicators across 4 categories:**
- Cohesion: Security Apparatus, Factionalized Elites, Group Grievance
- Economic: Economic Decline/Poverty, Uneven Economic Development, Human Flight/Brain Drain
- Political: State Legitimacy, Public Services, Human Rights/Rule of Law
- Social: Demographic Pressures, Refugees/IDPs, External Intervention

**Scoring**: Each indicator 0-10, total 0-120. Uses triangulated data: ~50M articles/year content analysis + quantitative datasets (UN, WHO, World Bank, Freedom House, Transparency International) + qualitative expert review.

**Geopol adaptation**: We lack expert review and comprehensive quantitative datasets. Our composite should be:
1. **GDELT event density** -- conflict/cooperation event counts per country (available now)
2. **ACLED conflict events** -- armed conflict, protests, riots (Phase 23 source expansion)
3. **UCDP fatality data** -- battle-related deaths (Phase 23 source expansion)
4. **Travel advisories** -- already ingested via `advisory_poller.py`
5. **Static FSI baseline** -- published annually, can be imported as CSV

**Composite formula recommendation** (0-100 scale):
```
risk = w1 * gdelt_conflict_density    # 0.30 -- real-time signal
     + w2 * acled_event_severity      # 0.25 -- conflict-specific
     + w3 * advisory_level            # 0.20 -- government assessment
     + w4 * fsi_normalized            # 0.15 -- structural fragility
     + w5 * ucdp_fatality_rate        # 0.10 -- lethality signal
```

Active forecast override: if a country has >= 3 active predictions, use the existing CTE-based risk score instead.

---

## 5. Polymarket Accuracy Tracking

### Table Stakes

| Feature | Why Expected | Complexity | Depends On | Notes |
|---------|--------------|------------|------------|-------|
| Resolution tracking | Record YES/NO outcome for each matched Polymarket question | Low | `polymarket_snapshots` table | Need resolution status + timestamp |
| Brier score per prediction | Geopol probability vs. actual outcome | Low | Resolved predictions + outcomes | Already have `BrierScorer` |
| Head-to-head accuracy display | "Geopol predicted 0.72, Polymarket was 0.65, outcome was YES" | Low | Matched predictions | Simple comparison card/table |
| Cumulative Brier score curve | Geopol vs. Polymarket accuracy over time | Med | Time-series of resolved comparisons | Line chart, lower is better |

### Differentiators

| Feature | Value Proposition | Complexity | Depends On | Notes |
|---------|-------------------|------------|------------|-------|
| Accuracy by category | "Geopol beats Polymarket on geopolitical events, loses on crypto" | Med | Category tagging on predictions | Breakdown by CAMEO category or topic |
| Time-to-resolution analysis | Accuracy at different lead times (1 day, 1 week, 1 month before resolution) | Med | Snapshot history | Polymarket accuracy page uses this exact pattern |
| Calibration comparison | Side-by-side reliability diagrams (Geopol vs. Polymarket) | Med | Sufficient resolved predictions | Need ~50+ resolved predictions per bucket for statistical significance |
| Win/loss record summary | "Geopol: 47 wins, 23 losses, 8 ties vs. Polymarket" | Low | Resolved comparisons | Simple count. "Win" = lower Brier score on that prediction. |

### Anti-Features

| Anti-Feature | Why Avoid | What to Do Instead |
|--------------|-----------|-------------------|
| Real-time Polymarket price display | Geopol is not a trading platform. Showing live prices implies actionability. | Show comparison at time-of-prediction and at resolution only. |
| Multi-platform comparison (Kalshi, Manifold, etc.) | Scope creep. Polymarket is the primary comparison target. | Polymarket only for v3.0. Architecture should allow adding platforms later. |
| Public leaderboard | v3.0 is internal tooling. Public accuracy claims require statistical rigor. | Internal admin dashboard. Public Brier.fyi-style page is v4+. |
| Automated trading signals | Crossing the line from forecasting into financial advice | Never. Forecast system produces probabilities, not trade recommendations. |

### Brier.fyi Patterns Worth Adopting

Brier.fyi (launched 2025) demonstrates the reference architecture for prediction accuracy tracking:
- **Matched-question methodology**: Link equivalent questions across platforms using embeddings + heuristics
- **Three scoring rules**: Brier (primary), Log (penalizes extreme errors), Spherical
- **Relative scoring**: Compare vs. median performance, not just absolute
- **Letter grades**: A-F translation for non-technical consumers
- **Daily relative score**: Granular time-series comparison

**Recommendation for geopol**: Implement Brier score only (simpler, well-understood). Add Log score later if needed. Skip letter grades -- internal tooling, operators understand Brier directly.

### Polymarket Accuracy Page Patterns

Polymarket's `/accuracy` page shows:
- Brier score at multiple time horizons: 1 month, 1 week, 1 day, 12 hours, 4 hours before resolution
- Percentage of markets resolving correctly at each horizon
- Overall accuracy percentage (90-95% range)

**Key insight**: Polymarket's Brier score of 0.058 (12h ahead) sets a high bar. Geopol should NOT expect to beat this on liquid markets. The value proposition is on low-liquidity or geopolitical-specific questions where prediction markets have thin coverage.

---

## 6. Globe Layer Data Wiring

### Table Stakes

| Feature | Why Expected | Complexity | Depends On | Notes |
|---------|--------------|------------|------------|-------|
| Arcs layer data | Globe arc visualization needs actual conflict/cooperation flows between countries | Med | GDELT dyadic event data | Extract country-pair event flows from GDELT. Aggregate by cooperation/conflict. |
| Heatmap layer data | Density visualization needs event concentration data | Med | GDELT geocoded events | Lat/lon event density aggregation. Already have GDELT events in SQLite. |
| Scenarios layer data | Show active forecast scenarios on globe | Low | Existing predictions table | Geocode predictions to country centroids, display as markers/arcs |

### Differentiators

| Feature | Value Proposition | Complexity | Depends On | Notes |
|---------|-------------------|------------|------------|-------|
| Animated temporal arcs | Show event flow evolution over time | High | Time-series arc data | DeckGL `TripLayer` or animated `ArcLayer`. Visually compelling but complex. |
| Risk heatmap gradient | Continuous color gradient from risk scores, not just markers | Med | Global risk seeding scores | Feed seeded risk scores into heatmap intensity |

### Anti-Features

| Anti-Feature | Why Avoid | What to Do Instead |
|--------------|-----------|-------------------|
| Real-time event streaming on globe | WebSocket + high-frequency updates for a daily-refresh system is nonsensical | Refresh globe data on page load and on 60s poll interval |
| 3D terrain/elevation | DeckGL supports it but adds no analytical value for geopolitical data | Flat globe projection is correct |

---

## Feature Dependencies

```
Phase 19 (Admin Dashboard) -----> No dependencies, can start immediately
    |
    v
Phase 20 (Daemon Consolidation) -> Enables admin "pause/resume" buttons to work
    |
    v
Phase 21 (Polymarket Hardening) -> Enables accuracy tracking (needs reliable resolution data)
    |
    v
Phase 22 (Backtesting) ---------> Requires resolved predictions from Phase 21
    |                              Requires walk-forward TKG training infrastructure
    v
Phase 23 (Source Expansion) -----> ICEWS + UCDP pollers (new daemon types)
    |                              Adds data sources for Phase 24
    v
Phase 24 (Global Seeding) ------> Requires ACLED/UCDP/ICEWS data from Phase 23
    |                              Requires composite scoring formula
    v
Phase 25 (Globe Data Wiring) ---> Requires global seeding scores (Phase 24)
                                   Requires GDELT dyadic data extraction
```

**Critical path**: Phases 19-20 are independent and can be built first. Phase 21-22 form a dependency chain (need resolved predictions for backtesting). Phase 23-25 form another chain (need sources before seeding before globe).

---

## MVP Recommendation

For each feature domain, the minimum viable implementation:

### Admin Dashboard MVP
1. Status overview page (daemon health, last run times, error counts)
2. Manual trigger buttons (run ingest now, trigger forecast)
3. Pause/resume daemon controls
4. Ingest history table (last 50 runs)

Defer: Config editing, audit trail, TKG training controls

### Feed Management MVP
1. Source health grid in admin (reuse WM status-dot pattern)
2. Per-source staleness alerts
3. Category/tier display

Defer: Add/remove feeds via UI (config file is fine for now), sparkline trends

### Backtesting MVP
1. Reliability diagram (calibration curve) from resolved predictions
2. Cumulative Brier score chart
3. TiRGN vs RE-GCN comparison table (MRR, Hits@K, Brier)

Defer: Walk-forward eval (computationally expensive, batch job), per-CAMEO heatmap, ensemble decomposition

### Global Seeding MVP
1. GDELT event density per country as baseline risk
2. Travel advisory level overlay
3. Active forecast override logic

Defer: ACLED/UCDP/FSI integration (requires Phase 23 source expansion first)

### Polymarket MVP
1. Resolution tracking (record outcomes)
2. Per-prediction Brier score
3. Head-to-head comparison table
4. Simple win/loss count

Defer: Calibration comparison, time-horizon analysis, category breakdown

---

## Sources

### Direct Code Examination (HIGH confidence)
- `/home/kondraki/personal/worldmonitor/src/components/UnifiedSettings.ts` -- WM settings/source management UI
- `/home/kondraki/personal/worldmonitor/src/components/StatusPanel.ts` -- WM feed/API health tracking
- `/home/kondraki/personal/worldmonitor/src/components/ServiceStatusPanel.ts` -- WM service status with category filters
- `/home/kondraki/personal/worldmonitor/src/components/RuntimeConfigPanel.ts` -- WM feature toggle + secret management
- `/home/kondraki/personal/worldmonitor/src/config/feeds.ts` -- WM 4-tier source system, region mapping
- `/home/kondraki/personal/geopol/src/api/routes/v1/sources.py` -- Existing source health endpoint
- `/home/kondraki/personal/geopol/src/monitoring/feed_monitor.py` -- Existing staleness detection
- `/home/kondraki/personal/geopol/src/ingest/feed_config.py` -- Existing feed tier/category model
- `/home/kondraki/personal/geopol/src/evaluation/evaluator.py` -- Existing evaluation orchestrator

### Official Documentation / Published Methodology (MEDIUM-HIGH confidence)
- [Fragile States Index Methodology](https://fragilestatesindex.org/methodology/) -- FSI composite scoring, 12 indicators, triangulated data
- [MLflow Agent Dashboard](https://mlflow.org/blog/mlflow-agent-dashboard) -- MLflow 3.9+ admin dashboard patterns
- [Brier.fyi About](https://brier.fyi/about/) -- Cross-platform prediction accuracy tracking methodology
- [Polymarket Accuracy](https://polymarket.com/accuracy) -- Brier score benchmarks, time-horizon accuracy
- [Metaculus Track Record](https://www.metaculus.com/questions/track-record/) -- Calibration curves, forecaster performance

### Web Research (MEDIUM confidence)
- [FastAPI-Scheduler](https://github.com/amisadmin/fastapi-scheduler) -- APScheduler + FastAPI admin integration
- [Maplecroft Country Risk Data](https://www.maplecroft.com/data/country-risk-data/) -- Commercial country risk methodology
- [Fensory Polymarket Analysis](https://www.fensory.com/intelligence/predict/polymarket-accuracy-analysis-track-record-2026) -- Polymarket accuracy benchmarks
