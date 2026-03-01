# Phase 13: Calibration, Monitoring & Hardening - Context

**Gathered:** 2026-03-02
**Status:** Ready for planning

<domain>
## Phase Boundary

System self-improves its ensemble weights from accumulated prediction-outcome data, health is continuously observable via alerts and dashboard, operational failures are detected and handled, and the system runs unattended for 7 days without degradation. Polymarket comparison validates calibration against prediction market benchmarks.

</domain>

<decisions>
## Implementation Decisions

### Alerting strategy
- Email via SMTP for all alerts (threshold breaches + daily digest)
- GDELT feed staleness threshold: 1 hour with no new data triggers alert
- Consecutive failure escalation: 1st failure logged as WARNING, 2nd consecutive failure sends email alert
- Daily digest email includes: predictions made, Brier score, Gemini budget remaining, feed health, Polymarket comparison summary
- Event-driven alerts fire immediately on threshold breach; daily digest sent regardless of alert status

### Polymarket comparison
- Matching: keyword + LLM hybrid — filter by geopolitical category/country first, then LLM ranks candidates
- Match confidence threshold: 0.6+ (medium) — accepted automatically, below 0.6 discarded
- Fetch cadence: hourly Polymarket price snapshots
- Display: API endpoint (`GET /api/v1/calibration/polymarket`) + frontend table in CalibrationPanel + inclusion in daily digest email
- Accuracy metric: Brier score comparison (same metric used internally)
- Resolution: when a matched Polymarket contract resolves, score both Geopol and Polymarket, then archive to resolved history table
- Retention: indefinite — all resolved comparisons kept for long-term calibration analysis
- Sparse data UX: show whatever active matches exist + "seeking more matches" indicator when fewer than 5 active overlaps
- Data stored in PostgreSQL (polymarket_comparisons table with active + resolved partitioning)

### Calibration recomputation
- Cadence: weekly recomputation of per-CAMEO alpha weights
- Approval: auto-apply with guardrails — new weights applied automatically unless any weight deviates >20% relative from current value; flagged weights trigger email alert and are held pending manual review
- Minimum samples: 10 resolved outcomes per CAMEO category before computing category-specific weights; fewer falls back to super-category (4 groups: Verbal Coop, Material Coop, Verbal Conflict, Material Conflict) or global
- Cold-start: literature-derived priors seeded from published TKG vs. LLM comparison studies (not naive alpha=0.6)
- Weight history: full version history — every weekly recomputation stored with timestamp in calibration_weight_history table; enables rollback to any past weight set
- Drift detection: independent 30-day rolling Brier score computed daily (not tied to weekly calibration cycle); alert fires when rolling Brier is 10% worse than all-time baseline
- Optimization: L-BFGS-B minimizing Brier score per category (from requirements)

### Unattended operation
- Target: 7 days hands-off operation without human intervention
- Self-healing scope: transient failures only — network timeouts, Gemini rate limits, brief Redis/PostgreSQL hiccups retry and recover; persistent failures (>5 retries) alert and stop
- Gemini budget exhaustion: queue pending forecast questions, process them when budget resets next day
- Process supervision: systemd services with Restart=on-failure for all daemons (ingest, forecast pipeline, RSS poller)
- Log retention: 30 days with daily rotation (RotatingFileHandler or equivalent)
- Disk monitoring: alert at 80% disk usage, emergency auto-cleanup at 90% (purge old GDELT data beyond retention window + oldest log files)
- Health endpoint: full process health — uptime, restart count, last restart reason, memory usage per daemon; SystemHealthPanel renders this
- PostgreSQL failure mode: graceful degradation — serve stale cached forecasts from Redis/memory, queue writes for replay when DB recovers, alert immediately

### Claude's Discretion
- SMTP configuration details (library choice, connection pooling, retry on send failure)
- Specific systemd unit file structure and restart delay intervals
- Polymarket API client implementation details
- L-BFGS-B optimization hyperparameters (bounds, max iterations, convergence criteria)
- Literature source selection for cold-start priors
- Emergency cleanup ordering (which old data gets purged first)
- Rolling Brier score computation window tuning
- Write queue implementation for PostgreSQL recovery replay

</decisions>

<specifics>
## Specific Ideas

- Daily digest email is a key operational artifact — should feel like a morning briefing (predictions made, accuracy trending, budget status, market comparison)
- Polymarket comparison serves dual purpose: calibration validation AND credibility signal for external visitors seeing the dashboard
- Weight version history enables "time travel" analysis of how calibration evolved — could be valuable for the research paper
- The 20% guardrail on auto-apply prevents runaway calibration from corrupted outcome data

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 13-calibration-monitoring-hardening*
*Context gathered: 2026-03-02*
