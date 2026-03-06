# Phase 22: Polymarket Hardening - Context

**Gathered:** 2026-03-06
**Status:** Ready for planning

<domain>
## Phase Boundary

Fix operational reliability of the Polymarket-driven forecasting pipeline (created_at overwrite bug, 429 cascade from shotgun forecasting), add rigorous cumulative accuracy metrics (Brier scores) comparing Geopol vs Polymarket on resolved questions, build an admin head-to-head accuracy panel, handle market resolution/voiding properly, and harden the poller against API failures with graceful degradation.

</domain>

<decisions>
## Implementation Decisions

### Question Selection & Pipeline Routing
- **Top 10 active set model**: forecast exactly the top 10 geo markets by volume (same set shown on the dashboard). New market enters top 10 -> auto-forecast. Market drops out of top 10 -> stop re-forecasting but keep existing prediction.
- Keep existing tag-based keyword filtering (GEO_INCLUDE/GEO_EXCLUDE) unchanged
- Keep single volume threshold (no tiered gating)
- Questions with no identifiable country (country_iso=NULL) are still forecasted -- global questions are valid
- Daily caps (3 new + 5 reforecast) stay as defaults but must be adjustable from admin ConfigEditor at runtime without restart

### Accuracy Metrics
- **Global cumulative Brier score** for both Geopol and Polymarket, updated after each resolution
- **Rolling 30-day window Brier score** to show trend over time
- No per-category breakdown
- **Win/loss determination**: per-question Brier score comparison -- (p - outcome)^2 for each side. Lower Brier = winner.
- **Score timing**: Geopol's last re-forecast probability before resolution is the scored value. No special cutoff snapshot.

### Resolution Handling
- Detect resolution via Polymarket Gamma API `resolved`/`resolutionSource` field -- authoritative source, not price convergence
- **Outcome mapping**: Yes = 1.0, No = 0.0 (direct binary)
- **Voided/cancelled markets**: mark comparison as `voided` status, keep in admin UI with a badge, exclude from all Brier calculations and win/loss record
- Voided data kept for informational purposes -- operator can see what was voided and why

### Admin Head-to-Head Panel
- **Table-first design**: sortable table of resolved comparisons with summary stats at top
- **Columns**: question title, Geopol probability, Polymarket price, outcome (Yes/No), Geopol Brier, Polymarket Brier, winner badge, category, country, resolution date, market volume, divergence at resolution
- **Resolved only** -- active comparisons stay in the public ComparisonPanel
- Summary stats row: total resolved, Geopol wins, Polymarket wins, draws, cumulative Brier (both), rolling 30d Brier (both)

### Claude's Discretion
- Degradation behavior during extended Polymarket outages (stale price thresholds, alerting, recovery)
- Exact backoff timing and circuit breaker tuning
- DB schema for polymarket_accuracy table
- Rolling window implementation details (materialized vs computed on-read)

</decisions>

<specifics>
## Specific Ideas

- The known bug is at `auto_forecaster.py:601` -- `existing.created_at = datetime.now(timezone.utc)` overwrites the original creation timestamp on re-forecast. Fix: add `reforecasted_at` column, preserve `created_at`.
- The 429 cascade happens because the auto-forecaster tries to force through every unmatched candidate, each burning 2-3 Gemini calls for extraction even before the prediction call. The top-10 model eliminates this entirely.
- Top 10 set should align with what `fetch_top_geopolitical(limit=10)` already returns for the dashboard -- single source of truth for which markets matter.

</specifics>

<deferred>
## Deferred Ideas

None -- discussion stayed within phase scope.

</deferred>

---

*Phase: 22-polymarket-hardening*
*Context gathered: 2026-03-06*
