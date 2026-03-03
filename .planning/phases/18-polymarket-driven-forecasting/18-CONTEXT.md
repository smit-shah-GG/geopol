# Phase 18: Polymarket-Driven Forecasting - Context

**Gathered:** 2026-03-04
**Status:** Ready for planning

<domain>
## Phase Boundary

Transform Polymarket from a passive comparison tool (Phase 13 matched against existing predictions) into an active forecast driver. When Polymarket has a geopolitical question, Geopol runs its forecasting pipeline to produce a competing prediction, tracks probability divergence over time via daily re-forecasting, and displays head-to-head comparisons on the dashboard. Polymarket-driven forecasts are first-class predictions — they appear in the main forecast list with a badge and inline comparison data.

**What Phase 13 already built:** PolymarketClient (Gamma API), PolymarketMatcher (keyword + LLM two-phase matching), PolymarketComparisonService (tracking, snapshots, resolution, Brier scores), DB tables (polymarket_comparisons, polymarket_snapshots), API routes (/calibration/polymarket, /calibration/polymarket/top), PolymarketPanel (Col 4 top-10 events table), background polling loop in app.py (3600s cycle).

**What Phase 18 adds:** Auto-forecast generation for unmatched Polymarket questions, periodic re-forecasting of active comparisons, badge system on forecast cards, inline comparison data in expanded cards, dedicated comparison panel in Col 2.

</domain>

<decisions>
## Implementation Decisions

### Auto-forecast trigger rules
- **Volume threshold filtering:** Only Polymarket questions above a trading volume threshold trigger pipeline runs. Rationale: the market itself signals which questions are significant — low-volume questions are noise or illiquid speculation. This focuses Geopol's API budget on questions the market considers worth pricing.
- **Hard cap of 5 new forecasts per day.** Prioritize by volume when more candidates exist. Rationale: Gemini API budget is finite and each forecast requires a full pipeline run (LLM scenario generation + TKG prediction + calibration). 5/day is enough to build a meaningful comparison corpus without runaway cost. The cap is configurable via Settings.
- **Daily re-forecasting of active comparisons.** Rationale: Polymarket prices shift continuously as news breaks; a one-shot Geopol probability would become stale and make divergence tracking meaningless. Daily re-runs capture how Geopol's model responds to new GDELT/ACLED data vs how the market responds to the same events. Re-forecasts count against the daily cap.

### Forecast identity and badge system
- **Two badge categories, one treatment:**
  - **"Polymarket-driven"** — new predictions auto-generated because Polymarket had a geopolitical question without a Geopol match. These are created by Phase 18's auto-forecast pipeline.
  - **"Polymarket-tracked"** — existing organic Geopol predictions that the Phase 13 matcher paired with a Polymarket market. These already exist in the predictions table; Phase 18 adds visual indication.
  - Rationale: both types have a market counterpart and produce comparison data. The badge signals "this prediction has a Polymarket price to compare against" regardless of which side initiated the pairing.
- **Badge appearance: icon only.** Small Polymarket icon on the forecast card collapsed state. No divergence number or market price on the badge itself. Rationale: the collapsed card is already dense (question text + probability bar + country code + age). Adding numbers to the badge would compete with the primary probability display. Details surface on expansion.
- **Main forecast list placement.** Polymarket-driven forecasts appear as regular cards in Col 2 Active Forecasts (with badge), not in a separate section. They are normal predictions — searchable, filterable, same lifecycle. Rationale: separating them would fragment the user's forecast view and create a false distinction. A forecast is a forecast; the badge indicates provenance, not priority.

### Expanded card inline comparison
- When a badged forecast card is expanded, the expanded view adds a comparison row showing: Polymarket current price, divergence from Geopol probability, and a mini sparkline of both probabilities over time. Rationale: the user should see the head-to-head data without leaving the card context. The sparkline uses existing snapshot data from polymarket_snapshots table. This is integrated into the existing two-column expanded card layout (Phase 15).

### Comparison panel (Col 2, below Active Forecasts)
- **Position:** Below the Active Forecasts list in Col 2, scrollable. Rationale: Active Forecasts remain the primary content; the comparison panel is supplementary analysis. Placing it below maintains the hierarchy — users who want Bloomberg-density can scroll to see head-to-head data.
- **Primary visual: dual probability bars.** Each comparison entry shows two horizontal bars side by side — Geopol probability vs Polymarket price — color-coded by divergence magnitude. Rationale: bars are compact and scannable at a glance. A sparkline-per-entry would consume too much vertical space in a scrollable list; the expanded card already provides the time-series view.
- **Resolved markets: inline with status badge.** Resolved entries stay in the same list with a "Resolved" badge + outcome indicator (correct/wrong for each side). No separate resolved section. Rationale: segregating resolved entries hides the track record from the default view. Inline display lets the user scan active and resolved together, building trust signal naturally.
- **No aggregate scorecard.** Rationale: too few resolved data points in the early period would make aggregate stats (win/loss, avg Brier) misleading or actively harmful to credibility. Each entry speaks for itself. Scorecard can be added in a future phase once the corpus is large enough for statistical significance.
- **5-minute auto-refresh.** Matches the existing PolymarketPanel in Col 4. Rationale: Polymarket prices update frequently, but Geopol probabilities only change daily. 5-minute refresh is the right cadence to pick up market price movement without excessive API calls.

### Question-to-pipeline mapping (extraction strategy)
- **Tiered extraction: heuristic first, LLM always for category.**
  - **Country:** Heuristic keyword matching from Polymarket event title and tags against existing country-name-to-ISO dict (already built in advisory_poller.py). Falls back to LLM extraction via Gemini if heuristic finds no country. Rationale: ~70-80% of geopolitical Polymarket questions mention a country explicitly ("Will Ukraine...", "Iran nuclear..."). Heuristic handles these for free. LLM fallback catches ambiguous cases ("Will NATO expand?", "OPEC production cuts?") where the LLM can infer the most relevant country or flag as multi-country.
  - **Horizon:** Computed directly from Polymarket event `endDate` field. No LLM needed. Rationale: the API provides structured market expiry dates — parsing them is trivial date arithmetic.
  - **CAMEO category:** Always extracted via LLM call, even when heuristic found the country. Rationale: per-CAMEO calibration weights (Phase 13 L-BFGS-B optimization) are a key accuracy lever. Defaulting to a generic category would bypass the dynamic calibration that differentiates Geopol's ensemble from a naive LLM forecast. The marginal cost of one Gemini parsing call per question is negligible against the full pipeline cost.
- **Standard scenario-tree output.** Polymarket-driven forecasts produce full Geopol output (scenarios, evidence, calibration), not a simplified binary probability. The top-level calibrated probability is what gets compared to market price. Rationale: the scenario tree is Geopol's core value proposition — explainability. Stripping it for Polymarket questions would produce a less useful forecast and lose the "why" behind the number.

### Claude's Discretion
- Volume threshold value (exact dollar amount — Claude determines based on typical Polymarket geopolitical question distribution)
- Sparkline rendering implementation (canvas, SVG, or CSS) and dimensions within expanded card
- Comparison panel empty state design
- Dual bar color scheme for divergence magnitude ranges
- How re-forecast results update existing prediction rows vs create new ones (append vs overwrite)
- Deduplication logic: ensuring the same Polymarket question doesn't trigger multiple pipeline runs across cycles
- Whether the daily cap is per-cycle or rolling 24-hour window

</decisions>

<specifics>
## Specific Ideas

- The two-badge distinction (driven vs tracked) emerged from recognizing that Phase 13's matcher already pairs organic predictions with markets — Phase 18 should surface those pairings visually, not just create new ones. The badge unifies both provenance types under one visual system.
- "Heuristic + LLM fallback" was chosen over "always LLM" because the country-name-to-ISO dict already exists (advisory_poller.py, ~200 entries) and handles the common case cheaply. The LLM is reserved for its actual value-add: CAMEO categorization and ambiguous-country inference.
- The comparison panel uses dual bars (not sparklines) as the primary visual because each entry needs to be glanceable in a scrollable list. The time-series sparkline lives in the expanded card where there's room for it — this avoids duplicating temporal data at two zoom levels.
- Daily re-forecasting was chosen over weekly because Polymarket prices move on daily news cycles. A weekly Geopol re-forecast would always be 1-6 days stale relative to market movement, making divergence tracking less meaningful.

</specifics>

<deferred>
## Deferred Ideas

- **Aggregate scorecard** (win/loss record, average Brier score vs market) — defer until enough resolved comparisons exist for statistical significance. Adding it prematurely with 2-3 data points would be misleading.
- **Multi-country question handling** — when LLM extracts multiple countries from a Polymarket question, generate forecasts for each country separately or treat as a single multi-country forecast. Defer — handle as single best-match country initially.
- **Polymarket-driven forecast notifications** — alert users when a new Polymarket question triggers a Geopol forecast. Defer — no notification infrastructure exists.
- **Market resolution verification** — cross-check Polymarket's outcome resolution against GDELT ground truth to validate the market's own accuracy. Defer — research-grade analysis, not a UX feature.

</deferred>

---

*Phase: 18-polymarket-driven-forecasting*
*Context gathered: 2026-03-04*
