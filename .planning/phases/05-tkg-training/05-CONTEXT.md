# Phase 5: TKG Training - Context

**Gathered:** 2026-01-13
**Status:** Ready for planning

<vision>
## How This Should Work

The TKG predictor should follow a periodic retraining model - start with an initial training on historical GDELT data, then retrain on a regular schedule (weekly or monthly) to keep patterns fresh. The system ingests all event types (conflicts, diplomatic, economic, social) because everything affects geopolitics - economic sanctions lead to conflicts, social unrest affects diplomacy.

The retraining happens on a predictable time-based schedule rather than being triggered by events or performance metrics. This keeps it simple and consistent - every Sunday night or first of the month, the model updates itself with the latest patterns.

Training should show visible progress - not just a black box that spits out a model at the end. Show what patterns are being learned, how the knowledge graph is evolving, maybe visualizations of key actor relationships. It should feel like watching the system learn about the world.

</vision>

<essential>
## What Must Be Nailed

- **Actually works** - The TKG predictor must produce real predictions that meaningfully improve forecasting accuracy, not just be a placeholder component

This is the core requirement. If the TKG doesn't contribute real value to predictions, the whole phase fails. The 40% weight in the ensemble needs to be justified by actual pattern matching that complements the LLM's reasoning.

</essential>

<boundaries>
## What's Out of Scope

- **Perfect accuracy** - This isn't about achieving state-of-the-art ML performance, just meaningful pattern matching
- **Real-time updates** - No need for streaming or live event ingestion, batch processing is fine
- **Production automation** - No CI/CD pipelines or automated deployments needed, manual training is acceptable
- **Complex architectures** - Not explicitly excluded, but the focus is on working predictions over architectural sophistication

</boundaries>

<specifics>
## Specific Ideas

- **Visible progress during training** - Show what's happening: patterns being learned, graph statistics, maybe visualizations of the evolving knowledge graph
- **Time-based retraining schedule** - Automatic retraining weekly or monthly for consistency
- **All QuadClass events** - Include all GDELT event types (1-4) since economic and social factors influence conflicts

</specifics>

<notes>
## Additional Context

The system currently runs in LLM-only mode with the TKG component returning placeholder predictions. This phase activates that missing 40% of the hybrid system's capability. The emphasis is on getting it working and contributing real value, not on optimization or perfection.

The periodic retraining model balances freshness with simplicity - we don't need complex triggers or monitoring, just a reliable schedule that keeps the model current with world events.

</notes>

---

*Phase: 05-tkg-training*
*Context gathered: 2026-01-13*