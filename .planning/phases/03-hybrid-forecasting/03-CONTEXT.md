# Phase 3: Hybrid Forecasting - Context

**Gathered:** 2026-01-09
**Status:** Ready for research

<vision>
## How This Should Work

The forecasting system uses an LLM-guided search approach where the language model generates hypotheses and the temporal knowledge graph validates them with historical patterns.

When given a geopolitical question or scenario, the LLM generates multiple future paths (scenario trees) - different ways events could unfold. Then the graph algorithms check these scenarios against historical precedents to assign probabilities and validate plausibility.

This is accessed through a simple command-line interface for the PoC stage - something like `python forecast.py "Will Russia-Ukraine conflict escalate?"` that returns structured predictions with clear reasoning.

</vision>

<essential>
## What Must Be Nailed

All three aspects are equally critical for this phase:

- **Explainability** - Every forecast must show clear reasoning paths from historical precedents
- **Probability calibration** - Getting accurate confidence scores, not just predictions
- **Scenario diversity** - Capturing multiple plausible futures, not just the most likely

</essential>

<boundaries>
## What's Out of Scope

- **Real-time processing** - Focus on batch analysis for the PoC, not streaming predictions
- **UI/visualization** - Just the prediction engine, command-line interface only
- **Production deployment** - This is a working PoC/MVP, not production-ready

</boundaries>

<specifics>
## Specific Ideas

- Command-line first interface for testing and experimentation
- LLM generates scenario trees, graph validates against history
- Simple invocation like `python forecast.py [question]`
- Focus on getting to working PoC/MVP stage first

</specifics>

<notes>
## Additional Context

The core innovation is the LLM-guided search pattern: instead of the graph algorithms driving and the LLM explaining, the LLM proposes scenarios and the graph validates them. This allows for more creative hypothesis generation while still grounding predictions in historical patterns.

Priority is on getting a working proof of concept that demonstrates the hybrid approach, not on UI or production readiness.

</notes>

---

*Phase: 03-hybrid-forecasting*
*Context gathered: 2026-01-09*