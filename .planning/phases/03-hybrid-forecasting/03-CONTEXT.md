# Phase 3: Hybrid Forecasting - Context

**Gathered:** 2026-01-10
**Status:** Ready for planning

<vision>
## How This Should Work

The forecasting system uses an LLM-first approach with Gemini API integration. When given a geopolitical question or scenario, Gemini generates multiple future paths (scenario trees) - different ways events could unfold. The system follows a multi-step reasoning process:

1. Gemini generates initial scenarios based on the question
2. The temporal knowledge graph validates each scenario against historical patterns
3. Gemini refines predictions based on the validation feedback
4. The system produces a final ensemble prediction with probabilities

The key is RAG integration - feeding relevant historical events from our knowledge graph to Gemini for grounded predictions. This ensures scenarios are anchored in real patterns, not just language model hallucinations.

This is accessed through a simple command-line interface for the PoC stage - something like `python forecast.py "Will Russia-Ukraine conflict escalate?"` that returns structured predictions with clear reasoning paths.

</vision>

<essential>
## What Must Be Nailed

All three aspects are equally critical for this phase:

- **Clear reasoning chains** - Every prediction step must be explainable, showing how historical patterns inform future scenarios
- **Scenario diversity** - Generate multiple distinct futures, not just variations of the same prediction
- **Confidence calibration** - Accurate probability scores that reflect genuine uncertainty, not overconfident predictions

</essential>

<boundaries>
## What's Out of Scope

- **Complex UI** - No web interface, dashboards, or visualization - just command-line
- **Production features** - No authentication, rate limiting, monitoring - just the core algorithm
- **Real-time processing** - No streaming updates or live data feeds - batch analysis only
- **Local LLM models** - Using Gemini API, not deploying 7B models locally

</boundaries>

<specifics>
## Specific Ideas

- RAG integration to feed relevant historical events from the graph to Gemini
- Multi-step reasoning flow with validation feedback loops
- Command-line first interface for testing and experimentation
- Simple invocation like `python forecast.py [question]`
- Focus on getting to working PoC/MVP stage first

</specifics>

<notes>
## Additional Context

The core innovation is the multi-step LLM-guided search pattern with graph validation feedback. Instead of the graph algorithms driving and the LLM explaining, Gemini proposes scenarios which get validated and refined based on historical patterns from our temporal knowledge graph.

Using Gemini API instead of local models simplifies deployment and gives access to more powerful reasoning capabilities. The RAG integration ensures predictions are grounded in actual historical precedents captured in Phase 2's knowledge graph.

Priority is on getting a working proof of concept that demonstrates the hybrid approach, not on UI or production readiness.

</notes>

---

*Phase: 03-hybrid-forecasting*
*Context gathered: 2026-01-10*