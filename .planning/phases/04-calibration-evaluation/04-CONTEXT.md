# Phase 4: Calibration & Evaluation - Context

**Gathered:** 2026-01-10
**Status:** Ready for planning

<vision>
## How This Should Work

The system automatically evaluates its predictions against real-world outcomes and self-improves its confidence calibration. As events unfold and predictions resolve, the system detects patterns of over or under-confidence and adjusts itself.

The key is that this happens automatically - no manual intervention needed. The system continuously learns from its mistakes, but crucially, it shows exactly WHY it's making adjustments. When it reduces confidence on Russia predictions by 15%, it explains that it detected systematic overconfidence in the last 20 predictions about Russian actions.

For predictions that haven't resolved yet, the system uses provisional scoring - if we predicted 70% chance of conflict by December and it's November with tensions rising, that partial information feeds into calibration even before the final outcome.

</vision>

<essential>
## What Must Be Nailed

- **Trust & transparency** - Every calibration adjustment must be explainable. Users need to see which predictions were wrong, what patterns were detected, and exactly how confidence is being adjusted.
- **Beat human baseline** - Target is achieving better Brier scores than human expert forecasters (typically 0.35). This proves the system works.
- **Self-improvement** - The system must genuinely learn from its errors and improve over time, not just report on them.

</essential>

<boundaries>
## What's Out of Scope

- **Retraining models** - Just calibrate confidence scores, don't retrain the underlying TKG or LLM models
- **New data sources** - Work with existing GDELT data, don't add new feeds or ground truth sources
- **UI/visualization** - Focus on core calibration logic, fancy dashboards come later

</boundaries>

<specifics>
## Specific Ideas

- **Isotonic regression** - Use the standard isotonic regression approach for monotonic calibration as mentioned in geopol.md
- **Per-category calibration** - Different calibration curves for different types of predictions (conflicts vs diplomatic events)
- **Explainable adjustments** - Each calibration comes with clear explanation, e.g., "reducing confidence on Russia predictions by 15% due to systematic overconfidence in 8 of last 10 predictions"
- **Provisional scoring** - Use partial information for unresolved predictions rather than waiting for complete outcomes
- **Alert & log failures** - When calibration detects systematic issues, log them and alert for manual review rather than trying to auto-correct everything

</specifics>

<notes>
## Additional Context

The emphasis throughout is on building trust through transparency. This isn't a black-box system that magically gets better - it's a system that clearly shows its reasoning for every adjustment it makes.

The self-improving aspect is critical but must be stable - no wild swings based on a few predictions. Statistical robustness matters more than speed of adaptation.

Beating human forecasters (0.35 Brier score) is the concrete target that proves the system works in practice, not just in theory.

</notes>

---

*Phase: 04-calibration-evaluation*
*Context gathered: 2026-01-10*