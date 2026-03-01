"""
Daily forecast pipeline components.

This package implements the autonomous daily forecast cycle:
1. QuestionGenerator -- LLM-based forecast question generation from GDELT events
2. BudgetTracker -- Gemini budget management with PendingQuestion queue
3. OutcomeResolver -- Ground-truth resolution comparing predictions to events
4. DailyPipeline -- Orchestrator tying all components into a 4-phase daily cycle

Lazy imports: use ``from src.pipeline.<module> import <Class>`` for
direct imports to avoid circular or missing-module errors during
incremental development.
"""
