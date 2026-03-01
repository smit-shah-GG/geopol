"""
Daily forecast pipeline components.

This package implements the autonomous daily forecast cycle:
1. QuestionGenerator -- LLM-based forecast question generation from GDELT events
2. BudgetTracker -- Gemini budget management with PendingQuestion queue
3. OutcomeResolver -- Ground-truth resolution comparing predictions to events
4. DailyPipeline -- Orchestrator tying all components into a 4-phase daily cycle
"""

from src.pipeline.budget_tracker import BudgetTracker
from src.pipeline.daily_forecast import DailyPipeline, PipelineResult
from src.pipeline.outcome_resolver import OutcomeResolver
from src.pipeline.question_generator import GeneratedQuestion, QuestionGenerator

__all__ = [
    "BudgetTracker",
    "DailyPipeline",
    "GeneratedQuestion",
    "OutcomeResolver",
    "PipelineResult",
    "QuestionGenerator",
]
