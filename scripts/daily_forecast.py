#!/usr/bin/env python3
"""
Entry point for the daily forecast pipeline (systemd timer target).

Initializes all pipeline components, runs the 4-phase daily cycle with
retry, and exits with code 0 on success or 1 on failure.

Usage:
    python scripts/daily_forecast.py [OPTIONS]

Options:
    --max-questions N    Override maximum questions to process (default: budget limit)
    --skip-outcomes      Skip phase 4 (outcome resolution)
    --dry-run            Generate questions only, do not predict or persist
    -h, --help           Show this help message
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Ensure project root is on sys.path for src.* imports
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.logging_config import setup_logging
from src.settings import get_settings


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Daily forecast pipeline -- generate, predict, persist, resolve.",
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=None,
        help="Override maximum questions to process (default: budget limit)",
    )
    parser.add_argument(
        "--skip-outcomes",
        action="store_true",
        help="Skip phase 4 (outcome resolution)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate questions only, do not predict or persist",
    )
    return parser.parse_args()


async def _run(args: argparse.Namespace) -> int:
    """Initialize components and run pipeline. Returns exit code."""
    settings = get_settings()
    setup_logging(level=settings.log_level, json_format=settings.log_json)
    logger = logging.getLogger(__name__)

    logger.info("Daily forecast pipeline starting (env=%s)", settings.environment)

    # -- Initialize infrastructure --

    # PostgreSQL
    from src.db.postgres import init_db, async_session_factory as get_factory

    init_db()
    from src.db import postgres

    session_factory = postgres.async_session_factory
    if session_factory is None:
        logger.critical("Failed to initialize PostgreSQL session factory")
        return 1

    # Redis
    from src.api.deps import get_redis

    redis_client = await get_redis()

    # -- Initialize pipeline components (sync v1.0 code via asyncio.to_thread) --

    # EventStorage (synchronous, SQLite)
    from src.database.storage import EventStorage

    event_storage = await asyncio.to_thread(
        EventStorage, settings.gdelt_db_path
    )

    # GeminiClient
    try:
        from src.forecasting.gemini_client import GeminiClient

        gemini_client = await asyncio.to_thread(GeminiClient)
    except Exception as exc:
        logger.warning("Gemini client initialization failed: %s", exc)
        gemini_client = None

    # EnsemblePredictor (with optional components)
    from src.forecasting.ensemble_predictor import EnsemblePredictor

    llm_orchestrator = None
    tkg_predictor = None

    if gemini_client is not None:
        try:
            from src.forecasting.rag_pipeline import RAGPipeline

            rag_pipeline = await asyncio.to_thread(RAGPipeline)

            from src.forecasting.scenario_generator import ScenarioGenerator

            scenario_gen = ScenarioGenerator(gemini_client)

            from src.forecasting.graph_validator import GraphValidator
            from src.forecasting.reasoning_orchestrator import ReasoningOrchestrator

            graph_validator = GraphValidator()
            llm_orchestrator = ReasoningOrchestrator(
                gemini_client=gemini_client,
                scenario_generator=scenario_gen,
                rag_pipeline=rag_pipeline,
                graph_validator=graph_validator,
            )
            logger.info("LLM orchestrator initialized")
        except Exception as exc:
            logger.warning("LLM orchestrator init failed (will use TKG-only): %s", exc)

    ensemble_predictor = EnsemblePredictor(
        llm_orchestrator=llm_orchestrator,
        tkg_predictor=tkg_predictor,
    )

    # Pipeline components
    from src.pipeline.question_generator import QuestionGenerator
    from src.pipeline.budget_tracker import BudgetTracker
    from src.pipeline.outcome_resolver import OutcomeResolver
    from src.pipeline.daily_forecast import DailyPipeline

    question_generator = QuestionGenerator(
        gemini_client=gemini_client,
        event_storage=event_storage,
    ) if gemini_client is not None else None

    budget_tracker = BudgetTracker(
        async_session_factory=session_factory,
        redis_client=redis_client,
    )

    outcome_resolver = OutcomeResolver(
        async_session_factory=session_factory,
        event_storage=event_storage,
        gemini_client=gemini_client,
    )

    if question_generator is None:
        logger.critical(
            "Cannot run pipeline without Gemini client. "
            "Set GEMINI_API_KEY environment variable."
        )
        return 1

    pipeline = DailyPipeline(
        question_generator=question_generator,
        budget_tracker=budget_tracker,
        outcome_resolver=outcome_resolver,
        ensemble_predictor=ensemble_predictor,
        async_session_factory=session_factory,
    )

    # -- Run pipeline --
    result = await pipeline.run_with_retry(
        max_retries=2,
        retry_delay_seconds=300.0,
        max_questions=args.max_questions,
        skip_outcomes=args.skip_outcomes,
        dry_run=args.dry_run,
    )

    if result.success:
        logger.info(
            "Pipeline completed successfully: generated=%d, produced=%d, "
            "queued=%d, resolved=%d, duration=%.1fs",
            result.questions_generated,
            result.forecasts_produced,
            result.questions_queued,
            result.outcomes_resolved,
            result.duration_seconds,
        )
        return 0
    else:
        logger.error(
            "Pipeline completed with errors: %s", "; ".join(result.errors)
        )
        return 1


def main() -> None:
    """Parse args and run the async pipeline."""
    args = _parse_args()
    exit_code = asyncio.run(_run(args))
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
