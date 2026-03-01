"""
Daily forecast pipeline orchestrator.

Runs the complete 4-phase daily forecast cycle:
1. Dequeue carryover questions from yesterday's budget exhaustion
2. Generate fresh questions from recent GDELT events (budget permitting)
3. Predict and persist forecasts (EnsemblePredictor -> ForecastService)
4. Resolve expired predictions against GDELT ground truth

Handles budget exhaustion by queuing overflow questions to PendingQuestion.
Consecutive failure alerting at >= 2 failures triggers CRITICAL log.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from src.api.services.forecast_service import ForecastService
from src.forecasting.ensemble_predictor import EnsemblePredictor
from src.pipeline.budget_tracker import BudgetTracker
from src.pipeline.outcome_resolver import OutcomeResolver
from src.pipeline.question_generator import GeneratedQuestion, QuestionGenerator

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Result of a daily pipeline run."""

    questions_generated: int = 0
    forecasts_produced: int = 0
    questions_queued: int = 0
    outcomes_resolved: int = 0
    errors: list[str] = field(default_factory=list)
    duration_seconds: float = 0.0
    success: bool = False


class DailyPipeline:
    """Orchestrate the complete daily forecast cycle.

    The pipeline runs four phases sequentially:
    1. Dequeue pending (carryover) questions from previous runs.
    2. Generate fresh questions from GDELT events via Gemini.
    3. Predict each question via EnsemblePredictor and persist via ForecastService.
    4. Resolve expired predictions against GDELT ground truth.

    Budget exhaustion during phase 3 queues remaining questions for the next run.

    Attributes:
        question_generator: QuestionGenerator for LLM-based question gen.
        budget_tracker: BudgetTracker for Gemini budget management.
        outcome_resolver: OutcomeResolver for expired prediction resolution.
        ensemble_predictor: EnsemblePredictor for forecast generation.
        async_session_factory: SQLAlchemy async session factory.
    """

    def __init__(
        self,
        question_generator: QuestionGenerator,
        budget_tracker: BudgetTracker,
        outcome_resolver: OutcomeResolver,
        ensemble_predictor: EnsemblePredictor,
        async_session_factory: async_sessionmaker[AsyncSession],
    ) -> None:
        self.question_generator = question_generator
        self.budget_tracker = budget_tracker
        self.outcome_resolver = outcome_resolver
        self.ensemble_predictor = ensemble_predictor
        self._session_factory = async_session_factory
        self._consecutive_failures: int = 0

    async def run_daily(
        self,
        max_questions: int | None = None,
        skip_outcomes: bool = False,
        dry_run: bool = False,
    ) -> PipelineResult:
        """Execute the complete daily forecast cycle.

        Args:
            max_questions: Override maximum questions to process.
            skip_outcomes: Skip phase 4 (outcome resolution).
            dry_run: Generate questions but don't predict or persist.

        Returns:
            PipelineResult with counts and success status.
        """
        start = time.monotonic()
        result = PipelineResult()
        all_questions: list[tuple[GeneratedQuestion, int | None]] = []
        # tuple: (question, pending_question_id or None for fresh)

        try:
            # Phase 1: Dequeue carryover questions
            logger.info("Phase 1: Dequeuing carryover questions")
            pending = await self.budget_tracker.dequeue_pending(limit=10)
            for p in pending:
                q = GeneratedQuestion(
                    question=p.question,
                    country_iso=p.country_iso or "XX",
                    horizon_days=p.horizon_days,
                    category=p.category,
                )
                all_questions.append((q, p.id))
            logger.info("Dequeued %d carryover questions", len(pending))

            # Phase 2: Generate fresh questions (budget permitting)
            budget_remaining = await self.budget_tracker.check_budget()
            logger.info("Phase 2: Budget remaining = %d", budget_remaining)

            if budget_remaining > len(all_questions):
                fresh_limit = budget_remaining - len(all_questions)
                if max_questions is not None:
                    fresh_limit = min(fresh_limit, max_questions - len(all_questions))
                    fresh_limit = max(0, fresh_limit)

                if fresh_limit > 0:
                    fresh = await self.question_generator.generate_questions(
                        n_questions=fresh_limit
                    )
                    result.questions_generated = len(fresh)
                    for q in fresh:
                        all_questions.append((q, None))
                    logger.info("Generated %d fresh questions", len(fresh))
            else:
                logger.warning("Budget exhausted, skipping fresh question generation")

            if max_questions is not None:
                all_questions = all_questions[:max_questions]

            if dry_run:
                logger.info("DRY RUN: skipping prediction and persistence")
                result.questions_generated = len(all_questions)
                result.success = True
                result.duration_seconds = time.monotonic() - start
                return result

            # Phase 3: Predict and persist
            logger.info("Phase 3: Predicting %d questions", len(all_questions))
            for question, pending_id in all_questions:
                # Check budget before each prediction (Gemini is called by EnsemblePredictor)
                remaining = await self.budget_tracker.check_budget()
                if remaining <= 0:
                    logger.warning(
                        "Budget exhausted mid-pipeline, queuing remaining questions"
                    )
                    # Queue this and all remaining questions
                    idx = all_questions.index((question, pending_id))
                    for q, pid in all_questions[idx:]:
                        if pid is None:  # Only queue fresh questions, carryover stays
                            await self.budget_tracker.queue_question(q)
                            result.questions_queued += 1
                    break

                try:
                    # EnsemblePredictor.predict() is synchronous
                    ensemble_pred, forecast_output = await asyncio.to_thread(
                        self.ensemble_predictor.predict,
                        question=question.question,
                        category=question.category,
                    )

                    # Persist via ForecastService
                    async with self._session_factory() as session:
                        service = ForecastService(session)
                        await service.persist_forecast(
                            forecast_output=forecast_output,
                            ensemble_prediction=ensemble_pred,
                            country_iso=question.country_iso,
                            horizon_days=question.horizon_days,
                        )
                        await session.commit()

                    # Increment budget counter
                    await self.budget_tracker.increment()

                    # Mark carryover question as completed
                    if pending_id is not None:
                        await self.budget_tracker.mark_completed(pending_id)

                    result.forecasts_produced += 1
                    logger.info(
                        "Forecast produced: p=%.3f country=%s q='%s'",
                        forecast_output.probability,
                        question.country_iso,
                        question.question[:60],
                    )

                except Exception as exc:
                    error_msg = f"Prediction failed for '{question.question[:50]}': {exc}"
                    logger.error(error_msg)
                    result.errors.append(error_msg)

            # Phase 4: Resolve outcomes
            if not skip_outcomes:
                logger.info("Phase 4: Resolving expired predictions")
                try:
                    outcomes = await self.outcome_resolver.resolve_expired_predictions()
                    result.outcomes_resolved = len(outcomes)
                    logger.info("Resolved %d expired predictions", len(outcomes))
                except Exception as exc:
                    error_msg = f"Outcome resolution failed: {exc}"
                    logger.error(error_msg)
                    result.errors.append(error_msg)

            result.success = len(result.errors) == 0
            result.duration_seconds = time.monotonic() - start

            logger.info(
                "Pipeline complete: generated=%d, produced=%d, queued=%d, "
                "resolved=%d, errors=%d, duration=%.1fs",
                result.questions_generated,
                result.forecasts_produced,
                result.questions_queued,
                result.outcomes_resolved,
                len(result.errors),
                result.duration_seconds,
            )
            return result

        except Exception as exc:
            result.errors.append(f"Pipeline failed: {exc}")
            result.duration_seconds = time.monotonic() - start
            logger.error("Pipeline failed with unhandled exception: %s", exc)
            raise

    async def run_with_retry(
        self,
        max_retries: int = 2,
        retry_delay_seconds: float = 300.0,
        **kwargs,
    ) -> PipelineResult:
        """Run the daily pipeline with retry on failure.

        On success, resets the consecutive failure counter.
        On failure, increments the counter and emits CRITICAL alert at >= 2.
        Retries up to max_retries with retry_delay_seconds between attempts.

        Args:
            max_retries: Maximum retry attempts.
            retry_delay_seconds: Delay between retries in seconds.
            **kwargs: Passed to run_daily().

        Returns:
            PipelineResult from the first successful run, or the last failed run.
        """
        last_result = PipelineResult()

        for attempt in range(max_retries + 1):
            try:
                result = await self.run_daily(**kwargs)
                if result.success:
                    self._consecutive_failures = 0
                    return result
                last_result = result
            except Exception as exc:
                last_result = PipelineResult(
                    errors=[str(exc)],
                    success=False,
                )

            self._consecutive_failures += 1
            logger.error(
                "Pipeline attempt %d/%d failed (consecutive failures: %d)",
                attempt + 1,
                max_retries + 1,
                self._consecutive_failures,
            )

            if self._consecutive_failures >= 2:
                logger.critical(
                    "ALERT: %d consecutive daily pipeline failures. "
                    "Manual investigation required.",
                    self._consecutive_failures,
                )

            if attempt < max_retries:
                logger.info(
                    "Retrying in %.0f seconds...", retry_delay_seconds
                )
                await asyncio.sleep(retry_delay_seconds)

        return last_result
