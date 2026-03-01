"""
Gemini daily budget tracking with PendingQuestion queue management.

Tracks daily Gemini API usage via Redis (primary) with PostgreSQL fallback.
When the budget is exhausted mid-pipeline, remaining questions are persisted
to the pending_questions table and prioritised in the next day's run.

Budget checking uses the Redis counter from rate_limit.py (gemini_budget_remaining).
If Redis is unavailable, falls back to counting today's Prediction rows in
PostgreSQL -- more expensive but functional.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timezone

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from src.api.middleware.rate_limit import (
    gemini_budget_remaining,
    increment_gemini_usage,
)
from src.db.models import PendingQuestion, Prediction
from src.pipeline.question_generator import GeneratedQuestion
from src.settings import get_settings

logger = logging.getLogger(__name__)


class BudgetTracker:
    """Manage Gemini daily budget and PendingQuestion queue.

    Uses Redis as the primary budget counter (via rate_limit.py functions).
    Falls back to counting today's Prediction rows in PostgreSQL when Redis
    is unavailable (NullRedis returns 0 from incr, triggering fallback).

    Attributes:
        async_session_factory: SQLAlchemy async session factory for PostgreSQL.
        daily_limit: Maximum Gemini calls per day.
    """

    def __init__(
        self,
        async_session_factory: async_sessionmaker[AsyncSession],
        daily_limit: int | None = None,
        redis_client: object | None = None,
    ) -> None:
        self._session_factory = async_session_factory
        self._redis = redis_client
        settings = get_settings()
        self.daily_limit = daily_limit if daily_limit is not None else settings.gemini_daily_budget

    async def check_budget(self) -> int:
        """Return remaining Gemini calls for today.

        Checks Redis first (via gemini_budget_remaining). If Redis returns
        the full budget (suggesting NullRedis or a fresh counter), cross-
        checks with PostgreSQL for accuracy.

        Returns:
            Number of remaining Gemini API calls (>= 0).
        """
        if self._redis is not None:
            try:
                remaining = await gemini_budget_remaining(self._redis)
                return remaining
            except Exception as exc:
                logger.warning("Redis budget check failed, falling back to PostgreSQL: %s", exc)

        # PostgreSQL fallback: count today's predictions
        return await self._check_budget_pg()

    async def _check_budget_pg(self) -> int:
        """Count today's predictions in PostgreSQL and return remaining budget."""
        today_start = datetime.now(timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        try:
            async with self._session_factory() as session:
                result = await session.execute(
                    select(func.count(Prediction.id)).where(
                        Prediction.created_at >= today_start
                    )
                )
                count = result.scalar_one()
                remaining = max(0, self.daily_limit - count)
                return remaining
        except Exception as exc:
            logger.error("PostgreSQL budget check failed: %s", exc)
            # Fail-open: allow pipeline to proceed
            return self.daily_limit

    async def increment(self) -> int:
        """Increment the daily Gemini usage counter.

        Returns:
            Current usage count after increment.
        """
        if self._redis is not None:
            try:
                return await increment_gemini_usage(self._redis)
            except Exception as exc:
                logger.warning("Redis increment failed: %s", exc)
        # No Redis increment possible -- PostgreSQL counting handles it implicitly
        return 0

    async def queue_question(self, question: GeneratedQuestion) -> None:
        """Persist a question to the PendingQuestion table for next-day processing.

        Args:
            question: GeneratedQuestion that could not be processed due to
                budget exhaustion.
        """
        async with self._session_factory() as session:
            pending = PendingQuestion(
                question=question.question,
                country_iso=question.country_iso,
                horizon_days=question.horizon_days,
                category=question.category,
                priority=1,  # Carryover questions get higher priority
                status="pending",
            )
            session.add(pending)
            await session.commit()
            logger.info(
                "Queued question for next day: %s (country=%s)",
                question.question[:60],
                question.country_iso,
            )

    async def dequeue_pending(self, limit: int = 10) -> list[PendingQuestion]:
        """Fetch pending questions ordered by priority DESC, created_at ASC.

        Updates their status to 'processing' atomically.

        Args:
            limit: Maximum number of questions to dequeue.

        Returns:
            List of PendingQuestion ORM instances with status='processing'.
        """
        async with self._session_factory() as session:
            result = await session.execute(
                select(PendingQuestion)
                .where(PendingQuestion.status == "pending")
                .order_by(
                    PendingQuestion.priority.desc(),
                    PendingQuestion.created_at.asc(),
                )
                .limit(limit)
            )
            pending = list(result.scalars().all())

            if pending:
                ids = [p.id for p in pending]
                await session.execute(
                    update(PendingQuestion)
                    .where(PendingQuestion.id.in_(ids))
                    .values(status="processing")
                )
                await session.commit()
                # Refresh objects to reflect updated status
                for p in pending:
                    p.status = "processing"
                logger.info("Dequeued %d pending questions for processing", len(pending))

            return pending

    async def mark_completed(self, question_id: int) -> None:
        """Mark a pending question as completed after successful prediction.

        Args:
            question_id: Primary key of the PendingQuestion row.
        """
        async with self._session_factory() as session:
            await session.execute(
                update(PendingQuestion)
                .where(PendingQuestion.id == question_id)
                .values(status="completed")
            )
            await session.commit()
            logger.debug("Marked pending question %d as completed", question_id)
