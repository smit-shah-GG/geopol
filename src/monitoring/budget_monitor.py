"""
Gemini API usage tracking for health reporting.

Read-only reporter that queries today's prediction count from PostgreSQL
and computes budget utilisation metrics.  This is NOT the budget enforcer
(that role belongs to BudgetTracker in src/pipeline/budget_tracker.py).

BudgetMonitor provides data for the health endpoint and daily digest
alerts.  It does not decrement or enforce limits.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from sqlalchemy import func, select

from src.db.models import Prediction

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

    from src.settings import Settings

logger = logging.getLogger(__name__)


class BudgetMonitor:
    """Read-only Gemini API budget status reporter.

    Attributes:
        _daily_budget: Maximum Gemini calls per day (from settings).
    """

    def __init__(
        self,
        async_session_factory: async_sessionmaker[AsyncSession],
        settings: Settings,
    ) -> None:
        self._session_factory = async_session_factory
        self._daily_budget: int = settings.gemini_daily_budget

    async def get_budget_status(self) -> dict[str, Any]:
        """Query today's prediction count and compute budget utilisation.

        Uses the same PostgreSQL counting logic as
        BudgetTracker._check_budget_pg, but returns a status dict
        instead of a bare integer.

        Returns:
            Dict with budget_total, budget_used, budget_remaining,
            pct_used, exhausted.
        """
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
                used = result.scalar_one()
        except Exception as exc:
            logger.error("Failed to query budget status: %s", exc)
            return {
                "budget_total": self._daily_budget,
                "budget_used": 0,
                "budget_remaining": self._daily_budget,
                "pct_used": 0.0,
                "exhausted": False,
                "error": str(exc)[:200],
            }

        remaining = max(0, self._daily_budget - used)
        pct_used = (used / self._daily_budget * 100) if self._daily_budget > 0 else 0.0

        return {
            "budget_total": self._daily_budget,
            "budget_used": used,
            "budget_remaining": remaining,
            "pct_used": round(pct_used, 1),
            "exhausted": remaining == 0,
        }
