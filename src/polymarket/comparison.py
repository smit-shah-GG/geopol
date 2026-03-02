"""
Polymarket comparison service: matching, snapshot capture, resolution, and querying.

Orchestrates the full lifecycle of Polymarket-vs-Geopol comparisons:
  1. Matching cycle: discover markets, match to predictions, create tracking rows
  2. Snapshot capture: periodic price/probability snapshots for time-series
  3. Resolution: score completed markets with Brier score comparison
  4. Querying: active/resolved comparisons and summary statistics
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models import (
    PolymarketComparison,
    PolymarketSnapshot,
    Prediction,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from src.polymarket.client import PolymarketClient
    from src.polymarket.matcher import PolymarketMatcher
    from src.settings import Settings

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _parse_outcome_price(markets: list[dict]) -> float | None:
    """Extract the 'Yes' outcome price from market data.

    Polymarket binary markets typically have two outcomes. The price
    of the 'Yes' outcome is the market-implied probability (0.0-1.0).

    Returns None if price cannot be determined.
    """
    for market in markets:
        outcome_prices = market.get("outcomePrices")
        if isinstance(outcome_prices, str):
            # Sometimes returned as JSON string: "[\"0.65\",\"0.35\"]"
            import json

            try:
                outcome_prices = json.loads(outcome_prices)
            except (json.JSONDecodeError, TypeError):
                continue

        if isinstance(outcome_prices, list) and len(outcome_prices) >= 1:
            try:
                return float(outcome_prices[0])
            except (ValueError, TypeError):
                continue

        # Fallback: bestBid as price proxy
        best_bid = market.get("bestBid")
        if best_bid is not None:
            try:
                return float(best_bid)
            except (ValueError, TypeError):
                continue

    return None


class PolymarketComparisonService:
    """Full lifecycle management for Polymarket-vs-Geopol comparisons.

    Args:
        async_session_factory: Callable returning an AsyncSession context manager
            (typically the SQLAlchemy async_sessionmaker).
        polymarket_client: PolymarketClient instance for API access.
        matcher: PolymarketMatcher instance for event-to-prediction matching.
        settings: Application settings (used for match threshold, enabled flag).
    """

    def __init__(
        self,
        async_session_factory: Callable[..., AsyncSession],
        polymarket_client: PolymarketClient,
        matcher: PolymarketMatcher,
        settings: Settings,
    ) -> None:
        self._session_factory = async_session_factory
        self._client = polymarket_client
        self._matcher = matcher
        self._settings = settings

    async def run_matching_cycle(self) -> dict[str, int]:
        """Fetch markets, match to predictions, create comparison rows.

        Returns:
            Dict with keys: markets_fetched, new_matches, already_tracked.
        """
        if not self._settings.polymarket_enabled:
            logger.debug("Polymarket comparison disabled in settings")
            return {"markets_fetched": 0, "new_matches": 0, "already_tracked": 0}

        events = await self._client.fetch_geopolitical_markets()
        result = {"markets_fetched": len(events), "new_matches": 0, "already_tracked": 0}

        if not events:
            return result

        async with self._session_factory() as session:
            # Load already-tracked Polymarket event IDs
            tracked_stmt = select(PolymarketComparison.polymarket_event_id)
            tracked_result = await session.execute(tracked_stmt)
            tracked_ids: set[str] = {row[0] for row in tracked_result.fetchall()}

            # Load active predictions (unexpired, with probability)
            now = _utcnow()
            pred_stmt = select(Prediction).where(
                Prediction.expires_at > now,
                Prediction.probability.isnot(None),
            )
            pred_result = await session.execute(pred_stmt)
            active_predictions: list[Prediction] = list(pred_result.scalars().all())

            if not active_predictions:
                logger.info("No active predictions available for matching")
                return result

            for event in events:
                event_id = str(event.get("id", ""))
                if event_id in tracked_ids:
                    result["already_tracked"] += 1
                    continue

                pred_id, confidence = await self._matcher.match_event_to_predictions(
                    event, active_predictions
                )

                if pred_id is None:
                    continue

                # Extract initial price if available
                markets = event.get("markets", [])
                initial_price = _parse_outcome_price(markets) if markets else None

                # Look up geopol probability
                matched_pred = next(
                    (p for p in active_predictions if p.id == pred_id), None
                )
                geopol_prob = matched_pred.probability if matched_pred else None

                comparison = PolymarketComparison(
                    polymarket_event_id=event_id,
                    polymarket_slug=event.get("slug", event_id),
                    polymarket_title=event.get("title", ""),
                    geopol_prediction_id=pred_id,
                    match_confidence=confidence,
                    polymarket_price=initial_price,
                    geopol_probability=geopol_prob,
                    status="active",
                )
                session.add(comparison)
                result["new_matches"] += 1

            await session.commit()

        logger.info(
            "Matching cycle: %d markets, %d new matches, %d already tracked",
            result["markets_fetched"],
            result["new_matches"],
            result["already_tracked"],
        )
        return result

    async def capture_snapshots(self) -> int:
        """Capture current price/probability snapshots for active comparisons.

        For each active comparison, fetches current Polymarket prices and
        re-queries the Geopol prediction for the latest probability.

        Returns:
            Number of snapshots captured.
        """
        count = 0

        async with self._session_factory() as session:
            # Load active comparisons
            stmt = select(PolymarketComparison).where(
                PolymarketComparison.status == "active"
            )
            result = await session.execute(stmt)
            comparisons: list[PolymarketComparison] = list(result.scalars().all())

            if not comparisons:
                return 0

            for comp in comparisons:
                # Fetch current Polymarket price
                markets = await self._client.fetch_event_prices(
                    comp.polymarket_event_id
                )
                pm_price = _parse_outcome_price(markets)
                if pm_price is None:
                    logger.debug(
                        "No price data for event %s, skipping snapshot",
                        comp.polymarket_event_id,
                    )
                    continue

                # Re-query prediction for latest probability
                pred_stmt = select(Prediction.probability).where(
                    Prediction.id == comp.geopol_prediction_id
                )
                pred_result = await session.execute(pred_stmt)
                pred_row = pred_result.first()
                geopol_prob = pred_row[0] if pred_row else comp.geopol_probability
                if geopol_prob is None:
                    geopol_prob = 0.5  # Fallback if prediction no longer exists

                snapshot = PolymarketSnapshot(
                    comparison_id=comp.id,
                    polymarket_price=pm_price,
                    geopol_probability=geopol_prob,
                )
                session.add(snapshot)

                # Update comparison with latest values
                comp.polymarket_price = pm_price
                comp.geopol_probability = geopol_prob
                comp.last_snapshot_at = _utcnow()
                count += 1

            await session.commit()

        logger.info("Captured %d snapshots for %d active comparisons", count, len(comparisons))
        return count

    async def resolve_completed(self) -> int:
        """Resolve comparisons where the Polymarket event has completed.

        Checks each active comparison's event status via the API. For
        resolved events, computes Brier scores for both Polymarket and
        Geopol and updates the comparison row.

        Brier score: (forecast - outcome)^2
          - outcome is 1.0 if event resolved 'Yes', 0.0 if 'No'
          - Lower is better

        Returns:
            Number of comparisons resolved.
        """
        resolved_count = 0

        async with self._session_factory() as session:
            stmt = select(PolymarketComparison).where(
                PolymarketComparison.status == "active"
            )
            result = await session.execute(stmt)
            comparisons: list[PolymarketComparison] = list(result.scalars().all())

            for comp in comparisons:
                # Fetch event to check resolution status
                markets = await self._client.fetch_event_prices(
                    comp.polymarket_event_id
                )
                if not markets:
                    continue

                # Check if any market in the event has resolved
                resolved_market = None
                for market in markets:
                    if market.get("resolved") or market.get("closed"):
                        resolved_market = market
                        break

                if resolved_market is None:
                    continue

                # Determine outcome from resolution data
                outcome = self._extract_outcome(resolved_market)
                if outcome is None:
                    logger.warning(
                        "Cannot determine outcome for resolved event %s",
                        comp.polymarket_event_id,
                    )
                    continue

                # Get final Polymarket price at resolution
                pm_final_price = _parse_outcome_price([resolved_market])
                if pm_final_price is None:
                    pm_final_price = comp.polymarket_price or 0.5

                # Get Geopol probability (use stored value)
                geopol_prob = comp.geopol_probability
                if geopol_prob is None:
                    geopol_prob = 0.5

                # Compute Brier scores
                geopol_brier = (geopol_prob - outcome) ** 2
                polymarket_brier = (pm_final_price - outcome) ** 2

                # Update comparison row
                comp.status = "resolved"
                comp.polymarket_outcome = outcome
                comp.geopol_brier = geopol_brier
                comp.polymarket_brier = polymarket_brier
                comp.resolved_at = _utcnow()
                resolved_count += 1

                logger.info(
                    "Resolved comparison %d: outcome=%.1f, "
                    "geopol_brier=%.4f, polymarket_brier=%.4f",
                    comp.id,
                    outcome,
                    geopol_brier,
                    polymarket_brier,
                )

            await session.commit()

        if resolved_count:
            logger.info("Resolved %d comparisons", resolved_count)
        return resolved_count

    @staticmethod
    def _extract_outcome(market: dict[str, Any]) -> float | None:
        """Extract binary outcome (1.0=Yes, 0.0=No) from a resolved market.

        Polymarket resolved markets typically have outcome prices converged
        to 1.0/0.0. We use the final outcome price of the first outcome
        (conventionally 'Yes').
        """
        # Method 1: resolved outcome prices (most reliable)
        outcome_prices = market.get("outcomePrices")
        if isinstance(outcome_prices, str):
            import json

            try:
                outcome_prices = json.loads(outcome_prices)
            except (json.JSONDecodeError, TypeError):
                outcome_prices = None

        if isinstance(outcome_prices, list) and len(outcome_prices) >= 1:
            try:
                price = float(outcome_prices[0])
                # Resolved markets should have prices near 0 or 1
                if price >= 0.95:
                    return 1.0
                elif price <= 0.05:
                    return 0.0
                # If not near extremes, market may not be fully resolved
                return price
            except (ValueError, TypeError):
                pass

        # Method 2: winner field
        winner = market.get("winner")
        if winner is not None:
            if str(winner).lower() in ("yes", "true", "1"):
                return 1.0
            elif str(winner).lower() in ("no", "false", "0"):
                return 0.0

        return None

    async def get_active_comparisons(self) -> list[dict[str, Any]]:
        """Query active comparisons with latest snapshot data.

        Returns:
            List of dicts with comparison details suitable for API response.
        """
        async with self._session_factory() as session:
            stmt = (
                select(PolymarketComparison)
                .where(PolymarketComparison.status == "active")
                .order_by(PolymarketComparison.created_at.desc())
            )
            result = await session.execute(stmt)
            comparisons = result.scalars().all()

            items: list[dict[str, Any]] = []
            for comp in comparisons:
                items.append({
                    "id": comp.id,
                    "polymarket_event_id": comp.polymarket_event_id,
                    "polymarket_slug": comp.polymarket_slug,
                    "polymarket_title": comp.polymarket_title,
                    "geopol_prediction_id": comp.geopol_prediction_id,
                    "match_confidence": comp.match_confidence,
                    "polymarket_price": comp.polymarket_price,
                    "geopol_probability": comp.geopol_probability,
                    "last_snapshot_at": (
                        comp.last_snapshot_at.isoformat()
                        if comp.last_snapshot_at
                        else None
                    ),
                    "created_at": comp.created_at.isoformat(),
                })

            return items

    async def get_resolved_comparisons(
        self, limit: int = 50
    ) -> list[dict[str, Any]]:
        """Query resolved comparisons ordered by resolution date.

        Args:
            limit: Maximum number of resolved comparisons to return.

        Returns:
            List of dicts with comparison details and Brier scores.
        """
        async with self._session_factory() as session:
            stmt = (
                select(PolymarketComparison)
                .where(PolymarketComparison.status == "resolved")
                .order_by(PolymarketComparison.resolved_at.desc())
                .limit(limit)
            )
            result = await session.execute(stmt)
            comparisons = result.scalars().all()

            items: list[dict[str, Any]] = []
            for comp in comparisons:
                items.append({
                    "id": comp.id,
                    "polymarket_event_id": comp.polymarket_event_id,
                    "polymarket_slug": comp.polymarket_slug,
                    "polymarket_title": comp.polymarket_title,
                    "geopol_prediction_id": comp.geopol_prediction_id,
                    "match_confidence": comp.match_confidence,
                    "polymarket_outcome": comp.polymarket_outcome,
                    "geopol_brier": comp.geopol_brier,
                    "polymarket_brier": comp.polymarket_brier,
                    "geopol_wins": (
                        comp.geopol_brier < comp.polymarket_brier
                        if comp.geopol_brier is not None
                        and comp.polymarket_brier is not None
                        else None
                    ),
                    "resolved_at": (
                        comp.resolved_at.isoformat() if comp.resolved_at else None
                    ),
                    "created_at": comp.created_at.isoformat(),
                })

            return items

    async def get_comparison_summary(self) -> dict[str, Any]:
        """Aggregate summary statistics across all comparisons.

        Returns:
            Dict with: active_count, resolved_count, geopol_avg_brier,
            polymarket_avg_brier, geopol_wins.
        """
        async with self._session_factory() as session:
            # Active count
            active_stmt = select(func.count()).where(
                PolymarketComparison.status == "active"
            ).select_from(PolymarketComparison)
            active_result = await session.execute(active_stmt)
            active_count = active_result.scalar() or 0

            # Resolved aggregates
            resolved_stmt = select(
                func.count(),
                func.avg(PolymarketComparison.geopol_brier),
                func.avg(PolymarketComparison.polymarket_brier),
            ).where(
                PolymarketComparison.status == "resolved"
            ).select_from(PolymarketComparison)
            resolved_result = await session.execute(resolved_stmt)
            row = resolved_result.first()
            resolved_count = row[0] if row else 0
            geopol_avg_brier = float(row[1]) if row and row[1] is not None else None
            polymarket_avg_brier = float(row[2]) if row and row[2] is not None else None

            # Geopol wins (lower Brier is better)
            wins_stmt = select(func.count()).where(
                PolymarketComparison.status == "resolved",
                PolymarketComparison.geopol_brier.isnot(None),
                PolymarketComparison.polymarket_brier.isnot(None),
                PolymarketComparison.geopol_brier < PolymarketComparison.polymarket_brier,
            ).select_from(PolymarketComparison)
            wins_result = await session.execute(wins_stmt)
            geopol_wins = wins_result.scalar() or 0

            return {
                "active_count": active_count,
                "resolved_count": resolved_count,
                "geopol_avg_brier": geopol_avg_brier,
                "polymarket_avg_brier": polymarket_avg_brier,
                "geopol_wins": geopol_wins,
            }
