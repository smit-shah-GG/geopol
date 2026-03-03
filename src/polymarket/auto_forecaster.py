"""
Auto-forecast pipeline for unmatched high-volume Polymarket questions.

Transforms Polymarket from a passive comparison tool into an active forecast
driver. When a geopolitical Polymarket question has high trading volume but
no matching Geopol prediction, this module generates one via the full
EnsemblePredictor pipeline.

Active comparisons are re-forecasted daily to track probability divergence.

Wired into the existing _polymarket_loop in app.py after the matching cycle.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models import (
    PolymarketComparison,
    PolymarketSnapshot,
    Prediction,
)
from src.ingest.advisory_poller import COUNTRY_NAME_TO_ISO

if TYPE_CHECKING:
    from collections.abc import Callable

    from src.forecasting.gemini_client import GeminiClient
    from src.settings import Settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CAMEO extraction prompt (from RESEARCH.md)
# ---------------------------------------------------------------------------

_CAMEO_EXTRACTION_PROMPT = """Classify this prediction market question into a CAMEO event category.

QUESTION: {question}

Return ONLY a JSON object:
{{"cameo_root_code": "<2-digit CAMEO code>", "category": "<one of: conflict, diplomatic, economic, security, political>"}}

CAMEO codes:
- 01-05: Verbal/material cooperation (diplomatic)
- 06-09: Verbal/material conflict (security)
- 10-14: Demands, protests, sanctions (political/economic)
- 15-17: Military action (conflict)
- 18-20: Physical assault, mass violence (conflict)

If ambiguous, choose the most likely category. Return ONLY JSON."""

_COUNTRY_EXTRACTION_PROMPT = """Given the following prediction market question, identify the single most relevant country.

QUESTION: {question}
DESCRIPTION: {description}

Return ONLY a JSON object:
{{"country_iso": "<ISO 3166-1 alpha-2 code>", "country_name": "<full country name>"}}

If no specific country is relevant, return:
{{"country_iso": null, "country_name": null}}

Return ONLY JSON."""


# ---------------------------------------------------------------------------
# Module-level helper functions
# ---------------------------------------------------------------------------


def compute_horizon_days(end_date_str: str | None) -> int | None:
    """Compute forecast horizon from Polymarket event endDate.

    Returns None if endDate is missing, unparseable, or outside valid range.
    Valid range: 7-365 days from now.
    """
    if not end_date_str:
        return None

    try:
        end_date = datetime.fromisoformat(end_date_str.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None

    now = datetime.now(timezone.utc)
    delta_days = (end_date - now).days

    if delta_days < 7 or delta_days > 365:
        return None  # Outside Geopol's calibrated range

    return delta_days


def extract_country_heuristic(title: str, tags: list[dict]) -> str | None:
    """Check title and tag labels against COUNTRY_NAME_TO_ISO dict.

    Returns ISO alpha-2 code if a country name is found, None otherwise.
    Checks both title text and tag label values (lowercased).
    """
    text = title.lower()
    for tag in tags:
        label = tag.get("label", "")
        if isinstance(label, str):
            text += " " + label.lower()

    # Check longer country names first to avoid "congo" matching before
    # "democratic republic of the congo". Sort by length DESC.
    for name in sorted(COUNTRY_NAME_TO_ISO, key=len, reverse=True):
        if name in text:
            return COUNTRY_NAME_TO_ISO[name]
    return None


async def extract_country_llm(
    title: str, description: str, gemini: GeminiClient
) -> str | None:
    """Ask Gemini to infer the most relevant country ISO code.

    Returns ISO alpha-2 code or None on failure. Runs the Gemini call
    in a thread to avoid blocking the event loop.
    """
    prompt = _COUNTRY_EXTRACTION_PROMPT.format(
        question=title, description=description or ""
    )
    try:
        response = await asyncio.to_thread(gemini.generate_content, prompt)
        text = response.text.strip()
        # Strip markdown code fences if present
        if text.startswith("```"):
            text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        parsed = json.loads(text)
        iso = parsed.get("country_iso")
        if iso and isinstance(iso, str) and len(iso) == 2:
            return iso.upper()
        return None
    except Exception as exc:
        logger.warning("LLM country extraction failed: %s", exc)
        return None


async def extract_cameo_category(
    title: str, gemini: GeminiClient
) -> str:
    """Extract CAMEO root code from a Polymarket question via Gemini.

    Returns the 2-digit CAMEO root code. Defaults to "14" (protest)
    on parse failure.
    """
    prompt = _CAMEO_EXTRACTION_PROMPT.format(question=title)
    try:
        response = await asyncio.to_thread(gemini.generate_content, prompt)
        text = response.text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        parsed = json.loads(text)
        code = parsed.get("cameo_root_code", "14")
        if isinstance(code, str) and code.isdigit():
            return code.zfill(2)
        return "14"
    except Exception as exc:
        logger.warning("CAMEO extraction failed (defaulting to 14): %s", exc)
        return "14"


async def count_today_new_forecasts(session: AsyncSession) -> int:
    """Count predictions with provenance='polymarket_driven' created today (UTC)."""
    today_start = datetime.now(timezone.utc).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    stmt = (
        select(func.count())
        .where(
            Prediction.provenance == "polymarket_driven",
            Prediction.created_at >= today_start,
        )
        .select_from(Prediction)
    )
    result = await session.execute(stmt)
    return result.scalar() or 0


async def count_today_reforecasts(session: AsyncSession) -> int:
    """Count snapshots captured today for active Polymarket-driven comparisons.

    Uses snapshot count as a proxy for re-forecast activity, since each
    re-forecast cycle also captures new snapshots.
    """
    today_start = datetime.now(timezone.utc).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    # Count predictions with polymarket provenance updated today
    stmt = (
        select(func.count())
        .where(
            Prediction.provenance.in_(["polymarket_driven", "polymarket_tracked"]),
            Prediction.created_at >= today_start,
        )
        .select_from(Prediction)
    )
    result = await session.execute(stmt)
    total = result.scalar() or 0
    # Subtract new forecasts (those are new, not re-forecasts)
    new = await count_today_new_forecasts(session)
    return max(0, total - new)


def _get_event_volume(event: dict) -> float:
    """Extract total volume from a Polymarket event dict.

    Volume can be on the event or summed from its markets.
    """
    # Direct volume field
    vol = event.get("volume")
    if vol is not None:
        try:
            return float(vol)
        except (ValueError, TypeError):
            pass

    # Sum from markets
    total = 0.0
    for market in event.get("markets", []):
        mv = market.get("volume")
        if mv is not None:
            try:
                total += float(mv)
            except (ValueError, TypeError):
                pass
    return total


# ---------------------------------------------------------------------------
# PolymarketAutoForecaster
# ---------------------------------------------------------------------------


class PolymarketAutoForecaster:
    """Generate Geopol forecasts for unmatched high-volume Polymarket questions.

    Runs inside the existing _polymarket_loop after the matching cycle.
    Filters unmatched events by volume threshold, extracts pipeline parameters
    (country, horizon, CAMEO), runs EnsemblePredictor.predict(), persists via
    ForecastService, and creates comparison rows.

    Also re-forecasts active comparisons daily (overwriting existing Prediction
    rows, with historical values preserved in polymarket_snapshots).
    """

    def __init__(
        self,
        async_session_factory: Callable[..., Any],
        gemini_client: GeminiClient,
        settings: Settings,
    ) -> None:
        self._session_factory = async_session_factory
        self._gemini = gemini_client
        self._settings = settings

    async def run(
        self,
        geo_events: list[dict],
        tracked_ids: set[str],
    ) -> dict[str, int]:
        """Filter unmatched events by volume, extract params, generate forecasts.

        Args:
            geo_events: Geopolitical events from PolymarketClient.
            tracked_ids: Event IDs already tracked by the matching cycle.
                If empty, the method queries tracked IDs internally.

        Returns:
            Dict with keys: candidates, generated, skipped_budget, skipped_cap.
        """
        result = {
            "candidates": 0,
            "generated": 0,
            "skipped_budget": 0,
            "skipped_cap": 0,
        }

        async with self._session_factory() as session:
            # Query tracked event IDs if not provided
            if not tracked_ids:
                tracked_stmt = select(PolymarketComparison.polymarket_event_id)
                tracked_result = await session.execute(tracked_stmt)
                tracked_ids = {row[0] for row in tracked_result.fetchall()}

            # Query existing polymarket_event_ids on predictions for dedup
            existing_stmt = select(Prediction.polymarket_event_id).where(
                Prediction.polymarket_event_id.isnot(None)
            )
            existing_result = await session.execute(existing_stmt)
            existing_event_ids: set[str] = {
                row[0] for row in existing_result.fetchall()
            }

            # How many new forecasts can we still generate today?
            today_count = await count_today_new_forecasts(session)
            remaining_cap = max(
                0, self._settings.polymarket_daily_new_forecast_cap - today_count
            )

            if remaining_cap == 0:
                logger.info(
                    "Daily new forecast cap reached (%d/%d)",
                    today_count,
                    self._settings.polymarket_daily_new_forecast_cap,
                )

            # Filter candidates
            candidates: list[dict] = []
            for event in geo_events:
                event_id = str(event.get("id", ""))
                if not event_id:
                    continue

                # Already tracked by matcher
                if event_id in tracked_ids:
                    continue

                # Already has a prediction (dedup)
                if event_id in existing_event_ids:
                    continue

                # Volume threshold
                volume = _get_event_volume(event)
                if volume < self._settings.polymarket_volume_threshold:
                    continue

                # Valid horizon
                end_date = event.get("endDate")
                horizon = compute_horizon_days(end_date)
                if horizon is None:
                    continue

                candidates.append(event)

            # Sort by volume DESC, take up to remaining cap
            candidates.sort(key=_get_event_volume, reverse=True)
            result["candidates"] = len(candidates)

            if not candidates:
                return result

            to_process = candidates[:remaining_cap]
            result["skipped_cap"] = len(candidates) - len(to_process)

            # Get redis for budget checks
            from src.api.deps import get_redis
            from src.api.middleware.rate_limit import (
                gemini_budget_remaining,
                increment_gemini_usage,
            )

            redis_client = await get_redis()

            for event in to_process:
                event_id = str(event.get("id", ""))
                title = event.get("title", "")
                description = event.get("description", "")
                tags = event.get("tags", [])
                end_date = event.get("endDate")

                # Check budget before burning API calls
                remaining_budget = await gemini_budget_remaining(redis_client)
                if remaining_budget <= 0:
                    result["skipped_budget"] += 1
                    logger.warning(
                        "Gemini budget exhausted, skipping auto-forecast for: %s",
                        title[:60],
                    )
                    continue

                try:
                    # Extract country (heuristic first, LLM fallback)
                    country_iso = extract_country_heuristic(title, tags if isinstance(tags, list) else [])
                    if country_iso is None:
                        country_iso = await extract_country_llm(
                            title, description, self._gemini
                        )
                        await increment_gemini_usage(redis_client)

                    # Extract CAMEO category (always LLM)
                    cameo_code = await extract_cameo_category(title, self._gemini)
                    await increment_gemini_usage(redis_client)

                    # Compute horizon
                    horizon = compute_horizon_days(end_date)
                    if horizon is None:
                        horizon = 30  # Fallback -- should not happen given filter above

                    # Check budget again before the expensive predict() call
                    remaining_budget = await gemini_budget_remaining(redis_client)
                    if remaining_budget <= 0:
                        result["skipped_budget"] += 1
                        logger.warning(
                            "Gemini budget exhausted before predict, skipping: %s",
                            title[:60],
                        )
                        continue

                    # Run EnsemblePredictor (fresh instance per prediction)
                    from src.forecasting.ensemble_predictor import EnsemblePredictor

                    predictor = EnsemblePredictor()
                    ensemble_pred, forecast_output = await asyncio.to_thread(
                        predictor.predict,
                        question=title,
                        cameo_root_code=cameo_code,
                    )

                    # Persist via ForecastService
                    from src.api.services.forecast_service import ForecastService

                    service = ForecastService(session)
                    prediction = await service.persist_forecast(
                        forecast_output=forecast_output,
                        ensemble_prediction=ensemble_pred,
                        country_iso=country_iso.upper() if country_iso else None,
                        horizon_days=horizon,
                    )

                    # Set polymarket-specific columns AFTER persist
                    prediction.provenance = "polymarket_driven"
                    prediction.polymarket_event_id = event_id
                    prediction.cameo_root_code = cameo_code
                    await session.flush()

                    # Create PolymarketComparison row
                    # Extract initial price from event markets
                    from src.polymarket.comparison import _parse_outcome_price

                    markets = event.get("markets", [])
                    initial_price = _parse_outcome_price(markets) if markets else None

                    comparison = PolymarketComparison(
                        polymarket_event_id=event_id,
                        polymarket_slug=event.get("slug", event_id),
                        polymarket_title=title,
                        geopol_prediction_id=prediction.id,
                        match_confidence=1.0,  # Auto-generated, not matched
                        polymarket_price=initial_price,
                        geopol_probability=prediction.probability,
                        status="active",
                    )
                    session.add(comparison)

                    await increment_gemini_usage(redis_client)
                    result["generated"] += 1

                    logger.info(
                        "Auto-forecasted Polymarket question: %s (event=%s, country=%s, p=%.3f)",
                        title[:60],
                        event_id,
                        country_iso,
                        prediction.probability,
                    )

                except Exception as exc:
                    logger.error(
                        "Auto-forecast failed for event %s: %s",
                        event_id,
                        exc,
                        exc_info=True,
                    )

            await session.commit()

        return result

    async def reforecast_active(self) -> dict[str, int]:
        """Re-forecast active comparisons with Polymarket provenance.

        Overwrites existing Prediction rows with fresh EnsemblePredictor
        output. Historical values are preserved in polymarket_snapshots.

        Returns:
            Dict with keys: active_comparisons, reforecasted, skipped_budget.
        """
        result = {
            "active_comparisons": 0,
            "reforecasted": 0,
            "skipped_budget": 0,
        }

        async with self._session_factory() as session:
            # Check if any re-forecasts already done today
            today_reforecasts = await count_today_reforecasts(session)
            remaining_cap = max(
                0, self._settings.polymarket_daily_reforecast_cap - today_reforecasts
            )

            if remaining_cap == 0:
                logger.info(
                    "Daily reforecast cap reached (%d/%d)",
                    today_reforecasts,
                    self._settings.polymarket_daily_reforecast_cap,
                )
                return result

            # Query active comparisons linked to Polymarket-driven/tracked predictions
            stmt = (
                select(PolymarketComparison)
                .where(PolymarketComparison.status == "active")
                .join(
                    Prediction,
                    Prediction.id == PolymarketComparison.geopol_prediction_id,
                )
                .where(
                    Prediction.provenance.in_(
                        ["polymarket_driven", "polymarket_tracked"]
                    )
                )
                .limit(remaining_cap)
            )
            comp_result = await session.execute(stmt)
            comparisons = list(comp_result.scalars().all())
            result["active_comparisons"] = len(comparisons)

            if not comparisons:
                return result

            # Get redis for budget checks
            from src.api.deps import get_redis
            from src.api.middleware.rate_limit import (
                gemini_budget_remaining,
                increment_gemini_usage,
            )

            redis_client = await get_redis()

            for comp in comparisons:
                remaining_budget = await gemini_budget_remaining(redis_client)
                if remaining_budget <= 0:
                    result["skipped_budget"] += 1
                    continue

                try:
                    # Load the existing prediction
                    pred_result = await session.execute(
                        select(Prediction).where(
                            Prediction.id == comp.geopol_prediction_id
                        )
                    )
                    existing = pred_result.scalar_one_or_none()
                    if existing is None:
                        logger.warning(
                            "Prediction %s not found for comparison %d, skipping",
                            comp.geopol_prediction_id,
                            comp.id,
                        )
                        continue

                    # Fresh EnsemblePredictor per prediction
                    from src.forecasting.ensemble_predictor import EnsemblePredictor

                    predictor = EnsemblePredictor()
                    ensemble_pred, forecast_output = await asyncio.to_thread(
                        predictor.predict,
                        question=existing.question,
                    )

                    # Overwrite the existing Prediction row
                    from src.api.services.forecast_service import ForecastService

                    existing.probability = forecast_output.probability
                    existing.confidence = forecast_output.confidence
                    existing.prediction = forecast_output.prediction
                    existing.reasoning_summary = forecast_output.reasoning_summary
                    existing.scenarios_json = ForecastService._scenarios_to_json(
                        forecast_output
                    )
                    existing.ensemble_info_json = {
                        "llm_probability": ensemble_pred.llm_prediction.probability,
                        "tkg_probability": (
                            ensemble_pred.tkg_prediction.probability
                            if ensemble_pred.tkg_prediction.available
                            else None
                        ),
                        "weights": {
                            "llm": ensemble_pred.weights_used[0],
                            "tkg": ensemble_pred.weights_used[1],
                        },
                        "temperature_applied": ensemble_pred.temperature,
                    }
                    existing.calibration_json = {
                        "category": ensemble_pred.category or "conflict",
                        "temperature": ensemble_pred.temperature,
                        "historical_accuracy": 0.0,
                        "brier_score": None,
                        "sample_size": 0,
                    }
                    existing.created_at = datetime.now(timezone.utc)
                    await session.flush()

                    # Update comparison's geopol_probability
                    comp.geopol_probability = forecast_output.probability

                    await increment_gemini_usage(redis_client)
                    result["reforecasted"] += 1

                    logger.info(
                        "Re-forecasted comparison %d: %s (new p=%.3f)",
                        comp.id,
                        existing.question[:60],
                        forecast_output.probability,
                    )

                except Exception as exc:
                    logger.error(
                        "Re-forecast failed for comparison %d: %s",
                        comp.id,
                        exc,
                        exc_info=True,
                    )

            await session.commit()

        return result
