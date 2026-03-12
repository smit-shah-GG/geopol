"""
Async background worker for processing confirmed forecast requests.

Architecture: asyncio.create_task + Semaphore(3) for bounded parallelism.
No external dependencies (no Celery, no Redis queue). The worker runs in
the same event loop as FastAPI.

Lifecycle per request:
    1. Acquire semaphore slot (max 3 concurrent)
    2. Load ForecastRequest with SELECT FOR UPDATE SKIP LOCKED
    3. Check Gemini budget
    4. For each country in country_iso_list:
       a. Run EnsemblePredictor.predict() via asyncio.to_thread
       b. Persist result via ForecastService
    5. Update status to 'complete'

On failure: retry up to 3 times with exponential backoff (30s, 2min, 10min).
Budget-exhausted requests are marked failed immediately (no retry).
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models import ForecastRequest

logger = logging.getLogger(__name__)

_worker_semaphore = asyncio.Semaphore(3)
_active_tasks: set[asyncio.Task] = set()  # type: ignore[type-arg]
_predictor_cache: tuple | None = None  # (orchestrator, tkg_pred) -- heavy init once

MAX_RETRIES = 3
RETRY_DELAYS = [30, 120, 600]  # seconds: 30s, 2min, 10min


def _build_predictor():
    """Build a properly-wired EnsemblePredictor, caching heavy components.

    Caches ReasoningOrchestrator and TKGPredictor across calls because
    they load sentence-transformers, ChromaDB, and TiRGN checkpoint (~4s).
    Returns a fresh EnsemblePredictor each time (mutable _forecast_output
    state prevents reuse).
    """
    global _predictor_cache

    from src.forecasting.ensemble_predictor import EnsemblePredictor

    if _predictor_cache is not None:
        orch, tkg_pred = _predictor_cache
        return EnsemblePredictor(llm_orchestrator=orch, tkg_predictor=tkg_pred)

    tkg_pred = None
    try:
        from src.forecasting.tkg_predictor import TKGPredictor

        tkg_pred = TKGPredictor()  # auto_load=True loads tirgn_best.npz
        if not tkg_pred.trained:
            logger.warning("TKG predictor has no trained model")
            tkg_pred = None
    except Exception as exc:
        logger.warning("TKG predictor init failed: %s", exc)

    orch = None
    try:
        from src.forecasting.graph_validator import GraphValidator
        from src.forecasting.reasoning_orchestrator import ReasoningOrchestrator

        graph_validator = GraphValidator(tkg_predictor=tkg_pred) if tkg_pred else None
        orch = ReasoningOrchestrator(graph_validator=graph_validator)
    except Exception as exc:
        logger.warning("ReasoningOrchestrator init failed: %s", exc)

    _predictor_cache = (orch, tkg_pred)
    return EnsemblePredictor(llm_orchestrator=orch, tkg_predictor=tkg_pred)


class BudgetExhaustedError(Exception):
    """Raised when Gemini API daily budget is exhausted."""

    pass


def schedule_processing(request_id: str) -> None:
    """Schedule a confirmed forecast request for background processing.

    Creates an asyncio task that acquires a semaphore slot before
    executing the forecast. The task is tracked in _active_tasks
    to prevent garbage collection.

    Args:
        request_id: UUID of the ForecastRequest to process.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        logger.error(
            "No running event loop -- cannot schedule processing for %s",
            request_id,
        )
        return

    task = loop.create_task(
        _process_request(request_id),
        name=f"forecast-worker-{request_id[:8]}",
    )
    _active_tasks.add(task)

    def _on_done(t: asyncio.Task) -> None:  # type: ignore[type-arg]
        _active_tasks.discard(t)
        if t.exception() is not None:
            logger.error(
                "Worker task for %s raised unhandled exception: %s",
                request_id,
                t.exception(),
            )

    task.add_done_callback(_on_done)

    logger.info(
        "Scheduled processing for request %s (active tasks: %d)",
        request_id,
        len(_active_tasks),
    )


async def _process_request(request_id: str) -> None:
    """Process a single forecast request under the concurrency semaphore.

    Acquires a semaphore slot (max 3 concurrent), gets its own DB
    session (the HTTP request's session is closed by now), and
    delegates to _execute_forecast.
    """
    async with _worker_semaphore:
        logger.info(
            "Worker acquired semaphore for request %s", request_id
        )

        from src.db.postgres import get_async_session

        async for session in get_async_session():
            try:
                await _execute_forecast(session, request_id)
            except BudgetExhaustedError as exc:
                logger.warning(
                    "Budget exhausted for request %s: %s",
                    request_id,
                    exc,
                )
                await _mark_failed(
                    session,
                    request_id,
                    f"Gemini daily budget exhausted: {exc}",
                )
            except Exception as exc:
                logger.error(
                    "Worker failed for request %s: %s",
                    request_id,
                    exc,
                    exc_info=True,
                )
                await _handle_failure(session, request_id, str(exc))


async def _execute_forecast(
    session: AsyncSession, request_id: str
) -> None:
    """Execute the forecast pipeline for a confirmed request.

    Uses SELECT FOR UPDATE SKIP LOCKED to prevent double-pickup
    by concurrent workers.
    """
    # Atomically claim the request
    stmt = (
        select(ForecastRequest)
        .where(ForecastRequest.id == request_id)
        .with_for_update(skip_locked=True)
    )
    result = await session.execute(stmt)
    request = result.scalar_one_or_none()

    if request is None or request.status not in ("confirmed", "processing"):
        logger.info(
            "Request %s already claimed or not ready (status=%s), skipping",
            request_id,
            request.status if request else "NOT_FOUND",
        )
        return

    # Transition to processing
    request.status = "processing"
    await session.flush()
    logger.info(
        "Processing request %s: question=%s..., countries=%s",
        request_id,
        request.question[:60],
        request.country_iso_list,
    )

    # Check Gemini budget before burning API calls
    from src.api.deps import get_redis
    from src.api.middleware.rate_limit import (
        gemini_budget_remaining,
        increment_gemini_usage,
    )

    redis_client = await get_redis()
    remaining = await gemini_budget_remaining(redis_client)
    if remaining <= 0:
        raise BudgetExhaustedError(
            f"Gemini daily budget exhausted ({remaining} remaining)"
        )

    # Process each country
    prediction_ids: list[str] = []

    for country_iso in request.country_iso_list:
        logger.info(
            "Running EnsemblePredictor for %s (request %s)",
            country_iso,
            request_id,
        )

        predictor = _build_predictor()

        # EnsemblePredictor.predict() is synchronous -- wrap in thread
        ensemble_pred, forecast_output = await asyncio.to_thread(
            predictor.predict,
            question=request.question,
        )

        # Persist via ForecastService
        from src.api.services.forecast_service import ForecastService

        service = ForecastService(session)
        prediction = await service.persist_forecast(
            forecast_output=forecast_output,
            ensemble_prediction=ensemble_pred,
            country_iso=country_iso.upper() if country_iso != "XX" else None,
            horizon_days=request.horizon_days,
        )

        prediction_ids.append(prediction.id)

        # Increment Gemini usage counter per country
        await increment_gemini_usage(redis_client)

        logger.info(
            "Forecast persisted for %s: prediction_id=%s",
            country_iso,
            prediction.id,
        )

    # Mark complete
    request.prediction_ids = prediction_ids
    request.status = "complete"
    request.completed_at = datetime.now(timezone.utc)
    request.error_message = None
    await session.flush()

    logger.info(
        "Request %s complete: %d predictions generated",
        request_id,
        len(prediction_ids),
    )


async def _handle_failure(
    session: AsyncSession, request_id: str, error_msg: str
) -> None:
    """Handle a failed forecast attempt with retry logic.

    Increments retry_count. If under MAX_RETRIES, schedules a delayed
    retry. If at or over MAX_RETRIES, marks the request as permanently
    failed.
    """
    result = await session.execute(
        select(ForecastRequest).where(ForecastRequest.id == request_id)
    )
    request = result.scalar_one_or_none()
    if request is None:
        logger.error("Request %s disappeared during failure handling", request_id)
        return

    request.retry_count = (request.retry_count or 0) + 1

    if request.retry_count < MAX_RETRIES:
        # Schedule retry with delay
        delay = RETRY_DELAYS[request.retry_count - 1]
        request.status = "confirmed"  # Worker will re-pick
        request.error_message = (
            f"Retry {request.retry_count}/{MAX_RETRIES} after error: {error_msg}"
        )
        await session.flush()

        logger.info(
            "Scheduling retry %d/%d for request %s in %ds",
            request.retry_count,
            MAX_RETRIES,
            request_id,
            delay,
        )

        # Schedule delayed retry via asyncio
        try:
            loop = asyncio.get_running_loop()
            loop.call_later(
                delay,
                lambda rid=request_id: asyncio.ensure_future(
                    _retry_wrapper(rid)
                ),
            )
        except RuntimeError:
            logger.error(
                "No event loop for retry scheduling (request %s)", request_id
            )
    else:
        await _mark_failed(session, request_id, error_msg)


async def _mark_failed(
    session: AsyncSession, request_id: str, error_msg: str
) -> None:
    """Mark a request as permanently failed."""
    result = await session.execute(
        select(ForecastRequest).where(ForecastRequest.id == request_id)
    )
    request = result.scalar_one_or_none()
    if request is None:
        return

    request.status = "failed"
    request.error_message = f"Failed after {request.retry_count} retries: {error_msg}"
    request.completed_at = datetime.now(timezone.utc)
    await session.flush()

    logger.error(
        "Request %s permanently failed: %s",
        request_id,
        error_msg,
    )


async def _retry_wrapper(request_id: str) -> None:
    """Thin wrapper for call_later -> schedule_processing bridge.

    call_later needs a plain callable; this bridges to the
    schedule_processing function which creates the async task.
    """
    logger.info("Retry triggered for request %s", request_id)
    schedule_processing(request_id)
