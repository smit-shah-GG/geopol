"""
Question submission endpoints for the two-phase submit/confirm flow.

Implements BAPI-03: users submit natural language forecast questions,
review the LLM-parsed structured form, confirm, and track processing
status. All endpoints require API key authentication. Queue visibility
is scoped per API key (users see only their own submissions).

Endpoints:
    POST /forecasts/submit                -- Parse question, create ForecastRequest
    POST /forecasts/submit/{id}/confirm   -- Confirm and queue for background processing
    GET  /forecasts/requests              -- List user's submitted requests

Mounted under the /forecasts prefix alongside the existing forecasts
router (no route conflicts -- different path segments).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.deps import get_db
from src.api.middleware.auth import verify_api_key
from src.api.schemas.submission import (
    ConfirmSubmissionResponse,
    ForecastRequestStatus,
    ParsedQuestionResponse,
    SubmitQuestionRequest,
)
from src.api.services.question_parser import parse_question
from src.db.models import ForecastRequest

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/submit",
    response_model=ParsedQuestionResponse,
    status_code=201,
    summary="Submit forecast question",
    description=(
        "Submit a natural language forecast question. The question is parsed "
        "by Gemini LLM into structured form (country codes, horizon, category). "
        "Returns the parsed interpretation for user review before confirmation."
    ),
)
async def submit_question(
    body: SubmitQuestionRequest,
    client: str = Depends(verify_api_key),
    db: AsyncSession = Depends(get_db),
) -> ParsedQuestionResponse:
    """Parse a natural language question and create a pending ForecastRequest."""
    logger.info(
        "Question submitted by %s: %s...",
        client,
        body.question[:60],
    )

    # Parse question via Gemini LLM
    parsed = await parse_question(body.question)

    now = datetime.now(timezone.utc)

    # Create ForecastRequest row
    request = ForecastRequest(
        question=body.question,
        country_iso_list=parsed["country_iso_list"],
        horizon_days=parsed["horizon_days"],
        category=parsed["category"],
        status="pending",
        submitted_by=client,
        submitted_at=now,
        parsed_at=now,
    )
    db.add(request)
    await db.flush()  # Assign id via default

    logger.info(
        "Created ForecastRequest %s: countries=%s, horizon=%d, category=%s",
        request.id,
        parsed["country_iso_list"],
        parsed["horizon_days"],
        parsed["category"],
    )

    return ParsedQuestionResponse(
        request_id=request.id,
        question=body.question,
        country_iso_list=parsed["country_iso_list"],
        horizon_days=parsed["horizon_days"],
        category=parsed["category"],
        status="pending",
        parsed_at=now,
    )


@router.post(
    "/submit/{request_id}/confirm",
    response_model=ConfirmSubmissionResponse,
    summary="Confirm submitted question",
    description=(
        "Confirm a previously submitted question after reviewing the parsed "
        "interpretation. Transitions the request to 'confirmed' and queues "
        "it for background processing by the forecast worker."
    ),
)
async def confirm_submission(
    request_id: str,
    client: str = Depends(verify_api_key),
    db: AsyncSession = Depends(get_db),
) -> ConfirmSubmissionResponse:
    """Confirm a pending ForecastRequest and trigger background processing."""
    result = await db.execute(
        select(ForecastRequest).where(ForecastRequest.id == request_id)
    )
    request = result.scalar_one_or_none()

    if request is None:
        raise HTTPException(
            status_code=404,
            detail=f"Forecast request '{request_id}' not found",
        )

    # Security: users can only confirm their own requests
    if request.submitted_by != client:
        raise HTTPException(
            status_code=403,
            detail="Not authorized to confirm this request",
        )

    # State guard: only pending requests can be confirmed
    if request.status != "pending":
        raise HTTPException(
            status_code=409,
            detail=f"Request is '{request.status}', only 'pending' requests can be confirmed",
        )

    request.status = "confirmed"
    await db.flush()

    logger.info("ForecastRequest %s confirmed by %s", request_id, client)

    # Trigger background processing via worker
    from src.api.services.submission_worker import schedule_processing

    schedule_processing(request_id)

    return ConfirmSubmissionResponse(
        request_id=request_id,
        status="confirmed",
    )


@router.get(
    "/requests",
    response_model=list[ForecastRequestStatus],
    summary="List submitted requests",
    description=(
        "List the authenticated user's submitted forecast requests. "
        "Optionally filter by status. Results are ordered by submission "
        "time descending (newest first). Queue visibility is scoped per API key."
    ),
)
async def list_requests(
    status_filter: Optional[str] = Query(
        default=None, description="Filter by status (pending|confirmed|processing|complete|failed)"
    ),
    limit: int = Query(default=20, ge=1, le=100),
    client: str = Depends(verify_api_key),
    db: AsyncSession = Depends(get_db),
) -> list[ForecastRequestStatus]:
    """Return the user's submitted forecast requests filtered by API key."""
    stmt = (
        select(ForecastRequest)
        .where(ForecastRequest.submitted_by == client)
        .order_by(ForecastRequest.submitted_at.desc())
    )

    if status_filter is not None:
        stmt = stmt.where(ForecastRequest.status == status_filter)

    stmt = stmt.limit(limit)

    result = await db.execute(stmt)
    rows = result.scalars().all()

    return [
        ForecastRequestStatus(
            request_id=row.id,
            question=row.question,
            country_iso_list=row.country_iso_list,
            horizon_days=row.horizon_days,
            category=row.category,
            status=row.status,
            submitted_at=row.submitted_at,
            completed_at=row.completed_at,
            prediction_ids=row.prediction_ids or [],
            error_message=row.error_message,
        )
        for row in rows
    ]
