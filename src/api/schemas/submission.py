"""
Pydantic V2 DTOs for the question submission flow.

Defines the request/response contracts for:
    POST /api/v1/forecasts/submit        -- Submit a natural language question
    POST /api/v1/forecasts/submit/{id}/confirm -- Confirm parsed question
    GET  /api/v1/forecasts/requests      -- List user's submitted requests

These DTOs bridge the frontend submission UI with the ForecastRequest ORM
model and the LLM question parser service.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class SubmitQuestionRequest(BaseModel):
    """POST body for question submission."""

    question: str = Field(
        ...,
        min_length=10,
        max_length=1000,
        description="Natural language forecast question",
    )


class ParsedQuestionResponse(BaseModel):
    """Response from submit endpoint showing LLM-parsed structured form.

    The user reviews this parsed interpretation before confirming
    that the system understood their question correctly.
    """

    request_id: str = Field(..., description="UUID of the forecast request")
    question: str = Field(..., description="Original question text")
    country_iso_list: list[str] = Field(
        ..., description="Parsed ISO 3166-1 alpha-2 country codes"
    )
    horizon_days: int = Field(..., description="Inferred forecast horizon in days")
    category: str = Field(..., description="Inferred CAMEO category")
    status: str = Field(default="pending", description="Current request status")
    parsed_at: datetime = Field(..., description="When the question was parsed")


class ForecastRequestStatus(BaseModel):
    """Status of a submitted forecast request.

    Lifecycle: pending -> confirmed -> processing -> complete | failed.
    """

    request_id: str
    question: str
    country_iso_list: list[str]
    horizon_days: int
    category: str
    status: str  # pending | confirmed | processing | complete | failed
    submitted_at: datetime
    completed_at: Optional[datetime] = None
    prediction_ids: list[str] = Field(default_factory=list)
    error_message: Optional[str] = None


class ConfirmSubmissionResponse(BaseModel):
    """Response from confirm endpoint."""

    request_id: str
    status: str  # "confirmed" -- worker will pick it up
    message: str = "Request confirmed and queued for processing"
