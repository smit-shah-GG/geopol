"""
Forecast-related Pydantic V2 DTOs.

These DTOs define the API contract for forecast data. They are the bridge between
the Python forecasting engine and the TypeScript frontend — both sides code to
these schemas. Changes here are breaking changes to the API contract.

Derived from WORLDMONITOR_INTEGRATION.md lines 328-382.
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class EvidenceDTO(BaseModel):
    """A single piece of evidence supporting a scenario.

    Links back to GDELT events, TKG structural patterns, or RAG-retrieved
    context to maintain the explainability chain.
    """

    model_config = ConfigDict(from_attributes=True)

    source: str = Field(
        ...,
        description='Evidence source type: "GDELT", "TKG pattern", "RAG match"',
    )
    description: str = Field(..., description="Human-readable evidence summary")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence in this evidence"
    )
    timestamp: Optional[datetime] = Field(
        None, description="When the evidence event occurred"
    )
    gdelt_event_id: Optional[str] = Field(
        None, description="Cross-reference to GDELT GlobalEventID"
    )


class EnsembleInfoDTO(BaseModel):
    """Breakdown of how the ensemble combined LLM and TKG predictions.

    Provides transparency into the weighting decision — which model
    contributed what, and what temperature scaling was applied.
    """

    model_config = ConfigDict(from_attributes=True)

    llm_probability: float = Field(
        ..., ge=0.0, le=1.0, description="Raw LLM probability estimate"
    )
    tkg_probability: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="TKG probability estimate (None if TKG disabled)",
    )
    weights: dict[str, float] = Field(
        ...,
        description='Model weight breakdown, e.g. {"llm": 0.6, "tkg": 0.4}',
    )
    temperature_applied: float = Field(
        ..., gt=0.0, description="Temperature scaling factor applied to logits"
    )


class CalibrationDTO(BaseModel):
    """Per-category calibration metadata.

    Reports how well-calibrated the model is for this event category,
    enabling the frontend to display confidence intervals and historical
    accuracy alongside the prediction.
    """

    model_config = ConfigDict(from_attributes=True)

    category: str = Field(
        ...,
        description='Event category: "conflict", "diplomatic", "economic", etc.',
    )
    temperature: float = Field(
        ..., description="Temperature scaling applied for this category"
    )
    historical_accuracy: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Historical accuracy at this confidence level",
    )
    brier_score: Optional[float] = Field(
        None, ge=0.0, le=2.0, description="Brier score if validation data available"
    )
    sample_size: int = Field(
        ..., ge=0, description="Number of past predictions calibrated against"
    )


class ScenarioDTO(BaseModel):
    """A single scenario branch in the forecast scenario tree.

    Scenarios are recursive: each scenario can contain child_scenarios
    representing further branching of that trajectory. This enables the
    frontend's interactive scenario explorer to render drill-down trees.

    IMPORTANT: model_rebuild() is called after class definition to resolve
    the self-referential forward reference in child_scenarios.
    """

    model_config = ConfigDict(from_attributes=True)

    scenario_id: str = Field(..., description="Unique scenario identifier")
    description: str = Field(
        ..., description="Natural language description of this scenario trajectory"
    )
    probability: float = Field(
        ..., ge=0.0, le=1.0, description="Probability of this scenario occurring"
    )
    answers_affirmative: bool = Field(
        ...,
        description='Whether this scenario constitutes a "yes" answer to the forecast question',
    )
    entities: list[str] = Field(
        ..., description="Actor names involved in this scenario"
    )
    timeline: list[str] = Field(
        ..., description="Sequence of expected events in chronological order"
    )
    evidence_sources: list[EvidenceDTO] = Field(
        default_factory=list, description="Evidence supporting this scenario"
    )
    child_scenarios: list["ScenarioDTO"] = Field(
        default_factory=list, description="Sub-scenarios branching from this one"
    )


# Resolve the self-referential forward reference. Without this call,
# Pydantic V2 raises PydanticUserError during schema generation because
# "ScenarioDTO" in the child_scenarios annotation is an unresolved string.
ScenarioDTO.model_rebuild()


class ForecastResponse(BaseModel):
    """Complete forecast response — the primary API contract.

    This is what GET /api/v1/forecasts/{id} returns. Contains the full
    prediction with nested scenarios, evidence chains, calibration data,
    and ensemble breakdown. The frontend renders this directly.

    Maps from the internal ForecastOutput + EnsemblePrediction objects
    produced by EnsemblePredictor.predict().
    """

    model_config = ConfigDict(from_attributes=True)

    forecast_id: str = Field(..., description="Unique forecast identifier (UUID)")
    question: str = Field(..., description="The forecasting question being answered")
    prediction: str = Field(
        ..., description="Natural language prediction summary"
    )
    probability: float = Field(
        ..., ge=0.0, le=1.0, description="Calibrated P(yes) — the headline number"
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Model confidence in this prediction (distinct from probability)",
    )
    horizon_days: int = Field(
        ..., gt=0, description="Forecast horizon in days from creation"
    )
    scenarios: list[ScenarioDTO] = Field(
        ..., description="Top-level scenario branches"
    )
    reasoning_summary: str = Field(
        ..., description="Human-readable reasoning chain summary"
    )
    evidence_count: int = Field(
        ..., ge=0, description="Total evidence sources across all scenarios"
    )
    ensemble_info: EnsembleInfoDTO = Field(
        ..., description="LLM/TKG ensemble weight breakdown"
    )
    calibration: CalibrationDTO = Field(
        ..., description="Calibration metadata for this forecast's category"
    )
    created_at: datetime = Field(
        ..., description="When this forecast was generated"
    )
    expires_at: datetime = Field(
        ..., description="When this forecast becomes stale and should be regenerated"
    )
