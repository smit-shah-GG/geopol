"""
Pydantic V2 DTO schemas for the geopolitical forecasting API.

All public DTOs are re-exported here for convenient import:

    from src.api.schemas import ForecastResponse, ScenarioDTO, CountryRiskSummary

These schemas define the API contract between the Python backend and the
TypeScript frontend. Both sides develop against these types — changes here
are breaking changes.
"""

from src.api.schemas.common import (
    PaginatedResponse,
    ProblemDetail,
    decode_cursor,
    encode_cursor,
)
from src.api.schemas.country import CountryRiskSummary
from src.api.schemas.forecast import (
    CalibrationDTO,
    EnsembleInfoDTO,
    EvidenceDTO,
    ForecastResponse,
    ScenarioDTO,
)
from src.api.schemas.health import (
    SUBSYSTEM_NAMES,
    HealthResponse,
    SubsystemStatus,
)

__all__ = [
    # Forecast DTOs
    "ForecastResponse",
    "ScenarioDTO",
    "EvidenceDTO",
    "CalibrationDTO",
    "EnsembleInfoDTO",
    # Country
    "CountryRiskSummary",
    # Health
    "HealthResponse",
    "SubsystemStatus",
    "SUBSYSTEM_NAMES",
    # Common
    "ProblemDetail",
    "PaginatedResponse",
    "encode_cursor",
    "decode_cursor",
]
