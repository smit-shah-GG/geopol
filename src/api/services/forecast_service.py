"""
ForecastService -- bridges EnsemblePredictor with PostgreSQL persistence.

Responsibility:
    1. Accept ForecastOutput + EnsemblePrediction from predict()
    2. Map them to the Prediction ORM model and persist
    3. Reconstruct ForecastResponse DTOs from persisted rows

This service does NOT call EnsemblePredictor.predict() itself -- callers
(routes, daily pipeline) invoke predict() then pass the results here.
Keeping the predictor pure and the persistence separate follows the existing
PredictionStore pattern from calibration.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.schemas.common import PaginatedResponse, decode_cursor, encode_cursor
from src.api.schemas.forecast import (
    CalibrationDTO,
    EnsembleInfoDTO,
    EvidenceDTO,
    ForecastResponse,
    ScenarioDTO,
)
from src.db.models import Prediction
from src.forecasting.ensemble_predictor import EnsemblePrediction
from src.forecasting.models import ForecastOutput, Scenario

logger = logging.getLogger(__name__)


class ForecastService:
    """Bridges EnsemblePredictor with PostgreSQL persistence.

    All methods are async and operate on a single session. The caller
    (FastAPI dependency or script) manages session lifecycle.
    """

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def persist_forecast(
        self,
        forecast_output: ForecastOutput,
        ensemble_prediction: EnsemblePrediction,
        country_iso: Optional[str] = None,
        horizon_days: int = 30,
    ) -> Prediction:
        """Map ForecastOutput + EnsemblePrediction to Prediction ORM and persist.

        Args:
            forecast_output: Full forecast from the prediction pipeline.
            ensemble_prediction: Ensemble metadata (weights, component probs).
            country_iso: ISO 3166-1 alpha-2/3 code, nullable for non-country queries.
            horizon_days: Forecast horizon for expiry calculation.

        Returns:
            The persisted Prediction ORM instance (with generated id).
        """
        now = datetime.now(timezone.utc)
        forecast_id = str(uuid.uuid4())

        # Build scenarios JSON from ScenarioTree
        scenarios_json = self._scenarios_to_json(forecast_output)

        # Build ensemble info JSON from EnsemblePrediction
        ensemble_info_json = {
            "llm_probability": ensemble_prediction.llm_prediction.probability,
            "tkg_probability": (
                ensemble_prediction.tkg_prediction.probability
                if ensemble_prediction.tkg_prediction.available
                else None
            ),
            "weights": {
                "llm": ensemble_prediction.weights_used[0],
                "tkg": ensemble_prediction.weights_used[1],
            },
            "temperature_applied": ensemble_prediction.temperature,
        }

        # Build calibration JSON
        calibration_json = {
            "category": ensemble_prediction.category or "conflict",
            "temperature": ensemble_prediction.temperature,
            "historical_accuracy": 0.0,  # No outcome data yet -- Phase 13
            "brier_score": None,
            "sample_size": 0,
        }

        # Extract entity names from scenario tree
        entities = self._extract_entities(forecast_output)

        prediction = Prediction(
            id=forecast_id,
            question=forecast_output.question,
            prediction=forecast_output.prediction,
            probability=forecast_output.probability,
            confidence=forecast_output.confidence,
            horizon_days=horizon_days,
            category=ensemble_prediction.category or "conflict",
            reasoning_summary=forecast_output.reasoning_summary,
            evidence_count=len(forecast_output.evidence_sources),
            scenarios_json=scenarios_json,
            ensemble_info_json=ensemble_info_json,
            calibration_json=calibration_json,
            entities=entities,
            country_iso=country_iso,
            created_at=now,
            expires_at=now + timedelta(days=horizon_days),
        )

        self.session.add(prediction)
        await self.session.flush()  # Assign defaults, raise constraint errors early

        logger.info(
            "Persisted forecast %s (country=%s, p=%.3f, category=%s)",
            forecast_id,
            country_iso,
            forecast_output.probability,
            ensemble_prediction.category,
        )

        return prediction

    async def get_forecast_by_id(
        self, forecast_id: str
    ) -> Optional[ForecastResponse]:
        """Retrieve a forecast from PostgreSQL and return as ForecastResponse DTO.

        Args:
            forecast_id: UUID string of the prediction.

        Returns:
            ForecastResponse if found, None otherwise.
        """
        result = await self.session.execute(
            select(Prediction).where(Prediction.id == forecast_id)
        )
        prediction = result.scalar_one_or_none()

        if prediction is None:
            return None

        return self.prediction_to_dto(prediction)

    async def get_forecasts_by_country(
        self,
        country_iso: str,
        cursor: Optional[str] = None,
        limit: int = 20,
    ) -> PaginatedResponse[ForecastResponse]:
        """Retrieve forecasts for a country with cursor-based pagination.

        Results are ordered by created_at DESC (newest first).
        Cursor encodes (id, created_at) for keyset pagination.

        Args:
            country_iso: ISO country code (case-insensitive, uppercased internally).
            cursor: Opaque pagination cursor from a previous response.
            limit: Maximum items per page.

        Returns:
            PaginatedResponse with items, next_cursor, has_more.
        """
        iso_upper = country_iso.upper()

        stmt = (
            select(Prediction)
            .where(Prediction.country_iso == iso_upper)
            .order_by(Prediction.created_at.desc(), Prediction.id.desc())
        )

        # Apply cursor filter if provided
        if cursor is not None:
            try:
                cursor_data = decode_cursor(cursor)
                cursor_ts = datetime.fromisoformat(cursor_data["ts"])
                cursor_id = cursor_data["id"]
                # Keyset: rows before the cursor point (older, since DESC)
                stmt = stmt.where(
                    (Prediction.created_at < cursor_ts)
                    | (
                        (Prediction.created_at == cursor_ts)
                        & (Prediction.id < cursor_id)
                    )
                )
            except (ValueError, KeyError) as exc:
                logger.warning("Invalid cursor ignored: %s", exc)

        # Fetch one extra to determine has_more
        stmt = stmt.limit(limit + 1)
        result = await self.session.execute(stmt)
        rows = result.scalars().all()

        has_more = len(rows) > limit
        page_rows = rows[:limit]

        items = [self.prediction_to_dto(row) for row in page_rows]

        next_cursor = None
        if has_more and page_rows:
            last = page_rows[-1]
            next_cursor = encode_cursor(
                last.id,
                last.created_at.isoformat(),
            )

        return PaginatedResponse[ForecastResponse](
            items=items,
            next_cursor=next_cursor,
            has_more=has_more,
        )

    @staticmethod
    def prediction_to_dto(prediction: Prediction) -> ForecastResponse:
        """Convert a Prediction ORM model to a ForecastResponse DTO.

        Reconstructs the fully-nested DTO tree from the JSON blobs
        stored in the Prediction row.
        """
        # Reconstruct scenarios from JSON
        scenarios_data = prediction.scenarios_json
        scenarios: list[ScenarioDTO] = []
        if isinstance(scenarios_data, list):
            for s in scenarios_data:
                scenarios.append(_scenario_dict_to_dto(s))

        # Reconstruct ensemble info
        ei = prediction.ensemble_info_json or {}
        ensemble_info = EnsembleInfoDTO(
            llm_probability=ei.get("llm_probability", 0.5),
            tkg_probability=ei.get("tkg_probability"),
            weights=ei.get("weights", {"llm": 0.6, "tkg": 0.4}),
            temperature_applied=ei.get("temperature_applied", 1.0),
        )

        # Reconstruct calibration
        cal = prediction.calibration_json or {}
        calibration = CalibrationDTO(
            category=cal.get("category", prediction.category),
            temperature=cal.get("temperature", 1.0),
            historical_accuracy=cal.get("historical_accuracy", 0.0),
            brier_score=cal.get("brier_score"),
            sample_size=cal.get("sample_size", 0),
        )

        return ForecastResponse(
            forecast_id=prediction.id,
            question=prediction.question,
            prediction=prediction.prediction,
            probability=prediction.probability,
            confidence=prediction.confidence,
            horizon_days=prediction.horizon_days,
            scenarios=scenarios,
            reasoning_summary=prediction.reasoning_summary,
            evidence_count=prediction.evidence_count,
            ensemble_info=ensemble_info,
            calibration=calibration,
            created_at=prediction.created_at,
            expires_at=prediction.expires_at,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _scenarios_to_json(forecast_output: ForecastOutput) -> list[dict]:
        """Flatten the ScenarioTree into a list of serializable scenario dicts."""
        result: list[dict] = []
        for scenario in forecast_output.scenario_tree.scenarios.values():
            result.append(_scenario_to_dict(scenario))
        return result

    @staticmethod
    def _extract_entities(forecast_output: ForecastOutput) -> list[str]:
        """Extract unique entity names from all scenarios."""
        names: list[str] = []
        seen: set[str] = set()
        for scenario in forecast_output.scenario_tree.scenarios.values():
            for entity in scenario.entities:
                if entity.name not in seen:
                    names.append(entity.name)
                    seen.add(entity.name)
        return names


def _scenario_to_dict(scenario: Scenario) -> dict:
    """Convert a Scenario Pydantic model to a JSON-serializable dict for storage."""
    return {
        "scenario_id": scenario.scenario_id,
        "description": scenario.description,
        "probability": scenario.probability,
        "answers_affirmative": scenario.answers_affirmative,
        "entities": [e.name for e in scenario.entities],
        "timeline": [te.description for te in scenario.timeline],
        "evidence_sources": [],  # Evidence stored separately, not per-scenario in v1
        "child_scenarios": [],  # Flat storage -- child_ids track hierarchy
    }


def _scenario_dict_to_dto(data: dict) -> ScenarioDTO:
    """Reconstruct a ScenarioDTO from a stored JSON dict.

    Handles nested child_scenarios recursively.
    """
    children: list[ScenarioDTO] = []
    for child in data.get("child_scenarios", []):
        children.append(_scenario_dict_to_dto(child))

    evidence: list[EvidenceDTO] = []
    for ev in data.get("evidence_sources", []):
        if isinstance(ev, dict):
            evidence.append(
                EvidenceDTO(
                    source=ev.get("source", "unknown"),
                    description=ev.get("description", ""),
                    confidence=ev.get("confidence", 0.5),
                    timestamp=ev.get("timestamp"),
                    gdelt_event_id=ev.get("gdelt_event_id"),
                )
            )

    return ScenarioDTO(
        scenario_id=data.get("scenario_id", "unknown"),
        description=data.get("description", ""),
        probability=data.get("probability", 0.5),
        answers_affirmative=data.get("answers_affirmative", False),
        entities=data.get("entities", []),
        timeline=data.get("timeline", []),
        evidence_sources=evidence,
        child_scenarios=children,
    )
