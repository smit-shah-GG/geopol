"""
Tests for ForecastService persistence round-trip.

Verifies:
    1. persist_forecast() writes a Prediction row to PostgreSQL
    2. get_forecast_by_id() retrieves and reconstructs ForecastResponse DTO
    3. get_forecasts_by_country() returns paginated results filtered by ISO
    4. prediction_to_dto() correctly maps ORM -> DTO including nested scenarios

Uses synthetic ForecastOutput / EnsemblePrediction -- does NOT invoke the
real EnsemblePredictor (no Gemini API, no TKG model needed).

Requires PostgreSQL. Tests skip gracefully if the database is unavailable.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone
from typing import Tuple

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.api.services.forecast_service import ForecastService
from src.db.models import Base, Prediction
from src.forecasting.ensemble_predictor import ComponentPrediction, EnsemblePrediction
from src.forecasting.models import (
    Entity,
    ForecastOutput,
    Scenario,
    ScenarioTree,
    TimelineEvent,
)
from src.settings import get_settings


def _pg_available() -> bool:
    """Check if PostgreSQL is reachable."""
    try:
        settings = get_settings()
        url = settings.database_url

        async def _probe() -> bool:
            eng = create_async_engine(url, pool_pre_ping=True)
            try:
                async with eng.connect() as conn:
                    await conn.execute(text("SELECT 1"))
                return True
            except Exception:
                return False
            finally:
                await eng.dispose()

        return asyncio.run(_probe())
    except Exception:
        return False


PG_AVAILABLE = _pg_available()
skip_no_pg = pytest.mark.skipif(not PG_AVAILABLE, reason="PostgreSQL not available")


def _make_synthetic_data() -> Tuple[ForecastOutput, EnsemblePrediction]:
    """Build synthetic ForecastOutput + EnsemblePrediction for testing."""
    scenario_a = Scenario(
        scenario_id="sc-escalation",
        description="Military escalation in eastern region",
        entities=[
            Entity(name="Country A", type="COUNTRY", role="ACTOR"),
            Entity(name="Country B", type="COUNTRY", role="TARGET"),
        ],
        timeline=[
            TimelineEvent(
                relative_time="T+1 week",
                description="Troop mobilization begins",
                probability=0.8,
            ),
            TimelineEvent(
                relative_time="T+2 weeks",
                description="Cross-border skirmish occurs",
                probability=0.5,
            ),
        ],
        probability=0.65,
        answers_affirmative=True,
        reasoning_path=[],
    )

    scenario_b = Scenario(
        scenario_id="sc-deescalation",
        description="Diplomatic resolution via mediation",
        entities=[
            Entity(name="Country A", type="COUNTRY", role="ACTOR"),
            Entity(name="Mediator Org", type="ORGANIZATION", role="MEDIATOR"),
        ],
        timeline=[
            TimelineEvent(
                relative_time="T+3 days",
                description="Emergency talks convened",
                probability=0.7,
            ),
        ],
        probability=0.35,
        answers_affirmative=False,
        reasoning_path=[],
    )

    tree = ScenarioTree(
        question="Will conflict escalate in eastern region within 30 days?",
        root_scenario=scenario_a,
        scenarios={"sc-escalation": scenario_a, "sc-deescalation": scenario_b},
    )

    forecast_output = ForecastOutput(
        question="Will conflict escalate in eastern region within 30 days?",
        prediction="Escalation is moderately likely given troop movements",
        probability=0.65,
        confidence=0.72,
        scenario_tree=tree,
        selected_scenario_ids=["sc-escalation"],
        reasoning_summary="Historical patterns and current posture suggest escalation",
        evidence_sources=["GDELT event analysis", "Graph pattern analysis (TKG)"],
        timestamp=datetime.now(timezone.utc),
    )

    ensemble_prediction = EnsemblePrediction(
        final_probability=0.65,
        final_confidence=0.72,
        raw_confidence=0.75,
        calibrated_confidence=0.72,
        llm_prediction=ComponentPrediction(
            component="llm",
            probability=0.70,
            confidence=0.80,
            available=True,
        ),
        tkg_prediction=ComponentPrediction(
            component="tkg",
            probability=0.55,
            confidence=0.60,
            available=True,
        ),
        weights_used=(0.6, 0.4),
        temperature=1.0,
        category="conflict",
    )

    return forecast_output, ensemble_prediction


async def _make_session():
    """Create a fresh async session for testing."""
    settings = get_settings()
    engine = create_async_engine(settings.database_url)

    # Ensure schema exists
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    return engine, factory


async def _cleanup(engine, tables=("predictions",)):
    """Remove test data."""
    async with engine.begin() as conn:
        for table in tables:
            await conn.execute(text(f"DELETE FROM {table}"))
    await engine.dispose()


@skip_no_pg
def test_forecast_persist_and_retrieve_by_id() -> None:
    """persist_forecast() -> get_forecast_by_id() round-trips correctly."""

    async def _run():
        engine, factory = await _make_session()
        try:
            forecast_output, ensemble_pred = _make_synthetic_data()

            async with factory() as session:
                service = ForecastService(session)
                prediction = await service.persist_forecast(
                    forecast_output, ensemble_pred, country_iso="UA"
                )
                await session.commit()

                assert prediction.id is not None
                assert len(prediction.id) == 36
                assert prediction.probability == pytest.approx(0.65, abs=0.001)
                assert prediction.country_iso == "UA"

            # Retrieve in separate session to confirm persistence
            async with factory() as session:
                service = ForecastService(session)
                dto = await service.get_forecast_by_id(prediction.id)
                assert dto is not None
                assert dto.forecast_id == prediction.id
                assert dto.question == forecast_output.question
                assert dto.probability == pytest.approx(0.65, abs=0.001)
                assert dto.confidence == pytest.approx(0.72, abs=0.001)
                assert dto.horizon_days == 30
                assert dto.evidence_count == 2
                assert dto.reasoning_summary == forecast_output.reasoning_summary
        finally:
            await _cleanup(engine)

    asyncio.run(_run())


@skip_no_pg
def test_forecast_persist_and_retrieve_by_country() -> None:
    """persist_forecast() -> get_forecasts_by_country() filters correctly."""

    async def _run():
        engine, factory = await _make_session()
        try:
            forecast_output, ensemble_pred = _make_synthetic_data()

            async with factory() as session:
                service = ForecastService(session)
                await service.persist_forecast(
                    forecast_output, ensemble_pred, country_iso="UA"
                )
                await service.persist_forecast(
                    forecast_output, ensemble_pred, country_iso="UA"
                )
                await service.persist_forecast(
                    forecast_output, ensemble_pred, country_iso="SY"
                )
                await session.commit()

            async with factory() as session:
                service = ForecastService(session)

                # UA should have 2
                page = await service.get_forecasts_by_country("UA", limit=10)
                assert len(page.items) == 2

                # SY should have 1
                page_sy = await service.get_forecasts_by_country("SY", limit=10)
                assert len(page_sy.items) == 1

                # XX should have 0
                page_empty = await service.get_forecasts_by_country("XX", limit=10)
                assert len(page_empty.items) == 0
        finally:
            await _cleanup(engine)

    asyncio.run(_run())


@skip_no_pg
def test_prediction_to_dto_reconstructs_scenarios() -> None:
    """prediction_to_dto() correctly reconstructs nested ScenarioDTOs from JSON."""

    async def _run():
        engine, factory = await _make_session()
        try:
            forecast_output, ensemble_pred = _make_synthetic_data()

            async with factory() as session:
                service = ForecastService(session)
                prediction = await service.persist_forecast(
                    forecast_output, ensemble_pred, country_iso="MM"
                )
                await session.commit()

                dto = ForecastService.prediction_to_dto(prediction)

            # Should have 2 scenarios from synthetic data
            assert len(dto.scenarios) == 2

            # Find the escalation scenario
            escalation = next(
                (s for s in dto.scenarios if s.scenario_id == "sc-escalation"), None
            )
            assert escalation is not None
            assert escalation.probability == pytest.approx(0.65, abs=0.001)
            assert escalation.answers_affirmative is True
            assert "Country A" in escalation.entities
            assert "Country B" in escalation.entities
            assert len(escalation.timeline) == 2

            # Check ensemble info reconstruction
            assert dto.ensemble_info.llm_probability == pytest.approx(0.70, abs=0.001)
            assert dto.ensemble_info.tkg_probability == pytest.approx(0.55, abs=0.001)
            assert dto.ensemble_info.weights == {"llm": 0.6, "tkg": 0.4}
            assert dto.ensemble_info.temperature_applied == pytest.approx(1.0)

            # Check calibration reconstruction
            assert dto.calibration.category == "conflict"
            assert dto.calibration.sample_size == 0
        finally:
            await _cleanup(engine)

    asyncio.run(_run())


@skip_no_pg
def test_get_forecast_by_id_returns_none_for_missing() -> None:
    """get_forecast_by_id() returns None for nonexistent ID."""

    async def _run():
        engine, factory = await _make_session()
        try:
            async with factory() as session:
                service = ForecastService(session)
                result = await service.get_forecast_by_id(str(uuid.uuid4()))
                assert result is None
        finally:
            await engine.dispose()

    asyncio.run(_run())
