"""
Concurrent multi-process PostgreSQL access test.

Verifies that 3 separate OS processes can read/write PostgreSQL simultaneously
without data corruption. This is the access pattern Phase 10 and Phase 13
depend on: FastAPI server + ingest daemon + prediction pipeline all hitting
PostgreSQL concurrently.

Additionally smoke-writes to ALL PostgreSQL tables (predictions, outcome_records,
calibration_weights, ingest_runs) to verify schema and write paths.

Uses subprocess.Popen for real process isolation (not just async tasks).

Requires PostgreSQL. Tests skip gracefully if the database is unavailable.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import textwrap
import uuid

import pytest
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.db.models import (
    Base,
    CalibrationWeight,
    IngestRun,
    OutcomeRecord,
    Prediction,
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


def _make_worker_script(worker_id: int, table: str, record_id: str) -> str:
    """Generate a self-contained Python script for a subprocess worker.

    Each worker:
    1. Creates a fresh async engine + session
    2. Writes a record to the specified table
    3. Reads it back and verifies
    4. Prints JSON result to stdout
    """
    settings = get_settings()
    db_url = settings.database_url

    if table == "prediction":
        return textwrap.dedent(f"""\
            import asyncio, json, sys
            from datetime import datetime, timedelta, timezone
            from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
            from sqlalchemy import select

            sys.path.insert(0, ".")
            from src.db.models import Base, Prediction

            async def main():
                engine = create_async_engine("{db_url}")
                async with engine.begin() as conn:
                    await conn.run_sync(Base.metadata.create_all)

                factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
                now = datetime.now(timezone.utc)

                async with factory() as session:
                    p = Prediction(
                        id="{record_id}",
                        question="Worker {worker_id} test question",
                        prediction="Worker {worker_id} prediction",
                        probability=0.{worker_id}5,
                        confidence=0.{worker_id}0,
                        horizon_days=30,
                        category="conflict",
                        reasoning_summary="Worker {worker_id} reasoning",
                        evidence_count={worker_id},
                        scenarios_json=[],
                        ensemble_info_json={{}},
                        calibration_json={{}},
                        entities=[],
                        country_iso="W{worker_id}",
                        created_at=now,
                        expires_at=now + timedelta(days=30),
                    )
                    session.add(p)
                    await session.commit()

                # Read back in new session
                async with factory() as session:
                    result = await session.execute(
                        select(Prediction).where(Prediction.id == "{record_id}")
                    )
                    row = result.scalar_one_or_none()
                    if row is None:
                        print(json.dumps({{"ok": False, "error": "Row not found"}}))
                    elif row.question != "Worker {worker_id} test question":
                        print(json.dumps({{"ok": False, "error": "Data mismatch"}}))
                    else:
                        print(json.dumps({{"ok": True, "id": row.id, "worker": {worker_id}}}))

                await engine.dispose()

            asyncio.run(main())
        """)

    elif table == "ingest_run":
        return textwrap.dedent(f"""\
            import asyncio, json, sys
            from datetime import datetime, timezone
            from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
            from sqlalchemy import select

            sys.path.insert(0, ".")
            from src.db.models import Base, IngestRun

            async def main():
                engine = create_async_engine("{db_url}")
                async with engine.begin() as conn:
                    await conn.run_sync(Base.metadata.create_all)

                factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
                now = datetime.now(timezone.utc)

                async with factory() as session:
                    run = IngestRun(
                        started_at=now,
                        completed_at=now,
                        status="success",
                        events_fetched=100,
                        events_new=50,
                        events_duplicate=50,
                    )
                    session.add(run)
                    await session.commit()
                    run_id = run.id

                # Read back
                async with factory() as session:
                    result = await session.execute(
                        select(IngestRun).where(IngestRun.id == run_id)
                    )
                    row = result.scalar_one_or_none()
                    if row is None:
                        print(json.dumps({{"ok": False, "error": "IngestRun not found"}}))
                    elif row.events_new != 50:
                        print(json.dumps({{"ok": False, "error": "Data mismatch"}}))
                    else:
                        print(json.dumps({{"ok": True, "id": row.id, "worker": {worker_id}}}))

                await engine.dispose()

            asyncio.run(main())
        """)

    raise ValueError(f"Unknown table: {table}")


async def _ensure_schema():
    """Create all tables in PostgreSQL."""
    settings = get_settings()
    engine = create_async_engine(settings.database_url)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    return engine


async def _cleanup_all(engine):
    """Remove all test data from all tables."""
    async with engine.begin() as conn:
        await conn.execute(text("DELETE FROM outcome_records"))
        await conn.execute(text("DELETE FROM calibration_weights"))
        await conn.execute(text("DELETE FROM ingest_runs"))
        await conn.execute(text("DELETE FROM predictions"))
    await engine.dispose()


@skip_no_pg
def test_concurrent_three_processes() -> None:
    """3 separate OS processes write/read PostgreSQL concurrently without corruption."""

    async def _setup():
        return await _ensure_schema()

    engine = asyncio.run(_setup())

    try:
        pred_id_1 = str(uuid.uuid4())
        pred_id_3 = str(uuid.uuid4())

        scripts = [
            _make_worker_script(1, "prediction", pred_id_1),
            _make_worker_script(2, "ingest_run", str(uuid.uuid4())),
            _make_worker_script(3, "prediction", pred_id_3),
        ]

        # Launch all 3 processes simultaneously
        processes = []
        for script in scripts:
            proc = subprocess.Popen(
                [sys.executable, "-c", script],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            processes.append(proc)

        # Wait for all to complete (10s timeout per process)
        results = []
        for i, proc in enumerate(processes):
            stdout, stderr = proc.communicate(timeout=10)
            if proc.returncode != 0:
                pytest.fail(
                    f"Worker {i + 1} exited with code {proc.returncode}.\n"
                    f"stderr: {stderr}\nstdout: {stdout}"
                )
            try:
                data = json.loads(stdout.strip())
            except json.JSONDecodeError:
                pytest.fail(
                    f"Worker {i + 1} produced invalid JSON.\n"
                    f"stdout: {stdout!r}\nstderr: {stderr}"
                )
            results.append(data)

        # All 3 must succeed
        for i, result in enumerate(results):
            assert result["ok"], (
                f"Worker {i + 1} failed: {result.get('error', 'unknown')}"
            )
    finally:
        asyncio.run(_cleanup_all(engine))


@skip_no_pg
def test_smoke_write_outcome_record() -> None:
    """Smoke-write an OutcomeRecord to verify the table and write path."""

    async def _run():
        engine = await _ensure_schema()
        factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

        from datetime import timedelta

        now = datetime.now(timezone.utc)

        # Must create a prediction first (outcome references prediction_id)
        pred_id = str(uuid.uuid4())
        async with factory() as session:
            pred = Prediction(
                id=pred_id,
                question="Smoke test question",
                prediction="Smoke test prediction",
                probability=0.5,
                confidence=0.5,
                horizon_days=30,
                category="conflict",
                reasoning_summary="smoke",
                evidence_count=0,
                scenarios_json=[],
                ensemble_info_json={},
                calibration_json={},
                entities=[],
                created_at=now,
                expires_at=now + timedelta(days=30),
            )
            session.add(pred)
            await session.commit()

        async with factory() as session:
            outcome = OutcomeRecord(
                prediction_id=pred_id,
                outcome=1.0,
                resolution_date=now,
                resolution_method="gdelt_automated",
                evidence_gdelt_ids=["EV001", "EV002"],
                notes="Smoke test outcome",
            )
            session.add(outcome)
            await session.commit()
            assert outcome.id is not None

        # Read back
        async with factory() as session:
            result = await session.execute(
                select(OutcomeRecord).where(OutcomeRecord.prediction_id == pred_id)
            )
            row = result.scalar_one()
            assert row.outcome == pytest.approx(1.0)
            assert row.resolution_method == "gdelt_automated"

        await _cleanup_all(engine)

    asyncio.run(_run())


@skip_no_pg
def test_smoke_write_calibration_weight() -> None:
    """Smoke-write a CalibrationWeight to verify the table and write path."""

    async def _run():
        engine = await _ensure_schema()
        factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

        cameo = f"TEST_{uuid.uuid4().hex[:6]}"  # Unique to avoid constraint clash

        async with factory() as session:
            weight = CalibrationWeight(
                cameo_code=cameo,
                alpha=0.65,
                sample_size=42,
                brier_score=0.18,
            )
            session.add(weight)
            await session.commit()
            assert weight.id is not None

        # Read back
        async with factory() as session:
            result = await session.execute(
                select(CalibrationWeight).where(CalibrationWeight.cameo_code == cameo)
            )
            row = result.scalar_one()
            assert row.alpha == pytest.approx(0.65)
            assert row.sample_size == 42

        await _cleanup_all(engine)

    asyncio.run(_run())


@skip_no_pg
def test_smoke_write_ingest_run() -> None:
    """Smoke-write an IngestRun to verify the table and write path."""

    async def _run():
        engine = await _ensure_schema()
        factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

        now = datetime.now(timezone.utc)

        async with factory() as session:
            run = IngestRun(
                started_at=now,
                completed_at=now,
                status="success",
                events_fetched=200,
                events_new=100,
                events_duplicate=100,
            )
            session.add(run)
            await session.commit()
            assert run.id is not None

        # Read back
        async with factory() as session:
            result = await session.execute(
                select(IngestRun).where(IngestRun.id == run.id)
            )
            row = result.scalar_one()
            assert row.events_new == 100
            assert row.status == "success"

        await _cleanup_all(engine)

    asyncio.run(_run())
