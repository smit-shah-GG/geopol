"""
Module-level functions for heavy job execution.

All functions are module-level (no bound methods, no closures) so they
are pickleable by ProcessPoolExecutor. Each function either:
  - Uses subprocess.run() for scripts/ (not a Python package)
  - Uses in-process imports for src/ (proper package)

daily_pipeline and tkg_retrain use subprocess.run() because:
  1. scripts/ has no __init__.py and is not in pyproject.toml packages
  2. retrain_tkg.main() calls argparse.parse_args() which reads sys.argv

polymarket uses in-process imports because src.polymarket.* is a proper
package and needs fine-grained async control (aiohttp sessions, DB txns).
"""

from __future__ import annotations

import logging
import subprocess

logger = logging.getLogger(__name__)

# Track last reforecast date for at-most-once-daily semantics
_last_reforecast_date: str | None = None


def run_daily_pipeline() -> int:
    """Execute scripts/daily_forecast.py via subprocess.

    Returns:
        0 on success, 1 on failure or timeout.
    """
    logger.info("Starting daily forecast pipeline (subprocess)")
    try:
        result = subprocess.run(
            ["uv", "run", "python", "scripts/daily_forecast.py"],
            capture_output=True,
            text=True,
            timeout=3600,
        )
        if result.stdout:
            for line in result.stdout.strip().split("\n"):
                logger.info("[daily_pipeline:stdout] %s", line)
        if result.stderr:
            for line in result.stderr.strip().split("\n"):
                logger.warning("[daily_pipeline:stderr] %s", line)

        if result.returncode != 0:
            logger.error(
                "daily_pipeline exited with code %d", result.returncode,
            )
        else:
            logger.info("daily_pipeline completed successfully")

        return result.returncode

    except subprocess.TimeoutExpired:
        logger.error("daily_pipeline timed out after 3600s")
        return 1
    except Exception:
        logger.exception("daily_pipeline subprocess failed")
        return 1


def run_polymarket_cycle() -> int:
    """Execute one Polymarket matching + auto-forecast cycle in-process.

    Creates its own asyncio event loop via asyncio.run() because this
    function runs inside a ProcessPoolExecutor worker (no existing loop).

    Cycle order (critical -- resolve BEFORE reforecast to prevent race):
      1. PolymarketComparisonService.run_matching_cycle()
      2. capture_snapshots()
      3. resolve_completed() -- detect resolutions + voided markets
      4. PolymarketAutoForecaster.run() with top-10 geo events
      5. reforecast_active() at most once per UTC day (top-10 only)

    Returns:
        0 on success, 1 on failure.
    """
    import asyncio as _asyncio
    import logging as _logging

    _logger = _logging.getLogger(__name__)

    async def _cycle() -> None:
        global _last_reforecast_date  # noqa: PLW0603

        import aiohttp

        from src.db import postgres as _pg
        from src.forecasting.gemini_client import GeminiClient
        from src.polymarket.auto_forecaster import PolymarketAutoForecaster
        from src.polymarket.client import PolymarketClient
        from src.polymarket.comparison import PolymarketComparisonService
        from src.polymarket.matcher import PolymarketMatcher
        from src.settings import get_settings

        settings = get_settings()

        # Ensure DB engine exists in this subprocess
        if _pg.async_session_factory is None:
            _pg.init_db()
        async_session_factory = _pg.async_session_factory

        gemini_client = GeminiClient()
        matcher = PolymarketMatcher(
            gemini_client,
            match_threshold=settings.polymarket_match_threshold,
        )

        async with aiohttp.ClientSession() as http_session:
            pm_client = PolymarketClient(session=http_session)
            service = PolymarketComparisonService(
                async_session_factory=async_session_factory,
                polymarket_client=pm_client,
                matcher=matcher,
                settings=settings,
            )

            # Step 1: Match predictions to Polymarket events
            result = await service.run_matching_cycle()
            _logger.info("Polymarket matching: %s", result)

            # Step 2: Capture price/probability snapshots
            await service.capture_snapshots()

            # Step 3: Resolve completed markets (detect resolutions + voided)
            # Must run BEFORE reforecast to prevent scoring post-reforecast probs
            resolve_result = await service.resolve_completed()
            _logger.info("Polymarket resolution: %s", resolve_result)

            # Step 4: Auto-forecast top-10 geopolitical markets by volume
            auto_forecaster = PolymarketAutoForecaster(
                async_session_factory=async_session_factory,
                gemini_client=gemini_client,
                settings=settings,
            )
            top_10_events, _ = await pm_client.fetch_top_geopolitical(limit=10)
            auto_result = await auto_forecaster.run(
                top_10_events, tracked_ids=set(),
            )
            _logger.info("Polymarket auto-forecast: %s", auto_result)

            # Step 5: Re-forecast active comparisons (at most once per UTC day)
            # Only reforecast comparisons in the current top-10 set
            from datetime import date

            today_str = date.today().isoformat()
            if _last_reforecast_date != today_str:
                top_10_ids = {
                    str(e.get("id", ""))
                    for e in top_10_events
                    if e.get("id")
                }
                reforecast_result = await auto_forecaster.reforecast_active(
                    active_event_ids=top_10_ids
                )
                _logger.info("Polymarket re-forecast: %s", reforecast_result)
                _last_reforecast_date = today_str

    _logger.info("Starting polymarket cycle (in-process)")
    try:
        _asyncio.run(_cycle())
        _logger.info("Polymarket cycle completed successfully")
        return 0
    except Exception:
        _logger.exception("Polymarket cycle failed")
        return 1


def run_backtest(config_json: str) -> int:
    """Execute a backtest run in-process with its own asyncio event loop.

    Creates its own DB engine (subprocess isolation), deserializes the
    BacktestRunConfig from the JSON string, and runs BacktestRunner.
    The config_json is a plain string -- passes through ProcessPoolExecutor
    pickling without issue.

    Args:
        config_json: JSON-serialized BacktestRunConfig (from to_json()).

    Returns:
        0 on success, 1 on failure.
    """
    import asyncio as _asyncio
    import logging as _logging

    _logger = _logging.getLogger(__name__)

    async def _run() -> None:
        from src.backtesting.runner import BacktestRunner
        from src.backtesting.schemas import BacktestRunConfig
        from src.db import postgres as _pg

        # Create own DB engine in subprocess (no shared state with parent)
        if _pg.async_session_factory is None:
            _pg.init_db()

        config = BacktestRunConfig.from_json(config_json)
        runner = BacktestRunner(
            run_config=config,
            async_session_factory=_pg.async_session_factory,
        )
        await runner.run()

    _logger.info("Starting backtest run (in-process)")
    try:
        _asyncio.run(_run())
        _logger.info("Backtest run completed successfully")
        return 0
    except Exception:
        _logger.exception("Backtest run failed")
        return 1


def run_tkg_retrain() -> int:
    """Execute scripts/retrain_tkg.py --force via subprocess.

    Returns:
        0 on success, 1 on failure or timeout.
    """
    logger.info("Starting TKG retrain (subprocess, --force)")
    try:
        result = subprocess.run(
            ["uv", "run", "python", "scripts/retrain_tkg.py", "--force"],
            capture_output=True,
            text=True,
            timeout=7200,
        )
        if result.stdout:
            for line in result.stdout.strip().split("\n"):
                logger.info("[tkg_retrain:stdout] %s", line)
        if result.stderr:
            for line in result.stderr.strip().split("\n"):
                logger.warning("[tkg_retrain:stderr] %s", line)

        if result.returncode != 0:
            logger.error(
                "tkg_retrain exited with code %d", result.returncode,
            )
        else:
            logger.info("TKG retrain completed successfully")

        return result.returncode

    except subprocess.TimeoutExpired:
        logger.error("tkg_retrain timed out after 7200s")
        return 1
    except Exception:
        logger.exception("tkg_retrain subprocess failed")
        return 1
