"""Phase 13 UAT — Calibration, Monitoring & Hardening.

Run: uv run pytest tests/test_uat_phase13.py -v
"""

import inspect
import logging
import os
import tempfile


def test_01_all_new_packages_import_cleanly():
    """All 12 new modules across calibration, monitoring, and polymarket import."""
    from src.calibration.priors import CAMEO_TO_SUPER, COLD_START_PRIORS
    from src.calibration.weight_optimizer import WeightOptimizer, optimize_alpha_for_category
    from src.calibration.weight_loader import WeightLoader
    from src.monitoring.alert_manager import AlertManager
    from src.monitoring.feed_monitor import FeedMonitor
    from src.monitoring.drift_monitor import DriftMonitor
    from src.monitoring.budget_monitor import BudgetMonitor
    from src.monitoring.disk_monitor import DiskMonitor
    from src.polymarket.client import PolymarketClient
    from src.polymarket.matcher import PolymarketMatcher
    from src.polymarket.comparison import PolymarketComparisonService


def test_02_lbfgsb_optimizer_produces_valid_alpha():
    """L-BFGS-B optimizer returns alpha in [0,1] with a Brier score."""
    from src.calibration.weight_optimizer import optimize_alpha_for_category

    alpha, brier = optimize_alpha_for_category(
        [1.0, 0.0, 1.0, 0.0],
        [0.8, 0.2, 0.7, 0.3],
        [0.9, 0.1, 0.8, 0.2],
    )
    print(f"alpha={alpha:.3f}, brier={brier:.4f}")
    assert 0.0 <= alpha <= 1.0, f"alpha {alpha} out of bounds"
    assert brier >= 0.0, f"brier {brier} negative"


def test_03_cold_start_priors_asymmetric_values():
    """Cold-start priors: verbal_coop=0.65, material_conflict=0.50, global=0.58, 20 codes."""
    from src.calibration.priors import COLD_START_PRIORS, CAMEO_TO_SUPER

    assert COLD_START_PRIORS["verbal_coop"] == 0.65
    assert COLD_START_PRIORS["material_conflict"] == 0.50
    assert COLD_START_PRIORS["global"] == 0.58
    assert len(CAMEO_TO_SUPER) == 20, f"Expected 20 CAMEO codes, got {len(CAMEO_TO_SUPER)}"


def test_04_disk_monitor_reports_real_usage():
    """DiskMonitor returns real system disk stats."""
    from src.monitoring.disk_monitor import DiskMonitor
    from src.settings import Settings

    dm = DiskMonitor(Settings())
    status = dm.check_disk()
    print(f"Status: {status['status']}, Used: {status['percent_used']:.1f}%, Free: {status['free_gb']:.1f} GB")
    assert status["status"] in ("ok", "warning", "critical")
    assert 0.0 <= status["percent_used"] <= 100.0
    assert status["free_gb"] >= 0.0


def test_05_log_rotation_creates_file():
    """setup_logging creates a JSON-formatted geopol.log on disk."""
    from src.logging_config import setup_logging

    d = tempfile.mkdtemp()
    setup_logging(log_dir=d)
    logging.getLogger("test").info("rotation test")
    logfile = os.path.join(d, "geopol.log")
    assert os.path.exists(logfile), f"Log file not created at {logfile}"
    line = open(logfile).readline()
    print(line[:80])
    assert "{" in line, "Expected JSON-formatted log line"


def test_06_systemd_units_exist():
    """4 systemd unit files with correct directives."""
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    systemd_dir = os.path.join(repo_root, "systemd")
    assert os.path.isdir(systemd_dir), f"systemd/ directory missing at {systemd_dir}"

    files = os.listdir(systemd_dir)
    assert len(files) == 4, f"Expected 4 systemd files, got {len(files)}: {files}"

    ingest = open(os.path.join(systemd_dir, "geopol-ingest.service")).read()
    assert "Restart=on-failure" in ingest, "Missing Restart=on-failure in ingest service"

    timer = open(os.path.join(systemd_dir, "geopol-daily-forecast.timer")).read()
    assert "OnCalendar" in timer, "Missing OnCalendar in daily forecast timer"


def test_07_ensemble_predictor_dynamic_alpha_params():
    """EnsemblePredictor.predict() has alpha_override and cameo_root_code params."""
    from src.forecasting.ensemble_predictor import EnsemblePredictor

    sig = inspect.signature(EnsemblePredictor.predict)
    params = list(sig.parameters)
    print(f"predict() params: {params}")
    assert "alpha_override" in params, f"Missing alpha_override in {params}"
    assert "cameo_root_code" in params, f"Missing cameo_root_code in {params}"


def test_08_health_schema_10_subsystems():
    """Health endpoint schema defines 10 subsystem names."""
    from src.api.schemas.health import SUBSYSTEM_NAMES

    print(f"Subsystems ({len(SUBSYSTEM_NAMES)}): {SUBSYSTEM_NAMES}")
    assert len(SUBSYSTEM_NAMES) == 10, f"Expected 10, got {len(SUBSYSTEM_NAMES)}"
    for expected in ("api_budget", "disk_usage", "calibration_freshness"):
        assert expected in SUBSYSTEM_NAMES, f"Missing {expected}"


def test_09_calibration_api_routes_registered():
    """Calibration router exposes /polymarket, /weights, /weights/history."""
    from src.api.routes.v1.calibration import router

    routes = [r.path for r in router.routes]
    print(f"Routes: {routes}")
    assert any("polymarket" in r for r in routes), f"Missing /polymarket in {routes}"
    assert any("weights" in r for r in routes), f"Missing /weights in {routes}"


def test_10_settings_phase13_fields():
    """Settings expose all Phase 13 fields with correct defaults."""
    from src.settings import Settings

    s = Settings()
    assert s.calibration_min_samples == 10
    assert s.smtp_port == 587
    assert s.polymarket_enabled is True
    assert s.log_dir == "data/logs"
    assert s.feed_staleness_hours == 1.0


def test_11_frontend_calibration_polymarket():
    """CalibrationPanel.ts has updatePolymarket, api.ts has PolymarketComparison."""
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    cal_panel = os.path.join(repo_root, "frontend", "src", "components", "CalibrationPanel.ts")
    assert os.path.exists(cal_panel), f"CalibrationPanel.ts not found at {cal_panel}"
    cal_content = open(cal_panel).read()
    assert "updatePolymarket" in cal_content, "Missing updatePolymarket method"

    api_types = os.path.join(repo_root, "frontend", "src", "types", "api.ts")
    assert os.path.exists(api_types), f"api.ts not found at {api_types}"
    api_content = open(api_types).read()
    assert "PolymarketComparison" in api_content, "Missing PolymarketComparison interface"


def test_12_alembic_migration_upgrade_downgrade():
    """Phase 13 Alembic migration has upgrade and downgrade functions."""
    import importlib.util

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    migration = os.path.join(repo_root, "alembic", "versions", "20260302_001_phase13_schema.py")
    assert os.path.exists(migration), f"Migration file not found at {migration}"

    spec = importlib.util.spec_from_file_location("phase13_migration", migration)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert hasattr(mod, "upgrade"), "Missing upgrade function"
    assert hasattr(mod, "downgrade"), "Missing downgrade function"
    assert callable(mod.upgrade)
    assert callable(mod.downgrade)
