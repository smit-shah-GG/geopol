"""Integration tests for TKG backend dispatch chain.

Verifies that TKG_BACKEND envvar controls model loading, prediction
dispatch, and scheduler training function selection -- end-to-end.
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _reset_settings():
    """Reset the settings singleton so TKG_BACKEND envvar is re-read."""
    import src.settings
    src.settings._settings = None


# ---------------------------------------------------------------------------
# 1. Settings
# ---------------------------------------------------------------------------


class TestSettingsTkgBackend:
    """Verify Settings.tkg_backend field."""

    def test_default_is_regcn(self):
        """Default backend must be regcn for backward compatibility."""
        from src.settings import Settings

        s = Settings()
        assert s.tkg_backend == "regcn"

    def test_tirgn_via_constructor(self):
        """Explicit constructor arg sets backend to tirgn."""
        from src.settings import Settings

        s = Settings(tkg_backend="tirgn")
        assert s.tkg_backend == "tirgn"

    def test_tirgn_via_envvar(self, monkeypatch):
        """TKG_BACKEND envvar propagates to Settings."""
        monkeypatch.setenv("TKG_BACKEND", "tirgn")
        _reset_settings()

        from src.settings import get_settings

        s = get_settings()
        assert s.tkg_backend == "tirgn"

        # Cleanup
        _reset_settings()


# ---------------------------------------------------------------------------
# 2. TKGPredictor backend dispatch
# ---------------------------------------------------------------------------


class TestTKGPredictorBackend:
    """Verify TKGPredictor initializes correct backend."""

    def test_regcn_backend(self, monkeypatch):
        """regcn backend creates REGCNWrapper, sets _backend='regcn'."""
        monkeypatch.setenv("TKG_BACKEND", "regcn")
        _reset_settings()

        from src.forecasting.tkg_predictor import TKGPredictor

        p = TKGPredictor(auto_load=False)
        assert p._backend == "regcn"
        assert p.model is not None  # REGCNWrapper instantiated
        assert p._tirgn_model is None

        _reset_settings()

    def test_tirgn_backend(self, monkeypatch):
        """tirgn backend sets model=None, _backend='tirgn'."""
        monkeypatch.setenv("TKG_BACKEND", "tirgn")
        _reset_settings()

        from src.forecasting.tkg_predictor import TKGPredictor

        p = TKGPredictor(auto_load=False)
        assert p._backend == "tirgn"
        assert p.model is None  # No REGCNWrapper
        assert p._tirgn_model is None  # Not loaded yet
        assert p.trained is False

        _reset_settings()

    def test_default_model_path_regcn(self, monkeypatch):
        """regcn default model path is .pt file."""
        monkeypatch.setenv("TKG_BACKEND", "regcn")
        _reset_settings()

        from src.forecasting.tkg_predictor import TKGPredictor

        p = TKGPredictor(auto_load=False)
        assert p.default_model_path.suffix == ".pt"
        assert "regcn" in p.default_model_path.name

        _reset_settings()

    def test_default_model_path_tirgn(self, monkeypatch):
        """tirgn default model path is .npz file."""
        monkeypatch.setenv("TKG_BACKEND", "tirgn")
        _reset_settings()

        from src.forecasting.tkg_predictor import TKGPredictor

        p = TKGPredictor(auto_load=False)
        assert p.default_model_path.suffix == ".npz"
        assert "tirgn" in p.default_model_path.name

        _reset_settings()


# ---------------------------------------------------------------------------
# 3. Scheduler dispatch
# ---------------------------------------------------------------------------


class TestSchedulerDispatch:
    """Verify scheduler dispatches to correct training function."""

    def test_dispatches_regcn(self, monkeypatch, tmp_path):
        """With regcn backend, _train_new_model calls train_regcn."""
        monkeypatch.setenv("TKG_BACKEND", "regcn")
        _reset_settings()

        from src.training.scheduler import RetrainingScheduler

        scheduler = RetrainingScheduler(
            model_dir=tmp_path / "models",
            log_dir=tmp_path / "logs",
        )

        mock_result = {"status": "complete", "metrics": {"mrr": 0.5}}

        with patch(
            "src.training.scheduler.RetrainingScheduler._train_regcn",
            return_value=mock_result,
        ) as mock_train:
            result = scheduler._train_new_model(Path("data/test.parquet"))
            mock_train.assert_called_once()
            assert result["status"] == "complete"

        _reset_settings()

    def test_dispatches_tirgn(self, monkeypatch, tmp_path):
        """With tirgn backend, _train_new_model calls _train_tirgn."""
        monkeypatch.setenv("TKG_BACKEND", "tirgn")
        _reset_settings()

        from src.training.scheduler import RetrainingScheduler

        scheduler = RetrainingScheduler(
            model_dir=tmp_path / "models",
            log_dir=tmp_path / "logs",
        )

        mock_result = {"status": "complete", "metrics": {"mrr": 0.45}}

        with patch(
            "src.training.scheduler.RetrainingScheduler._train_tirgn",
            return_value=mock_result,
        ) as mock_train:
            result = scheduler._train_new_model(Path("data/test.parquet"))
            mock_train.assert_called_once()
            assert result["status"] == "complete"

        _reset_settings()

    def test_set_last_trained_includes_model_type(self, monkeypatch, tmp_path):
        """_set_last_trained_time includes model_type in JSON."""
        monkeypatch.setenv("TKG_BACKEND", "tirgn")
        _reset_settings()

        from datetime import datetime

        from src.training.scheduler import RetrainingScheduler

        scheduler = RetrainingScheduler(
            model_dir=tmp_path / "models",
            log_dir=tmp_path / "logs",
        )

        scheduler._set_last_trained_time(datetime(2026, 3, 1, 12, 0, 0))

        with open(scheduler.last_trained_path) as f:
            data = json.load(f)

        assert data["model_type"] == "tirgn"
        assert "tirgn" in data["model_path"]

        _reset_settings()


# ---------------------------------------------------------------------------
# 4. EnsemblePredictor agnostic
# ---------------------------------------------------------------------------


class TestEnsemblePredictorAgnostic:
    """Verify EnsemblePredictor works with TiRGN backend (graceful fallback)."""

    def test_untrained_tirgn_falls_back(self, monkeypatch):
        """EnsemblePredictor with untrained TiRGN TKGPredictor falls back to LLM."""
        monkeypatch.setenv("TKG_BACKEND", "tirgn")
        _reset_settings()

        from src.forecasting.tkg_predictor import TKGPredictor

        tkg = TKGPredictor(auto_load=False)
        assert tkg.trained is False

        # EnsemblePredictor._get_tkg_prediction checks tkg_predictor.trained
        # If not trained, it returns available=False -- no crash.
        from src.forecasting.ensemble_predictor import EnsemblePredictor

        ensemble = EnsemblePredictor(
            llm_orchestrator=None,
            tkg_predictor=tkg,
            alpha=0.6,
        )

        # _get_tkg_prediction should return unavailable gracefully
        pred = ensemble._get_tkg_prediction("RUSSIA", "CONFLICT", "UKRAINE")
        assert pred.available is False
        assert pred.component == "tkg"

        _reset_settings()


# ---------------------------------------------------------------------------
# 5. Checkpoint model_type validation
# ---------------------------------------------------------------------------


class TestCheckpointModelTypeValidation:
    """Cross-loading checkpoints with wrong backend must fail gracefully."""

    def test_regcn_checkpoint_rejected_by_tirgn_backend(self, monkeypatch, tmp_path):
        """Loading a checkpoint with model_type='regcn' when backend=tirgn returns False."""
        monkeypatch.setenv("TKG_BACKEND", "tirgn")
        _reset_settings()

        from src.forecasting.tkg_predictor import TKGPredictor

        # Create fake checkpoint with wrong model_type
        npz_path = tmp_path / "tirgn_best.npz"
        json_path = tmp_path / "tirgn_best.json"

        import numpy as np

        np.savez(npz_path, dummy=np.zeros(1))

        metadata = {
            "model_type": "regcn",  # Wrong type!
            "config": {
                "num_entities": 10,
                "num_relations": 5,
                "embedding_dim": 32,
                "num_layers": 1,
            },
            "entity_to_id": {"A": 0},
            "relation_to_id": {"R": 0},
        }
        with open(json_path, "w") as f:
            json.dump(metadata, f)

        p = TKGPredictor(auto_load=False)
        result = p._load_tirgn_checkpoint(npz_path)
        assert result is False
        assert p.trained is False

        _reset_settings()

    def test_missing_json_rejected(self, monkeypatch, tmp_path):
        """Loading checkpoint without JSON sidecar returns False."""
        monkeypatch.setenv("TKG_BACKEND", "tirgn")
        _reset_settings()

        from src.forecasting.tkg_predictor import TKGPredictor

        import numpy as np

        npz_path = tmp_path / "tirgn_best.npz"
        np.savez(npz_path, dummy=np.zeros(1))
        # No JSON file

        p = TKGPredictor(auto_load=False)
        result = p._load_tirgn_checkpoint(npz_path)
        assert result is False

        _reset_settings()


# ---------------------------------------------------------------------------
# 6. Retrain script CLI
# ---------------------------------------------------------------------------


class TestRetrainScriptCLI:
    """Verify scripts/retrain_tkg.py --help shows --backend arg."""

    def test_help_shows_backend(self):
        """--help output includes --backend argument."""
        result = subprocess.run(
            [sys.executable, "scripts/retrain_tkg.py", "--help"],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).parent.parent),
        )
        assert result.returncode == 0
        assert "--backend" in result.stdout
        assert "tirgn" in result.stdout
        assert "regcn" in result.stdout


# ---------------------------------------------------------------------------
# 7. Backup backend-awareness
# ---------------------------------------------------------------------------


class TestBackupBackendAwareness:
    """Verify _backup_current_model uses correct file patterns."""

    @staticmethod
    def _write_minimal_config(path: Path, model_dir: Path, log_dir: Path) -> None:
        """Write a minimal retraining.yaml that does NOT override model_dir."""
        import yaml

        config = {
            "schedule": {"frequency": "weekly", "day_of_week": 0, "hour": 2},
            "data": {"data_window": 30, "max_events": 100000},
            "model": {"epochs": 1},
            "versioning": {
                "backup_count": 3,
                "model_dir": str(model_dir),
                "log_dir": str(log_dir),
            },
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(config, f)

    def test_regcn_backup(self, monkeypatch, tmp_path):
        """regcn backup copies .pt file."""
        monkeypatch.setenv("TKG_BACKEND", "regcn")
        _reset_settings()

        from src.training.scheduler import RetrainingScheduler

        model_dir = tmp_path / "models"
        log_dir = tmp_path / "logs"
        config_path = tmp_path / "config.yaml"
        self._write_minimal_config(config_path, model_dir, log_dir)

        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "regcn_trained.pt").write_bytes(b"fake_model")

        scheduler = RetrainingScheduler(
            config_path=config_path,
            model_dir=model_dir,
            log_dir=log_dir,
        )

        backup = scheduler._backup_current_model()
        assert backup is not None
        assert "regcn_backup_" in backup.name
        assert backup.suffix == ".pt"

        _reset_settings()

    def test_tirgn_backup(self, monkeypatch, tmp_path):
        """tirgn backup copies both .npz and .json files."""
        monkeypatch.setenv("TKG_BACKEND", "tirgn")
        _reset_settings()

        from src.training.scheduler import RetrainingScheduler

        model_dir = tmp_path / "models"
        log_dir = tmp_path / "logs"
        config_path = tmp_path / "config.yaml"
        self._write_minimal_config(config_path, model_dir, log_dir)

        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "tirgn_best.npz").write_bytes(b"fake_npz")
        (model_dir / "tirgn_best.json").write_text('{"model_type": "tirgn"}')

        scheduler = RetrainingScheduler(
            config_path=config_path,
            model_dir=model_dir,
            log_dir=log_dir,
        )

        backup = scheduler._backup_current_model()
        assert backup is not None
        assert "tirgn_backup_" in backup.name
        assert backup.suffix == ".npz"
        # JSON companion should also exist
        assert backup.with_suffix(".json").exists()

        _reset_settings()
