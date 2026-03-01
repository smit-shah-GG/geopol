"""
Retraining scheduler for periodic TKG model updates.

Implements time-based retraining (weekly/monthly) to keep the TKG predictor
current with evolving world events. The scheduler:
- Tracks last training time
- Determines when retraining is due
- Orchestrates the full retraining pipeline
- Manages model versioning and backup

The scheduler is model-agnostic: it dispatches to the correct training
function (RE-GCN or TiRGN) based on ``Settings.tkg_backend``.
"""

import json
import logging
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

import yaml

from src.settings import get_settings

logger = logging.getLogger(__name__)

# Default paths
DEFAULT_CONFIG_PATH = Path("config/retraining.yaml")
DEFAULT_MODEL_DIR = Path("models/tkg")
DEFAULT_LOG_DIR = Path("logs/retraining")
LAST_TRAINED_FILE = "last_trained.json"


class RetrainingScheduler:
    """
    Manages periodic retraining of TKG models on a configurable schedule.

    The scheduler follows a time-based approach (weekly or monthly) rather than
    performance-triggered retraining for simplicity and predictability.  It
    dispatches to the correct training function based on ``Settings.tkg_backend``
    (RE-GCN or TiRGN).

    Attributes:
        config: Retraining configuration dictionary
        model_dir: Directory containing model checkpoints
        log_dir: Directory for retraining logs
        last_trained_path: Path to file tracking last training timestamp
    """

    def __init__(
        self,
        config_path: Optional[Path] = None,
        model_dir: Optional[Path] = None,
        log_dir: Optional[Path] = None,
    ):
        """
        Initialize the retraining scheduler.

        Args:
            config_path: Path to YAML configuration file
            model_dir: Directory for model checkpoints
            log_dir: Directory for retraining logs
        """
        self.config_path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
        self.config = self._load_config()

        self.model_dir = Path(model_dir) if model_dir else DEFAULT_MODEL_DIR
        self.log_dir = Path(log_dir) if log_dir else DEFAULT_LOG_DIR

        # Override from config if specified
        if "versioning" in self.config:
            if "model_dir" in self.config["versioning"]:
                self.model_dir = Path(self.config["versioning"]["model_dir"])
            if "log_dir" in self.config["versioning"]:
                self.log_dir = Path(self.config["versioning"]["log_dir"])

        self.last_trained_path = self.model_dir / LAST_TRAINED_FILE

        # Ensure directories exist
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initialized RetrainingScheduler with {self.config_path}")
        logger.info(f"  Frequency: {self.config['schedule']['frequency']}")
        logger.info(f"  Model dir: {self.model_dir}")
        logger.info(f"  Backend: {get_settings().tkg_backend}")

    # ------------------------------------------------------------------
    # Active backend helpers
    # ------------------------------------------------------------------

    @property
    def _backend(self) -> str:
        """Return the active TKG backend from settings."""
        return get_settings().tkg_backend

    @property
    def _production_model_path(self) -> Path:
        """Path to the production model checkpoint for the active backend."""
        if self._backend == "tirgn":
            return self.model_dir / "tirgn_best.npz"
        return self.model_dir / "regcn_trained.pt"

    def _load_config(self) -> dict[str, Any]:
        """
        Load configuration from YAML file.

        Returns:
            Configuration dictionary

        Raises:
            FileNotFoundError: If config file doesn't exist
        """
        if not self.config_path.exists():
            logger.warning(f"Config file not found: {self.config_path}, using defaults")
            return self._default_config()

        with open(self.config_path) as f:
            config = yaml.safe_load(f)

        return config

    def _default_config(self) -> dict[str, Any]:
        """Return default configuration."""
        return {
            "schedule": {
                "frequency": "weekly",
                "day_of_week": 0,  # Sunday
                "day_of_month": 1,
                "hour": 2,
            },
            "data": {
                "data_window": 30,
                "max_events": 1000000,
            },
            "model": {
                "epochs": 50,
                "batch_size": 1024,
                "embedding_dim": 200,
                "num_layers": 2,
                "learning_rate": 0.001,
            },
            "model_tirgn": {
                "epochs": 100,
                "batch_size": 1024,
                "embedding_dim": 200,
                "num_layers": 2,
                "learning_rate": 0.001,
                "history_rate": 0.3,
                "history_window": 50,
                "patience": 15,
                "logdir": "runs/tirgn_retrain",
            },
            "versioning": {
                "backup_count": 3,
                "model_dir": "models/tkg",
                "log_dir": "logs/retraining",
            },
            "validation": {
                "min_improvement": 0.0,
                "validation_samples": 1000,
            },
        }

    def get_last_trained_time(self) -> Optional[datetime]:
        """
        Get timestamp of last successful training.

        Returns:
            datetime of last training, or None if never trained
        """
        if not self.last_trained_path.exists():
            return None

        with open(self.last_trained_path) as f:
            data = json.load(f)

        timestamp_str = data.get("last_trained")
        if timestamp_str:
            return datetime.fromisoformat(timestamp_str)
        return None

    def _set_last_trained_time(self, timestamp: datetime) -> None:
        """
        Record timestamp of successful training.

        Includes ``model_type`` so downstream tooling knows which backend
        was used for the most recent training run.

        Args:
            timestamp: Training completion time
        """
        data = {
            "last_trained": timestamp.isoformat(),
            "model_type": self._backend,
            "model_path": str(self._production_model_path),
        }

        with open(self.last_trained_path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Updated last trained time: {timestamp.isoformat()}")

    def should_retrain(self, now: Optional[datetime] = None) -> bool:
        """
        Check if retraining is due based on schedule.

        Args:
            now: Current timestamp (defaults to datetime.now())

        Returns:
            True if retraining should occur
        """
        if now is None:
            now = datetime.now()

        last_trained = self.get_last_trained_time()

        # If never trained, retraining is needed
        if last_trained is None:
            logger.info("No previous training found, retraining needed")
            return True

        schedule = self.config["schedule"]
        frequency = schedule["frequency"]

        if frequency == "weekly":
            # Check if it's past the weekly retraining time
            days_since = (now - last_trained).days
            if days_since >= 7:
                logger.info(f"Weekly retraining due: {days_since} days since last training")
                return True

            # Check if we're on the scheduled day and past the scheduled hour
            target_day = schedule["day_of_week"]
            target_hour = schedule["hour"]

            if now.weekday() == target_day and now.hour >= target_hour:
                # Check if last training was before this week's scheduled time
                this_weeks_target = now.replace(hour=target_hour, minute=0, second=0, microsecond=0)
                if last_trained < this_weeks_target:
                    logger.info("Weekly retraining due: scheduled time reached")
                    return True

        elif frequency == "monthly":
            # Check if it's past the monthly retraining time
            days_since = (now - last_trained).days
            if days_since >= 28:  # At least 4 weeks
                logger.info(f"Monthly retraining due: {days_since} days since last training")
                return True

            # Check if we're on the scheduled day and past the scheduled hour
            target_day = schedule["day_of_month"]
            target_hour = schedule["hour"]

            if now.day == target_day and now.hour >= target_hour:
                # Check if last training was before this month's scheduled time
                this_months_target = now.replace(day=target_day, hour=target_hour, minute=0, second=0, microsecond=0)
                if last_trained < this_months_target:
                    logger.info("Monthly retraining due: scheduled time reached")
                    return True

        logger.debug(f"Retraining not due (last trained: {last_trained})")
        return False

    def get_next_retrain_time(self, now: Optional[datetime] = None) -> datetime:
        """
        Calculate when the next retraining will occur.

        Args:
            now: Current timestamp (defaults to datetime.now())

        Returns:
            datetime of next scheduled retraining
        """
        if now is None:
            now = datetime.now()

        schedule = self.config["schedule"]
        frequency = schedule["frequency"]
        target_hour = schedule["hour"]

        if frequency == "weekly":
            target_day = schedule["day_of_week"]

            # Find the next occurrence of target day
            days_ahead = target_day - now.weekday()
            if days_ahead < 0:  # Target day already passed this week
                days_ahead += 7
            elif days_ahead == 0 and now.hour >= target_hour:
                # Today is target day but hour passed
                days_ahead = 7

            next_date = now + timedelta(days=days_ahead)
            return next_date.replace(hour=target_hour, minute=0, second=0, microsecond=0)

        elif frequency == "monthly":
            target_day = schedule["day_of_month"]

            # Start from current month
            next_date = now.replace(day=target_day, hour=target_hour, minute=0, second=0, microsecond=0)

            if next_date <= now:
                # Move to next month
                if now.month == 12:
                    next_date = next_date.replace(year=now.year + 1, month=1)
                else:
                    next_date = next_date.replace(month=now.month + 1)

            return next_date

        # Fallback (shouldn't reach here with valid config)
        return now + timedelta(days=7)

    def _backup_current_model(self) -> Optional[Path]:
        """
        Backup current model before retraining.

        Backend-aware: backs up the correct file(s) depending on whether
        RE-GCN (.pt) or TiRGN (.npz + .json) is active.

        Returns:
            Path to backup file, or None if no model exists
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if self._backend == "tirgn":
            current_npz = self.model_dir / "tirgn_best.npz"
            current_json = self.model_dir / "tirgn_best.json"
            if not current_npz.exists():
                logger.info("No existing TiRGN model to backup")
                return None

            backup_npz = self.model_dir / f"tirgn_backup_{timestamp}.npz"
            backup_json = self.model_dir / f"tirgn_backup_{timestamp}.json"
            shutil.copy2(current_npz, backup_npz)
            if current_json.exists():
                shutil.copy2(current_json, backup_json)
            logger.info(f"Backed up TiRGN model to {backup_npz}")

            self._cleanup_old_backups()
            return backup_npz
        else:
            current_model = self.model_dir / "regcn_trained.pt"
            if not current_model.exists():
                logger.info("No existing model to backup")
                return None

            backup_path = self.model_dir / f"regcn_backup_{timestamp}.pt"
            shutil.copy2(current_model, backup_path)
            logger.info(f"Backed up model to {backup_path}")

            self._cleanup_old_backups()
            return backup_path

    def _cleanup_old_backups(self) -> None:
        """Remove old backup files exceeding backup_count.

        Uses backend-appropriate glob patterns.
        """
        backup_count = self.config.get("versioning", {}).get("backup_count", 3)

        if self._backend == "tirgn":
            # Clean up .npz backups
            npz_backups = sorted(
                self.model_dir.glob("tirgn_backup_*.npz"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            for old in npz_backups[backup_count:]:
                logger.info(f"Removing old backup: {old}")
                old.unlink()
                # Also remove companion .json if it exists
                json_companion = old.with_suffix(".json")
                if json_companion.exists():
                    json_companion.unlink()
        else:
            backups = sorted(
                self.model_dir.glob("regcn_backup_*.pt"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            for old_backup in backups[backup_count:]:
                logger.info(f"Removing old backup: {old_backup}")
                old_backup.unlink()

    def _collect_fresh_data(self) -> Path:
        """
        Collect fresh GDELT data for retraining.

        Returns:
            Path to processed data file

        Raises:
            RuntimeError: If data collection fails
        """
        from src.training.data_collector import GDELTHistoricalCollector

        data_config = self.config.get("data", {})
        data_window = data_config.get("data_window", 30)

        logger.info(f"Collecting {data_window} days of fresh GDELT data")

        collector = GDELTHistoricalCollector()
        df = collector.collect_last_n_days(n_days=data_window)

        if df.empty:
            raise RuntimeError("Data collection returned no events")

        # Process and save to parquet
        output_path = Path("data/gdelt/processed/events.parquet")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Apply same processing as collect_training_data.py
        from scripts.collect_training_data import process_events_for_tkg

        processed_df = process_events_for_tkg(df)
        processed_df.to_parquet(output_path, index=False)

        logger.info(f"Saved {len(processed_df)} processed events to {output_path}")
        return output_path

    def _train_new_model(self, data_path: Path) -> dict[str, Any]:
        """
        Train a new model on the collected data.

        Dispatches to ``train_tirgn`` or ``train_regcn`` depending on
        ``Settings.tkg_backend``.

        Args:
            data_path: Path to processed data

        Returns:
            Training result dictionary with metrics
        """
        data_config = self.config.get("data", {})

        if self._backend == "tirgn":
            return self._train_tirgn(data_path, data_config)
        else:
            return self._train_regcn(data_path, data_config)

    def _train_regcn(self, data_path: Path, data_config: dict[str, Any]) -> dict[str, Any]:
        """Train RE-GCN model (original behaviour)."""
        from scripts.train_tkg import train_regcn

        model_config = self.config.get("model", {})

        logger.info("Starting RE-GCN model training")

        result = train_regcn(
            epochs=model_config.get("epochs", 50),
            learning_rate=model_config.get("learning_rate", 0.001),
            batch_size=model_config.get("batch_size", 1024),
            embedding_dim=model_config.get("embedding_dim", 200),
            num_layers=model_config.get("num_layers", 2),
            max_events=data_config.get("max_events", 1000000),
            num_days=data_config.get("data_window", 30),
            dry_run=False,
        )

        return result

    def _train_tirgn(self, data_path: Path, data_config: dict[str, Any]) -> dict[str, Any]:
        """Train TiRGN model using the TiRGN training pipeline."""
        from src.training.train_tirgn import TiRGNTrainingConfig, train_tirgn

        tirgn_config = self.config.get("model_tirgn", {})

        logger.info("Starting TiRGN model training")

        config = TiRGNTrainingConfig(
            epochs=tirgn_config.get("epochs", 100),
            learning_rate=tirgn_config.get("learning_rate", 0.001),
            batch_size=tirgn_config.get("batch_size", 1024),
            history_rate=tirgn_config.get("history_rate", 0.3),
            history_window=tirgn_config.get("history_window", 50),
            patience=tirgn_config.get("patience", 15),
            logdir=tirgn_config.get("logdir", "runs/tirgn_retrain"),
        )

        result = train_tirgn(
            data_path=data_path,
            config=config,
            model_dir=self.model_dir,
            max_events=data_config.get("max_events", 1000000),
            num_days=data_config.get("data_window", 30),
            embedding_dim=tirgn_config.get("embedding_dim", 200),
            num_layers=tirgn_config.get("num_layers", 2),
        )

        return result

    def _validate_new_model(self, new_model_path: Path, old_model_path: Optional[Path]) -> bool:
        """
        Validate that new model meets quality threshold.

        Backend-aware: loads ``.json`` metadata for TiRGN or ``.pt`` via
        torch for RE-GCN.

        Args:
            new_model_path: Path to newly trained model
            old_model_path: Path to previous model (for comparison)

        Returns:
            True if new model should be deployed
        """
        validation_config = self.config.get("validation", {})
        min_improvement = validation_config.get("min_improvement", 0.0)

        if min_improvement <= 0.0:
            logger.info("No minimum improvement threshold, accepting new model")
            return True

        if old_model_path is None or not old_model_path.exists():
            logger.info("No previous model for comparison, accepting new model")
            return True

        if self._backend == "tirgn":
            return self._validate_tirgn_model(new_model_path, old_model_path, min_improvement)
        else:
            return self._validate_regcn_model(new_model_path, old_model_path, min_improvement)

    def _validate_regcn_model(
        self, new_model_path: Path, old_model_path: Path, min_improvement: float
    ) -> bool:
        """Validate RE-GCN model using torch checkpoint."""
        import torch

        new_checkpoint = torch.load(new_model_path, map_location="cpu", weights_only=False)
        old_checkpoint = torch.load(old_model_path, map_location="cpu", weights_only=False)

        new_mrr = new_checkpoint.get("metrics", {}).get("mrr", 0.0)
        old_mrr = old_checkpoint.get("metrics", {}).get("mrr", 0.0)

        improvement = new_mrr - old_mrr
        logger.info(f"Model comparison: old MRR={old_mrr:.4f}, new MRR={new_mrr:.4f}, improvement={improvement:.4f}")

        if improvement >= min_improvement:
            logger.info(f"New model meets improvement threshold ({improvement:.4f} >= {min_improvement:.4f})")
            return True
        else:
            logger.warning(f"New model below improvement threshold ({improvement:.4f} < {min_improvement:.4f})")
            return False

    def _validate_tirgn_model(
        self, new_model_path: Path, old_model_path: Path, min_improvement: float
    ) -> bool:
        """Validate TiRGN model using JSON metadata sidecar."""
        new_json = new_model_path.with_suffix(".json")
        old_json = old_model_path.with_suffix(".json")

        if not new_json.exists() or not old_json.exists():
            logger.info("Missing JSON metadata for TiRGN comparison, accepting new model")
            return True

        with open(new_json) as f:
            new_meta = json.load(f)
        with open(old_json) as f:
            old_meta = json.load(f)

        new_mrr = new_meta.get("metrics", {}).get("mrr", 0.0)
        old_mrr = old_meta.get("metrics", {}).get("mrr", 0.0)

        improvement = new_mrr - old_mrr
        logger.info(f"TiRGN comparison: old MRR={old_mrr:.4f}, new MRR={new_mrr:.4f}, improvement={improvement:.4f}")

        if improvement >= min_improvement:
            logger.info(f"New TiRGN meets improvement threshold ({improvement:.4f} >= {min_improvement:.4f})")
            return True
        else:
            logger.warning(f"New TiRGN below improvement threshold ({improvement:.4f} < {min_improvement:.4f})")
            return False

    def retrain(self, dry_run: bool = False, skip_data_collection: bool = False) -> dict[str, Any]:
        """
        Execute full retraining pipeline.

        Steps:
        1. Backup existing model
        2. Collect fresh GDELT data (unless skipped)
        3. Train new model (RE-GCN or TiRGN per Settings.tkg_backend)
        4. Validate against baseline
        5. Replace production model if validation passes
        6. Update last trained timestamp

        Args:
            dry_run: If True, validate pipeline without actual training
            skip_data_collection: If True, use existing data

        Returns:
            Dictionary with retraining results:
            - status: "success", "skipped", "failed"
            - metrics: Training metrics if successful
            - reason: Description of outcome
        """
        start_time = datetime.now()
        logger.info("=" * 60)
        logger.info("Starting TKG Retraining Pipeline")
        logger.info("=" * 60)
        logger.info(f"Started: {start_time.isoformat()}")
        logger.info(f"Backend: {self._backend}")

        if dry_run:
            logger.info("DRY RUN - simulating retraining pipeline")

            # Validate configuration
            schedule = self.config["schedule"]
            logger.info(f"Schedule: {schedule['frequency']}")
            logger.info(f"Next retrain: {self.get_next_retrain_time()}")
            logger.info(f"Last trained: {self.get_last_trained_time()}")
            logger.info(f"Should retrain: {self.should_retrain()}")

            return {
                "status": "dry_run",
                "schedule": schedule,
                "next_retrain": self.get_next_retrain_time().isoformat(),
                "last_trained": self.get_last_trained_time().isoformat() if self.get_last_trained_time() else None,
                "should_retrain": self.should_retrain(),
            }

        try:
            # Step 1: Backup current model
            logger.info("Step 1: Backing up current model")
            backup_path = self._backup_current_model()

            # Step 2: Collect fresh data
            if skip_data_collection:
                logger.info("Step 2: Using existing data (skipped collection)")
                data_path = Path("data/gdelt/processed/events.parquet")
                if not data_path.exists():
                    raise RuntimeError(f"Data file not found: {data_path}")
            else:
                logger.info("Step 2: Collecting fresh GDELT data")
                data_path = self._collect_fresh_data()

            # Step 3: Train new model
            logger.info("Step 3: Training new model (%s)", self._backend)
            train_result = self._train_new_model(data_path)

            if train_result.get("status") != "complete":
                raise RuntimeError(f"Training failed: {train_result}")

            # Step 4: Validate new model
            logger.info("Step 4: Validating new model")
            new_model_path = self._production_model_path
            if self._validate_new_model(new_model_path, backup_path):
                logger.info("Validation passed - new model deployed")
                deploy_status = "deployed"
            else:
                logger.warning("Validation failed - keeping previous model")
                if backup_path and backup_path.exists():
                    shutil.copy2(backup_path, new_model_path)
                    if self._backend == "tirgn":
                        # Also restore the JSON sidecar
                        backup_json = backup_path.with_suffix(".json")
                        prod_json = new_model_path.with_suffix(".json")
                        if backup_json.exists():
                            shutil.copy2(backup_json, prod_json)
                deploy_status = "rolled_back"

            # Step 5: Update timestamp
            logger.info("Step 5: Updating training timestamp")
            self._set_last_trained_time(datetime.now())

            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            result = {
                "status": "success",
                "deploy_status": deploy_status,
                "started": start_time.isoformat(),
                "completed": end_time.isoformat(),
                "duration_seconds": duration,
                "metrics": train_result.get("metrics", {}),
                "model_path": str(new_model_path),
                "backend": self._backend,
            }

            # Log to file
            self._log_retraining_result(result)

            logger.info("=" * 60)
            logger.info(f"Retraining Complete: {deploy_status}")
            logger.info(f"Duration: {duration:.1f}s")
            logger.info("=" * 60)

            return result

        except Exception as e:
            logger.error(f"Retraining failed: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "started": start_time.isoformat(),
            }

    def _log_retraining_result(self, result: dict[str, Any]) -> None:
        """Log retraining result to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = self.log_dir / f"retraining_{timestamp}.json"

        with open(log_path, "w") as f:
            json.dump(result, f, indent=2)

        logger.info(f"Logged result to {log_path}")
