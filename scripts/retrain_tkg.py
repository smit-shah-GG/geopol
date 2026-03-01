#!/usr/bin/env python
"""
Automated TKG retraining script.

This script is designed to be called by cron/systemd for periodic retraining:
1. Checks if retraining is due based on schedule
2. If yes: collects fresh 30-day window of GDELT data
3. Trains new model with configured hyperparameters
4. Compares performance to current model
5. Replaces if better, keeps if not
6. Logs all metrics and decisions

Usage:
    uv run python scripts/retrain_tkg.py              # Normal execution (uses TKG_BACKEND env)
    uv run python scripts/retrain_tkg.py --backend tirgn  # Override backend for this run
    uv run python scripts/retrain_tkg.py --dry-run    # Verify pipeline without training
    uv run python scripts/retrain_tkg.py --force      # Force retraining even if not scheduled
    uv run python scripts/retrain_tkg.py --check-schedule  # Only check schedule, don't retrain
"""

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.scheduler import RetrainingScheduler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def setup_file_logging(log_dir: Path) -> None:
    """Add file handler for persistent logging."""
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"retrain_{timestamp}.log"

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    )

    logging.getLogger().addHandler(file_handler)
    logger.info(f"Logging to {log_file}")


def check_schedule(scheduler: RetrainingScheduler) -> None:
    """Display schedule information and exit."""
    print("\n=== TKG Retraining Schedule ===\n")

    config = scheduler.config
    schedule = config.get("schedule", {})

    print(f"Frequency:       {schedule.get('frequency', 'weekly')}")

    if schedule.get("frequency") == "weekly":
        days = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
        day_name = days[schedule.get("day_of_week", 0)]
        print(f"Day:             {day_name}")
    else:
        print(f"Day of month:    {schedule.get('day_of_month', 1)}")

    print(f"Hour:            {schedule.get('hour', 2):02d}:00")
    print()

    last_trained = scheduler.get_last_trained_time()
    if last_trained:
        print(f"Last trained:    {last_trained.isoformat()}")
        days_ago = (datetime.now() - last_trained).days
        print(f"                 ({days_ago} days ago)")
    else:
        print("Last trained:    Never")

    print()
    print(f"Next scheduled:  {scheduler.get_next_retrain_time().isoformat()}")
    print(f"Should retrain:  {'Yes' if scheduler.should_retrain() else 'No'}")
    print()


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Automated TKG retraining script",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Verify pipeline without actual training",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force retraining even if not scheduled",
    )
    parser.add_argument(
        "--check-schedule",
        action="store_true",
        help="Only check schedule, don't retrain",
    )
    parser.add_argument(
        "--skip-data-collection",
        action="store_true",
        help="Use existing data instead of collecting fresh",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["tirgn", "regcn"],
        default=None,
        help="Override TKG_BACKEND for this run (default: from env/settings)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/retraining.yaml",
        help="Path to configuration file",
    )

    args = parser.parse_args()

    # Apply backend override via environment variable before settings are loaded
    if args.backend is not None:
        os.environ["TKG_BACKEND"] = args.backend
        # Reset settings singleton so it picks up the new envvar
        import src.settings
        src.settings._settings = None
        logger.info("Backend override: %s", args.backend)

    # Initialize scheduler
    try:
        scheduler = RetrainingScheduler(config_path=Path(args.config))
    except Exception as e:
        logger.error(f"Failed to initialize scheduler: {e}")
        return 1

    # Check schedule mode
    if args.check_schedule:
        check_schedule(scheduler)
        return 0

    # Dry run mode
    if args.dry_run:
        logger.info("DRY RUN MODE - Validating pipeline")
        result = scheduler.retrain(dry_run=True)
        print("\n=== Dry Run Result ===")
        for key, value in result.items():
            print(f"  {key}: {value}")
        return 0

    # Check if retraining is due (unless forced)
    if not args.force and not scheduler.should_retrain():
        logger.info("Retraining not due, exiting")
        logger.info(f"Next scheduled: {scheduler.get_next_retrain_time()}")
        return 0

    # Set up file logging for the actual retraining run
    setup_file_logging(scheduler.log_dir)

    # Execute retraining
    from src.settings import get_settings
    backend = get_settings().tkg_backend
    logger.info("=" * 70)
    logger.info("TKG Automated Retraining (backend: %s)", backend)
    logger.info("=" * 70)
    logger.info(f"Started: {datetime.now().isoformat()}")

    if args.force:
        logger.info("Forced retraining (ignoring schedule)")

    result = scheduler.retrain(
        dry_run=False,
        skip_data_collection=args.skip_data_collection,
    )

    # Report result
    status = result.get("status", "unknown")
    logger.info(f"Retraining status: {status}")

    if status == "success":
        logger.info(f"Deploy status: {result.get('deploy_status')}")
        logger.info(f"Duration: {result.get('duration_seconds', 0):.1f}s")
        metrics = result.get("metrics", {})
        if metrics:
            logger.info(f"Final MRR: {metrics.get('mrr', 'N/A')}")
        return 0
    elif status == "failed":
        logger.error(f"Retraining failed: {result.get('error')}")
        return 1
    else:
        logger.warning(f"Unexpected status: {status}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
