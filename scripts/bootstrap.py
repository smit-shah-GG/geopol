#!/usr/bin/env python
"""
Bootstrap the geopolitical forecasting system from zero data to operational.

Usage:
    uv run python scripts/bootstrap.py [--force-stage STAGE] [--dry-run]

Stages:
    1. collect   - Fetch GDELT events (30 days)
    2. process   - Transform to TKG format + load into SQLite
    3. graph     - Build temporal knowledge graph
    4. persist   - Save graph to GraphML
    5. index     - Index patterns in RAG store
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.bootstrap import CheckpointManager, StageOrchestrator, ProgressReporter
from src.bootstrap.stages import (
    GDELTCollectStage,
    ProcessEventsStage,
    BuildGraphStage,
    PersistGraphStage,
    IndexRAGStage,
)


def setup_logging(verbose: bool = False) -> None:
    """Configure logging for bootstrap."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    # Reduce noise from third-party libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Bootstrap geopolitical forecasting system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--force-stage",
        type=str,
        help="Force re-run of specific stage (collect, process, graph, persist, index)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show stages that would run without executing",
    )
    parser.add_argument(
        "--n-days",
        type=int,
        default=30,
        help="Number of days to collect (default: 30)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--state-file",
        type=Path,
        default=Path("data/bootstrap_state.json"),
        help="Path to checkpoint state file",
    )
    return parser.parse_args()


def print_dry_run(plan: list) -> None:
    """Print dry run plan."""
    print("\n[BOOTSTRAP] Dry run - stages that would execute:\n")
    print(f"{'#':<4} {'Stage':<12} {'Action':<8} {'Reason'}")
    print("-" * 60)
    for i, stage in enumerate(plan, 1):
        print(f"{i:<4} {stage['name']:<12} {stage['action']:<8} {stage['reason']}")
    print()


def print_summary(summary: dict) -> None:
    """Print execution summary."""
    print("\n" + "=" * 60)
    print("[BOOTSTRAP] Execution Summary")
    print("=" * 60)
    print(f"Total stages:  {summary['total_stages']}")
    print(f"Completed:     {summary['completed']}")
    print(f"Skipped:       {summary['skipped']}")
    print(f"Failed:        {summary['failed']}")
    print(f"Duration:      {summary['duration_seconds']:.1f}s")

    if summary["failed"] == 0:
        print("\n[BOOTSTRAP] Outputs:")
        for stage in summary["stages"]:
            if stage["status"] == "completed":
                print(f"  - {stage['name']}: OK")


def main() -> int:
    """Main entry point."""
    args = parse_args()
    setup_logging(args.verbose)

    print("\n" + "=" * 60)
    print("[BOOTSTRAP] Geopolitical Forecasting System Bootstrap")
    print("=" * 60)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Days to collect: {args.n_days}")
    print()

    # Initialize orchestrator
    checkpoint = CheckpointManager(state_file=args.state_file)
    reporter = ProgressReporter(prefix="STAGE")
    orchestrator = StageOrchestrator(checkpoint_manager=checkpoint, reporter=reporter)

    # Register stages
    stages = [
        GDELTCollectStage(n_days=args.n_days),
        ProcessEventsStage(),
        BuildGraphStage(),
        PersistGraphStage(),
        IndexRAGStage(),
    ]

    for stage in stages:
        orchestrator.register_stage(stage)

    # Dry run mode
    if args.dry_run:
        plan = orchestrator.dry_run()
        print_dry_run(plan)
        return 0

    # Force specific stage if requested
    force_stages = []
    if args.force_stage:
        force_stages = [args.force_stage]
        print(f"[BOOTSTRAP] Forcing re-run of stage: {args.force_stage}")

    # Execute pipeline
    summary = orchestrator.run_all(force_stages=force_stages)

    # Print summary
    print_summary(summary)

    # Return non-zero on failure
    if summary["failed"] > 0:
        print("\n[BOOTSTRAP] Pipeline failed. Check logs for details.")
        return 1

    print("\n[BOOTSTRAP] Pipeline complete. System is operational.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
