#!/usr/bin/env python
"""
Main entry point for GDELT data processing pipeline.

Usage:
    python run_pipeline.py --date 2026-01-09
    python run_pipeline.py --timespan 7d
    python run_pipeline.py --date 2026-01-09 --max-events 5000
"""

import sys
import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.pipeline import GDELTPipeline
from src.gdelt_client import GDELTClient
from src.database import EventStorage
from src.monitoring import DataQualityMonitor

# Configure logging
def setup_logging(log_level: str = "INFO"):
    """Configure logging for the application."""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    # Setup file logging
    log_file = log_dir / f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    # Reduce verbosity of some libraries
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('src.deduplication').setLevel(logging.INFO)  # Hide timestamp parsing debug messages

    return logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run GDELT data processing pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Process today's events:
    python run_pipeline.py

  Process specific date:
    python run_pipeline.py --date 2026-01-09

  Process last 7 days:
    python run_pipeline.py --timespan 7d

  Process with custom limits:
    python run_pipeline.py --max-events 5000 --min-mentions 50

  Dry run (no database storage):
    python run_pipeline.py --dry-run

  Debug mode:
    python run_pipeline.py --log-level DEBUG
        """
    )

    # Time range options
    time_group = parser.add_mutually_exclusive_group()
    time_group.add_argument(
        '--date',
        type=str,
        help='Specific date to process (YYYY-MM-DD)'
    )
    time_group.add_argument(
        '--timespan',
        type=str,
        default='24h',
        help='Time window to process (e.g., 24h, 7d, 1w)'
    )

    # Filtering options
    parser.add_argument(
        '--quadclasses',
        type=int,
        nargs='+',
        default=[1, 4],
        choices=[1, 2, 3, 4],
        help='QuadClass values to keep (1=Verbal Coop, 2=Material Coop, 3=Verbal Conflict, 4=Material Conflict)'
    )
    parser.add_argument(
        '--min-mentions',
        type=int,
        default=100,
        help='Minimum mentions for high-confidence filtering (GDELT100)'
    )

    # Sampling options
    parser.add_argument(
        '--max-events',
        type=int,
        default=10000,
        help='Maximum events to process (sampling applied if exceeded)'
    )
    parser.add_argument(
        '--no-sampling',
        action='store_true',
        help='Disable sampling (process all events)'
    )

    # Processing options
    parser.add_argument(
        '--no-filtering',
        action='store_true',
        help='Disable filtering stage'
    )
    parser.add_argument(
        '--no-deduplication',
        action='store_true',
        help='Disable deduplication stage'
    )

    # Execution options
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Run pipeline without storing events in database'
    )
    parser.add_argument(
        '--max-retries',
        type=int,
        default=3,
        help='Maximum retry attempts on failure'
    )

    # Logging options
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging verbosity level'
    )

    # Database options
    parser.add_argument(
        '--db-path',
        type=str,
        default='data/events.db',
        help='Path to SQLite database file'
    )

    return parser.parse_args()


def main():
    """Main pipeline execution."""
    args = parse_arguments()

    # Setup logging
    logger = setup_logging(args.log_level)

    logger.info("=" * 70)
    logger.info("GDELT Data Processing Pipeline")
    logger.info("=" * 70)
    logger.info(f"Configuration:")
    logger.info(f"  Date/Timespan: {args.date or args.timespan}")
    logger.info(f"  QuadClasses: {args.quadclasses}")
    logger.info(f"  Min Mentions: {args.min_mentions}")
    logger.info(f"  Max Events: {args.max_events}")
    logger.info(f"  Database: {args.db_path}")
    logger.info(f"  Dry Run: {args.dry_run}")
    logger.info("-" * 70)

    try:
        # Initialize components
        logger.info("Initializing pipeline components...")

        client = GDELTClient()
        storage = EventStorage(db_path=args.db_path)
        monitor = DataQualityMonitor()

        # Create pipeline
        pipeline = GDELTPipeline(
            client=client,
            storage=storage,
            monitor=monitor
        )

        # Configure pipeline
        pipeline.configure(
            use_filtering=not args.no_filtering,
            use_deduplication=not args.no_deduplication,
            use_sampling=not args.no_sampling,
            max_events=args.max_events,
            min_mentions=args.min_mentions,
            quadclasses=args.quadclasses,
        )

        # Run pipeline
        logger.info("Starting pipeline execution...")

        if args.max_retries > 1:
            results = pipeline.run_with_recovery(
                date_str=args.date,
                timespan=args.timespan,
                max_retries=args.max_retries
            )
        else:
            results = pipeline.run(
                date_str=args.date,
                timespan=args.timespan,
                dry_run=args.dry_run
            )

        # Display results
        logger.info("\n" + "=" * 70)
        logger.info("Pipeline Results")
        logger.info("=" * 70)

        if results['success']:
            logger.info("✓ Pipeline completed successfully!")
            logger.info(f"  Events processed: {results['events_processed']:,}")
            logger.info(f"  Events stored: {results['events_stored']:,}")

            if 'overall_reduction' in results:
                logger.info(f"  Overall reduction: {results['overall_reduction']:.1%}")
            if 'duplicate_rate' in results:
                logger.info(f"  Duplicate rate: {results['duplicate_rate']:.1%}")
            if 'high_confidence_rate' in results:
                logger.info(f"  High confidence rate: {results['high_confidence_rate']:.1%}")
            if 'estimated_accuracy' in results:
                logger.info(f"  Estimated accuracy: {results['estimated_accuracy']:.1%}")
            if 'total_time' in results:
                logger.info(f"  Processing time: {results['total_time']:.1f}s")

            # Get database statistics
            db_stats = storage.get_statistics()
            logger.info(f"\nDatabase Statistics:")
            logger.info(f"  Total events: {db_stats['total_events']:,}")
            if db_stats.get('date_range'):
                logger.info(f"  Date range: {db_stats['date_range']['earliest']} to {db_stats['date_range']['latest']}")
            if db_stats.get('high_confidence_events'):
                logger.info(f"  High confidence events: {db_stats['high_confidence_events']:,}")

        else:
            logger.error("✗ Pipeline failed!")
            logger.error(f"  Events processed: {results.get('events_processed', 0)}")
            if results.get('errors'):
                logger.error("  Errors:")
                for error in results['errors']:
                    logger.error(f"    - {error}")

            sys.exit(1)

    except KeyboardInterrupt:
        logger.info("\nPipeline interrupted by user")
        sys.exit(130)

    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()