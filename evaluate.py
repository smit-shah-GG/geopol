#!/usr/bin/env python
"""
Evaluation CLI for geopolitical forecasting system.

Commands:
- score: Calculate current performance metrics
- trend: Show performance trend over time
- calibrate: Trigger recalibration if drift detected
- report: Generate comprehensive HTML report
- diagrams: Generate reliability diagrams

Usage:
    python evaluate.py score
    python evaluate.py report --format html
    python evaluate.py trend --window 90
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from evaluation.evaluator import Evaluator
from calibration.prediction_store import PredictionStore


def cmd_score(args):
    """Calculate and display current performance metrics."""
    print("Loading predictions and calculating metrics...")

    # Initialize
    store = PredictionStore(args.db_path)
    evaluator = Evaluator(prediction_store=store)

    # Evaluate
    results = evaluator.evaluate_current_performance(
        use_calibrated=not args.raw,
        include_provisional=not args.no_provisional,
    )

    # Display
    if args.json:
        import json

        print(json.dumps(results, indent=2))
    else:
        report = evaluator.generate_performance_report(
            output_format="text"
        )
        print(report)


def cmd_trend(args):
    """Show performance trend over time."""
    print(f"Analyzing performance trend (window: {args.window} days)...")

    store = PredictionStore(args.db_path)
    evaluator = Evaluator(prediction_store=store)

    trend = evaluator.get_performance_trend(window_days=args.window)

    # Display trend statistics
    if trend["ece_trend"]:
        stats = trend["ece_trend"]
        print("\n--- ECE Trend Statistics ---")
        print(f"Mean ECE: {stats['mean_ece']:.4f}")
        print(f"Median ECE: {stats['median_ece']:.4f}")
        print(f"Std Dev: {stats['std_ece']:.4f}")
        print(f"Min ECE (best): {stats['min_ece']:.4f}")
        print(f"Max ECE (worst): {stats['max_ece']:.4f}")

        if "trend" in stats:
            if stats["trend"] < 0:
                print(f"✓ IMPROVING: Trend coefficient = {stats['trend']:.6f}")
            else:
                print(f"✗ DEGRADING: Trend coefficient = {stats['trend']:.6f}")
    else:
        print("Insufficient history for trend analysis")

    # Show recent metrics
    if trend["recent_metrics"]:
        print("\n--- Recent Metrics (last 10 evaluations) ---")
        for entry in trend["recent_metrics"][-10:]:
            print(
                f"{entry['timestamp']}: ECE={entry['ece']:.4f}, "
                f"MCE={entry['mce']:.4f}, n={entry['n_predictions']}"
            )


def cmd_calibrate(args):
    """Check for drift and trigger recalibration if needed."""
    print("Checking for calibration drift...")

    store = PredictionStore(args.db_path)
    evaluator = Evaluator(prediction_store=store)

    # Load predictions
    predictions = evaluator.load_predictions()

    # Detect drift
    drift_result = evaluator.drift_detector.detect_drift(
        predictions, use_calibrated=not args.raw
    )

    print(f"\n{drift_result['recommendation']}")

    if drift_result["drift_detected"]:
        print("\n⚠️  RECALIBRATION REQUIRED")
        print("Run isotonic recalibration to restore performance.")
        print("\nCommand:")
        print("  from calibration.isotonic_calibrator import IsotonicCalibrator")
        print("  calibrator = IsotonicCalibrator(prediction_store)")
        print("  calibrator.recalibrate()")

        return 1  # Exit code 1 to indicate action needed
    elif drift_result.get("warning"):
        print("\n⚠️  WARNING: Monitor calibration closely")
        return 0
    else:
        print("\n✓ Calibration is healthy")
        return 0


def cmd_report(args):
    """Generate comprehensive performance report."""
    print(f"Generating {args.format} report...")

    store = PredictionStore(args.db_path)
    evaluator = Evaluator(prediction_store=store, output_dir=args.output_dir)

    # Generate report
    report = evaluator.generate_performance_report(
        use_calibrated=not args.raw,
        output_format=args.format,
    )

    if args.format == "html":
        print(f"\n✓ Report saved to: {report}")
    else:
        print(report)


def cmd_diagrams(args):
    """Generate reliability diagrams."""
    print("Generating reliability diagrams...")

    store = PredictionStore(args.db_path)
    evaluator = Evaluator(prediction_store=store, output_dir=args.output_dir)

    # Generate diagrams
    plot_paths = evaluator.generate_reliability_diagrams(
        use_calibrated=not args.raw
    )

    print("\n✓ Reliability diagrams generated:")
    for category, path in plot_paths.items():
        if path:
            print(f"  {category}: {path}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluation CLI for geopolitical forecasting system"
    )

    # Global arguments
    parser.add_argument(
        "--db-path",
        default="./data/predictions.db",
        help="Path to predictions database (default: ./data/predictions.db)",
    )

    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # score command
    score_parser = subparsers.add_parser("score", help="Calculate current performance metrics")
    score_parser.add_argument("--raw", action="store_true", help="Use raw probabilities")
    score_parser.add_argument(
        "--no-provisional", action="store_true", help="Exclude provisional scoring"
    )
    score_parser.add_argument("--json", action="store_true", help="Output as JSON")

    # trend command
    trend_parser = subparsers.add_parser("trend", help="Show performance trend over time")
    trend_parser.add_argument(
        "--window", type=int, default=90, help="Window in days (default: 90)"
    )

    # calibrate command
    calibrate_parser = subparsers.add_parser(
        "calibrate", help="Check drift and trigger recalibration"
    )
    calibrate_parser.add_argument("--raw", action="store_true", help="Use raw probabilities")

    # report command
    report_parser = subparsers.add_parser("report", help="Generate comprehensive report")
    report_parser.add_argument(
        "--format",
        choices=["text", "json", "html"],
        default="text",
        help="Output format (default: text)",
    )
    report_parser.add_argument("--raw", action="store_true", help="Use raw probabilities")
    report_parser.add_argument(
        "--output-dir", default="./outputs", help="Output directory (default: ./outputs)"
    )

    # diagrams command
    diagrams_parser = subparsers.add_parser("diagrams", help="Generate reliability diagrams")
    diagrams_parser.add_argument("--raw", action="store_true", help="Use raw probabilities")
    diagrams_parser.add_argument(
        "--output-dir", default="./outputs", help="Output directory (default: ./outputs)"
    )

    args = parser.parse_args()

    # Execute command
    if not args.command:
        parser.print_help()
        return 1

    try:
        if args.command == "score":
            cmd_score(args)
        elif args.command == "trend":
            cmd_trend(args)
        elif args.command == "calibrate":
            return cmd_calibrate(args)
        elif args.command == "report":
            cmd_report(args)
        elif args.command == "diagrams":
            cmd_diagrams(args)
        else:
            parser.print_help()
            return 1

        return 0

    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
