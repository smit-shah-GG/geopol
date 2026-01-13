"""
Comprehensive evaluation orchestrator for geopolitical forecasting system.

The Evaluator coordinates all evaluation components:
- Brier scoring (resolved + provisional)
- Calibration metrics (ECE, MCE, ACE)
- Drift detection
- Human baseline comparison
- Performance reporting

It loads predictions from SQLite and generates comprehensive performance reports.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from .benchmark import Benchmark, HumanBaseline
from .brier_scorer import BrierScorer
from .calibration_metrics import CalibrationMetrics
from .drift_detector import DriftDetector
from .provisional_scorer import ProvisionalScorer

logger = logging.getLogger(__name__)


class Evaluator:
    """
    Orchestrates comprehensive evaluation of forecasting system.

    Combines all evaluation metrics into unified reports.
    """

    def __init__(
        self,
        prediction_store=None,
        gdelt_client=None,
        output_dir: Optional[str] = None,
    ):
        """
        Initialize evaluator.

        Args:
            prediction_store: PredictionStore instance for loading predictions
            gdelt_client: GDELTClient for provisional scoring
            output_dir: Directory for output files (plots, reports)
        """
        self.prediction_store = prediction_store
        self.gdelt_client = gdelt_client

        # Output directory
        if output_dir is None:
            output_dir = "./outputs"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Initialize components
        self.brier_scorer = BrierScorer()
        self.provisional_scorer = ProvisionalScorer(
            gdelt_client=gdelt_client
        )
        self.calibration_metrics = CalibrationMetrics(n_bins=10)
        self.drift_detector = DriftDetector(
            window_days=30,
            alert_threshold=0.15,
            warning_threshold=0.10,
        )
        self.benchmark = Benchmark()

        logger.info("Evaluator initialized")

    def load_predictions(
        self,
        category: Optional[str] = None,
        resolved_only: bool = False,
        min_date: Optional[datetime] = None,
    ) -> List[Dict]:
        """
        Load predictions from prediction store.

        Args:
            category: Filter by category (optional)
            resolved_only: Only load resolved predictions
            min_date: Minimum timestamp

        Returns:
            List of prediction dicts
        """
        if self.prediction_store is None:
            raise ValueError("No prediction_store provided to Evaluator")

        predictions = self.prediction_store.get_predictions_for_calibration(
            category=category,
            resolved_only=resolved_only,
            min_date=min_date,
        )

        logger.info(f"Loaded {len(predictions)} predictions from store")

        return predictions

    def evaluate_current_performance(
        self,
        predictions: Optional[List[Dict]] = None,
        use_calibrated: bool = True,
        include_provisional: bool = True,
    ) -> Dict:
        """
        Evaluate current system performance.

        Args:
            predictions: List of predictions (loads from store if None)
            use_calibrated: Use calibrated vs raw probabilities
            include_provisional: Include provisional scoring for unresolved predictions

        Returns:
            Dict with complete evaluation results
        """
        # Load predictions if not provided
        if predictions is None:
            predictions = self.load_predictions()

        if not predictions:
            raise ValueError("No predictions available for evaluation")

        # Split resolved/unresolved
        resolved = [p for p in predictions if p.get("outcome") is not None]
        unresolved = [p for p in predictions if p.get("outcome") is None]

        logger.info(
            f"Evaluating {len(resolved)} resolved + {len(unresolved)} unresolved predictions"
        )

        # Calculate Brier scores
        if resolved:
            brier_result = self.brier_scorer.score_batch(resolved, use_calibrated)
            brier_detailed = self.brier_scorer.get_detailed_breakdown(
                resolved, use_calibrated
            )
        else:
            brier_result = None
            brier_detailed = None

        # Provisional scoring
        if include_provisional and unresolved:
            provisional_result = self.provisional_scorer.score_all(
                predictions, use_calibrated, provisional_weight=0.5
            )
        else:
            provisional_result = None

        # Calibration metrics (resolved only)
        if resolved and len(resolved) >= 10:
            calibration_result = self.calibration_metrics.calculate_metrics(
                resolved, use_calibrated
            )

            # Per-category calibration
            per_category_cal = self.calibration_metrics.calculate_per_category_metrics(
                resolved, use_calibrated
            )
        else:
            calibration_result = None
            per_category_cal = None

        # Drift detection
        if resolved and len(resolved) >= 10:
            drift_result = self.drift_detector.detect_drift(predictions, use_calibrated)

            # Record metrics
            if calibration_result:
                self.drift_detector.record_metrics(
                    ece=calibration_result["ece"],
                    mce=calibration_result["mce"],
                    ace=calibration_result["ace"],
                    n_predictions=len(resolved),
                )
        else:
            drift_result = None

        # Benchmark comparison
        if brier_result and calibration_result:
            benchmark_result = self.benchmark.benchmark_predictions(
                predictions=predictions,
                brier_score=brier_result["overall"],
                ece=calibration_result["ece"],
                use_calibrated=use_calibrated,
            )
        else:
            benchmark_result = None

        return {
            "timestamp": datetime.now().isoformat(),
            "brier_scores": brier_result,
            "brier_detailed": brier_detailed,
            "provisional_scores": provisional_result,
            "calibration_metrics": calibration_result,
            "per_category_calibration": per_category_cal,
            "drift_detection": drift_result,
            "benchmark": benchmark_result,
            "prediction_counts": {
                "total": len(predictions),
                "resolved": len(resolved),
                "unresolved": len(unresolved),
            },
            "methodology": "calibrated" if use_calibrated else "raw",
        }

    def generate_reliability_diagrams(
        self,
        predictions: Optional[List[Dict]] = None,
        use_calibrated: bool = True,
    ) -> Dict[str, str]:
        """
        Generate reliability diagrams for all categories.

        Args:
            predictions: List of predictions (loads if None)
            use_calibrated: Use calibrated vs raw probabilities

        Returns:
            Dict mapping category -> plot path
        """
        if predictions is None:
            predictions = self.load_predictions()

        resolved = [p for p in predictions if p.get("outcome") is not None]

        if not resolved:
            logger.warning("No resolved predictions for reliability diagrams")
            return {}

        # Generate diagrams
        plot_paths = self.calibration_metrics.generate_per_category_diagrams(
            resolved,
            use_calibrated=use_calibrated,
            output_dir=str(self.output_dir),
        )

        # Also generate overall diagram
        overall_path = self.calibration_metrics.generate_reliability_diagram(
            resolved,
            use_calibrated=use_calibrated,
            output_path=str(self.output_dir / "reliability_overall.png"),
            title="Overall",
        )

        plot_paths["overall"] = overall_path

        return plot_paths

    def generate_performance_report(
        self,
        predictions: Optional[List[Dict]] = None,
        use_calibrated: bool = True,
        output_format: str = "text",
    ) -> str:
        """
        Generate comprehensive performance report.

        Args:
            predictions: List of predictions (loads if None)
            use_calibrated: Use calibrated vs raw probabilities
            output_format: 'text', 'json', or 'html'

        Returns:
            Report as string (or path if HTML)
        """
        # Evaluate
        results = self.evaluate_current_performance(
            predictions, use_calibrated, include_provisional=True
        )

        if output_format == "json":
            return json.dumps(results, indent=2)

        elif output_format == "text":
            # Generate text summary
            if results["benchmark"]:
                summary = self.benchmark.generate_performance_summary(results["benchmark"])
            else:
                summary = "Insufficient data for benchmark report"

            # Add detailed metrics
            lines = [summary, "\n\n--- DETAILED METRICS ---\n"]

            if results["brier_scores"]:
                bs = results["brier_scores"]
                lines.append(f"Overall Brier Score: {bs['overall']:.4f}")
                lines.append(f"  Conflict: {bs['conflict'] if bs['conflict'] else 'N/A'}")
                lines.append(f"  Diplomatic: {bs['diplomatic'] if bs['diplomatic'] else 'N/A'}")
                lines.append(f"  Economic: {bs['economic'] if bs['economic'] else 'N/A'}")

            if results["calibration_metrics"]:
                cm = results["calibration_metrics"]
                lines.append(f"\nCalibration Metrics:")
                lines.append(f"  ECE: {cm['ece']:.4f}")
                lines.append(f"  MCE: {cm['mce']:.4f}")
                lines.append(f"  ACE: {cm['ace']:.4f}")

            if results["drift_detection"]:
                dd = results["drift_detection"]
                lines.append(f"\nDrift Detection:")
                lines.append(f"  Status: {'DRIFT DETECTED' if dd['drift_detected'] else 'OK'}")
                lines.append(f"  Current ECE: {dd['current_ece']:.4f}")
                lines.append(f"  {dd['recommendation']}")

            if results["provisional_scores"]:
                ps = results["provisional_scores"]
                lines.append(f"\nProvisional Scoring:")
                lines.append(f"  Combined Brier: {ps['combined_brier']:.4f}")
                lines.append(f"  Provisional predictions: {ps['n_provisional']}")

            return "\n".join(lines)

        elif output_format == "html":
            # Generate HTML report
            html_path = self.output_dir / f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"

            html_content = self._generate_html_report(results)

            with open(html_path, "w") as f:
                f.write(html_content)

            logger.info(f"Generated HTML report: {html_path}")

            return str(html_path)

        else:
            raise ValueError(f"Unknown output format: {output_format}")

    def _generate_html_report(self, results: Dict) -> str:
        """Generate HTML report from results."""
        # Simple HTML template
        benchmark = results.get("benchmark", {})
        brier = results.get("brier_scores", {})
        calibration = results.get("calibration_metrics", {})

        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Forecasting System Performance Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        h1 {{ color: #333; }}
        h2 {{ color: #666; margin-top: 30px; }}
        table {{ border-collapse: collapse; margin: 20px 0; }}
        td, th {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        .metric {{ font-size: 24px; font-weight: bold; }}
        .good {{ color: green; }}
        .warning {{ color: orange; }}
        .bad {{ color: red; }}
    </style>
</head>
<body>
    <h1>Geopolitical Forecasting System - Performance Report</h1>
    <p>Generated: {results['timestamp']}</p>

    <h2>Overall Performance</h2>
    <div class="metric">Brier Score: {brier.get('overall', 'N/A'):.4f}</div>
    <div class="metric">ECE: {calibration.get('ece', 'N/A'):.4f}</div>

    <h2>Human Baseline Comparison</h2>
    <table>
        <tr><th>Metric</th><th>Value</th></tr>
        <tr><td>Performance Level</td><td>{benchmark.get('human_comparison', {}).get('performance_level', 'N/A')}</td></tr>
        <tr><td>Percentile</td><td>{benchmark.get('human_comparison', {}).get('percentile', 'N/A'):.1f}th</td></tr>
        <tr><td>Beats Expert</td><td>{'Yes' if benchmark.get('human_comparison', {}).get('beats_expert') else 'No'}</td></tr>
    </table>

    <h2>Prediction Statistics</h2>
    <p>Total Predictions: {results['prediction_counts']['total']}</p>
    <p>Resolved: {results['prediction_counts']['resolved']}</p>
    <p>Unresolved: {results['prediction_counts']['unresolved']}</p>

    <h2>Calibration Quality</h2>
    <p>Calibration Level: {benchmark.get('calibration_quality', {}).get('calibration_level', 'N/A').upper()}</p>
    <p>Well Calibrated: {'Yes' if benchmark.get('calibration_quality', {}).get('is_well_calibrated') else 'No'}</p>

</body>
</html>
"""
        return html

    def get_performance_trend(
        self, window_days: int = 90
    ) -> Dict[str, List[float]]:
        """
        Get performance trend over time.

        Args:
            window_days: Window for trend analysis

        Returns:
            Dict with time series of metrics
        """
        trend_stats = self.drift_detector.get_trend_statistics()

        return {
            "ece_trend": trend_stats,
            "recent_metrics": self.drift_detector.get_recent_metrics(n=20),
        }
