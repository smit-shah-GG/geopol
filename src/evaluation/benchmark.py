"""
Human baseline comparison for geopolitical forecasting.

Benchmarks system performance against human forecasters:
- Expert forecasters: Professionals with domain knowledge (~0.35 Brier)
- Superforecasters: Top 2% of forecasters (~0.25 Brier)
- Good Judgment Project results: IARPA tournament data

Historical context:
- GJP superforecasters beat intelligence analysts by 30%
- Aggregation techniques (extremizing) improve scores by 10-15%
- Optimal calibration is key to superforecaster performance
"""

import logging
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class HumanBaseline:
    """
    Human forecaster baseline performance metrics.

    Based on published research from:
    - Good Judgment Project (Tetlock et al.)
    - IARPA ACE/SAGE programs
    - Superforecasting literature
    """

    # Baseline Brier scores
    BASELINES = {
        "random": 0.25,  # Coin flip / no information
        "novice": 0.40,  # Casual forecaster
        "expert": 0.35,  # Domain expert (geopolitical analyst)
        "trained": 0.30,  # Expert with forecasting training
        "superforecaster": 0.25,  # Top 2% of forecasters
        "superforecaster_aggregated": 0.20,  # Aggregated superforecasters
    }

    # Calibration targets (ECE)
    CALIBRATION_TARGETS = {
        "acceptable": 0.15,
        "good": 0.10,
        "excellent": 0.05,
        "superforecaster": 0.03,
    }

    @classmethod
    def get_baseline(cls, level: str) -> float:
        """
        Get baseline Brier score for forecaster level.

        Args:
            level: Forecaster level (random/novice/expert/trained/superforecaster)

        Returns:
            Baseline Brier score
        """
        return cls.BASELINES.get(level, cls.BASELINES["expert"])

    @classmethod
    def compare_to_human(cls, brier_score: float) -> Dict[str, any]:
        """
        Compare system Brier score to human baselines.

        Args:
            brier_score: System's Brier score

        Returns:
            Dict with comparison results:
                - performance_level: Best matched human level
                - beats_expert: True if better than expert
                - beats_superforecaster: True if better than superforecaster
                - percentile: Estimated percentile vs human forecasters
        """
        # Determine performance level
        if brier_score <= cls.BASELINES["superforecaster_aggregated"]:
            level = "superforecaster_aggregated"
        elif brier_score <= cls.BASELINES["superforecaster"]:
            level = "superforecaster"
        elif brier_score <= cls.BASELINES["trained"]:
            level = "trained"
        elif brier_score <= cls.BASELINES["expert"]:
            level = "expert"
        elif brier_score <= cls.BASELINES["novice"]:
            level = "novice"
        else:
            level = "random"

        # Calculate percentile (rough estimate)
        # Based on GJP data: superforecasters are top 2%, experts ~50th percentile
        percentile_map = {
            "superforecaster_aggregated": 99.5,
            "superforecaster": 98.0,
            "trained": 75.0,
            "expert": 50.0,
            "novice": 25.0,
            "random": 5.0,
        }

        percentile = percentile_map.get(level, 50.0)

        return {
            "performance_level": level,
            "beats_expert": brier_score < cls.BASELINES["expert"],
            "beats_superforecaster": brier_score < cls.BASELINES["superforecaster"],
            "percentile": percentile,
            "delta_vs_expert": brier_score - cls.BASELINES["expert"],
            "delta_vs_superforecaster": brier_score - cls.BASELINES["superforecaster"],
        }

    @classmethod
    def compare_calibration(cls, ece: float) -> Dict[str, any]:
        """
        Compare system calibration to human targets.

        Args:
            ece: System's Expected Calibration Error

        Returns:
            Dict with calibration comparison
        """
        if ece <= cls.CALIBRATION_TARGETS["superforecaster"]:
            level = "superforecaster"
        elif ece <= cls.CALIBRATION_TARGETS["excellent"]:
            level = "excellent"
        elif ece <= cls.CALIBRATION_TARGETS["good"]:
            level = "good"
        elif ece <= cls.CALIBRATION_TARGETS["acceptable"]:
            level = "acceptable"
        else:
            level = "poor"

        return {
            "calibration_level": level,
            "is_well_calibrated": ece <= cls.CALIBRATION_TARGETS["good"],
            "needs_recalibration": ece > cls.CALIBRATION_TARGETS["acceptable"],
        }


class Benchmark:
    """
    Comprehensive benchmarking system for geopolitical forecasting.

    Tracks performance over time, compares to baselines, and generates reports.
    """

    def __init__(self):
        """Initialize benchmark system."""
        self.human_baseline = HumanBaseline()

    def benchmark_predictions(
        self,
        predictions: List[Dict],
        brier_score: float,
        ece: float,
        use_calibrated: bool = True,
    ) -> Dict:
        """
        Generate comprehensive benchmark report.

        Args:
            predictions: List of prediction dicts
            brier_score: Overall Brier score
            ece: Expected Calibration Error
            use_calibrated: Whether calibrated probabilities were used

        Returns:
            Dict with complete benchmark results
        """
        # Compare to human baselines
        brier_comparison = self.human_baseline.compare_to_human(brier_score)
        calibration_comparison = self.human_baseline.compare_calibration(ece)

        # Calculate prediction statistics
        resolved = [p for p in predictions if p.get("outcome") is not None]
        unresolved = [p for p in predictions if p.get("outcome") is None]

        # Category breakdown
        category_stats = {}
        for category in ["conflict", "diplomatic", "economic"]:
            cat_preds = [p for p in resolved if p["category"] == category]
            if cat_preds:
                category_stats[category] = {
                    "n_predictions": len(cat_preds),
                    "base_rate": float(
                        np.mean([p["outcome"] for p in cat_preds])
                    ),
                }

        # Performance trajectory
        if len(resolved) >= 10:
            # Calculate Brier scores over time to detect improvement/degradation
            sorted_preds = sorted(resolved, key=lambda p: p.get("timestamp", ""))
            n_windows = 5
            window_size = len(sorted_preds) // n_windows

            trajectory = []
            for i in range(n_windows):
                start = i * window_size
                end = start + window_size if i < n_windows - 1 else len(sorted_preds)
                window_preds = sorted_preds[start:end]

                from .brier_scorer import BrierScorer

                scorer = BrierScorer()
                try:
                    window_score = scorer.score_batch(window_preds, use_calibrated)
                    trajectory.append(window_score["overall"])
                except ValueError:
                    continue

            # Calculate trend
            if len(trajectory) >= 2:
                x = np.arange(len(trajectory))
                trend_coef = np.polyfit(x, trajectory, 1)[0]
                improving = trend_coef < 0  # Negative trend = improving (lower Brier)
            else:
                improving = None
        else:
            trajectory = None
            improving = None

        return {
            "brier_score": brier_score,
            "ece": ece,
            "human_comparison": brier_comparison,
            "calibration_quality": calibration_comparison,
            "prediction_counts": {
                "total": len(predictions),
                "resolved": len(resolved),
                "unresolved": len(unresolved),
            },
            "category_stats": category_stats,
            "performance_trajectory": trajectory,
            "improving": improving,
            "methodology": "calibrated" if use_calibrated else "raw",
        }

    def generate_performance_summary(self, benchmark_results: Dict) -> str:
        """
        Generate human-readable performance summary.

        Args:
            benchmark_results: Results from benchmark_predictions()

        Returns:
            Formatted summary string
        """
        brier = benchmark_results["brier_score"]
        ece = benchmark_results["ece"]
        human_comp = benchmark_results["human_comparison"]
        cal_quality = benchmark_results["calibration_quality"]

        lines = [
            "=" * 70,
            "GEOPOLITICAL FORECASTING SYSTEM - PERFORMANCE BENCHMARK",
            "=" * 70,
            "",
            f"Overall Brier Score: {brier:.4f}",
            f"Expected Calibration Error (ECE): {ece:.4f}",
            "",
            "--- Human Baseline Comparison ---",
            f"Performance Level: {human_comp['performance_level'].upper()}",
            f"Estimated Percentile: {human_comp['percentile']:.1f}th",
            "",
        ]

        if human_comp["beats_superforecaster"]:
            lines.append(
                "🏆 EXCELLENT: Outperforms superforecasters "
                f"(Δ = {abs(human_comp['delta_vs_superforecaster']):.4f})"
            )
        elif human_comp["beats_expert"]:
            lines.append(
                "✓ GOOD: Outperforms expert forecasters "
                f"(Δ = {abs(human_comp['delta_vs_expert']):.4f})"
            )
        else:
            lines.append(
                "✗ BELOW TARGET: Underperforms expert baseline "
                f"(Δ = {human_comp['delta_vs_expert']:.4f})"
            )

        lines.extend(
            [
                "",
                "--- Calibration Quality ---",
                f"Calibration Level: {cal_quality['calibration_level'].upper()}",
            ]
        )

        if cal_quality["is_well_calibrated"]:
            lines.append("✓ Well calibrated (probabilities align with outcomes)")
        elif cal_quality["needs_recalibration"]:
            lines.append(
                "✗ NEEDS RECALIBRATION: Probabilities misaligned with outcomes"
            )

        lines.extend(
            [
                "",
                "--- Prediction Statistics ---",
                f"Total Predictions: {benchmark_results['prediction_counts']['total']}",
                f"Resolved: {benchmark_results['prediction_counts']['resolved']}",
                f"Unresolved: {benchmark_results['prediction_counts']['unresolved']}",
            ]
        )

        # Category breakdown
        if benchmark_results["category_stats"]:
            lines.append("\nCategory Breakdown:")
            for category, stats in benchmark_results["category_stats"].items():
                lines.append(
                    f"  {category.capitalize()}: {stats['n_predictions']} predictions, "
                    f"{stats['base_rate']:.2%} base rate"
                )

        # Performance trajectory
        if benchmark_results["improving"] is not None:
            lines.append("\n--- Performance Trajectory ---")
            if benchmark_results["improving"]:
                lines.append("✓ IMPROVING: Brier score trending downward over time")
            else:
                lines.append("✗ DEGRADING: Brier score trending upward over time")

        lines.append("=" * 70)

        return "\n".join(lines)
