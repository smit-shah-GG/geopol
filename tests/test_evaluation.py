"""
Tests for evaluation module.

Covers:
- Brier scoring (resolved and provisional)
- Calibration metrics (ECE, MCE, ACE)
- Drift detection
- Human baseline comparison
- Evaluator orchestration
"""

import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from src.evaluation.brier_scorer import BrierScorer
from src.evaluation.provisional_scorer import ProvisionalScorer
from src.evaluation.calibration_metrics import CalibrationMetrics
from src.evaluation.drift_detector import DriftDetector
from src.evaluation.benchmark import HumanBaseline, Benchmark


def create_test_predictions(n=50, resolved_fraction=0.7, seed=42):
    """Create synthetic test predictions."""
    np.random.seed(seed)

    predictions = []

    # Resolved predictions
    n_resolved = int(n * resolved_fraction)
    for i in range(n_resolved):
        # Generate with some calibration error
        raw_prob = np.random.uniform(0.2, 0.8)

        # Add calibration bias (overconfident)
        calibrated_prob = raw_prob * 1.2
        calibrated_prob = np.clip(calibrated_prob, 0.0, 1.0)

        # Outcome based on probability (with noise)
        outcome = 1.0 if np.random.random() < calibrated_prob else 0.0

        predictions.append(
            {
                "id": i,
                "query": f"Test prediction {i}",
                "raw_probability": raw_prob,
                "calibrated_probability": calibrated_prob,
                "category": np.random.choice(["conflict", "diplomatic", "economic"]),
                "entities": ["EntityA", "EntityB"],
                "outcome": outcome,
                "timestamp": datetime.now() - timedelta(days=np.random.randint(1, 60)),
                "resolution_date": datetime.now() - timedelta(days=np.random.randint(0, 10)),
            }
        )

    # Unresolved predictions
    n_unresolved = n - n_resolved
    for i in range(n_resolved, n):
        raw_prob = np.random.uniform(0.2, 0.8)
        calibrated_prob = raw_prob * 1.2
        calibrated_prob = np.clip(calibrated_prob, 0.0, 1.0)

        predictions.append(
            {
                "id": i,
                "query": f"Test prediction {i}",
                "raw_probability": raw_prob,
                "calibrated_probability": calibrated_prob,
                "category": np.random.choice(["conflict", "diplomatic", "economic"]),
                "entities": ["EntityA", "EntityB"],
                "outcome": None,  # Unresolved
                "timestamp": datetime.now() - timedelta(days=np.random.randint(1, 30)),
                "resolution_date": None,
            }
        )

    return predictions


class TestBrierScorer:
    """Test Brier score calculation."""

    def test_perfect_predictions(self):
        """Test Brier score for perfect predictions (should be 0)."""
        scorer = BrierScorer()

        # Perfect predictions
        predictions = [
            {"raw_probability": 1.0, "outcome": 1.0, "category": "conflict"},
            {"raw_probability": 0.0, "outcome": 0.0, "category": "conflict"},
            {"raw_probability": 1.0, "outcome": 1.0, "category": "diplomatic"},
            {"raw_probability": 0.0, "outcome": 0.0, "category": "diplomatic"},
        ]

        result = scorer.score_batch(predictions, use_calibrated=False)

        assert result["overall"] == pytest.approx(0.0, abs=1e-6)
        assert result["beats_expert"]
        assert result["beats_superforecaster"]

    def test_worst_predictions(self):
        """Test Brier score for worst predictions (should be 1.0)."""
        scorer = BrierScorer()

        # Maximally wrong predictions
        predictions = [
            {"raw_probability": 1.0, "outcome": 0.0, "category": "conflict"},
            {"raw_probability": 0.0, "outcome": 1.0, "category": "conflict"},
            {"raw_probability": 1.0, "outcome": 0.0, "category": "diplomatic"},
            {"raw_probability": 0.0, "outcome": 1.0, "category": "diplomatic"},
        ]

        result = scorer.score_batch(predictions, use_calibrated=False)

        assert result["overall"] == pytest.approx(1.0, abs=1e-6)
        assert not result["beats_expert"]

    def test_coin_flip_baseline(self):
        """Test that 0.5 probabilities give Brier score of 0.25."""
        scorer = BrierScorer()

        # Coin flip predictions
        predictions = [
            {"raw_probability": 0.5, "outcome": 1.0, "category": "conflict"},
            {"raw_probability": 0.5, "outcome": 0.0, "category": "conflict"},
        ] * 10

        result = scorer.score_batch(predictions, use_calibrated=False)

        assert result["overall"] == pytest.approx(0.25, abs=1e-6)

    def test_per_category_scores(self):
        """Test per-category Brier scores."""
        scorer = BrierScorer()

        predictions = create_test_predictions(n=60, resolved_fraction=1.0)

        result = scorer.score_batch(predictions, use_calibrated=True)

        # Check all categories have scores
        for category in ["conflict", "diplomatic", "economic"]:
            assert result[category] is not None
            assert 0.0 <= result[category] <= 1.0

    def test_confidence_interval(self):
        """Test bootstrap confidence interval calculation."""
        scorer = BrierScorer()

        predictions = create_test_predictions(n=100, resolved_fraction=1.0)

        ci = scorer.get_confidence_interval(predictions, use_calibrated=True)

        # Check CI structure
        assert "mean" in ci
        assert "lower" in ci
        assert "upper" in ci

        # Check bounds
        assert ci["lower"] <= ci["mean"] <= ci["upper"]

    def test_detailed_breakdown(self):
        """Test Brier score decomposition."""
        scorer = BrierScorer()

        predictions = create_test_predictions(n=100, resolved_fraction=1.0)

        breakdown = scorer.get_detailed_breakdown(predictions, use_calibrated=True)

        # Check decomposition components
        assert "decomposition" in breakdown
        assert "reliability" in breakdown["decomposition"]
        assert "resolution" in breakdown["decomposition"]
        assert "uncertainty" in breakdown["decomposition"]


class TestProvisionalScorer:
    """Test provisional scoring for unresolved predictions."""

    def test_time_decay_weight(self):
        """Test time decay weight calculation."""
        scorer = ProvisionalScorer()

        timestamp = datetime.now() - timedelta(days=15)
        deadline = timestamp + timedelta(days=30)

        # 50% elapsed
        weight = scorer.calculate_time_decay_weight(timestamp, deadline)

        assert 0.4 <= weight <= 0.6  # Should be around 0.5

    def test_time_decay_extremes(self):
        """Test time decay at extremes."""
        scorer = ProvisionalScorer()

        now = datetime.now()

        # Just created
        weight_start = scorer.calculate_time_decay_weight(now, now + timedelta(days=30))
        assert weight_start == pytest.approx(0.1, abs=0.01)

        # Past deadline
        weight_end = scorer.calculate_time_decay_weight(
            now - timedelta(days=31), now - timedelta(days=1)
        )
        assert weight_end == pytest.approx(1.0, abs=0.01)

    def test_provisional_outcome_estimation(self):
        """Test provisional outcome estimation."""
        scorer = ProvisionalScorer()

        # Conflict prediction with positive tension
        conflict_pred = {"category": "conflict", "entities": ["USA", "Iran"]}
        outcome = scorer.estimate_provisional_outcome(conflict_pred, tension_index=0.5)

        # Positive tension should suggest conflict likely
        assert outcome > 0.5

        # Diplomatic prediction with negative tension (cooperation)
        diplomatic_pred = {"category": "diplomatic", "entities": ["USA", "Canada"]}
        outcome = scorer.estimate_provisional_outcome(diplomatic_pred, tension_index=-0.5)

        # Negative tension (cooperation) should suggest diplomatic success
        assert outcome > 0.5

    def test_score_all_combined(self):
        """Test combined scoring with resolved and provisional predictions."""
        scorer = ProvisionalScorer()

        predictions = create_test_predictions(n=50, resolved_fraction=0.6)

        result = scorer.score_all(predictions, use_calibrated=True, provisional_weight=0.5)

        # Check structure
        assert "combined_brier" in result
        assert "resolved_brier" in result
        assert "provisional_brier" in result

        # Combined should be between resolved and provisional
        if result["resolved_brier"] and result["provisional_brier"]:
            assert result["combined_brier"] is not None


class TestCalibrationMetrics:
    """Test calibration metrics calculation."""

    def test_ece_calculation(self):
        """Test ECE calculation on synthetic data."""
        metrics = CalibrationMetrics(n_bins=10)

        # Perfectly calibrated predictions
        predictions = []
        for i in range(100):
            prob = i / 100
            outcome = 1.0 if np.random.random() < prob else 0.0
            predictions.append(
                {
                    "raw_probability": prob,
                    "outcome": outcome,
                    "category": "conflict",
                }
            )

        result = metrics.calculate_metrics(predictions, use_calibrated=False)

        # ECE should be small for perfectly calibrated
        assert result["ece"] < 0.2  # Allow some randomness

    def test_per_category_metrics(self):
        """Test per-category calibration metrics."""
        metrics = CalibrationMetrics(n_bins=10)

        predictions = create_test_predictions(n=90, resolved_fraction=1.0)

        result = metrics.calculate_per_category_metrics(predictions, use_calibrated=True)

        # Check all categories processed
        assert "conflict" in result
        assert "diplomatic" in result
        assert "economic" in result

    def test_bin_statistics(self):
        """Test bin statistics calculation."""
        metrics = CalibrationMetrics(n_bins=10)

        predictions = create_test_predictions(n=100, resolved_fraction=1.0)

        bin_stats = metrics.calculate_bin_statistics(predictions, use_calibrated=True)

        # Check structure
        assert len(bin_stats) > 0

        for stat in bin_stats:
            assert "bin_lower" in stat
            assert "bin_upper" in stat
            assert "n_predictions" in stat
            assert "avg_predicted" in stat
            assert "avg_observed" in stat
            assert "calibration_error" in stat


class TestDriftDetector:
    """Test calibration drift detection."""

    def test_metrics_recording(self):
        """Test recording calibration metrics."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            metrics_file = f.name

        try:
            detector = DriftDetector(metrics_file=metrics_file)

            # Record some metrics
            detector.record_metrics(ece=0.08, mce=0.15, ace=0.10, n_predictions=100)
            detector.record_metrics(ece=0.12, mce=0.20, ace=0.14, n_predictions=120)

            # Check history
            assert len(detector.metrics_history) == 2
            assert detector.metrics_history[0]["ece"] == 0.08

        finally:
            Path(metrics_file).unlink(missing_ok=True)

    def test_drift_detection(self):
        """Test drift detection logic."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            metrics_file = f.name

        try:
            detector = DriftDetector(
                alert_threshold=0.15, warning_threshold=0.10, metrics_file=metrics_file
            )

            # Create predictions with poor calibration (drift)
            predictions = []
            for i in range(50):
                # Highly miscalibrated
                prob = np.random.uniform(0.2, 0.8)
                outcome = 1.0 if np.random.random() > prob else 0.0  # Inverted

                predictions.append(
                    {
                        "raw_probability": prob * 0.9,  # Add raw_probability
                        "calibrated_probability": prob,
                        "outcome": outcome,
                        "category": "conflict",
                        "timestamp": datetime.now() - timedelta(days=i % 10),
                    }
                )

            result = detector.detect_drift(predictions, use_calibrated=True)

            # Should detect drift (ECE likely > 0.15)
            assert "drift_detected" in result
            assert "recommendation" in result

        finally:
            Path(metrics_file).unlink(missing_ok=True)

    def test_trend_statistics(self):
        """Test trend statistics calculation."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            metrics_file = f.name

        try:
            detector = DriftDetector(metrics_file=metrics_file)

            # Record increasing ECE (degrading)
            for i in range(10):
                ece = 0.05 + i * 0.01  # Increasing trend
                detector.record_metrics(ece=ece, mce=0.15, ace=0.10, n_predictions=100)

            stats = detector.get_trend_statistics()

            # Check trend is positive (degrading)
            assert "trend" in stats
            assert stats["trend"] > 0

        finally:
            Path(metrics_file).unlink(missing_ok=True)


class TestHumanBaseline:
    """Test human baseline comparison."""

    def test_baseline_levels(self):
        """Test baseline score retrieval."""
        assert HumanBaseline.get_baseline("expert") == 0.35
        assert HumanBaseline.get_baseline("superforecaster") == 0.25
        assert HumanBaseline.get_baseline("random") == 0.25

    def test_compare_to_human(self):
        """Test comparison to human performance."""
        # Superforecaster level
        result = HumanBaseline.compare_to_human(0.20)
        assert result["performance_level"] == "superforecaster_aggregated"
        assert result["beats_expert"]
        assert result["beats_superforecaster"]

        # Expert level
        result = HumanBaseline.compare_to_human(0.33)
        assert result["performance_level"] == "expert"
        assert result["beats_expert"]
        assert not result["beats_superforecaster"]

        # Below expert
        result = HumanBaseline.compare_to_human(0.40)
        assert not result["beats_expert"]

    def test_calibration_comparison(self):
        """Test calibration quality comparison."""
        # Excellent calibration
        result = HumanBaseline.compare_calibration(0.04)
        assert result["calibration_level"] == "excellent"
        assert result["is_well_calibrated"]

        # Poor calibration
        result = HumanBaseline.compare_calibration(0.20)
        assert result["calibration_level"] == "poor"
        assert not result["is_well_calibrated"]
        assert result["needs_recalibration"]


class TestBenchmark:
    """Test benchmarking system."""

    def test_benchmark_predictions(self):
        """Test comprehensive benchmark generation."""
        benchmark = Benchmark()

        predictions = create_test_predictions(n=100, resolved_fraction=0.8)

        result = benchmark.benchmark_predictions(
            predictions=predictions,
            brier_score=0.30,
            ece=0.08,
            use_calibrated=True,
        )

        # Check structure
        assert "brier_score" in result
        assert "ece" in result
        assert "human_comparison" in result
        assert "calibration_quality" in result
        assert "prediction_counts" in result

        # Check human comparison
        assert result["human_comparison"]["beats_expert"]

    def test_performance_summary(self):
        """Test human-readable summary generation."""
        benchmark = Benchmark()

        predictions = create_test_predictions(n=50, resolved_fraction=1.0)

        result = benchmark.benchmark_predictions(
            predictions=predictions,
            brier_score=0.25,
            ece=0.07,
            use_calibrated=True,
        )

        summary = benchmark.generate_performance_summary(result)

        # Check summary contains key information
        assert "Brier Score" in summary
        assert "ECE" in summary
        assert "Performance Level" in summary


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
