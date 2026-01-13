"""
Calibration drift detection and monitoring.

Calibration drift occurs when model probabilities become less aligned with outcomes over time.
This can happen due to:
- Changing event patterns (geopolitical shifts)
- Model staleness (training data becomes outdated)
- Adversarial dynamics (actors adapt to predictions)

The drift detector:
1. Monitors ECE over sliding time windows
2. Compares recent vs historical calibration
3. Triggers recalibration alerts when drift detected
4. Stores metrics history for trend analysis
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class DriftDetector:
    """
    Monitors calibration drift over time using sliding windows.

    Tracks ECE progression and alerts when calibration degrades beyond thresholds.
    Maintains historical metrics in SQLite for trend analysis.
    """

    def __init__(
        self,
        window_days: int = 30,
        alert_threshold: float = 0.15,
        warning_threshold: float = 0.10,
        metrics_file: Optional[str] = None,
    ):
        """
        Initialize drift detector.

        Args:
            window_days: Size of sliding window for drift detection
            alert_threshold: ECE threshold for recalibration alert (default 0.15)
            warning_threshold: ECE threshold for warning (default 0.10)
            metrics_file: Path to metrics history JSON file
        """
        self.window_days = window_days
        self.alert_threshold = alert_threshold
        self.warning_threshold = warning_threshold

        # Metrics history file
        if metrics_file is None:
            metrics_file = "./data/calibration_metrics_history.json"

        self.metrics_file = Path(metrics_file)
        self.metrics_file.parent.mkdir(parents=True, exist_ok=True)

        # Initialize or load history
        self.metrics_history = self._load_history()

    def _load_history(self) -> List[Dict]:
        """Load metrics history from JSON file."""
        if not self.metrics_file.exists():
            return []

        try:
            with open(self.metrics_file, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load metrics history: {e}")
            return []

    def _save_history(self) -> None:
        """Save metrics history to JSON file."""
        try:
            with open(self.metrics_file, "w") as f:
                json.dump(self.metrics_history, f, indent=2)
        except Exception as e:
            logger.error(f"Could not save metrics history: {e}")

    def record_metrics(
        self, ece: float, mce: float, ace: float, n_predictions: int, metadata: Optional[Dict] = None
    ) -> None:
        """
        Record calibration metrics with timestamp.

        Args:
            ece: Expected Calibration Error
            mce: Maximum Calibration Error
            ace: Adaptive Calibration Error
            n_predictions: Number of predictions used
            metadata: Optional additional metadata
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "ece": ece,
            "mce": mce,
            "ace": ace,
            "n_predictions": n_predictions,
            "metadata": metadata or {},
        }

        self.metrics_history.append(entry)
        self._save_history()

        logger.info(f"Recorded metrics: ECE={ece:.4f}, MCE={mce:.4f}, n={n_predictions}")

    def detect_drift(
        self, predictions: List[Dict], use_calibrated: bool = True
    ) -> Dict[str, any]:
        """
        Detect calibration drift in recent predictions.

        Compares recent window ECE to historical baseline.
        Triggers alerts if drift exceeds thresholds.

        Args:
            predictions: List of prediction dicts with timestamps
            use_calibrated: Use calibrated vs raw probabilities

        Returns:
            Dict with drift detection results:
                - drift_detected: True if ECE > alert_threshold
                - current_ece: ECE in recent window
                - historical_ece: Baseline ECE (if available)
                - delta: Change in ECE vs baseline
                - recommendation: Action to take
        """
        # Calculate ECE for recent window
        cutoff_date = datetime.now() - timedelta(days=self.window_days)

        recent_preds = [
            p
            for p in predictions
            if p.get("timestamp") and p["timestamp"] >= cutoff_date and p.get("outcome") is not None
        ]

        if len(recent_preds) < 10:
            logger.warning(
                f"Only {len(recent_preds)} predictions in recent window; drift detection unreliable"
            )
            return {
                "drift_detected": False,
                "current_ece": None,
                "warning": "Insufficient recent predictions",
            }

        # Calculate current ECE
        from .calibration_metrics import CalibrationMetrics

        metrics_calc = CalibrationMetrics()

        try:
            current_metrics = metrics_calc.calculate_metrics(recent_preds, use_calibrated)
            current_ece = current_metrics["ece"]
        except ValueError as e:
            logger.error(f"Could not calculate current ECE: {e}")
            return {"drift_detected": False, "error": str(e)}

        # Get historical baseline (average ECE from older data)
        historical_ece = self._get_historical_baseline()

        # Determine drift status
        drift_detected = current_ece > self.alert_threshold
        warning = (
            current_ece > self.warning_threshold and current_ece <= self.alert_threshold
        )

        # Calculate delta if we have baseline
        delta = None
        if historical_ece is not None:
            delta = current_ece - historical_ece

        # Generate recommendation
        if drift_detected:
            recommendation = (
                f"RECALIBRATE: ECE={current_ece:.4f} exceeds alert threshold "
                f"({self.alert_threshold}). Run isotonic recalibration."
            )
        elif warning:
            recommendation = (
                f"WARNING: ECE={current_ece:.4f} above target ({self.warning_threshold}). "
                f"Monitor closely; consider recalibration if trend continues."
            )
        else:
            recommendation = f"OK: ECE={current_ece:.4f} within acceptable range."

        result = {
            "drift_detected": drift_detected,
            "warning": warning,
            "current_ece": current_ece,
            "historical_ece": historical_ece,
            "delta": delta,
            "window_days": self.window_days,
            "n_predictions": len(recent_preds),
            "recommendation": recommendation,
        }

        # Log result
        if drift_detected:
            logger.warning(recommendation)
        elif warning:
            logger.info(recommendation)
        else:
            logger.info(recommendation)

        return result

    def _get_historical_baseline(self) -> Optional[float]:
        """
        Calculate historical baseline ECE.

        Uses median ECE from metrics history (more robust than mean).

        Returns:
            Median historical ECE or None if insufficient history
        """
        if len(self.metrics_history) < 5:
            return None

        # Get ECE values
        ece_values = [entry["ece"] for entry in self.metrics_history]

        # Return median
        return float(np.median(ece_values))

    def get_trend_statistics(self) -> Dict[str, float]:
        """
        Calculate trend statistics from metrics history.

        Returns:
            Dict with trend statistics:
                - mean_ece: Average ECE over time
                - median_ece: Median ECE
                - std_ece: Standard deviation
                - min_ece: Best calibration achieved
                - max_ece: Worst calibration
                - trend: Linear trend coefficient (positive = degrading)
        """
        if not self.metrics_history:
            return {}

        ece_values = np.array([entry["ece"] for entry in self.metrics_history])

        # Calculate statistics
        stats = {
            "mean_ece": float(np.mean(ece_values)),
            "median_ece": float(np.median(ece_values)),
            "std_ece": float(np.std(ece_values)),
            "min_ece": float(np.min(ece_values)),
            "max_ece": float(np.max(ece_values)),
        }

        # Calculate linear trend
        if len(ece_values) >= 2:
            x = np.arange(len(ece_values))
            trend_coef = np.polyfit(x, ece_values, 1)[0]
            stats["trend"] = float(trend_coef)

        return stats

    def detect_per_category_drift(
        self, predictions: List[Dict], use_calibrated: bool = True
    ) -> Dict[str, Dict]:
        """
        Detect drift separately for each category.

        Args:
            predictions: List of prediction dicts
            use_calibrated: Use calibrated vs raw probabilities

        Returns:
            Dict mapping category -> drift detection results
        """
        results = {}

        for category in ["conflict", "diplomatic", "economic"]:
            category_preds = [p for p in predictions if p["category"] == category]

            if not category_preds:
                results[category] = None
                continue

            try:
                drift_result = self.detect_drift(category_preds, use_calibrated)
                results[category] = drift_result
            except Exception as e:
                logger.warning(f"Could not detect drift for {category}: {e}")
                results[category] = {"error": str(e)}

        return results

    def get_recent_metrics(self, n: int = 10) -> List[Dict]:
        """
        Get the N most recent metrics entries.

        Args:
            n: Number of recent entries to return

        Returns:
            List of recent metrics dicts
        """
        return self.metrics_history[-n:] if self.metrics_history else []

    def clear_history(self) -> None:
        """Clear all metrics history (use with caution)."""
        self.metrics_history = []
        self._save_history()
        logger.warning("Cleared all metrics history")
