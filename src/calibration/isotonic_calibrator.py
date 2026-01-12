"""
Isotonic regression calibration for geopolitical predictions.

This module implements per-category isotonic calibration using scikit-learn's
CalibratedClassifierCV. It:
- Trains separate calibration curves per event category (conflict/diplomatic/economic)
- Automatically selects isotonic vs sigmoid methods based on dataset size
- Persists calibration curves to disk for reuse
- Provides recalibration for periodic updates

Isotonic regression is non-parametric and makes no assumptions about the form
of the calibration function, making it ideal for small datasets where the
true relationship between predicted and actual probabilities is unknown.
"""

import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression

logger = logging.getLogger(__name__)


class IsotonicCalibrator:
    """
    Per-category isotonic regression calibrator.

    Maintains separate calibration curves for each event category to account
    for systematic biases in different prediction domains (e.g., conflict
    predictions may be overconfident while economic predictions are underconfident).

    Architecture:
    1. Train separate IsotonicRegression models per category
    2. Use isotonic method for >1000 samples, sigmoid for smaller datasets
    3. Store curves in pickle files for persistence
    4. Provide calibrate() method for inference

    Attributes:
        calibrators: Dict mapping category -> IsotonicRegression model
        sample_counts: Dict mapping category -> number of training samples
        calibration_dir: Directory for persisting calibration curves
    """

    def __init__(self, calibration_dir: str = "./data/calibration"):
        """
        Initialize isotonic calibrator.

        Args:
            calibration_dir: Directory to store calibration curves
        """
        self.calibrators: Dict[str, IsotonicRegression] = {}
        self.sample_counts: Dict[str, int] = {}
        self.calibration_dir = Path(calibration_dir)
        self.calibration_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initialized IsotonicCalibrator with dir={calibration_dir}")

    def fit(
        self,
        predictions: List[float],
        outcomes: List[float],
        categories: List[str],
    ) -> Dict[str, int]:
        """
        Fit per-category calibration curves.

        Args:
            predictions: Raw model probabilities (0-1)
            outcomes: True outcomes (0.0 or 1.0)
            categories: Event categories for each prediction

        Returns:
            Dict mapping category -> number of samples used for training

        Raises:
            ValueError: If input arrays have mismatched lengths or invalid values
        """
        # Validate inputs
        if not (len(predictions) == len(outcomes) == len(categories)):
            raise ValueError(
                f"Input arrays must have same length: predictions={len(predictions)}, "
                f"outcomes={len(outcomes)}, categories={len(categories)}"
            )

        if not all(0.0 <= p <= 1.0 for p in predictions):
            raise ValueError("All predictions must be in [0, 1]")

        if not all(o in [0.0, 1.0] for o in outcomes):
            raise ValueError("All outcomes must be 0.0 or 1.0")

        # Convert to numpy arrays
        predictions = np.array(predictions)
        outcomes = np.array(outcomes)
        categories_arr = np.array(categories)

        # Train per-category calibrators
        results = {}
        for category in ["conflict", "diplomatic", "economic"]:
            # Filter data for this category
            mask = categories_arr == category
            cat_preds = predictions[mask]
            cat_outcomes = outcomes[mask]

            n_samples = len(cat_preds)
            results[category] = n_samples

            if n_samples < 10:
                logger.warning(
                    f"Category '{category}' has only {n_samples} samples - "
                    "skipping calibration (minimum 10 required)"
                )
                continue

            # Choose calibration method based on sample size
            # Isotonic for >1000 samples (more flexible, but needs more data)
            # Sigmoid (Platt scaling) for smaller datasets (more stable)
            if n_samples >= 1000:
                logger.info(f"Using isotonic regression for '{category}' ({n_samples} samples)")
                calibrator = IsotonicRegression(out_of_bounds="clip")
            else:
                logger.info(
                    f"Using sigmoid calibration for '{category}' ({n_samples} samples, <1000)"
                )
                # Sigmoid is implemented as a simple logistic regression
                from sklearn.linear_model import LogisticRegression

                calibrator = LogisticRegression()

            # Fit calibrator
            if isinstance(calibrator, IsotonicRegression):
                calibrator.fit(cat_preds, cat_outcomes)
            else:
                # LogisticRegression needs 2D input
                calibrator.fit(cat_preds.reshape(-1, 1), cat_outcomes)

            self.calibrators[category] = calibrator
            self.sample_counts[category] = n_samples

            logger.info(f"Trained calibrator for '{category}' with {n_samples} samples")

        # Persist calibrators to disk
        self._save_calibrators()

        return results

    def calibrate(self, probability: float, category: str) -> float:
        """
        Calibrate a raw probability using the category-specific calibrator.

        Args:
            probability: Raw model probability (0-1)
            category: Event category (conflict/diplomatic/economic)

        Returns:
            Calibrated probability (0-1)

        Raises:
            ValueError: If category invalid or not trained
        """
        if not 0.0 <= probability <= 1.0:
            raise ValueError(f"Probability must be in [0, 1], got {probability}")

        category = category.lower()
        if category not in ["conflict", "diplomatic", "economic"]:
            raise ValueError(
                f"Category must be one of [conflict, diplomatic, economic], got {category}"
            )

        # Check if calibrator exists for this category
        if category not in self.calibrators:
            logger.warning(
                f"No calibrator trained for category '{category}' - returning raw probability"
            )
            return probability

        calibrator = self.calibrators[category]

        # Apply calibration
        if isinstance(calibrator, IsotonicRegression):
            calibrated = calibrator.predict([probability])[0]
        else:
            # LogisticRegression
            calibrated = calibrator.predict_proba([[probability]])[0, 1]

        # Ensure valid range (should be handled by out_of_bounds="clip", but be defensive)
        calibrated = np.clip(calibrated, 0.0, 1.0)

        logger.debug(
            f"Calibrated {category} prediction: {probability:.3f} -> {calibrated:.3f} "
            f"(Δ={calibrated - probability:+.3f})"
        )

        return float(calibrated)

    def calibrate_batch(
        self, probabilities: List[float], categories: List[str]
    ) -> List[float]:
        """
        Calibrate multiple predictions efficiently.

        Args:
            probabilities: List of raw probabilities
            categories: List of categories (same length as probabilities)

        Returns:
            List of calibrated probabilities
        """
        if len(probabilities) != len(categories):
            raise ValueError(
                f"Probabilities and categories must have same length: "
                f"{len(probabilities)} != {len(categories)}"
            )

        return [self.calibrate(p, c) for p, c in zip(probabilities, categories)]

    def recalibrate(
        self,
        predictions: List[float],
        outcomes: List[float],
        categories: List[str],
    ) -> Dict[str, int]:
        """
        Update calibrators with new data (periodic recalibration).

        This is a convenience method that calls fit() with new data.
        In production, you might want to implement incremental updates
        or weighted combinations of old and new calibrators.

        Args:
            predictions: New raw predictions
            outcomes: New outcomes
            categories: Categories for new predictions

        Returns:
            Dict mapping category -> number of samples used
        """
        logger.info("Recalibrating with new data")
        return self.fit(predictions, outcomes, categories)

    def get_calibration_info(self) -> Dict:
        """
        Get information about trained calibrators.

        Returns:
            Dict with keys: categories (list), sample_counts (dict), trained (bool)
        """
        return {
            "categories": list(self.calibrators.keys()),
            "sample_counts": self.sample_counts,
            "trained": len(self.calibrators) > 0,
            "total_samples": sum(self.sample_counts.values()),
        }

    def _save_calibrators(self) -> None:
        """Save calibrators to disk for persistence."""
        for category, calibrator in self.calibrators.items():
            filepath = self.calibration_dir / f"{category}_calibrator.pkl"
            with open(filepath, "wb") as f:
                pickle.dump(
                    {
                        "calibrator": calibrator,
                        "sample_count": self.sample_counts[category],
                    },
                    f,
                )
            logger.info(f"Saved calibrator for '{category}' to {filepath}")

    def load_calibrators(self) -> bool:
        """
        Load calibrators from disk.

        Returns:
            True if calibrators loaded successfully, False otherwise
        """
        loaded_any = False

        for category in ["conflict", "diplomatic", "economic"]:
            filepath = self.calibration_dir / f"{category}_calibrator.pkl"

            if not filepath.exists():
                logger.debug(f"No saved calibrator found for '{category}' at {filepath}")
                continue

            try:
                with open(filepath, "rb") as f:
                    data = pickle.load(f)
                    self.calibrators[category] = data["calibrator"]
                    self.sample_counts[category] = data["sample_count"]
                    logger.info(
                        f"Loaded calibrator for '{category}' "
                        f"({data['sample_count']} samples)"
                    )
                    loaded_any = True
            except Exception as e:
                logger.error(f"Failed to load calibrator for '{category}': {e}")

        return loaded_any

    def get_calibration_curve(self, category: str, n_points: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get calibration curve for visualization.

        Args:
            category: Event category
            n_points: Number of points to sample

        Returns:
            Tuple of (raw_probabilities, calibrated_probabilities) arrays

        Raises:
            ValueError: If category not trained
        """
        category = category.lower()
        if category not in self.calibrators:
            raise ValueError(f"No calibrator trained for category '{category}'")

        # Sample points uniformly in [0, 1]
        raw_probs = np.linspace(0.0, 1.0, n_points)

        # Get calibrated values
        calibrator = self.calibrators[category]
        if isinstance(calibrator, IsotonicRegression):
            cal_probs = calibrator.predict(raw_probs)
        else:
            cal_probs = calibrator.predict_proba(raw_probs.reshape(-1, 1))[:, 1]

        return raw_probs, cal_probs
