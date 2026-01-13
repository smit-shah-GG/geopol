"""
Temperature scaling optimization for confidence calibration.

This module implements temperature scaling - a post-hoc calibration method
that learns an optimal temperature parameter T from validation data using
log loss minimization.

Temperature scaling:
- Scales logits: logits' = logits / T
- Scales confidence: c' = c^(1/T)
- T > 1: Smooths (reduces confidence)
- T < 1: Sharpens (increases confidence)
- T = 1: No change

Unlike isotonic regression, temperature scaling is parametric (single parameter T)
and preserves model ranking/ordering. It's particularly effective for neural
network calibration.
"""

import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


class TemperatureScaler:
    """
    Temperature scaling for confidence calibration.

    Learns optimal temperature T for each category using log loss minimization
    on validation data. The temperature parameter scales model confidence to
    better match true outcome probabilities.

    Architecture:
    1. Collect validation predictions and outcomes
    2. Optimize temperature T using scipy.optimize.minimize (L-BFGS)
    3. Apply T to calibrate confidence: c' = c^(1/T)
    4. Store per-category temperatures

    Attributes:
        temperatures: Dict mapping category -> optimal temperature
        sample_counts: Dict mapping category -> number of validation samples
        temperature_dir: Directory for persisting temperatures
    """

    def __init__(self, temperature_dir: str = "./data/temperature"):
        """
        Initialize temperature scaler.

        Args:
            temperature_dir: Directory to store temperature parameters
        """
        self.temperatures: Dict[str, float] = {}
        self.sample_counts: Dict[str, int] = {}
        self.temperature_dir = Path(temperature_dir)
        self.temperature_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initialized TemperatureScaler with dir={temperature_dir}")

    def fit(
        self,
        logits: List[float],
        labels: List[float],
        categories: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """
        Learn optimal temperature from validation data.

        Args:
            logits: Model logits (raw outputs before softmax). If unavailable,
                   can use log-odds: logit = log(p / (1-p)) for probability p.
            labels: True binary labels (0.0 or 1.0)
            categories: Optional per-sample categories for category-specific T.
                       If None, fits single global temperature.

        Returns:
            Dict mapping category -> optimal temperature

        Raises:
            ValueError: If input arrays have mismatched lengths or invalid values
        """
        # Validate inputs
        if not (len(logits) == len(labels)):
            raise ValueError(
                f"Logits and labels must have same length: "
                f"logits={len(logits)}, labels={len(labels)}"
            )

        if categories is not None and len(categories) != len(logits):
            raise ValueError(
                f"Categories must match logits length: "
                f"categories={len(categories)}, logits={len(logits)}"
            )

        if not all(label in [0.0, 1.0] for label in labels):
            raise ValueError("All labels must be 0.0 or 1.0")

        # Convert to numpy
        logits_arr = np.array(logits)
        labels_arr = np.array(labels)

        # Fit per-category or global temperature
        if categories is not None:
            categories_arr = np.array(categories)
            results = {}

            for category in ["conflict", "diplomatic", "economic"]:
                mask = categories_arr == category
                cat_logits = logits_arr[mask]
                cat_labels = labels_arr[mask]

                if len(cat_logits) < 10:
                    logger.warning(
                        f"Category '{category}' has only {len(cat_logits)} samples - "
                        "using default T=1.0"
                    )
                    self.temperatures[category] = 1.0
                    self.sample_counts[category] = len(cat_logits)
                    results[category] = 1.0
                    continue

                # Optimize temperature for this category
                T_opt = self._optimize_temperature(cat_logits, cat_labels)
                self.temperatures[category] = T_opt
                self.sample_counts[category] = len(cat_logits)
                results[category] = T_opt

                logger.info(
                    f"Optimized temperature for '{category}': T={T_opt:.3f} "
                    f"({len(cat_logits)} samples)"
                )
        else:
            # Global temperature
            T_opt = self._optimize_temperature(logits_arr, labels_arr)
            self.temperatures["global"] = T_opt
            self.sample_counts["global"] = len(logits_arr)
            results = {"global": T_opt}

            logger.info(f"Optimized global temperature: T={T_opt:.3f} ({len(logits_arr)} samples)")

        # Persist temperatures
        self._save_temperatures()

        return results

    def _optimize_temperature(self, logits: np.ndarray, labels: np.ndarray) -> float:
        """
        Optimize temperature T to minimize log loss.

        Uses L-BFGS optimization to find T that minimizes:
        log_loss = -mean(y * log(p') + (1-y) * log(1-p'))
        where p' = sigmoid(logit / T)

        Args:
            logits: Model logits
            labels: True labels (0 or 1)

        Returns:
            Optimal temperature T
        """

        def log_loss_with_temperature(T: float) -> float:
            """Compute log loss with temperature scaling."""
            # Apply temperature scaling: logit' = logit / T
            scaled_logits = logits / T

            # Convert to probabilities with sigmoid
            probs = 1.0 / (1.0 + np.exp(-scaled_logits))

            # Clip to avoid log(0)
            probs = np.clip(probs, 1e-7, 1 - 1e-7)

            # Compute binary cross-entropy (log loss)
            loss = -np.mean(labels * np.log(probs) + (1 - labels) * np.log(1 - probs))

            return loss

        # Optimize using L-BFGS
        # Initial guess: T = 1.0 (no scaling)
        # Bounds: T in [0.1, 10.0] (reasonable range for temperature)
        result = minimize(
            log_loss_with_temperature,
            x0=[1.0],
            method="L-BFGS-B",
            bounds=[(0.1, 10.0)],
            options={"maxiter": 100},
        )

        T_opt = float(result.x[0])

        logger.debug(
            f"Temperature optimization: T={T_opt:.3f}, "
            f"loss={result.fun:.4f}, success={result.success}"
        )

        return T_opt

    def calibrate_confidence(self, confidence: float, temperature: float) -> float:
        """
        Apply temperature scaling to calibrate confidence.

        Formula: c' = c^(1/T)
        - T > 1: Reduces confidence (smoother)
        - T < 1: Increases confidence (sharper)
        - T = 1: No change

        Args:
            confidence: Raw confidence score (0-1)
            temperature: Temperature parameter

        Returns:
            Calibrated confidence (0-1)

        Raises:
            ValueError: If confidence not in [0,1] or temperature <= 0
        """
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(f"Confidence must be in [0, 1], got {confidence}")

        if temperature <= 0:
            raise ValueError(f"Temperature must be > 0, got {temperature}")

        # Apply power scaling: c' = c^(1/T)
        # Note: This is equivalent to scaling logits before sigmoid
        calibrated = confidence ** (1.0 / temperature)

        # Ensure valid range
        calibrated = np.clip(calibrated, 0.0, 1.0)

        return float(calibrated)

    def calibrate(self, confidence: float, category: Optional[str] = None) -> float:
        """
        Calibrate confidence using learned temperature.

        Args:
            confidence: Raw confidence score (0-1)
            category: Optional category for category-specific temperature.
                     If None, uses global temperature.

        Returns:
            Calibrated confidence (0-1)
        """
        # Get temperature for category
        if category and category in self.temperatures:
            T = self.temperatures[category]
        elif "global" in self.temperatures:
            T = self.temperatures["global"]
        else:
            logger.warning("No temperature trained - returning raw confidence")
            return confidence

        return self.calibrate_confidence(confidence, T)

    def calibrate_batch(
        self,
        confidences: List[float],
        categories: Optional[List[str]] = None,
    ) -> List[float]:
        """
        Calibrate multiple confidences efficiently.

        Args:
            confidences: List of raw confidence scores
            categories: Optional list of categories (same length as confidences)

        Returns:
            List of calibrated confidences
        """
        if categories is None:
            categories = [None] * len(confidences)

        if len(confidences) != len(categories):
            raise ValueError(
                f"Confidences and categories must have same length: "
                f"{len(confidences)} != {len(categories)}"
            )

        return [self.calibrate(c, cat) for c, cat in zip(confidences, categories)]

    def get_temperature_info(self) -> Dict:
        """
        Get information about learned temperatures.

        Returns:
            Dict with keys: categories, temperatures, sample_counts, trained
        """
        return {
            "categories": list(self.temperatures.keys()),
            "temperatures": self.temperatures,
            "sample_counts": self.sample_counts,
            "trained": len(self.temperatures) > 0,
        }

    def _save_temperatures(self) -> None:
        """Save temperatures to disk for persistence."""
        filepath = self.temperature_dir / "temperatures.pkl"
        with open(filepath, "wb") as f:
            pickle.dump(
                {
                    "temperatures": self.temperatures,
                    "sample_counts": self.sample_counts,
                },
                f,
            )
        logger.info(f"Saved temperatures to {filepath}")

    def load_temperatures(self) -> bool:
        """
        Load temperatures from disk.

        Returns:
            True if temperatures loaded successfully, False otherwise
        """
        filepath = self.temperature_dir / "temperatures.pkl"

        if not filepath.exists():
            logger.debug(f"No saved temperatures found at {filepath}")
            return False

        try:
            with open(filepath, "rb") as f:
                data = pickle.load(f)
                self.temperatures = data["temperatures"]
                self.sample_counts = data["sample_counts"]
                logger.info(f"Loaded temperatures from {filepath}: {self.temperatures}")
                return True
        except Exception as e:
            logger.error(f"Failed to load temperatures: {e}")
            return False


def probabilities_to_logits(probabilities: List[float]) -> List[float]:
    """
    Convert probabilities to logits for temperature scaling.

    Formula: logit = log(p / (1-p))

    This is useful when you only have probabilities but need logits for
    temperature scaling optimization.

    Args:
        probabilities: List of probabilities (0-1, excluding 0 and 1)

    Returns:
        List of logits

    Raises:
        ValueError: If any probability is 0 or 1 (undefined logit)
    """
    logits = []

    for p in probabilities:
        if p <= 0.0 or p >= 1.0:
            # Clip to avoid division by zero
            p = np.clip(p, 1e-7, 1 - 1e-7)

        logit = np.log(p / (1 - p))
        logits.append(float(logit))

    return logits
