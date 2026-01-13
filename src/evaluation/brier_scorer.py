"""
Brier score calculation for resolved geopolitical predictions.

The Brier score is the gold standard metric for probabilistic forecasting accuracy:
    BS = (1/N) * sum((p_i - o_i)^2)
where p_i is the predicted probability and o_i is the actual outcome (0 or 1).

Lower is better: 0 = perfect, 0.25 = coin flip baseline, 1 = maximally wrong.

Human baseline:
- Expert forecasters: ~0.35 Brier score
- Superforecasters: ~0.25 Brier score
- Our target: <0.35 to beat experts
"""

import logging
from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import brier_score_loss

logger = logging.getLogger(__name__)


class BrierScorer:
    """
    Calculates Brier scores for resolved predictions.

    Provides:
    - Overall Brier score across all predictions
    - Per-category Brier scores (conflict/diplomatic/economic)
    - Detailed scoring breakdowns with confidence intervals
    - Comparison to human baseline performance
    """

    def __init__(self):
        """Initialize Brier scorer."""
        self.human_baseline = {
            "expert": 0.35,  # Typical expert forecaster
            "superforecaster": 0.25,  # Top 2% of forecasters
            "random": 0.25,  # Coin flip baseline
        }

    def score_batch(
        self, predictions: List[Dict], use_calibrated: bool = True
    ) -> Dict[str, float]:
        """
        Calculate Brier scores for a batch of predictions.

        Args:
            predictions: List of prediction dicts with keys:
                - 'calibrated_probability' or 'raw_probability': Predicted probability
                - 'outcome': Actual outcome (0.0 or 1.0)
                - 'category': Event category (conflict/diplomatic/economic)
            use_calibrated: If True, use calibrated_probability; else use raw_probability

        Returns:
            Dict with keys:
                - 'overall': Overall Brier score
                - 'conflict': Brier score for conflict predictions
                - 'diplomatic': Brier score for diplomatic predictions
                - 'economic': Brier score for economic predictions
                - 'n_predictions': Total number of predictions
                - 'beats_expert': True if overall < 0.35
                - 'beats_superforecaster': True if overall < 0.25

        Raises:
            ValueError: If no resolved predictions provided
        """
        # Filter resolved predictions only
        resolved = [p for p in predictions if p.get("outcome") is not None]

        if not resolved:
            raise ValueError("No resolved predictions provided for Brier scoring")

        # Extract probabilities and outcomes
        prob_key = "calibrated_probability" if use_calibrated else "raw_probability"

        y_true = np.array([p["outcome"] for p in resolved])
        y_prob = np.array([p.get(prob_key, p["raw_probability"]) for p in resolved])

        # Calculate overall Brier score
        overall_brier = brier_score_loss(y_true, y_prob)

        # Calculate per-category Brier scores
        category_scores = {}
        for category in ["conflict", "diplomatic", "economic"]:
            category_preds = [p for p in resolved if p["category"] == category]

            if category_preds:
                cat_y_true = np.array([p["outcome"] for p in category_preds])
                cat_y_prob = np.array(
                    [p.get(prob_key, p["raw_probability"]) for p in category_preds]
                )
                category_scores[category] = float(brier_score_loss(cat_y_true, cat_y_prob))
            else:
                category_scores[category] = None

        # Compare to baselines
        beats_expert = overall_brier < self.human_baseline["expert"]
        beats_superforecaster = overall_brier < self.human_baseline["superforecaster"]

        result = {
            "overall": float(overall_brier),
            "conflict": category_scores["conflict"],
            "diplomatic": category_scores["diplomatic"],
            "economic": category_scores["economic"],
            "n_predictions": len(resolved),
            "beats_expert": beats_expert,
            "beats_superforecaster": beats_superforecaster,
        }

        # Log summary
        logger.info(
            f"Brier Score: {overall_brier:.4f} on {len(resolved)} predictions "
            f"({'BEATS' if beats_expert else 'BELOW'} expert baseline)"
        )

        return result

    def score_single(self, predicted_prob: float, outcome: float) -> float:
        """
        Calculate Brier score for a single prediction.

        Args:
            predicted_prob: Predicted probability (0-1)
            outcome: Actual outcome (0 or 1)

        Returns:
            Brier score for this prediction

        Raises:
            ValueError: If inputs out of range
        """
        if not 0.0 <= predicted_prob <= 1.0:
            raise ValueError(f"Predicted probability must be in [0, 1], got {predicted_prob}")

        if outcome not in [0.0, 1.0]:
            raise ValueError(f"Outcome must be 0 or 1, got {outcome}")

        return float((predicted_prob - outcome) ** 2)

    def get_confidence_interval(
        self, predictions: List[Dict], use_calibrated: bool = True, confidence: float = 0.95
    ) -> Dict[str, float]:
        """
        Calculate confidence interval for Brier score using bootstrap.

        Args:
            predictions: List of prediction dicts
            use_calibrated: Use calibrated vs raw probabilities
            confidence: Confidence level (default 0.95)

        Returns:
            Dict with 'mean', 'lower', 'upper' keys
        """
        resolved = [p for p in predictions if p.get("outcome") is not None]

        if len(resolved) < 30:
            logger.warning(
                f"Only {len(resolved)} predictions; confidence interval may be unreliable"
            )

        # Bootstrap resampling
        n_bootstrap = 1000
        bootstrap_scores = []

        prob_key = "calibrated_probability" if use_calibrated else "raw_probability"

        for _ in range(n_bootstrap):
            # Resample with replacement
            sample = np.random.choice(resolved, size=len(resolved), replace=True)

            y_true = np.array([p["outcome"] for p in sample])
            y_prob = np.array([p.get(prob_key, p["raw_probability"]) for p in sample])

            bs = brier_score_loss(y_true, y_prob)
            bootstrap_scores.append(bs)

        # Calculate percentiles
        alpha = 1 - confidence
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100

        return {
            "mean": float(np.mean(bootstrap_scores)),
            "lower": float(np.percentile(bootstrap_scores, lower_percentile)),
            "upper": float(np.percentile(bootstrap_scores, upper_percentile)),
        }

    def get_detailed_breakdown(
        self, predictions: List[Dict], use_calibrated: bool = True
    ) -> Dict:
        """
        Generate detailed scoring breakdown with statistics.

        Args:
            predictions: List of prediction dicts
            use_calibrated: Use calibrated vs raw probabilities

        Returns:
            Dict with comprehensive statistics including:
            - Brier score components (reliability, resolution, uncertainty)
            - Per-category performance
            - Calibration quality indicators
            - Baseline comparisons
        """
        resolved = [p for p in predictions if p.get("outcome") is not None]

        if not resolved:
            return {"error": "No resolved predictions"}

        # Get basic scores
        scores = self.score_batch(predictions, use_calibrated)

        # Calculate Brier score decomposition
        # BS = Reliability - Resolution + Uncertainty
        prob_key = "calibrated_probability" if use_calibrated else "raw_probability"

        y_true = np.array([p["outcome"] for p in resolved])
        y_prob = np.array([p.get(prob_key, p["raw_probability"]) for p in resolved])

        # Uncertainty: variance of outcomes
        uncertainty = float(np.var(y_true))

        # Resolution: how well predictions separate outcomes
        # Higher is better - measures how much predictions vary from base rate
        base_rate = float(np.mean(y_true))
        resolution = float(np.mean((y_prob - base_rate) ** 2))

        # Reliability: calibration error
        # Lower is better - measures how close predictions are to outcomes
        reliability = float(np.mean((y_prob - y_true) ** 2))

        # Verify decomposition
        brier_check = reliability - resolution + uncertainty
        logger.debug(f"Brier decomposition check: {brier_check:.4f} vs {scores['overall']:.4f}")

        return {
            **scores,
            "decomposition": {
                "reliability": reliability,
                "resolution": resolution,
                "uncertainty": uncertainty,
            },
            "base_rate": base_rate,
            "confidence_interval": self.get_confidence_interval(predictions, use_calibrated),
        }
