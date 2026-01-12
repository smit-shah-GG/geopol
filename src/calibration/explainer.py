"""
Explainable calibration adjustments for geopolitical predictions.

This module generates human-readable explanations for why calibration adjusted
a prediction's probability. It analyzes historical patterns to provide context
like "Reduced confidence by 15% - similar Russia predictions overconfident in
8/10 recent cases."

The explainer uses:
1. Historical prediction performance data
2. Entity similarity matching
3. Category-specific bias patterns
4. Recent calibration trends
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class CalibrationExplainer:
    """
    Generates explanations for calibration adjustments.

    Analyzes historical prediction patterns to explain why a calibration
    adjustment was made, providing transparency and trust in the system.

    Architecture:
    1. Store historical predictions with outcomes
    2. Match current prediction to similar historical predictions
    3. Compute bias patterns (overconfidence vs underconfidence)
    4. Generate natural language explanation
    """

    def __init__(self, prediction_store=None):
        """
        Initialize calibration explainer.

        Args:
            prediction_store: Optional PredictionStore instance for accessing history
        """
        self.prediction_store = prediction_store
        logger.info("Initialized CalibrationExplainer")

    def explain_adjustment(
        self,
        raw_probability: float,
        calibrated_probability: float,
        category: str,
        entities: List[str],
        query: str,
    ) -> str:
        """
        Generate explanation for calibration adjustment.

        Args:
            raw_probability: Original model prediction
            calibrated_probability: Calibrated prediction
            category: Event category
            entities: Entities involved in prediction
            query: Original forecasting question

        Returns:
            Human-readable explanation string
        """
        # Calculate adjustment magnitude and direction
        adjustment = calibrated_probability - raw_probability
        adjustment_pct = abs(adjustment) * 100

        # No significant adjustment
        if abs(adjustment) < 0.01:
            return "No calibration adjustment needed (model well-calibrated for this category)"

        # Direction of adjustment
        if adjustment > 0:
            direction = "increased"
            bias_type = "underconfident"
        else:
            direction = "decreased"
            bias_type = "overconfident"

        # Build explanation components
        explanation_parts = [
            f"Calibration {direction} confidence by {adjustment_pct:.1f}%"
        ]

        # Add historical context if prediction store available
        if self.prediction_store:
            historical_context = self._get_historical_context(
                category, entities, raw_probability
            )
            if historical_context:
                explanation_parts.append(historical_context)
        else:
            # Generic explanation without historical data
            explanation_parts.append(
                f"model tends to be {bias_type} for {category} predictions in this probability range"
            )

        # Add category-specific insight
        category_insight = self._get_category_insight(category, adjustment)
        if category_insight:
            explanation_parts.append(category_insight)

        return " - ".join(explanation_parts)

    def _get_historical_context(
        self,
        category: str,
        entities: List[str],
        raw_probability: float,
        lookback_days: int = 180,
    ) -> Optional[str]:
        """
        Get historical context for similar predictions.

        Args:
            category: Event category
            entities: Entities in current prediction
            raw_probability: Raw probability value
            lookback_days: How many days of history to consider

        Returns:
            Historical context string or None
        """
        if not self.prediction_store:
            return None

        try:
            # Get historical predictions for this category with outcomes
            from datetime import datetime, timedelta

            min_date = datetime.now() - timedelta(days=lookback_days)
            historical = self.prediction_store.get_predictions_for_calibration(
                category=category,
                resolved_only=True,
                min_date=min_date,
            )

            if len(historical) < 5:
                logger.debug(f"Insufficient historical data ({len(historical)} predictions)")
                return None

            # Find similar predictions (by entity overlap and probability range)
            similar = self._find_similar_predictions(
                historical, entities, raw_probability
            )

            if len(similar) < 3:
                logger.debug(f"Insufficient similar predictions ({len(similar)})")
                return None

            # Analyze bias pattern in similar predictions
            bias_pattern = self._analyze_bias_pattern(similar)

            if bias_pattern:
                return bias_pattern

        except Exception as e:
            logger.error(f"Error generating historical context: {e}")

        return None

    def _find_similar_predictions(
        self,
        historical: List[Dict],
        entities: List[str],
        raw_probability: float,
        prob_tolerance: float = 0.15,
    ) -> List[Dict]:
        """
        Find similar historical predictions.

        Similarity criteria:
        1. Entity overlap (at least one entity in common)
        2. Probability within ±tolerance
        3. Has resolved outcome

        Args:
            historical: List of historical prediction dicts
            entities: Current entities
            raw_probability: Current raw probability
            prob_tolerance: Probability range for similarity

        Returns:
            List of similar prediction dicts
        """
        similar = []

        # Convert entities to lowercase for comparison
        entities_lower = {e.lower() for e in entities}

        for pred in historical:
            # Check entity overlap
            pred_entities = {e.lower() for e in pred.get("entities", [])}
            if not entities_lower.intersection(pred_entities):
                continue

            # Check probability range
            pred_prob = pred["raw_probability"]
            if abs(pred_prob - raw_probability) > prob_tolerance:
                continue

            # Must have outcome
            if pred["outcome"] is None:
                continue

            similar.append(pred)

        return similar

    def _analyze_bias_pattern(self, similar_predictions: List[Dict]) -> Optional[str]:
        """
        Analyze bias pattern in similar predictions.

        Computes:
        - Overconfidence: predicted > actual outcome rate
        - Underconfidence: predicted < actual outcome rate

        Args:
            similar_predictions: List of similar prediction dicts

        Returns:
            Bias pattern description or None
        """
        if len(similar_predictions) < 3:
            return None

        # Extract predictions and outcomes
        predictions = [p["raw_probability"] for p in similar_predictions]
        outcomes = [p["outcome"] for p in similar_predictions]

        # Compute calibration error
        mean_prediction = np.mean(predictions)
        actual_rate = np.mean(outcomes)
        calibration_error = mean_prediction - actual_rate

        n_correct = sum(1 for o in outcomes if o == 1.0)
        n_total = len(outcomes)

        # Build description
        if abs(calibration_error) < 0.05:
            return None  # Well-calibrated

        if calibration_error > 0:
            # Overconfident
            return (
                f"similar predictions were overconfident "
                f"({n_correct}/{n_total} actually occurred, "
                f"but predicted {mean_prediction:.0%})"
            )
        else:
            # Underconfident
            return (
                f"similar predictions were underconfident "
                f"({n_correct}/{n_total} actually occurred, "
                f"but predicted {mean_prediction:.0%})"
            )

    def _get_category_insight(self, category: str, adjustment: float) -> Optional[str]:
        """
        Get category-specific insight about calibration.

        Args:
            category: Event category
            adjustment: Calibration adjustment value

        Returns:
            Insight string or None
        """
        # Category-specific patterns based on domain knowledge
        insights = {
            "conflict": {
                "negative": "conflict predictions often overestimate escalation likelihood",
                "positive": "conflict predictions sometimes miss emerging patterns",
            },
            "diplomatic": {
                "negative": "diplomatic predictions tend to overestimate agreement likelihood",
                "positive": "diplomatic predictions may underestimate breakthrough probability",
            },
            "economic": {
                "negative": "economic predictions can overestimate shock impact",
                "positive": "economic predictions may underestimate systemic risk",
            },
        }

        category_insights = insights.get(category, {})

        if adjustment < 0 and "negative" in category_insights:
            return category_insights["negative"]
        elif adjustment > 0 and "positive" in category_insights:
            return category_insights["positive"]

        return None

    def explain_batch(
        self,
        adjustments: List[Tuple[float, float, str, List[str], str]],
    ) -> List[str]:
        """
        Generate explanations for multiple calibration adjustments.

        Args:
            adjustments: List of tuples (raw_prob, cal_prob, category, entities, query)

        Returns:
            List of explanation strings
        """
        explanations = []

        for raw_prob, cal_prob, category, entities, query in adjustments:
            explanation = self.explain_adjustment(
                raw_prob, cal_prob, category, entities, query
            )
            explanations.append(explanation)

        return explanations

    def get_calibration_statistics(
        self, category: Optional[str] = None, days: int = 30
    ) -> Dict:
        """
        Get calibration statistics for analysis.

        Args:
            category: Optional category filter
            days: Number of days to analyze

        Returns:
            Dict with statistics (mean_adjustment, overconfident_rate, etc.)
        """
        if not self.prediction_store:
            return {}

        from datetime import datetime, timedelta

        min_date = datetime.now() - timedelta(days=days)

        predictions = self.prediction_store.get_predictions_for_calibration(
            category=category,
            resolved_only=True,
            min_date=min_date,
        )

        if not predictions:
            return {"error": "No predictions found"}

        # Compute statistics
        adjustments = []
        for pred in predictions:
            if pred["calibrated_probability"] is not None:
                adj = pred["calibrated_probability"] - pred["raw_probability"]
                adjustments.append(adj)

        if not adjustments:
            return {"error": "No calibrated predictions found"}

        adjustments = np.array(adjustments)

        return {
            "n_predictions": len(adjustments),
            "mean_adjustment": float(np.mean(adjustments)),
            "std_adjustment": float(np.std(adjustments)),
            "overconfident_rate": float(np.mean(adjustments < 0)),
            "underconfident_rate": float(np.mean(adjustments > 0)),
            "max_adjustment": float(np.max(np.abs(adjustments))),
            "category": category or "all",
            "days_analyzed": days,
        }
