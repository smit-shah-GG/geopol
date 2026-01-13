"""
Provisional scoring for unresolved geopolitical predictions.

For predictions that haven't yet resolved, we can't calculate true Brier scores.
Instead, we calculate "provisional outcomes" based on current GDELT event signals
and weight them by time decay.

This allows us to:
1. Track performance on recent predictions before formal resolution
2. Get early warning of calibration drift
3. Provide interim performance metrics

Methodology:
- Calculate tension/cooperation indices from recent GDELT events
- Apply time decay weight (predictions closer to deadline get higher weight)
- Combine with resolved predictions using weighted average
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class ProvisionalScorer:
    """
    Calculates provisional scores for unresolved predictions.

    Uses GDELT event streams to estimate current trajectory toward outcomes.
    Weights provisional scores by time progression (more confidence as deadline approaches).
    """

    def __init__(self, gdelt_client=None, prediction_window_days: int = 30):
        """
        Initialize provisional scorer.

        Args:
            gdelt_client: Optional GDELTClient for fetching recent events
            prediction_window_days: Default prediction window (days)
        """
        self.gdelt_client = gdelt_client
        self.prediction_window_days = prediction_window_days

    def calculate_tension_index(
        self, entities: List[str], lookback_days: int = 7
    ) -> Optional[float]:
        """
        Calculate tension index from recent GDELT events.

        Tension index = (conflict events - cooperation events) / total events
        Range: [-1, 1] where -1 = pure cooperation, +1 = pure conflict

        Args:
            entities: List of entity names involved in prediction
            lookback_days: How many days to look back for events

        Returns:
            Tension index or None if insufficient data
        """
        if not self.gdelt_client or not entities:
            return None

        try:
            # Fetch recent events involving these entities
            # We'll look for both conflict (QuadClass 4) and cooperation (QuadClass 1)
            conflict_events = self._fetch_events_for_entities(
                entities, lookback_days, quad_class=4
            )

            cooperation_events = self._fetch_events_for_entities(
                entities, lookback_days, quad_class=1
            )

            total_events = len(conflict_events) + len(cooperation_events)

            if total_events == 0:
                logger.debug(f"No events found for entities {entities}")
                return None

            # Calculate tension index
            tension = (len(conflict_events) - len(cooperation_events)) / total_events

            logger.debug(
                f"Tension index for {entities}: {tension:.3f} "
                f"({len(conflict_events)} conflict, {len(cooperation_events)} cooperation)"
            )

            return float(tension)

        except Exception as e:
            logger.warning(f"Failed to calculate tension index: {e}")
            return None

    def _fetch_events_for_entities(
        self, entities: List[str], lookback_days: int, quad_class: int
    ) -> List[Dict]:
        """
        Fetch GDELT events for entities and quad class.

        Args:
            entities: Entity names
            lookback_days: Lookback window
            quad_class: CAMEO quad class (1-4)

        Returns:
            List of event dicts
        """
        # Build keyword query from entities
        keyword = " OR ".join(entities)

        # Fetch events
        events = self.gdelt_client.fetch_recent_events(
            timespan=f"{lookback_days}d", quad_classes=[quad_class], keyword=keyword
        )

        return events.to_dict("records") if not events.empty else []

    def calculate_time_decay_weight(
        self, timestamp: datetime, deadline: Optional[datetime] = None
    ) -> float:
        """
        Calculate time decay weight for provisional score.

        Weight increases linearly as prediction approaches deadline:
        - At creation (t=0): weight = 0.1 (very uncertain)
        - At 50% elapsed: weight = 0.5 (moderate confidence)
        - At deadline: weight = 1.0 (high confidence, treat as resolved)

        Args:
            timestamp: When prediction was made
            deadline: When prediction resolves (defaults to timestamp + prediction_window_days)

        Returns:
            Weight in [0.1, 1.0]
        """
        if deadline is None:
            deadline = timestamp + timedelta(days=self.prediction_window_days)

        now = datetime.now()

        # Handle edge cases
        if now <= timestamp:
            return 0.1  # Prediction just made, very uncertain

        if now >= deadline:
            return 1.0  # Past deadline, treat as fully resolved

        # Linear interpolation
        elapsed = (now - timestamp).total_seconds()
        total_duration = (deadline - timestamp).total_seconds()

        progress = elapsed / total_duration

        # Map progress [0, 1] to weight [0.1, 1.0]
        weight = 0.1 + 0.9 * progress

        return float(weight)

    def estimate_provisional_outcome(
        self, prediction: Dict, tension_index: Optional[float] = None
    ) -> float:
        """
        Estimate provisional outcome based on current signals.

        For conflict predictions:
        - tension_index > 0 suggests outcome = 1 (conflict likely)
        - tension_index < 0 suggests outcome = 0 (cooperation)

        For diplomatic predictions:
        - tension_index < 0 suggests outcome = 1 (cooperation likely)
        - tension_index > 0 suggests outcome = 0 (tension)

        Args:
            prediction: Prediction dict with 'category' and 'entities'
            tension_index: Pre-calculated tension index (optional)

        Returns:
            Provisional outcome probability in [0, 1]
        """
        category = prediction.get("category", "conflict")

        # Calculate tension index if not provided
        if tension_index is None:
            entities = prediction.get("entities", [])
            tension_index = self.calculate_tension_index(entities)

            # If we can't calculate tension, use neutral baseline
            if tension_index is None:
                return 0.5

        # Map tension to outcome based on category
        if category == "conflict":
            # Positive tension → higher conflict probability
            # Map [-1, 1] to [0, 1]
            outcome_prob = (tension_index + 1) / 2
        elif category == "diplomatic":
            # Negative tension (cooperation) → higher diplomatic success probability
            # Map [-1, 1] to [1, 0] (inverted)
            outcome_prob = (1 - tension_index) / 2
        else:
            # Economic: use neutral with slight positive bias for growth
            outcome_prob = 0.5 + 0.1 * tension_index

        # Clip to [0, 1]
        return float(np.clip(outcome_prob, 0.0, 1.0))

    def score_provisional(
        self, predictions: List[Dict], use_calibrated: bool = True
    ) -> Dict[str, float]:
        """
        Score unresolved predictions using provisional outcomes.

        Args:
            predictions: List of unresolved prediction dicts
            use_calibrated: Use calibrated vs raw probabilities

        Returns:
            Dict with provisional Brier score and metadata
        """
        unresolved = [p for p in predictions if p.get("outcome") is None]

        if not unresolved:
            return {
                "provisional_brier": None,
                "n_provisional": 0,
                "avg_weight": 0.0,
            }

        prob_key = "calibrated_probability" if use_calibrated else "raw_probability"

        provisional_scores = []
        weights = []

        for pred in unresolved:
            # Get predicted probability
            predicted_prob = pred.get(prob_key, pred["raw_probability"])

            # Calculate provisional outcome
            provisional_outcome = self.estimate_provisional_outcome(pred)

            # Calculate time decay weight
            timestamp = pred.get("timestamp", datetime.now())
            weight = self.calculate_time_decay_weight(timestamp)

            # Calculate Brier score for this prediction
            brier = (predicted_prob - provisional_outcome) ** 2

            provisional_scores.append(brier)
            weights.append(weight)

        # Weighted average Brier score
        weights_array = np.array(weights)
        scores_array = np.array(provisional_scores)

        weighted_brier = float(np.average(scores_array, weights=weights_array))

        return {
            "provisional_brier": weighted_brier,
            "n_provisional": len(unresolved),
            "avg_weight": float(np.mean(weights)),
        }

    def score_all(
        self,
        predictions: List[Dict],
        use_calibrated: bool = True,
        provisional_weight: float = 0.5,
    ) -> Dict[str, float]:
        """
        Score all predictions: resolved at full weight, provisional at reduced weight.

        Args:
            predictions: List of all prediction dicts (resolved + unresolved)
            use_calibrated: Use calibrated vs raw probabilities
            provisional_weight: Weight for provisional scores vs resolved (default 0.5)

        Returns:
            Dict with combined score and breakdown
        """
        resolved = [p for p in predictions if p.get("outcome") is not None]
        unresolved = [p for p in predictions if p.get("outcome") is None]

        # Score resolved predictions
        from .brier_scorer import BrierScorer

        resolved_score = None
        if resolved:
            scorer = BrierScorer()
            resolved_result = scorer.score_batch(resolved, use_calibrated)
            resolved_score = resolved_result["overall"]

        # Score provisional predictions
        provisional_result = self.score_provisional(unresolved, use_calibrated)
        provisional_score = provisional_result.get("provisional_brier")

        # Combine scores
        if resolved_score is not None and provisional_score is not None:
            # Weighted combination
            total_weight = len(resolved) * 1.0 + len(unresolved) * provisional_weight
            combined_score = (
                len(resolved) * 1.0 * resolved_score
                + len(unresolved) * provisional_weight * provisional_score
            ) / total_weight
        elif resolved_score is not None:
            combined_score = resolved_score
        elif provisional_score is not None:
            combined_score = provisional_score
        else:
            combined_score = None

        return {
            "combined_brier": combined_score,
            "resolved_brier": resolved_score,
            "provisional_brier": provisional_score,
            "n_resolved": len(resolved),
            "n_provisional": len(unresolved),
            "provisional_weight": provisional_weight,
        }
