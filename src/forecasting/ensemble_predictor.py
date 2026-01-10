"""
Ensemble predictor combining LLM and TKG predictions.

This module implements weighted voting between:
1. ReasoningOrchestrator (LLM-based scenario analysis)
2. TKGPredictor (graph-based pattern matching)

The ensemble:
- Normalizes probability distributions from both models
- Applies configurable weights (default: 0.6 LLM, 0.4 TKG)
- Performs temperature scaling for confidence calibration
- Tracks component contributions for explainability
- Provides graceful degradation if one model fails
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.forecasting.models import ForecastOutput, Scenario
from src.forecasting.reasoning_orchestrator import ReasoningOrchestrator
from src.forecasting.tkg_predictor import TKGPredictor

logger = logging.getLogger(__name__)


@dataclass
class ComponentPrediction:
    """Individual component's prediction with metadata."""

    component: str  # "llm" or "tkg"
    probability: float  # Raw probability
    confidence: float  # Model's confidence in this prediction
    available: bool  # Whether this component provided a prediction
    error: Optional[str] = None  # Error message if unavailable


@dataclass
class EnsemblePrediction:
    """Combined ensemble prediction with explainability."""

    final_probability: float  # Ensemble probability
    final_confidence: float  # Calibrated confidence
    llm_prediction: ComponentPrediction
    tkg_prediction: ComponentPrediction
    weights_used: Tuple[float, float]  # (llm_weight, tkg_weight)
    temperature: float  # Temperature scaling factor


class EnsemblePredictor:
    """
    Combines LLM and TKG predictions using weighted voting.

    Architecture:
    1. Both models make independent predictions
    2. Probabilities are normalized to [0, 1]
    3. Weighted average is computed: P = α*P_LLM + (1-α)*P_TKG
    4. Temperature scaling calibrates confidence
    5. Component contributions tracked for explainability

    Attributes:
        llm_orchestrator: ReasoningOrchestrator for LLM predictions
        tkg_predictor: TKGPredictor for graph-based predictions
        alpha: Weight for LLM predictions (default: 0.6)
        temperature: Temperature scaling factor (default: 1.0)
    """

    def __init__(
        self,
        llm_orchestrator: Optional[ReasoningOrchestrator] = None,
        tkg_predictor: Optional[TKGPredictor] = None,
        alpha: float = 0.6,
        temperature: float = 1.0,
    ):
        """
        Initialize ensemble predictor.

        Args:
            llm_orchestrator: ReasoningOrchestrator instance
            tkg_predictor: TKGPredictor instance
            alpha: Weight for LLM (0.0-1.0), TKG gets (1-alpha)
            temperature: Temperature for confidence calibration (>0)
                        - T < 1: Sharpen (more confident)
                        - T = 1: No change
                        - T > 1: Smooth (less confident)

        Raises:
            ValueError: If alpha not in [0, 1] or temperature <= 0
        """
        if not 0 <= alpha <= 1:
            raise ValueError(f"Alpha must be in [0, 1], got {alpha}")
        if temperature <= 0:
            raise ValueError(f"Temperature must be > 0, got {temperature}")

        self.llm_orchestrator = llm_orchestrator
        self.tkg_predictor = tkg_predictor
        self.alpha = alpha
        self.temperature = temperature

        logger.info(
            f"Initialized ensemble with α={alpha:.2f} (LLM), "
            f"β={1-alpha:.2f} (TKG), T={temperature:.2f}"
        )

    def predict(
        self,
        question: str,
        context: Optional[List[str]] = None,
        entity1: Optional[str] = None,
        relation: Optional[str] = None,
        entity2: Optional[str] = None,
    ) -> Tuple[EnsemblePrediction, ForecastOutput]:
        """
        Generate ensemble forecast combining LLM and TKG predictions.

        Args:
            question: Natural language forecasting question
            context: Optional context strings for LLM
            entity1: Source entity for TKG query (extracted from LLM if None)
            relation: Relation type for TKG query (extracted from LLM if None)
            entity2: Target entity for TKG query (extracted from LLM if None)

        Returns:
            Tuple of:
            - EnsemblePrediction: Component contributions and final prediction
            - ForecastOutput: Full forecast with reasoning chains

        Note:
            If one component fails, falls back to other component with notification.
        """
        # Get LLM prediction
        llm_pred = self._get_llm_prediction(question, context)

        # Extract entities from LLM for TKG query if not provided
        if llm_pred.available and not all([entity1, relation, entity2]):
            entity1, relation, entity2 = self._extract_entities_from_llm(
                llm_pred, question
            )

        # Get TKG prediction
        tkg_pred = self._get_tkg_prediction(entity1, relation, entity2)

        # Combine predictions
        final_prob, final_conf = self._combine_predictions(llm_pred, tkg_pred)

        # Create ensemble prediction metadata
        ensemble_pred = EnsemblePrediction(
            final_probability=final_prob,
            final_confidence=final_conf,
            llm_prediction=llm_pred,
            tkg_prediction=tkg_pred,
            weights_used=(self.alpha, 1 - self.alpha),
            temperature=self.temperature,
        )

        # Build full forecast output (use LLM's ForecastOutput as base)
        if llm_pred.available:
            # We stored the full ForecastOutput in the prediction
            forecast = self._forecast_output
            # Update with ensemble values
            forecast.probability = final_prob
            forecast.confidence = final_conf

            # Add ensemble explanation to reasoning
            ensemble_info = self._build_ensemble_explanation(ensemble_pred)
            forecast.reasoning_summary += f"\n\nEnsemble: {ensemble_info}"

            # Add TKG patterns to evidence if available
            if tkg_pred.available:
                forecast.evidence_sources.append("Graph pattern analysis (TKG)")
        else:
            # Fallback: create basic forecast if LLM failed
            from src.forecasting.models import ForecastOutput, ScenarioTree, Scenario
            from datetime import datetime

            forecast = ForecastOutput(
                question=question,
                prediction=f"Ensemble prediction (probability: {final_prob:.2f})",
                probability=final_prob,
                confidence=final_conf,
                scenario_tree=ScenarioTree(
                    question=question,
                    root_scenario=Scenario(
                        scenario_id="ensemble",
                        description="Ensemble prediction (LLM unavailable)",
                        probability=final_prob,
                    ),
                    scenarios={},
                ),
                selected_scenario_ids=[],
                reasoning_summary=self._build_ensemble_explanation(ensemble_pred),
                evidence_sources=["Graph pattern analysis (TKG)"]
                if tkg_pred.available
                else [],
                timestamp=datetime.now(),
            )

        return ensemble_pred, forecast

    def _get_llm_prediction(
        self, question: str, context: Optional[List[str]]
    ) -> ComponentPrediction:
        """
        Get prediction from LLM orchestrator.

        Returns:
            ComponentPrediction with LLM's probability and confidence
        """
        if self.llm_orchestrator is None:
            return ComponentPrediction(
                component="llm",
                probability=0.5,
                confidence=0.0,
                available=False,
                error="LLM orchestrator not initialized",
            )

        try:
            forecast = self.llm_orchestrator.forecast(
                question=question, context=context or []
            )

            # Store forecast for later use
            self._forecast_output = forecast

            return ComponentPrediction(
                component="llm",
                probability=forecast.probability,
                confidence=forecast.confidence,
                available=True,
            )

        except Exception as e:
            logger.error(f"LLM prediction failed: {e}")
            return ComponentPrediction(
                component="llm",
                probability=0.5,
                confidence=0.0,
                available=False,
                error=str(e),
            )

    def _get_tkg_prediction(
        self,
        entity1: Optional[str],
        relation: Optional[str],
        entity2: Optional[str],
    ) -> ComponentPrediction:
        """
        Get prediction from TKG predictor.

        Returns:
            ComponentPrediction with TKG's probability and confidence
        """
        if self.tkg_predictor is None or not self.tkg_predictor.trained:
            return ComponentPrediction(
                component="tkg",
                probability=0.5,
                confidence=0.0,
                available=False,
                error="TKG predictor not initialized or not trained",
            )

        # Need at least 2 out of 3 components for query
        if sum([x is not None for x in [entity1, relation, entity2]]) < 2:
            return ComponentPrediction(
                component="tkg",
                probability=0.5,
                confidence=0.0,
                available=False,
                error="Insufficient entities for TKG query",
            )

        try:
            # Query TKG for event probability
            predictions = self.tkg_predictor.predict_future_events(
                entity1=entity1,
                relation=relation,
                entity2=entity2,
                k=1,
                apply_decay=True,
            )

            if predictions:
                # Use top prediction's confidence as both probability and confidence
                confidence = predictions[0]["confidence"]
                return ComponentPrediction(
                    component="tkg",
                    probability=confidence,
                    confidence=confidence,
                    available=True,
                )
            else:
                return ComponentPrediction(
                    component="tkg",
                    probability=0.5,
                    confidence=0.0,
                    available=False,
                    error="No patterns found in graph",
                )

        except Exception as e:
            logger.error(f"TKG prediction failed: {e}")
            return ComponentPrediction(
                component="tkg",
                probability=0.5,
                confidence=0.0,
                available=False,
                error=str(e),
            )

    def _extract_entities_from_llm(
        self, llm_pred: ComponentPrediction, question: str
    ) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """
        Extract entities from LLM forecast for TKG query.

        Simple extraction: takes first two entities from top scenario
        and infers relation from question keywords.

        Returns:
            Tuple of (entity1, relation, entity2)
        """
        if not hasattr(self, "_forecast_output"):
            return None, None, None

        forecast = self._forecast_output
        scenarios = list(forecast.scenario_tree.scenarios.values())

        if not scenarios:
            return None, None, None

        # Get top scenario
        top_scenario = max(scenarios, key=lambda s: s.probability)

        # Extract entities
        entities = top_scenario.entities
        if len(entities) >= 2:
            entity1 = entities[0].name
            entity2 = entities[1].name
        elif len(entities) == 1:
            entity1 = entities[0].name
            entity2 = None
        else:
            return None, None, None

        # Infer relation from question (simple heuristics)
        question_lower = question.lower()
        if any(
            word in question_lower
            for word in ["conflict", "escalate", "war", "attack"]
        ):
            relation = "CONFLICT"
        elif any(
            word in question_lower
            for word in ["cooperate", "alliance", "agreement", "treaty"]
        ):
            relation = "COOPERATION"
        elif any(word in question_lower for word in ["sanction", "restrict"]):
            relation = "SANCTION"
        else:
            relation = "INTERACT"  # Generic relation

        return entity1, relation, entity2

    def _combine_predictions(
        self,
        llm_pred: ComponentPrediction,
        tkg_pred: ComponentPrediction,
    ) -> Tuple[float, float]:
        """
        Combine LLM and TKG predictions using weighted voting.

        Implements:
        - Weighted average: P = α*P_LLM + (1-α)*P_TKG
        - Confidence combination: weighted by availability
        - Temperature scaling for calibration
        - Graceful degradation if one component unavailable

        Returns:
            Tuple of (final_probability, final_confidence)
        """
        # Case 1: Both available - weighted average
        if llm_pred.available and tkg_pred.available:
            prob = self.alpha * llm_pred.probability + (1 - self.alpha) * tkg_pred.probability
            conf = self.alpha * llm_pred.confidence + (1 - self.alpha) * tkg_pred.confidence

            logger.info(
                f"Ensemble: LLM={llm_pred.probability:.3f}, "
                f"TKG={tkg_pred.probability:.3f}, "
                f"Combined={prob:.3f}"
            )

        # Case 2: Only LLM available
        elif llm_pred.available:
            prob = llm_pred.probability
            conf = llm_pred.confidence * 0.8  # Penalty for missing TKG
            logger.warning(f"TKG unavailable: {tkg_pred.error}. Using LLM only.")

        # Case 3: Only TKG available
        elif tkg_pred.available:
            prob = tkg_pred.probability
            conf = tkg_pred.confidence * 0.8  # Penalty for missing LLM
            logger.warning(f"LLM unavailable: {llm_pred.error}. Using TKG only.")

        # Case 4: Neither available - return uninformative prior
        else:
            prob = 0.5
            conf = 0.0
            logger.error(
                f"Both components unavailable. LLM: {llm_pred.error}, "
                f"TKG: {tkg_pred.error}"
            )

        # Apply temperature scaling to confidence
        calibrated_conf = self._apply_temperature_scaling(conf)

        # Ensure valid range
        prob = np.clip(prob, 0.0, 1.0)
        calibrated_conf = np.clip(calibrated_conf, 0.0, 1.0)

        return float(prob), float(calibrated_conf)

    def _apply_temperature_scaling(self, confidence: float) -> float:
        """
        Apply temperature scaling to calibrate confidence.

        Formula: c' = c^(1/T)
        - T < 1: Sharpen (increase confidence)
        - T = 1: No change
        - T > 1: Smooth (decrease confidence)

        Args:
            confidence: Raw confidence score

        Returns:
            Calibrated confidence score
        """
        if self.temperature == 1.0:
            return confidence

        # Apply power scaling
        calibrated = confidence ** (1.0 / self.temperature)

        return calibrated

    def _build_ensemble_explanation(self, ensemble_pred: EnsemblePrediction) -> str:
        """
        Build human-readable explanation of ensemble decision.

        Returns:
            Explanation string showing component contributions
        """
        parts = []

        llm = ensemble_pred.llm_prediction
        tkg = ensemble_pred.tkg_prediction

        # Component status
        if llm.available and tkg.available:
            parts.append(
                f"Combined LLM (α={self.alpha:.2f}, P={llm.probability:.3f}) "
                f"and TKG (β={1-self.alpha:.2f}, P={tkg.probability:.3f})"
            )
        elif llm.available:
            parts.append(f"Used LLM only (TKG unavailable: {tkg.error})")
        elif tkg.available:
            parts.append(f"Used TKG only (LLM unavailable: {llm.error})")
        else:
            parts.append("Both components unavailable - returning uninformative prior")

        # Final values
        parts.append(
            f"Final probability: {ensemble_pred.final_probability:.3f}, "
            f"confidence: {ensemble_pred.final_confidence:.3f} "
            f"(T={self.temperature:.2f})"
        )

        return " ".join(parts)

    def update_weights(self, alpha: float) -> None:
        """
        Update ensemble weights dynamically.

        Args:
            alpha: New LLM weight (0.0-1.0)

        Raises:
            ValueError: If alpha not in [0, 1]
        """
        if not 0 <= alpha <= 1:
            raise ValueError(f"Alpha must be in [0, 1], got {alpha}")

        old_alpha = self.alpha
        self.alpha = alpha
        logger.info(f"Updated ensemble weights: α {old_alpha:.2f} -> {alpha:.2f}")

    def update_temperature(self, temperature: float) -> None:
        """
        Update temperature scaling factor.

        Args:
            temperature: New temperature (> 0)

        Raises:
            ValueError: If temperature <= 0
        """
        if temperature <= 0:
            raise ValueError(f"Temperature must be > 0, got {temperature}")

        old_temp = self.temperature
        self.temperature = temperature
        logger.info(f"Updated temperature: {old_temp:.2f} -> {temperature:.2f}")
