"""
Graph-based validation for LLM-generated scenarios.

This module provides graph-based scenario validation using temporal
knowledge graph predictions. It validates LLM scenarios against:
1. Historical graph patterns (TKG predictions)
2. Entity relationship plausibility
3. Temporal consistency with recent events
4. Structural consistency (event chains)

The validator generates structured feedback that the LLM can use
to refine scenarios, creating a bidirectional feedback loop between
symbolic graph reasoning and neural language modeling.
"""

import logging
from typing import Dict, List, Optional, Union

import networkx as nx

from src.forecasting.models import Scenario, ValidationFeedback
from src.forecasting.tkg_predictor import TKGPredictor

logger = logging.getLogger(__name__)


class GraphValidator:
    """
    Validates LLM scenarios against temporal knowledge graph patterns.

    The validator:
    1. Extracts structured events from scenario descriptions
    2. Queries TKG predictor for plausibility scores
    3. Identifies contradictions with graph patterns
    4. Generates concrete suggestions based on historical precedents
    5. Computes precedent score for overall scenario

    Attributes:
        tkg_predictor: TKGPredictor instance for graph-based predictions
        confidence_threshold: Minimum confidence for valid events (default: 0.15)
        precedent_weight: Weight for precedent in final scoring (default: 0.7)
    """

    def __init__(
        self,
        tkg_predictor: Optional[TKGPredictor] = None,
        confidence_threshold: float = 0.15,
        precedent_weight: float = 0.7,
    ):
        """
        Initialize graph validator.

        Args:
            tkg_predictor: TKGPredictor instance (created if None)
            confidence_threshold: Minimum confidence for plausibility
            precedent_weight: Weight for precedent in scoring [0, 1]
        """
        self.tkg_predictor = tkg_predictor
        self.confidence_threshold = confidence_threshold
        self.precedent_weight = precedent_weight

        logger.info(f"Initialized GraphValidator with threshold={confidence_threshold}, "
                   f"precedent_weight={precedent_weight}")

    def validate_scenario(
        self,
        scenario: Union[Scenario, Dict],
        extract_events: bool = True
    ) -> ValidationFeedback:
        """
        Validate scenario against graph patterns.

        Args:
            scenario: Scenario object or dict with scenario data
            extract_events: Whether to extract events from description

        Returns:
            ValidationFeedback with graph-based validation results
        """
        if not self.tkg_predictor or not self.tkg_predictor.trained:
            logger.warning("TKG predictor not available or not trained, using fallback")
            return self._fallback_validation(scenario)

        # Extract scenario components
        if isinstance(scenario, dict):
            scenario_id = scenario.get('id', scenario.get('scenario_id', 'unknown'))
            description = scenario.get('description', '')
            probability = scenario.get('probability', 0.5)
            entities = scenario.get('entities', [])
        else:
            scenario_id = scenario.scenario_id
            description = scenario.description
            probability = scenario.probability
            entities = [{'name': e.name, 'type': e.type} for e in scenario.entities]

        # Extract events from scenario
        if extract_events:
            events = self._extract_events_from_scenario(scenario)
        else:
            events = scenario.get('events', []) if isinstance(scenario, dict) else []

        # Validate each event against graph
        event_validations = []
        for event in events:
            validation = self._validate_event(event)
            event_validations.append(validation)

        # Compute overall precedent score
        precedent_score = self._compute_precedent_score(event_validations)

        # Identify contradictions
        contradictions = self._identify_contradictions(
            event_validations,
            entities,
            description
        )

        # Generate suggestions
        suggestions = self._generate_suggestions(
            event_validations,
            contradictions,
            entities
        )

        # Extract historical patterns
        historical_patterns = self._extract_historical_patterns(event_validations)

        # Determine overall validity
        is_valid = (
            precedent_score >= self.confidence_threshold and
            len(contradictions) == 0
        )

        # Combine graph precedent with scenario probability
        final_confidence = (
            self.precedent_weight * precedent_score +
            (1 - self.precedent_weight) * probability
        )

        return ValidationFeedback(
            scenario_id=scenario_id,
            is_valid=is_valid,
            confidence_score=final_confidence,
            historical_patterns=historical_patterns,
            contradictions=contradictions,
            suggestions=suggestions,
        )

    def _extract_events_from_scenario(
        self,
        scenario: Union[Scenario, Dict]
    ) -> List[Dict[str, str]]:
        """
        Extract structured events from scenario.

        For now, uses simple entity-pair extraction.
        TODO: Use NLP to extract full (entity1, relation, entity2) triples.

        Args:
            scenario: Scenario object or dict

        Returns:
            List of event dictionaries with keys: entity1, relation, entity2
        """
        events = []

        # Extract entities
        if isinstance(scenario, dict):
            entities = scenario.get('entities', [])
            description = scenario.get('description', '')
        else:
            entities = [e.name for e in scenario.entities]
            description = scenario.description

        # Simple heuristic: create events between entity pairs
        # Infer relation type from keywords in description
        for i, entity1 in enumerate(entities):
            for entity2 in entities[i+1:]:
                # Infer relation from description keywords
                relation = self._infer_relation_from_text(description)

                events.append({
                    'entity1': entity1,
                    'relation': relation,
                    'entity2': entity2
                })

        return events

    def _infer_relation_from_text(self, text: str) -> str:
        """
        Infer relation type from text using keywords.

        Args:
            text: Description text

        Returns:
            Inferred relation type
        """
        text_lower = text.lower()

        # Conflict keywords
        if any(kw in text_lower for kw in ['conflict', 'attack', 'strike', 'war', 'military']):
            return 'MILITARY_CONFLICT'

        # Cooperation keywords
        if any(kw in text_lower for kw in ['cooperate', 'partner', 'agree', 'alliance', 'diplomatic']):
            return 'DIPLOMATIC_COOPERATION'

        # Trade keywords
        if any(kw in text_lower for kw in ['trade', 'economic', 'sanction', 'embargo']):
            return 'TRADE_AGREEMENT'

        # Default
        return 'UNKNOWN_RELATION'

    def _validate_event(self, event: Dict[str, str]) -> Dict[str, Union[str, float, bool]]:
        """
        Validate single event against graph patterns.

        Args:
            event: Event dict with entity1, relation, entity2

        Returns:
            Validation result with confidence and similar events
        """
        try:
            result = self.tkg_predictor.validate_scenario_event(event)
            return result
        except Exception as e:
            logger.warning(f"Event validation failed: {e}")
            return {
                'valid': False,
                'confidence': 0.0,
                'reason': str(e),
                'similar_events': []
            }

    def _compute_precedent_score(
        self,
        event_validations: List[Dict[str, Union[str, float, bool]]]
    ) -> float:
        """
        Compute overall precedent score from event validations.

        Uses average confidence of valid events.

        Args:
            event_validations: List of event validation results

        Returns:
            Precedent score in [0, 1]
        """
        if not event_validations:
            return 0.5  # Neutral if no events

        # Average confidence of all events
        confidences = [v['confidence'] for v in event_validations]
        return sum(confidences) / len(confidences)

    def _identify_contradictions(
        self,
        event_validations: List[Dict[str, Union[str, float, bool]]],
        entities: List[Union[str, Dict]],
        description: str
    ) -> List[str]:
        """
        Identify contradictions between scenario and graph patterns.

        Args:
            event_validations: Event validation results
            entities: List of entities in scenario
            description: Scenario description

        Returns:
            List of contradiction descriptions
        """
        contradictions = []

        # Check for low-confidence events
        for validation in event_validations:
            if validation['confidence'] < self.confidence_threshold:
                entity1 = validation.get('entity1', 'unknown')
                entity2 = validation.get('entity2', 'unknown')
                relation = validation.get('relation', 'unknown')

                contradictions.append(
                    f"Event ({entity1}, {relation}, {entity2}) unlikely based on "
                    f"historical patterns (confidence: {validation['confidence']:.2f})"
                )

        # Check for conflicting patterns
        for validation in event_validations:
            similar_events = validation.get('similar_events', [])
            if similar_events:
                # Check if similar events have different relations
                relations = set(e['relation'] for e in similar_events)
                if len(relations) > 1 and validation['confidence'] < 0.3:
                    contradictions.append(
                        f"Mixed historical precedent for {validation.get('entity1')} - "
                        f"{validation.get('entity2')} relationship"
                    )

        return contradictions

    def _generate_suggestions(
        self,
        event_validations: List[Dict[str, Union[str, float, bool]]],
        contradictions: List[str],
        entities: List[Union[str, Dict]]
    ) -> List[str]:
        """
        Generate suggestions for scenario refinement.

        Args:
            event_validations: Event validation results
            contradictions: Identified contradictions
            entities: Entities in scenario

        Returns:
            List of concrete suggestions
        """
        suggestions = []

        # Suggest higher-confidence alternatives
        for validation in event_validations:
            if validation['confidence'] < self.confidence_threshold:
                similar = validation.get('similar_events', [])
                if similar:
                    top_alternative = similar[0]
                    suggestions.append(
                        f"Consider {top_alternative['relation']} instead of "
                        f"{validation.get('relation')} for {validation.get('entity1')} - "
                        f"{validation.get('entity2')} (historical confidence: "
                        f"{top_alternative['confidence']:.2f})"
                    )

        # Suggest based on entity patterns
        if len(event_validations) == 0 and len(entities) >= 2:
            suggestions.append(
                "Consider incorporating more specific event details to enable "
                "graph-based validation"
            )

        # Limit suggestions to most actionable
        return suggestions[:3]

    def _extract_historical_patterns(
        self,
        event_validations: List[Dict[str, Union[str, float, bool]]]
    ) -> List[str]:
        """
        Extract historical pattern descriptions from validations.

        Args:
            event_validations: Event validation results

        Returns:
            List of historical pattern descriptions
        """
        patterns = []

        for validation in event_validations:
            similar = validation.get('similar_events', [])
            for event in similar[:2]:  # Top 2 similar events
                pattern_desc = (
                    f"Historical pattern: {event['entity1']} - {event['relation']} - "
                    f"{event['entity2']} (confidence: {event['confidence']:.2f})"
                )
                patterns.append(pattern_desc)

        # Deduplicate and limit
        unique_patterns = list(dict.fromkeys(patterns))
        return unique_patterns[:5]

    def _fallback_validation(
        self,
        scenario: Union[Scenario, Dict]
    ) -> ValidationFeedback:
        """
        Fallback validation when TKG predictor unavailable.

        Returns neutral validation feedback.

        Args:
            scenario: Scenario object or dict

        Returns:
            ValidationFeedback with neutral scores
        """
        if isinstance(scenario, dict):
            scenario_id = scenario.get('id', scenario.get('scenario_id', 'unknown'))
        else:
            scenario_id = scenario.scenario_id

        return ValidationFeedback(
            scenario_id=scenario_id,
            is_valid=True,
            confidence_score=0.5,
            historical_patterns=["TKG validation unavailable - using fallback"],
            contradictions=[],
            suggestions=["Train TKG predictor on historical graph for validation"],
        )

    def set_predictor(self, predictor: TKGPredictor) -> None:
        """
        Set or update TKG predictor.

        Args:
            predictor: TKGPredictor instance
        """
        self.tkg_predictor = predictor
        logger.info("TKG predictor updated")
