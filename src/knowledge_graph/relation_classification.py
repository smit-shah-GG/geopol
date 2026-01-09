"""
Relation classification system for converting CAMEO event codes to typed relations.

This module handles:
1. Mapping event codes + quad_class to relation types
2. Confidence calculation from NumMentions, GoldsteinScale, and Tone
3. Bayesian confidence aggregation for duplicate events
4. Temporal granularity detection
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, List
from enum import Enum
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class RelationType(str, Enum):
    """Standardized relation types in knowledge graphs."""
    # Diplomatic relations
    MAKES_STATEMENT = "makes_statement"
    APPEALS = "appeals"
    EXPRESSES_INTENT = "expresses_intent_to_cooperate"
    CONSULTS = "consults"
    NEGOTIATES = "negotiates"
    COOPERATES = "cooperates"
    PRAISES = "praises"
    DEFENDS = "defends"
    RECOGNIZES = "grants_recognition"
    APOLOGIZES = "apologizes"
    FORGIVES = "forgives"
    SIGNS_AGREEMENT = "signs_agreement"

    # Conflict relations
    THREATENS = "threatens"
    ACCUSES = "accuses"
    DEMANDS = "demands"
    PROTESTS = "protests"
    ASSAULTS = "assaults"
    FIGHTS = "fights"
    USES_WEAPONS = "uses_weapons"
    BOMBARDS = "bombards"
    OCCUPIES = "occupies"
    VIOLATES_CEASEFIRE = "violates_ceasefire"
    KILLS = "kills"
    TORTURES = "tortures"
    ABDUCTS = "abducts"

    # Material cooperation
    PROVIDES_AID = "provides_aid"
    PROVIDES_TRADE = "provides_trade"
    PROVIDES_MATERIAL = "provides_material"

    # Verbal conflict
    VERBAL_CONFLICT = "verbal_conflict"

    # Generic fallback
    UNKNOWN = "unknown"


class TemporalGranularity(str, Enum):
    """Temporal granularity for different event types."""
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"


@dataclass
class Relation:
    """Canonical relation representation."""
    source_entity: str
    target_entity: str
    relation_type: RelationType
    timestamp: str  # ISO format
    confidence: float  # [0, 1]
    quad_class: int  # 1-4
    num_mentions: int = 1
    goldstein_scale: Optional[float] = None
    tone: Optional[float] = None
    event_codes: List[str] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)

    def __hash__(self):
        return hash((self.source_entity, self.target_entity, self.relation_type, self.timestamp))

    def __eq__(self, other):
        if isinstance(other, Relation):
            return (self.source_entity == other.source_entity and
                    self.target_entity == other.target_entity and
                    self.relation_type == other.relation_type and
                    self.timestamp == other.timestamp)
        return False


class RelationClassifier:
    """
    Converts CAMEO event codes and QuadClass to typed relations with confidence scoring.
    """

    def __init__(self):
        """Initialize relation classifier with event code mappings."""
        self.event_code_map = self._build_event_code_map()
        self.quadclass_map = self._build_quadclass_map()

    def _build_event_code_map(self) -> Dict[str, RelationType]:
        """
        Build mapping from CAMEO event codes to relation types.

        Returns:
            Dictionary mapping event codes to RelationType
        """
        mapping = {
            # Make public statement
            '01': RelationType.MAKES_STATEMENT,
            '010': RelationType.MAKES_STATEMENT,

            # Appeal
            '02': RelationType.APPEALS,
            '020': RelationType.APPEALS,

            # Express intent to cooperate
            '03': RelationType.EXPRESSES_INTENT,
            '030': RelationType.EXPRESSES_INTENT,

            # Consult
            '04': RelationType.CONSULTS,
            '040': RelationType.CONSULTS,
            '041': RelationType.CONSULTS,

            # Diplomatic cooperation
            '05': RelationType.COOPERATES,
            '050': RelationType.COOPERATES,
            '051': RelationType.PRAISES,
            '052': RelationType.DEFENDS,
            '053': RelationType.DEFENDS,
            '054': RelationType.RECOGNIZES,
            '055': RelationType.APOLOGIZES,
            '056': RelationType.FORGIVES,
            '057': RelationType.SIGNS_AGREEMENT,

            # Negotiate
            '06': RelationType.NEGOTIATES,
            '060': RelationType.NEGOTIATES,
            '061': RelationType.NEGOTIATES,

            # Protest/demand
            '07': RelationType.PROTESTS,
            '070': RelationType.PROTESTS,
            '071': RelationType.DEMANDS,
            '072': RelationType.DEMANDS,

            # Threats
            '08': RelationType.THREATENS,
            '080': RelationType.THREATENS,
            '081': RelationType.THREATENS,

            # Accusations
            '09': RelationType.ACCUSES,
            '090': RelationType.ACCUSES,

            # Assault/violence
            '18': RelationType.ASSAULTS,
            '180': RelationType.ASSAULTS,
            '181': RelationType.ABDUCTS,
            '182': RelationType.ASSAULTS,
            '183': RelationType.TORTURES,
            '184': RelationType.KILLS,

            # Fight
            '19': RelationType.FIGHTS,
            '190': RelationType.USES_WEAPONS,
            '191': RelationType.BOMBARDS,
            '192': RelationType.OCCUPIES,
            '193': RelationType.FIGHTS,
            '194': RelationType.FIGHTS,
            '195': RelationType.USES_WEAPONS,
            '196': RelationType.VIOLATES_CEASEFIRE,

            # Mass violence
            '20': RelationType.USES_WEAPONS,
            '200': RelationType.USES_WEAPONS,

            # Material cooperation
            '11': RelationType.PROVIDES_AID,
            '110': RelationType.PROVIDES_AID,
            '111': RelationType.PROVIDES_TRADE,
            '112': RelationType.PROVIDES_MATERIAL,

            # Default for unmapped codes
        }
        return mapping

    def _build_quadclass_map(self) -> Dict[int, Tuple[List[str], TemporalGranularity]]:
        """
        Build mapping from QuadClass to expected relation types and temporal granularity.

        Returns:
            Dictionary mapping quad_class to (expected_types, temporal_granularity)
        """
        return {
            1: (  # Verbal cooperation
                [
                    RelationType.MAKES_STATEMENT,
                    RelationType.APPEALS,
                    RelationType.EXPRESSES_INTENT,
                    RelationType.CONSULTS,
                    RelationType.COOPERATES,
                    RelationType.PRAISES,
                    RelationType.DEFENDS,
                    RelationType.RECOGNIZES,
                    RelationType.NEGOTIATES,
                    RelationType.SIGNS_AGREEMENT,
                ],
                TemporalGranularity.WEEKLY
            ),
            2: (  # Material cooperation
                [
                    RelationType.PROVIDES_AID,
                    RelationType.PROVIDES_TRADE,
                    RelationType.PROVIDES_MATERIAL,
                ],
                TemporalGranularity.MONTHLY
            ),
            3: (  # Verbal conflict
                [
                    RelationType.THREATENS,
                    RelationType.ACCUSES,
                    RelationType.DEMANDS,
                    RelationType.PROTESTS,
                    RelationType.VERBAL_CONFLICT,
                ],
                TemporalGranularity.DAILY
            ),
            4: (  # Material conflict
                [
                    RelationType.ASSAULTS,
                    RelationType.FIGHTS,
                    RelationType.USES_WEAPONS,
                    RelationType.BOMBARDS,
                    RelationType.OCCUPIES,
                    RelationType.VIOLATES_CEASEFIRE,
                    RelationType.KILLS,
                    RelationType.TORTURES,
                    RelationType.ABDUCTS,
                ],
                TemporalGranularity.DAILY
            ),
        }

    def classify_event(self,
                      source_entity: str,
                      target_entity: str,
                      event_code: Optional[str],
                      quad_class: Optional[int],
                      timestamp: str,
                      num_mentions: Optional[int] = None,
                      goldstein_scale: Optional[float] = None,
                      tone: Optional[float] = None) -> Optional[Relation]:
        """
        Classify a GDELT event to a typed relation with confidence.

        Args:
            source_entity: Source entity ID
            target_entity: Target entity ID
            event_code: CAMEO event code (e.g., '184' for KILL)
            quad_class: GDELT QuadClass (1-4)
            timestamp: ISO format timestamp
            num_mentions: Number of mentions (for confidence)
            goldstein_scale: Goldstein scale score (-10 to +10)
            tone: Average tone (-100 to +100)

        Returns:
            Relation object with calculated confidence or None if classification fails
        """
        if not source_entity or not target_entity or quad_class is None:
            return None

        # Map event code to relation type
        if event_code and event_code.strip() in self.event_code_map:
            relation_type = self.event_code_map[event_code.strip()]
        else:
            # Fallback based on quad_class
            relation_type = RelationType.UNKNOWN

        # Calculate confidence
        confidence = self._calculate_confidence(
            num_mentions=num_mentions,
            goldstein_scale=goldstein_scale,
            tone=tone,
            quad_class=quad_class
        )

        # Get temporal granularity
        temporal_gran = self._get_temporal_granularity(quad_class)

        relation = Relation(
            source_entity=source_entity,
            target_entity=target_entity,
            relation_type=relation_type,
            timestamp=timestamp,
            confidence=confidence,
            quad_class=quad_class,
            num_mentions=num_mentions or 1,
            goldstein_scale=goldstein_scale,
            tone=tone,
            event_codes=[event_code] if event_code else [],
            metadata={'temporal_granularity': temporal_gran.value}
        )

        return relation

    def _calculate_confidence(self,
                             num_mentions: Optional[int] = None,
                             goldstein_scale: Optional[float] = None,
                             tone: Optional[float] = None,
                             quad_class: Optional[int] = None) -> float:
        """
        Calculate confidence score from multiple factors.

        Confidence components:
        - num_mentions: log-scaled (1 mention = 0.3, 10 = 0.5, 100 = 0.7, 1000 = 0.9)
        - goldstein_scale: magnitude indicates strength
        - tone: consistency of coverage (close to scale indicates consistent reporting)

        Args:
            num_mentions: Number of mentions
            goldstein_scale: Goldstein scale (-10 to +10)
            tone: Average tone (-100 to +100)
            quad_class: GDELT QuadClass for calibration

        Returns:
            Confidence score [0, 1]
        """
        components = []

        # Mentions component: log scale, capped at 1.0
        if num_mentions and num_mentions > 0:
            # Map: 1->0.3, 10->0.5, 100->0.7, 1000->0.9
            import math
            mention_score = min(0.9, 0.3 + (math.log10(num_mentions) * 0.2))
            components.append(mention_score)

        # Goldstein component: magnitude + consistency
        if goldstein_scale is not None:
            # Normalize to [0, 1]
            goldstein_norm = min(1.0, abs(goldstein_scale) / 10.0)
            components.append(goldstein_norm * 0.8)  # Weight at 0.8

        # Tone component: consistency indicator
        if tone is not None and goldstein_scale is not None:
            # Check if tone aligns with goldstein scale
            goldstein_sign = 1 if goldstein_scale > 0 else (-1 if goldstein_scale < 0 else 0)
            tone_sign = 1 if tone > 0 else (-1 if tone < 0 else 0)

            # If signs align (or scale is near zero), increase confidence
            if goldstein_sign == tone_sign or abs(goldstein_scale) < 1:
                tone_consistency = 0.8
            else:
                tone_consistency = 0.4

            components.append(tone_consistency * 0.5)  # Weight at 0.5

        # Baseline confidence based on quad_class
        if quad_class in [1, 3]:  # Diplomatic or verbal conflict
            baseline = 0.4
        elif quad_class in [2, 4]:  # Material cooperation or conflict
            baseline = 0.5
        else:
            baseline = 0.3

        if not components:
            return baseline

        # Average components with baseline
        avg_component = sum(components) / len(components)
        confidence = (baseline * 0.3) + (avg_component * 0.7)

        return min(1.0, max(0.1, confidence))

    def aggregate_confidence_bayesian(self, confidences: List[float]) -> float:
        """
        Aggregate multiple confidence scores using Bayesian updating.

        Assumes each confidence represents independent evidence of relation strength.

        Args:
            confidences: List of individual confidence scores [0, 1]

        Returns:
            Aggregated confidence [0, 1]
        """
        if not confidences:
            return 0.5

        # Use Bayesian model: P(relation) ∝ ∏ P(evidence_i)
        # But convert odds ratios to avoid numerical issues
        log_odds = 0.0
        prior_odds = 1.0  # Prior = 0.5 (neutral)

        for conf in confidences:
            if conf > 0.99:
                conf = 0.99  # Avoid log(0)
            if conf < 0.01:
                conf = 0.01

            # Likelihood ratio for this evidence
            odds = conf / (1 - conf)
            log_odds += (1 if conf > 0.5 else -1) * abs(conf - 0.5) * 5

        # Convert back to probability
        from math import tanh
        aggregated = 0.5 + (tanh(log_odds) / 2)  # Normalized to [0, 1]

        return min(1.0, max(0.0, aggregated))

    def aggregate_duplicate_relations(self, relations: List[Relation]) -> Relation:
        """
        Aggregate duplicate relations with Bayesian confidence fusion.

        Assumes all relations refer to same source-target-type triple at same time.

        Args:
            relations: List of duplicate relations

        Returns:
            Aggregated relation with fused confidence
        """
        if not relations:
            raise ValueError("Cannot aggregate empty relation list")

        if len(relations) == 1:
            return relations[0]

        # Use first relation as template
        base = relations[0]

        # Aggregate confidences
        confidences = [r.confidence for r in relations]
        aggregated_confidence = self.aggregate_confidence_bayesian(confidences)

        # Sum mentions
        total_mentions = sum(r.num_mentions for r in relations)

        # Average numeric fields
        goldstein_scales = [r.goldstein_scale for r in relations if r.goldstein_scale is not None]
        avg_goldstein = sum(goldstein_scales) / len(goldstein_scales) if goldstein_scales else None

        tones = [r.tone for r in relations if r.tone is not None]
        avg_tone = sum(tones) / len(tones) if tones else None

        # Merge event codes
        all_event_codes = []
        for r in relations:
            all_event_codes.extend(r.event_codes)

        aggregated_relation = Relation(
            source_entity=base.source_entity,
            target_entity=base.target_entity,
            relation_type=base.relation_type,
            timestamp=base.timestamp,
            confidence=aggregated_confidence,
            quad_class=base.quad_class,
            num_mentions=total_mentions,
            goldstein_scale=avg_goldstein,
            tone=avg_tone,
            event_codes=list(set(all_event_codes)),
            metadata={**base.metadata, 'aggregated_from': len(relations)}
        )

        return aggregated_relation

    def _get_temporal_granularity(self, quad_class: Optional[int]) -> TemporalGranularity:
        """Get recommended temporal granularity for quad class."""
        if quad_class in self.quadclass_map:
            return self.quadclass_map[quad_class][1]
        return TemporalGranularity.DAILY

    def get_statistics(self) -> Dict:
        """Get classifier statistics."""
        return {
            'total_event_codes': len(self.event_code_map),
            'relation_types': len(RelationType),
            'quadclass_mappings': len(self.quadclass_map),
        }


def create_classifier() -> RelationClassifier:
    """Factory function to create initialized relation classifier."""
    return RelationClassifier()
