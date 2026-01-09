"""
Knowledge Graph Engine - Temporal Knowledge Graph construction from GDELT events.

Modules:
- entity_normalization: CAMEO actor code to canonical entity mapping
- relation_classification: Event code to typed relation mapping with confidence
- graph_builder: NetworkX temporal MultiDiGraph construction
- temporal_index: Efficient temporal query structures
- persistence: Graph serialization and deserialization
"""

from .entity_normalization import EntityNormalizer, Entity, create_normalizer
from .relation_classification import (
    RelationClassifier, Relation, RelationType, TemporalGranularity,
    create_classifier
)

__all__ = [
    'EntityNormalizer',
    'Entity',
    'create_normalizer',
    'RelationClassifier',
    'Relation',
    'RelationType',
    'TemporalGranularity',
    'create_classifier',
]
