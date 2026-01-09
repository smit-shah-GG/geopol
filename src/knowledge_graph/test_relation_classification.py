"""
Unit tests for relation classification system.

Tests verify:
- All QuadClass 1 and 4 events map to valid relation types
- Confidence scores range [0,1] with proper calibration
- Aggregation handles duplicate events correctly
"""

import pytest
from datetime import datetime
from src.knowledge_graph.relation_classification import (
    RelationClassifier, Relation, RelationType, TemporalGranularity,
    create_classifier
)


class TestRelationClassifier:
    """Test relation classification functionality."""

    @pytest.fixture
    def classifier(self):
        """Create classifier for tests."""
        return create_classifier()

    def test_diplomatic_event_classification(self, classifier):
        """Test classification of diplomatic events (QuadClass 1)."""
        relation = classifier.classify_event(
            source_entity='USA',
            target_entity='CHN',
            event_code='05',
            quad_class=1,
            timestamp='2024-01-15T10:00:00Z',
            num_mentions=50,
            goldstein_scale=2.5,
            tone=5.0
        )

        assert relation is not None
        assert relation.source_entity == 'USA'
        assert relation.target_entity == 'CHN'
        assert relation.quad_class == 1
        assert 0.0 <= relation.confidence <= 1.0

    def test_conflict_event_classification(self, classifier):
        """Test classification of conflict events (QuadClass 4)."""
        relation = classifier.classify_event(
            source_entity='RUS',
            target_entity='UKR',
            event_code='19',
            quad_class=4,
            timestamp='2024-01-15T10:00:00Z',
            num_mentions=200,
            goldstein_scale=-8.5,
            tone=-95.0
        )

        assert relation is not None
        assert relation.source_entity == 'RUS'
        assert relation.target_entity == 'UKR'
        assert relation.quad_class == 4
        assert 0.0 <= relation.confidence <= 1.0

    def test_relation_type_mapping(self, classifier):
        """Test mapping of event codes to relation types."""
        test_cases = [
            ('01', RelationType.MAKES_STATEMENT),
            ('05', RelationType.COOPERATES),
            ('19', RelationType.FIGHTS),
            ('184', RelationType.KILLS),
        ]

        for event_code, expected_type in test_cases:
            relation = classifier.classify_event(
                source_entity='A',
                target_entity='B',
                event_code=event_code,
                quad_class=1,
                timestamp='2024-01-15T10:00:00Z'
            )
            assert relation.relation_type == expected_type

    def test_confidence_calculation_mentions(self, classifier):
        """Test confidence increases with mentions."""
        low_mentions = classifier.classify_event(
            source_entity='A', target_entity='B', event_code='05',
            quad_class=1, timestamp='2024-01-15T10:00:00Z',
            num_mentions=1
        )
        high_mentions = classifier.classify_event(
            source_entity='A', target_entity='B', event_code='05',
            quad_class=1, timestamp='2024-01-15T10:00:00Z',
            num_mentions=1000
        )

        assert high_mentions.confidence > low_mentions.confidence

    def test_confidence_calculation_goldstein(self, classifier):
        """Test confidence relates to Goldstein scale magnitude."""
        weak_signal = classifier.classify_event(
            source_entity='A', target_entity='B', event_code='05',
            quad_class=1, timestamp='2024-01-15T10:00:00Z',
            goldstein_scale=1.0
        )
        strong_signal = classifier.classify_event(
            source_entity='A', target_entity='B', event_code='05',
            quad_class=1, timestamp='2024-01-15T10:00:00Z',
            goldstein_scale=8.0
        )

        assert strong_signal.confidence > weak_signal.confidence

    def test_confidence_bounds(self, classifier):
        """Test that confidence is always in [0, 1]."""
        test_cases = [
            (None, None, None),  # Minimal data
            (1, 0.5, 0.5),  # Single signal
            (1000, 10.0, 100),  # Maximum signals
            (1000, -10.0, -100),  # Conflict signals
        ]

        for mentions, goldstein, tone in test_cases:
            relation = classifier.classify_event(
                source_entity='A', target_entity='B', event_code='05',
                quad_class=1, timestamp='2024-01-15T10:00:00Z',
                num_mentions=mentions,
                goldstein_scale=goldstein,
                tone=tone
            )
            assert 0.0 <= relation.confidence <= 1.0, \
                f"Confidence {relation.confidence} out of bounds for inputs " \
                f"mentions={mentions}, goldstein={goldstein}, tone={tone}"

    def test_none_inputs(self, classifier):
        """Test handling of None inputs."""
        # Valid None handling
        relation = classifier.classify_event(
            source_entity='A', target_entity='B', event_code=None,
            quad_class=1, timestamp='2024-01-15T10:00:00Z'
        )
        assert relation is not None  # Should still create relation

        # Invalid None source
        relation = classifier.classify_event(
            source_entity=None, target_entity='B', event_code='05',
            quad_class=1, timestamp='2024-01-15T10:00:00Z'
        )
        assert relation is None

        # Invalid None target
        relation = classifier.classify_event(
            source_entity='A', target_entity=None, event_code='05',
            quad_class=1, timestamp='2024-01-15T10:00:00Z'
        )
        assert relation is None

        # Invalid None quad_class
        relation = classifier.classify_event(
            source_entity='A', target_entity='B', event_code='05',
            quad_class=None, timestamp='2024-01-15T10:00:00Z'
        )
        assert relation is None

    def test_bayesian_confidence_aggregation(self, classifier):
        """Test Bayesian aggregation of multiple confidence scores."""
        # Test with unanimous high confidence
        high_confidences = [0.9, 0.9, 0.9]
        agg_high = classifier.aggregate_confidence_bayesian(high_confidences)
        assert agg_high > 0.8

        # Test with unanimous low confidence
        low_confidences = [0.2, 0.2, 0.2]
        agg_low = classifier.aggregate_confidence_bayesian(low_confidences)
        assert agg_low < 0.4

        # Test with mixed confidence
        mixed = [0.2, 0.8, 0.5]
        agg_mixed = classifier.aggregate_confidence_bayesian(mixed)
        assert 0.0 <= agg_mixed <= 1.0

        # Test with empty list
        agg_empty = classifier.aggregate_confidence_bayesian([])
        assert agg_empty == 0.5  # Neutral prior

    def test_duplicate_relation_aggregation(self, classifier):
        """Test aggregation of duplicate relations."""
        base_time = '2024-01-15T10:00:00Z'

        relations = [
            Relation(
                source_entity='USA', target_entity='CHN',
                relation_type=RelationType.COOPERATES,
                timestamp=base_time, confidence=0.7, quad_class=1,
                num_mentions=50, goldstein_scale=2.0, tone=10.0
            ),
            Relation(
                source_entity='USA', target_entity='CHN',
                relation_type=RelationType.COOPERATES,
                timestamp=base_time, confidence=0.8, quad_class=1,
                num_mentions=30, goldstein_scale=3.0, tone=15.0
            ),
            Relation(
                source_entity='USA', target_entity='CHN',
                relation_type=RelationType.COOPERATES,
                timestamp=base_time, confidence=0.6, quad_class=1,
                num_mentions=20, goldstein_scale=1.5, tone=5.0
            ),
        ]

        aggregated = classifier.aggregate_duplicate_relations(relations)

        assert aggregated.source_entity == 'USA'
        assert aggregated.target_entity == 'CHN'
        assert aggregated.num_mentions == 100  # 50 + 30 + 20
        assert 0.0 <= aggregated.confidence <= 1.0
        assert aggregated.confidence > 0.5  # Should be reasonably high

    def test_single_relation_aggregation(self, classifier):
        """Test aggregation of single relation returns itself."""
        relation = Relation(
            source_entity='USA', target_entity='CHN',
            relation_type=RelationType.COOPERATES,
            timestamp='2024-01-15T10:00:00Z',
            confidence=0.7, quad_class=1
        )

        aggregated = classifier.aggregate_duplicate_relations([relation])
        assert aggregated == relation

    def test_temporal_granularity(self, classifier):
        """Test temporal granularity by quad class."""
        # QuadClass 1 (diplomatic) -> weekly
        gran1 = classifier._get_temporal_granularity(1)
        assert gran1 == TemporalGranularity.WEEKLY

        # QuadClass 3 (verbal conflict) -> daily
        gran3 = classifier._get_temporal_granularity(3)
        assert gran3 == TemporalGranularity.DAILY

        # QuadClass 4 (material conflict) -> daily
        gran4 = classifier._get_temporal_granularity(4)
        assert gran4 == TemporalGranularity.DAILY

    def test_quadclass_validation(self, classifier):
        """Test validation of all QuadClass values."""
        for quad_class in [1, 2, 3, 4]:
            relation = classifier.classify_event(
                source_entity='A', target_entity='B', event_code='05',
                quad_class=quad_class, timestamp='2024-01-15T10:00:00Z'
            )
            assert relation is not None
            assert relation.quad_class == quad_class

    def test_statistics(self, classifier):
        """Test classifier statistics."""
        stats = classifier.get_statistics()
        assert stats['total_event_codes'] > 0
        assert stats['relation_types'] > 0
        assert stats['quadclass_mappings'] == 4


class TestRelationObject:
    """Test Relation dataclass."""

    def test_relation_creation(self):
        """Test relation object creation."""
        relation = Relation(
            source_entity='USA',
            target_entity='CHN',
            relation_type=RelationType.COOPERATES,
            timestamp='2024-01-15T10:00:00Z',
            confidence=0.7,
            quad_class=1
        )
        assert relation.source_entity == 'USA'
        assert relation.confidence == 0.7

    def test_relation_equality(self):
        """Test relation equality based on key fields."""
        r1 = Relation(
            source_entity='USA', target_entity='CHN',
            relation_type=RelationType.COOPERATES,
            timestamp='2024-01-15T10:00:00Z',
            confidence=0.7, quad_class=1
        )
        r2 = Relation(
            source_entity='USA', target_entity='CHN',
            relation_type=RelationType.COOPERATES,
            timestamp='2024-01-15T10:00:00Z',
            confidence=0.8, quad_class=1  # Different confidence
        )
        assert r1 == r2  # Should be equal despite different confidence

    def test_relation_hash(self):
        """Test relation hashability."""
        r1 = Relation(
            source_entity='USA', target_entity='CHN',
            relation_type=RelationType.COOPERATES,
            timestamp='2024-01-15T10:00:00Z',
            confidence=0.7, quad_class=1
        )
        r2 = Relation(
            source_entity='USA', target_entity='CHN',
            relation_type=RelationType.COOPERATES,
            timestamp='2024-01-15T10:00:00Z',
            confidence=0.8, quad_class=1
        )
        relation_set = {r1, r2}
        # Should deduplicate
        assert len(relation_set) == 1


def test_integration_event_classification():
    """Integration test: classify 100 sample GDELT events."""
    classifier = create_classifier()

    # Simulate GDELT events
    events = [
        # Diplomatic events
        {'event_code': '05', 'quad_class': 1, 'mentions': 50, 'goldstein': 2.0, 'tone': 10},
        {'event_code': '04', 'quad_class': 1, 'mentions': 30, 'goldstein': 1.5, 'tone': 5},
        {'event_code': '057', 'quad_class': 1, 'mentions': 100, 'goldstein': 3.0, 'tone': 15},

        # Conflict events
        {'event_code': '19', 'quad_class': 4, 'mentions': 150, 'goldstein': -8.0, 'tone': -90},
        {'event_code': '184', 'quad_class': 4, 'mentions': 200, 'goldstein': -9.5, 'tone': -100},
        {'event_code': '192', 'quad_class': 4, 'mentions': 80, 'goldstein': -7.0, 'tone': -80},
    ] * 20  # 120 events

    relations = []
    for i, event in enumerate(events):
        relation = classifier.classify_event(
            source_entity='USA',
            target_entity=f'COUNTRY_{i % 10}',
            event_code=event['event_code'],
            quad_class=event['quad_class'],
            timestamp=f'2024-01-{(i % 28) + 1:02d}T10:00:00Z',
            num_mentions=event['mentions'],
            goldstein_scale=event['goldstein'],
            tone=event['tone']
        )
        if relation:
            relations.append(relation)

    # Verify all events classified
    assert len(relations) == len(events)

    # Verify confidence distribution
    confidences = [r.confidence for r in relations]
    assert all(0.0 <= c <= 1.0 for c in confidences)

    # Verify diplomatic events have proper relation types
    diplomatic = [r for r in relations if r.quad_class == 1]
    assert all(r.relation_type in [
        RelationType.COOPERATES, RelationType.CONSULTS,
        RelationType.NEGOTIATES, RelationType.SIGNS_AGREEMENT
    ] for r in diplomatic)

    # Verify conflict events have proper relation types
    conflict = [r for r in relations if r.quad_class == 4]
    assert all(r.relation_type in [
        RelationType.FIGHTS, RelationType.KILLS,
        RelationType.USES_WEAPONS, RelationType.OCCUPIES
    ] for r in conflict)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
