"""
Unit tests for entity normalization module.

Tests verify:
- All GDELT actor codes resolve to valid entities
- Cache lookup performance < 1ms per entity
- Unknown actors get consistent IDs across runs
"""

import pytest
import time
from entity_normalization import EntityNormalizer, Entity, create_normalizer


class TestEntityNormalization:
    """Test entity normalization functionality."""

    @pytest.fixture
    def normalizer(self):
        """Create normalizer for tests."""
        return create_normalizer()

    def test_known_country_resolution(self, normalizer):
        """Test resolution of known country codes."""
        assert normalizer.resolve_actor('USA') == 'USA'
        assert normalizer.resolve_actor('CHN') == 'CHN'
        assert normalizer.resolve_actor('RUS') == 'RUS'
        assert normalizer.resolve_actor('GBR') == 'GBR'

    def test_case_insensitive_resolution(self, normalizer):
        """Test that resolution is case-insensitive."""
        assert normalizer.resolve_actor('usa') == 'USA'
        assert normalizer.resolve_actor('USA') == 'USA'
        assert normalizer.resolve_actor('UsA') == 'USA'

    def test_whitespace_handling(self, normalizer):
        """Test that whitespace is handled correctly."""
        assert normalizer.resolve_actor('  USA  ') == 'USA'
        assert normalizer.resolve_actor('USA ') == 'USA'

    def test_none_actor_code(self, normalizer):
        """Test handling of None actor codes."""
        assert normalizer.resolve_actor(None) is None

    def test_empty_string_actor_code(self, normalizer):
        """Test handling of empty actor codes."""
        assert normalizer.resolve_actor('') is None
        assert normalizer.resolve_actor('   ') is None

    def test_unknown_actor_consistent_id(self, normalizer):
        """Test that unknown actors get consistent IDs."""
        entity_id_1 = normalizer.resolve_actor('UNKNOWN_REBEL_GROUP_1')
        entity_id_2 = normalizer.resolve_actor('UNKNOWN_REBEL_GROUP_1')
        assert entity_id_1 == entity_id_2
        assert entity_id_1 is not None
        assert 'UNKNOWN_' in entity_id_1

    def test_unknown_actor_different_ids(self, normalizer):
        """Test that different unknown actors get different IDs."""
        id1 = normalizer.resolve_actor('UNKNOWN_REBEL_1')
        id2 = normalizer.resolve_actor('UNKNOWN_REBEL_2')
        assert id1 != id2
        assert id1 is not None
        assert id2 is not None

    def test_entity_pair_resolution(self, normalizer):
        """Test resolution of actor pairs."""
        entity1, entity2 = normalizer.resolve_entity_pair('USA', 'CHN')
        assert entity1 == 'USA'
        assert entity2 == 'CHN'

    def test_entity_pair_with_unknowns(self, normalizer):
        """Test resolution of pairs with unknown actors."""
        entity1, entity2 = normalizer.resolve_entity_pair('USA', 'REBEL_UNKNOWN')
        assert entity1 == 'USA'
        assert entity2 is not None
        assert 'UNKNOWN_' in entity2

    def test_entity_pair_both_none(self, normalizer):
        """Test that both None returns None."""
        entity1, entity2 = normalizer.resolve_entity_pair(None, None)
        assert entity1 is None
        assert entity2 is None

    def test_get_entity_metadata(self, normalizer):
        """Test retrieving entity metadata."""
        entity = normalizer.get_entity('USA')
        assert entity is not None
        assert entity.entity_id == 'USA'
        assert entity.entity_type == 'country'
        assert entity.name == 'United States'

    def test_get_unknown_entity_metadata(self, normalizer):
        """Test retrieving unknown entity metadata."""
        entity_id = normalizer.resolve_actor('CUSTOM_REBEL')
        entity = normalizer.get_entity(entity_id)
        assert entity is not None
        assert entity.entity_type == 'unknown'
        assert 'CUSTOM_REBEL' in entity.metadata.get('original_code', '')

    def test_is_known_entity(self, normalizer):
        """Test known/unknown entity detection."""
        assert normalizer.is_known_entity('USA') is True
        assert normalizer.is_known_entity('CHN') is True

        unknown_id = normalizer.resolve_actor('UNKNOWN_ACTOR')
        assert normalizer.is_known_entity(unknown_id) is False

    def test_cache_performance(self, normalizer):
        """Test that cached lookups are fast (< 1ms per entity)."""
        # First lookup (may cache miss)
        normalizer.resolve_actor('USA')

        # Measure cached lookups
        start = time.perf_counter()
        for _ in range(1000):
            normalizer.resolve_actor('USA')
        elapsed = time.perf_counter() - start

        avg_time_ms = (elapsed * 1000) / 1000
        assert avg_time_ms < 1.0, f"Average lookup time {avg_time_ms}ms exceeds 1ms"

    def test_statistics(self, normalizer):
        """Test normalizer statistics."""
        stats = normalizer.get_statistics()
        assert stats['total_entities'] > 0
        assert stats['known_entities'] > 0
        assert stats['cache_entries'] > 0
        assert 'entities_by_type' in stats

    def test_international_organization_resolution(self, normalizer):
        """Test resolution of international organizations."""
        assert normalizer.resolve_actor('UN') == 'UN'
        assert normalizer.resolve_actor('NATO') == 'NATO'
        assert normalizer.resolve_actor('EU') == 'EU'

    def test_alternative_country_codes(self, normalizer):
        """Test resolution of alternative codes for same country."""
        # Some countries have multiple CAMEO codes
        assert normalizer.resolve_actor('CHN') == 'CHN'
        assert normalizer.resolve_actor('PRC') == 'CHN'

    def test_lru_cache_behavior(self, normalizer):
        """Test that LRU cache prevents memory explosion."""
        # Resolve many different unknown actors
        entity_ids = set()
        for i in range(5000):
            entity_id = normalizer.resolve_actor(f'ACTOR_{i}')
            entity_ids.add(entity_id)

        assert len(entity_ids) == 5000
        # Cache should have limited size despite 5000 different entries
        assert len(normalizer._lookup_cache) <= normalizer.cache_size + 100


class TestEntityObject:
    """Test Entity dataclass."""

    def test_entity_creation(self):
        """Test entity object creation."""
        entity = Entity(
            entity_id='USA',
            entity_type='country',
            name='United States'
        )
        assert entity.entity_id == 'USA'
        assert entity.entity_type == 'country'
        assert entity.name == 'United States'

    def test_entity_equality(self):
        """Test entity equality based on ID."""
        entity1 = Entity(entity_id='USA', entity_type='country', name='United States')
        entity2 = Entity(entity_id='USA', entity_type='country', name='US')
        assert entity1 == entity2

    def test_entity_hash(self):
        """Test entity hashability."""
        entity1 = Entity(entity_id='USA', entity_type='country', name='United States')
        entity2 = Entity(entity_id='USA', entity_type='country', name='US')
        entity_set = {entity1, entity2}
        # Should deduplicate based on hash
        assert len(entity_set) == 1


def test_integration_sample_events():
    """Integration test: resolve sample of 1000 event actors."""
    normalizer = create_normalizer()

    # Simulate sample of GDELT actor codes
    sample_actors = [
        'USA', 'CHN', 'RUS', 'GBR', 'FRA', 'DEU', 'JPN', 'IND',
        'ISR', 'IRN', 'SAU', 'TUR', 'KOR', 'PRK',  # Known countries
        'UN', 'NATO', 'EU',  # International orgs
        'REBEL_GROUP_1', 'ETHNIC_GROUP_2', 'UNKNOWN_ACTOR_3',  # Unknown
    ] * 50  # Repeat 50 times to get ~1000 samples

    start = time.perf_counter()
    resolved = set()
    for actor in sample_actors:
        entity_id = normalizer.resolve_actor(actor)
        resolved.add(entity_id)
    elapsed = time.perf_counter() - start

    # All should resolve to some entity
    assert len(resolved) > 0
    assert None not in resolved

    # Performance should be good (1000 actors in < 10ms with caching)
    throughput = len(sample_actors) / elapsed
    assert throughput > 100000, f"Throughput {throughput}/sec is too low"

    # Check statistics
    stats = normalizer.get_statistics()
    assert stats['total_entities'] > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
