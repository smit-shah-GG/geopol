"""
Tests for Query Parser and Validator.

Tests query parsing, validation, optimization, and natural language understanding.
"""

import pytest
from datetime import datetime, timedelta
from src.knowledge_graph.query_parser import (
    QueryParser, QueryOptimizer, QueryType, SearchMode,
    TimeWindow, QueryConstraints, ParsedQuery,
    create_parser, create_optimizer
)


class TestTimeWindow:
    """Test TimeWindow functionality."""

    def test_explicit_start_end(self):
        """Test time window with explicit start and end."""
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 31)
        tw = TimeWindow(start=start, end=end)

        assert tw.start == start
        assert tw.end == end
        assert tw.validate() == (True, None)

    def test_relative_days(self):
        """Test time window with relative days."""
        tw = TimeWindow(days=30)

        assert tw.start is not None
        assert tw.end is not None
        assert (tw.end - tw.start).days == 30

    def test_relative_weeks(self):
        """Test time window with relative weeks."""
        tw = TimeWindow(weeks=4)

        assert tw.start is not None
        assert tw.end is not None
        assert (tw.end - tw.start).days == 28

    def test_relative_months(self):
        """Test time window with relative months."""
        tw = TimeWindow(months=2)

        assert tw.start is not None
        assert tw.end is not None
        # Approximately 60 days (2 months * 30 days)
        assert 58 <= (tw.end - tw.start).days <= 62

    def test_invalid_start_after_end(self):
        """Test validation catches start after end."""
        tw = TimeWindow(
            start=datetime(2024, 2, 1),
            end=datetime(2024, 1, 1)
        )

        valid, error = tw.validate()
        assert not valid
        assert "before" in error

    def test_span_too_large(self):
        """Test validation catches excessive time span."""
        tw = TimeWindow(
            start=datetime(2023, 1, 1),
            end=datetime(2025, 1, 1)
        )

        valid, error = tw.validate()
        assert not valid
        assert "365 days" in error


class TestQueryConstraints:
    """Test QueryConstraints functionality."""

    def test_default_constraints(self):
        """Test default constraint values."""
        c = QueryConstraints()

        assert c.min_confidence == 0.0
        assert c.max_confidence == 1.0
        assert c.quad_class is None
        assert c.min_mentions == 1
        assert c.validate() == (True, None)

    def test_valid_constraints(self):
        """Test valid constraint configuration."""
        c = QueryConstraints(
            min_confidence=0.5,
            max_confidence=0.9,
            quad_class=4,
            min_mentions=10
        )

        assert c.validate() == (True, None)

    def test_invalid_confidence_range(self):
        """Test validation catches invalid confidence."""
        c = QueryConstraints(min_confidence=-0.1)
        valid, error = c.validate()
        assert not valid
        assert "min_confidence" in error

        c = QueryConstraints(max_confidence=1.5)
        valid, error = c.validate()
        assert not valid
        assert "max_confidence" in error

    def test_invalid_confidence_order(self):
        """Test validation catches min > max."""
        c = QueryConstraints(min_confidence=0.8, max_confidence=0.5)
        valid, error = c.validate()
        assert not valid
        assert "min_confidence must be <=" in error

    def test_invalid_quad_class(self):
        """Test validation catches invalid quad_class."""
        c = QueryConstraints(quad_class=5)
        valid, error = c.validate()
        assert not valid
        assert "quad_class" in error

    def test_invalid_min_mentions(self):
        """Test validation catches invalid min_mentions."""
        c = QueryConstraints(min_mentions=0)
        valid, error = c.validate()
        assert not valid
        assert "min_mentions" in error


class TestParsedQuery:
    """Test ParsedQuery functionality."""

    def test_entity_pair_valid(self):
        """Test valid entity pair query."""
        q = ParsedQuery(
            query_type=QueryType.ENTITY_PAIR,
            entity1="RUS",
            entity2="UKR"
        )

        assert q.validate() == (True, None)

    def test_entity_pair_missing_entity(self):
        """Test entity pair query missing entity2."""
        q = ParsedQuery(
            query_type=QueryType.ENTITY_PAIR,
            entity1="RUS"
        )

        valid, error = q.validate()
        assert not valid
        assert "requires both entity1 and entity2" in error

    def test_entity_relations_valid(self):
        """Test valid entity relations query."""
        q = ParsedQuery(
            query_type=QueryType.ENTITY_RELATIONS,
            entity1="USA"
        )

        assert q.validate() == (True, None)

    def test_temporal_path_valid(self):
        """Test valid temporal path query."""
        q = ParsedQuery(
            query_type=QueryType.TEMPORAL_PATH,
            entity1="USA",
            entity2="CHN",
            path_length=3
        )

        assert q.validate() == (True, None)

    def test_temporal_path_invalid_length(self):
        """Test temporal path with invalid path length."""
        q = ParsedQuery(
            query_type=QueryType.TEMPORAL_PATH,
            entity1="USA",
            entity2="CHN",
            path_length=10
        )

        valid, error = q.validate()
        assert not valid
        assert "path_length must be between 1 and 5" in error

    def test_similarity_search_valid(self):
        """Test valid similarity search query."""
        q = ParsedQuery(
            query_type=QueryType.SIMILARITY_SEARCH,
            entity_similarity_query="NATO",
            similarity_threshold=0.85
        )

        assert q.validate() == (True, None)

    def test_similarity_search_missing_query(self):
        """Test similarity search without query string."""
        q = ParsedQuery(query_type=QueryType.SIMILARITY_SEARCH)

        valid, error = q.validate()
        assert not valid
        assert "requires entity_similarity_query" in error

    def test_pattern_match_valid(self):
        """Test valid pattern match query."""
        q = ParsedQuery(
            query_type=QueryType.PATTERN_MATCH,
            pattern_sequence=[
                {"relation": "threaten", "quad_class": 3},
                {"relation": "military_action", "quad_class": 4}
            ]
        )

        assert q.validate() == (True, None)

    def test_invalid_max_results(self):
        """Test validation catches invalid max_results."""
        q = ParsedQuery(
            query_type=QueryType.ENTITY_RELATIONS,
            entity1="USA",
            max_results=100000
        )

        valid, error = q.validate()
        assert not valid
        assert "max_results" in error


class TestQueryParser:
    """Test QueryParser functionality."""

    def test_parse_entity_pair_dict(self):
        """Test parsing entity pair query from dict."""
        parser = create_parser()

        query_dict = {
            "actor1": "RUS",
            "actor2": "UKR",
            "quad_class": 4,
            "time_window": {"days": 30},
            "min_confidence": 0.7
        }

        parsed = parser.parse_dict(query_dict)

        assert parsed.query_type == QueryType.ENTITY_PAIR
        assert parsed.entity1 == "RUS"
        assert parsed.entity2 == "UKR"
        assert parsed.constraints.quad_class == 4
        assert parsed.constraints.min_confidence == 0.7
        assert parsed.time_window is not None
        assert parsed.time_window.days == 30

    def test_parse_similarity_search_dict(self):
        """Test parsing similarity search from dict."""
        parser = create_parser()

        query_dict = {
            "entity_similarity": "NATO",
            "similarity_threshold": 0.8,
            "quad_class": 1,
            "time_window": {"days": 7}
        }

        parsed = parser.parse_dict(query_dict)

        assert parsed.query_type == QueryType.SIMILARITY_SEARCH
        assert parsed.entity_similarity_query == "NATO"
        assert parsed.similarity_threshold == 0.8
        assert parsed.search_mode == SearchMode.SEMANTIC

    def test_parse_temporal_path_dict(self):
        """Test parsing temporal path query from dict."""
        parser = create_parser()

        query_dict = {
            "target_event": "RUS_UKR_conflict_2024",
            "entity1": "RUS",
            "entity2": "UKR",
            "path_length": 3,
            "time_window": {"days": 90}
        }

        parsed = parser.parse_dict(query_dict)

        assert parsed.query_type == QueryType.TEMPORAL_PATH
        assert parsed.entity1 == "RUS"
        assert parsed.entity2 == "UKR"
        assert parsed.path_length == 3

    def test_parse_with_explicit_query_type(self):
        """Test parsing with explicit query_type field."""
        parser = create_parser()

        query_dict = {
            "query_type": "entity_relations",
            "entity1": "USA"
        }

        parsed = parser.parse_dict(query_dict)

        assert parsed.query_type == QueryType.ENTITY_RELATIONS
        assert parsed.entity1 == "USA"

    def test_parse_datetime_iso_format(self):
        """Test datetime parsing from ISO format."""
        parser = create_parser()

        query_dict = {
            "entity1": "USA",
            "time_window": {
                "start": "2024-01-01T00:00:00",
                "end": "2024-01-31T23:59:59"
            }
        }

        parsed = parser.parse_dict(query_dict)

        assert parsed.time_window.start == datetime(2024, 1, 1, 0, 0, 0)
        assert parsed.time_window.end == datetime(2024, 1, 31, 23, 59, 59)

    def test_parse_datetime_date_format(self):
        """Test datetime parsing from date-only format."""
        parser = create_parser()

        query_dict = {
            "entity1": "USA",
            "time_window": {
                "start": "2024-01-01",
                "end": "2024-01-31"
            }
        }

        parsed = parser.parse_dict(query_dict)

        assert parsed.time_window.start.date() == datetime(2024, 1, 1).date()
        assert parsed.time_window.end.date() == datetime(2024, 1, 31).date()

    def test_parse_natural_language_conflicts(self):
        """Test parsing 'conflicts between X and Y' pattern."""
        parser = create_parser()

        parsed = parser.parse_natural_language("conflicts between RUS and UKR")

        assert parsed.query_type == QueryType.ENTITY_PAIR
        assert parsed.entity1 == "RUS"
        assert parsed.entity2 == "UKR"

    def test_parse_natural_language_relations(self):
        """Test parsing 'relations with X' pattern."""
        parser = create_parser()

        parsed = parser.parse_natural_language("relations with USA")

        assert parsed.query_type == QueryType.ENTITY_RELATIONS
        assert parsed.entity1 == "USA"

    def test_parse_natural_language_path(self):
        """Test parsing 'path from X to Y' pattern."""
        parser = create_parser()

        parsed = parser.parse_natural_language("path from USA to CHN")

        assert parsed.query_type == QueryType.TEMPORAL_PATH
        assert parsed.entity1 == "USA"
        assert parsed.entity2 == "CHN"

    def test_parse_natural_language_similarity(self):
        """Test parsing 'similar to X' pattern."""
        parser = create_parser()

        parsed = parser.parse_natural_language("similar to NATO")

        assert parsed.query_type == QueryType.SIMILARITY_SEARCH
        assert parsed.entity_similarity_query == "NATO"
        assert parsed.search_mode == SearchMode.SEMANTIC

    def test_parse_natural_language_invalid(self):
        """Test parsing invalid natural language query."""
        parser = create_parser()

        with pytest.raises(ValueError, match="Could not parse"):
            parser.parse_natural_language("this is not a valid query")

    def test_validate_valid_query(self):
        """Test validation of valid query."""
        parser = create_parser()

        query = ParsedQuery(
            query_type=QueryType.ENTITY_PAIR,
            entity1="USA",
            entity2="CHN"
        )

        # Should not raise
        parser.validate(query)

    def test_validate_invalid_query(self):
        """Test validation of invalid query."""
        parser = create_parser()

        query = ParsedQuery(
            query_type=QueryType.ENTITY_PAIR,
            entity1="USA"  # Missing entity2
        )

        with pytest.raises(ValueError, match="Invalid query"):
            parser.validate(query)


class TestQueryOptimizer:
    """Test QueryOptimizer functionality."""

    def test_optimize_entity_pair(self):
        """Test optimization for entity pair query."""
        optimizer = create_optimizer()

        query = ParsedQuery(
            query_type=QueryType.ENTITY_PAIR,
            entity1="RUS",
            entity2="UKR"
        )

        plan = optimizer.optimize(query)

        assert plan["query_type"] == "entity_pair"
        assert plan["estimated_cost"] > 0
        assert len(plan["execution_steps"]) > 0
        assert "actor_pair_index_lookup" in str(plan["execution_steps"])

    def test_optimize_entity_relations(self):
        """Test optimization for entity relations query."""
        optimizer = create_optimizer()

        query = ParsedQuery(
            query_type=QueryType.ENTITY_RELATIONS,
            entity1="USA"
        )

        plan = optimizer.optimize(query)

        assert plan["query_type"] == "entity_relations"
        assert plan["estimated_cost"] > 0
        assert "get_neighbors" in str(plan["execution_steps"])

    def test_optimize_temporal_path(self):
        """Test optimization for temporal path query."""
        optimizer = create_optimizer()

        query = ParsedQuery(
            query_type=QueryType.TEMPORAL_PATH,
            entity1="USA",
            entity2="CHN",
            path_length=3
        )

        plan = optimizer.optimize(query)

        assert plan["query_type"] == "temporal_path"
        # Cost should scale with path length
        assert plan["estimated_cost"] >= 30
        assert "temporal_bfs" in str(plan["execution_steps"])

    def test_optimize_similarity_search(self):
        """Test optimization for similarity search."""
        optimizer = create_optimizer()

        query = ParsedQuery(
            query_type=QueryType.SIMILARITY_SEARCH,
            entity_similarity_query="NATO"
        )

        plan = optimizer.optimize(query)

        assert plan["query_type"] == "similarity_search"
        assert "vector_similarity_search" in str(plan["execution_steps"])

    def test_optimize_hybrid_search(self):
        """Test optimization for hybrid search."""
        optimizer = create_optimizer()

        query = ParsedQuery(
            query_type=QueryType.HYBRID_SEARCH,
            entity1="USA",
            entity_similarity_query="NATO"
        )

        plan = optimizer.optimize(query)

        assert plan["query_type"] == "hybrid_search"
        assert plan["parallel_execution"] is True

    def test_optimize_pattern_match(self):
        """Test optimization for pattern matching."""
        optimizer = create_optimizer()

        query = ParsedQuery(
            query_type=QueryType.PATTERN_MATCH,
            pattern_sequence=[
                {"relation": "threaten"},
                {"relation": "military_action"}
            ]
        )

        plan = optimizer.optimize(query)

        assert plan["query_type"] == "pattern_match"
        # Cost scales with pattern length
        assert plan["estimated_cost"] >= 30

    def test_optimize_with_small_time_window(self):
        """Test cost reduction for small time window."""
        optimizer = create_optimizer()

        query = ParsedQuery(
            query_type=QueryType.ENTITY_RELATIONS,
            entity1="USA",
            time_window=TimeWindow(days=7)
        )

        plan = optimizer.optimize(query)

        # Should have reduced cost due to small time window
        base_query = ParsedQuery(
            query_type=QueryType.ENTITY_RELATIONS,
            entity1="USA"
        )
        base_plan = optimizer.optimize(base_query)

        assert plan["estimated_cost"] < base_plan["estimated_cost"]

    def test_optimize_with_quad_class_filter(self):
        """Test cost reduction for quad_class filtering."""
        optimizer = create_optimizer()

        query = ParsedQuery(
            query_type=QueryType.ENTITY_RELATIONS,
            entity1="USA",
            constraints=QueryConstraints(quad_class=4)
        )

        plan = optimizer.optimize(query)

        # Should have reduced cost due to filtering
        base_query = ParsedQuery(
            query_type=QueryType.ENTITY_RELATIONS,
            entity1="USA"
        )
        base_plan = optimizer.optimize(base_query)

        assert plan["estimated_cost"] <= base_plan["estimated_cost"]


class TestIntegration:
    """Integration tests for full query pipeline."""

    def test_full_pipeline_entity_pair(self):
        """Test full pipeline: parse -> validate -> optimize."""
        parser = create_parser()
        optimizer = create_optimizer()

        query_dict = {
            "actor1": "RUS",
            "actor2": "UKR",
            "quad_class": 4,
            "time_window": {"days": 30},
            "min_confidence": 0.7
        }

        # Parse
        parsed = parser.parse_dict(query_dict)

        # Validate
        parser.validate(parsed)

        # Optimize
        plan = optimizer.optimize(parsed)

        assert parsed.query_type == QueryType.ENTITY_PAIR
        assert plan["estimated_cost"] > 0

    def test_full_pipeline_similarity_search(self):
        """Test full pipeline for similarity search."""
        parser = create_parser()
        optimizer = create_optimizer()

        query_dict = {
            "entity_similarity": "NATO",
            "similarity_threshold": 0.8,
            "time_window": {"weeks": 2},
            "quad_class": 1
        }

        parsed = parser.parse_dict(query_dict)
        parser.validate(parsed)
        plan = optimizer.optimize(parsed)

        assert parsed.query_type == QueryType.SIMILARITY_SEARCH
        assert parsed.similarity_threshold == 0.8
        assert plan["use_cache"] is True

    def test_error_messages_helpful(self):
        """Test that error messages are helpful."""
        parser = create_parser()

        # Invalid confidence range
        query = ParsedQuery(
            query_type=QueryType.ENTITY_PAIR,
            entity1="USA",
            entity2="CHN",
            constraints=QueryConstraints(min_confidence=2.0)
        )

        with pytest.raises(ValueError) as exc_info:
            parser.validate(query)

        assert "min_confidence" in str(exc_info.value)
        assert "[0,1]" in str(exc_info.value)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
