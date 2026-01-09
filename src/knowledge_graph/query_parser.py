"""
Query Parser and Validator for Temporal Knowledge Graph.

Parses and validates user queries into structured format for efficient execution.
Supports entity pairs, time windows, relation types, and confidence constraints.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Any
import re


class QueryType(Enum):
    """Type of query operation."""
    ENTITY_PAIR = "entity_pair"           # Find relations between two entities
    ENTITY_RELATIONS = "entity_relations"  # Find all relations for one entity
    TEMPORAL_PATH = "temporal_path"        # Find causal path to event
    PATTERN_MATCH = "pattern_match"        # Match event sequence pattern
    SIMILARITY_SEARCH = "similarity_search"  # Semantic entity search
    HYBRID_SEARCH = "hybrid_search"        # Combined graph + vector search


class SearchMode(Enum):
    """Entity matching mode."""
    EXACT = "exact"           # Exact entity ID match
    SEMANTIC = "semantic"     # Vector similarity match
    HYBRID = "hybrid"         # Both exact and semantic


@dataclass
class TimeWindow:
    """Time window specification for temporal filtering."""
    start: Optional[datetime] = None
    end: Optional[datetime] = None
    days: Optional[int] = None
    weeks: Optional[int] = None
    months: Optional[int] = None

    def __post_init__(self):
        """Compute start/end from relative time if provided."""
        if self.days is not None or self.weeks is not None or self.months is not None:
            if self.end is None:
                self.end = datetime.now()

            delta = timedelta(
                days=self.days or 0,
                weeks=self.weeks or 0
            )
            # Approximate months as 30 days
            if self.months:
                delta += timedelta(days=30 * self.months)

            if self.start is None:
                self.start = self.end - delta

    def validate(self) -> Tuple[bool, Optional[str]]:
        """Validate time window constraints.

        Returns:
            (is_valid, error_message)
        """
        if self.start and self.end and self.start > self.end:
            return False, "start time must be before end time"

        # Limit query span to avoid performance issues
        if self.start and self.end:
            span_days = (self.end - self.start).days
            if span_days > 365:
                return False, "time window span cannot exceed 365 days"

        return True, None


@dataclass
class QueryConstraints:
    """Query constraints and filters."""
    min_confidence: float = 0.0
    max_confidence: float = 1.0
    quad_class: Optional[int] = None
    relation_types: Set[str] = field(default_factory=set)
    min_mentions: int = 1
    excluded_entities: Set[str] = field(default_factory=set)
    required_entities: Set[str] = field(default_factory=set)

    def validate(self) -> Tuple[bool, Optional[str]]:
        """Validate constraints.

        Returns:
            (is_valid, error_message)
        """
        if not (0.0 <= self.min_confidence <= 1.0):
            return False, "min_confidence must be in [0,1]"
        if not (0.0 <= self.max_confidence <= 1.0):
            return False, "max_confidence must be in [0,1]"
        if self.min_confidence > self.max_confidence:
            return False, "min_confidence must be <= max_confidence"
        if self.quad_class is not None and self.quad_class not in [1, 2, 3, 4]:
            return False, "quad_class must be 1, 2, 3, or 4"
        if self.min_mentions < 1:
            return False, "min_mentions must be >= 1"

        return True, None


@dataclass
class ParsedQuery:
    """Structured representation of parsed query."""
    query_type: QueryType
    search_mode: SearchMode = SearchMode.EXACT

    # Entity specifications
    entity1: Optional[str] = None
    entity2: Optional[str] = None
    entity_similarity_query: Optional[str] = None  # For semantic search
    similarity_threshold: float = 0.8

    # Temporal specifications
    time_window: Optional[TimeWindow] = None

    # Constraints
    constraints: QueryConstraints = field(default_factory=QueryConstraints)

    # Path/pattern specifications
    path_length: int = 2  # Max hops for path queries
    pattern_sequence: List[Dict[str, Any]] = field(default_factory=list)

    # Query optimization hints
    max_results: int = 1000
    result_ranking: str = "confidence"  # or "recency" or "relevance"
    include_explanations: bool = True

    def validate(self) -> Tuple[bool, Optional[str]]:
        """Validate complete query structure.

        Returns:
            (is_valid, error_message)
        """
        # Validate constraints
        valid, error = self.constraints.validate()
        if not valid:
            return False, f"Invalid constraints: {error}"

        # Validate time window if present
        if self.time_window:
            valid, error = self.time_window.validate()
            if not valid:
                return False, f"Invalid time window: {error}"

        # Query type specific validation
        if self.query_type == QueryType.ENTITY_PAIR:
            if not self.entity1 or not self.entity2:
                return False, "ENTITY_PAIR query requires both entity1 and entity2"

        elif self.query_type == QueryType.ENTITY_RELATIONS:
            if not self.entity1:
                return False, "ENTITY_RELATIONS query requires entity1"

        elif self.query_type == QueryType.TEMPORAL_PATH:
            if not self.entity1 or not self.entity2:
                return False, "TEMPORAL_PATH query requires both entity1 and entity2"
            if self.path_length < 1 or self.path_length > 5:
                return False, "path_length must be between 1 and 5"

        elif self.query_type == QueryType.SIMILARITY_SEARCH:
            if not self.entity_similarity_query:
                return False, "SIMILARITY_SEARCH query requires entity_similarity_query"
            if not (0.0 <= self.similarity_threshold <= 1.0):
                return False, "similarity_threshold must be in [0,1]"

        elif self.query_type == QueryType.PATTERN_MATCH:
            if not self.pattern_sequence:
                return False, "PATTERN_MATCH query requires pattern_sequence"

        # Validate search mode
        if self.search_mode not in SearchMode:
            return False, f"Invalid search_mode: {self.search_mode}"

        # Validate result ranking
        if self.result_ranking not in ["confidence", "recency", "relevance"]:
            return False, f"Invalid result_ranking: {self.result_ranking}"

        # Validate max_results
        if self.max_results < 1 or self.max_results > 10000:
            return False, "max_results must be between 1 and 10000"

        return True, None


class QueryParser:
    """Parse and validate knowledge graph queries."""

    # Natural language pattern templates
    NL_PATTERNS = {
        # "conflicts between X and Y" -> entity pair query
        r"conflicts? between (\w+) and (\w+)": QueryType.ENTITY_PAIR,
        # "relations with X" -> entity relations
        r"relations? (?:with|for|of) (\w+)": QueryType.ENTITY_RELATIONS,
        # "path from X to Y" -> temporal path
        r"path from (\w+) to (\w+)": QueryType.TEMPORAL_PATH,
        # "similar to X" -> similarity search
        r"similar to (\w+)": QueryType.SIMILARITY_SEARCH,
    }

    def __init__(self):
        """Initialize query parser."""
        pass

    def parse_dict(self, query_dict: Dict[str, Any]) -> ParsedQuery:
        """Parse query from dictionary format.

        Args:
            query_dict: Query specification as dictionary

        Returns:
            Parsed query object

        Raises:
            ValueError: If query is invalid
        """
        # Determine query type
        query_type = self._infer_query_type(query_dict)

        # Parse time window
        time_window = None
        if "time_window" in query_dict:
            tw_dict = query_dict["time_window"]
            time_window = TimeWindow(
                start=self._parse_datetime(tw_dict.get("start")),
                end=self._parse_datetime(tw_dict.get("end")),
                days=tw_dict.get("days"),
                weeks=tw_dict.get("weeks"),
                months=tw_dict.get("months")
            )

        # Parse constraints
        constraints = QueryConstraints(
            min_confidence=query_dict.get("min_confidence", 0.0),
            max_confidence=query_dict.get("max_confidence", 1.0),
            quad_class=query_dict.get("quad_class"),
            relation_types=set(query_dict.get("relation_types", [])),
            min_mentions=query_dict.get("min_mentions", 1),
            excluded_entities=set(query_dict.get("excluded_entities", [])),
            required_entities=set(query_dict.get("required_entities", []))
        )

        # Determine search mode
        search_mode = SearchMode.EXACT
        if query_dict.get("entity_similarity"):
            search_mode = SearchMode.SEMANTIC
        elif query_dict.get("search_mode"):
            search_mode = SearchMode[query_dict["search_mode"].upper()]

        # Build parsed query
        parsed = ParsedQuery(
            query_type=query_type,
            search_mode=search_mode,
            entity1=query_dict.get("actor1") or query_dict.get("entity1"),
            entity2=query_dict.get("actor2") or query_dict.get("entity2"),
            entity_similarity_query=query_dict.get("entity_similarity"),
            similarity_threshold=query_dict.get("similarity_threshold", 0.8),
            time_window=time_window,
            constraints=constraints,
            path_length=query_dict.get("path_length", 2),
            pattern_sequence=query_dict.get("pattern_sequence", []),
            max_results=query_dict.get("max_results", 1000),
            result_ranking=query_dict.get("result_ranking", "confidence"),
            include_explanations=query_dict.get("include_explanations", True)
        )

        return parsed

    def parse_natural_language(self, query_text: str) -> ParsedQuery:
        """Parse query from natural language text.

        Args:
            query_text: Natural language query string

        Returns:
            Parsed query object

        Raises:
            ValueError: If query cannot be parsed
        """
        query_text_lower = query_text.lower().strip()

        # Try to match against known patterns
        for pattern, query_type in self.NL_PATTERNS.items():
            match = re.search(pattern, query_text_lower)
            if match:
                return self._build_query_from_match(query_type, match, query_text)

        # No pattern matched
        raise ValueError(
            f"Could not parse natural language query: '{query_text}'. "
            "Please use dictionary format or rephrase."
        )

    def validate(self, query: ParsedQuery) -> None:
        """Validate parsed query.

        Args:
            query: Parsed query to validate

        Raises:
            ValueError: If query is invalid
        """
        valid, error = query.validate()
        if not valid:
            raise ValueError(f"Invalid query: {error}")

    def _infer_query_type(self, query_dict: Dict[str, Any]) -> QueryType:
        """Infer query type from dictionary keys.

        Args:
            query_dict: Query dictionary

        Returns:
            Inferred query type
        """
        # Explicit query type
        if "query_type" in query_dict:
            return QueryType[query_dict["query_type"].upper()]

        # Infer from keys
        if "entity_similarity" in query_dict:
            if "actor1" in query_dict or "entity1" in query_dict:
                return QueryType.HYBRID_SEARCH
            return QueryType.SIMILARITY_SEARCH

        if "pattern_sequence" in query_dict:
            return QueryType.PATTERN_MATCH

        if "target_event" in query_dict or "path_length" in query_dict:
            return QueryType.TEMPORAL_PATH

        if ("actor1" in query_dict or "entity1" in query_dict) and \
           ("actor2" in query_dict or "entity2" in query_dict):
            return QueryType.ENTITY_PAIR

        if "actor1" in query_dict or "entity1" in query_dict:
            return QueryType.ENTITY_RELATIONS

        raise ValueError("Could not infer query type from dictionary keys")

    def _parse_datetime(self, dt_value: Any) -> Optional[datetime]:
        """Parse datetime from various formats.

        Args:
            dt_value: Datetime value (datetime, str, or None)

        Returns:
            Parsed datetime or None
        """
        if dt_value is None:
            return None
        if isinstance(dt_value, datetime):
            return dt_value
        if isinstance(dt_value, str):
            # Try ISO format
            try:
                return datetime.fromisoformat(dt_value)
            except ValueError:
                pass
            # Try common formats
            for fmt in ["%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%Y/%m/%d"]:
                try:
                    return datetime.strptime(dt_value, fmt)
                except ValueError:
                    continue
            raise ValueError(f"Could not parse datetime: {dt_value}")
        raise ValueError(f"Invalid datetime type: {type(dt_value)}")

    def _build_query_from_match(
        self, query_type: QueryType, match: re.Match, original_text: str
    ) -> ParsedQuery:
        """Build ParsedQuery from regex match.

        Args:
            query_type: Type of query matched
            match: Regex match object
            original_text: Original query text

        Returns:
            Parsed query
        """
        if query_type == QueryType.ENTITY_PAIR:
            return ParsedQuery(
                query_type=query_type,
                entity1=match.group(1).upper(),
                entity2=match.group(2).upper()
            )

        elif query_type == QueryType.ENTITY_RELATIONS:
            return ParsedQuery(
                query_type=query_type,
                entity1=match.group(1).upper()
            )

        elif query_type == QueryType.TEMPORAL_PATH:
            return ParsedQuery(
                query_type=query_type,
                entity1=match.group(1).upper(),
                entity2=match.group(2).upper()
            )

        elif query_type == QueryType.SIMILARITY_SEARCH:
            return ParsedQuery(
                query_type=query_type,
                search_mode=SearchMode.SEMANTIC,
                entity_similarity_query=match.group(1).upper()
            )

        raise ValueError(f"Unsupported query type: {query_type}")


class QueryOptimizer:
    """Optimize query execution plans."""

    def __init__(self):
        """Initialize query optimizer."""
        pass

    def optimize(self, query: ParsedQuery) -> Dict[str, Any]:
        """Generate optimized execution plan for query.

        Args:
            query: Parsed query

        Returns:
            Execution plan with optimization hints
        """
        plan = {
            "query_type": query.query_type.value,
            "estimated_cost": 0,
            "execution_steps": [],
            "use_cache": True,
            "parallel_execution": False
        }

        # ENTITY_PAIR: Direct lookup in actor-pair index
        if query.query_type == QueryType.ENTITY_PAIR:
            plan["execution_steps"] = [
                {"step": "actor_pair_index_lookup", "cost": 1},
                {"step": "temporal_filter", "cost": 2},
                {"step": "confidence_filter", "cost": 1}
            ]
            plan["estimated_cost"] = 4

        # ENTITY_RELATIONS: Get neighbors from graph
        elif query.query_type == QueryType.ENTITY_RELATIONS:
            plan["execution_steps"] = [
                {"step": "get_neighbors", "cost": 5},
                {"step": "temporal_filter", "cost": 3},
                {"step": "confidence_filter", "cost": 2}
            ]
            plan["estimated_cost"] = 10

        # TEMPORAL_PATH: BFS with temporal constraints
        elif query.query_type == QueryType.TEMPORAL_PATH:
            cost = query.path_length * 10  # Cost scales with path length
            plan["execution_steps"] = [
                {"step": "temporal_bfs", "cost": cost},
                {"step": "path_ranking", "cost": 5}
            ]
            plan["estimated_cost"] = cost + 5

        # SIMILARITY_SEARCH: Vector search in Qdrant
        elif query.query_type == QueryType.SIMILARITY_SEARCH:
            plan["execution_steps"] = [
                {"step": "vector_similarity_search", "cost": 8},
                {"step": "graph_lookup", "cost": 5},
                {"step": "temporal_filter", "cost": 3}
            ]
            plan["estimated_cost"] = 16

        # HYBRID_SEARCH: Both graph and vector
        elif query.query_type == QueryType.HYBRID_SEARCH:
            plan["parallel_execution"] = True
            plan["execution_steps"] = [
                {"step": "parallel_graph_vector_search", "cost": 15},
                {"step": "result_fusion", "cost": 5},
                {"step": "temporal_filter", "cost": 3}
            ]
            plan["estimated_cost"] = 23

        # PATTERN_MATCH: Complex pattern matching
        elif query.query_type == QueryType.PATTERN_MATCH:
            cost = len(query.pattern_sequence) * 15
            plan["execution_steps"] = [
                {"step": "pattern_matching", "cost": cost},
                {"step": "result_ranking", "cost": 5}
            ]
            plan["estimated_cost"] = cost + 5

        # Add optimization hints
        if query.time_window and query.time_window.start:
            # Small time window = cheaper
            span = (query.time_window.end - query.time_window.start).days
            if span < 30:
                plan["estimated_cost"] = int(plan["estimated_cost"] * 0.7)

        if query.constraints.quad_class:
            # Filtering by quad_class reduces search space
            plan["estimated_cost"] = int(plan["estimated_cost"] * 0.8)

        return plan


def create_parser() -> QueryParser:
    """Create query parser instance.

    Returns:
        Initialized QueryParser
    """
    return QueryParser()


def create_optimizer() -> QueryOptimizer:
    """Create query optimizer instance.

    Returns:
        Initialized QueryOptimizer
    """
    return QueryOptimizer()
