"""
Pydantic models for structured scenario generation.

This module defines the data structures used for scenario trees, reasoning paths,
and forecasting outputs from the Gemini LLM.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class Entity(BaseModel):
    """Represents an entity (actor/country/organization) in a scenario."""

    name: str = Field(..., description="Name of the entity")
    type: str = Field(..., description="Type: COUNTRY, ORGANIZATION, PERSON, etc.")
    role: str = Field(..., description="Role in the scenario: ACTOR, TARGET, MEDIATOR, etc.")
    attributes: Dict[str, Any] = Field(default_factory=dict, description="Additional attributes")


class TimelineEvent(BaseModel):
    """Represents a single event in a scenario timeline."""

    timestamp: Optional[datetime] = Field(None, description="When event might occur")
    relative_time: str = Field(..., description="Relative timing: T+1 week, T+1 month, etc.")
    description: str = Field(..., description="What happens")
    probability: float = Field(0.5, description="Probability of this event occurring", ge=0.0, le=1.0)
    preconditions: List[str] = Field(default_factory=list, description="What must happen first")
    effects: List[str] = Field(default_factory=list, description="Consequences of this event")


class ReasoningStep(BaseModel):
    """Represents a step in the reasoning chain."""

    step_number: int = Field(..., description="Order in reasoning chain")
    claim: str = Field(..., description="The claim or assertion being made")
    evidence: List[str] = Field(default_factory=list, description="Supporting evidence")
    confidence: float = Field(0.5, description="Confidence in this step", ge=0.0, le=1.0)
    assumptions: List[str] = Field(default_factory=list, description="Assumptions made")


class Scenario(BaseModel):
    """Represents a single scenario in the scenario tree."""

    scenario_id: str = Field(..., description="Unique identifier for the scenario")
    description: str = Field(..., description="Natural language description of the scenario")
    entities: List[Entity] = Field(default_factory=list, description="Entities involved")
    timeline: List[TimelineEvent] = Field(default_factory=list, description="Sequence of events")
    probability: float = Field(0.5, description="Overall probability", ge=0.0, le=1.0)
    reasoning_path: List[ReasoningStep] = Field(default_factory=list, description="How we arrived at this scenario")
    parent_id: Optional[str] = Field(None, description="ID of parent scenario if branching")
    child_ids: List[str] = Field(default_factory=list, description="IDs of child scenarios")

    @field_validator('timeline')
    @classmethod
    def sort_timeline(cls, v: List[TimelineEvent]) -> List[TimelineEvent]:
        """Ensure timeline events are sorted by relative time."""
        # Sort by extracting numeric value from relative_time (e.g., "T+1" -> 1)
        import re

        def extract_time_value(event: TimelineEvent) -> float:
            match = re.search(r'T\+(\d+\.?\d*)\s*(\w+)', event.relative_time)
            if match:
                value = float(match.group(1))
                unit = match.group(2).lower()
                # Convert to days for sorting
                multipliers = {'day': 1, 'week': 7, 'month': 30, 'year': 365}
                return value * multipliers.get(unit.rstrip('s'), 1)
            return 0

        return sorted(v, key=extract_time_value)


class ScenarioTree(BaseModel):
    """Represents a tree of branching scenarios."""

    question: str = Field(..., description="The forecasting question being answered")
    root_scenario: Scenario = Field(..., description="The root/initial scenario")
    scenarios: Dict[str, Scenario] = Field(default_factory=dict, description="All scenarios by ID")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    def add_scenario(self, scenario: Scenario, parent_id: Optional[str] = None) -> None:
        """Add a scenario to the tree."""
        self.scenarios[scenario.scenario_id] = scenario
        if parent_id and parent_id in self.scenarios:
            scenario.parent_id = parent_id
            self.scenarios[parent_id].child_ids.append(scenario.scenario_id)

    def get_leaf_scenarios(self) -> List[Scenario]:
        """Get all leaf scenarios (no children)."""
        return [s for s in self.scenarios.values() if not s.child_ids]

    def get_path_to_scenario(self, scenario_id: str) -> List[Scenario]:
        """Get the path from root to a specific scenario."""
        path = []
        current = self.scenarios.get(scenario_id)
        while current:
            path.append(current)
            current = self.scenarios.get(current.parent_id) if current.parent_id else None
        return list(reversed(path))


class ValidationFeedback(BaseModel):
    """Feedback from graph validation for scenario refinement."""

    scenario_id: str = Field(..., description="Scenario being validated")
    is_valid: bool = Field(..., description="Whether scenario passes validation")
    confidence_score: float = Field(0.0, description="Graph-based confidence score", ge=0.0, le=1.0)
    historical_patterns: List[str] = Field(default_factory=list, description="Similar historical patterns found")
    contradictions: List[str] = Field(default_factory=list, description="Contradictions with historical data")
    suggestions: List[str] = Field(default_factory=list, description="Suggestions for improvement")


class ForecastOutput(BaseModel):
    """Final forecast output combining LLM and graph predictions."""

    question: str = Field(..., description="The forecasting question")
    prediction: str = Field(..., description="The main prediction")
    probability: float = Field(0.5, description="Overall probability", ge=0.0, le=1.0)
    confidence: float = Field(0.5, description="Confidence in the prediction", ge=0.0, le=1.0)
    scenario_tree: ScenarioTree = Field(..., description="Full scenario tree")
    selected_scenario_ids: List[str] = Field(default_factory=list, description="Most likely scenario IDs")
    reasoning_summary: str = Field(..., description="Summary of reasoning")
    evidence_sources: List[str] = Field(default_factory=list, description="Sources of evidence used")
    timestamp: datetime = Field(default_factory=datetime.now, description="When forecast was made")


# Schema definitions for Gemini structured output
def get_scenario_schema() -> Dict:
    """Get JSON schema for Scenario model for Gemini response_schema."""
    return {
        "type": "object",
        "properties": {
            "scenario_id": {"type": "string"},
            "description": {"type": "string"},
            "entities": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "type": {"type": "string"},
                        "role": {"type": "string"},
                        "attributes": {
                            "type": "object",
                            "properties": {
                                "category": {"type": "string"},
                                "confidence": {"type": "number"}
                            }
                        }
                    },
                    "required": ["name", "type", "role"]
                }
            },
            "timeline": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "relative_time": {"type": "string"},
                        "description": {"type": "string"},
                        "probability": {"type": "number"},
                        "preconditions": {"type": "array", "items": {"type": "string"}},
                        "effects": {"type": "array", "items": {"type": "string"}}
                    },
                    "required": ["relative_time", "description", "probability"]
                }
            },
            "probability": {"type": "number"},
            "reasoning_path": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "step_number": {"type": "integer"},
                        "claim": {"type": "string"},
                        "evidence": {"type": "array", "items": {"type": "string"}},
                        "confidence": {"type": "number"},
                        "assumptions": {"type": "array", "items": {"type": "string"}}
                    },
                    "required": ["step_number", "claim", "confidence"]
                }
            }
        },
        "required": ["scenario_id", "description", "probability", "timeline"]
    }


def get_scenario_tree_schema() -> Dict:
    """Get JSON schema for ScenarioTree model for Gemini response_schema."""
    scenario_schema = get_scenario_schema()
    return {
        "type": "object",
        "properties": {
            "question": {"type": "string"},
            "root_scenario": scenario_schema,
            "scenarios": {
                "type": "array",
                "items": scenario_schema
            },
            "metadata": {
                "type": "object",
                "properties": {
                    "generated_at": {"type": "string"},
                    "model": {"type": "string"},
                    "context_count": {"type": "integer"}
                }
            }
        },
        "required": ["question", "root_scenario"]
    }