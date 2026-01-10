"""
Tests for the ReasoningOrchestrator with mock Gemini responses.

This module tests the multi-step reasoning flow with mocked API responses
to ensure the orchestrator works correctly without requiring API keys.
"""

import json
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

import pytest

from src.forecasting.gemini_client import GeminiClient
from src.forecasting.models import (
    Entity,
    ForecastOutput,
    ReasoningStep,
    Scenario,
    ScenarioTree,
    TimelineEvent,
)
from src.forecasting.reasoning_orchestrator import ReasoningOrchestrator, ReasoningState
from src.forecasting.scenario_generator import ScenarioGenerator


@pytest.fixture
def mock_gemini_client():
    """Create a mock GeminiClient that doesn't require API key."""
    client = Mock(spec=GeminiClient)
    client.generate_content = Mock()
    return client


@pytest.fixture
def mock_scenario_tree():
    """Create a mock scenario tree for testing."""
    scenario1 = Scenario(
        scenario_id="scenario_1",
        description="Diplomatic resolution through negotiations",
        entities=[
            Entity(name="Country A", type="COUNTRY", role="ACTOR"),
            Entity(name="Country B", type="COUNTRY", role="TARGET"),
        ],
        timeline=[
            TimelineEvent(
                relative_time="T+1 week",
                description="Initial diplomatic contact",
                probability=0.7,
            ),
            TimelineEvent(
                relative_time="T+2 weeks",
                description="Formal negotiations begin",
                probability=0.6,
            ),
        ],
        probability=0.65,
        reasoning_path=[
            ReasoningStep(
                step_number=1,
                claim="Historical precedent suggests diplomatic resolution",
                confidence=0.7,
            ),
        ],
    )

    scenario2 = Scenario(
        scenario_id="scenario_2",
        description="Economic sanctions escalation",
        entities=[
            Entity(name="Country A", type="COUNTRY", role="ACTOR"),
            Entity(name="International Body", type="ORGANIZATION", role="MEDIATOR"),
        ],
        timeline=[
            TimelineEvent(
                relative_time="T+1 week",
                description="Sanctions announced",
                probability=0.4,
            ),
        ],
        probability=0.35,
        reasoning_path=[
            ReasoningStep(
                step_number=1,
                claim="Economic pressure as alternative to conflict",
                confidence=0.5,
            ),
        ],
    )

    tree = ScenarioTree(
        question="Will Country A and Country B resolve their dispute peacefully?",
        root_scenario=scenario1,
        scenarios={"scenario_1": scenario1, "scenario_2": scenario2},
    )

    return tree


@pytest.fixture
def mock_generator(mock_scenario_tree):
    """Create a mock ScenarioGenerator."""
    generator = Mock(spec=ScenarioGenerator)
    generator.generate_scenarios = Mock(return_value=mock_scenario_tree)
    generator.refine_scenarios = Mock(return_value=mock_scenario_tree)
    return generator


def test_orchestrator_initialization(mock_gemini_client):
    """Test that orchestrator initializes correctly."""
    orchestrator = ReasoningOrchestrator(
        client=mock_gemini_client,
        enable_rag=False,
        enable_graph_validation=False
    )
    assert orchestrator.client == mock_gemini_client
    assert orchestrator.generator is not None
    assert orchestrator.rag_pipeline is None
    assert orchestrator.graph_validator is None


def test_reasoning_state():
    """Test ReasoningState dataclass."""
    state = ReasoningState(
        question="Test question",
        context=["Context 1", "Context 2"],
    )

    assert state.question == "Test question"
    assert len(state.context) == 2
    assert state.initial_scenarios is None
    assert state.validation_feedback == []
    assert state.step_outputs == {}
    assert state.errors == []

    # Test adding step output
    state.add_step_output("test_step", {"result": "success"})
    assert "test_step" in state.step_outputs
    assert state.step_outputs["test_step"]["output"]["result"] == "success"

    # Test adding error
    state.add_error("test_step", "Test error")
    assert len(state.errors) == 1
    assert state.errors[0]["step"] == "test_step"
    assert state.errors[0]["error"] == "Test error"


def test_forecast_full_pipeline(mock_gemini_client, mock_generator):
    """Test the full forecasting pipeline with mocks."""
    orchestrator = ReasoningOrchestrator(
        client=mock_gemini_client,
        generator=mock_generator,
        enable_rag=False,
        enable_graph_validation=False
    )

    # Execute forecast
    forecast = orchestrator.forecast(
        question="Will Country A and Country B resolve their dispute peacefully?",
        context=["Recent tensions", "Historical precedent"],
        num_scenarios=2,
        enable_validation=True,
        enable_refinement=True,
    )

    # Verify forecast output
    assert isinstance(forecast, ForecastOutput)
    assert forecast.question == "Will Country A and Country B resolve their dispute peacefully?"
    assert forecast.probability > 0 and forecast.probability <= 1
    assert forecast.confidence > 0 and forecast.confidence <= 1
    assert len(forecast.selected_scenario_ids) > 0
    assert forecast.reasoning_summary != ""
    assert isinstance(forecast.timestamp, datetime)

    # Verify generator was called
    mock_generator.generate_scenarios.assert_called_once()
    mock_generator.refine_scenarios.assert_called_once()


def test_forecast_without_validation(mock_gemini_client, mock_generator):
    """Test forecasting without validation and refinement."""
    orchestrator = ReasoningOrchestrator(
        client=mock_gemini_client,
        generator=mock_generator,
    )

    forecast = orchestrator.forecast(
        question="Test question",
        enable_validation=False,
        enable_refinement=False,
    )

    assert isinstance(forecast, ForecastOutput)
    # Refinement should not be called when validation is disabled
    mock_generator.refine_scenarios.assert_not_called()


def test_forecast_error_handling(mock_gemini_client):
    """Test error handling in forecast pipeline."""
    # Create generator that raises error
    error_generator = Mock(spec=ScenarioGenerator)
    error_generator.generate_scenarios = Mock(side_effect=Exception("API Error"))

    orchestrator = ReasoningOrchestrator(
        client=mock_gemini_client,
        generator=error_generator,
    )

    forecast = orchestrator.forecast(
        question="Test question",
        context=["Context"],
    )

    # Should return error forecast
    assert isinstance(forecast, ForecastOutput)
    assert forecast.probability == 0.0
    assert forecast.confidence == 0.0
    assert "Unable to complete forecast" in forecast.prediction
    assert "API Error" in forecast.prediction


def test_validation_feedback_generation(mock_gemini_client, mock_scenario_tree):
    """Test that validation feedback is generated correctly."""
    orchestrator = ReasoningOrchestrator(client=mock_gemini_client)

    state = ReasoningState(
        question="Test question",
        context=[],
        initial_scenarios=mock_scenario_tree,
    )

    feedback = orchestrator._validate_scenarios(state)

    assert len(feedback) == 2  # One for each scenario
    for fb in feedback:
        assert fb.scenario_id in ["scenario_1", "scenario_2"]
        assert isinstance(fb.is_valid, bool)
        assert 0 <= fb.confidence_score <= 1
        assert isinstance(fb.historical_patterns, list)
        assert isinstance(fb.suggestions, list)


def test_step_explanation(mock_gemini_client, mock_generator):
    """Test step explanation functionality."""
    orchestrator = ReasoningOrchestrator(
        client=mock_gemini_client,
        generator=mock_generator,
    )

    forecast = orchestrator.forecast("Test question")

    # Get explanation for a step
    explanation = orchestrator.get_step_explanation(forecast, "generate_scenarios")

    assert explanation["step"] == "generate_scenarios"
    assert "Generate initial scenario tree" in explanation["description"]
    assert "Created" in explanation["impact"]


def test_mock_pattern_generation(mock_gemini_client, mock_scenario_tree):
    """Test mock pattern generation for validation."""
    orchestrator = ReasoningOrchestrator(client=mock_gemini_client)

    scenario = mock_scenario_tree.scenarios["scenario_1"]
    patterns = orchestrator._find_mock_patterns(scenario)

    assert isinstance(patterns, list)
    assert len(patterns) <= 3
    # Should generate patterns based on entities
    assert any("role" in p.lower() or "conflict" in p.lower() for p in patterns)


def test_reasoning_summary_building(mock_gemini_client, mock_scenario_tree):
    """Test building reasoning summary."""
    orchestrator = ReasoningOrchestrator(client=mock_gemini_client)

    state = ReasoningState(
        question="Test question",
        context=[],
        initial_scenarios=mock_scenario_tree,
    )

    # Add mock validation feedback
    state.validation_feedback = orchestrator._validate_scenarios(state)

    selected = list(mock_scenario_tree.scenarios.values())[:1]
    summary = orchestrator._build_reasoning_summary(state, selected)

    assert isinstance(summary, str)
    assert "Generated" in summary
    assert "scenarios" in summary
    assert "Validation" in summary


def test_evidence_extraction(mock_gemini_client, mock_scenario_tree):
    """Test evidence source extraction."""
    orchestrator = ReasoningOrchestrator(client=mock_gemini_client)

    scenarios = list(mock_scenario_tree.scenarios.values())
    sources = orchestrator._extract_evidence_sources(scenarios)

    assert isinstance(sources, list)
    assert len(sources) <= 10
    # Should include default sources
    assert "Historical pattern analysis" in sources
    assert "Graph-based validation" in sources


if __name__ == "__main__":
    pytest.main([__file__, "-v"])