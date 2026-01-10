"""
Multi-step reasoning orchestrator for geopolitical forecasting.

This module coordinates the full forecasting pipeline:
1. Generate initial scenarios from question
2. Prepare validation feedback (placeholder for graph integration)
3. Refine scenarios based on feedback
4. Extract final predictions with confidence scores

The orchestrator preserves reasoning chains for explainability throughout the process.
"""

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from src.forecasting.gemini_client import GeminiClient
from src.forecasting.models import (
    ForecastOutput,
    Scenario,
    ScenarioTree,
    ValidationFeedback,
)
from src.forecasting.scenario_generator import ScenarioGenerator


@dataclass
class ReasoningState:
    """Tracks state throughout the reasoning process."""

    question: str
    context: List[str]
    initial_scenarios: Optional[ScenarioTree] = None
    validation_feedback: List[ValidationFeedback] = None
    refined_scenarios: Optional[ScenarioTree] = None
    final_prediction: Optional[ForecastOutput] = None
    step_outputs: Dict[str, Any] = None
    errors: List[str] = None

    def __post_init__(self):
        if self.validation_feedback is None:
            self.validation_feedback = []
        if self.step_outputs is None:
            self.step_outputs = {}
        if self.errors is None:
            self.errors = []

    def add_step_output(self, step_name: str, output: Any) -> None:
        """Record output from a reasoning step."""
        self.step_outputs[step_name] = {
            "timestamp": datetime.now().isoformat(),
            "output": output,
        }

    def add_error(self, step_name: str, error: str) -> None:
        """Record an error during processing."""
        self.errors.append({
            "step": step_name,
            "error": error,
            "timestamp": datetime.now().isoformat(),
        })


class ReasoningOrchestrator:
    """
    Orchestrates the multi-step reasoning flow for forecasting.

    This class manages the complete pipeline from question to prediction,
    coordinating between scenario generation, validation, and refinement.
    """

    def __init__(
        self,
        client: Optional[GeminiClient] = None,
        generator: Optional[ScenarioGenerator] = None,
    ):
        """
        Initialize the orchestrator.

        Args:
            client: Optional GeminiClient instance.
            generator: Optional ScenarioGenerator instance.
        """
        self.client = client or GeminiClient()
        self.generator = generator or ScenarioGenerator(client=self.client)

        # Placeholder for future components
        self.rag_pipeline = None  # Will be added in 03-02
        self.graph_validator = None  # Will be added in 03-03

    def forecast(
        self,
        question: str,
        context: Optional[List[str]] = None,
        num_scenarios: int = 3,
        enable_validation: bool = True,
        enable_refinement: bool = True,
    ) -> ForecastOutput:
        """
        Execute the full forecasting pipeline.

        Args:
            question: The geopolitical forecasting question.
            context: Optional list of context strings.
            num_scenarios: Number of scenarios to generate.
            enable_validation: Whether to validate scenarios (uses mock for now).
            enable_refinement: Whether to refine based on validation.

        Returns:
            ForecastOutput with final prediction and reasoning.
        """
        # Initialize state
        state = ReasoningState(
            question=question,
            context=context or [],
        )

        try:
            # Step 1: Generate initial scenarios
            state.initial_scenarios = self._generate_scenarios(
                state, num_scenarios
            )

            # Step 2: Validate scenarios (placeholder for now)
            if enable_validation:
                state.validation_feedback = self._validate_scenarios(state)

            # Step 3: Refine scenarios based on feedback
            if enable_refinement and state.validation_feedback:
                state.refined_scenarios = self._refine_scenarios(state)
            else:
                state.refined_scenarios = state.initial_scenarios

            # Step 4: Extract final prediction
            state.final_prediction = self._extract_prediction(state)

            return state.final_prediction

        except Exception as e:
            state.add_error("orchestrator", str(e))
            # Return a basic forecast with error information
            return self._create_error_forecast(state, str(e))

    def _generate_scenarios(
        self,
        state: ReasoningState,
        num_scenarios: int,
    ) -> ScenarioTree:
        """Step 1: Generate initial scenarios."""
        try:
            scenarios = self.generator.generate_scenarios(
                question=state.question,
                context=state.context,
                num_scenarios=num_scenarios,
                include_branches=True,
            )
            state.add_step_output("generate_scenarios", {
                "num_scenarios": len(scenarios.scenarios),
                "has_branches": any(s.child_ids for s in scenarios.scenarios.values()),
            })
            return scenarios
        except Exception as e:
            state.add_error("generate_scenarios", str(e))
            raise

    def _validate_scenarios(self, state: ReasoningState) -> List[ValidationFeedback]:
        """
        Step 2: Validate scenarios against historical patterns.

        This is a placeholder that generates mock feedback.
        In 03-02 and 03-03, this will:
        1. Use RAG pipeline to find similar historical events
        2. Use graph validator to check pattern consistency
        3. Generate real validation feedback
        """
        feedback = []

        # Mock validation for now
        for scenario_id, scenario in state.initial_scenarios.scenarios.items():
            # Simulate validation with mock scores
            is_valid = scenario.probability > 0.3  # Simple mock rule
            confidence = min(0.9, scenario.probability + 0.2)  # Mock confidence

            fb = ValidationFeedback(
                scenario_id=scenario_id,
                is_valid=is_valid,
                confidence_score=confidence,
                historical_patterns=self._find_mock_patterns(scenario),
                contradictions=self._find_mock_contradictions(scenario),
                suggestions=self._generate_mock_suggestions(scenario),
            )
            feedback.append(fb)

        state.add_step_output("validate_scenarios", {
            "num_validated": len(feedback),
            "num_valid": sum(1 for f in feedback if f.is_valid),
            "avg_confidence": sum(f.confidence_score for f in feedback) / len(feedback) if feedback else 0,
        })

        return feedback

    def _find_mock_patterns(self, scenario: Scenario) -> List[str]:
        """Generate mock historical patterns for testing."""
        patterns = []

        # Generate based on entities in scenario
        for entity in scenario.entities[:2]:  # First 2 entities
            if entity.type == "COUNTRY":
                patterns.append(f"Similar {entity.role.lower()} role in 2020 conflict")
            elif entity.type == "ORGANIZATION":
                patterns.append(f"{entity.name} involvement in regional disputes")

        # Add generic patterns
        if scenario.probability > 0.5:
            patterns.append("High-probability pattern matches historical precedent")

        return patterns[:3]  # Limit to 3 patterns

    def _find_mock_contradictions(self, scenario: Scenario) -> List[str]:
        """Generate mock contradictions for testing."""
        contradictions = []

        # Generate based on probability
        if scenario.probability < 0.2:
            contradictions.append("Low historical precedent for this scenario")

        # Check timeline
        if len(scenario.timeline) > 5:
            contradictions.append("Timeline complexity exceeds typical patterns")

        return contradictions

    def _generate_mock_suggestions(self, scenario: Scenario) -> List[str]:
        """Generate mock improvement suggestions."""
        suggestions = []

        # Check for missing elements
        if not scenario.entities:
            suggestions.append("Add key actors and stakeholders")

        if len(scenario.reasoning_path) < 3:
            suggestions.append("Expand reasoning chain with more evidence")

        # Generic suggestions
        suggestions.append("Consider economic impact factors")
        suggestions.append("Include regional alliance dynamics")

        return suggestions[:2]  # Limit to 2 suggestions

    def _refine_scenarios(self, state: ReasoningState) -> ScenarioTree:
        """Step 3: Refine scenarios based on validation feedback."""
        try:
            refined = self.generator.refine_scenarios(
                scenario_tree=state.initial_scenarios,
                feedback=state.validation_feedback,
            )

            state.add_step_output("refine_scenarios", {
                "num_scenarios": len(refined.scenarios),
                "refinement_applied": True,
            })

            return refined
        except Exception as e:
            state.add_error("refine_scenarios", str(e))
            # Return original scenarios if refinement fails
            return state.initial_scenarios

    def _extract_prediction(self, state: ReasoningState) -> ForecastOutput:
        """Step 4: Extract final prediction from refined scenarios."""
        scenarios = state.refined_scenarios or state.initial_scenarios

        # Select most likely scenarios
        sorted_scenarios = sorted(
            scenarios.scenarios.values(),
            key=lambda s: s.probability,
            reverse=True,
        )
        selected = sorted_scenarios[:2]  # Top 2 scenarios

        # Calculate weighted prediction
        if selected:
            # Weight by probability
            total_prob = sum(s.probability for s in selected)
            weighted_prob = sum(s.probability ** 2 for s in selected) / total_prob if total_prob > 0 else 0.5
        else:
            weighted_prob = 0.5

        # Calculate confidence based on validation feedback
        if state.validation_feedback:
            avg_confidence = sum(f.confidence_score for f in state.validation_feedback) / len(state.validation_feedback)
        else:
            avg_confidence = 0.5

        # Build reasoning summary
        reasoning_summary = self._build_reasoning_summary(state, selected)

        # Extract evidence sources
        evidence_sources = self._extract_evidence_sources(selected)

        # Create final forecast
        forecast = ForecastOutput(
            question=state.question,
            prediction=selected[0].description if selected else "Unable to generate prediction",
            probability=weighted_prob,
            confidence=avg_confidence,
            scenario_tree=scenarios,
            selected_scenario_ids=[s.scenario_id for s in selected],
            reasoning_summary=reasoning_summary,
            evidence_sources=evidence_sources,
            timestamp=datetime.now(),
        )

        state.add_step_output("extract_prediction", {
            "selected_scenarios": len(selected),
            "final_probability": weighted_prob,
            "final_confidence": avg_confidence,
        })

        return forecast

    def _build_reasoning_summary(
        self,
        state: ReasoningState,
        selected_scenarios: List[Scenario],
    ) -> str:
        """Build a summary of the reasoning process."""
        parts = []

        # Summarize scenarios
        parts.append(f"Generated {len(state.initial_scenarios.scenarios)} initial scenarios.")

        # Validation results
        if state.validation_feedback:
            valid_count = sum(1 for f in state.validation_feedback if f.is_valid)
            parts.append(f"Validation: {valid_count}/{len(state.validation_feedback)} scenarios validated.")

        # Selected scenarios
        if selected_scenarios:
            parts.append(f"Selected top {len(selected_scenarios)} scenarios based on probability and validation.")

            # Add reasoning from top scenario
            top_scenario = selected_scenarios[0]
            if top_scenario.reasoning_path:
                key_claims = [step.claim for step in top_scenario.reasoning_path[:2]]
                parts.append(f"Key reasoning: {'; '.join(key_claims)}")

        return " ".join(parts)

    def _extract_evidence_sources(self, scenarios: List[Scenario]) -> List[str]:
        """Extract unique evidence sources from scenarios."""
        sources = set()

        for scenario in scenarios:
            # Extract from reasoning steps
            for step in scenario.reasoning_path:
                sources.update(step.evidence)

            # Extract from validation feedback patterns
            # (In real implementation, this would come from RAG results)
            sources.add("Historical pattern analysis")
            sources.add("Graph-based validation")

        return list(sources)[:10]  # Limit to 10 sources

    def _create_error_forecast(self, state: ReasoningState, error_msg: str) -> ForecastOutput:
        """Create a forecast output when an error occurs."""
        # Use initial scenarios if available
        scenarios = state.initial_scenarios or ScenarioTree(
            question=state.question,
            root_scenario=Scenario(
                scenario_id="error",
                description=f"Error during forecasting: {error_msg}",
                probability=0.0,
            ),
            scenarios={},
        )

        return ForecastOutput(
            question=state.question,
            prediction=f"Unable to complete forecast: {error_msg}",
            probability=0.0,
            confidence=0.0,
            scenario_tree=scenarios,
            selected_scenario_ids=[],
            reasoning_summary=f"Forecasting failed at step: {state.errors[-1]['step'] if state.errors else 'unknown'}",
            evidence_sources=[],
            timestamp=datetime.now(),
        )

    def get_step_explanation(
        self,
        forecast: ForecastOutput,
        step_name: str,
    ) -> Dict[str, Any]:
        """
        Get detailed explanation of a specific reasoning step.

        This supports explainability by allowing inspection of intermediate results.

        Args:
            forecast: The forecast output.
            step_name: Name of the step to explain.

        Returns:
            Dictionary with step details.
        """
        # Extract from forecast metadata
        # In a full implementation, we'd store step outputs in the forecast
        explanation = {
            "step": step_name,
            "description": self._get_step_description(step_name),
            "impact": self._assess_step_impact(forecast, step_name),
        }

        return explanation

    def _get_step_description(self, step_name: str) -> str:
        """Get human-readable description of a step."""
        descriptions = {
            "generate_scenarios": "Generate initial scenario tree from the forecasting question",
            "validate_scenarios": "Validate scenarios against historical patterns and graph data",
            "refine_scenarios": "Refine scenarios based on validation feedback",
            "extract_prediction": "Extract final prediction from refined scenarios",
        }
        return descriptions.get(step_name, "Unknown step")

    def _assess_step_impact(self, forecast: ForecastOutput, step_name: str) -> str:
        """Assess the impact of a step on the final prediction."""
        if step_name == "generate_scenarios":
            return f"Created {len(forecast.scenario_tree.scenarios)} scenarios"
        elif step_name == "validate_scenarios":
            return f"Validated scenarios with average confidence {forecast.confidence:.2f}"
        elif step_name == "refine_scenarios":
            return "Refined scenarios based on historical patterns"
        elif step_name == "extract_prediction":
            return f"Selected {len(forecast.selected_scenario_ids)} scenarios for final prediction"
        return "Impact not assessed"