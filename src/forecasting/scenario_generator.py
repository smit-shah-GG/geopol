"""
Scenario generation using Gemini with structured output.

This module handles the creation of scenario trees from geopolitical questions,
using Gemini's structured output capabilities to ensure consistent formatting.
"""

import json
import uuid
from datetime import datetime
from typing import Dict, List, Optional

from src.forecasting.gemini_client import GeminiClient
from src.forecasting.models import (
    Entity,
    Scenario,
    ScenarioTree,
    TimelineEvent,
    ReasoningStep,
    ValidationFeedback,
    get_scenario_schema,
    get_scenario_tree_schema,
)


class ScenarioGenerator:
    """
    Generates structured scenario trees from geopolitical questions using Gemini.

    This class orchestrates the multi-step scenario generation process:
    1. Initial scenario generation from a question
    2. Placeholder for validation feedback (to be integrated later)
    3. Refinement based on feedback
    """

    def __init__(self, client: Optional[GeminiClient] = None):
        """
        Initialize the scenario generator.

        Args:
            client: Optional GeminiClient instance. Creates default if not provided.
        """
        self.client = client or GeminiClient()
        self.system_instruction = """You are a geopolitical forecasting expert.
Your task is to generate detailed, plausible scenarios for future events based on current
geopolitical context. Each scenario should include:
1. Clear description of what happens
2. Key entities involved (countries, organizations, leaders)
3. Timeline of events with relative timing
4. Reasoning chain explaining why this scenario is plausible
5. Probability assessment

Be specific, grounded in current events, and maintain logical consistency."""

    def generate_scenarios(
        self,
        question: str,
        context: Optional[List[str]] = None,
        num_scenarios: int = 3,
        include_branches: bool = True,
    ) -> ScenarioTree:
        """
        Generate initial scenarios for a forecasting question.

        Args:
            question: The geopolitical forecasting question.
            context: Optional list of context strings (recent events, constraints).
            num_scenarios: Number of alternative scenarios to generate.
            include_branches: Whether to include branching scenarios.

        Returns:
            ScenarioTree with initial scenarios.
        """
        # Build the prompt
        prompt = self._build_generation_prompt(question, context, num_scenarios, include_branches)

        # Generate with structured output
        try:
            response = self.client.generate_content(
                prompt=prompt,
                system_instruction=self.system_instruction,
                response_schema=get_scenario_tree_schema(),
            )

            # Parse the JSON response
            json_text = response.text
            scenario_data = json.loads(json_text)

            # Ensure metadata has context_count
            if "metadata" not in scenario_data:
                scenario_data["metadata"] = {}
            scenario_data["metadata"]["context_count"] = len(context) if context else 0

            # Convert to Pydantic models
            return self._parse_scenario_tree(scenario_data)

        except Exception as e:
            # Fallback: Generate without schema and parse manually
            print(f"Structured generation failed: {e}. Trying unstructured generation.")
            return self._generate_unstructured(question, context, num_scenarios)

    def _build_generation_prompt(
        self,
        question: str,
        context: Optional[List[str]],
        num_scenarios: int,
        include_branches: bool,
    ) -> str:
        """Build the prompt for scenario generation."""
        prompt_parts = [
            f"Forecasting Question: {question}",
            "",
            f"Generate {num_scenarios} alternative scenarios that could answer this question.",
        ]

        if context:
            prompt_parts.extend([
                "",
                "Context and Recent Events:",
                *context,
            ])

        if include_branches:
            prompt_parts.extend([
                "",
                "For each main scenario, consider 1-2 branching points where events could diverge.",
                "These branches represent critical decision points or uncertain events.",
            ])

        prompt_parts.extend([
            "",
            "For each scenario, provide:",
            "1. A unique scenario_id (e.g., 'scenario_1', 'branch_1a')",
            "2. Clear description of the outcome",
            "3. List of key entities with their roles",
            "4. Timeline with relative timing (T+1 week, T+1 month, etc.)",
            "5. Step-by-step reasoning chain",
            "6. Probability assessment (0.0 to 1.0)",
            "7. answers_affirmative (true/false): Does this scenario constitute a 'yes' answer to the question?",
            "   For example, if the question is 'Will X happen?', set true only for scenarios where X actually happens.",
            "",
            "IMPORTANT: Scenario probabilities must sum to approximately 1.0.",
            "Ensure scenarios are mutually exclusive and collectively exhaustive.",
        ])

        return "\n".join(prompt_parts)

    def _parse_scenario_tree(self, data: Dict) -> ScenarioTree:
        """Parse JSON data into ScenarioTree model."""
        # Create root scenario
        root_data = data.get("root_scenario", {})
        root_scenario = self._parse_scenario(root_data)

        # Parse or create metadata with required fields
        metadata = data.get("metadata", {})
        if "generated_at" not in metadata:
            metadata["generated_at"] = datetime.now().isoformat()
        if "model" not in metadata:
            metadata["model"] = getattr(self.client, "model_name", "gemini-3-pro-preview")
        if "context_count" not in metadata:
            metadata["context_count"] = 0

        # Create tree
        tree = ScenarioTree(
            question=data.get("question", ""),
            root_scenario=root_scenario,
            scenarios={root_scenario.scenario_id: root_scenario},
            metadata=metadata,
        )

        # Add additional scenarios - handle both array and dict formats
        scenarios_data = data.get("scenarios", [])
        if isinstance(scenarios_data, dict):
            # Legacy dict format
            for scenario_id, scenario_data in scenarios_data.items():
                if scenario_id != root_scenario.scenario_id:
                    scenario = self._parse_scenario(scenario_data)
                    tree.scenarios[scenario_id] = scenario
        elif isinstance(scenarios_data, list):
            # New array format from Gemini schema
            for scenario_data in scenarios_data:
                scenario = self._parse_scenario(scenario_data)
                if scenario.scenario_id != root_scenario.scenario_id:
                    tree.scenarios[scenario.scenario_id] = scenario

        return tree

    def _parse_scenario(self, data: Dict) -> Scenario:
        """Parse scenario data into Scenario model."""
        # Parse entities
        entities = []
        for entity_data in data.get("entities", []):
            entities.append(Entity(
                name=entity_data.get("name", "Unknown"),
                type=entity_data.get("type", "UNKNOWN"),
                role=entity_data.get("role", "UNKNOWN"),
                attributes=entity_data.get("attributes", {}),
            ))

        # Parse timeline
        timeline = []
        for event_data in data.get("timeline", []):
            timeline.append(TimelineEvent(
                relative_time=event_data.get("relative_time", "T+0"),
                description=event_data.get("description", ""),
                probability=event_data.get("probability", 0.5),
                preconditions=event_data.get("preconditions", []),
                effects=event_data.get("effects", []),
            ))

        # Parse reasoning path
        reasoning = []
        for step_data in data.get("reasoning_path", []):
            reasoning.append(ReasoningStep(
                step_number=step_data.get("step_number", 0),
                claim=step_data.get("claim", ""),
                evidence=step_data.get("evidence", []),
                confidence=step_data.get("confidence", 0.5),
                assumptions=step_data.get("assumptions", []),
            ))

        return Scenario(
            scenario_id=data.get("scenario_id", str(uuid.uuid4())),
            description=data.get("description", ""),
            entities=entities,
            timeline=timeline,
            probability=data.get("probability", 0.5),
            answers_affirmative=data.get("answers_affirmative", False),
            reasoning_path=reasoning,
            parent_id=data.get("parent_id"),
            child_ids=data.get("child_ids", []),
        )

    def _generate_unstructured(
        self,
        question: str,
        context: Optional[List[str]],
        num_scenarios: int,
    ) -> ScenarioTree:
        """Fallback: Generate scenarios without structured output."""
        prompt = self._build_generation_prompt(question, context, num_scenarios, False)
        prompt += "\n\nFormat your response as valid JSON matching the ScenarioTree structure."

        response = self.client.generate_content(
            prompt=prompt,
            system_instruction=self.system_instruction,
        )

        # Try to extract JSON from the response
        text = response.text

        # Look for JSON between ```json and ``` markers
        import re
        json_match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
        if json_match:
            json_text = json_match.group(1)
        else:
            # Try to find JSON-like content
            json_text = text

        try:
            data = json.loads(json_text)
            return self._parse_scenario_tree(data)
        except json.JSONDecodeError:
            # Last resort: Create a basic scenario tree from text
            return self._create_basic_scenario_tree(question, text)

    def _create_basic_scenario_tree(self, question: str, response_text: str) -> ScenarioTree:
        """Create a basic scenario tree from unstructured text."""
        # Create a single scenario from the response
        scenario = Scenario(
            scenario_id="scenario_1",
            description=response_text[:500],  # First 500 chars as description
            entities=[],
            timeline=[
                TimelineEvent(
                    relative_time="T+1 month",
                    description="Potential outcome based on analysis",
                    probability=0.5,
                )
            ],
            probability=0.5,
            reasoning_path=[
                ReasoningStep(
                    step_number=1,
                    claim="Analysis suggests this outcome",
                    confidence=0.5,
                )
            ],
        )

        return ScenarioTree(
            question=question,
            root_scenario=scenario,
            scenarios={"scenario_1": scenario},
        )

    def refine_scenarios(
        self,
        scenario_tree: ScenarioTree,
        feedback: List[ValidationFeedback],
    ) -> ScenarioTree:
        """
        Refine scenarios based on validation feedback.

        Args:
            scenario_tree: The original scenario tree.
            feedback: List of validation feedback for scenarios.

        Returns:
            Refined ScenarioTree.
        """
        # Build refinement prompt
        prompt = self._build_refinement_prompt(scenario_tree, feedback)

        # Generate refined scenarios
        response = self.client.generate_content(
            prompt=prompt,
            system_instruction=self.system_instruction + "\nRefine the scenarios based on the validation feedback provided.",
            response_schema=get_scenario_tree_schema(),
        )

        # Parse and return refined tree
        try:
            json_text = response.text
            scenario_data = json.loads(json_text)
            return self._parse_scenario_tree(scenario_data)
        except Exception as e:
            print(f"Refinement failed: {e}. Returning original tree.")
            return scenario_tree

    def _build_refinement_prompt(
        self,
        scenario_tree: ScenarioTree,
        feedback: List[ValidationFeedback],
    ) -> str:
        """Build prompt for scenario refinement."""
        prompt_parts = [
            f"Original Question: {scenario_tree.question}",
            "",
            "Original Scenarios:",
        ]

        # Summarize original scenarios
        for scenario_id, scenario in scenario_tree.scenarios.items():
            prompt_parts.append(f"- {scenario_id}: {scenario.description[:200]}...")

        prompt_parts.extend([
            "",
            "Validation Feedback:",
        ])

        # Add feedback
        for fb in feedback:
            prompt_parts.extend([
                f"\nScenario {fb.scenario_id}:",
                f"- Valid: {fb.is_valid}",
                f"- Confidence: {fb.confidence_score:.2f}",
            ])

            if fb.historical_patterns:
                prompt_parts.append(f"- Similar patterns: {', '.join(fb.historical_patterns[:3])}")

            if fb.contradictions:
                prompt_parts.append(f"- Issues: {', '.join(fb.contradictions[:3])}")

            if fb.suggestions:
                prompt_parts.append(f"- Suggestions: {', '.join(fb.suggestions[:3])}")

        prompt_parts.extend([
            "",
            "Please refine the scenarios based on this feedback:",
            "1. Address any contradictions with historical patterns",
            "2. Incorporate suggested improvements",
            "3. Adjust probabilities based on validation confidence",
            "4. Maintain scenario diversity and mutual exclusivity",
            "5. Preserve answers_affirmative tagging (true if scenario means 'yes' to the question)",
            "6. Ensure scenario probabilities still sum to approximately 1.0",
        ])

        return "\n".join(prompt_parts)

    def generate_with_validation_placeholder(
        self,
        question: str,
        context: Optional[List[str]] = None,
    ) -> ScenarioTree:
        """
        Generate scenarios with placeholder for validation.

        This method demonstrates the multi-step flow:
        1. Generate initial scenarios
        2. Prepare for validation (placeholder)
        3. Refine based on mock feedback

        Args:
            question: The forecasting question.
            context: Optional context.

        Returns:
            ScenarioTree ready for validation.
        """
        # Step 1: Generate initial scenarios
        initial_tree = self.generate_scenarios(question, context)

        # Step 2: Placeholder for validation
        # In the next plan, this will call the RAG pipeline and graph validator
        mock_feedback = []
        for scenario_id in list(initial_tree.scenarios.keys())[:2]:  # Mock feedback for first 2
            mock_feedback.append(ValidationFeedback(
                scenario_id=scenario_id,
                is_valid=True,
                confidence_score=0.75,
                historical_patterns=["Similar event in 2020", "Pattern matches regional conflict"],
                contradictions=[],
                suggestions=["Consider economic factors", "Include regional allies"],
            ))

        # Step 3: Refine if we have feedback
        if mock_feedback:
            refined_tree = self.refine_scenarios(initial_tree, mock_feedback)
            return refined_tree

        return initial_tree