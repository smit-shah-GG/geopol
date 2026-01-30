"""
Unified forecasting engine orchestrating all components.

This is the main entry point for the hybrid forecasting system:
1. ReasoningOrchestrator (LLM-based scenario generation)
2. TKGPredictor (graph-based pattern matching)
3. RAGPipeline (historical grounding)
4. EnsemblePredictor (weighted combination)

The engine:
- Accepts natural language questions
- Coordinates all components
- Returns structured forecasts with reasoning chains
- Provides explainability throughout
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.forecasting.ensemble_predictor import EnsemblePredictor, EnsemblePrediction
from src.forecasting.gemini_client import GeminiClient
from src.forecasting.graph_validator import GraphValidator
from src.forecasting.models import ForecastOutput
from src.forecasting.rag_pipeline import RAGPipeline
from src.forecasting.reasoning_orchestrator import ReasoningOrchestrator
from src.forecasting.tkg_predictor import TKGPredictor

logger = logging.getLogger(__name__)


class ForecastEngine:
    """
    Main entry point for hybrid geopolitical forecasting.

    Architecture:
    1. Question Analysis: Parse natural language query
    2. LLM Reasoning: Generate scenarios via ReasoningOrchestrator
    3. Graph Analysis: Validate patterns via TKGPredictor
    4. Historical Grounding: Retrieve context via RAGPipeline
    5. Ensemble Combination: Merge predictions with weights
    6. Output Formatting: Structure results for consumption

    Attributes:
        gemini_client: Client for Gemini API
        rag_pipeline: RAG pipeline for historical context
        tkg_predictor: TKG predictor for graph patterns
        reasoning_orchestrator: LLM orchestrator
        ensemble_predictor: Ensemble combiner
        alpha: Weight for LLM predictions (default: 0.6)
        temperature: Temperature for confidence calibration (default: 1.0)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        rag_pipeline: Optional[RAGPipeline] = None,
        tkg_predictor: Optional[TKGPredictor] = None,
        alpha: float = 0.6,
        temperature: float = 1.0,
        enable_rag: bool = True,
        enable_tkg: bool = True,
    ):
        """
        Initialize forecast engine.

        Args:
            api_key: Gemini API key (loads from env if None)
            rag_pipeline: Pre-configured RAG pipeline
            tkg_predictor: Pre-trained TKG predictor
            alpha: LLM weight for ensemble (0.0-1.0)
            temperature: Temperature for confidence calibration
            enable_rag: Whether to use RAG for historical grounding
            enable_tkg: Whether to use TKG for graph validation

        Note:
            If enable_tkg=True but tkg_predictor=None, TKG will be
            initialized but not used until trained via fit_tkg().
        """
        # Initialize Gemini client
        self.gemini_client = GeminiClient(api_key=api_key)

        # Initialize RAG pipeline
        if enable_rag:
            self.rag_pipeline = rag_pipeline or RAGPipeline()
        else:
            self.rag_pipeline = None

        # Initialize TKG predictor
        if enable_tkg:
            self.tkg_predictor = tkg_predictor or TKGPredictor()
        else:
            self.tkg_predictor = None

        # Initialize graph validator with TKG predictor (if enabled)
        graph_validator = None
        if enable_tkg and self.tkg_predictor:
            graph_validator = GraphValidator(tkg_predictor=self.tkg_predictor)

        # Initialize reasoning orchestrator
        self.reasoning_orchestrator = ReasoningOrchestrator(
            client=self.gemini_client,
            rag_pipeline=self.rag_pipeline,
            graph_validator=graph_validator,
            enable_rag=enable_rag,
            enable_graph_validation=enable_tkg,
        )

        # Initialize ensemble predictor
        self.ensemble_predictor = EnsemblePredictor(
            llm_orchestrator=self.reasoning_orchestrator,
            tkg_predictor=self.tkg_predictor,
            alpha=alpha,
            temperature=temperature,
        )

        # Configuration
        self.alpha = alpha
        self.temperature = temperature

        logger.info(
            f"ForecastEngine initialized (RAG={enable_rag}, "
            f"TKG={enable_tkg}, α={alpha:.2f}, T={temperature:.2f})"
        )

    def forecast(
        self,
        question: str,
        context: Optional[List[str]] = None,
        verbose: bool = False,
        use_cache: bool = True,
    ) -> Dict[str, Any]:
        """
        Generate forecast for a geopolitical question.

        This is the main forecasting method. It:
        1. Validates the question
        2. Retrieves historical context (if RAG enabled)
        3. Generates LLM scenarios
        4. Validates against graph patterns (if TKG trained)
        5. Combines predictions via ensemble
        6. Formats output

        Args:
            question: Natural language forecasting question
            context: Optional additional context strings
            verbose: Whether to include detailed reasoning steps
            use_cache: Whether to use RAG cache (if enabled)

        Returns:
            Dictionary with keys:
            - question: Original question
            - prediction: Main prediction text
            - probability: Forecast probability [0, 1]
            - confidence: Confidence score [0, 1]
            - scenarios: List of top scenarios
            - reasoning_chain: Step-by-step reasoning
            - evidence: Supporting evidence
            - ensemble_info: Component contributions
            - metadata: Additional information

        Raises:
            ValueError: If question is empty or invalid
        """
        if not question or not question.strip():
            raise ValueError("Question cannot be empty")

        logger.info(f"Forecasting: {question}")

        # Step 1: Retrieve historical context (if RAG enabled)
        if self.rag_pipeline and use_cache:
            if verbose:
                print("Retrieving historical context from RAG...", file=sys.stderr)
            context = self._retrieve_historical_context(question, context or [])
            if verbose:
                print(f"  Retrieved {len(context)} context items", file=sys.stderr)

        # Step 2: Generate ensemble forecast
        if verbose:
            print("Generating ensemble forecast...", file=sys.stderr)

        ensemble_pred, forecast = self.ensemble_predictor.predict(
            question=question, context=context
        )

        # Step 3: Format output
        output = self._format_output(
            question=question,
            forecast=forecast,
            ensemble_pred=ensemble_pred,
            verbose=verbose,
        )

        logger.info(
            f"Forecast complete: P={output['probability']:.3f}, "
            f"C={output['confidence']:.3f}"
        )

        return output

    def _retrieve_historical_context(
        self, question: str, context: List[str]
    ) -> List[str]:
        """
        Retrieve historical context via RAG pipeline.

        Args:
            question: Forecasting question
            context: Existing context strings

        Returns:
            Extended context list with historical patterns
        """
        if not self.rag_pipeline or not self.rag_pipeline.index:
            return context

        try:
            # Extract entities from question (simple heuristics)
            entities = self._extract_entities_from_question(question)

            # Query RAG pipeline
            rag_context = self.rag_pipeline.query_historical_context(
                scenario_description=question,
                entities=entities,
                lookback_days=365 * 2,  # 2 years
            )

            # Add patterns to context
            extended_context = context.copy()
            for pattern in rag_context["retrieved_patterns"][:3]:
                extended_context.append(pattern["summary"][:200])

            return extended_context

        except Exception as e:
            logger.warning(f"Failed to retrieve historical context: {e}")
            return context

    def _extract_entities_from_question(self, question: str) -> List[str]:
        """
        Extract entities from question using simple heuristics.

        This is a basic implementation. In production, use NER.

        Args:
            question: Question text

        Returns:
            List of entity names
        """
        # Common geopolitical entities
        common_entities = [
            "Russia",
            "Ukraine",
            "China",
            "Taiwan",
            "USA",
            "Iran",
            "Israel",
            "North Korea",
            "South Korea",
            "NATO",
            "UN",
            "EU",
        ]

        entities = []
        question_upper = question.upper()

        for entity in common_entities:
            if entity.upper() in question_upper:
                entities.append(entity)

        return entities

    def _format_output(
        self,
        question: str,
        forecast: ForecastOutput,
        ensemble_pred: EnsemblePrediction,
        verbose: bool,
    ) -> Dict[str, Any]:
        """
        Format forecast output as structured dictionary.

        Args:
            question: Original question
            forecast: ForecastOutput from ensemble
            ensemble_pred: Ensemble prediction metadata
            verbose: Whether to include detailed info

        Returns:
            Structured output dictionary
        """
        # Extract top scenarios
        scenarios = []
        scenario_tree = forecast.scenario_tree
        sorted_scenarios = sorted(
            scenario_tree.scenarios.values(),
            key=lambda s: s.probability,
            reverse=True,
        )

        for scenario in sorted_scenarios[:3]:  # Top 3
            scenarios.append(
                {
                    "description": scenario.description,
                    "probability": scenario.probability,
                    "entities": [
                        {
                            "name": e.name,
                            "type": e.type,
                            "role": e.role,
                        }
                        for e in scenario.entities
                    ],
                    "timeline": [
                        {
                            "time": event.relative_time,
                            "description": event.description,
                            "probability": event.probability,
                        }
                        for event in scenario.timeline[:5]  # First 5 events
                    ],
                }
            )

        # Build reasoning chain
        reasoning_chain = []
        if verbose and sorted_scenarios:
            top_scenario = sorted_scenarios[0]
            for step in top_scenario.reasoning_path:
                reasoning_chain.append(
                    {
                        "step": step.step_number,
                        "claim": step.claim,
                        "confidence": step.confidence,
                        "evidence": step.evidence,
                    }
                )

        # Ensemble information
        ensemble_info = {
            "llm_available": ensemble_pred.llm_prediction.available,
            "tkg_available": ensemble_pred.tkg_prediction.available,
            "llm_probability": ensemble_pred.llm_prediction.probability
            if ensemble_pred.llm_prediction.available
            else None,
            "tkg_probability": ensemble_pred.tkg_prediction.probability
            if ensemble_pred.tkg_prediction.available
            else None,
            "weights": {
                "llm": ensemble_pred.weights_used[0],
                "tkg": ensemble_pred.weights_used[1],
            },
            "temperature": ensemble_pred.temperature,
        }

        # Build output
        output = {
            "question": question,
            "prediction": forecast.prediction,
            "probability": forecast.probability,
            "confidence": forecast.confidence,
            "scenarios": scenarios,
            "reasoning_summary": forecast.reasoning_summary,
            "evidence": forecast.evidence_sources,
            "ensemble_info": ensemble_info,
            "metadata": {
                "timestamp": forecast.timestamp.isoformat(),
                "num_scenarios": len(scenario_tree.scenarios),
            },
        }

        if verbose:
            output["reasoning_chain"] = reasoning_chain

        return output

    def fit_tkg(self, graph, recent_days: int = 30) -> None:
        """
        Train TKG predictor on historical graph data.

        Args:
            graph: NetworkX MultiDiGraph with temporal edges
            recent_days: Number of recent days to use for training

        Raises:
            ValueError: If TKG not enabled or graph invalid
        """
        if self.tkg_predictor is None:
            raise ValueError("TKG predictor not enabled")

        logger.info(f"Training TKG predictor on {recent_days} days of data")
        self.tkg_predictor.fit(graph, recent_days=recent_days)
        logger.info("TKG predictor training complete")

    def load_tkg(self, path: Path) -> None:
        """
        Load pre-trained TKG predictor.

        Args:
            path: Path to TKG checkpoint directory

        Raises:
            ValueError: If TKG not enabled
        """
        if self.tkg_predictor is None:
            raise ValueError("TKG predictor not enabled")

        logger.info(f"Loading TKG predictor from {path}")
        self.tkg_predictor.load(path)
        logger.info("TKG predictor loaded successfully")

    def save_tkg(self, path: Path) -> None:
        """
        Save trained TKG predictor.

        Args:
            path: Path to save checkpoint

        Raises:
            ValueError: If TKG not enabled or not trained
        """
        if self.tkg_predictor is None:
            raise ValueError("TKG predictor not enabled")

        if not self.tkg_predictor.trained:
            raise ValueError("TKG predictor not trained")

        logger.info(f"Saving TKG predictor to {path}")
        path.mkdir(parents=True, exist_ok=True)
        self.tkg_predictor.save(path)
        logger.info("TKG predictor saved successfully")

    def update_ensemble_weights(self, alpha: float) -> None:
        """
        Update ensemble weights dynamically.

        Args:
            alpha: New LLM weight (0.0-1.0)

        Raises:
            ValueError: If alpha not in [0, 1]
        """
        self.ensemble_predictor.update_weights(alpha)
        self.alpha = alpha
        logger.info(f"Updated ensemble weights: α={alpha:.2f}")

    def update_temperature(self, temperature: float) -> None:
        """
        Update temperature scaling factor.

        Args:
            temperature: New temperature (> 0)

        Raises:
            ValueError: If temperature <= 0
        """
        self.ensemble_predictor.update_temperature(temperature)
        self.temperature = temperature
        logger.info(f"Updated temperature: T={temperature:.2f}")

    def get_engine_status(self) -> Dict[str, Any]:
        """
        Get status of all engine components.

        Returns:
            Dictionary with component status information
        """
        return {
            "gemini_client": {"initialized": self.gemini_client is not None},
            "rag_pipeline": {
                "enabled": self.rag_pipeline is not None,
                "has_index": self.rag_pipeline.index is not None
                if self.rag_pipeline
                else False,
            },
            "tkg_predictor": {
                "enabled": self.tkg_predictor is not None,
                "trained": self.tkg_predictor.trained
                if self.tkg_predictor
                else False,
            },
            "ensemble": {
                "alpha": self.alpha,
                "temperature": self.temperature,
            },
        }
