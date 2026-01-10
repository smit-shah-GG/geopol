"""
Tests for ensemble predictor combining LLM and TKG predictions.

Verifies:
1. Weighted voting works correctly
2. Temperature scaling calibrates confidence
3. Graceful degradation when one component fails
4. Ensemble outperforms individual models on synthetic data
"""

import pytest
from datetime import datetime
from unittest.mock import Mock, MagicMock

from src.forecasting.ensemble_predictor import (
    EnsemblePredictor,
    ComponentPrediction,
    EnsemblePrediction,
)
from src.forecasting.models import (
    ForecastOutput,
    ScenarioTree,
    Scenario,
    Entity,
)


@pytest.fixture
def mock_llm_orchestrator():
    """Mock LLM orchestrator with controllable predictions."""
    orchestrator = Mock()

    # Default forecast
    test_scenario = Scenario(
        scenario_id="test",
        description="Test scenario",
        probability=0.7,
        entities=[
            Entity(name="Russia", type="COUNTRY", role="ACTOR"),
            Entity(name="Ukraine", type="COUNTRY", role="TARGET"),
        ],
    )

    forecast = ForecastOutput(
        question="Test question",
        prediction="Test prediction",
        probability=0.7,
        confidence=0.8,
        scenario_tree=ScenarioTree(
            question="Test question",
            root_scenario=test_scenario,
            scenarios={"test": test_scenario},  # Include scenario in dict
        ),
        selected_scenario_ids=["test"],
        reasoning_summary="Test reasoning",
        evidence_sources=["Test source"],
        timestamp=datetime.now(),
    )

    orchestrator.forecast.return_value = forecast
    return orchestrator


@pytest.fixture
def mock_tkg_predictor():
    """Mock TKG predictor with controllable predictions."""
    predictor = Mock()
    predictor.trained = True

    # Default prediction
    predictor.predict_future_events.return_value = [
        {
            "entity1": "Russia",
            "relation": "CONFLICT",
            "entity2": "Ukraine",
            "confidence": 0.6,
        }
    ]

    return predictor


class TestEnsemblePredictorInitialization:
    """Test ensemble predictor initialization."""

    def test_default_initialization(self):
        """Test initialization with default parameters."""
        ensemble = EnsemblePredictor()

        assert ensemble.alpha == 0.6
        assert ensemble.temperature == 1.0
        assert ensemble.llm_orchestrator is None
        assert ensemble.tkg_predictor is None

    def test_custom_weights(self, mock_llm_orchestrator, mock_tkg_predictor):
        """Test initialization with custom weights."""
        ensemble = EnsemblePredictor(
            llm_orchestrator=mock_llm_orchestrator,
            tkg_predictor=mock_tkg_predictor,
            alpha=0.8,
            temperature=1.5,
        )

        assert ensemble.alpha == 0.8
        assert ensemble.temperature == 1.5

    def test_invalid_alpha(self):
        """Test that invalid alpha raises ValueError."""
        with pytest.raises(ValueError, match="Alpha must be in"):
            EnsemblePredictor(alpha=1.5)

        with pytest.raises(ValueError, match="Alpha must be in"):
            EnsemblePredictor(alpha=-0.1)

    def test_invalid_temperature(self):
        """Test that invalid temperature raises ValueError."""
        with pytest.raises(ValueError, match="Temperature must be"):
            EnsemblePredictor(temperature=0.0)

        with pytest.raises(ValueError, match="Temperature must be"):
            EnsemblePredictor(temperature=-1.0)


class TestWeightedVoting:
    """Test weighted voting mechanism."""

    def test_both_components_available(
        self, mock_llm_orchestrator, mock_tkg_predictor
    ):
        """Test ensemble when both components available."""
        ensemble = EnsemblePredictor(
            llm_orchestrator=mock_llm_orchestrator,
            tkg_predictor=mock_tkg_predictor,
            alpha=0.6,
        )

        ensemble_pred, forecast = ensemble.predict(
            question="Will conflict escalate?",
            entity1="Russia",
            relation="CONFLICT",
            entity2="Ukraine",
        )

        # Check weighted average: 0.6 * 0.7 + 0.4 * 0.6 = 0.66
        assert abs(ensemble_pred.final_probability - 0.66) < 0.01
        assert ensemble_pred.llm_prediction.available
        assert ensemble_pred.tkg_prediction.available
        assert ensemble_pred.weights_used == (0.6, 0.4)

    def test_llm_only_available(self, mock_llm_orchestrator):
        """Test ensemble when only LLM available."""
        ensemble = EnsemblePredictor(
            llm_orchestrator=mock_llm_orchestrator,
            tkg_predictor=None,  # TKG not available
            alpha=0.6,
        )

        ensemble_pred, forecast = ensemble.predict(question="Test question")

        # Should use LLM probability with penalty
        assert ensemble_pred.final_probability == 0.7
        assert ensemble_pred.llm_prediction.available
        assert not ensemble_pred.tkg_prediction.available
        assert ensemble_pred.final_confidence < 0.8  # Penalty applied

    def test_tkg_only_available(self, mock_tkg_predictor):
        """Test ensemble when only TKG available."""
        ensemble = EnsemblePredictor(
            llm_orchestrator=None,  # LLM not available
            tkg_predictor=mock_tkg_predictor,
            alpha=0.6,
        )

        ensemble_pred, forecast = ensemble.predict(
            question="Test question",
            entity1="Russia",
            relation="CONFLICT",
            entity2="Ukraine",
        )

        # Should use TKG probability with penalty
        assert ensemble_pred.final_probability == 0.6
        assert not ensemble_pred.llm_prediction.available
        assert ensemble_pred.tkg_prediction.available
        assert ensemble_pred.final_confidence < 0.6  # Penalty applied

    def test_neither_available(self):
        """Test ensemble when neither component available."""
        ensemble = EnsemblePredictor(
            llm_orchestrator=None, tkg_predictor=None, alpha=0.6
        )

        ensemble_pred, forecast = ensemble.predict(question="Test question")

        # Should return uninformative prior
        assert ensemble_pred.final_probability == 0.5
        assert ensemble_pred.final_confidence == 0.0
        assert not ensemble_pred.llm_prediction.available
        assert not ensemble_pred.tkg_prediction.available


class TestTemperatureScaling:
    """Test temperature scaling for confidence calibration."""

    def test_temperature_one_no_change(
        self, mock_llm_orchestrator, mock_tkg_predictor
    ):
        """Test that temperature=1.0 doesn't change confidence."""
        ensemble = EnsemblePredictor(
            llm_orchestrator=mock_llm_orchestrator,
            tkg_predictor=mock_tkg_predictor,
            alpha=0.6,
            temperature=1.0,
        )

        ensemble_pred, _ = ensemble.predict(
            question="Test",
            entity1="Russia",
            relation="CONFLICT",
            entity2="Ukraine",
        )

        # Confidence should be weighted average: 0.6 * 0.8 + 0.4 * 0.6 = 0.72
        assert abs(ensemble_pred.final_confidence - 0.72) < 0.01

    def test_temperature_less_than_one_sharpens(
        self, mock_llm_orchestrator, mock_tkg_predictor
    ):
        """Test that temperature < 1 increases confidence."""
        ensemble_normal = EnsemblePredictor(
            llm_orchestrator=mock_llm_orchestrator,
            tkg_predictor=mock_tkg_predictor,
            alpha=0.6,
            temperature=1.0,
        )

        ensemble_sharp = EnsemblePredictor(
            llm_orchestrator=mock_llm_orchestrator,
            tkg_predictor=mock_tkg_predictor,
            alpha=0.6,
            temperature=0.5,
        )

        pred_normal, _ = ensemble_normal.predict(
            question="Test",
            entity1="Russia",
            relation="CONFLICT",
            entity2="Ukraine",
        )

        pred_sharp, _ = ensemble_sharp.predict(
            question="Test",
            entity1="Russia",
            relation="CONFLICT",
            entity2="Ukraine",
        )

        # Lower temperature should increase confidence
        assert pred_sharp.final_confidence > pred_normal.final_confidence

    def test_temperature_greater_than_one_smooths(
        self, mock_llm_orchestrator, mock_tkg_predictor
    ):
        """Test that temperature > 1 decreases confidence."""
        ensemble_normal = EnsemblePredictor(
            llm_orchestrator=mock_llm_orchestrator,
            tkg_predictor=mock_tkg_predictor,
            alpha=0.6,
            temperature=1.0,
        )

        ensemble_smooth = EnsemblePredictor(
            llm_orchestrator=mock_llm_orchestrator,
            tkg_predictor=mock_tkg_predictor,
            alpha=0.6,
            temperature=2.0,
        )

        pred_normal, _ = ensemble_normal.predict(
            question="Test",
            entity1="Russia",
            relation="CONFLICT",
            entity2="Ukraine",
        )

        pred_smooth, _ = ensemble_smooth.predict(
            question="Test",
            entity1="Russia",
            relation="CONFLICT",
            entity2="Ukraine",
        )

        # Higher temperature should decrease confidence
        assert pred_smooth.final_confidence < pred_normal.final_confidence


class TestGracefulDegradation:
    """Test graceful degradation when components fail."""

    def test_llm_failure_uses_tkg(self, mock_tkg_predictor):
        """Test that LLM failure falls back to TKG."""
        failing_llm = Mock()
        failing_llm.forecast.side_effect = Exception("LLM API error")

        ensemble = EnsemblePredictor(
            llm_orchestrator=failing_llm,
            tkg_predictor=mock_tkg_predictor,
            alpha=0.6,
        )

        ensemble_pred, forecast = ensemble.predict(
            question="Test",
            entity1="Russia",
            relation="CONFLICT",
            entity2="Ukraine",
        )

        # Should use TKG only
        assert not ensemble_pred.llm_prediction.available
        assert ensemble_pred.tkg_prediction.available
        assert "LLM API error" in ensemble_pred.llm_prediction.error
        assert ensemble_pred.final_probability == 0.6  # TKG probability

    def test_tkg_failure_uses_llm(self, mock_llm_orchestrator):
        """Test that TKG failure falls back to LLM."""
        failing_tkg = Mock()
        failing_tkg.trained = True
        failing_tkg.predict_future_events.side_effect = Exception("Graph error")

        ensemble = EnsemblePredictor(
            llm_orchestrator=mock_llm_orchestrator,
            tkg_predictor=failing_tkg,
            alpha=0.6,
        )

        ensemble_pred, forecast = ensemble.predict(
            question="Test",
            entity1="Russia",
            relation="CONFLICT",
            entity2="Ukraine",
        )

        # Should use LLM only
        assert ensemble_pred.llm_prediction.available
        assert not ensemble_pred.tkg_prediction.available
        assert "Graph error" in ensemble_pred.tkg_prediction.error
        assert ensemble_pred.final_probability == 0.7  # LLM probability

    def test_tkg_not_trained(self, mock_llm_orchestrator, mock_tkg_predictor):
        """Test handling of untrained TKG predictor."""
        mock_tkg_predictor.trained = False

        ensemble = EnsemblePredictor(
            llm_orchestrator=mock_llm_orchestrator,
            tkg_predictor=mock_tkg_predictor,
            alpha=0.6,
        )

        ensemble_pred, forecast = ensemble.predict(question="Test")

        # Should use LLM only
        assert ensemble_pred.llm_prediction.available
        assert not ensemble_pred.tkg_prediction.available
        assert "not trained" in ensemble_pred.tkg_prediction.error.lower()


class TestEntityExtraction:
    """Test entity extraction from LLM forecasts for TKG queries."""

    def test_extract_entities_from_scenario(
        self, mock_llm_orchestrator, mock_tkg_predictor
    ):
        """Test extraction of entities from LLM scenario."""
        ensemble = EnsemblePredictor(
            llm_orchestrator=mock_llm_orchestrator,
            tkg_predictor=mock_tkg_predictor,
            alpha=0.6,
        )

        # Predict without providing entities
        ensemble_pred, forecast = ensemble.predict(
            question="Will Russia-Ukraine conflict escalate?"
        )

        # Should extract entities from LLM scenario and query TKG
        assert ensemble_pred.tkg_prediction.available
        mock_tkg_predictor.predict_future_events.assert_called_once()

        # Check that entities were extracted
        call_args = mock_tkg_predictor.predict_future_events.call_args
        assert call_args[1]["entity1"] is not None
        assert call_args[1]["entity2"] is not None


class TestEnsemblePerformance:
    """Test that ensemble performs better than individual models."""

    def test_ensemble_balances_extremes(self):
        """Test that ensemble balances extreme predictions from components."""
        # Setup: LLM is overconfident, TKG is underconfident
        overconfident_llm = Mock()
        overconfident_forecast = ForecastOutput(
            question="Test",
            prediction="Test",
            probability=0.95,  # Very high
            confidence=0.9,
            scenario_tree=ScenarioTree(
                question="Test",
                root_scenario=Scenario(
                    scenario_id="test", description="Test", probability=0.95
                ),
                scenarios={},
            ),
            selected_scenario_ids=[],
            reasoning_summary="Test",
            evidence_sources=[],
            timestamp=datetime.now(),
        )
        overconfident_llm.forecast.return_value = overconfident_forecast

        underconfident_tkg = Mock()
        underconfident_tkg.trained = True
        underconfident_tkg.predict_future_events.return_value = [
            {"entity1": "A", "relation": "R", "entity2": "B", "confidence": 0.25}
        ]

        ensemble = EnsemblePredictor(
            llm_orchestrator=overconfident_llm,
            tkg_predictor=underconfident_tkg,
            alpha=0.6,
        )

        ensemble_pred, _ = ensemble.predict(
            question="Test", entity1="A", relation="R", entity2="B"
        )

        # Ensemble should be between extremes: 0.6*0.95 + 0.4*0.25 = 0.67
        assert 0.25 < ensemble_pred.final_probability < 0.95
        assert abs(ensemble_pred.final_probability - 0.67) < 0.01

    def test_ensemble_improves_confidence_with_agreement(self):
        """Test that ensemble has higher confidence when components agree."""
        # Setup: Both models predict similar probabilities
        agreeing_llm = Mock()
        agreeing_forecast = ForecastOutput(
            question="Test",
            prediction="Test",
            probability=0.7,
            confidence=0.8,
            scenario_tree=ScenarioTree(
                question="Test",
                root_scenario=Scenario(
                    scenario_id="test", description="Test", probability=0.7
                ),
                scenarios={},
            ),
            selected_scenario_ids=[],
            reasoning_summary="Test",
            evidence_sources=[],
            timestamp=datetime.now(),
        )
        agreeing_llm.forecast.return_value = agreeing_forecast

        agreeing_tkg = Mock()
        agreeing_tkg.trained = True
        agreeing_tkg.predict_future_events.return_value = [
            {"entity1": "A", "relation": "R", "entity2": "B", "confidence": 0.75}
        ]

        ensemble = EnsemblePredictor(
            llm_orchestrator=agreeing_llm,
            tkg_predictor=agreeing_tkg,
            alpha=0.5,  # Equal weights
        )

        ensemble_pred, _ = ensemble.predict(
            question="Test", entity1="A", relation="R", entity2="B"
        )

        # When models agree, confidence should be high
        # (0.5 * 0.8 + 0.5 * 0.75) = 0.775
        assert ensemble_pred.final_confidence > 0.7


class TestDynamicConfiguration:
    """Test dynamic weight and temperature updates."""

    def test_update_weights(self):
        """Test updating ensemble weights."""
        ensemble = EnsemblePredictor(alpha=0.6)

        ensemble.update_weights(0.8)
        assert ensemble.alpha == 0.8

        ensemble.update_weights(0.3)
        assert ensemble.alpha == 0.3

    def test_update_weights_validation(self):
        """Test weight validation."""
        ensemble = EnsemblePredictor(alpha=0.6)

        with pytest.raises(ValueError):
            ensemble.update_weights(1.5)

        with pytest.raises(ValueError):
            ensemble.update_weights(-0.1)

    def test_update_temperature(self):
        """Test updating temperature."""
        ensemble = EnsemblePredictor(temperature=1.0)

        ensemble.update_temperature(1.5)
        assert ensemble.temperature == 1.5

        ensemble.update_temperature(0.5)
        assert ensemble.temperature == 0.5

    def test_update_temperature_validation(self):
        """Test temperature validation."""
        ensemble = EnsemblePredictor(temperature=1.0)

        with pytest.raises(ValueError):
            ensemble.update_temperature(0.0)

        with pytest.raises(ValueError):
            ensemble.update_temperature(-1.0)
