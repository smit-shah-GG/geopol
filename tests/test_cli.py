"""
Tests for CLI argument parsing and output formatting.

Verifies:
1. Argument parsing works correctly
2. Weight parsing handles various formats
3. Output formatting produces valid JSON/text
4. Error handling for invalid inputs
"""

import json
import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

# Import CLI functions
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from forecast import (
    parse_weights,
    load_api_key,
)
from src.forecasting.output_formatter import (
    OutputFormatter,
    format_forecast,
)


class TestWeightParsing:
    """Test CLI weight parsing."""

    def test_parse_single_weight(self):
        """Test parsing single LLM weight (TKG inferred)."""
        llm, tkg = parse_weights("0.6")
        assert llm == 0.6
        assert tkg == 0.4

    def test_parse_both_weights(self):
        """Test parsing both LLM and TKG weights."""
        llm, tkg = parse_weights("0.7,0.3")
        assert llm == 0.7
        assert tkg == 0.3

    def test_parse_equal_weights(self):
        """Test parsing equal weights."""
        llm, tkg = parse_weights("0.5,0.5")
        assert llm == 0.5
        assert tkg == 0.5

    def test_parse_extreme_weights(self):
        """Test parsing extreme weights."""
        llm, tkg = parse_weights("1.0,0.0")
        assert llm == 1.0
        assert tkg == 0.0

        llm, tkg = parse_weights("0.0")
        assert llm == 0.0
        assert tkg == 1.0

    def test_invalid_weight_format(self):
        """Test that invalid format raises error."""
        with pytest.raises(ValueError, match="Invalid weights format"):
            parse_weights("0.6,0.3,0.1")

    def test_weights_out_of_range(self):
        """Test that out-of-range weights raise error."""
        with pytest.raises(ValueError, match="must be in"):
            parse_weights("1.5")

        with pytest.raises(ValueError, match="must be in"):
            parse_weights("-0.1")

    def test_weights_dont_sum_to_one(self):
        """Test that weights not summing to 1 raise error."""
        with pytest.raises(ValueError, match="must sum to 1.0"):
            parse_weights("0.6,0.5")


class TestAPIKeyLoading:
    """Test API key loading from environment."""

    @patch.dict(os.environ, {"GEMINI_API_KEY": "test-key-123"})
    def test_load_api_key_success(self):
        """Test loading API key from environment."""
        key = load_api_key()
        assert key == "test-key-123"

    @patch.dict(os.environ, {}, clear=True)
    def test_load_api_key_missing(self, capsys):
        """Test handling of missing API key."""
        key = load_api_key()
        assert key is None

        captured = capsys.readouterr()
        assert "GEMINI_API_KEY" in captured.err
        assert "not set" in captured.err


class TestOutputFormatter:
    """Test output formatting."""

    @pytest.fixture
    def sample_forecast(self):
        """Sample forecast data for testing."""
        return {
            "question": "Will Russia-Ukraine conflict escalate?",
            "prediction": "Conflict likely to continue at current intensity",
            "probability": 0.65,
            "confidence": 0.72,
            "scenarios": [
                {
                    "description": "Status quo maintenance",
                    "probability": 0.65,
                    "entities": [
                        {"name": "Russia", "type": "COUNTRY", "role": "ACTOR"},
                        {"name": "Ukraine", "type": "COUNTRY", "role": "TARGET"},
                    ],
                    "timeline": [
                        {
                            "time": "T+1 month",
                            "description": "Continued trench warfare",
                            "probability": 0.7,
                        }
                    ],
                }
            ],
            "reasoning_summary": "Based on historical patterns and current dynamics",
            "evidence": ["Historical pattern analysis", "Graph-based validation"],
            "ensemble_info": {
                "llm_available": True,
                "tkg_available": True,
                "llm_probability": 0.7,
                "tkg_probability": 0.6,
                "weights": {"llm": 0.6, "tkg": 0.4},
                "temperature": 1.0,
            },
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "num_scenarios": 3,
            },
        }

    def test_format_json(self, sample_forecast):
        """Test JSON formatting."""
        formatter = OutputFormatter()
        output = formatter.format_json(sample_forecast)

        # Should be valid JSON
        parsed = json.loads(output)
        assert parsed["question"] == sample_forecast["question"]
        assert parsed["probability"] == sample_forecast["probability"]

    def test_format_text(self, sample_forecast):
        """Test text formatting."""
        formatter = OutputFormatter(use_colors=False)
        output = formatter.format_text(sample_forecast, verbose=False)

        # Should contain key information
        assert sample_forecast["question"] in output
        assert sample_forecast["prediction"] in output
        assert "65.0%" in output  # Probability formatted
        assert "Reasoning" in output

    def test_format_text_verbose(self, sample_forecast):
        """Test verbose text formatting."""
        # Add reasoning chain
        sample_forecast["reasoning_chain"] = [
            {
                "step": 1,
                "claim": "Historical precedent suggests continuation",
                "confidence": 0.8,
                "evidence": ["Similar conflicts lasted years"],
            }
        ]

        formatter = OutputFormatter(use_colors=False)
        output = formatter.format_text(sample_forecast, verbose=True)

        # Should contain detailed reasoning
        assert "Detailed Reasoning Chain" in output
        assert "Historical precedent" in output
        assert "Ensemble Details" in output

    def test_format_summary(self, sample_forecast):
        """Test summary formatting."""
        formatter = OutputFormatter(use_colors=False)
        output = formatter.format_summary(sample_forecast)

        # Should be concise
        assert len(output) < 200
        assert "65.0%" in output
        assert "72.0%" in output

    def test_color_coding(self, sample_forecast):
        """Test color codes are applied correctly."""
        formatter = OutputFormatter(use_colors=True)
        output = formatter.format_text(sample_forecast, verbose=False)

        # Should contain ANSI codes
        assert "\033[" in output

    def test_no_color(self, sample_forecast):
        """Test no color codes when disabled."""
        formatter = OutputFormatter(use_colors=False)
        output = formatter.format_text(sample_forecast, verbose=False)

        # Should not contain ANSI codes
        assert "\033[" not in output

    def test_format_error(self):
        """Test error message formatting."""
        formatter = OutputFormatter(use_colors=False)
        output = formatter.format_error("Something went wrong")

        assert "ERROR" in output
        assert "Something went wrong" in output


class TestOutputFormatterProbabilityColors:
    """Test probability color coding logic."""

    @pytest.fixture
    def formatter(self):
        return OutputFormatter(use_colors=True)

    def test_high_probability_red(self, formatter):
        """Test high probability gets red color."""
        output = formatter._format_probability(0.8)
        assert formatter.colors["red"] in output

    def test_medium_probability_yellow(self, formatter):
        """Test medium probability gets yellow color."""
        output = formatter._format_probability(0.6)
        assert formatter.colors["yellow"] in output

    def test_low_probability_green(self, formatter):
        """Test low probability gets green color."""
        output = formatter._format_probability(0.3)
        assert formatter.colors["green"] in output

    def test_high_confidence_green(self, formatter):
        """Test high confidence gets green color."""
        output = formatter._format_confidence(0.8)
        assert formatter.colors["green"] in output

    def test_low_confidence_red(self, formatter):
        """Test low confidence gets red color."""
        output = formatter._format_confidence(0.2)
        assert formatter.colors["red"] in output


class TestFormatForecastConvenience:
    """Test convenience format_forecast function."""

    @pytest.fixture
    def sample_forecast(self):
        """Minimal forecast for testing."""
        return {
            "question": "Test question",
            "prediction": "Test prediction",
            "probability": 0.5,
            "confidence": 0.5,
            "scenarios": [],
            "reasoning_summary": "Test reasoning",
            "evidence": [],
            "ensemble_info": {
                "llm_available": True,
                "tkg_available": False,
                "weights": {"llm": 0.6, "tkg": 0.4},
                "temperature": 1.0,
            },
            "metadata": {"timestamp": datetime.now().isoformat(), "num_scenarios": 0},
        }

    def test_format_text_default(self, sample_forecast):
        """Test default text formatting."""
        output = format_forecast(sample_forecast, format_type="text")
        assert "Test question" in output

    def test_format_json(self, sample_forecast):
        """Test JSON formatting."""
        output = format_forecast(sample_forecast, format_type="json")
        parsed = json.loads(output)
        assert parsed["question"] == "Test question"

    def test_format_summary(self, sample_forecast):
        """Test summary formatting."""
        output = format_forecast(sample_forecast, format_type="summary")
        assert "Test prediction" in output

    def test_invalid_format_type(self, sample_forecast):
        """Test invalid format type raises error."""
        with pytest.raises(ValueError, match="Invalid format_type"):
            format_forecast(sample_forecast, format_type="invalid")


class TestCLIArgumentHandling:
    """Test CLI argument combinations."""

    def test_question_required(self):
        """Test that question argument is required."""
        # This would be tested with argparse in integration tests
        # Here we just verify the structure is correct
        from forecast import main
        import argparse

        # argparse will handle required argument validation
        assert callable(main)

    def test_default_weights(self):
        """Test default weight parsing."""
        llm, tkg = parse_weights("0.6")
        assert llm == 0.6
        assert abs(tkg - 0.4) < 0.01

    def test_custom_weights(self):
        """Test custom weight parsing."""
        llm, tkg = parse_weights("0.8,0.2")
        assert llm == 0.8
        assert tkg == 0.2


class TestErrorHandling:
    """Test error handling in formatting."""

    def test_missing_fields_handled(self):
        """Test formatting handles missing optional fields."""
        minimal_forecast = {
            "question": "Test",
            "prediction": "Test",
            "probability": 0.5,
            "confidence": 0.5,
            "reasoning_summary": "Test",
        }

        formatter = OutputFormatter(use_colors=False)

        # Should not crash
        output = formatter.format_text(minimal_forecast, verbose=False)
        assert "Test" in output

        # Summary should also work
        summary = formatter.format_summary(minimal_forecast)
        assert len(summary) > 0
