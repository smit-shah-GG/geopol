"""
Output formatter for forecast results.

Provides formatting utilities for forecast outputs:
1. JSON formatting (machine-readable)
2. Text formatting (human-readable)
3. Detailed vs. summary modes
4. Color-coded terminal output (optional)
"""

import json
from datetime import datetime
from typing import Any, Dict, Optional


class OutputFormatter:
    """
    Formats forecast outputs for display.

    Supports multiple output formats:
    - JSON: Machine-readable structured output
    - Text: Human-readable formatted text
    - Summary: Condensed overview
    - Detailed: Full reasoning chains and evidence

    Attributes:
        use_colors: Whether to use ANSI color codes (for terminal)
    """

    def __init__(self, use_colors: bool = True):
        """
        Initialize formatter.

        Args:
            use_colors: Whether to use ANSI color codes in text output
        """
        self.use_colors = use_colors

        # ANSI color codes
        self.colors = {
            "reset": "\033[0m" if use_colors else "",
            "bold": "\033[1m" if use_colors else "",
            "red": "\033[91m" if use_colors else "",
            "green": "\033[92m" if use_colors else "",
            "yellow": "\033[93m" if use_colors else "",
            "blue": "\033[94m" if use_colors else "",
            "magenta": "\033[95m" if use_colors else "",
            "cyan": "\033[96m" if use_colors else "",
        }

    def format_json(self, forecast: Dict[str, Any], indent: int = 2) -> str:
        """
        Format forecast as JSON.

        Args:
            forecast: Forecast dictionary from ForecastEngine
            indent: JSON indentation level

        Returns:
            JSON string
        """
        return json.dumps(forecast, indent=indent, default=str)

    def format_text(
        self, forecast: Dict[str, Any], verbose: bool = False
    ) -> str:
        """
        Format forecast as human-readable text.

        Args:
            forecast: Forecast dictionary from ForecastEngine
            verbose: Whether to include detailed reasoning

        Returns:
            Formatted text string
        """
        lines = []

        # Header
        lines.append(self._format_header("GEOPOLITICAL FORECAST"))
        lines.append("")

        # Question
        lines.append(self._format_section("Question"))
        lines.append(forecast["question"])
        lines.append("")

        # Main prediction
        lines.append(self._format_section("Prediction"))
        prob = forecast["probability"]
        conf = forecast["confidence"]

        # Probability with color coding
        prob_str = self._format_probability(prob)
        conf_str = self._format_confidence(conf)

        lines.append(f"{forecast['prediction']}")
        lines.append(f"Probability: {prob_str}")
        lines.append(f"Confidence: {conf_str}")
        lines.append("")

        # Top scenarios
        if forecast.get("scenarios"):
            lines.append(self._format_section("Top Scenarios"))
            for i, scenario in enumerate(forecast["scenarios"][:3], 1):
                lines.append(
                    f"{i}. {scenario['description']} "
                    f"({self._format_probability(scenario['probability'])})"
                )

                # Entities
                if scenario.get("entities"):
                    entity_names = [e["name"] for e in scenario["entities"][:3]]
                    lines.append(f"   Actors: {', '.join(entity_names)}")

                # Timeline preview
                if scenario.get("timeline"):
                    first_event = scenario["timeline"][0]
                    lines.append(
                        f"   First event: {first_event['description']} "
                        f"({first_event['time']})"
                    )

                lines.append("")

        # Reasoning summary
        lines.append(self._format_section("Reasoning"))
        lines.append(forecast["reasoning_summary"])
        lines.append("")

        # Detailed reasoning (if verbose)
        if verbose and forecast.get("reasoning_chain"):
            lines.append(self._format_section("Detailed Reasoning Chain"))
            for step in forecast["reasoning_chain"]:
                lines.append(
                    f"Step {step['step']}: {step['claim']} "
                    f"(confidence: {self._format_confidence(step['confidence'])})"
                )
                if step.get("evidence"):
                    for evidence in step["evidence"][:2]:
                        lines.append(f"  - {evidence}")
            lines.append("")

        # Evidence sources
        if forecast.get("evidence"):
            lines.append(self._format_section("Evidence Sources"))
            for source in forecast["evidence"][:5]:
                lines.append(f"- {source}")
            lines.append("")

        # Ensemble information (if verbose)
        if verbose and forecast.get("ensemble_info"):
            lines.append(self._format_section("Ensemble Details"))
            ensemble = forecast["ensemble_info"]

            components = []
            if ensemble["llm_available"]:
                components.append(
                    f"LLM (weight={ensemble['weights']['llm']:.2f}, "
                    f"P={ensemble['llm_probability']:.3f})"
                )
            else:
                components.append("LLM (unavailable)")

            if ensemble["tkg_available"]:
                components.append(
                    f"TKG (weight={ensemble['weights']['tkg']:.2f}, "
                    f"P={ensemble['tkg_probability']:.3f})"
                )
            else:
                components.append("TKG (unavailable)")

            lines.append(f"Components: {' + '.join(components)}")
            lines.append(f"Temperature: {ensemble['temperature']:.2f}")
            lines.append("")

        # Metadata
        metadata = forecast.get("metadata", {})
        timestamp = metadata.get("timestamp", datetime.now().isoformat())
        lines.append(self._format_section("Metadata"))
        lines.append(f"Generated: {timestamp}")
        lines.append(f"Scenarios analyzed: {metadata.get('num_scenarios', 'N/A')}")

        return "\n".join(lines)

    def format_summary(self, forecast: Dict[str, Any]) -> str:
        """
        Format forecast as brief summary.

        Args:
            forecast: Forecast dictionary from ForecastEngine

        Returns:
            Summary string (1-3 lines)
        """
        prob = forecast["probability"]
        conf = forecast["confidence"]

        prob_str = self._format_probability(prob)
        conf_str = self._format_confidence(conf)

        summary = (
            f"Forecast: {forecast['prediction'][:100]}... "
            f"[P={prob_str}, C={conf_str}]"
        )

        return summary

    def _format_header(self, text: str) -> str:
        """Format section header with colors."""
        return f"{self.colors['bold']}{self.colors['cyan']}{text}{self.colors['reset']}"

    def _format_section(self, text: str) -> str:
        """Format section title."""
        return f"{self.colors['bold']}{text}{self.colors['reset']}"

    def _format_probability(self, prob: float) -> str:
        """Format probability with color coding."""
        if prob >= 0.7:
            color = self.colors["red"]  # High probability - red alert
        elif prob >= 0.5:
            color = self.colors["yellow"]  # Medium probability
        else:
            color = self.colors["green"]  # Low probability

        return f"{color}{prob:.1%}{self.colors['reset']}"

    def _format_confidence(self, conf: float) -> str:
        """Format confidence score with color coding."""
        if conf >= 0.7:
            color = self.colors["green"]  # High confidence
        elif conf >= 0.4:
            color = self.colors["yellow"]  # Medium confidence
        else:
            color = self.colors["red"]  # Low confidence

        return f"{color}{conf:.1%}{self.colors['reset']}"

    def format_error(self, error_msg: str) -> str:
        """
        Format error message.

        Args:
            error_msg: Error message

        Returns:
            Formatted error string
        """
        return (
            f"{self.colors['bold']}{self.colors['red']}"
            f"ERROR: {error_msg}"
            f"{self.colors['reset']}"
        )


def format_forecast(
    forecast: Dict[str, Any],
    format_type: str = "text",
    verbose: bool = False,
    use_colors: bool = True,
) -> str:
    """
    Convenience function to format forecast output.

    Args:
        forecast: Forecast dictionary from ForecastEngine
        format_type: Output format ("text", "json", or "summary")
        verbose: Whether to include detailed information
        use_colors: Whether to use ANSI color codes

    Returns:
        Formatted string

    Raises:
        ValueError: If format_type is invalid
    """
    formatter = OutputFormatter(use_colors=use_colors)

    if format_type == "text":
        return formatter.format_text(forecast, verbose=verbose)
    elif format_type == "json":
        return formatter.format_json(forecast)
    elif format_type == "summary":
        return formatter.format_summary(forecast)
    else:
        raise ValueError(
            f"Invalid format_type: {format_type}. "
            "Must be 'text', 'json', or 'summary'"
        )
