#!/usr/bin/env python3
"""
Command-line interface for geopolitical forecasting.

Usage:
    python forecast.py "Will Russia-Ukraine conflict escalate?"
    python forecast.py "Will China-Taiwan tensions escalate?" --verbose
    python forecast.py "Will Iran develop nuclear weapons?" --output-format json

Requirements:
    - GEMINI_API_KEY environment variable (or .env file with GEMINI_API_KEY)
    - Optional: Pre-trained TKG model at checkpoints/tkg/
    - Optional: RAG index at data/rag_index/
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from src.forecasting.forecast_engine import ForecastEngine
from src.forecasting.output_formatter import format_forecast

# Load environment variables from .env file if it exists
load_dotenv()


def setup_logging(verbose: bool = False) -> None:
    """
    Configure logging for CLI.

    Args:
        verbose: Whether to enable debug logging
    """
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(levelname)s: %(message)s",
    )


def load_api_key() -> Optional[str]:
    """
    Load Gemini API key from environment.

    Returns:
        API key string or None if not found
    """
    api_key = os.environ.get("GEMINI_API_KEY")

    if not api_key:
        print(
            "ERROR: GEMINI_API_KEY environment variable not set.",
            file=sys.stderr,
        )
        print("\nTo set your API key:", file=sys.stderr)
        print("  export GEMINI_API_KEY='your-api-key-here'", file=sys.stderr)
        print(
            "\nGet an API key at: https://aistudio.google.com/apikey",
            file=sys.stderr,
        )
        return None

    return api_key


def parse_weights(weights_str: str) -> tuple[float, float]:
    """
    Parse ensemble weights from string.

    Args:
        weights_str: Weights in format "0.6,0.4" or "0.6" (TKG inferred)

    Returns:
        Tuple of (llm_weight, tkg_weight)

    Raises:
        ValueError: If weights are invalid
    """
    parts = weights_str.split(",")

    if len(parts) == 1:
        # Only LLM weight provided, infer TKG weight
        llm_weight = float(parts[0])
        tkg_weight = 1.0 - llm_weight
    elif len(parts) == 2:
        # Both weights provided
        llm_weight = float(parts[0])
        tkg_weight = float(parts[1])
    else:
        raise ValueError(
            f"Invalid weights format: {weights_str}. "
            "Expected: '0.6' or '0.6,0.4'"
        )

    # Validate
    if not (0 <= llm_weight <= 1 and 0 <= tkg_weight <= 1):
        raise ValueError("Weights must be in [0, 1]")

    # Check sum (allow small tolerance)
    if abs(llm_weight + tkg_weight - 1.0) > 0.01:
        raise ValueError(f"Weights must sum to 1.0, got {llm_weight + tkg_weight}")

    return llm_weight, tkg_weight


def initialize_engine(
    api_key: str,
    alpha: float,
    temperature: float,
    no_cache: bool,
    tkg_path: Optional[Path],
) -> ForecastEngine:
    """
    Initialize forecast engine with components.

    Args:
        api_key: Gemini API key
        alpha: LLM weight for ensemble
        temperature: Temperature for confidence calibration
        no_cache: Whether to disable RAG cache
        tkg_path: Path to pre-trained TKG model (optional)

    Returns:
        Initialized ForecastEngine

    Raises:
        Exception: If initialization fails
    """
    try:
        # Initialize engine
        engine = ForecastEngine(
            api_key=api_key,
            alpha=alpha,
            temperature=temperature,
            enable_rag=not no_cache,  # Disable RAG if no_cache=True
            enable_tkg=tkg_path is not None,  # Enable TKG if path provided
        )

        # Load pre-trained TKG if available
        if tkg_path and tkg_path.exists():
            print(f"Loading TKG model from {tkg_path}...", file=sys.stderr)
            engine.load_tkg(tkg_path)
            print("TKG model loaded successfully.", file=sys.stderr)
        elif tkg_path:
            print(
                f"WARNING: TKG path {tkg_path} not found. "
                "Proceeding with LLM-only forecasting.",
                file=sys.stderr,
            )

        return engine

    except Exception as e:
        print(f"ERROR: Failed to initialize engine: {e}", file=sys.stderr)
        raise


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Geopolitical forecasting with hybrid LLM-TKG system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic forecast
  python forecast.py "Will Russia-Ukraine conflict escalate?"

  # Detailed output
  python forecast.py "Will China invade Taiwan?" --verbose

  # JSON output
  python forecast.py "Will Iran develop nukes?" --output-format json

  # Custom ensemble weights (70% LLM, 30% TKG)
  python forecast.py "Will NATO expand?" --weights 0.7,0.3

  # Disable RAG cache
  python forecast.py "Will conflict occur?" --no-cache

Environment Variables:
  GEMINI_API_KEY    Gemini API key (required)
                    Can be set via export or .env file
        """,
    )

    # Required arguments
    parser.add_argument(
        "question",
        type=str,
        help="Forecasting question (natural language)",
    )

    # Output options
    parser.add_argument(
        "--output-format",
        choices=["text", "json", "summary"],
        default="text",
        help="Output format (default: text)",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed reasoning steps and debug info",
    )

    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable colored output",
    )

    # Engine options
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Bypass RAG cache and disable historical grounding",
    )

    parser.add_argument(
        "--weights",
        type=str,
        default="0.6",
        metavar="LLM[,TKG]",
        help="Ensemble weights (default: 0.6 for LLM, 0.4 for TKG)",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        metavar="T",
        help="Temperature for confidence calibration (default: 1.0)",
    )

    parser.add_argument(
        "--tkg-path",
        type=str,
        default="checkpoints/tkg",
        metavar="PATH",
        help="Path to pre-trained TKG model (default: checkpoints/tkg)",
    )

    # Parse arguments
    args = parser.parse_args()

    # Setup logging
    setup_logging(verbose=args.verbose)

    try:
        # Load API key
        api_key = load_api_key()
        if not api_key:
            return 1

        # Parse weights
        llm_weight, tkg_weight = parse_weights(args.weights)
        if args.verbose:
            print(
                f"Ensemble weights: LLM={llm_weight:.2f}, TKG={tkg_weight:.2f}",
                file=sys.stderr,
            )

        # Initialize engine
        tkg_path = Path(args.tkg_path) if args.tkg_path else None
        engine = initialize_engine(
            api_key=api_key,
            alpha=llm_weight,
            temperature=args.temperature,
            no_cache=args.no_cache,
            tkg_path=tkg_path,
        )

        # Display engine status if verbose
        if args.verbose:
            status = engine.get_engine_status()
            print("\nEngine status:", file=sys.stderr)
            print(f"  Gemini client: {'OK' if status['gemini_client']['initialized'] else 'FAILED'}", file=sys.stderr)
            print(f"  RAG pipeline: {'Enabled' if status['rag_pipeline']['enabled'] else 'Disabled'}", file=sys.stderr)
            print(f"  TKG predictor: {'Trained' if status['tkg_predictor']['trained'] else 'Not trained'}", file=sys.stderr)
            print(f"  Ensemble: α={status['ensemble']['alpha']:.2f}, T={status['ensemble']['temperature']:.2f}", file=sys.stderr)
            print("", file=sys.stderr)

        # Generate forecast
        if args.verbose:
            print(f"Forecasting: {args.question}", file=sys.stderr)
            print("", file=sys.stderr)

        forecast = engine.forecast(
            question=args.question,
            verbose=args.verbose,
            use_cache=not args.no_cache,
        )

        # Format output
        output = format_forecast(
            forecast=forecast,
            format_type=args.output_format,
            verbose=args.verbose,
            use_colors=not args.no_color,
        )

        # Print to stdout
        print(output)

        return 0

    except KeyboardInterrupt:
        print("\n\nInterrupted by user.", file=sys.stderr)
        return 130

    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
