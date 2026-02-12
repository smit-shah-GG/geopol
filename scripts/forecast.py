#!/usr/bin/env python
"""
CLI wrapper for the ForecastEngine.

Generates geopolitical forecasts from natural language questions.

Usage:
    uv run python scripts/forecast.py -q "Will NATO expand in the next 6 months?"
    uv run python scripts/forecast.py -q "..." --verbose
    uv run python scripts/forecast.py -q "..." --json --no-tkg
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load environment variables
from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stderr,
)

# Reduce noise from third-party libraries
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("chromadb").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate geopolitical forecasts using hybrid LLM + TKG engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--question", "-q",
        required=True,
        type=str,
        help="The forecast question (required)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Include detailed reasoning steps in output",
    )
    parser.add_argument(
        "--no-rag",
        action="store_true",
        help="Disable RAG pipeline (no historical context retrieval)",
    )
    parser.add_argument(
        "--no-tkg",
        action="store_true",
        help="Disable TKG predictor (LLM-only mode)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.6,
        help="LLM ensemble weight, 0.0-1.0 (default: 0.6)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Confidence calibration temperature (default: 1.0)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Output raw JSON instead of formatted text",
    )
    return parser.parse_args()


def load_rag(disable: bool) -> Optional[object]:
    """Load RAG pipeline. Returns None if disabled or unavailable."""
    if disable:
        logger.info("RAG disabled via --no-rag")
        return None

    try:
        from src.forecasting.rag_pipeline import RAGPipeline

        rag = RAGPipeline(persist_dir="./chroma_db")
        loaded = rag.load_existing_index()
        if loaded:
            logger.info("RAG pipeline loaded")
            return rag
        logger.warning("RAG index not found — continuing without RAG")
        return None
    except Exception as e:
        print(f"Warning: RAG pipeline unavailable: {e}", file=sys.stderr)
        return None


def load_tkg(disable: bool) -> Optional[object]:
    """Load TKG predictor. Returns None if disabled or unavailable."""
    if disable:
        logger.info("TKG disabled via --no-tkg")
        return None

    checkpoint_path = Path("models/tkg/regcn_trained.pt")
    if not checkpoint_path.exists():
        logger.info("TKG checkpoint not found — LLM-only mode")
        return None

    try:
        from src.forecasting.tkg_predictor import TKGPredictor

        tkg = TKGPredictor(auto_load=False)
        tkg.load_pretrained(checkpoint_path)
        logger.info("TKG predictor loaded")
        return tkg
    except Exception as e:
        print(f"Warning: TKG predictor unavailable: {e}", file=sys.stderr)
        return None


def format_output(result: dict[str, object]) -> str:
    """Format forecast result as human-readable text."""
    lines: list[str] = []

    lines.append(f"QUESTION: {result['question']}")
    lines.append("")
    lines.append(f"PREDICTION: {result['prediction']}")
    lines.append("")
    lines.append(f"PROBABILITY: {result['probability']:.1%}")
    lines.append(f"CONFIDENCE:  {result['confidence']:.1%}")
    lines.append("")

    lines.append("REASONING:")
    reasoning = result.get("reasoning_summary", "N/A")
    lines.append(str(reasoning))
    lines.append("")

    scenarios = result.get("scenarios", [])
    if scenarios:
        lines.append("TOP SCENARIOS:")
        for i, scenario in enumerate(scenarios[:3], 1):
            desc = scenario.get("description", "N/A")
            prob = scenario.get("probability", 0.0)
            lines.append(f"  {i}. {desc} (P={prob:.1%})")
        lines.append("")

    ens = result.get("ensemble_info", {})
    llm_prob = ens.get("llm_probability")
    tkg_prob = ens.get("tkg_probability")
    weights = ens.get("weights", {})

    lines.append("ENSEMBLE:")
    llm_str = f"{llm_prob:.3f}" if llm_prob is not None else "N/A"
    tkg_str = f"{tkg_prob:.3f}" if tkg_prob is not None else "N/A"
    lines.append(f"  LLM: {llm_str}  weight={weights.get('llm', 0.0):.2f}")
    lines.append(f"  TKG: {tkg_str}  weight={weights.get('tkg', 0.0):.2f}")

    return "\n".join(lines)


def main() -> int:
    """Run forecast from CLI arguments."""
    args = parse_args()

    # Load components
    rag = load_rag(args.no_rag)
    tkg = load_tkg(args.no_tkg)

    tkg_trained = False
    if tkg is not None:
        tkg_trained = getattr(tkg, "trained", False)

    # Verify at least the LLM path is viable
    import os

    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key or api_key.strip().lower() in {"your-key-here", "placeholder"}:
        if rag is None and tkg is None:
            print(
                "System not ready. Run `uv run python scripts/preflight.py` to diagnose.",
                file=sys.stderr,
            )
            return 1

    # Initialize engine
    try:
        from src.forecasting.forecast_engine import ForecastEngine

        engine = ForecastEngine(
            rag_pipeline=rag,
            tkg_predictor=tkg,
            alpha=args.alpha,
            temperature=args.temperature,
            enable_rag=rag is not None,
            enable_tkg=tkg_trained,
        )
    except Exception as e:
        print(f"Error: Failed to initialize ForecastEngine: {e}", file=sys.stderr)
        print(
            "Run `uv run python scripts/preflight.py` to diagnose.",
            file=sys.stderr,
        )
        return 1

    # Run forecast
    try:
        result = engine.forecast(args.question, verbose=args.verbose)
    except Exception as e:
        print(f"Error: Forecast failed: {e}", file=sys.stderr)
        return 1

    # Output
    if args.json_output:
        print(json.dumps(result, indent=2, default=str))
    else:
        print(format_output(result))

    return 0


if __name__ == "__main__":
    sys.exit(main())
