#!/usr/bin/env python
"""
End-to-end forecast test.

Tests the full pipeline: RAG retrieval → LLM reasoning → TKG patterns → Ensemble output.
"""

import logging
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load environment variables
from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)

# Reduce noise from third-party libraries
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("chromadb").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)


def main():
    print("=" * 60)
    print("E2E FORECAST TEST")
    print("=" * 60)
    print()

    # Step 1: Load RAG pipeline
    print("[1/4] Loading RAG pipeline...")
    from src.forecasting.rag_pipeline import RAGPipeline

    rag = RAGPipeline(persist_dir="./chroma_db")
    rag_loaded = rag.load_existing_index()
    print(f"      RAG index loaded: {rag_loaded}")
    print()

    # Step 2: Load TKG predictor (optional - may not have trained model)
    print("[2/4] Loading TKG predictor...")
    tkg = None
    try:
        from src.forecasting.tkg_predictor import TKGPredictor

        tkg = TKGPredictor(auto_load=False)
        # Use the pretrained checkpoint from training pipeline
        checkpoint_path = Path("models/tkg/regcn_trained.pt")
        if checkpoint_path.exists():
            tkg.load_pretrained(checkpoint_path)
            print(f"      TKG model loaded: {tkg.trained}")
        else:
            print("      TKG checkpoint not found (will use LLM only)")
            tkg = None
    except Exception as e:
        print(f"      TKG loading failed: {e}")
        print("      Continuing with LLM only...")
        tkg = None
    print()

    # Step 3: Initialize forecast engine
    print("[3/4] Initializing forecast engine...")
    from src.forecasting.forecast_engine import ForecastEngine

    engine = ForecastEngine(
        rag_pipeline=rag if rag_loaded else None,
        tkg_predictor=tkg,
        alpha=0.6,  # 60% LLM, 40% TKG
        enable_rag=rag_loaded,
        enable_tkg=tkg is not None and tkg.trained if hasattr(tkg, "trained") else False,
    )

    status = engine.get_engine_status()
    print(f"      Engine status:")
    print(f"        - RAG enabled: {status['rag_pipeline']['enabled']}")
    print(f"        - RAG has index: {status['rag_pipeline']['has_index']}")
    print(f"        - TKG enabled: {status['tkg_predictor']['enabled']}")
    print(f"        - TKG trained: {status['tkg_predictor']['trained']}")
    print()

    # Step 4: Run forecast
    print("[4/4] Generating forecast...")
    print()

    question = "Will Russia and Ukraine engage in direct diplomatic talks in the next 30 days?"

    print("=" * 60)
    print(f"QUESTION: {question}")
    print("=" * 60)
    print()

    try:
        result = engine.forecast(question, verbose=True)

        print(f"PREDICTION: {result['prediction'][:200]}...")
        print()
        print(f"PROBABILITY: {result['probability']:.1%}")
        print(f"CONFIDENCE:  {result['confidence']:.1%}")
        print()

        print("REASONING SUMMARY:")
        print("-" * 40)
        print(result["reasoning_summary"][:500])
        print()

        print("TOP SCENARIOS:")
        print("-" * 40)
        for i, scenario in enumerate(result["scenarios"][:3], 1):
            desc = scenario["description"][:80] + "..." if len(scenario["description"]) > 80 else scenario["description"]
            print(f"  {i}. {desc}")
            print(f"     Probability: {scenario['probability']:.1%}")
            print()

        print("ENSEMBLE INFO:")
        print("-" * 40)
        ens = result["ensemble_info"]
        print(f"  LLM available: {ens['llm_available']}")
        print(f"  TKG available: {ens['tkg_available']}")
        if ens["llm_probability"] is not None:
            print(f"  LLM probability: {ens['llm_probability']:.3f}")
        if ens["tkg_probability"] is not None:
            print(f"  TKG probability: {ens['tkg_probability']:.3f}")
        print(f"  Weights: LLM={ens['weights']['llm']:.2f}, TKG={ens['weights']['tkg']:.2f}")
        print()

        print("=" * 60)
        print("E2E TEST COMPLETE ✓")
        print("=" * 60)

    except Exception as e:
        logging.exception("Forecast failed")
        print(f"\nERROR: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
