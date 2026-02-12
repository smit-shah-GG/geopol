#!/usr/bin/env python
"""
Preflight system readiness validator.

Checks all components required for forecasting and reports pass/fail status.
Does NOT import any src/ modules — must work even if src/ is broken.

Usage:
    uv run python scripts/preflight.py [--quiet]
"""

from __future__ import annotations

import argparse
import glob
import os
import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Validate system readiness for geopolitical forecasting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress PASS lines (only show FAIL/SKIP)",
    )
    return parser.parse_args()


def print_pass(msg: str, quiet: bool) -> None:
    """Print a passing check line."""
    if not quiet:
        print(f"[PASS] {msg}")


def print_fail(msg: str, fix: str) -> None:
    """Print a failing check line with fix guidance."""
    print(f"[FAIL] {msg}")
    print(f"       Fix: {fix}")


def print_skip(msg: str) -> None:
    """Print a skipped (optional) check line."""
    print(f"[SKIP] {msg}")


def check_imports(quiet: bool) -> int:
    """Check required Python package imports. Returns number of failures."""
    packages = {
        "dotenv": "python-dotenv",
        "chromadb": "chromadb",
        "google.generativeai": "google-generativeai",
        "networkx": "networkx",
        "pandas": "pandas",
        "llama_index": "llama-index",
    }

    failures = 0
    for import_name, install_name in packages.items():
        try:
            __import__(import_name)
            print_pass(f"Import {import_name}", quiet)
        except ImportError:
            print_fail(
                f"Import {import_name}",
                f"uv add {install_name}",
            )
            failures += 1

    return failures


def check_environment(quiet: bool) -> int:
    """Check .env file and API key. Returns number of failures."""
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        print_fail(
            "Environment file (.env)",
            "cp .env.example .env && edit .env with your API keys",
        )
        return 1

    # Load dotenv if available (best-effort; we already checked import above)
    try:
        from dotenv import load_dotenv

        load_dotenv(env_path)
    except ImportError:
        pass

    api_key = os.environ.get("GEMINI_API_KEY", "")
    placeholders = {"", "your-key-here", "PLACEHOLDER", "your_key_here"}

    if api_key.strip().lower() in {p.lower() for p in placeholders}:
        print_fail(
            "GEMINI_API_KEY",
            "Set GEMINI_API_KEY in .env (get one at https://aistudio.google.com/apikey)",
        )
        return 1

    masked = api_key[:8] + "****" if len(api_key) > 8 else api_key[:4] + "****"
    print_pass(f"GEMINI_API_KEY ({masked})", quiet)
    return 0


def check_event_database(quiet: bool) -> int:
    """Check events.db exists and has data. Returns number of failures."""
    db_path = PROJECT_ROOT / "data" / "events.db"
    if not db_path.exists():
        print_fail(
            "Event database (data/events.db)",
            "uv run python scripts/bootstrap.py",
        )
        return 1

    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.execute("SELECT COUNT(*) FROM events")
        count = cursor.fetchone()[0]
        conn.close()
        print_pass(f"Event database ({count:,} events)", quiet)
        return 0
    except sqlite3.Error as e:
        print_fail(
            f"Event database (query failed: {e})",
            "uv run python scripts/bootstrap.py --force-stage process",
        )
        return 1


def check_knowledge_graphs(quiet: bool) -> int:
    """Check graph files exist. Returns number of failures."""
    graphs_dir = PROJECT_ROOT / "data" / "graphs"
    if not graphs_dir.exists():
        print_fail(
            "Knowledge graphs (data/graphs/)",
            "uv run python scripts/bootstrap.py --force-stage graph",
        )
        return 1

    graphml_files = glob.glob(str(graphs_dir / "*.graphml"))
    if not graphml_files:
        print_fail(
            "Knowledge graphs (no .graphml files)",
            "uv run python scripts/bootstrap.py --force-stage persist",
        )
        return 1

    print_pass(f"Knowledge graphs ({len(graphml_files)} .graphml files)", quiet)
    return 0


def check_rag_store(quiet: bool) -> int:
    """Check ChromaDB vector store exists. Returns number of failures."""
    chroma_dir = PROJECT_ROOT / "chroma_db"
    if not chroma_dir.exists():
        print_fail(
            "RAG vector store (chroma_db/)",
            "uv run python scripts/bootstrap.py --force-stage index",
        )
        return 1

    # Check non-empty (has subdirectories or files)
    contents = list(chroma_dir.iterdir())
    if not contents:
        print_fail(
            "RAG vector store (chroma_db/ is empty)",
            "uv run python scripts/bootstrap.py --force-stage index",
        )
        return 1

    print_pass(f"RAG vector store ({len(contents)} entries)", quiet)
    return 0


def check_tkg_model(quiet: bool) -> int:
    """Check TKG model checkpoint. Returns 0 (always optional)."""
    model_path = PROJECT_ROOT / "models" / "tkg" / "regcn_trained.pt"
    if not model_path.exists():
        print_skip(
            "TKG model not found — forecasting works without it (LLM-only mode)"
        )
    else:
        print_pass("TKG model (models/tkg/regcn_trained.pt)", quiet)
    # TKG is optional, never counts as a failure
    return 0


def main() -> int:
    """Run all preflight checks and report summary."""
    args = parse_args()
    quiet: bool = args.quiet

    print()
    print("=" * 50)
    print("PREFLIGHT SYSTEM CHECK")
    print("=" * 50)
    print()

    total_checks = 0
    total_failures = 0

    # 1. Python imports (6 checks)
    print("--- Python imports ---")
    import_failures = check_imports(quiet)
    total_checks += 6
    total_failures += import_failures
    print()

    # 2. Environment (1 check)
    print("--- Environment ---")
    env_failures = check_environment(quiet)
    total_checks += 1
    total_failures += env_failures
    print()

    # 3. Event database (1 check)
    print("--- Event database ---")
    db_failures = check_event_database(quiet)
    total_checks += 1
    total_failures += db_failures
    print()

    # 4. Knowledge graphs (1 check)
    print("--- Knowledge graphs ---")
    graph_failures = check_knowledge_graphs(quiet)
    total_checks += 1
    total_failures += graph_failures
    print()

    # 5. RAG vector store (1 check)
    print("--- RAG vector store ---")
    rag_failures = check_rag_store(quiet)
    total_checks += 1
    total_failures += rag_failures
    print()

    # 6. TKG model (optional, 1 check)
    print("--- TKG model (optional) ---")
    check_tkg_model(quiet)
    total_checks += 1
    # TKG failures not counted (optional)
    print()

    # Summary
    passed = total_checks - total_failures
    print("=" * 50)
    print(f"{passed}/{total_checks} checks passed")

    if total_failures > 0:
        print()
        print(
            "Run `uv run python scripts/bootstrap.py` to initialize the system."
        )
        print("=" * 50)
        return 1

    print("System is ready for forecasting.")
    print("=" * 50)
    return 0


if __name__ == "__main__":
    sys.exit(main())
