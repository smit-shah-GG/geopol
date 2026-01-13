#!/usr/bin/env python
"""Diagnostic script to test where the forecast CLI hangs."""

import os
import sys
import time

print("Testing forecast system diagnostics...")

# Check API key
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("❌ GEMINI_API_KEY not set - this will cause initialization to fail")
    print("   Fix: export GEMINI_API_KEY='your-key-here'")
    sys.exit(1)
else:
    print(f"✓ GEMINI_API_KEY set ({len(api_key)} chars)")

# Test imports
print("\nTesting imports...")
try:
    from src.forecasting.forecast_engine import ForecastEngine
    print("✓ ForecastEngine imported")
except Exception as e:
    print(f"❌ Failed to import ForecastEngine: {e}")
    sys.exit(1)

try:
    from src.calibration import TemperatureScaler
    print("✓ Calibration modules imported")
except Exception as e:
    print(f"❌ Failed to import calibration: {e}")
    sys.exit(1)

# Test initialization
print("\nTesting engine initialization...")
try:
    start = time.time()
    engine = ForecastEngine(
        alpha=0.6,
        temperature=1.0,
        enable_rag=True,
        enable_tkg=True
    )
    elapsed = time.time() - start
    print(f"✓ Engine initialized in {elapsed:.2f}s")
except Exception as e:
    print(f"❌ Failed to initialize engine: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test a simple forecast
print("\nTesting forecast (this may take 30-60s)...")
test_question = "Will Russia escalate military operations in Ukraine in Q1 2024?"
print(f"Query: {test_question}")

try:
    start = time.time()
    print("  Step 1: Starting forecast call...")

    # Add timeout wrapper to detect hangs
    import signal

    def timeout_handler(signum, frame):
        raise TimeoutError("Forecast timed out after 120 seconds")

    # Set 2 minute timeout
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(120)

    try:
        result = engine.forecast(
            question=test_question,
            verbose=True,
            use_cache=True
        )
        signal.alarm(0)  # Cancel alarm

        elapsed = time.time() - start
        print(f"✓ Forecast completed in {elapsed:.2f}s")
        print(f"  Probability: {result.get('probability', 'N/A')}")
        print(f"  Confidence: {result.get('confidence', 'N/A')}")

    except TimeoutError as e:
        print(f"❌ Forecast timed out: {e}")
        print("   The forecast is hanging somewhere in the execution")
        sys.exit(1)

except Exception as e:
    print(f"❌ Forecast failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n✓ All tests passed - system is working")