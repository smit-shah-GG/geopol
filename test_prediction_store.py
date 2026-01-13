from src.calibration import PredictionStore
import datetime

# Test prediction storage
store = PredictionStore(":memory:")  # Use in-memory DB for testing

# Store a test prediction
pred_id = store.store_prediction(
    query="Will there be a conflict between Russia and Ukraine in Q1 2024?",
    raw_probability=0.75,
    category="conflict",
    entities=["Russia", "Ukraine"],
    metadata={"test": True}
)
print(f"✓ Stored prediction with ID: {pred_id}")

# Retrieve predictions
predictions = store.get_predictions_for_calibration(category="conflict")
print(f"✓ Retrieved {len(predictions)} conflict predictions")

# Check fields
pred = predictions[0]
print(f"✓ Prediction has required fields:")
print(f"  - Query: {pred['query'][:50]}...")
print(f"  - Raw probability: {pred['raw_probability']}")
print(f"  - Category: {pred['category']}")
print(f"  - Calibrated probability: {pred['calibrated_probability']} (None expected)")

store.close()
print("✓ SQLite prediction tracking working correctly")
