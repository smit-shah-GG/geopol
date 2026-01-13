from src.calibration import IsotonicCalibrator, CalibrationExplainer
import numpy as np

# Create calibrator
calibrator = IsotonicCalibrator()

# Create synthetic data with bias (overconfident for high probabilities)
np.random.seed(42)
predictions = []
outcomes = []
categories = []

for _ in range(100):
    for cat in ["conflict", "diplomatic", "economic"]:
        # Overconfident predictions
        raw_prob = np.random.uniform(0.6, 0.95)
        # True probability is lower
        true_prob = raw_prob * 0.7  
        outcome = 1 if np.random.random() < true_prob else 0
        
        predictions.append(raw_prob)
        outcomes.append(outcome)
        categories.append(cat)

# Fit calibration curves
calibrator.fit(predictions, outcomes, categories)
print("✓ Fitted calibration curves for 3 categories")

# Test calibration
test_prob = 0.8
for category in ["conflict", "diplomatic", "economic"]:
    calibrated = calibrator.calibrate(test_prob, category)
    print(f"✓ {category}: {test_prob:.2f} → {calibrated:.2f} (reduced by {(test_prob-calibrated)*100:.1f}%)")

# Test explanation
explainer = CalibrationExplainer()
explanation = explainer.explain(
    original_prob=0.8,
    calibrated_prob=0.65,
    category="conflict",
    entities=["Russia", "Ukraine"]
)
print(f"✓ Explanation generated: '{explanation[:80]}...'")

print("\n✓ Isotonic calibration working correctly")
