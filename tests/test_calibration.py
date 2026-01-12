"""
Tests for calibration system (isotonic calibration and explanations).

This test suite verifies:
1. Isotonic calibration training and inference
2. Per-category calibration curves
3. Explanation generation for adjustments
4. Calibration curve persistence
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.calibration.explainer import CalibrationExplainer
from src.calibration.isotonic_calibrator import IsotonicCalibrator
from src.calibration.prediction_store import PredictionStore


class TestIsotonicCalibrator:
    """Test isotonic calibration functionality."""

    def test_fit_isotonic_curves(self, tmp_path):
        """Test fitting per-category isotonic curves."""
        # Create calibrator
        calibrator = IsotonicCalibrator(calibration_dir=str(tmp_path / "calibration"))

        # Generate synthetic calibration data
        # Simulate overconfident conflict predictions
        n_conflict = 100
        conflict_preds = np.random.uniform(0.4, 0.9, n_conflict)
        # Actual outcomes are lower than predictions (overconfident)
        conflict_outcomes = (conflict_preds - 0.15 + np.random.normal(0, 0.1, n_conflict) > 0.5).astype(float)

        # Simulate well-calibrated diplomatic predictions
        n_diplomatic = 100
        diplomatic_preds = np.random.uniform(0.3, 0.8, n_diplomatic)
        diplomatic_outcomes = (diplomatic_preds + np.random.normal(0, 0.1, n_diplomatic) > 0.5).astype(float)

        # Simulate underconfident economic predictions
        n_economic = 100
        economic_preds = np.random.uniform(0.3, 0.7, n_economic)
        economic_outcomes = (economic_preds + 0.15 + np.random.normal(0, 0.1, n_economic) > 0.5).astype(float)

        # Combine data
        predictions = np.concatenate([conflict_preds, diplomatic_preds, economic_preds])
        outcomes = np.concatenate([conflict_outcomes, diplomatic_outcomes, economic_outcomes])
        categories = (
            ["conflict"] * n_conflict +
            ["diplomatic"] * n_diplomatic +
            ["economic"] * n_economic
        )

        # Fit calibrator
        results = calibrator.fit(
            predictions.tolist(),
            outcomes.tolist(),
            categories
        )

        # Verify all categories trained
        assert "conflict" in results
        assert "diplomatic" in results
        assert "economic" in results
        assert results["conflict"] == n_conflict
        assert results["diplomatic"] == n_diplomatic
        assert results["economic"] == n_economic

        # Verify calibrators exist
        info = calibrator.get_calibration_info()
        assert info["trained"] is True
        assert len(info["categories"]) == 3
        assert info["total_samples"] == n_conflict + n_diplomatic + n_economic

        print(f"✓ Trained {len(info['categories'])} category calibrators")
        print(f"  Sample counts: {info['sample_counts']}")

    def test_calibrate_per_category(self, tmp_path):
        """Test per-category calibration adjustments."""
        calibrator = IsotonicCalibrator(calibration_dir=str(tmp_path / "calibration"))

        # Create simple training data with known bias
        # Conflict: overconfident (predictions too high)
        conflict_preds = [0.7, 0.8, 0.9, 0.6, 0.75]
        conflict_outcomes = [0.0, 1.0, 1.0, 0.0, 0.0]  # Only 2/5 = 40% actually happened

        # Repeat data to meet minimum sample requirements
        conflict_preds = conflict_preds * 20  # 100 samples
        conflict_outcomes = conflict_outcomes * 20

        categories = ["conflict"] * len(conflict_preds)

        calibrator.fit(conflict_preds, conflict_outcomes, categories)

        # Test calibration - high predictions should be reduced
        cal_high = calibrator.calibrate(0.8, "conflict")
        assert cal_high < 0.8, "Overconfident predictions should be reduced"

        # Test batch calibration
        batch_preds = [0.7, 0.8, 0.9]
        batch_cats = ["conflict"] * 3
        batch_cal = calibrator.calibrate_batch(batch_preds, batch_cats)

        assert len(batch_cal) == 3
        assert all(cal <= raw for cal, raw in zip(batch_cal, batch_preds))

        print(f"✓ Per-category calibration working")
        print(f"  Raw: {batch_preds}")
        print(f"  Calibrated: {[f'{c:.3f}' for c in batch_cal]}")

    def test_calibration_persistence(self, tmp_path):
        """Test saving and loading calibration curves."""
        calibration_dir = tmp_path / "calibration"

        # Train calibrator
        calibrator1 = IsotonicCalibrator(calibration_dir=str(calibration_dir))

        predictions = [0.3, 0.4, 0.5, 0.6, 0.7] * 20
        outcomes = [0.0, 0.0, 1.0, 1.0, 1.0] * 20
        categories = ["conflict"] * len(predictions)

        calibrator1.fit(predictions, outcomes, categories)
        cal_before = calibrator1.calibrate(0.6, "conflict")

        # Create new calibrator and load saved curves
        calibrator2 = IsotonicCalibrator(calibration_dir=str(calibration_dir))
        loaded = calibrator2.load_calibrators()

        assert loaded is True, "Should load calibrators successfully"

        # Verify same calibration result
        cal_after = calibrator2.calibrate(0.6, "conflict")
        assert abs(cal_before - cal_after) < 0.01, "Loaded calibrator should match original"

        # Verify calibration file exists
        calibration_file = calibration_dir / "conflict_calibrator.pkl"
        assert calibration_file.exists(), "Calibration file should be saved"

        print(f"✓ Calibration curves persisted and loaded correctly")
        print(f"  Before: {cal_before:.3f}, After: {cal_after:.3f}")

    def test_explanation_generation(self, tmp_path):
        """Test generating explanations for calibration adjustments."""
        # Create prediction store with historical data
        store = PredictionStore(db_path=str(tmp_path / "predictions.db"))

        # Store historical predictions with outcomes
        for i in range(10):
            pred_id = store.store_prediction(
                query=f"Will Russia escalate in scenario {i}?",
                raw_probability=0.7,  # Consistently predict 70%
                category="conflict",
                entities=["Russia", "Ukraine"],
                metadata={"test": True}
            )
            # Only 3/10 actually happened (overconfident)
            store.update_outcome(pred_id, 1.0 if i < 3 else 0.0)

        # Create explainer with prediction store
        explainer = CalibrationExplainer(prediction_store=store)

        # Generate explanation for calibration adjustment
        explanation = explainer.explain_adjustment(
            raw_probability=0.7,
            calibrated_probability=0.5,  # Reduced by 20%
            category="conflict",
            entities=["Russia", "Ukraine"],
            query="Will Russia escalate next month?"
        )

        assert "decreased" in explanation.lower() or "reduced" in explanation.lower()
        assert "20" in explanation or "0.2" in explanation  # Adjustment percentage
        print(f"✓ Generated explanation: {explanation}")

    def test_get_calibration_curve(self, tmp_path):
        """Test retrieving calibration curve for visualization."""
        calibrator = IsotonicCalibrator(calibration_dir=str(tmp_path / "calibration"))

        # Train simple calibrator
        predictions = np.linspace(0.1, 0.9, 50).tolist()
        outcomes = (np.array(predictions) - 0.1 > 0.5).astype(float).tolist()  # Slightly overconfident
        categories = ["conflict"] * len(predictions)

        calibrator.fit(predictions, outcomes, categories)

        # Get calibration curve
        raw_probs, cal_probs = calibrator.get_calibration_curve("conflict", n_points=20)

        assert len(raw_probs) == 20
        assert len(cal_probs) == 20
        assert all(0 <= p <= 1 for p in raw_probs)
        assert all(0 <= p <= 1 for p in cal_probs)

        print(f"✓ Retrieved calibration curve with {len(raw_probs)} points")


class TestCalibrationExplainer:
    """Test calibration explanation functionality."""

    def test_explain_without_history(self):
        """Test explanation generation without historical data."""
        explainer = CalibrationExplainer(prediction_store=None)

        explanation = explainer.explain_adjustment(
            raw_probability=0.8,
            calibrated_probability=0.6,
            category="conflict",
            entities=["Russia", "Ukraine"],
            query="Will conflict escalate?"
        )

        assert len(explanation) > 0
        assert "decreased" in explanation.lower() or "reduced" in explanation.lower()
        assert "conflict" in explanation.lower()

        print(f"✓ Generated generic explanation: {explanation}")

    def test_no_adjustment_explanation(self):
        """Test explanation when no calibration needed."""
        explainer = CalibrationExplainer(prediction_store=None)

        explanation = explainer.explain_adjustment(
            raw_probability=0.5,
            calibrated_probability=0.501,  # Tiny adjustment
            category="diplomatic",
            entities=["EU", "US"],
            query="Will agreement be reached?"
        )

        assert "no" in explanation.lower() and "adjustment" in explanation.lower()

        print(f"✓ Generated no-adjustment explanation: {explanation}")


def test_full_calibration_workflow(tmp_path):
    """Integration test for full calibration workflow."""
    # 1. Create prediction store
    store = PredictionStore(db_path=str(tmp_path / "predictions.db"))

    # 2. Store predictions with outcomes
    predictions_data = []
    for i in range(50):
        category = ["conflict", "diplomatic", "economic"][i % 3]
        raw_prob = 0.5 + np.random.uniform(-0.3, 0.3)
        pred_id = store.store_prediction(
            query=f"Test prediction {i}",
            raw_probability=raw_prob,
            category=category,
            entities=["Entity1", "Entity2"],
            metadata={"test": True}
        )
        # Simulate outcomes
        outcome = 1.0 if np.random.random() < raw_prob else 0.0
        store.update_outcome(pred_id, outcome)

        predictions_data.append({
            "id": pred_id,
            "raw_prob": raw_prob,
            "category": category,
            "outcome": outcome
        })

    # 3. Train calibrator
    calibrator = IsotonicCalibrator(calibration_dir=str(tmp_path / "calibration"))

    resolved = store.get_predictions_for_calibration(resolved_only=True)
    assert len(resolved) == 50

    predictions = [p["raw_probability"] for p in resolved]
    outcomes = [p["outcome"] for p in resolved]
    categories = [p["category"] for p in resolved]

    results = calibrator.fit(predictions, outcomes, categories)

    # 4. Apply calibration to new predictions
    new_pred_id = store.store_prediction(
        query="New prediction",
        raw_probability=0.7,
        category="conflict",
        entities=["Russia", "Ukraine"],
        metadata={}
    )

    calibrated_prob = calibrator.calibrate(0.7, "conflict")

    # Update with calibrated probability
    store.update_calibrated_probability(new_pred_id, calibrated_prob)

    # 5. Generate explanation
    explainer = CalibrationExplainer(prediction_store=store)
    explanation = explainer.explain_adjustment(
        raw_probability=0.7,
        calibrated_probability=calibrated_prob,
        category="conflict",
        entities=["Russia", "Ukraine"],
        query="New prediction"
    )

    print(f"\n✓ Full calibration workflow completed:")
    print(f"  Trained on {len(resolved)} predictions")
    print(f"  Categories: {list(results.keys())}")
    print(f"  Raw prob: 0.7 -> Calibrated: {calibrated_prob:.3f}")
    print(f"  Explanation: {explanation}")

    assert 0.0 <= calibrated_prob <= 1.0


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
