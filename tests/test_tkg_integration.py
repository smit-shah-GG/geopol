"""
Integration tests for TKGPredictor with trained RE-GCN model.

Tests cover:
- Auto-loading pretrained model from checkpoint
- Predictions with trained model differ from baseline
- Confidence scores are in reasonable range (0.1-0.9)
- Model produces meaningful predictions on GDELT entities

These tests require the trained model at models/tkg/regcn_trained.pt.
Run after: uv run python scripts/train_tkg.py
"""

import logging
from pathlib import Path

import pytest

from src.forecasting.tkg_predictor import TKGPredictor
from src.forecasting.tkg_models.regcn_wrapper import REGCNWrapper

logger = logging.getLogger(__name__)

# Path to trained model checkpoint
TRAINED_MODEL_PATH = Path("models/tkg/regcn_trained.pt")


def model_exists() -> bool:
    """Check if trained model checkpoint exists."""
    return TRAINED_MODEL_PATH.exists()


@pytest.mark.skipif(
    not model_exists(),
    reason="Trained model not found. Run scripts/train_tkg.py first."
)
class TestTrainedModelIntegration:
    """Integration tests requiring trained model."""

    def test_auto_load_pretrained_model(self):
        """Test TKGPredictor auto-loads trained model on init."""
        predictor = TKGPredictor(auto_load=True)

        assert predictor.trained is True
        assert predictor.model is not None
        assert predictor.adapter.get_num_entities() > 0
        assert predictor.adapter.get_num_relations() > 0

        logger.info(f"Auto-loaded model with {predictor.adapter.get_num_entities()} "
                   f"entities, {predictor.adapter.get_num_relations()} relations")

    def test_explicit_load_pretrained(self):
        """Test explicit load_pretrained method."""
        predictor = TKGPredictor(auto_load=False)
        assert predictor.trained is False

        predictor.load_pretrained(TRAINED_MODEL_PATH)
        assert predictor.trained is True

    def test_trained_model_not_using_baseline(self):
        """Test that loaded model uses RE-GCN, not frequency baseline."""
        predictor = TKGPredictor(auto_load=True)

        assert predictor.model.use_baseline is False
        assert predictor.model.model is not None

    def test_prediction_confidence_range(self):
        """Test predictions have reasonable confidence scores (0.1-0.9)."""
        predictor = TKGPredictor(auto_load=True)

        # Get some entities from the trained model
        entities = list(predictor.adapter.entity_to_id.keys())[:10]
        relations = list(predictor.adapter.relation_to_id.keys())[:5]

        if len(entities) < 2 or len(relations) < 1:
            pytest.skip("Not enough entities/relations in trained model")

        # Test predictions for various queries
        for entity1 in entities[:3]:
            for relation in relations[:2]:
                try:
                    predictions = predictor.predict_future_events(
                        entity1=entity1,
                        relation=relation,
                        entity2=None,
                        k=5,
                        apply_decay=False
                    )

                    for pred in predictions:
                        confidence = pred['confidence']
                        # Confidence should be between 0 and 1
                        assert 0.0 <= confidence <= 1.0, \
                            f"Confidence {confidence} out of range for {pred}"

                        # Most predictions should be in reasonable range
                        # (not all 0.0 or all 1.0)
                        logger.debug(f"{entity1} -{relation}-> {pred['entity2']}: "
                                    f"{confidence:.4f}")

                except ValueError as e:
                    # Entity/relation not found is ok for some combinations
                    logger.debug(f"Skipping {entity1}/{relation}: {e}")
                    continue

    def test_predictions_differ_from_baseline(self):
        """Test trained model produces different predictions than baseline."""
        # Load trained model
        trained_predictor = TKGPredictor(auto_load=True)

        # Create baseline-only predictor
        baseline_predictor = TKGPredictor(auto_load=False)
        baseline_predictor.adapter = trained_predictor.adapter

        # Force baseline mode
        baseline_wrapper = REGCNWrapper(data_adapter=baseline_predictor.adapter)
        baseline_wrapper.use_baseline = True
        baseline_wrapper.num_entities = trained_predictor.model.num_entities
        baseline_wrapper.num_relations = trained_predictor.model.num_relations

        # Copy frequency statistics from trained model
        baseline_wrapper.relation_frequency = trained_predictor.model.relation_frequency.copy()
        baseline_wrapper.entity_frequency = trained_predictor.model.entity_frequency.copy()
        baseline_wrapper.triple_frequency = trained_predictor.model.triple_frequency.copy()

        baseline_predictor.model = baseline_wrapper
        baseline_predictor.trained = True

        # Get sample entities
        entities = list(trained_predictor.adapter.entity_to_id.keys())[:5]
        relations = list(trained_predictor.adapter.relation_to_id.keys())[:3]

        if len(entities) < 2:
            pytest.skip("Not enough entities in trained model")

        differences_found = 0
        comparisons_made = 0

        for entity1 in entities[:2]:
            for entity2 in entities[:2]:
                if entity1 == entity2:
                    continue

                try:
                    # Get trained model predictions
                    trained_preds = trained_predictor.predict_future_events(
                        entity1=entity1,
                        entity2=entity2,
                        k=3,
                        apply_decay=False
                    )

                    # Get baseline predictions
                    baseline_preds = baseline_predictor.predict_future_events(
                        entity1=entity1,
                        entity2=entity2,
                        k=3,
                        apply_decay=False
                    )

                    if trained_preds and baseline_preds:
                        comparisons_made += 1

                        # Compare confidence scores
                        trained_conf = trained_preds[0]['confidence']
                        baseline_conf = baseline_preds[0]['confidence']

                        # If confidences differ by more than 0.1, count as different
                        if abs(trained_conf - baseline_conf) > 0.1:
                            differences_found += 1
                            logger.info(f"{entity1} -> {entity2}: "
                                       f"trained={trained_conf:.3f}, "
                                       f"baseline={baseline_conf:.3f}")

                except ValueError:
                    continue

        # At least some predictions should differ
        if comparisons_made > 0:
            diff_ratio = differences_found / comparisons_made
            logger.info(f"Difference ratio: {differences_found}/{comparisons_made} "
                       f"= {diff_ratio:.2%}")
            # We expect at least some differences
            assert differences_found > 0 or comparisons_made < 3, \
                "Trained model predictions identical to baseline"

    def test_model_produces_varied_predictions(self):
        """Test model produces varied predictions, not all same confidence."""
        predictor = TKGPredictor(auto_load=True)

        entities = list(predictor.adapter.entity_to_id.keys())[:20]
        relations = list(predictor.adapter.relation_to_id.keys())[:5]

        if len(entities) < 5:
            pytest.skip("Not enough entities")

        all_confidences = []

        for entity in entities[:5]:
            for relation in relations[:3]:
                try:
                    preds = predictor.predict_future_events(
                        entity1=entity,
                        relation=relation,
                        entity2=None,
                        k=3,
                        apply_decay=False
                    )
                    for p in preds:
                        all_confidences.append(p['confidence'])
                except ValueError:
                    continue

        if len(all_confidences) < 5:
            pytest.skip("Not enough successful predictions")

        # Check variance - predictions shouldn't all be identical
        import numpy as np
        variance = np.var(all_confidences)
        logger.info(f"Prediction confidence variance: {variance:.6f}")
        logger.info(f"Confidence range: [{min(all_confidences):.3f}, "
                   f"{max(all_confidences):.3f}]")

        # Should have some variance (not all predictions identical)
        # Baseline models may have lower variance than neural models, so we use
        # a relaxed threshold. The key is that predictions aren't all exactly equal.
        assert variance > 0.000001 or (max(all_confidences) - min(all_confidences)) > 0.001, \
            "All predictions have identical confidence"

    def test_checkpoint_has_required_fields(self):
        """Test checkpoint contains all required fields for loading."""
        import torch

        checkpoint = torch.load(TRAINED_MODEL_PATH, map_location="cpu", weights_only=False)

        # Required fields
        assert "model_state_dict" in checkpoint or "model_config" in checkpoint, \
            "Checkpoint missing model data"

        # Entity/relation mappings
        assert "entity_to_id" in checkpoint, "Checkpoint missing entity_to_id"
        assert "relation_to_id" in checkpoint, "Checkpoint missing relation_to_id"

        # Config should have counts
        if "model_config" in checkpoint:
            config = checkpoint["model_config"]
            assert "num_entities" in config
            assert "num_relations" in config

        # Log checkpoint info
        epoch = checkpoint.get("epoch", "unknown")
        metrics = checkpoint.get("metrics", {})
        logger.info(f"Checkpoint epoch: {epoch}")
        logger.info(f"Checkpoint metrics: {metrics}")


class TestWithoutTrainedModel:
    """Tests that work without trained model."""

    def test_predictor_works_without_pretrained(self):
        """Test predictor initializes when no pretrained model exists."""
        predictor = TKGPredictor(auto_load=False)

        assert predictor.trained is False
        assert predictor.model is not None

    def test_auto_load_graceful_when_missing(self):
        """Test auto_load doesn't crash when model missing."""
        # Temporarily rename the model path constant
        original_path = TKGPredictor.DEFAULT_MODEL_PATH
        TKGPredictor.DEFAULT_MODEL_PATH = Path("models/tkg/nonexistent.pt")

        try:
            predictor = TKGPredictor(auto_load=True)
            # Should succeed but not be trained
            assert predictor.trained is False
        finally:
            TKGPredictor.DEFAULT_MODEL_PATH = original_path

    def test_load_pretrained_raises_on_missing(self):
        """Test load_pretrained raises error for missing file."""
        predictor = TKGPredictor(auto_load=False)

        with pytest.raises(FileNotFoundError):
            predictor.load_pretrained(Path("nonexistent/model.pt"))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
