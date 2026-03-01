"""Unit tests for TiRGN training loop components.

Tests training infrastructure (config, early stopping, history vocab,
checkpointing, logger, comparison report) WITHOUT running actual GPU
training. All JAX model operations use tiny synthetic data.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.training.train_tirgn import (
    TiRGNTrainingConfig,
    build_history_vocabulary_from_snapshots,
    load_tirgn_checkpoint,
    save_tirgn_checkpoint,
)
from src.training.training_logger import TrainingLogger


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

NUM_ENTITIES = 10
NUM_RELATIONS = 5


@pytest.fixture()
def synthetic_snapshots() -> list[np.ndarray]:
    """Three small synthetic snapshot triple arrays."""
    rng = np.random.RandomState(42)
    snapshots = []
    for _ in range(3):
        n_triples = rng.randint(5, 15)
        subjects = rng.randint(0, NUM_ENTITIES, size=n_triples)
        relations = rng.randint(0, NUM_RELATIONS * 2, size=n_triples)
        objects = rng.randint(0, NUM_ENTITIES, size=n_triples)
        snapshots.append(np.stack([subjects, relations, objects], axis=1))
    return snapshots


@pytest.fixture()
def tmp_model_dir() -> Path:
    """Temporary directory for checkpoint I/O tests."""
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


# ---------------------------------------------------------------------------
# Test 1: build_history_vocabulary returns expected structure
# ---------------------------------------------------------------------------


def test_build_history_vocabulary(synthetic_snapshots: list[np.ndarray]) -> None:
    """History vocabulary has (int, int) -> set[int] entries."""
    vocab = build_history_vocabulary_from_snapshots(
        synthetic_snapshots,
        num_entities=NUM_ENTITIES,
        num_relations=NUM_RELATIONS * 2,
        window_size=50,
    )

    assert isinstance(vocab, dict)
    assert len(vocab) > 0, "Vocabulary should have at least one entry"

    for key, val in vocab.items():
        assert isinstance(key, tuple) and len(key) == 2
        assert isinstance(key[0], int) and isinstance(key[1], int)
        assert isinstance(val, set)
        for obj in val:
            assert isinstance(obj, int)
            assert 0 <= obj < NUM_ENTITIES


# ---------------------------------------------------------------------------
# Test 2: Early stopping triggers after patience epochs without improvement
# ---------------------------------------------------------------------------


def test_early_stopping_triggers() -> None:
    """Early stopping fires when epochs_without_improvement >= patience."""
    config = TiRGNTrainingConfig(patience=10, eval_interval=5)

    # Simulate the early stopping logic from train_tirgn
    best_mrr = 0.5
    epochs_without_improvement = 0
    early_stopped = False

    # Simulate 4 eval rounds with no improvement
    for _ in range(4):
        current_mrr = 0.3  # worse than best
        if current_mrr > best_mrr:
            best_mrr = current_mrr
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += config.eval_interval

        if epochs_without_improvement >= config.patience:
            early_stopped = True
            break

    assert early_stopped, (
        f"Early stopping should have triggered: "
        f"epochs_without_improvement={epochs_without_improvement}, "
        f"patience={config.patience}"
    )


def test_early_stopping_resets_on_improvement() -> None:
    """Early stopping counter resets when MRR improves."""
    config = TiRGNTrainingConfig(patience=15, eval_interval=5)

    best_mrr = 0.0
    epochs_without_improvement = 0

    # Round 1: improvement
    mrr_sequence = [0.2, 0.1, 0.3, 0.05, 0.05, 0.05]
    for mrr in mrr_sequence:
        if mrr > best_mrr:
            best_mrr = mrr
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += config.eval_interval

    # After [0.2, 0.1, 0.3, 0.05, 0.05, 0.05]:
    # 0.2 -> reset, 0.1 -> +5, 0.3 -> reset, 0.05 -> +5, 0.05 -> +10, 0.05 -> +15
    assert epochs_without_improvement == 15
    assert best_mrr == 0.3


# ---------------------------------------------------------------------------
# Test 3: TiRGNTrainingConfig defaults
# ---------------------------------------------------------------------------


def test_training_config_defaults() -> None:
    """Default config has correct TiRGN-specific values."""
    config = TiRGNTrainingConfig()

    assert config.epochs == 100
    assert config.learning_rate == 0.001
    assert config.batch_size == 1024
    assert config.num_negatives == 10
    assert config.grad_clip == 1.0
    assert config.checkpoint_interval == 10
    assert config.eval_interval == 5

    # TiRGN-specific
    assert config.history_rate == 0.3
    assert config.history_window == 50
    assert config.patience == 15
    assert config.logdir == "runs/tirgn"


# ---------------------------------------------------------------------------
# Test 4: Checkpoint metadata includes model_type
# ---------------------------------------------------------------------------


def test_checkpoint_metadata_includes_model_type(tmp_model_dir: Path) -> None:
    """TiRGN checkpoint JSON contains model_type: 'tirgn'."""
    from src.training.models.tirgn_jax import create_tirgn_model

    model = create_tirgn_model(
        num_entities=NUM_ENTITIES,
        num_relations=NUM_RELATIONS,
        embedding_dim=32,
        num_layers=1,
        seed=0,
    )

    ckpt_path = tmp_model_dir / "tirgn_test.npz"
    save_tirgn_checkpoint(
        model,
        ckpt_path,
        epoch=5,
        metrics={"mrr": 0.42},
        entity_to_id={"A": 0, "B": 1},
        relation_to_id={"R1": 0},
    )

    # Verify JSON sidecar exists and has correct fields
    meta = load_tirgn_checkpoint(ckpt_path)
    assert meta["model_type"] == "tirgn"
    assert meta["epoch"] == 5
    assert meta["metrics"]["mrr"] == 0.42
    assert meta["config"]["history_rate"] == 0.3
    assert meta["config"]["history_window"] == 50
    assert meta["config"]["num_entities"] == NUM_ENTITIES
    assert meta["config"]["num_relations"] == NUM_RELATIONS
    assert meta["config"]["embedding_dim"] == 32

    # Verify .npz file exists
    assert ckpt_path.exists()


# ---------------------------------------------------------------------------
# Test 5: TrainingLogger writes expected metric keys
# ---------------------------------------------------------------------------


def test_logger_metrics_format() -> None:
    """TrainingLogger.log_metrics writes to TensorBoard with expected keys."""
    with tempfile.TemporaryDirectory() as logdir:
        with TrainingLogger(logdir, run_name="test_metrics") as tl:
            metrics = {
                "train/loss": 0.45,
                "train/lr": 0.001,
                "eval/mrr": 0.32,
                "eval/hits_at_1": 0.15,
                "eval/hits_at_3": 0.25,
                "eval/hits_at_10": 0.40,
                "system/vram_used_mb": 1024.5,
                "system/epoch_duration_s": 12.3,
            }
            # Should not raise
            tl.log_metrics(metrics, step=1)
            tl.log_metrics({"system/total_params": 50000.0}, step=0)
            tl.log_text("config", "test run", step=0)

        # Verify TensorBoard event files were created
        event_files = list(Path(logdir).glob("events.out.tfevents.*"))
        assert len(event_files) > 0, "TensorBoard event file not created"
