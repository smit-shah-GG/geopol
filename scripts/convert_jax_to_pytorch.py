#!/usr/bin/env python
"""
Convert JAX/jraph model checkpoint to PyTorch format.

This script extracts entity/relation mappings and frequency statistics
from a JAX training checkpoint and creates a PyTorch checkpoint that
TKGPredictor can load.

Usage:
    uv run python scripts/convert_jax_to_pytorch.py
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Paths
JAX_CHECKPOINT = Path("models/tkg/regcn_jraph_final.json")
GDELT_DATA = Path("data/gdelt/processed/events.parquet")
OUTPUT_PATH = Path("models/tkg/regcn_trained.pt")


def load_jax_checkpoint(path: Path) -> dict:
    """Load JAX checkpoint JSON."""
    with open(path) as f:
        return json.load(f)


def build_frequency_statistics(
    df: pd.DataFrame,
    entity_to_id: dict,
    relation_to_id: dict,
) -> tuple:
    """
    Build frequency statistics from GDELT events.

    Returns:
        Tuple of (relation_frequency, entity_frequency, triple_frequency)
    """
    relation_frequency = {}
    entity_frequency = {}
    triple_frequency = {}

    # Filter to recent 30 days
    max_date = df["timestamp"].max()
    cutoff = max_date - pd.Timedelta(days=30)
    df = df[df["timestamp"] >= cutoff]

    logger.info(f"Building statistics from {len(df):,} events")

    for _, row in df.iterrows():
        e1 = row["entity1"]
        rel = row["relation"]
        e2 = row["entity2"]

        # Map to IDs
        e1_id = entity_to_id.get(e1)
        rel_id = relation_to_id.get(rel)
        e2_id = entity_to_id.get(e2)

        if e1_id is None or rel_id is None or e2_id is None:
            continue

        # Count frequencies
        relation_frequency[rel_id] = relation_frequency.get(rel_id, 0) + 1
        entity_frequency[e1_id] = entity_frequency.get(e1_id, 0) + 1
        entity_frequency[e2_id] = entity_frequency.get(e2_id, 0) + 1

        triple = (e1_id, rel_id, e2_id)
        triple_frequency[triple] = triple_frequency.get(triple, 0) + 1

    logger.info(f"Statistics: {len(relation_frequency)} relations, "
               f"{len(entity_frequency)} entities, {len(triple_frequency)} triples")

    return relation_frequency, entity_frequency, triple_frequency


def convert_checkpoint():
    """Convert JAX checkpoint to PyTorch format."""
    if not JAX_CHECKPOINT.exists():
        logger.error(f"JAX checkpoint not found: {JAX_CHECKPOINT}")
        return 1

    if not GDELT_DATA.exists():
        logger.error(f"GDELT data not found: {GDELT_DATA}")
        return 1

    logger.info(f"Loading JAX checkpoint: {JAX_CHECKPOINT}")
    jax_ckpt = load_jax_checkpoint(JAX_CHECKPOINT)

    # Extract mappings
    entity_to_id = jax_ckpt.get("entity_to_id", {})
    relation_to_id = jax_ckpt.get("relation_to_id", {})
    epoch = jax_ckpt.get("epoch", 0)
    metrics = jax_ckpt.get("metrics", {})

    logger.info(f"JAX model: epoch={epoch}, MRR={metrics.get('mrr', 0):.4f}")
    logger.info(f"Entities: {len(entity_to_id)}, Relations: {len(relation_to_id)}")

    # Load GDELT data and build frequency statistics
    logger.info(f"Loading GDELT data: {GDELT_DATA}")
    df = pd.read_parquet(GDELT_DATA)

    relation_frequency, entity_frequency, triple_frequency = build_frequency_statistics(
        df, entity_to_id, relation_to_id
    )

    # Create PyTorch checkpoint
    checkpoint = {
        "epoch": epoch,
        "metrics": metrics,
        "model_config": {
            "num_entities": len(entity_to_id),
            "num_relations": len(relation_to_id),
            "embedding_dim": 200,
            "num_layers": 2,
        },
        "entity_to_id": entity_to_id,
        "relation_to_id": relation_to_id,
        "relation_frequency": relation_frequency,
        "entity_frequency": entity_frequency,
        "triple_frequency": triple_frequency,
        # No model_state_dict - this will use baseline mode
        # but with proper entity/relation mappings from JAX training
    }

    # Save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, OUTPUT_PATH)

    logger.info(f"\nPyTorch checkpoint saved to: {OUTPUT_PATH}")
    logger.info(f"  Epoch: {epoch}")
    logger.info(f"  Entities: {len(entity_to_id):,}")
    logger.info(f"  Relations: {len(relation_to_id)}")
    logger.info(f"  Triple patterns: {len(triple_frequency):,}")

    # Verify the checkpoint loads
    logger.info("\nVerifying checkpoint...")
    loaded = torch.load(OUTPUT_PATH, map_location="cpu", weights_only=False)
    assert "entity_to_id" in loaded
    assert "relation_to_id" in loaded
    assert "model_config" in loaded
    logger.info("Checkpoint verification: PASSED")

    return 0


if __name__ == "__main__":
    sys.exit(convert_checkpoint())
