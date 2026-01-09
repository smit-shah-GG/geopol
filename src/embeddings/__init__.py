"""
Vector embedding module for temporal knowledge graphs.

Implements RotatE and HyTE embedding models for entity/relation representation.
"""

from .rotate import RotatEModel
from .hyte import HyTEModel
from .trainer import EmbeddingTrainer

__all__ = ["RotatEModel", "HyTEModel", "EmbeddingTrainer"]
