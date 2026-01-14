"""
Vector embedding module for temporal knowledge graphs.

Implements RotatE embedding model for entity/relation representation.
HyTE and trainer modules planned for Phase 5.
"""

from .rotate import RotatEModel

__all__ = ["RotatEModel"]
