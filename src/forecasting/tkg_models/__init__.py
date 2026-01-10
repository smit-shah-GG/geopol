"""
Temporal Knowledge Graph prediction models.

This subpackage provides interfaces to TKG prediction algorithms,
specifically RE-GCN for temporal link prediction.
"""

from .data_adapter import DataAdapter
from .regcn_wrapper import REGCNWrapper

__all__ = ['DataAdapter', 'REGCNWrapper']
