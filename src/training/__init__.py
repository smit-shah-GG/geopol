"""
Training components for TKG predictor.
"""

from .data_collector import GDELTHistoricalCollector
from .data_processor import GDELTDataProcessor

__all__ = ["GDELTHistoricalCollector", "GDELTDataProcessor"]
