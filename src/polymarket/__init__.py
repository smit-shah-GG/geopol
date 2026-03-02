"""
Polymarket prediction market comparison subsystem.

Provides API client, LLM-assisted matcher, and comparison service
for benchmarking Geopol forecasts against Polymarket prediction markets.
"""

from src.polymarket.client import PolymarketClient
from src.polymarket.comparison import PolymarketComparisonService
from src.polymarket.matcher import PolymarketMatcher

__all__ = [
    "PolymarketClient",
    "PolymarketMatcher",
    "PolymarketComparisonService",
]
