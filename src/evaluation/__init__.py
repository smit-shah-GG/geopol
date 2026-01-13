"""
Evaluation module for geopolitical forecasting system.

Provides:
- Brier score calculation for resolved predictions
- Provisional scoring for unresolved predictions
- Calibration metrics (ECE, MCE, ACE)
- Drift detection and monitoring
- Human baseline comparison
- Comprehensive evaluation framework
"""

from .brier_scorer import BrierScorer
from .provisional_scorer import ProvisionalScorer

__all__ = [
    "BrierScorer",
    "ProvisionalScorer",
]
