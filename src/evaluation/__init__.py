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
from .calibration_metrics import CalibrationMetrics
from .drift_detector import DriftDetector
from .benchmark import Benchmark, HumanBaseline
from .evaluator import Evaluator

__all__ = [
    "BrierScorer",
    "ProvisionalScorer",
    "CalibrationMetrics",
    "DriftDetector",
    "Benchmark",
    "HumanBaseline",
    "Evaluator",
]
