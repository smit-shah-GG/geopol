"""
Probability calibration system for geopolitical forecasting.

This module implements:
1. SQLite-backed prediction tracking (prediction_store.py)
2. Isotonic regression calibration (isotonic_calibrator.py)
3. Temperature scaling optimization (temperature_scaler.py)
4. Explainable calibration adjustments (explainer.py)

The calibration system transforms raw model predictions into calibrated probabilities
that accurately reflect true outcome likelihoods, enabling trustworthy confidence scores.
"""

from src.calibration.explainer import CalibrationExplainer
from src.calibration.isotonic_calibrator import IsotonicCalibrator
from src.calibration.prediction_store import Prediction, PredictionStore
from src.calibration.temperature_scaler import TemperatureScaler, probabilities_to_logits

__all__ = [
    "PredictionStore",
    "Prediction",
    "IsotonicCalibrator",
    "CalibrationExplainer",
    "TemperatureScaler",
    "probabilities_to_logits",
]
