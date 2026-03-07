"""
Metric computation for backtesting evaluation windows.

All functions are stateless and operate on plain lists of floats.
No database access, no side effects -- pure numpy computation.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


def compute_brier_score(
    predictions: list[float],
    outcomes: list[float],
) -> float:
    """Compute Brier score: mean squared error between predicted and actual.

    Lower is better. Perfect = 0.0, random baseline = 0.25, worst = 1.0.

    Args:
        predictions: Predicted probabilities in [0, 1].
        outcomes: Actual outcomes (0.0 or 1.0).

    Returns:
        Brier score as a float.

    Raises:
        ValueError: If inputs are empty or have mismatched lengths.
    """
    if len(predictions) != len(outcomes):
        raise ValueError(
            f"Length mismatch: predictions={len(predictions)}, "
            f"outcomes={len(outcomes)}"
        )
    if len(predictions) == 0:
        raise ValueError("Cannot compute Brier score with empty data")

    p = np.asarray(predictions, dtype=np.float64)
    o = np.asarray(outcomes, dtype=np.float64)
    return float(np.mean((p - o) ** 2))


def compute_calibration_bins(
    predictions: list[float],
    outcomes: list[float],
    n_bins: int = 10,
) -> dict[str, Any]:
    """Compute calibration reliability diagram bins.

    Buckets predictions into equally-spaced probability bins, then
    computes the mean predicted probability and observed frequency
    within each bin. Used to construct reliability diagrams.

    Args:
        predictions: Predicted probabilities in [0, 1].
        outcomes: Actual outcomes (0.0 or 1.0).
        n_bins: Number of bins (default 10 for 0.0-0.1, 0.1-0.2, ...).

    Returns:
        Dict with keys:
            bins: List of bin edges [0.0, 0.1, ..., 1.0].
            predicted_avg: Mean predicted probability per bin (None if empty).
            observed_freq: Observed outcome frequency per bin (None if empty).
            counts: Number of predictions in each bin.
    """
    if len(predictions) != len(outcomes):
        raise ValueError(
            f"Length mismatch: predictions={len(predictions)}, "
            f"outcomes={len(outcomes)}"
        )

    p = np.asarray(predictions, dtype=np.float64)
    o = np.asarray(outcomes, dtype=np.float64)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)

    predicted_avg: list[Optional[float]] = []
    observed_freq: list[Optional[float]] = []
    counts: list[int] = []

    for i in range(n_bins):
        low = bin_edges[i]
        high = bin_edges[i + 1]

        # Include right edge for last bin to capture predictions == 1.0
        if i == n_bins - 1:
            mask = (p >= low) & (p <= high)
        else:
            mask = (p >= low) & (p < high)

        n = int(np.sum(mask))
        counts.append(n)

        if n == 0:
            predicted_avg.append(None)
            observed_freq.append(None)
        else:
            predicted_avg.append(float(np.mean(p[mask])))
            observed_freq.append(float(np.mean(o[mask])))

    return {
        "bins": bin_edges.tolist(),
        "predicted_avg": predicted_avg,
        "observed_freq": observed_freq,
        "counts": counts,
    }


def compute_hit_rate(
    predictions: list[float],
    outcomes: list[float],
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Compute binary classification hit rate at a given threshold.

    A prediction is "correct" (a hit) if:
    - predicted >= threshold AND outcome == 1.0, OR
    - predicted < threshold AND outcome == 0.0

    Args:
        predictions: Predicted probabilities in [0, 1].
        outcomes: Actual outcomes (0.0 or 1.0).
        threshold: Decision boundary (default 0.5).

    Returns:
        Dict with keys: total, correct, hit_rate.
    """
    if len(predictions) != len(outcomes):
        raise ValueError(
            f"Length mismatch: predictions={len(predictions)}, "
            f"outcomes={len(outcomes)}"
        )

    total = len(predictions)
    if total == 0:
        return {"total": 0, "correct": 0, "hit_rate": 0.0}

    p = np.asarray(predictions, dtype=np.float64)
    o = np.asarray(outcomes, dtype=np.float64)

    correct_positive = (p >= threshold) & (o == 1.0)
    correct_negative = (p < threshold) & (o == 0.0)
    correct = int(np.sum(correct_positive) + np.sum(correct_negative))

    return {
        "total": total,
        "correct": correct,
        "hit_rate": correct / total,
    }


def compute_mrr(rankings: list[int]) -> float:
    """Compute Mean Reciprocal Rank from TKG prediction rankings.

    Each ranking value represents the position (1-based) of the correct
    entity in the TKG model's ranked prediction list. MRR is the mean
    of 1/rank across all queries.

    Args:
        rankings: List of 1-based rank positions. Empty list returns 0.0.

    Returns:
        MRR as a float in [0, 1].
    """
    if not rankings:
        return 0.0

    reciprocals = [1.0 / r for r in rankings if r > 0]
    if not reciprocals:
        return 0.0

    return float(np.mean(reciprocals))
