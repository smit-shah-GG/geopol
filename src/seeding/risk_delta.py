"""7-day risk score change computation.

Compares current baseline risk scores against previous values and
identifies countries where the risk changed by more than a configurable
threshold. Used by the globe "scenarios" layer to visualize where
things are getting worse (red shift) or better (green shift).
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def compute_risk_deltas(
    current_risks: dict[str, float],
    previous_risks: dict[str, float],
    threshold: float = 3.0,
) -> list[dict]:
    """Compute significant risk score changes per country.

    For each country in ``current_risks``, computes delta = current - previous.
    Countries absent from ``previous_risks`` use 0.0 as the baseline (new
    entry). Only deltas with |delta| >= threshold are returned.

    Args:
        current_risks: Mapping of ISO alpha-2 -> current risk score (0-100).
        previous_risks: Mapping of ISO alpha-2 -> previous risk score (0-100).
        threshold: Minimum absolute delta to include (default 10.0).

    Returns:
        List of dicts: country_iso, current_risk, previous_risk, delta.
        Sorted by absolute delta descending.
    """
    deltas: list[dict] = []

    for country_iso, current in current_risks.items():
        previous = previous_risks.get(country_iso, 0.0)
        delta = current - previous

        if abs(delta) >= threshold:
            deltas.append(
                {
                    "country_iso": country_iso,
                    "current_risk": round(current, 1),
                    "previous_risk": round(previous, 1),
                    "delta": round(delta, 1),
                }
            )

    # Sort by absolute delta descending -- biggest changes first
    deltas.sort(key=lambda d: abs(d["delta"]), reverse=True)

    logger.info(
        "risk_delta: %d countries checked, %d significant changes (threshold=%.1f)",
        len(current_risks),
        len(deltas),
        threshold,
    )
    return deltas
