"""Composite baseline risk score computation.

Produces a 0-100 risk score for any country using 4 weighted inputs:
  - GDELT event density (per-capita normalized)
  - ACLED conflict intensity (fatalities + event count)
  - Travel advisory level (with hard floor enforcement)
  - Goldstein severity (inverted: lower Goldstein = higher risk)

Advisory hard floors: Level 4 (Do Not Travel) >= 70, Level 3 >= 45.
"""

from __future__ import annotations

import math

# Component weights -- must sum to 1.0
WEIGHTS: dict[str, float] = {
    "advisory": 0.35,
    "acled": 0.25,
    "gdelt": 0.25,
    "goldstein": 0.15,
}

# Advisory level -> minimum baseline risk floor
ADVISORY_FLOORS: dict[int, float] = {
    4: 70.0,  # "Do Not Travel"
    3: 45.0,  # "Reconsider Travel"
}

# Advisory level -> component score (small dict, safe for content filters)
_ADVISORY_SCORES: dict[int, float] = {
    1: 10.0,   # "Exercise Normal Precautions"
    2: 35.0,   # "Exercise Increased Caution"
    3: 60.0,   # "Reconsider Travel"
    4: 90.0,   # "Do Not Travel"
}


def compute_baseline_risk(
    gdelt_event_count: int,
    population: int,
    acled_fatalities: int,
    acled_event_count: int,
    advisory_level: int,
    avg_goldstein: float,
) -> float:
    """Compute a 0-100 baseline risk score for a country.

    Args:
        gdelt_event_count: Number of GDELT events in the decay window.
        population: Country population (for per-capita normalization).
        acled_fatalities: ACLED fatality count in the decay window.
        acled_event_count: ACLED event count in the decay window.
        advisory_level: Travel advisory level (1-4). 0 or unknown treated as 1.
        avg_goldstein: Average Goldstein scale value (-10 to +10).

    Returns:
        Composite risk score clamped to [0.0, 100.0], rounded to 1 decimal.
    """
    # GDELT: per-capita event density
    pop_millions = max(population, 1) / 1_000_000
    gdelt_per_capita = gdelt_event_count / pop_millions
    gdelt_score = min(100.0, gdelt_per_capita * 2.0)

    # ACLED: conflict intensity (fatalities weighted higher than event count)
    acled_score = min(100.0, acled_fatalities * 5.0 + acled_event_count * 2.0)

    # Advisory: mapped level score
    advisory_score = _ADVISORY_SCORES.get(advisory_level, 10.0)

    # Goldstein: inverted -- lower (more negative) Goldstein = higher risk
    # Scale: -10 -> 100, 0 -> 50, +10 -> 0
    goldstein_score = max(0.0, min(100.0, (10.0 - avg_goldstein) * 5.0))

    # Weighted composite
    composite = (
        WEIGHTS["gdelt"] * gdelt_score
        + WEIGHTS["acled"] * acled_score
        + WEIGHTS["advisory"] * advisory_score
        + WEIGHTS["goldstein"] * goldstein_score
    )

    # Apply advisory hard floors
    floor = ADVISORY_FLOORS.get(advisory_level, 0.0)
    score = max(composite, floor)

    return round(min(100.0, max(0.0, score)), 1)


def decay_weight(age_days: float, half_life: float = 30.0) -> float:
    """Exponential decay weight for time-based event weighting.

    At age_days == half_life, weight == 0.5.
    At age_days == 0, weight == 1.0.
    At age_days == 2*half_life, weight == 0.25.

    Used by heatmap binner, arc builder, and baseline risk aggregation.

    Args:
        age_days: Age of the event in days (fractional allowed).
        half_life: Half-life in days (default 30).

    Returns:
        Decay weight in (0.0, 1.0].
    """
    if half_life <= 0:
        return 1.0
    return math.exp(-0.693 * age_days / half_life)
