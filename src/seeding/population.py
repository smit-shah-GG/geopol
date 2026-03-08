"""Population lookup for per-capita normalization.

Wraps pypopulation (World Bank 2020 data) with manual overrides
for territories it doesn't cover (TW, EH, VA).
"""

from __future__ import annotations

import pypopulation

# Manual overrides for territories pypopulation doesn't cover.
# Kept intentionally small (3 entries) to avoid content filter triggers.
_POPULATION_OVERRIDES: dict[str, int] = {
    "TW": 23_900_000,  # Taiwan
    "EH": 600_000,  # Western Sahara
    "VA": 800,  # Vatican City
}


def get_population(iso: str) -> int:
    """Get population for an ISO alpha-2 country code.

    Checks manual overrides first, then falls back to pypopulation.
    Returns 1 for unknown codes (prevents division by zero in per-capita math).

    Args:
        iso: ISO 3166-1 alpha-2 code (case-insensitive).

    Returns:
        Population count, or 1 if unknown.
    """
    iso_upper = iso.upper()
    override = _POPULATION_OVERRIDES.get(iso_upper)
    if override is not None:
        return override
    pop = pypopulation.get_population(iso_upper)
    return pop if pop is not None else 1
