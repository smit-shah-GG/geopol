"""Bilateral country relationship extraction from event pairs.

Extracts top-N bilateral relationships by grouping events on
(actor1_country, event_country) pairs. Both codes go through FIPS-to-ISO
conversion since GDELT stores FIPS 10-4 codes. Domestic events
(same source and target country) are excluded.

Canonical pair ordering (min_iso, max_iso) ensures US-RU and RU-US
aggregate into the same relationship.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Callable

logger = logging.getLogger(__name__)


def _extract_actor_country(actor_code: str) -> str | None:
    """Extract country prefix from a GDELT actor code.

    GDELT actor codes encode country as the first 3 characters when the
    code is longer than 3 chars (e.g., "USAGOV" -> "USA", "CHNCOP" -> "CHN").
    Codes of exactly 3 characters are often country alpha-3 codes directly.
    Codes of 2 characters may be FIPS country codes.

    Returns the raw code prefix (caller must convert via fips_to_iso).
    """
    if not actor_code:
        return None
    code = actor_code.strip().upper()
    if len(code) == 2:
        return code
    if len(code) >= 3:
        return code[:3]
    return None


def extract_bilateral_arcs(
    events: list[dict],
    fips_fn: Callable[[str], str | None],
    top_n: int = 50,
) -> list[dict]:
    """Extract top-N bilateral country relationships from event pairs.

    For each event, extracts:
      - Source country: from actor1_code country prefix
      - Target country: from country_iso (event location)

    Both codes pass through ``fips_fn`` for FIPS-to-ISO conversion.
    Domestic events (same source and target) are excluded.

    Pairs are canonicalized as (min_iso, max_iso) so US-CN and CN-US
    merge into a single bilateral relationship.

    Args:
        events: List of event dicts with keys: actor1_code, country_iso,
                goldstein_scale.
        fips_fn: Callable that converts a FIPS/ISO-3 code to ISO alpha-2.
                 Returns None for unmapped codes.
        top_n: Number of top bilateral pairs to return, ranked by event count.

    Returns:
        List of dicts: source_iso, target_iso, event_count, avg_goldstein.
        Sorted by event_count descending.
    """
    pair_stats: dict[tuple[str, str], dict] = defaultdict(
        lambda: {"count": 0, "goldstein_sum": 0.0}
    )

    for event in events:
        actor1_code = event.get("actor1_code")
        action_country = event.get("country_iso")

        if not actor1_code or not action_country:
            continue

        # Extract actor1 country prefix, then convert both through FIPS
        actor1_prefix = _extract_actor_country(actor1_code)
        if actor1_prefix is None:
            continue

        iso1 = fips_fn(actor1_prefix)
        iso2 = fips_fn(action_country) if len(action_country) != 2 else action_country
        # If country_iso is already a 2-letter ISO code (post-FIPS migration),
        # use it directly. Otherwise, convert.
        if iso2 is None:
            iso2 = action_country.upper() if len(action_country) == 2 else None

        if not iso1 or not iso2 or iso1 == iso2:
            continue  # Skip domestic events or unmapped codes

        # Canonical ordering: alphabetical
        pair = (min(iso1, iso2), max(iso1, iso2))
        goldstein = float(event.get("goldstein_scale") or 0.0)
        pair_stats[pair]["count"] += 1
        pair_stats[pair]["goldstein_sum"] += goldstein

    # Rank by event count, take top_n
    ranked = sorted(
        pair_stats.items(),
        key=lambda x: x[1]["count"],
        reverse=True,
    )[:top_n]

    result = [
        {
            "source_iso": pair[0],
            "target_iso": pair[1],
            "event_count": stats["count"],
            "avg_goldstein": round(
                stats["goldstein_sum"] / max(stats["count"], 1), 3
            ),
        }
        for pair, stats in ranked
    ]

    logger.info(
        "arc_builder: %d events -> %d bilateral pairs (top %d)",
        len(events),
        len(pair_stats),
        top_n,
    )
    return result
