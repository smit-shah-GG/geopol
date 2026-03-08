"""H3 hexagonal binning of geocoded events for the globe heatmap layer.

Aggregates events with lat/lon into H3 hex cells at a configurable resolution.
Each hex cell accumulates a time-decayed, severity-weighted score from its
constituent events. Events without coordinates are silently skipped.

Resolution 3 yields ~41,162 cells globally at ~9,229 km^2 each -- a reasonable
density for a world-scale choropleth without drowning the frontend in geometry.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime, timezone

import h3

from src.seeding.baseline_risk import decay_weight

logger = logging.getLogger(__name__)

# H3 resolution 3: ~9,229 km^2 per hex, 41,162 total cells globally
HEATMAP_RESOLUTION = 3


def bin_events_to_h3(
    events: list[dict],
    resolution: int = HEATMAP_RESOLUTION,
    decay_half_life: float = 30.0,
) -> list[dict]:
    """Aggregate geocoded events into H3 hex bins with time-decayed weights.

    Each event contributes: severity * mentions_norm * decay_weight, where
      - severity = abs(goldstein_scale) / 10.0  (0-1 range)
      - mentions_norm = min(num_mentions, 100) / 100.0  (0-1 range)
      - decay_weight = exponential decay based on event age

    Events missing lat or lon are silently skipped.

    Args:
        events: List of event dicts with keys: lat, lon, goldstein_scale,
                num_mentions, event_date.
        resolution: H3 resolution level (default 3).
        decay_half_life: Half-life in days for exponential decay weighting.

    Returns:
        List of dicts with keys: h3_index (str), weight (float), event_count (int).
    """
    now = datetime.now(timezone.utc)
    bins: dict[str, dict] = defaultdict(lambda: {"weight": 0.0, "count": 0})
    skipped = 0

    for event in events:
        lat = event.get("lat")
        lon = event.get("lon")
        if lat is None or lon is None:
            skipped += 1
            continue

        try:
            h3_index = h3.latlng_to_cell(float(lat), float(lon), resolution)
        except Exception:
            skipped += 1
            continue

        # Severity: normalize abs(goldstein) to 0-1 range
        goldstein = event.get("goldstein_scale") or 0.0
        severity = abs(float(goldstein)) / 10.0

        # Mentions: normalize to 0-1, capped at 100
        mentions = event.get("num_mentions") or 1
        mentions_norm = min(int(mentions), 100) / 100.0

        # Time decay
        event_date_str = event.get("event_date", "")
        try:
            if "T" in str(event_date_str):
                event_dt = datetime.fromisoformat(str(event_date_str))
            else:
                event_dt = datetime.strptime(str(event_date_str)[:10], "%Y-%m-%d")
            if event_dt.tzinfo is None:
                event_dt = event_dt.replace(tzinfo=timezone.utc)
            age_days = max(0.0, (now - event_dt).total_seconds() / 86400.0)
        except (ValueError, TypeError):
            age_days = 0.0

        weight = severity * mentions_norm * decay_weight(age_days, decay_half_life)
        bins[h3_index]["weight"] += weight
        bins[h3_index]["count"] += 1

    if skipped > 0:
        logger.debug("heatmap_binner: skipped %d events without lat/lon", skipped)

    result = [
        {
            "h3_index": idx,
            "weight": round(data["weight"], 6),
            "event_count": data["count"],
        }
        for idx, data in bins.items()
    ]

    logger.info(
        "heatmap_binner: %d events -> %d hex bins (resolution=%d, %d skipped)",
        len(events),
        len(result),
        resolution,
        skipped,
    )
    return result
