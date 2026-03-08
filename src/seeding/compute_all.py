"""Master orchestrator: compute all globe layer data in a single pass.

Reads from SQLite events + PostgreSQL advisories, computes:
  1. Baseline risk for ~195-250 countries (GDELT + ACLED + advisory + Goldstein)
  2. H3 heatmap hex bins from geocoded events
  3. Bilateral country arcs from event actor pairs
  4. Risk deltas (7-day change) by comparing against previous baseline

Writes everything to PostgreSQL in a single transaction (full table replace).
Called hourly by the APScheduler heavy job via heavy_runner.run_baseline_risk().
"""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime, timedelta, timezone

from sqlalchemy import delete, select

from src.db.models import (
    BaselineCountryRisk,
    CountryArc,
    HeatmapHexbin,
    RiskDelta,
    TravelAdvisory,
)
from src.seeding.arc_builder import extract_bilateral_arcs
from src.seeding.baseline_risk import compute_baseline_risk, decay_weight
from src.seeding.fips import fips_to_iso, get_sovereign_isos
from src.seeding.heatmap_binner import bin_events_to_h3
from src.seeding.population import get_population
from src.seeding.risk_delta import compute_risk_deltas

logger = logging.getLogger(__name__)

# Disputed territories -- scored independently but flagged in API response
_DISPUTED_ISOS = frozenset({"XK", "TW", "PS", "EH"})

# Decay windows
_BASELINE_DECAY_DAYS = 90
_HEATMAP_DECAY_DAYS = 30


def _load_events_from_sqlite(days: int) -> list[dict]:
    """Load recent events from SQLite within the given day window.

    Returns list of dicts with all relevant fields for downstream computations.
    Uses a fresh EventStorage connection (safe for subprocess context).
    """
    from src.database.storage import EventStorage

    storage = EventStorage()
    cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    today = datetime.now().strftime("%Y-%m-%d")

    events = storage.get_events(start_date=cutoff, end_date=today)

    return [
        {
            "country_iso": e.country_iso,
            "actor1_code": e.actor1_code,
            "actor2_code": e.actor2_code,
            "goldstein_scale": e.goldstein_scale,
            "num_mentions": e.num_mentions,
            "event_date": e.event_date,
            "source": e.source,
            "lat": e.lat,
            "lon": e.lon,
        }
        for e in events
    ]


async def _load_advisories() -> dict[str, int]:
    """Load travel advisories from PostgreSQL.

    Returns dict of {country_iso: max_level} (max across all sources per country).
    """
    from src.db import postgres

    if postgres.async_session_factory is None:
        postgres.init_db()

    advisory_map: dict[str, int] = {}

    async with postgres.async_session_factory() as session:
        result = await session.execute(select(TravelAdvisory))
        rows = result.scalars().all()

        for row in rows:
            iso = row.country_iso.upper()
            level = row.level
            # Take max level across sources per country
            if iso not in advisory_map or level > advisory_map[iso]:
                advisory_map[iso] = level

    logger.info("Loaded %d advisory entries from PostgreSQL", len(advisory_map))
    return advisory_map


async def _load_previous_risks() -> dict[str, float]:
    """Load the current baseline risk scores from PostgreSQL (for delta computation).

    These represent the "previous" scores before the current recompute.
    """
    from src.db import postgres

    if postgres.async_session_factory is None:
        postgres.init_db()

    risks: dict[str, float] = {}
    async with postgres.async_session_factory() as session:
        result = await session.execute(select(BaselineCountryRisk))
        for row in result.scalars().all():
            risks[row.country_iso] = row.baseline_risk

    return risks


def _aggregate_country_stats(
    events: list[dict],
    window_days: int,
) -> dict[str, dict]:
    """Aggregate per-country statistics from events.

    Groups events by country_iso, computing decay-weighted counts and
    Goldstein sums. Separates GDELT and ACLED sources.

    Returns dict of {country_iso: {gdelt_count, acled_count, goldstein_sum,
    goldstein_weight_sum}} where counts are decay-weighted.
    """
    now = datetime.now(timezone.utc)
    stats: dict[str, dict] = defaultdict(
        lambda: {
            "gdelt_count": 0.0,
            "acled_count": 0.0,
            "goldstein_sum": 0.0,
            "goldstein_weight_sum": 0.0,
        }
    )

    for event in events:
        iso = event.get("country_iso")
        if not iso or len(iso) != 2:
            continue

        iso = iso.upper()

        # Compute age-based decay weight
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

        w = decay_weight(age_days, half_life=30.0)
        source = event.get("source", "gdelt")

        if source == "acled":
            stats[iso]["acled_count"] += w
        else:
            stats[iso]["gdelt_count"] += w

        goldstein = float(event.get("goldstein_scale") or 0.0)
        stats[iso]["goldstein_sum"] += goldstein * w
        stats[iso]["goldstein_weight_sum"] += w

    return dict(stats)


async def compute_all_layers() -> dict[str, int]:
    """Compute all globe layer data in a single pass.

    Orchestrates:
      1. Load events from SQLite (90-day window)
      2. Load advisories from PostgreSQL
      3. Load previous baseline risks (for delta computation)
      4. Compute baseline risk for each sovereign country
      5. Compute H3 heatmap hex bins from geocoded events
      6. Compute bilateral arcs from event pairs
      7. Compute risk deltas
      8. Write everything to PostgreSQL (full table replace)

    Returns:
        Dict with counts: countries, hexbins, arcs, deltas.
    """
    from src.db import postgres

    if postgres.async_session_factory is None:
        postgres.init_db()

    now = datetime.now(timezone.utc)
    logger.info("compute_all_layers: starting full globe data computation")

    # Step 1: Load events from SQLite
    events = _load_events_from_sqlite(days=_BASELINE_DECAY_DAYS)
    logger.info("Loaded %d events from SQLite (90-day window)", len(events))

    # Step 2: Load advisory data from PostgreSQL
    advisory_map = await _load_advisories()

    # Step 3: Load previous baseline risks for delta computation
    previous_risks = await _load_previous_risks()

    # Step 4: Aggregate per-country stats
    country_stats = _aggregate_country_stats(events, _BASELINE_DECAY_DAYS)

    # Step 5: Compute baseline risk for each sovereign country
    sovereign_isos = get_sovereign_isos()
    risk_rows: list[BaselineCountryRisk] = []
    current_risks: dict[str, float] = {}

    for iso in sorted(sovereign_isos):
        stats = country_stats.get(iso, {
            "gdelt_count": 0.0,
            "acled_count": 0.0,
            "goldstein_sum": 0.0,
            "goldstein_weight_sum": 0.0,
        })

        gdelt_count = int(round(stats["gdelt_count"]))
        acled_count = int(round(stats["acled_count"]))
        # ACLED fatalities not available in schema -- use event count as proxy
        acled_fatalities = 0
        advisory_level = advisory_map.get(iso, 1)
        population = get_population(iso)

        # Weighted average Goldstein score
        total_weight = stats["goldstein_weight_sum"]
        avg_goldstein = (
            stats["goldstein_sum"] / total_weight if total_weight > 0 else 0.0
        )

        risk_score = compute_baseline_risk(
            gdelt_event_count=gdelt_count,
            population=population,
            acled_fatalities=acled_fatalities,
            acled_event_count=acled_count,
            advisory_level=advisory_level,
            avg_goldstein=avg_goldstein,
        )

        # Compute component scores for audit (same logic as compute_baseline_risk)
        pop_millions = max(population, 1) / 1_000_000
        gdelt_per_capita = gdelt_count / pop_millions
        gdelt_score = min(100.0, gdelt_per_capita * 2.0)
        acled_score = min(100.0, acled_fatalities * 5.0 + acled_count * 2.0)
        advisory_score_map = {1: 10.0, 2: 35.0, 3: 60.0, 4: 90.0}
        advisory_score = advisory_score_map.get(advisory_level, 10.0)
        goldstein_score = max(0.0, min(100.0, (10.0 - avg_goldstein) * 5.0))

        row = BaselineCountryRisk(
            country_iso=iso,
            baseline_risk=risk_score,
            gdelt_score=round(gdelt_score, 1),
            acled_score=round(acled_score, 1),
            advisory_score=round(advisory_score, 1),
            goldstein_score=round(goldstein_score, 1),
            advisory_level=advisory_level,
            gdelt_event_count=gdelt_count,
            acled_event_count=acled_count,
            disputed=iso in _DISPUTED_ISOS,
            computed_at=now,
        )
        risk_rows.append(row)
        current_risks[iso] = risk_score

    logger.info("Computed baseline risk for %d countries", len(risk_rows))

    # Step 6: Compute heatmap hex bins (30-day window for heatmap, events
    # already loaded for 90 days -- filter to 30 for heatmap)
    cutoff_30d = (now - timedelta(days=_HEATMAP_DECAY_DAYS)).strftime("%Y-%m-%d")
    heatmap_events = [
        e for e in events
        if (e.get("event_date") or "") >= cutoff_30d
    ]
    hex_bins = bin_events_to_h3(heatmap_events, decay_half_life=float(_HEATMAP_DECAY_DAYS))
    hexbin_rows = [
        HeatmapHexbin(
            h3_index=hb["h3_index"],
            weight=hb["weight"],
            event_count=hb["event_count"],
            computed_at=now,
        )
        for hb in hex_bins
    ]

    # Step 7: Compute bilateral arcs from events (90-day window)
    arcs = extract_bilateral_arcs(events, fips_fn=fips_to_iso, top_n=50)
    arc_rows = [
        CountryArc(
            source_iso=a["source_iso"],
            target_iso=a["target_iso"],
            event_count=a["event_count"],
            avg_goldstein=a["avg_goldstein"],
            computed_at=now,
        )
        for a in arcs
    ]

    # Step 8: Compute risk deltas
    deltas = compute_risk_deltas(current_risks, previous_risks, threshold=10.0)
    delta_rows = [
        RiskDelta(
            country_iso=d["country_iso"],
            current_risk=d["current_risk"],
            previous_risk=d["previous_risk"],
            delta=d["delta"],
            computed_at=now,
        )
        for d in deltas
    ]

    # Step 9: Write everything to PostgreSQL in a single transaction
    async with postgres.async_session_factory() as session:
        # Full table replace: DELETE all rows, then INSERT new ones
        await session.execute(delete(RiskDelta))
        await session.execute(delete(CountryArc))
        await session.execute(delete(HeatmapHexbin))
        await session.execute(delete(BaselineCountryRisk))

        session.add_all(risk_rows)
        session.add_all(hexbin_rows)
        session.add_all(arc_rows)
        session.add_all(delta_rows)

        await session.commit()

    counts = {
        "countries": len(risk_rows),
        "hexbins": len(hexbin_rows),
        "arcs": len(arc_rows),
        "deltas": len(delta_rows),
    }
    logger.info(
        "compute_all_layers: complete -- %d countries, %d hexbins, %d arcs, %d deltas",
        counts["countries"],
        counts["hexbins"],
        counts["arcs"],
        counts["deltas"],
    )
    return counts
