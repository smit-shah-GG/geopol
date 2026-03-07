"""
Calibration weight time-travel for look-ahead bias prevention.

Reconstructs the per-CAMEO calibration weight state as it existed at a
given point in time by querying calibration_weight_history. Codes with
no history before the target date fall back to cold-start priors from
priors.py -- the same default behavior the production system used
before calibration was deployed.

The returned dict is suitable for direct injection into a WeightLoader's
_weights cache, bypassing the DB query and TTL-based cache invalidation.
"""

from __future__ import annotations

import logging
from datetime import datetime

from sqlalchemy import and_, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.calibration.priors import CAMEO_TO_SUPER, COLD_START_PRIORS
from src.db.models import CalibrationWeightHistory

logger = logging.getLogger(__name__)


async def snapshot_calibration_weights(
    session: AsyncSession,
    as_of: datetime,
) -> dict[str, float]:
    """Reconstruct calibration weights as they existed at ``as_of``.

    Resolution strategy:
      1. Query calibration_weight_history for the latest entry per
         cameo_code where computed_at <= as_of AND auto_applied = True.
      2. Fall back to cold-start priors for codes with no qualifying
         history entry.

    The result includes all CAMEO root codes (01-20), super-category
    keys (super:verbal_coop, etc.), and the global key.

    Args:
        session: Async SQLAlchemy session (not committed by this function).
        as_of: Point-in-time for weight reconstruction.

    Returns:
        Dict of cameo_code -> alpha, suitable for injection into
        WeightLoader._weights cache. Covers all standard codes with
        either historical or cold-start values.
    """
    # Subquery: max computed_at per cameo_code where computed_at <= as_of
    # and auto_applied = True.
    latest_per_code = (
        select(
            CalibrationWeightHistory.cameo_code,
            func.max(CalibrationWeightHistory.computed_at).label("max_computed"),
        )
        .where(
            and_(
                CalibrationWeightHistory.computed_at <= as_of,
                CalibrationWeightHistory.auto_applied.is_(True),
            )
        )
        .group_by(CalibrationWeightHistory.cameo_code)
        .subquery()
    )

    # Join back to get the alpha for each (cameo_code, max_computed) pair.
    stmt = (
        select(
            CalibrationWeightHistory.cameo_code,
            CalibrationWeightHistory.alpha,
        )
        .join(
            latest_per_code,
            and_(
                CalibrationWeightHistory.cameo_code == latest_per_code.c.cameo_code,
                CalibrationWeightHistory.computed_at == latest_per_code.c.max_computed,
            ),
        )
        .where(CalibrationWeightHistory.auto_applied.is_(True))
    )

    result = await session.execute(stmt)
    rows = result.all()

    # Build snapshot from history.
    snapshot: dict[str, float] = {}
    for cameo_code, alpha in rows:
        snapshot[cameo_code] = float(alpha)

    # Fill in cold-start priors for codes with no history before as_of.
    # CAMEO root codes 01-20.
    for code in CAMEO_TO_SUPER:
        if code not in snapshot:
            super_cat = CAMEO_TO_SUPER[code]
            snapshot[code] = COLD_START_PRIORS.get(
                super_cat, COLD_START_PRIORS["global"]
            )

    # Super-category keys.
    for super_cat, prior in COLD_START_PRIORS.items():
        key = f"super:{super_cat}" if super_cat != "global" else "global"
        if key not in snapshot:
            snapshot[key] = prior

    # Global fallback.
    if "global" not in snapshot:
        snapshot["global"] = COLD_START_PRIORS["global"]

    history_count = len(rows)
    prior_count = len(snapshot) - history_count
    logger.info(
        "Weight snapshot at %s: %d from history, %d from cold-start priors",
        as_of.isoformat(),
        history_count,
        prior_count,
    )

    return snapshot
