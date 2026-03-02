"""
Hierarchical weight loader with TTL-based in-memory caching.

Resolves the ensemble alpha weight for a given prediction through a
five-level fallback chain:

    1. CAMEO root code (e.g. "14") -- most specific, from calibration_weights
    2. Super-category (e.g. "super:verbal_conflict") -- aggregated calibration_weights
    3. Global ("global") -- all-data calibration_weights
    4. Cold-start super-category prior -- from priors.py, no data needed
    5. Cold-start global prior (0.58) -- absolute fallback

The loader is designed as a long-lived singleton: instantiated once at
API server startup and reused across all prediction requests. An
in-memory cache with configurable TTL (default 5 min) prevents
per-request PostgreSQL queries while ensuring the weekly calibration
results propagate within a bounded window.

Thread safety: The loader is async and uses a single _loaded_at timestamp
for cache invalidation. Concurrent callers may trigger a redundant reload
(benign -- not a correctness issue, just a wasted query).
"""

from __future__ import annotations

import logging
import time
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.calibration.priors import (
    COLD_START_PRIORS,
    infer_super_category,
)
from src.db.models import CalibrationWeight

logger = logging.getLogger(__name__)


class WeightLoader:
    """Load and resolve calibration weights with hierarchical fallback."""

    def __init__(
        self,
        async_session_factory: Any,
        min_samples: int = 10,
        cache_ttl_seconds: int = 300,
    ) -> None:
        """
        Args:
            async_session_factory: Callable returning an async context
                manager yielding AsyncSession (e.g. async_sessionmaker).
            min_samples: Minimum sample_size required to trust a
                calibration weight. Weights below this threshold are
                treated as absent and resolution falls through.
            cache_ttl_seconds: Time-to-live for the in-memory weight
                cache, in seconds. Default 300 (5 minutes).
        """
        self._session_factory = async_session_factory
        self._min_samples = min_samples
        self._cache_ttl = cache_ttl_seconds

        # In-memory cache: cameo_code -> CalibrationWeight-like dict
        self._weights: dict[str, _CachedWeight] = {}
        self._loaded_at: float = 0.0  # epoch seconds

    @property
    def weights_loaded(self) -> bool:
        """True if weights have been loaded at least once."""
        return self._loaded_at > 0.0

    async def load_weights(self) -> None:
        """Load all CalibrationWeight rows from PostgreSQL.

        Populates the in-memory cache. Also registers cold-start priors
        as fallback entries (they never expire but are only used when
        no database weight qualifies).
        """
        async with self._session_factory() as session:
            stmt = select(CalibrationWeight)
            result = await session.execute(stmt)
            rows = result.scalars().all()

        self._weights.clear()

        for row in rows:
            self._weights[row.cameo_code] = _CachedWeight(
                alpha=row.alpha,
                sample_size=row.sample_size,
                source="database",
            )

        self._loaded_at = time.monotonic()

        logger.info(
            "Loaded %d calibration weights from database", len(rows),
        )

    async def _ensure_loaded(self) -> None:
        """Reload weights if cache has expired or never loaded."""
        elapsed = time.monotonic() - self._loaded_at
        if self._loaded_at == 0.0 or elapsed > self._cache_ttl:
            await self.load_weights()

    async def resolve_alpha(
        self,
        cameo_root_code: str | None = None,
        keyword_category: str | None = None,
    ) -> float:
        """Resolve the ensemble alpha weight through hierarchical fallback.

        Resolution order:
            1. CAMEO root code (if present and sample_size >= min_samples)
            2. Super-category key "super:{cat}" (if resolvable and qualified)
            3. Global key "global" (if qualified)
            4. Cold-start prior for the super-category
            5. Cold-start global prior (0.58)

        Args:
            cameo_root_code: Two-digit CAMEO root code, or None.
            keyword_category: Heuristic category from EnsemblePredictor, or None.

        Returns:
            Alpha value (LLM weight) in [0.0, 1.0].
        """
        info = await self.get_weight_info(cameo_root_code, keyword_category)
        return info["alpha"]

    async def get_weight_info(
        self,
        cameo_root_code: str | None = None,
        keyword_category: str | None = None,
    ) -> dict[str, Any]:
        """Resolve alpha with full diagnostic metadata.

        Returns:
            Dict with keys:
                alpha: float -- resolved weight
                resolution_level: str -- one of "cameo", "super", "global",
                    "prior_super", "prior_global"
                sample_size: int -- sample size backing the weight (0 for priors)
        """
        await self._ensure_loaded()

        # Level 1: CAMEO root code
        if cameo_root_code is not None:
            code = cameo_root_code.strip().zfill(2)
            w = self._weights.get(code)
            if w is not None and w.sample_size >= self._min_samples:
                return {
                    "alpha": w.alpha,
                    "resolution_level": "cameo",
                    "sample_size": w.sample_size,
                }

        # Level 2: Super-category
        super_cat = infer_super_category(cameo_root_code, keyword_category)
        if super_cat is not None:
            super_key = f"super:{super_cat}"
            w = self._weights.get(super_key)
            if w is not None and w.sample_size >= self._min_samples:
                return {
                    "alpha": w.alpha,
                    "resolution_level": "super",
                    "sample_size": w.sample_size,
                }

        # Level 3: Global
        w = self._weights.get("global")
        if w is not None and w.sample_size >= self._min_samples:
            return {
                "alpha": w.alpha,
                "resolution_level": "global",
                "sample_size": w.sample_size,
            }

        # Level 4: Cold-start super-category prior
        if super_cat is not None and super_cat in COLD_START_PRIORS:
            return {
                "alpha": COLD_START_PRIORS[super_cat],
                "resolution_level": "prior_super",
                "sample_size": 0,
            }

        # Level 5: Cold-start global prior (absolute fallback)
        return {
            "alpha": COLD_START_PRIORS["global"],
            "resolution_level": "prior_global",
            "sample_size": 0,
        }


class _CachedWeight:
    """Lightweight in-memory representation of a calibration weight."""

    __slots__ = ("alpha", "sample_size", "source")

    def __init__(self, alpha: float, sample_size: int, source: str) -> None:
        self.alpha = alpha
        self.sample_size = sample_size
        self.source = source

    def __repr__(self) -> str:
        return (
            f"_CachedWeight(alpha={self.alpha:.4f}, "
            f"n={self.sample_size}, src={self.source!r})"
        )
