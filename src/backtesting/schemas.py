"""
Internal dataclasses for backtesting run configuration and results.

These are pure-Python data structures used within the backtesting engine.
BacktestRunConfig serializes to/from JSON for ProcessPoolExecutor transport
(pickle-free argument passing).
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Optional


@dataclass
class BacktestRunConfig:
    """Configuration for a single backtest run.

    Frozen at run creation and passed to ProcessPoolExecutor workers
    as a JSON string (no unpickleable objects).

    Attributes:
        label: Human-readable run name (e.g. "Q1 2026 Full Evaluation").
        checkpoints: Model checkpoint mapping, e.g.
            {"tirgn": "tirgn_best.npz", "regcn": "regcn_jraph_best.npz"}.
        window_size_days: Width of each evaluation window in days.
        slide_step_days: Step size between consecutive windows.
        min_predictions_per_window: Minimum resolved predictions required
            for a window to be included in the evaluation.
        description: Optional long-form description.
        run_id: UUID assigned at creation time.
    """

    label: str
    checkpoints: dict[str, str]
    window_size_days: int = 14
    slide_step_days: int = 7
    min_predictions_per_window: int = 3
    description: Optional[str] = None
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def to_json(self) -> str:
        """Serialize to a plain JSON string for ProcessPoolExecutor transport."""
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_json(cls, data: str) -> BacktestRunConfig:
        """Reconstruct from a JSON string produced by to_json()."""
        parsed = json.loads(data)
        return cls(**parsed)


@dataclass
class WindowResult:
    """Metrics from a single evaluation window.

    One WindowResult is produced per (window, checkpoint) pair. Stored
    as a BacktestResult row in PostgreSQL.
    """

    window_start: datetime
    window_end: datetime
    prediction_start: datetime
    prediction_end: datetime
    checkpoint_name: str
    num_predictions: int
    brier_score: Optional[float] = None
    mrr: Optional[float] = None
    hits_at_1: Optional[float] = None
    hits_at_10: Optional[float] = None
    calibration_bins: Optional[dict[str, Any]] = None
    prediction_details: Optional[list[dict[str, Any]]] = None
    polymarket_brier: Optional[float] = None
    geopol_vs_pm_wins: Optional[int] = None
    pm_vs_geopol_wins: Optional[int] = None
    weight_snapshot: Optional[dict[str, float]] = None


@dataclass
class BacktestRunResult:
    """Aggregate result of a complete backtest run.

    Returned by BacktestRunner.run() after all windows have been
    evaluated (or after cancellation with partial results).
    """

    run_id: str
    status: str  # completed | cancelled | failed
    windows: list[WindowResult] = field(default_factory=list)
    aggregate_brier: Optional[float] = None
    aggregate_mrr: Optional[float] = None
    vs_polymarket_record: Optional[dict[str, int]] = None
    error_message: Optional[str] = None
