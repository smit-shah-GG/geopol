"""Unified training logger: TensorBoard always-on, W&B optional.

Provides a single abstraction for logging training metrics, text, and
system stats. TensorBoard is the primary backend (via tensorboardX,
which has no PyTorch dependency). Weights & Biases is enabled only
when WANDB_API_KEY is present in the environment AND the wandb package
is installed.

Metric naming convention:
    train/loss              -- per-epoch training loss
    train/lr                -- learning rate
    eval/mrr               -- validation MRR
    eval/hits_at_1          -- Hits@1
    eval/hits_at_3          -- Hits@3
    eval/hits_at_10         -- Hits@10
    system/vram_used_mb     -- GPU memory usage
    system/epoch_duration_s -- wall-clock time per epoch
    system/total_params     -- logged once at step 0
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from types import TracebackType

logger = logging.getLogger(__name__)


class TrainingLogger:
    """Unified TensorBoard + optional W&B logger.

    Always writes to TensorBoard via tensorboardX.  If ``WANDB_API_KEY``
    is set in the environment AND wandb is importable, metrics are also
    forwarded to Weights & Biases.  All wandb calls are guarded by
    try/except so a wandb failure never crashes training.
    """

    def __init__(
        self,
        logdir: str | Path,
        project: str = "geopol-tkg",
        run_name: str | None = None,
        config: dict | None = None,
    ) -> None:
        from tensorboardX import SummaryWriter

        self._logdir = Path(logdir)
        self._logdir.mkdir(parents=True, exist_ok=True)

        self.writer = SummaryWriter(str(self._logdir))
        logger.info("TensorBoard writer initialized at %s", self._logdir)

        # Attempt W&B init
        self._wandb_run = None
        api_key = os.environ.get("WANDB_API_KEY")
        if api_key:
            try:
                import wandb  # type: ignore[import-untyped]

                self._wandb_run = wandb.init(
                    project=project,
                    name=run_name,
                    config=config,
                )
                logger.info("W&B run initialized: %s/%s", project, run_name)
            except Exception as exc:
                logger.info(
                    "W&B init failed (%s), TensorBoard only", exc
                )
                self._wandb_run = None
        else:
            logger.info("W&B not configured (WANDB_API_KEY unset), TensorBoard only")

    # ------------------------------------------------------------------
    # Metric logging
    # ------------------------------------------------------------------

    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        """Log scalar metrics to TensorBoard and optionally W&B.

        Args:
            metrics: Mapping of metric name to scalar value.
            step: Global step (typically epoch number).
        """
        for key, value in metrics.items():
            self.writer.add_scalar(key, value, step)

        if self._wandb_run is not None:
            try:
                import wandb  # type: ignore[import-untyped]

                wandb.log(metrics, step=step)
            except Exception:
                pass  # Never crash training for a logging failure

    # ------------------------------------------------------------------
    # Text logging
    # ------------------------------------------------------------------

    def log_text(self, tag: str, text: str, step: int) -> None:
        """Log a text blob to TensorBoard.

        Args:
            tag: Text tag / category.
            text: Content string.
            step: Global step.
        """
        self.writer.add_text(tag, text, step)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Flush and close all logging backends."""
        self.writer.close()

        if self._wandb_run is not None:
            try:
                import wandb  # type: ignore[import-untyped]

                wandb.finish()
            except Exception:
                pass

    def __enter__(self) -> TrainingLogger:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.close()
