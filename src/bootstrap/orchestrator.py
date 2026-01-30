"""
Stage orchestrator for bootstrap pipeline.

Executes stages in order, respecting checkpoint state and validating outputs.
"""

from __future__ import annotations

import logging
import time
from abc import abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Tuple, runtime_checkable

from .checkpoint import (
    CheckpointManager,
    ConsoleReporter,
    ProgressReporter,
    StageStatus,
    should_skip_stage,
)

logger = logging.getLogger(__name__)


@runtime_checkable
class Stage(Protocol):
    """Protocol defining the interface for pipeline stages."""

    @property
    def name(self) -> str:
        """Unique stage name."""
        ...

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the stage.

        Args:
            context: Shared context dict for passing data between stages

        Returns:
            Dict with execution statistics
        """
        ...

    def validate_output(self) -> Tuple[bool, str]:
        """
        Validate stage output exists and is valid.

        Returns:
            Tuple of (is_valid, reason_message)
        """
        ...

    def get_output_path(self) -> Optional[str]:
        """
        Get path to stage output.

        Returns:
            Path string or None if no persistent output
        """
        ...


# ProgressReporter protocol and ConsoleReporter implementation are now in checkpoint.py
# Re-exported above for backwards compatibility


class StageOrchestrator:
    """
    Orchestrates execution of pipeline stages.

    Handles:
    - Stage registration and ordering
    - Checkpoint-based resumption
    - Output validation
    - Progress reporting
    - Context passing between stages
    """

    def __init__(
        self,
        checkpoint_manager: CheckpointManager,
        reporter: Optional[ProgressReporter] = None,
    ):
        """
        Initialize orchestrator.

        Args:
            checkpoint_manager: CheckpointManager for state persistence
            reporter: ProgressReporter for console output
        """
        self.checkpoint = checkpoint_manager
        self.reporter = reporter or ConsoleReporter()
        self._stages: List[Stage] = []
        self._context: Dict[str, Any] = {}

    def register_stage(self, stage: Stage) -> None:
        """
        Register a stage for execution.

        Args:
            stage: Stage instance to register
        """
        if not isinstance(stage, Stage):
            raise TypeError(f"Stage must implement Stage protocol, got {type(stage)}")
        self._stages.append(stage)
        logger.debug(f"Registered stage: {stage.name}")

    def should_skip(self, stage: Stage) -> Tuple[bool, str]:
        """
        Determine if stage should be skipped using dual idempotency check.

        Delegates to should_skip_stage() which checks:
        1. Checkpoint status is COMPLETED
        2. Output validation passes

        Args:
            stage: Stage to check

        Returns:
            Tuple of (should_skip, reason)
        """
        state = self.checkpoint.load()

        # Use stage's validate_output method as the validator callable
        skip, reason = should_skip_stage(
            stage_name=stage.name,
            state=state,
            validator=stage.validate_output,
        )

        # If checkpoint was stale (marked complete but output invalid), reset it
        if not skip and stage.name in state.stages:
            stage_state = state.stages[stage.name]
            if stage_state.status == StageStatus.COMPLETED:
                # should_skip_stage returned False for a COMPLETED stage = stale checkpoint
                self.checkpoint.reset_stage(stage.name)

        return skip, reason

    def execute_stage(self, stage: Stage) -> bool:
        """
        Execute a single stage.

        Args:
            stage: Stage to execute

        Returns:
            True if successful, False if failed
        """
        start_time = time.time()

        # Mark running
        self.checkpoint.mark_running(stage.name)
        self.reporter.stage_start(stage.name)

        try:
            # Execute stage
            stats = stage.run(self._context)
            duration = time.time() - start_time
            stats["duration_seconds"] = round(duration, 2)

            # Validate output
            is_valid, validation_msg = stage.validate_output()
            if not is_valid:
                raise RuntimeError(f"Output validation failed: {validation_msg}")

            # Mark completed
            output_path = stage.get_output_path()
            self.checkpoint.mark_completed(stage.name, output_path, stats)
            self.reporter.stage_complete(stage.name, duration, stats)

            return True

        except Exception as e:
            duration = time.time() - start_time
            error_msg = str(e)
            self.checkpoint.mark_failed(stage.name, error_msg)
            self.reporter.stage_error(stage.name, error_msg)
            logger.exception(f"Stage '{stage.name}' failed after {duration:.1f}s")
            return False

    def run_all(self, force_stages: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Execute all registered stages in order.

        Args:
            force_stages: Optional list of stage names to force re-run

        Returns:
            Dict with execution summary
        """
        force_stages = force_stages or []
        pipeline_start = time.time()

        summary = {
            "total_stages": len(self._stages),
            "completed": 0,
            "skipped": 0,
            "failed": 0,
            "stages": [],
        }

        for stage in self._stages:
            # Check if we should force this stage
            if stage.name in force_stages:
                self.checkpoint.reset_stage(stage.name)

            # Check if we should skip
            should_skip, skip_reason = self.should_skip(stage)
            if should_skip:
                self.reporter.stage_skipped(stage.name, skip_reason)
                summary["skipped"] += 1
                summary["stages"].append({"name": stage.name, "status": "skipped"})
                continue

            # Execute
            success = self.execute_stage(stage)

            if success:
                summary["completed"] += 1
                summary["stages"].append({"name": stage.name, "status": "completed"})
            else:
                summary["failed"] += 1
                summary["stages"].append({"name": stage.name, "status": "failed"})
                # Stop on failure
                break

        pipeline_duration = time.time() - pipeline_start
        summary["duration_seconds"] = round(pipeline_duration, 2)

        if summary["failed"] == 0:
            self.checkpoint.mark_pipeline_complete()
            self.reporter.pipeline_complete(len(self._stages), pipeline_duration)

        return summary

    def dry_run(self) -> List[Dict[str, str]]:
        """
        Show what stages would run without executing.

        Returns:
            List of dicts with stage name and action
        """
        plan = []
        for stage in self._stages:
            should_skip, skip_reason = self.should_skip(stage)
            if should_skip:
                plan.append({
                    "name": stage.name,
                    "action": "skip",
                    "reason": skip_reason,
                })
            else:
                plan.append({
                    "name": stage.name,
                    "action": "run",
                    "reason": "pending or incomplete",
                })
        return plan

    def get_context(self) -> Dict[str, Any]:
        """Get shared context dict."""
        return self._context

    def set_context(self, key: str, value: Any) -> None:
        """Set value in shared context."""
        self._context[key] = value
