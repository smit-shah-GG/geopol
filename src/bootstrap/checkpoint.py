"""
Checkpoint state management for bootstrap pipeline.

Provides atomic state persistence to allow pipeline resumption after
interruption or failure.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Callable, Dict, Optional, Protocol, Tuple, runtime_checkable

logger = logging.getLogger(__name__)


class StageStatus(Enum):
    """Status of a pipeline stage."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class StageState:
    """State of a single pipeline stage."""
    name: str
    status: StageStatus = StageStatus.PENDING
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    output_path: Optional[str] = None
    error: Optional[str] = None
    stats: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to serializable dictionary."""
        return {
            "name": self.name,
            "status": self.status.value,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "output_path": self.output_path,
            "error": self.error,
            "stats": self.stats,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> StageState:
        """Create StageState from dictionary."""
        return cls(
            name=data["name"],
            status=StageStatus(data.get("status", "pending")),
            started_at=data.get("started_at"),
            completed_at=data.get("completed_at"),
            output_path=data.get("output_path"),
            error=data.get("error"),
            stats=data.get("stats", {}),
        )


@dataclass
class BootstrapState:
    """Complete bootstrap pipeline state."""
    stages: Dict[str, StageState] = field(default_factory=dict)
    last_updated: Optional[str] = None
    pipeline_started: Optional[str] = None
    pipeline_completed: Optional[str] = None

    def to_dict(self) -> Dict:
        """Convert to serializable dictionary."""
        return {
            "stages": {name: state.to_dict() for name, state in self.stages.items()},
            "last_updated": self.last_updated,
            "pipeline_started": self.pipeline_started,
            "pipeline_completed": self.pipeline_completed,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> BootstrapState:
        """Create BootstrapState from dictionary."""
        stages = {
            name: StageState.from_dict(state_data)
            for name, state_data in data.get("stages", {}).items()
        }
        return cls(
            stages=stages,
            last_updated=data.get("last_updated"),
            pipeline_started=data.get("pipeline_started"),
            pipeline_completed=data.get("pipeline_completed"),
        )


class CheckpointManager:
    """
    Manages bootstrap pipeline checkpoint state.

    Uses atomic file writes (tempfile + os.replace) to prevent
    state corruption on interrupt.
    """

    def __init__(self, state_file: Optional[Path] = None):
        """
        Initialize checkpoint manager.

        Args:
            state_file: Path to state file (default: data/bootstrap_state.json)
        """
        self.state_file = state_file or Path("data/bootstrap_state.json")
        self.state_file = Path(self.state_file)
        self._state: Optional[BootstrapState] = None

    def load(self) -> BootstrapState:
        """
        Load state from file.

        Returns:
            BootstrapState (empty if file doesn't exist)
        """
        if self._state is not None:
            return self._state

        if not self.state_file.exists():
            logger.info(f"No existing state file at {self.state_file}, starting fresh")
            self._state = BootstrapState()
            return self._state

        try:
            with open(self.state_file, "r") as f:
                data = json.load(f)
            self._state = BootstrapState.from_dict(data)
            logger.info(f"Loaded state from {self.state_file}")
            return self._state
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Corrupted state file, starting fresh: {e}")
            self._state = BootstrapState()
            return self._state

    def save(self, state: BootstrapState) -> None:
        """
        Save state to file atomically.

        Uses tempfile + os.replace pattern to prevent corruption.

        Args:
            state: BootstrapState to save
        """
        state.last_updated = datetime.utcnow().isoformat()
        self._state = state

        # Ensure parent directory exists
        self.state_file.parent.mkdir(parents=True, exist_ok=True)

        # Atomic write: write to temp file, then replace
        fd, temp_path = tempfile.mkstemp(
            dir=self.state_file.parent,
            prefix=".bootstrap_state_",
            suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(state.to_dict(), f, indent=2)

            # Atomic replace
            os.replace(temp_path, self.state_file)
            logger.debug(f"Saved state to {self.state_file}")

        except Exception as e:
            # Clean up temp file on failure
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            raise RuntimeError(f"Failed to save checkpoint: {e}") from e

    def mark_running(self, stage_name: str) -> None:
        """
        Mark stage as running.

        Args:
            stage_name: Name of the stage
        """
        state = self.load()

        if stage_name not in state.stages:
            state.stages[stage_name] = StageState(name=stage_name)

        stage = state.stages[stage_name]
        stage.status = StageStatus.RUNNING
        stage.started_at = datetime.utcnow().isoformat()
        stage.error = None

        if state.pipeline_started is None:
            state.pipeline_started = datetime.utcnow().isoformat()

        self.save(state)
        logger.info(f"Stage '{stage_name}' marked as RUNNING")

    def mark_completed(self, stage_name: str, output_path: Optional[str] = None,
                       stats: Optional[Dict] = None) -> None:
        """
        Mark stage as completed.

        Args:
            stage_name: Name of the stage
            output_path: Path to stage output
            stats: Stage execution statistics
        """
        state = self.load()

        if stage_name not in state.stages:
            state.stages[stage_name] = StageState(name=stage_name)

        stage = state.stages[stage_name]
        stage.status = StageStatus.COMPLETED
        stage.completed_at = datetime.utcnow().isoformat()
        stage.output_path = output_path
        stage.stats = stats or {}
        stage.error = None

        self.save(state)
        logger.info(f"Stage '{stage_name}' marked as COMPLETED")

    def mark_failed(self, stage_name: str, error: str) -> None:
        """
        Mark stage as failed.

        Args:
            stage_name: Name of the stage
            error: Error message
        """
        state = self.load()

        if stage_name not in state.stages:
            state.stages[stage_name] = StageState(name=stage_name)

        stage = state.stages[stage_name]
        stage.status = StageStatus.FAILED
        stage.completed_at = datetime.utcnow().isoformat()
        stage.error = error

        self.save(state)
        logger.error(f"Stage '{stage_name}' marked as FAILED: {error}")

    def get_status(self, stage_name: str) -> StageStatus:
        """
        Get status of a stage.

        Args:
            stage_name: Name of the stage

        Returns:
            StageStatus (PENDING if stage not found)
        """
        state = self.load()
        if stage_name in state.stages:
            return state.stages[stage_name].status
        return StageStatus.PENDING

    def get_stage_state(self, stage_name: str) -> Optional[StageState]:
        """
        Get full state of a stage.

        Args:
            stage_name: Name of the stage

        Returns:
            StageState or None if not found
        """
        state = self.load()
        return state.stages.get(stage_name)

    def reset_stage(self, stage_name: str) -> None:
        """
        Reset a stage to PENDING status.

        Args:
            stage_name: Name of the stage
        """
        state = self.load()
        if stage_name in state.stages:
            state.stages[stage_name] = StageState(name=stage_name)
            self.save(state)
            logger.info(f"Stage '{stage_name}' reset to PENDING")

    def reset_all(self) -> None:
        """Reset all stages to PENDING status."""
        state = self.load()
        for stage_name in state.stages:
            state.stages[stage_name] = StageState(name=stage_name)
        state.pipeline_started = None
        state.pipeline_completed = None
        self.save(state)
        logger.info("All stages reset to PENDING")

    def mark_pipeline_complete(self) -> None:
        """Mark the entire pipeline as complete."""
        state = self.load()
        state.pipeline_completed = datetime.utcnow().isoformat()
        self.save(state)
        logger.info("Pipeline marked as COMPLETE")

    def get_summary(self) -> Dict:
        """
        Get summary of pipeline state.

        Returns:
            Dict with stage counts by status
        """
        state = self.load()
        summary = {
            "total_stages": len(state.stages),
            "pending": 0,
            "running": 0,
            "completed": 0,
            "failed": 0,
            "pipeline_started": state.pipeline_started,
            "pipeline_completed": state.pipeline_completed,
            "last_updated": state.last_updated,
        }

        for stage in state.stages.values():
            summary[stage.status.value] += 1

        return summary


# --- Progress Reporting ---


@runtime_checkable
class ProgressReporter(Protocol):
    """Protocol defining the interface for progress reporting."""

    def stage_start(self, name: str) -> None:
        """Report stage starting."""
        ...

    def stage_progress(self, name: str, message: str) -> None:
        """Report stage progress update."""
        ...

    def stage_complete(self, name: str, duration_sec: float, stats: Dict) -> None:
        """Report stage completion with stats."""
        ...

    def stage_error(self, name: str, error: str) -> None:
        """Report stage error."""
        ...

    def stage_skipped(self, name: str, reason: str) -> None:
        """Report stage skipped."""
        ...


class ConsoleReporter:
    """
    Reports bootstrap progress to stdout with [STAGE] prefix format.

    IMPORTANT: ALL output (including errors) goes to stdout per success
    criterion: "The bootstrap script reports progress for each stage
    (stage name, status, errors if any) to stdout"
    """

    def stage_start(self, name: str) -> None:
        """Report stage starting."""
        print(f"[STAGE] {name}: Starting...", flush=True)

    def stage_progress(self, name: str, message: str) -> None:
        """Report stage progress update."""
        print(f"[STAGE] {name}: {message}", flush=True)

    def stage_complete(self, name: str, duration_sec: float, stats: Dict) -> None:
        """Report stage completion with stats."""
        summary = ", ".join(
            f"{k}={v}" for k, v in stats.items() if k != "duration_seconds"
        )
        print(f"[STAGE] {name}: Completed in {duration_sec:.1f}s ({summary})", flush=True)

    def stage_error(self, name: str, error: str) -> None:
        """Report stage error to stdout (NOT stderr)."""
        print(f"[STAGE] {name}: FAILED - {error}", flush=True)

    def stage_skipped(self, name: str, reason: str) -> None:
        """Report stage skipped."""
        print(f"[STAGE] {name}: Skipped ({reason})", flush=True)

    def pipeline_complete(self, total_stages: int, duration_sec: float) -> None:
        """Report pipeline completion."""
        print(
            f"[BOOTSTRAP] Complete: {total_stages}/{total_stages} stages in {duration_sec:.1f}s",
            flush=True,
        )


# --- Dual Idempotency Check ---


def should_skip_stage(
    stage_name: str,
    state: BootstrapState,
    validator: Callable[[], Tuple[bool, str]],
) -> Tuple[bool, str]:
    """
    Dual idempotency check: checkpoint status AND output validation.

    This function implements the core resume logic for bootstrap stages.
    A stage should only be skipped if BOTH conditions are true:
    1. Checkpoint status is COMPLETED
    2. Output validation passes (output exists and is valid)

    Args:
        stage_name: Name of the stage to check
        state: Current bootstrap state
        validator: Callable that returns (is_valid, reason) for output validation

    Returns:
        Tuple of (should_skip, reason) where reason explains the decision.

    Logic:
    - Output valid -> skip (output exists regardless of checkpoint state)
    - COMPLETED but output invalid -> don't skip (stale checkpoint, re-run)
    - PENDING/RUNNING/FAILED with no output -> don't skip (need to run/re-run)
    """
    # Always check output first — if valid output exists, skip regardless
    # of checkpoint state. Handles deleted state files, fresh clones with
    # pre-existing data, and interrupted stages whose output survived.
    try:
        is_valid, validation_reason = validator()
    except Exception as e:
        logger.error(f"Output validator for '{stage_name}' raised exception: {e}")
        is_valid = False
        validation_reason = f"Validation exception: {e}"

    if is_valid:
        return True, f"Valid output exists: {validation_reason}"

    # Output doesn't exist — check checkpoint for context on why
    if stage_name not in state.stages:
        return False, f"No checkpoint and no valid output"

    stage_state = state.stages[stage_name]
    status = stage_state.status

    if status == StageStatus.RUNNING:
        logger.warning(
            f"Stage '{stage_name}' was interrupted (status=RUNNING), will re-run"
        )
        return False, "Stage was interrupted (status=RUNNING), needs re-run"

    if status == StageStatus.FAILED:
        return False, f"Stage previously failed, needs re-run"

    if status == StageStatus.COMPLETED:
        logger.warning(
            f"Stage '{stage_name}' marked COMPLETED but output invalid: {validation_reason}"
        )
        return False, f"Checkpoint stale (output invalid: {validation_reason}), needs re-run"

    return False, "Stage is pending (never completed)"
