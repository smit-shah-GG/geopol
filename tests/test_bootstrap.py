"""
Tests for bootstrap checkpoint, validation, and resume behavior.

These tests prove:
1. Checkpoint manager correctly saves and loads state
2. Dual idempotency (checkpoint + output validation) works correctly
3. Resume scenarios work as expected after interrupt
4. Validation functions correctly identify valid/invalid outputs
"""

from __future__ import annotations

import pytest
from pathlib import Path
import tempfile

from src.bootstrap.checkpoint import (
    CheckpointManager,
    BootstrapState,
    StageStatus,
    StageState,
    should_skip_stage,
    ConsoleReporter,
)
from src.bootstrap.validation import validate_gdelt_output


class TestCheckpointManager:
    """Tests for CheckpointManager state persistence."""

    def test_load_missing_file_returns_empty_state(self, tmp_path: Path) -> None:
        """Loading from non-existent file returns empty state."""
        mgr = CheckpointManager(tmp_path / "state.json")
        state = mgr.load()
        assert len(state.stages) == 0
        assert state.pipeline_started is None
        assert state.pipeline_completed is None

    def test_save_and_load_roundtrip(self, tmp_path: Path) -> None:
        """State survives save/load cycle."""
        mgr = CheckpointManager(tmp_path / "state.json")
        state = BootstrapState()
        state.stages["test"] = StageState(
            name="test",
            status=StageStatus.COMPLETED,
            output_path="/some/path",
        )
        state.stages["test"].stats = {"events": 100}
        mgr.save(state)

        # Force reload by creating new manager
        mgr2 = CheckpointManager(tmp_path / "state.json")
        loaded = mgr2.load()
        assert "test" in loaded.stages
        assert loaded.stages["test"].status == StageStatus.COMPLETED
        assert loaded.stages["test"].output_path == "/some/path"
        assert loaded.stages["test"].stats == {"events": 100}

    def test_atomic_write_creates_file(self, tmp_path: Path) -> None:
        """Atomic write creates state file."""
        state_file = tmp_path / "state.json"
        mgr = CheckpointManager(state_file)
        mgr.save(BootstrapState())
        assert state_file.exists()

    def test_corrupted_state_file_returns_empty_state(self, tmp_path: Path) -> None:
        """Corrupted JSON returns empty state instead of raising."""
        state_file = tmp_path / "state.json"
        state_file.write_text("{ invalid json }")

        mgr = CheckpointManager(state_file)
        state = mgr.load()
        assert len(state.stages) == 0

    def test_mark_running_updates_state(self, tmp_path: Path) -> None:
        """mark_running sets stage to RUNNING."""
        mgr = CheckpointManager(tmp_path / "state.json")
        mgr.mark_running("test")

        state = mgr.load()
        assert state.stages["test"].status == StageStatus.RUNNING
        assert state.stages["test"].started_at is not None

    def test_mark_completed_updates_state(self, tmp_path: Path) -> None:
        """mark_completed sets stage to COMPLETED with stats."""
        mgr = CheckpointManager(tmp_path / "state.json")
        mgr.mark_completed("test", "/output/path", {"events": 42})

        state = mgr.load()
        assert state.stages["test"].status == StageStatus.COMPLETED
        assert state.stages["test"].output_path == "/output/path"
        assert state.stages["test"].stats["events"] == 42

    def test_mark_failed_updates_state(self, tmp_path: Path) -> None:
        """mark_failed sets stage to FAILED with error."""
        mgr = CheckpointManager(tmp_path / "state.json")
        mgr.mark_failed("test", "Something went wrong")

        state = mgr.load()
        assert state.stages["test"].status == StageStatus.FAILED
        assert state.stages["test"].error == "Something went wrong"

    def test_reset_stage_clears_state(self, tmp_path: Path) -> None:
        """reset_stage resets to PENDING."""
        mgr = CheckpointManager(tmp_path / "state.json")
        mgr.mark_completed("test", "/path", {})
        mgr.reset_stage("test")

        state = mgr.load()
        assert state.stages["test"].status == StageStatus.PENDING


class TestDualIdempotency:
    """Tests for should_skip_stage dual idempotency logic."""

    def test_skip_when_output_valid(self) -> None:
        """Skip stage when output is valid, regardless of checkpoint status."""
        state = BootstrapState()
        state.stages["test"] = StageState(name="test", status=StageStatus.COMPLETED)

        skip, reason = should_skip_stage("test", state, lambda: (True, "ok"))
        assert skip is True
        assert "valid output" in reason.lower()

    def test_skip_when_output_valid_but_pending(self) -> None:
        """Skip when output exists even if checkpoint says PENDING."""
        state = BootstrapState()
        state.stages["test"] = StageState(name="test", status=StageStatus.PENDING)

        skip, reason = should_skip_stage("test", state, lambda: (True, "ok"))
        assert skip is True
        assert "valid output" in reason.lower()

    def test_skip_when_output_valid_but_running(self) -> None:
        """Skip when output exists even if checkpoint says RUNNING (interrupted but output survived)."""
        state = BootstrapState()
        state.stages["test"] = StageState(name="test", status=StageStatus.RUNNING)

        skip, reason = should_skip_stage("test", state, lambda: (True, "ok"))
        assert skip is True
        assert "valid output" in reason.lower()

    def test_skip_when_output_valid_but_failed(self) -> None:
        """Skip when output exists even if checkpoint says FAILED (output from earlier run)."""
        state = BootstrapState()
        state.stages["test"] = StageState(name="test", status=StageStatus.FAILED)

        skip, reason = should_skip_stage("test", state, lambda: (True, "ok"))
        assert skip is True
        assert "valid output" in reason.lower()

    def test_skip_when_output_valid_stage_not_in_state(self) -> None:
        """Skip when output exists even if no checkpoint record (deleted state file)."""
        state = BootstrapState()

        skip, reason = should_skip_stage("new_stage", state, lambda: (True, "ok"))
        assert skip is True
        assert "valid output" in reason.lower()

    def test_no_skip_when_complete_but_invalid(self) -> None:
        """Don't skip when checkpoint=COMPLETED but output missing/invalid."""
        state = BootstrapState()
        state.stages["test"] = StageState(name="test", status=StageStatus.COMPLETED)

        skip, reason = should_skip_stage("test", state, lambda: (False, "file missing"))
        assert skip is False
        assert "stale" in reason.lower() or "invalid" in reason.lower()

    def test_no_skip_when_no_output_and_no_checkpoint(self) -> None:
        """Don't skip when no output and no checkpoint record."""
        state = BootstrapState()

        skip, reason = should_skip_stage("new_stage", state, lambda: (False, "missing"))
        assert skip is False
        assert "no checkpoint" in reason.lower() or "no valid output" in reason.lower()

    def test_no_skip_when_no_output_and_running(self) -> None:
        """Don't skip when no output and checkpoint says RUNNING."""
        state = BootstrapState()
        state.stages["test"] = StageState(name="test", status=StageStatus.RUNNING)

        skip, reason = should_skip_stage("test", state, lambda: (False, "missing"))
        assert skip is False
        assert "interrupt" in reason.lower() or "running" in reason.lower()

    def test_validator_exception_returns_no_skip(self) -> None:
        """If validator raises exception, don't skip (re-run to be safe)."""
        state = BootstrapState()
        state.stages["test"] = StageState(name="test", status=StageStatus.COMPLETED)

        def bad_validator():
            raise RuntimeError("Validator crashed")

        skip, reason = should_skip_stage("test", state, bad_validator)
        assert skip is False
        assert "stale" in reason.lower() or "invalid" in reason.lower()


class TestValidation:
    """Tests for output validation functions."""

    def test_gdelt_validation_fails_for_nonexistent_dir(self) -> None:
        """GDELT validation fails for non-existent directory."""
        valid, reason = validate_gdelt_output(Path("/nonexistent_dir"))
        assert valid is False
        assert "not exist" in reason.lower() or "not found" in reason.lower()

    def test_gdelt_validation_fails_for_empty_dir(self, tmp_path: Path) -> None:
        """GDELT validation fails for directory with no CSVs."""
        valid, reason = validate_gdelt_output(tmp_path)
        assert valid is False
        assert "no" in reason.lower() and "csv" in reason.lower()

    def test_gdelt_validation_fails_for_empty_csvs(self, tmp_path: Path) -> None:
        """GDELT validation fails when all CSVs are empty."""
        csv_file = tmp_path / "gdelt_2025-01-01.csv"
        csv_file.write_text("")  # Empty file

        valid, reason = validate_gdelt_output(tmp_path)
        assert valid is False
        assert "empty" in reason.lower()

    def test_gdelt_validation_fails_for_header_only_csvs(self, tmp_path: Path) -> None:
        """GDELT validation fails when CSVs only have headers."""
        csv_file = tmp_path / "gdelt_2025-01-01.csv"
        csv_file.write_text("col1,col2,col3\n")  # Just header, <100 bytes

        valid, reason = validate_gdelt_output(tmp_path)
        assert valid is False

    def test_gdelt_validation_passes_with_csv(self, tmp_path: Path) -> None:
        """GDELT validation passes when CSV exists with content."""
        csv_file = tmp_path / "gdelt_2025-01-01.csv"
        # Write enough content to exceed 100 byte threshold
        csv_file.write_text("col1,col2,col3\n" + "val1,val2,val3\n" * 10)

        valid, reason = validate_gdelt_output(tmp_path)
        assert valid is True
        assert "csv" in reason.lower()


class TestResumeScenario:
    """Integration tests for resume after interrupt scenarios."""

    def test_resume_from_checkpoint(self, tmp_path: Path) -> None:
        """Simulate interrupt and resume scenario."""
        state_file = tmp_path / "state.json"
        mgr = CheckpointManager(state_file)

        # Simulate: stages 1-2 completed, stage 3 was running (interrupted)
        state = BootstrapState()
        state.stages["collect"] = StageState(name="collect", status=StageStatus.COMPLETED)
        state.stages["process"] = StageState(name="process", status=StageStatus.COMPLETED)
        state.stages["graph"] = StageState(name="graph", status=StageStatus.RUNNING)
        mgr.save(state)

        # Resume: load state, check which stages to run
        loaded = mgr.load()

        # collect and process should skip (outputs valid)
        skip_collect, _ = should_skip_stage("collect", loaded, lambda: (True, "ok"))
        skip_process, _ = should_skip_stage("process", loaded, lambda: (True, "ok"))
        assert skip_collect is True, "Should skip stage with valid output"
        assert skip_process is True, "Should skip stage with valid output"

        # graph with valid output should skip even if interrupted
        skip_graph_valid, _ = should_skip_stage("graph", loaded, lambda: (True, "ok"))
        assert skip_graph_valid is True, "Should skip interrupted stage if output survived"

        # graph with invalid output should NOT skip
        skip_graph_invalid, _ = should_skip_stage("graph", loaded, lambda: (False, "missing"))
        assert skip_graph_invalid is False, "Should NOT skip interrupted stage with no output"

    def test_resume_with_stale_checkpoint(self, tmp_path: Path) -> None:
        """Resume when checkpoint says complete but output is missing."""
        state_file = tmp_path / "state.json"
        mgr = CheckpointManager(state_file)

        # Stage marked complete but output doesn't exist
        state = BootstrapState()
        state.stages["collect"] = StageState(
            name="collect",
            status=StageStatus.COMPLETED,
            output_path="/path/that/does/not/exist",
        )
        mgr.save(state)

        # Resume: output validation fails, should not skip
        loaded = mgr.load()
        skip, reason = should_skip_stage(
            "collect",
            loaded,
            lambda: (False, "output directory missing"),
        )

        assert skip is False, "Should NOT skip when output invalid"
        assert "stale" in reason.lower() or "invalid" in reason.lower()

    def test_full_resume_flow(self, tmp_path: Path) -> None:
        """Test complete resume flow with checkpointing."""
        state_file = tmp_path / "state.json"
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        mgr = CheckpointManager(state_file)

        # First run: mark collect as running (simulating start)
        mgr.mark_running("collect")
        state = mgr.load()
        assert state.stages["collect"].status == StageStatus.RUNNING

        # First run: complete collect
        mgr.mark_completed("collect", str(output_dir), {"files": 5})
        state = mgr.load()
        assert state.stages["collect"].status == StageStatus.COMPLETED

        # Simulate restart: create fresh manager, check resume logic
        mgr2 = CheckpointManager(state_file)
        state2 = mgr2.load()

        # Create valid output for collect stage
        csv_file = output_dir / "gdelt_2025-01-01.csv"
        csv_file.write_text("col1,col2,col3\n" + "val1,val2,val3\n" * 10)

        skip, reason = should_skip_stage(
            "collect",
            state2,
            lambda: validate_gdelt_output(output_dir),
        )

        assert skip is True, "Should skip collect since output exists"


class TestConsoleReporter:
    """Tests for ConsoleReporter output formatting."""

    def test_reporter_outputs_to_stdout(self, capsys) -> None:
        """Reporter outputs to stdout, not stderr."""
        reporter = ConsoleReporter()
        reporter.stage_start("test")
        reporter.stage_error("test", "error message")

        captured = capsys.readouterr()
        assert "[STAGE] test: Starting..." in captured.out
        assert "[STAGE] test: FAILED - error message" in captured.out
        assert captured.err == ""  # Nothing to stderr

    def test_reporter_format_prefix(self, capsys) -> None:
        """Reporter uses [STAGE] prefix."""
        reporter = ConsoleReporter()
        reporter.stage_progress("mytest", "processing")

        captured = capsys.readouterr()
        assert "[STAGE] mytest: processing" in captured.out

    def test_reporter_complete_format(self, capsys) -> None:
        """Reporter formats completion with duration and stats."""
        reporter = ConsoleReporter()
        reporter.stage_complete("test", 2.5, {"events": 100, "files": 3})

        captured = capsys.readouterr()
        assert "Completed in 2.5s" in captured.out
        assert "events=100" in captured.out
        assert "files=3" in captured.out

    def test_reporter_skipped_format(self, capsys) -> None:
        """Reporter formats skipped stage."""
        reporter = ConsoleReporter()
        reporter.stage_skipped("test", "already complete")

        captured = capsys.readouterr()
        assert "[STAGE] test: Skipped (already complete)" in captured.out

    def test_pipeline_complete_format(self, capsys) -> None:
        """Reporter formats pipeline completion."""
        reporter = ConsoleReporter()
        reporter.pipeline_complete(5, 120.3)

        captured = capsys.readouterr()
        assert "[BOOTSTRAP] Complete: 5/5 stages in 120.3s" in captured.out
