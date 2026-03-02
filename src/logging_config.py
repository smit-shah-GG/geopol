"""Structured logging configuration for the geopol forecasting engine.

Provides two output modes:
- Human-readable (dev): timestamped, leveled, module-attributed lines.
- JSON (production): machine-parseable structured log lines for log
  aggregation pipelines (ELK, CloudWatch, etc.).

File rotation: When ``log_dir`` is provided, a daily-rotated JSON log
file is written alongside the stderr handler.  Rotation occurs at
midnight UTC; ``log_retention_days`` controls how many rotated files
are kept (default 30).

Usage:
    from src.logging_config import setup_logging

    setup_logging()                         # INFO, human-readable
    setup_logging(level="DEBUG")            # DEBUG, human-readable
    setup_logging(json_format=True)         # INFO, JSON lines
    setup_logging(log_dir="/var/log/geopol")# + daily-rotated file
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.settings import Settings


class _JSONFormatter(logging.Formatter):
    """Emit each log record as a single JSON line.

    Fields: timestamp (ISO-8601 UTC), severity, module, message, plus
    any extra attributes attached to the record.
    """

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat(),
            "severity": record.levelname,
            "module": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info and record.exc_info[0] is not None:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


_HUMAN_FMT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"


def setup_logging(
    level: str = "INFO",
    json_format: bool = False,
    log_dir: str | None = None,
    log_retention_days: int = 30,
) -> None:
    """Configure the root logger with stderr and optional file handlers.

    Idempotent: clears existing handlers before adding new ones so
    repeated calls (e.g. in tests) don't produce duplicate output.

    The stderr handler respects ``json_format``.  When ``log_dir`` is
    provided the file handler **always** uses JSON formatting regardless
    of ``json_format`` -- structured logs on disk are mandatory for
    machine parsing.

    Args:
        level: Log level name (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        json_format: If True, emit JSON lines on stderr instead of
            human-readable output.
        log_dir: Directory for rotated log files.  ``None`` or empty
            string disables file logging.
        log_retention_days: Number of daily rotated log files to retain.
            Only meaningful when ``log_dir`` is set.

    Raises:
        ValueError: If *level* is not a recognised log level name.
    """
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {level!r}")

    root = logging.getLogger()
    root.setLevel(numeric_level)

    # Remove all existing handlers to prevent duplicates
    for handler in root.handlers[:]:
        root.removeHandler(handler)
        handler.close()

    # -- stderr handler (always present) --
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(numeric_level)

    if json_format:
        stderr_handler.setFormatter(_JSONFormatter())
    else:
        stderr_handler.setFormatter(logging.Formatter(_HUMAN_FMT))

    root.addHandler(stderr_handler)

    # -- File handler (opt-in via log_dir) --
    if log_dir:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)

        file_handler = TimedRotatingFileHandler(
            filename=str(log_path / "geopol.log"),
            when="midnight",
            interval=1,
            backupCount=log_retention_days,
            encoding="utf-8",
            utc=True,
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(_JSONFormatter())
        root.addHandler(file_handler)


def setup_logging_from_settings(settings: Settings) -> None:
    """Configure logging from a :class:`Settings` instance.

    Convenience wrapper that extracts ``log_level``, ``log_json``,
    ``log_dir``, and ``log_retention_days`` from the settings object
    and forwards them to :func:`setup_logging`.
    """
    setup_logging(
        level=settings.log_level,
        json_format=settings.log_json,
        log_dir=settings.log_dir,
        log_retention_days=settings.log_retention_days,
    )
