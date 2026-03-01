"""Structured logging configuration for the geopol forecasting engine.

Provides two output modes:
- Human-readable (dev): timestamped, leveled, module-attributed lines.
- JSON (production): machine-parseable structured log lines for log
  aggregation pipelines (ELK, CloudWatch, etc.).

Usage:
    from src.logging_config import setup_logging

    setup_logging()                         # INFO, human-readable
    setup_logging(level="DEBUG")            # DEBUG, human-readable
    setup_logging(json_format=True)         # INFO, JSON lines
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any


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
) -> None:
    """Configure the root logger with a single stderr handler.

    Idempotent: clears existing handlers before adding a new one so
    repeated calls (e.g. in tests) don't produce duplicate output.

    Args:
        level: Log level name (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        json_format: If True, emit JSON lines instead of human-readable.

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

    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(numeric_level)

    if json_format:
        handler.setFormatter(_JSONFormatter())
    else:
        handler.setFormatter(logging.Formatter(_HUMAN_FMT))

    root.addHandler(handler)
