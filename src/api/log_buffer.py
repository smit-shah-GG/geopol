"""In-memory ring buffer log handler for the admin dashboard.

Captures the last N log records in a thread-safe deque (GIL-protected
under workers=1) and exposes filtered retrieval for the admin logs API.

Usage:
    from src.api.log_buffer import get_ring_buffer

    ring = get_ring_buffer()
    logging.getLogger().addHandler(ring)

    # Later, in admin endpoint:
    entries = ring.get_entries(severity="ERROR", subsystem="ingest")
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass(frozen=True, slots=True)
class LogEntry:
    """Immutable structured log record for API consumption."""

    timestamp: str  # ISO-8601 UTC
    severity: str
    module: str
    message: str


_DEFAULT_CAPACITY = 1000


class RingBufferHandler(logging.Handler):
    """Logging handler backed by a bounded deque.

    Thread-safe under CPython GIL with a single uvicorn worker.
    The deque automatically evicts the oldest entry when capacity is
    reached -- no manual eviction logic required.
    """

    def __init__(self, capacity: int = _DEFAULT_CAPACITY) -> None:
        super().__init__()
        self._buffer: deque[LogEntry] = deque(maxlen=capacity)

    def emit(self, record: logging.LogRecord) -> None:
        """Convert a LogRecord to a LogEntry and append to the ring."""
        try:
            entry = LogEntry(
                timestamp=datetime.fromtimestamp(
                    record.created, tz=timezone.utc
                ).isoformat(),
                severity=record.levelname,
                module=record.name,
                message=self.format(record) if self.formatter else record.getMessage(),
            )
            self._buffer.append(entry)
        except Exception:
            # Handler.emit contract: never raise
            self.handleError(record)

    def get_entries(
        self,
        severity: str | None = None,
        subsystem: str | None = None,
    ) -> list[LogEntry]:
        """Return filtered log entries from the buffer.

        Args:
            severity: Exact match on log level (e.g. "ERROR", "WARNING").
            subsystem: Substring match on the module name.

        Returns:
            Filtered list ordered oldest-first (deque insertion order).
        """
        entries: list[LogEntry] = list(self._buffer)
        if severity:
            sev_upper = severity.upper()
            entries = [e for e in entries if e.severity == sev_upper]
        if subsystem:
            sub_lower = subsystem.lower()
            entries = [e for e in entries if sub_lower in e.module.lower()]
        return entries

    @property
    def size(self) -> int:
        """Current number of entries in the buffer."""
        return len(self._buffer)


# Module-level singleton -- lazy-initialized via get_ring_buffer().
_ring_buffer: RingBufferHandler | None = None


def get_ring_buffer(capacity: int = _DEFAULT_CAPACITY) -> RingBufferHandler:
    """Return (or create) the module-level RingBufferHandler singleton."""
    global _ring_buffer  # noqa: PLW0603
    if _ring_buffer is None:
        _ring_buffer = RingBufferHandler(capacity=capacity)
    return _ring_buffer
