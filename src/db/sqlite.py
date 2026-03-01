"""
SQLite connection manager for the GDELT event store.

Retained from v1.x -- GDELT events and partition index remain in SQLite
because they are single-writer workloads where WAL mode + busy_timeout
is sufficient. PostgreSQL handles the multi-writer forecast persistence.
"""

from __future__ import annotations

import logging
import sqlite3
from contextlib import contextmanager
from collections.abc import Generator
from pathlib import Path

from src.settings import get_settings

logger = logging.getLogger(__name__)


class SQLiteConnection:
    """Context-managed SQLite connection with WAL mode and tuned pragmas.

    Usage::

        sqlite = SQLiteConnection()
        with sqlite.get_connection() as conn:
            rows = conn.execute("SELECT * FROM events LIMIT 10").fetchall()
    """

    def __init__(self, db_path: str | None = None) -> None:
        settings = get_settings()
        self.db_path = db_path or settings.gdelt_db_path

        # Ensure parent directory exists for fresh installs
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Yield a configured SQLite connection.

        Pragmas applied on every connection:
            - journal_mode = WAL  (concurrent readers, single writer)
            - busy_timeout = 30000  (30s retry on SQLITE_BUSY)
            - synchronous = NORMAL  (safe with WAL, faster than FULL)
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA busy_timeout = 30000")
        conn.execute("PRAGMA synchronous = NORMAL")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
