"""
Database connection management with safe connection handling.
"""

import sqlite3
from contextlib import contextmanager
from typing import Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class DatabaseConnection:
    """Manages SQLite database connections with context manager support."""

    def __init__(self, db_path: str = "data/events.db"):
        """
        Initialize database connection manager.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.connection: Optional[sqlite3.Connection] = None

        # Ensure data directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def get_connection(self):
        """
        Context manager for safe database connection handling.

        Yields:
            sqlite3.Connection: Database connection

        Example:
            with db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM events")
        """
        conn = None
        try:
            # Create connection with row factory for dict-like access
            conn = sqlite3.connect(str(self.db_path))
            conn.row_factory = sqlite3.Row

            # Enable foreign keys and optimize for batch operations
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute("PRAGMA journal_mode = WAL")  # Write-Ahead Logging for better concurrency
            conn.execute("PRAGMA synchronous = NORMAL")  # Faster writes with acceptable safety
            conn.execute("PRAGMA cache_size = 10000")  # Larger cache for better performance

            logger.debug(f"Database connection opened to {self.db_path}")
            yield conn

            # Commit any pending transactions
            conn.commit()

        except sqlite3.Error as e:
            logger.error(f"Database error: {e}")
            if conn:
                conn.rollback()
            raise

        finally:
            if conn:
                conn.close()
                logger.debug("Database connection closed")

    @contextmanager
    def transaction(self):
        """
        Context manager for explicit transaction handling.

        Example:
            with db.transaction() as conn:
                cursor = conn.cursor()
                cursor.execute("INSERT INTO events ...")
                # Automatically commits on success, rolls back on error
        """
        with self.get_connection() as conn:
            try:
                conn.execute("BEGIN TRANSACTION")
                yield conn
                conn.commit()
                logger.debug("Transaction committed")
            except Exception as e:
                conn.rollback()
                logger.error(f"Transaction rolled back due to error: {e}")
                raise

    def execute_script(self, script_path: str):
        """
        Execute SQL script file.

        Args:
            script_path: Path to SQL script file
        """
        with open(script_path, 'r') as f:
            script = f.read()

        with self.get_connection() as conn:
            conn.executescript(script)
            logger.info(f"Executed SQL script: {script_path}")

    def table_exists(self, table_name: str) -> bool:
        """
        Check if a table exists in the database.

        Args:
            table_name: Name of the table to check

        Returns:
            True if table exists, False otherwise
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                (table_name,)
            )
            return cursor.fetchone() is not None


# Global connection instance for module-level use
default_connection = DatabaseConnection()
get_connection = default_connection.get_connection