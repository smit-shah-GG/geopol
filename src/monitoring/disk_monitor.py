"""
Disk usage monitoring with emergency cleanup.

Monitors root partition utilisation via psutil.  Three status levels:

- ok:       percent_used < warning_threshold (default 80%)
- warning:  percent_used >= warning_threshold
- critical: percent_used >= critical_threshold (default 90%)

At critical level, emergency_cleanup() purges old log files and GDELT
data to reclaim space, supporting the 7-day unattended operation target.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import psutil

if TYPE_CHECKING:
    from src.monitoring.alert_manager import AlertManager
    from src.settings import Settings

logger = logging.getLogger(__name__)


class DiskMonitor:
    """Monitor disk usage and perform emergency cleanup when critical.

    Attributes:
        _warning_pct: Percentage threshold for "warning" status.
        _critical_pct: Percentage threshold for "critical" status + cleanup.
        _log_dir: Directory containing rotated log files.
        _gdelt_db_path: Path to GDELT SQLite database (parent dir used
            for locating data files).
    """

    def __init__(self, settings: Settings) -> None:
        self._warning_pct: float = settings.disk_warning_pct
        self._critical_pct: float = settings.disk_critical_pct
        self._log_dir = Path(settings.log_dir)
        self._gdelt_db_path = Path(settings.gdelt_db_path)
        self._log_retention_days: int = settings.log_retention_days

    def check_disk(self) -> dict[str, Any]:
        """Check root partition disk usage via psutil.

        Returns:
            Dict with percent_used, free_gb, total_gb, status,
            warning_threshold, critical_threshold.
        """
        try:
            usage = psutil.disk_usage("/")
        except OSError as exc:
            logger.error("psutil.disk_usage failed: %s", exc)
            return {
                "percent_used": 0.0,
                "free_gb": 0.0,
                "total_gb": 0.0,
                "status": "unknown",
                "warning_threshold": self._warning_pct,
                "critical_threshold": self._critical_pct,
                "error": str(exc)[:200],
            }

        percent = usage.percent
        free_gb = usage.free / (1024**3)
        total_gb = usage.total / (1024**3)

        if percent >= self._critical_pct:
            status = "critical"
        elif percent >= self._warning_pct:
            status = "warning"
        else:
            status = "ok"

        return {
            "percent_used": round(percent, 1),
            "free_gb": round(free_gb, 2),
            "total_gb": round(total_gb, 2),
            "status": status,
            "warning_threshold": self._warning_pct,
            "critical_threshold": self._critical_pct,
        }

    async def emergency_cleanup(self) -> dict[str, Any]:
        """Purge old logs and GDELT data to free disk space.

        Cleanup order:
          1. Log files older than retention_days/2 in log_dir.
          2. GDELT CSV/zip data files older than 90 days in data/ directory.

        Returns:
            Dict with files_removed, bytes_freed.
        """
        files_removed = 0
        bytes_freed = 0
        now = time.time()

        # 1. Purge old log files (half the normal retention)
        half_retention_secs = (self._log_retention_days / 2) * 86400
        if self._log_dir.is_dir():
            for log_file in self._log_dir.iterdir():
                if not log_file.is_file():
                    continue
                try:
                    age = now - log_file.stat().st_mtime
                    if age > half_retention_secs:
                        size = log_file.stat().st_size
                        log_file.unlink()
                        files_removed += 1
                        bytes_freed += size
                        logger.debug("Purged log: %s (%d bytes)", log_file.name, size)
                except OSError as exc:
                    logger.warning("Failed to purge %s: %s", log_file, exc)

        # 2. Purge old GDELT data files (>90 days)
        data_dir = self._gdelt_db_path.parent
        gdelt_retention_secs = 90 * 86400
        if data_dir.is_dir():
            for data_file in data_dir.iterdir():
                if not data_file.is_file():
                    continue
                # Only target GDELT CSV/zip files, not databases
                if data_file.suffix not in (".csv", ".zip", ".gz", ".CSV"):
                    continue
                try:
                    age = now - data_file.stat().st_mtime
                    if age > gdelt_retention_secs:
                        size = data_file.stat().st_size
                        data_file.unlink()
                        files_removed += 1
                        bytes_freed += size
                        logger.debug("Purged data: %s (%d bytes)", data_file.name, size)
                except OSError as exc:
                    logger.warning("Failed to purge %s: %s", data_file, exc)

        logger.info(
            "Emergency cleanup: removed %d files, freed %.1f MB",
            files_removed,
            bytes_freed / (1024 * 1024),
        )

        return {
            "files_removed": files_removed,
            "bytes_freed": bytes_freed,
        }

    async def check_and_alert(self, alert_manager: AlertManager) -> dict[str, Any]:
        """Check disk usage, alert if needed, and run cleanup if critical.

        Args:
            alert_manager: AlertManager instance for email dispatch.

        Returns:
            Disk status dict, possibly augmented with cleanup_result.
        """
        status = self.check_disk()
        disk_status = status.get("status", "unknown")

        if disk_status == "critical":
            body = (
                f"CRITICAL: Disk usage at {status['percent_used']:.1f}% "
                f"(threshold: {self._critical_pct}%).\n"
                f"Free: {status['free_gb']:.1f} GB / {status['total_gb']:.1f} GB\n\n"
                f"Running emergency cleanup..."
            )
            await alert_manager.send_alert("disk_critical", "Disk Usage CRITICAL", body)
            cleanup = await self.emergency_cleanup()
            status["cleanup_result"] = cleanup

        elif disk_status == "warning":
            body = (
                f"WARNING: Disk usage at {status['percent_used']:.1f}% "
                f"(threshold: {self._warning_pct}%).\n"
                f"Free: {status['free_gb']:.1f} GB / {status['total_gb']:.1f} GB\n\n"
                f"No cleanup triggered yet. Investigate and free space manually."
            )
            await alert_manager.send_alert("disk_warning", "Disk Usage Warning", body)

        return status
