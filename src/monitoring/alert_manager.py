"""
SMTP alert manager with per-type cooldown and async-safe sending.

Sends email alerts for monitoring events (feed staleness, calibration drift,
disk pressure, budget exhaustion). All SMTP I/O is offloaded to a thread
via asyncio.to_thread() so the async event loop is never blocked.

Rate limiting is per alert_type with a configurable cooldown window.
SMTP failures are logged but NEVER propagated -- alerts are fire-and-forget.
"""

from __future__ import annotations

import asyncio
import logging
import smtplib
from datetime import datetime, timedelta, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.settings import Settings

logger = logging.getLogger(__name__)


class AlertManager:
    """SMTP alert sender with per-type cooldown enforcement.

    Attributes:
        _enabled: False if smtp_host is empty (no SMTP configured).
        _cooldown: Minimum interval between alerts of the same type.
        _last_sent: Per-type timestamp of the last successfully sent alert.
    """

    def __init__(self, settings: Settings) -> None:
        self._smtp_host: str = settings.smtp_host
        self._smtp_port: int = settings.smtp_port
        self._smtp_username: str = settings.smtp_username
        self._smtp_password: str = settings.smtp_password
        self._smtp_sender: str = settings.smtp_sender
        self._recipient: str = settings.alert_recipient
        self._cooldown = timedelta(minutes=settings.alert_cooldown_minutes)
        self._last_sent: dict[str, datetime] = {}

        if not self._smtp_host:
            self._enabled = False
            logger.warning("SMTP not configured, alerts disabled")
        else:
            self._enabled = True
            logger.info(
                "AlertManager enabled: host=%s port=%d recipient=%s cooldown=%dm",
                self._smtp_host,
                self._smtp_port,
                self._recipient,
                settings.alert_cooldown_minutes,
            )

    @property
    def is_enabled(self) -> bool:
        """Whether SMTP alerting is operational."""
        return self._enabled

    async def send_alert(self, alert_type: str, subject: str, body: str) -> bool:
        """Send a rate-limited email alert.

        Args:
            alert_type: Identifier for cooldown grouping (e.g. "feed_stale").
            subject: Email subject (automatically prefixed with [Geopol]).
            body: Plain-text email body.

        Returns:
            True if the alert was sent, False if suppressed, disabled, or failed.
        """
        if not self._enabled:
            logger.debug("Alert suppressed (SMTP disabled): type=%s", alert_type)
            return False

        # Cooldown check
        now = datetime.now(timezone.utc)
        last = self._last_sent.get(alert_type)
        if last is not None and (now - last) < self._cooldown:
            remaining = self._cooldown - (now - last)
            logger.debug(
                "Alert suppressed (cooldown): type=%s, %ds remaining",
                alert_type,
                int(remaining.total_seconds()),
            )
            return False

        # Offload synchronous SMTP to thread pool
        try:
            await asyncio.to_thread(self._send_sync, subject, body)
            self._last_sent[alert_type] = now
            logger.info("Alert sent: type=%s subject=%r", alert_type, subject)
            return True
        except Exception:
            logger.error(
                "Failed to send alert: type=%s subject=%r",
                alert_type,
                subject,
                exc_info=True,
            )
            return False

    def _send_sync(self, subject: str, body: str) -> None:
        """Synchronous SMTP send -- called via asyncio.to_thread().

        Raises on failure; caller catches and logs.
        """
        msg = MIMEMultipart("alternative")
        msg["Subject"] = f"[Geopol] {subject}"
        msg["From"] = self._smtp_sender
        msg["To"] = self._recipient
        msg.attach(MIMEText(body, "plain"))

        with smtplib.SMTP(self._smtp_host, self._smtp_port, timeout=10) as server:
            server.starttls()
            server.login(self._smtp_username, self._smtp_password)
            server.send_message(msg)

    def reset_cooldown(self, alert_type: str) -> None:
        """Clear cooldown for an alert type (for testing or manual override)."""
        removed = self._last_sent.pop(alert_type, None)
        if removed is not None:
            logger.debug("Cooldown reset for alert_type=%s", alert_type)
