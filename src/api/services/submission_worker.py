"""
Async background worker for processing confirmed forecast requests.

Placeholder -- full implementation in Task 3.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
RETRY_DELAYS = [30, 120, 600]  # seconds: 30s, 2min, 10min


def schedule_processing(request_id: str) -> None:
    """Schedule a confirmed forecast request for background processing.

    Stub -- full implementation in Task 3.
    """
    logger.info("Scheduling processing for request %s (stub)", request_id)
