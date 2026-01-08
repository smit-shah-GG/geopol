"""
GDELT Doc API client with rate limiting and error handling.
"""

import logging
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
import pandas as pd
from gdeltdoc import GdeltDoc, Filters
import requests

from .rate_limiter import RateLimiter
from .config import (
    GDELT_REQUEST_TIMEOUT,
    GDELT_MAX_RETRIES,
    GDELT_BASE_DELAY,
    QUADCLASS_VERBAL_COOPERATION,
    QUADCLASS_MATERIAL_CONFLICT,
    GDELT_DATE_FORMAT,
)

logger = logging.getLogger(__name__)


class GDELTClient:
    """Wrapper for gdeltdoc with rate limiting and error handling."""

    def __init__(
        self,
        base_delay: float = GDELT_BASE_DELAY,
        max_retries: int = GDELT_MAX_RETRIES,
        timeout: int = GDELT_REQUEST_TIMEOUT,
    ):
        """
        Initialize GDELT client.

        Args:
            base_delay: Base delay for exponential backoff (seconds)
            max_retries: Maximum number of retry attempts
            timeout: Request timeout in seconds
        """
        self.gd = GdeltDoc()
        self.rate_limiter = RateLimiter(base_delay, max_retries)
        self.timeout = timeout

        # Configure requests session timeout
        self._configure_session()

    def _configure_session(self):
        """Configure the underlying requests session if accessible."""
        # gdeltdoc doesn't expose session directly, so we set a global timeout
        # This is a workaround - in production, we might fork gdeltdoc
        import socket
        socket.setdefaulttimeout(self.timeout)

    def fetch_recent_events(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        timespan: Optional[str] = None,
        quad_classes: Optional[List[int]] = None,
        themes: Optional[str] = None,
        keyword: Optional[str] = None,
        domain: Optional[str] = None,
        country: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Fetch recent events from GDELT Doc API with filters.

        Args:
            start_date: Start date in format "YYYY-MM-DD"
            end_date: End date in format "YYYY-MM-DD"
            timespan: Alternative to dates - e.g., "24h", "7d", "1w"
            quad_classes: List of QuadClass values to filter (1-4)
            themes: GDELT theme filter (e.g., "DIPLOMATIC_EXCHANGE OR ARMED_CONFLICT")
            keyword: Keyword search in article text
            domain: Specific news domain to filter
            country: Country code to filter events

        Returns:
            DataFrame with GDELT events
        """
        # Build filters - use either timespan OR date range
        filter_args = {}

        if timespan:
            # Use timespan if provided (more reliable for recent data)
            filter_args["timespan"] = timespan
        else:
            # Use date range if no timespan
            # Default to last 24 hours if no dates provided
            if not end_date:
                end_date = datetime.now().strftime("%Y-%m-%d")
            if not start_date:
                # Default to yesterday for a 2-day window
                start_datetime = datetime.now() - timedelta(days=1)
                start_date = start_datetime.strftime("%Y-%m-%d")

            filter_args["start_date"] = start_date
            filter_args["end_date"] = end_date

        if themes:
            filter_args["theme"] = themes
        if keyword:
            filter_args["keyword"] = keyword
        if domain:
            filter_args["domain"] = domain
        if country:
            filter_args["country"] = country

        try:
            filters = Filters(**filter_args)
        except Exception as e:
            logger.error(f"Invalid filter parameters: {e}")
            raise ValueError(f"Failed to create GDELT filters: {e}")

        # Wrap API call with rate limiting
        @self.rate_limiter.with_exponential_backoff
        def _fetch():
            logger.info(
                f"Fetching GDELT events from {start_date} to {end_date}"
                f" with filters: {filter_args}"
            )

            try:
                # Use article_search for Doc API access
                articles_df = self.gd.article_search(filters)

                if articles_df is None or articles_df.empty:
                    logger.warning("No events returned from GDELT")
                    return pd.DataFrame()

                logger.info(f"Retrieved {len(articles_df)} articles from GDELT")

                # Post-process for QuadClass filtering if needed
                if quad_classes and 'QuadClass' in articles_df.columns:
                    original_count = len(articles_df)
                    articles_df = articles_df[articles_df['QuadClass'].isin(quad_classes)]
                    logger.info(
                        f"Filtered {original_count} events to {len(articles_df)} "
                        f"for QuadClasses {quad_classes}"
                    )

                return articles_df

            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429:
                    # Will be caught by rate limiter decorator
                    raise
                logger.error(f"HTTP error from GDELT API: {e}")
                raise
            except requests.exceptions.Timeout:
                logger.error(f"Request timed out after {self.timeout} seconds")
                raise TimeoutError(f"GDELT request timed out after {self.timeout}s")
            except Exception as e:
                logger.error(f"Unexpected error fetching GDELT data: {e}")
                raise

        # Execute with rate limiting
        return _fetch()

    def fetch_conflict_diplomatic_events(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Fetch specifically conflict and diplomatic events.

        This is a convenience method that fetches QuadClass 1 (verbal cooperation)
        and QuadClass 4 (material conflict) events.

        Args:
            start_date: Start date for event search
            end_date: End date for event search

        Returns:
            DataFrame with conflict and diplomatic events
        """
        return self.fetch_recent_events(
            start_date=start_date,
            end_date=end_date,
            quad_classes=[QUADCLASS_VERBAL_COOPERATION, QUADCLASS_MATERIAL_CONFLICT],
            themes="(DIPLOMATIC_EXCHANGE OR ARMED_CONFLICT OR MILITARY_CONFLICT)",
        )

    def test_connection(self) -> bool:
        """
        Test connection to GDELT API.

        Returns:
            True if connection successful, False otherwise
        """
        try:
            # Try to fetch events from last 24 hours using timespan
            # This is more reliable than specific dates
            df = self.fetch_recent_events(
                timespan="24h",
                keyword="(conflict OR diplomatic)",  # Relevant keywords with parentheses
            )

            logger.info("GDELT connection test successful")
            return True

        except Exception as e:
            logger.error(f"GDELT connection test failed: {e}")
            return False