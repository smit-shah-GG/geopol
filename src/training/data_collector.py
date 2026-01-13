"""
GDELT historical data collector for TKG training using gdeltPyR.

Uses the GDELT Events table (v2) for bulk historical collection.
Unlike gdeltdoc (Doc API), gdeltPyR provides access to structured event data
with Actor1/Actor2, EventCode, and QuadClass fields needed for TKG construction.
"""

import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Default output directory
DEFAULT_RAW_DIR = Path(__file__).parent.parent.parent / "data" / "gdelt" / "raw"


class GDELTHistoricalCollector:
    """
    Collects historical GDELT event data for TKG training.

    Uses gdeltPyR (gdelt package) to access the GDELT Events table v2,
    which provides structured actor/event data unlike the Doc API.
    """

    def __init__(
        self,
        output_dir: Optional[Path] = None,
        base_delay: float = 2.0,
        max_retries: int = 3,
    ):
        """
        Initialize the historical collector.

        Args:
            output_dir: Directory for raw CSV output (default: data/gdelt/raw/)
            base_delay: Base delay between requests in seconds
            max_retries: Maximum retry attempts per day
        """
        self.output_dir = output_dir or DEFAULT_RAW_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.base_delay = base_delay
        self.max_retries = max_retries

        # Import gdelt here to avoid import errors if not installed
        try:
            import gdelt
            self.gd = gdelt.gdelt(version=2)
        except ImportError as e:
            raise ImportError(
                "gdelt package not installed. Install with: pip install gdelt"
            ) from e

    def collect_single_day(
        self,
        date: datetime,
        quad_classes: Optional[list[int]] = None,
    ) -> Optional[pd.DataFrame]:
        """
        Collect events for a single day with retry logic.

        Args:
            date: Date to collect events for
            quad_classes: List of QuadClass values to include (1-4), None for all

        Returns:
            DataFrame of events, or None if collection failed
        """
        date_str = date.strftime("%Y %m %d")

        for attempt in range(self.max_retries):
            try:
                logger.debug(f"Fetching events for {date_str} (attempt {attempt + 1})")

                # Fetch events table for this date
                # gdeltPyR returns events when table='events' (default)
                df = self.gd.Search(date_str, table="events", coverage=True)

                if df is None or (hasattr(df, "empty") and df.empty):
                    logger.warning(f"No events returned for {date_str}")
                    return pd.DataFrame()

                # Filter by QuadClass if specified
                if quad_classes and "QuadClass" in df.columns:
                    original_count = len(df)
                    df = df[df["QuadClass"].isin(quad_classes)]
                    logger.debug(
                        f"Filtered {original_count} → {len(df)} events for QuadClasses {quad_classes}"
                    )

                logger.info(f"Collected {len(df)} events for {date_str}")
                return df

            except Exception as e:
                delay = self.base_delay * (2 ** attempt)
                logger.warning(
                    f"Failed to fetch {date_str} (attempt {attempt + 1}/{self.max_retries}): {e}"
                )
                if attempt < self.max_retries - 1:
                    logger.info(f"Retrying in {delay:.1f}s...")
                    time.sleep(delay)

        logger.error(f"All {self.max_retries} attempts failed for {date_str}")
        return None

    def collect_historical_data(
        self,
        start_date: datetime,
        end_date: datetime,
        quad_classes: Optional[list[int]] = None,
        save_daily: bool = True,
    ) -> pd.DataFrame:
        """
        Collect historical GDELT events over a date range.

        Args:
            start_date: Start of date range (inclusive)
            end_date: End of date range (inclusive)
            quad_classes: List of QuadClass values (1-4), None for all
            save_daily: Whether to save each day as separate CSV

        Returns:
            Combined DataFrame of all collected events
        """
        all_events = []
        current = start_date
        days_total = (end_date - start_date).days + 1
        days_processed = 0
        days_failed = []

        logger.info(
            f"Starting historical collection: {start_date.date()} to {end_date.date()} "
            f"({days_total} days)"
        )

        while current <= end_date:
            date_str = current.strftime("%Y-%m-%d")
            days_processed += 1

            logger.info(f"Processing day {days_processed}/{days_total}: {date_str}")

            df = self.collect_single_day(current, quad_classes)

            if df is not None and not df.empty:
                if save_daily:
                    output_path = self.output_dir / f"gdelt_{date_str}.csv"
                    df.to_csv(output_path, index=False)
                    logger.info(f"Saved {len(df)} events to {output_path.name}")

                all_events.append(df)
            elif df is None:
                days_failed.append(date_str)

            # Rate limiting between days
            time.sleep(self.base_delay)
            current += timedelta(days=1)

        if days_failed:
            logger.warning(f"Failed to collect {len(days_failed)} days: {days_failed}")

        if not all_events:
            logger.error("No events collected!")
            return pd.DataFrame()

        combined = pd.concat(all_events, ignore_index=True)
        logger.info(
            f"Collection complete: {len(combined)} total events from "
            f"{days_total - len(days_failed)}/{days_total} days"
        )

        return combined

    def collect_last_n_days(
        self,
        n_days: int = 30,
        quad_classes: Optional[list[int]] = None,
    ) -> pd.DataFrame:
        """
        Convenience method to collect the last N days of data.

        Args:
            n_days: Number of days to collect
            quad_classes: List of QuadClass values (1-4), None for all

        Returns:
            Combined DataFrame of events
        """
        end_date = datetime.now() - timedelta(days=1)  # Yesterday
        start_date = end_date - timedelta(days=n_days - 1)

        return self.collect_historical_data(
            start_date=start_date,
            end_date=end_date,
            quad_classes=quad_classes,
            save_daily=True,
        )
