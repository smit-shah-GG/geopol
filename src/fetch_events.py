"""
Event fetching with filtering for conflicts and diplomatic events.
"""

import logging
from typing import Optional, Tuple
import pandas as pd
from datetime import datetime

from .gdelt_client import GDELTClient
from .config import (
    MIN_GOLDSTEIN_CONFLICT,
    TONE_RANGE_DIPLOMATIC,
    QUADCLASS_VERBAL_COOPERATION,
    QUADCLASS_MATERIAL_CONFLICT,
)

logger = logging.getLogger(__name__)


def fetch_conflict_diplomatic_events(
    date_str: Optional[str] = None,
    timespan: Optional[str] = "24h",
    client: Optional[GDELTClient] = None,
) -> pd.DataFrame:
    """
    Fetch conflict and diplomatic events with tone-based filtering.

    Args:
        date_str: Optional date string in format "YYYY-MM-DD" (deprecated - use timespan)
        timespan: Time window to fetch (e.g., "24h", "7d", "1w")
        client: Optional GDELTClient instance (creates new if not provided)

    Returns:
        DataFrame with filtered and deduplicated events
    """
    if client is None:
        client = GDELTClient()

    try:
        logger.info(f"Fetching events for timespan: {timespan}")

        # Fetch events with theme filtering using timespan
        # Timespan is more reliable than specific dates for recent data
        articles_df = client.fetch_recent_events(
            timespan=timespan,
            themes="(DIPLOMATIC_EXCHANGE OR ARMED_CONFLICT OR MILITARY_CONFLICT OR NEGOTIATE OR PEACE_PROCESS)",
        )

        if articles_df.empty:
            logger.warning(f"No events found for timespan: {timespan}")
            return pd.DataFrame()

        logger.info(f"Retrieved {len(articles_df)} raw articles for timespan: {timespan}")

        # Apply tone-based filtering if tone column exists
        if 'tone' in articles_df.columns:
            filtered_df = filter_by_tone(articles_df)
        else:
            logger.warning("Tone column not found, skipping tone filtering")
            filtered_df = articles_df

        # Deduplicate results
        deduplicated_df = deduplicate_articles(filtered_df)

        logger.info(
            f"Final event count: {len(deduplicated_df)} "
            f"(reduced from {len(articles_df)} raw articles)"
        )

        return deduplicated_df

    except ValueError as e:
        logger.error(f"Invalid parameters: {e}")
        raise
    except Exception as e:
        logger.error(f"Failed to fetch events: {e}")
        raise


def filter_by_tone(articles_df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter articles by tone to identify conflicts and diplomatic events.

    Args:
        articles_df: DataFrame with GDELT articles containing 'tone' column

    Returns:
        Filtered DataFrame with conflict and diplomatic events
    """
    if articles_df.empty or 'tone' not in articles_df.columns:
        return articles_df

    original_count = len(articles_df)

    # Create tone-based categories
    conflict_mask = articles_df['tone'] < MIN_GOLDSTEIN_CONFLICT
    diplomatic_mask = (
        (articles_df['tone'] >= TONE_RANGE_DIPLOMATIC[0]) &
        (articles_df['tone'] <= TONE_RANGE_DIPLOMATIC[1])
    )

    # Combine masks
    filtered_df = articles_df[conflict_mask | diplomatic_mask].copy()

    # Add event type classification
    filtered_df.loc[:, 'event_type'] = 'other'
    filtered_df.loc[conflict_mask, 'event_type'] = 'conflict'
    filtered_df.loc[diplomatic_mask, 'event_type'] = 'diplomatic'

    conflict_count = (filtered_df['event_type'] == 'conflict').sum()
    diplomatic_count = (filtered_df['event_type'] == 'diplomatic').sum()

    logger.info(
        f"Tone filtering: {original_count} → {len(filtered_df)} events "
        f"(conflicts: {conflict_count}, diplomatic: {diplomatic_count})"
    )

    return filtered_df


def deduplicate_articles(articles_df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate articles based on URL and title similarity.

    Args:
        articles_df: DataFrame with GDELT articles

    Returns:
        Deduplicated DataFrame
    """
    if articles_df.empty:
        return articles_df

    original_count = len(articles_df)

    # First pass: Remove exact URL duplicates
    if 'url' in articles_df.columns:
        articles_df = articles_df.drop_duplicates(subset=['url'], keep='first')
        logger.debug(f"URL deduplication: {original_count} → {len(articles_df)}")

    # Second pass: Remove duplicate titles (case-insensitive)
    if 'title' in articles_df.columns:
        # Create lowercase title for comparison
        articles_df['title_lower'] = articles_df['title'].str.lower()
        articles_df = articles_df.drop_duplicates(subset=['title_lower'], keep='first')
        articles_df = articles_df.drop(columns=['title_lower'])

        final_count = len(articles_df)
        logger.info(f"Deduplication complete: {original_count} → {final_count} articles")

    return articles_df


def get_event_statistics(events_df: pd.DataFrame) -> dict:
    """
    Calculate statistics about the events DataFrame.

    Args:
        events_df: DataFrame with events

    Returns:
        Dictionary with event statistics
    """
    stats = {
        'total_events': len(events_df),
        'unique_urls': events_df['url'].nunique() if 'url' in events_df.columns else 0,
        'unique_domains': events_df['domain'].nunique() if 'domain' in events_df.columns else 0,
    }

    # Add event type distribution if available
    if 'event_type' in events_df.columns:
        type_counts = events_df['event_type'].value_counts().to_dict()
        stats['event_types'] = type_counts

    # Add tone statistics if available
    if 'tone' in events_df.columns:
        stats['tone_stats'] = {
            'mean': float(events_df['tone'].mean()),
            'std': float(events_df['tone'].std()),
            'min': float(events_df['tone'].min()),
            'max': float(events_df['tone'].max()),
        }

    # Add QuadClass distribution if available
    if 'QuadClass' in events_df.columns:
        quad_counts = events_df['QuadClass'].value_counts().to_dict()
        stats['quad_classes'] = quad_counts

    return stats