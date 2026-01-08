"""
Filtering functions for GDELT events using QuadClass and quality metrics.
"""

import logging
from typing import List, Optional, Dict, Any, Tuple
import pandas as pd
import numpy as np

from .constants import (
    QUADCLASS_VERBAL_COOPERATION,
    QUADCLASS_MATERIAL_CONFLICT,
    GDELT100_THRESHOLD,
    MIN_GOLDSTEIN_CONFLICT,
    TONE_RANGE_DIPLOMATIC,
    TONE_RANGE_CONFLICT,
    MIN_SOURCES_THRESHOLD,
    MIN_TONE_CONFIDENCE,
    MAX_TONE_CONFIDENCE,
    CONFLICT_EVENT_CODES,
    DIPLOMATIC_EVENT_CODES,
    MAJOR_ACTORS,
)

logger = logging.getLogger(__name__)


def filter_by_quadclass(
    events_df: pd.DataFrame,
    classes: List[int] = [QUADCLASS_VERBAL_COOPERATION, QUADCLASS_MATERIAL_CONFLICT]
) -> pd.DataFrame:
    """
    Filter events by QuadClass categories.

    Args:
        events_df: DataFrame with GDELT events
        classes: List of QuadClass values to keep (1-4)

    Returns:
        Filtered DataFrame containing only specified QuadClasses
    """
    if events_df.empty:
        return events_df

    if 'QuadClass' not in events_df.columns and 'quad_class' not in events_df.columns:
        logger.warning("QuadClass column not found, returning all events")
        return events_df

    # Handle different column naming conventions
    quad_col = 'QuadClass' if 'QuadClass' in events_df.columns else 'quad_class'

    original_count = len(events_df)
    filtered_df = events_df[events_df[quad_col].isin(classes)].copy()
    filtered_count = len(filtered_df)

    # Log statistics
    reduction_pct = ((original_count - filtered_count) / original_count * 100) if original_count > 0 else 0
    logger.info(
        f"QuadClass filter: {original_count} → {filtered_count} events "
        f"({reduction_pct:.1f}% reduction), Classes: {classes}"
    )

    # Log distribution of filtered events
    if filtered_count > 0:
        distribution = filtered_df[quad_col].value_counts().to_dict()
        class_names = {
            1: 'Verbal Cooperation',
            2: 'Material Cooperation',
            3: 'Verbal Conflict',
            4: 'Material Conflict'
        }
        for class_id, count in distribution.items():
            pct = (count / filtered_count) * 100
            name = class_names.get(class_id, f'Class {class_id}')
            logger.debug(f"  {name}: {count} events ({pct:.1f}%)")

    return filtered_df


def filter_high_confidence(
    events_df: pd.DataFrame,
    min_mentions: int = GDELT100_THRESHOLD
) -> pd.DataFrame:
    """
    Apply GDELT100 quality filtering based on mention count.

    Args:
        events_df: DataFrame with GDELT events
        min_mentions: Minimum number of mentions required

    Returns:
        DataFrame with only high-confidence events
    """
    if events_df.empty:
        return events_df

    # Check for mentions columns (multiple possible names)
    mentions_cols = ['NumMentions', 'num_mentions', 'nummentions']
    mentions_col = None
    for col in mentions_cols:
        if col in events_df.columns:
            mentions_col = col
            break

    if not mentions_col:
        logger.warning("NumMentions column not found, skipping confidence filtering")
        return events_df

    original_count = len(events_df)
    filtered_df = events_df[events_df[mentions_col] >= min_mentions].copy()
    filtered_count = len(filtered_df)

    reduction_pct = ((original_count - filtered_count) / original_count * 100) if original_count > 0 else 0
    logger.info(
        f"GDELT100 filter (>={min_mentions} mentions): "
        f"{original_count} → {filtered_count} events ({reduction_pct:.1f}% reduction)"
    )

    if filtered_count > 0:
        avg_mentions = filtered_df[mentions_col].mean()
        median_mentions = filtered_df[mentions_col].median()
        logger.debug(
            f"  Filtered events: avg={avg_mentions:.1f} mentions, "
            f"median={median_mentions:.0f} mentions"
        )

    return filtered_df


def filter_by_tone_and_goldstein(
    events_df: pd.DataFrame,
    conflict_goldstein: float = MIN_GOLDSTEIN_CONFLICT,
    diplomatic_tone: Tuple[float, float] = TONE_RANGE_DIPLOMATIC
) -> pd.DataFrame:
    """
    Filter events based on tone and Goldstein scale for conflict/diplomatic classification.

    Args:
        events_df: DataFrame with GDELT events
        conflict_goldstein: Maximum Goldstein value for conflicts (negative)
        diplomatic_tone: Tone range tuple for diplomatic events

    Returns:
        DataFrame with classified events
    """
    if events_df.empty:
        return events_df

    filtered_df = events_df.copy()
    has_classification = False

    # Check for Goldstein scale
    goldstein_cols = ['GoldsteinScale', 'goldstein_scale', 'AvgTone']
    goldstein_col = None
    for col in goldstein_cols:
        if col in events_df.columns:
            goldstein_col = col
            break

    # Check for tone column
    tone_col = 'tone' if 'tone' in events_df.columns else 'Tone' if 'Tone' in events_df.columns else None

    # Apply Goldstein filtering if available
    if goldstein_col:
        conflict_mask = filtered_df[goldstein_col] <= conflict_goldstein
        filtered_df.loc[conflict_mask, 'event_classification'] = 'conflict'
        has_classification = True

        conflict_count = conflict_mask.sum()
        if conflict_count > 0:
            logger.debug(
                f"  Goldstein conflict filter: {conflict_count} events with "
                f"Goldstein <= {conflict_goldstein}"
            )

    # Apply tone filtering if available
    if tone_col:
        # Diplomatic events (neutral tone)
        diplomatic_mask = (
            (filtered_df[tone_col] >= diplomatic_tone[0]) &
            (filtered_df[tone_col] <= diplomatic_tone[1])
        )
        filtered_df.loc[diplomatic_mask, 'event_classification'] = 'diplomatic'

        # Conflict events (negative tone)
        conflict_tone_mask = filtered_df[tone_col] < diplomatic_tone[0]
        filtered_df.loc[conflict_tone_mask, 'event_classification'] = 'conflict'

        has_classification = True

        diplomatic_count = diplomatic_mask.sum()
        conflict_tone_count = conflict_tone_mask.sum()
        logger.debug(
            f"  Tone filter: {diplomatic_count} diplomatic "
            f"(tone {diplomatic_tone[0]} to {diplomatic_tone[1]}), "
            f"{conflict_tone_count} conflict (tone < {diplomatic_tone[0]})"
        )

    # Filter out extreme tone values (likely errors)
    if tone_col:
        valid_tone_mask = (
            (filtered_df[tone_col] >= MIN_TONE_CONFIDENCE) &
            (filtered_df[tone_col] <= MAX_TONE_CONFIDENCE)
        )
        invalid_count = (~valid_tone_mask).sum()
        if invalid_count > 0:
            logger.debug(f"  Removing {invalid_count} events with invalid tone values")
            filtered_df = filtered_df[valid_tone_mask]

    # If classification was applied, optionally filter unclassified
    if has_classification and 'event_classification' in filtered_df.columns:
        classified_count = filtered_df['event_classification'].notna().sum()
        total_count = len(filtered_df)
        logger.info(
            f"Tone/Goldstein classification: {classified_count}/{total_count} events classified"
        )

        # Log classification distribution
        if classified_count > 0:
            class_dist = filtered_df['event_classification'].value_counts().to_dict()
            for class_name, count in class_dist.items():
                pct = (count / classified_count) * 100
                logger.debug(f"  {class_name}: {count} ({pct:.1f}%)")

    return filtered_df


def filter_by_actors(
    events_df: pd.DataFrame,
    actor_codes: Optional[List[str]] = None,
    use_major_actors: bool = False
) -> pd.DataFrame:
    """
    Filter events by specific actors or major actors only.

    Args:
        events_df: DataFrame with GDELT events
        actor_codes: Specific actor codes to filter for
        use_major_actors: If True, filter for major world actors only

    Returns:
        DataFrame with events involving specified actors
    """
    if events_df.empty:
        return events_df

    if not actor_codes and not use_major_actors:
        return events_df

    # Use major actors if specified
    if use_major_actors:
        actor_codes = list(MAJOR_ACTORS.keys())

    # Check for actor columns
    actor1_col = 'Actor1Code' if 'Actor1Code' in events_df.columns else 'actor1_code' if 'actor1_code' in events_df.columns else None
    actor2_col = 'Actor2Code' if 'Actor2Code' in events_df.columns else 'actor2_code' if 'actor2_code' in events_df.columns else None

    if not actor1_col and not actor2_col:
        logger.warning("Actor columns not found, skipping actor filtering")
        return events_df

    original_count = len(events_df)

    # Filter for events where either actor matches
    mask = pd.Series([False] * len(events_df))

    if actor1_col:
        mask |= events_df[actor1_col].isin(actor_codes)
    if actor2_col:
        mask |= events_df[actor2_col].isin(actor_codes)

    filtered_df = events_df[mask].copy()
    filtered_count = len(filtered_df)

    reduction_pct = ((original_count - filtered_count) / original_count * 100) if original_count > 0 else 0
    logger.info(
        f"Actor filter: {original_count} → {filtered_count} events "
        f"({reduction_pct:.1f}% reduction)"
    )

    return filtered_df


def filter_by_sources(
    events_df: pd.DataFrame,
    min_sources: int = MIN_SOURCES_THRESHOLD
) -> pd.DataFrame:
    """
    Filter events by minimum number of sources for reliability.

    Args:
        events_df: DataFrame with GDELT events
        min_sources: Minimum number of sources required

    Returns:
        DataFrame with multi-source verified events
    """
    if events_df.empty:
        return events_df

    # Check for sources column
    sources_cols = ['NumSources', 'num_sources', 'numsources']
    sources_col = None
    for col in sources_cols:
        if col in events_df.columns:
            sources_col = col
            break

    if not sources_col:
        logger.debug("NumSources column not found, skipping source filtering")
        return events_df

    original_count = len(events_df)
    filtered_df = events_df[events_df[sources_col] >= min_sources].copy()
    filtered_count = len(filtered_df)

    reduction_pct = ((original_count - filtered_count) / original_count * 100) if original_count > 0 else 0
    logger.info(
        f"Source filter (>={min_sources} sources): "
        f"{original_count} → {filtered_count} events ({reduction_pct:.1f}% reduction)"
    )

    return filtered_df


def combine_filters(
    events_df: pd.DataFrame,
    use_quadclass: bool = True,
    quadclasses: List[int] = [QUADCLASS_VERBAL_COOPERATION, QUADCLASS_MATERIAL_CONFLICT],
    use_confidence: bool = True,
    min_mentions: int = GDELT100_THRESHOLD,
    use_tone: bool = True,
    use_sources: bool = False,
    min_sources: int = MIN_SOURCES_THRESHOLD,
    use_actors: bool = False,
    actor_codes: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Apply all filtering stages with statistics tracking.

    Args:
        events_df: DataFrame with GDELT events
        use_quadclass: Apply QuadClass filtering
        quadclasses: QuadClass values to keep
        use_confidence: Apply GDELT100 confidence filtering
        min_mentions: Minimum mentions for confidence filter
        use_tone: Apply tone and Goldstein filtering
        use_sources: Apply source count filtering
        min_sources: Minimum sources required
        use_actors: Apply actor filtering
        actor_codes: Specific actors to filter for

    Returns:
        Fully filtered DataFrame with statistics logged
    """
    if events_df.empty:
        logger.warning("No events to filter")
        return events_df

    original_count = len(events_df)
    filtered_df = events_df.copy()

    logger.info(f"Starting combined filtering on {original_count} events")
    logger.info("-" * 50)

    # Track reduction at each step
    reductions = []

    # 1. QuadClass filtering
    if use_quadclass:
        prev_count = len(filtered_df)
        filtered_df = filter_by_quadclass(filtered_df, quadclasses)
        reductions.append(('QuadClass', prev_count - len(filtered_df)))

    # 2. Confidence filtering (GDELT100)
    if use_confidence:
        prev_count = len(filtered_df)
        filtered_df = filter_high_confidence(filtered_df, min_mentions)
        reductions.append(('Confidence', prev_count - len(filtered_df)))

    # 3. Tone and Goldstein filtering
    if use_tone:
        prev_count = len(filtered_df)
        filtered_df = filter_by_tone_and_goldstein(filtered_df)
        reductions.append(('Tone/Goldstein', prev_count - len(filtered_df)))

    # 4. Source filtering
    if use_sources:
        prev_count = len(filtered_df)
        filtered_df = filter_by_sources(filtered_df, min_sources)
        reductions.append(('Sources', prev_count - len(filtered_df)))

    # 5. Actor filtering
    if use_actors:
        prev_count = len(filtered_df)
        filtered_df = filter_by_actors(filtered_df, actor_codes)
        reductions.append(('Actors', prev_count - len(filtered_df)))

    # Summary statistics
    final_count = len(filtered_df)
    total_reduction = original_count - final_count
    reduction_pct = (total_reduction / original_count * 100) if original_count > 0 else 0

    logger.info("-" * 50)
    logger.info(f"Filtering complete: {original_count} → {final_count} events")
    logger.info(f"Total reduction: {total_reduction} events ({reduction_pct:.1f}%)")

    logger.info("\nReduction by filter:")
    for filter_name, reduction in reductions:
        if reduction > 0:
            pct = (reduction / original_count) * 100
            logger.info(f"  {filter_name}: -{reduction} ({pct:.1f}%)")

    return filtered_df


def get_filter_statistics(events_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate statistics about filtering potential.

    Args:
        events_df: DataFrame to analyze

    Returns:
        Dictionary with filter statistics
    """
    stats = {
        'total_events': len(events_df),
        'filters_available': [],
        'potential_reduction': {}
    }

    # Check QuadClass distribution
    if 'QuadClass' in events_df.columns or 'quad_class' in events_df.columns:
        quad_col = 'QuadClass' if 'QuadClass' in events_df.columns else 'quad_class'
        stats['filters_available'].append('QuadClass')

        # Count events that would be filtered
        target_classes = [QUADCLASS_VERBAL_COOPERATION, QUADCLASS_MATERIAL_CONFLICT]
        would_remain = events_df[quad_col].isin(target_classes).sum()
        would_filter = len(events_df) - would_remain
        stats['potential_reduction']['QuadClass'] = {
            'would_filter': would_filter,
            'would_remain': would_remain,
            'reduction_pct': (would_filter / len(events_df) * 100) if len(events_df) > 0 else 0
        }

    # Check confidence filtering potential
    mentions_col = None
    for col in ['NumMentions', 'num_mentions']:
        if col in events_df.columns:
            mentions_col = col
            break

    if mentions_col:
        stats['filters_available'].append('Confidence')
        would_remain = (events_df[mentions_col] >= GDELT100_THRESHOLD).sum()
        would_filter = len(events_df) - would_remain
        stats['potential_reduction']['Confidence'] = {
            'would_filter': would_filter,
            'would_remain': would_remain,
            'reduction_pct': (would_filter / len(events_df) * 100) if len(events_df) > 0 else 0
        }

    # Check tone filtering potential
    if 'tone' in events_df.columns or 'Tone' in events_df.columns:
        stats['filters_available'].append('Tone')
        tone_col = 'tone' if 'tone' in events_df.columns else 'Tone'

        # Count valid tone range
        valid_tone = (
            (events_df[tone_col] >= MIN_TONE_CONFIDENCE) &
            (events_df[tone_col] <= MAX_TONE_CONFIDENCE)
        )
        would_remain = valid_tone.sum()
        would_filter = len(events_df) - would_remain
        stats['potential_reduction']['Tone'] = {
            'would_filter': would_filter,
            'would_remain': would_remain,
            'reduction_pct': (would_filter / len(events_df) * 100) if len(events_df) > 0 else 0
        }

    return stats