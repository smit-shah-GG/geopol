"""
Stratified sampling strategies for managing GDELT data volume.
"""

import logging
from typing import Optional, Dict, Any, Union
import pandas as pd
import numpy as np
from datetime import datetime

from .constants import (
    DEFAULT_SAMPLE_SIZE,
    MIN_CLASS_REPRESENTATION,
    QUADCLASS_VERBAL_COOPERATION,
    QUADCLASS_MATERIAL_COOPERATION,
    QUADCLASS_VERBAL_CONFLICT,
    QUADCLASS_MATERIAL_CONFLICT,
)

logger = logging.getLogger(__name__)


def sample_events_stratified(
    events_df: pd.DataFrame,
    max_events: int = DEFAULT_SAMPLE_SIZE,
    stratify_by: str = 'QuadClass',
    min_class_ratio: float = MIN_CLASS_REPRESENTATION,
    prioritize_by: Optional[str] = 'NumMentions',
) -> pd.DataFrame:
    """
    Perform stratified sampling while preserving class distribution.

    Args:
        events_df: DataFrame with events to sample
        max_events: Maximum number of events to return
        stratify_by: Column to use for stratification
        min_class_ratio: Minimum representation for minority classes
        prioritize_by: Column to use for prioritization within classes

    Returns:
        Sampled DataFrame maintaining class distribution
    """
    if events_df.empty:
        return events_df

    if len(events_df) <= max_events:
        logger.info(f"No sampling needed: {len(events_df)} events <= {max_events}")
        return events_df

    # Check if stratification column exists
    if stratify_by not in events_df.columns:
        # Try lowercase version
        stratify_by_lower = stratify_by.lower()
        if stratify_by_lower in events_df.columns:
            stratify_by = stratify_by_lower
        else:
            logger.warning(f"Column '{stratify_by}' not found, using random sampling")
            return events_df.sample(n=max_events, random_state=42)

    # Calculate class distribution in original data
    class_counts = events_df[stratify_by].value_counts()
    class_ratios = class_counts / len(events_df)

    logger.info(f"Original distribution ({len(events_df)} events):")
    for class_val, count in class_counts.items():
        logger.info(f"  Class {class_val}: {count} ({class_ratios[class_val]:.1%})")

    # Calculate target samples per class
    samples_per_class = {}
    allocated = 0

    # First pass: Allocate proportionally
    for class_val, ratio in class_ratios.items():
        target = int(max_events * ratio)
        # Ensure minimum representation
        min_samples = max(1, int(max_events * min_class_ratio))
        samples_per_class[class_val] = max(min_samples, target)
        allocated += samples_per_class[class_val]

    # Adjust if over-allocated
    if allocated > max_events:
        # Reduce largest classes proportionally
        excess = allocated - max_events
        sorted_classes = sorted(samples_per_class.items(), key=lambda x: x[1], reverse=True)

        for class_val, samples in sorted_classes:
            if excess <= 0:
                break
            # Don't reduce below minimum
            min_samples = max(1, int(max_events * min_class_ratio))
            can_reduce = samples - min_samples
            if can_reduce > 0:
                reduction = min(can_reduce, excess)
                samples_per_class[class_val] -= reduction
                excess -= reduction

    # Perform sampling
    sampled_dfs = []

    for class_val, n_samples in samples_per_class.items():
        class_df = events_df[events_df[stratify_by] == class_val]

        # Limit samples to available events
        n_samples = min(n_samples, len(class_df))

        if n_samples > 0:
            # Prioritize if column specified
            if prioritize_by and prioritize_by in class_df.columns:
                # Handle different prioritization column names
                priority_col = prioritize_by
                if prioritize_by == 'NumMentions' and prioritize_by not in class_df.columns:
                    if 'num_mentions' in class_df.columns:
                        priority_col = 'num_mentions'

                if priority_col in class_df.columns:
                    # Sort by priority (descending) and take top n
                    class_df_sorted = class_df.sort_values(priority_col, ascending=False)
                    sampled = class_df_sorted.head(n_samples)
                else:
                    sampled = class_df.sample(n=n_samples, random_state=42)
            else:
                sampled = class_df.sample(n=n_samples, random_state=42)

            sampled_dfs.append(sampled)

    # Combine sampled data
    sampled_df = pd.concat(sampled_dfs, ignore_index=True)

    # Maintain temporal order if event_date exists
    if 'event_date' in sampled_df.columns:
        sampled_df = sampled_df.sort_values('event_date')
    elif 'seendate' in sampled_df.columns:
        sampled_df = sampled_df.sort_values('seendate')

    # Log sampling results
    sampled_counts = sampled_df[stratify_by].value_counts()
    sampled_ratios = sampled_counts / len(sampled_df)

    logger.info(f"Sampled distribution ({len(sampled_df)} events):")
    for class_val, count in sampled_counts.items():
        original_ratio = class_ratios.get(class_val, 0)
        sampled_ratio = sampled_ratios[class_val]
        logger.info(
            f"  Class {class_val}: {count} ({sampled_ratio:.1%}) "
            f"[original: {original_ratio:.1%}]"
        )

    return sampled_df


def adaptive_sampling(
    events_df: pd.DataFrame,
    compute_budget: float,
    base_cost_per_event: float = 0.001,
    complexity_factors: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """
    Adjust sample size based on available compute resources.

    Args:
        events_df: DataFrame with events
        compute_budget: Available compute budget (abstract units)
        base_cost_per_event: Base processing cost per event
        complexity_factors: Multipliers for different event types

    Returns:
        Sampled DataFrame within compute budget
    """
    if events_df.empty:
        return events_df

    # Default complexity factors if not provided
    if complexity_factors is None:
        complexity_factors = {
            1: 1.0,   # Verbal cooperation - simple
            2: 1.2,   # Material cooperation - moderate
            3: 1.1,   # Verbal conflict - simple
            4: 1.5,   # Material conflict - complex
        }

    # Calculate processing cost for each event
    events_df = events_df.copy()
    events_df['processing_cost'] = base_cost_per_event

    # Apply complexity factors if QuadClass exists
    quad_col = None
    for col in ['QuadClass', 'quad_class']:
        if col in events_df.columns:
            quad_col = col
            break

    if quad_col:
        for class_val, factor in complexity_factors.items():
            mask = events_df[quad_col] == class_val
            events_df.loc[mask, 'processing_cost'] *= factor

    # Sort by cost-effectiveness (high priority, low cost)
    priority_col = None
    for col in ['NumMentions', 'num_mentions']:
        if col in events_df.columns:
            priority_col = col
            break

    if priority_col:
        # Calculate cost-effectiveness score
        events_df['cost_effectiveness'] = events_df[priority_col] / events_df['processing_cost']
        events_df = events_df.sort_values('cost_effectiveness', ascending=False)
    else:
        # Sort by processing cost (ascending) if no priority metric
        events_df = events_df.sort_values('processing_cost')

    # Select events within budget
    events_df['cumulative_cost'] = events_df['processing_cost'].cumsum()
    sampled_df = events_df[events_df['cumulative_cost'] <= compute_budget].copy()

    # Clean up temporary columns
    sampled_df = sampled_df.drop(columns=['processing_cost', 'cumulative_cost'], errors='ignore')
    if 'cost_effectiveness' in sampled_df.columns:
        sampled_df = sampled_df.drop(columns=['cost_effectiveness'])

    # Log results
    logger.info(
        f"Adaptive sampling: {len(sampled_df)} events selected from {len(events_df)} "
        f"within budget {compute_budget:.2f}"
    )

    if len(sampled_df) < len(events_df):
        reduction_pct = ((len(events_df) - len(sampled_df)) / len(events_df)) * 100
        logger.info(f"  Budget constraint reduced events by {reduction_pct:.1f}%")

    return sampled_df


def progressive_sampling(
    events_df: pd.DataFrame,
    initial_sample: int = 1000,
    max_iterations: int = 5,
    growth_factor: float = 2.0,
    early_stop_metric: Optional[str] = 'diversity',
    early_stop_threshold: float = 0.95,
) -> pd.DataFrame:
    """
    Progressive sampling that increases sample size until diversity plateaus.

    Args:
        events_df: DataFrame with events
        initial_sample: Initial sample size
        max_iterations: Maximum sampling iterations
        growth_factor: Sample size multiplier per iteration
        early_stop_metric: Metric for early stopping ('diversity' or 'coverage')
        early_stop_threshold: Threshold for early stopping

    Returns:
        Progressively sampled DataFrame
    """
    if events_df.empty or len(events_df) <= initial_sample:
        return events_df

    current_sample_size = initial_sample
    best_sample = None
    prev_metric = 0

    for iteration in range(max_iterations):
        # Don't exceed available data
        current_sample_size = min(current_sample_size, len(events_df))

        # Sample data
        sampled = events_df.sample(n=current_sample_size, random_state=42 + iteration)

        # Calculate stopping metric
        if early_stop_metric == 'diversity':
            # Measure diversity as unique value coverage
            metric = 0
            diversity_cols = ['actor1_code', 'actor2_code', 'event_code']
            valid_cols = [col for col in diversity_cols if col in sampled.columns]

            if valid_cols:
                for col in valid_cols:
                    if col in events_df.columns:
                        unique_sampled = sampled[col].nunique()
                        unique_total = events_df[col].nunique()
                        if unique_total > 0:
                            metric += unique_sampled / unique_total

                metric /= len(valid_cols) if valid_cols else 1
            else:
                # Fallback to simple ratio
                metric = len(sampled) / len(events_df)

        else:  # coverage
            # Measure coverage as proportion of data sampled
            metric = len(sampled) / len(events_df)

        logger.debug(
            f"Iteration {iteration + 1}: Sample size={current_sample_size}, "
            f"{early_stop_metric}={metric:.3f}"
        )

        # Check for early stopping
        if metric >= early_stop_threshold:
            logger.info(
                f"Early stop at iteration {iteration + 1}: "
                f"{early_stop_metric}={metric:.3f} >= {early_stop_threshold}"
            )
            best_sample = sampled
            break

        # Check if improvement is marginal
        if iteration > 0 and (metric - prev_metric) < 0.01:
            logger.info(
                f"Marginal improvement at iteration {iteration + 1}: "
                f"delta={metric - prev_metric:.4f}"
            )
            best_sample = sampled
            break

        best_sample = sampled
        prev_metric = metric

        # Increase sample size
        current_sample_size = int(current_sample_size * growth_factor)

    logger.info(
        f"Progressive sampling complete: {len(best_sample)} events selected "
        f"from {len(events_df)}"
    )

    return best_sample


def time_stratified_sampling(
    events_df: pd.DataFrame,
    max_events: int = DEFAULT_SAMPLE_SIZE,
    time_column: str = 'event_date',
    time_bins: int = 24,  # Default to hourly bins for a day
) -> pd.DataFrame:
    """
    Sample events stratified by time periods to maintain temporal distribution.

    Args:
        events_df: DataFrame with events
        max_events: Maximum number of events
        time_column: Column containing timestamps
        time_bins: Number of time bins to create

    Returns:
        Time-stratified sample
    """
    if events_df.empty or len(events_df) <= max_events:
        return events_df

    # Check for time column
    if time_column not in events_df.columns:
        logger.warning(f"Time column '{time_column}' not found, using random sampling")
        return events_df.sample(n=max_events, random_state=42)

    # Convert to datetime if needed
    events_df = events_df.copy()
    if not pd.api.types.is_datetime64_any_dtype(events_df[time_column]):
        try:
            events_df[time_column] = pd.to_datetime(events_df[time_column])
        except:
            logger.warning(f"Could not convert {time_column} to datetime, using random sampling")
            return events_df.sample(n=max_events, random_state=42)

    # Create time bins
    events_df['time_bin'] = pd.cut(events_df[time_column], bins=time_bins, labels=False)

    # Sample stratified by time bins
    sampled_df = sample_events_stratified(
        events_df,
        max_events=max_events,
        stratify_by='time_bin',
        prioritize_by='NumMentions' if 'NumMentions' in events_df.columns else None
    )

    # Remove temporary column
    sampled_df = sampled_df.drop(columns=['time_bin'], errors='ignore')

    logger.info(
        f"Time-stratified sampling: {len(sampled_df)} events from {len(events_df)} "
        f"across {time_bins} time periods"
    )

    return sampled_df


def get_sampling_statistics(events_df: pd.DataFrame, sampled_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate statistics comparing original and sampled datasets.

    Args:
        events_df: Original DataFrame
        sampled_df: Sampled DataFrame

    Returns:
        Dictionary with comparison statistics
    """
    stats = {
        'original_size': len(events_df),
        'sampled_size': len(sampled_df),
        'sampling_rate': len(sampled_df) / len(events_df) if len(events_df) > 0 else 0,
        'reduction_pct': ((len(events_df) - len(sampled_df)) / len(events_df) * 100)
                        if len(events_df) > 0 else 0
    }

    # Compare distributions
    quad_col = None
    for col in ['QuadClass', 'quad_class']:
        if col in events_df.columns and col in sampled_df.columns:
            quad_col = col
            break

    if quad_col:
        original_dist = events_df[quad_col].value_counts(normalize=True)
        sampled_dist = sampled_df[quad_col].value_counts(normalize=True)

        stats['distribution_comparison'] = {}
        for class_val in original_dist.index:
            stats['distribution_comparison'][f'class_{class_val}'] = {
                'original': original_dist.get(class_val, 0),
                'sampled': sampled_dist.get(class_val, 0),
                'difference': abs(original_dist.get(class_val, 0) - sampled_dist.get(class_val, 0))
            }

        # Calculate distribution similarity (1 - max difference)
        max_diff = max([v['difference'] for v in stats['distribution_comparison'].values()])
        stats['distribution_similarity'] = 1 - max_diff

    # Compare average metrics
    metrics_cols = ['NumMentions', 'num_mentions', 'tone']
    for col in metrics_cols:
        if col in events_df.columns and col in sampled_df.columns:
            col_clean = col.lower().replace('num', '').replace('_', '')
            stats[f'avg_{col_clean}_original'] = events_df[col].mean()
            stats[f'avg_{col_clean}_sampled'] = sampled_df[col].mean()

    return stats