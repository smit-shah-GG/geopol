#!/usr/bin/env python
"""
Test script demonstrating sampling on large dataset.
"""

import sys
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.sampling import (
    sample_events_stratified,
    adaptive_sampling,
    progressive_sampling,
    time_stratified_sampling,
    get_sampling_statistics
)
from src.filtering import combine_filters, get_filter_statistics
from src.fetch_events import fetch_conflict_diplomatic_events
from src.gdelt_client import GDELTClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_fake_events(n: int = 50000, seed: int = 42) -> pd.DataFrame:
    """Generate large fake dataset for testing."""
    np.random.seed(seed)

    # Generate events with realistic distribution
    quad_classes = np.random.choice(
        [1, 2, 3, 4],
        size=n,
        p=[0.25, 0.15, 0.30, 0.30]  # More conflicts than cooperation
    )

    # Generate mentions with power law distribution
    num_mentions = np.random.pareto(2, n) * 50
    num_mentions = np.clip(num_mentions, 1, 10000).astype(int)

    # Generate tone correlated with QuadClass
    tone = np.zeros(n)
    for i in range(n):
        if quad_classes[i] in [1, 2]:  # Cooperation
            tone[i] = np.random.normal(0, 2)
        else:  # Conflict
            tone[i] = np.random.normal(-5, 3)
    tone = np.clip(tone, -10, 10)

    # Generate timestamps over 7 days
    base_date = datetime.now() - timedelta(days=7)
    timestamps = [
        base_date + timedelta(seconds=np.random.randint(0, 7*24*3600))
        for _ in range(n)
    ]

    # Create DataFrame
    df = pd.DataFrame({
        'event_id': range(n),
        'QuadClass': quad_classes,
        'NumMentions': num_mentions,
        'tone': tone,
        'event_date': timestamps,
        'actor1_code': np.random.choice(['USA', 'CHN', 'RUS', 'GBR', 'FRA'], n),
        'actor2_code': np.random.choice(['USA', 'CHN', 'RUS', 'GBR', 'FRA'], n),
    })

    return df.sort_values('event_date').reset_index(drop=True)


def test_sampling():
    """Test sampling functionality."""

    logger.info("=" * 60)
    logger.info("GDELT Sampling System Test")
    logger.info("=" * 60)

    # Option 1: Use real GDELT data
    use_real_data = False

    if use_real_data:
        logger.info("\n1. Fetching real GDELT data...")
        client = GDELTClient()
        events_df = fetch_conflict_diplomatic_events(timespan="7d", client=client)

        if events_df.empty:
            logger.warning("No real data available, switching to synthetic data")
            use_real_data = False

    if not use_real_data:
        # Option 2: Generate synthetic data
        logger.info("\n1. Generating synthetic test data...")
        events_df = generate_fake_events(50000)
        logger.info(f"Generated {len(events_df)} fake events for testing")

    # Show original distribution
    logger.info("\n2. Original data statistics:")
    logger.info(f"  Total events: {len(events_df)}")

    if 'QuadClass' in events_df.columns:
        dist = events_df['QuadClass'].value_counts()
        logger.info("  QuadClass distribution:")
        for class_id, count in dist.items():
            pct = (count / len(events_df)) * 100
            logger.info(f"    Class {class_id}: {count} ({pct:.1f}%)")

    if 'NumMentions' in events_df.columns:
        logger.info(f"  Average mentions: {events_df['NumMentions'].mean():.1f}")
        logger.info(f"  High-confidence (100+ mentions): {(events_df['NumMentions'] >= 100).sum()}")

    # Test 1: Stratified Sampling
    logger.info("\n3. Testing stratified sampling (target: 10,000 events)...")
    sampled_stratified = sample_events_stratified(
        events_df,
        max_events=10000,
        stratify_by='QuadClass',
        prioritize_by='NumMentions'
    )

    stats = get_sampling_statistics(events_df, sampled_stratified)
    logger.info(f"  Sampled: {stats['sampled_size']} events ({stats['sampling_rate']:.1%})")
    logger.info(f"  Distribution similarity: {stats.get('distribution_similarity', 0):.3f}")

    # Verify distribution preserved
    if 'distribution_comparison' in stats:
        logger.info("  Class distribution comparison:")
        for class_key, comparison in stats['distribution_comparison'].items():
            class_num = class_key.split('_')[1]
            logger.info(
                f"    Class {class_num}: "
                f"Original {comparison['original']:.1%} → "
                f"Sampled {comparison['sampled']:.1%} "
                f"(diff: {comparison['difference']:.1%})"
            )

    # Verify high-mention events prioritized
    if 'NumMentions' in sampled_stratified.columns:
        avg_mentions_original = events_df['NumMentions'].mean()
        avg_mentions_sampled = sampled_stratified['NumMentions'].mean()
        logger.info(
            f"  Average mentions: Original {avg_mentions_original:.1f} → "
            f"Sampled {avg_mentions_sampled:.1f}"
        )

        if avg_mentions_sampled > avg_mentions_original:
            logger.info("  ✓ High-mention events successfully prioritized")

    # Test 2: Adaptive Sampling
    logger.info("\n4. Testing adaptive sampling (compute budget: 5.0)...")
    sampled_adaptive = adaptive_sampling(
        events_df,
        compute_budget=5.0,
        base_cost_per_event=0.001
    )
    logger.info(f"  Adaptive sampling selected: {len(sampled_adaptive)} events")

    # Test 3: Progressive Sampling
    logger.info("\n5. Testing progressive sampling...")
    sampled_progressive = progressive_sampling(
        events_df,
        initial_sample=1000,
        max_iterations=5,
        growth_factor=2.0,
        early_stop_metric='diversity'
    )
    logger.info(f"  Progressive sampling selected: {len(sampled_progressive)} events")

    # Test 4: Time-Stratified Sampling
    logger.info("\n6. Testing time-stratified sampling...")
    sampled_time = time_stratified_sampling(
        events_df,
        max_events=5000,
        time_column='event_date',
        time_bins=24  # Daily bins for week of data
    )
    logger.info(f"  Time-stratified sampling selected: {len(sampled_time)} events")

    # Test 5: Combined Filtering + Sampling
    logger.info("\n7. Testing combined filtering + sampling pipeline...")

    # First apply filters
    logger.info("  Applying filters...")
    filtered_df = combine_filters(
        events_df,
        use_quadclass=True,
        quadclasses=[1, 4],  # Verbal cooperation + Material conflict
        use_confidence=True,
        min_mentions=100,
        use_tone=True
    )

    # Then apply sampling if still too large
    if len(filtered_df) > 5000:
        logger.info(f"  Filtered data still large ({len(filtered_df)} events), applying sampling...")
        final_df = sample_events_stratified(filtered_df, max_events=5000)
    else:
        final_df = filtered_df

    logger.info(f"  Final pipeline result: {len(final_df)} events")
    logger.info(f"  Reduction: {len(events_df)} → {len(final_df)} ({(len(final_df)/len(events_df)*100):.1f}% retained)")

    # Summary
    logger.info("\n8. Summary of sampling methods:")
    logger.info("-" * 40)
    methods = [
        ("Stratified", len(sampled_stratified)),
        ("Adaptive", len(sampled_adaptive)),
        ("Progressive", len(sampled_progressive)),
        ("Time-stratified", len(sampled_time)),
        ("Filtered+Sampled", len(final_df))
    ]

    for method, count in methods:
        retention = (count / len(events_df)) * 100
        logger.info(f"  {method:20}: {count:6} events ({retention:5.1f}%)")

    logger.info("\n" + "=" * 60)
    logger.info("Sampling test completed successfully!")
    logger.info("=" * 60)


if __name__ == "__main__":
    test_sampling()