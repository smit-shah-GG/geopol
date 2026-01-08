"""
Monitoring and data quality metrics for GDELT pipeline.
"""

import logging
import json
import time
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path
import pandas as pd

from .constants import GDELT100_THRESHOLD

logger = logging.getLogger(__name__)


class DataQualityMonitor:
    """Tracks data quality metrics throughout the pipeline."""

    def __init__(self, metrics_dir: str = "data/metrics"):
        """
        Initialize monitoring system.

        Args:
            metrics_dir: Directory to save metrics files
        """
        self.metrics_dir = Path(metrics_dir)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

        self.metrics = {
            'pipeline_start': None,
            'pipeline_end': None,
            'stages': {},
            'event_counts': {
                'raw': 0,
                'filtered': 0,
                'deduplicated': 0,
                'sampled': 0,
                'stored': 0
            },
            'quality_metrics': {
                'duplicate_rate': 0.0,
                'high_confidence_rate': 0.0,
                'quadclass_distribution': {},
                'average_tone': None,
                'average_mentions': None,
                'estimated_accuracy': 0.55  # Baseline from research
            },
            'processing_times': {},
            'errors': []
        }

        self.stage_start_times = {}

    def start_pipeline(self):
        """Mark pipeline start time."""
        self.metrics['pipeline_start'] = datetime.now().isoformat()
        logger.info(f"Pipeline started at {self.metrics['pipeline_start']}")

    def end_pipeline(self):
        """Mark pipeline end time and calculate total duration."""
        self.metrics['pipeline_end'] = datetime.now().isoformat()

        if self.metrics['pipeline_start']:
            start = datetime.fromisoformat(self.metrics['pipeline_start'])
            end = datetime.fromisoformat(self.metrics['pipeline_end'])
            duration = (end - start).total_seconds()
            self.metrics['total_processing_time'] = duration
            logger.info(f"Pipeline completed in {duration:.2f} seconds")

    def start_stage(self, stage_name: str):
        """
        Start timing a pipeline stage.

        Args:
            stage_name: Name of the stage
        """
        self.stage_start_times[stage_name] = time.time()
        logger.debug(f"Started stage: {stage_name}")

    def end_stage(self, stage_name: str, **kwargs):
        """
        End timing a pipeline stage and record metrics.

        Args:
            stage_name: Name of the stage
            **kwargs: Additional metrics to record for this stage
        """
        if stage_name in self.stage_start_times:
            duration = time.time() - self.stage_start_times[stage_name]
            self.metrics['processing_times'][stage_name] = duration

            # Record stage-specific metrics
            if stage_name not in self.metrics['stages']:
                self.metrics['stages'][stage_name] = {}

            self.metrics['stages'][stage_name].update(kwargs)
            self.metrics['stages'][stage_name]['duration'] = duration

            logger.info(f"Stage '{stage_name}' completed in {duration:.2f}s")
            del self.stage_start_times[stage_name]

    def record_event_count(self, stage: str, count: int):
        """
        Record event count at a specific stage.

        Args:
            stage: Stage name (raw, filtered, deduplicated, sampled, stored)
            count: Number of events
        """
        if stage in self.metrics['event_counts']:
            self.metrics['event_counts'][stage] = count
            logger.debug(f"Event count at '{stage}': {count}")

    def calculate_data_quality(self, events_df: pd.DataFrame):
        """
        Calculate data quality metrics from events DataFrame.

        Args:
            events_df: DataFrame with events
        """
        if events_df.empty:
            return

        # Duplicate rate (if deduplication info available)
        if 'is_duplicate' in events_df.columns:
            duplicate_rate = events_df['is_duplicate'].sum() / len(events_df)
            self.metrics['quality_metrics']['duplicate_rate'] = duplicate_rate

        # High confidence rate
        mentions_col = None
        for col in ['NumMentions', 'num_mentions']:
            if col in events_df.columns:
                mentions_col = col
                break

        if mentions_col:
            high_conf = (events_df[mentions_col] >= GDELT100_THRESHOLD).sum()
            high_conf_rate = high_conf / len(events_df) if len(events_df) > 0 else 0
            self.metrics['quality_metrics']['high_confidence_rate'] = high_conf_rate
            self.metrics['quality_metrics']['average_mentions'] = float(events_df[mentions_col].mean())

        # QuadClass distribution
        quad_col = None
        for col in ['QuadClass', 'quad_class']:
            if col in events_df.columns:
                quad_col = col
                break

        if quad_col:
            distribution = events_df[quad_col].value_counts().to_dict()
            self.metrics['quality_metrics']['quadclass_distribution'] = {
                str(k): v for k, v in distribution.items()
            }

        # Average tone
        if 'tone' in events_df.columns or 'Tone' in events_df.columns:
            tone_col = 'tone' if 'tone' in events_df.columns else 'Tone'
            self.metrics['quality_metrics']['average_tone'] = float(events_df[tone_col].mean())

    def calculate_reduction_rates(self):
        """Calculate reduction rates between pipeline stages."""
        counts = self.metrics['event_counts']
        reductions = {}

        if counts['raw'] > 0:
            if counts['filtered'] > 0:
                reductions['filtering_reduction'] = 1 - (counts['filtered'] / counts['raw'])

            if counts['deduplicated'] > 0:
                reductions['deduplication_reduction'] = 1 - (counts['deduplicated'] / counts['filtered'])

            if counts['sampled'] > 0 and counts['deduplicated'] > 0:
                reductions['sampling_reduction'] = 1 - (counts['sampled'] / counts['deduplicated'])

            if counts['stored'] > 0:
                reductions['overall_reduction'] = 1 - (counts['stored'] / counts['raw'])

        self.metrics['reduction_rates'] = reductions

    def add_error(self, error_msg: str, stage: Optional[str] = None):
        """
        Record an error that occurred during processing.

        Args:
            error_msg: Error message
            stage: Optional stage where error occurred
        """
        error_entry = {
            'timestamp': datetime.now().isoformat(),
            'message': error_msg
        }
        if stage:
            error_entry['stage'] = stage

        self.metrics['errors'].append(error_entry)
        logger.error(f"Error in stage '{stage}': {error_msg}")

    def estimate_accuracy(self, high_confidence_rate: Optional[float] = None):
        """
        Estimate accuracy based on data quality indicators.

        Args:
            high_confidence_rate: Proportion of high-confidence events
        """
        # Start with baseline accuracy from research (55%)
        baseline = 0.55

        # Adjust based on high-confidence rate
        if high_confidence_rate is not None:
            # High-confidence events have better accuracy
            accuracy_boost = high_confidence_rate * 0.2  # Up to 20% boost
            self.metrics['quality_metrics']['estimated_accuracy'] = min(0.75, baseline + accuracy_boost)
        else:
            # Use stored high confidence rate if available
            hcr = self.metrics['quality_metrics'].get('high_confidence_rate', 0)
            if hcr > 0:
                accuracy_boost = hcr * 0.2
                self.metrics['quality_metrics']['estimated_accuracy'] = min(0.75, baseline + accuracy_boost)

    def log_metrics(self):
        """Log current metrics to console."""
        logger.info("=" * 60)
        logger.info("Pipeline Metrics Summary")
        logger.info("=" * 60)

        # Event counts
        logger.info("\nEvent Counts:")
        for stage, count in self.metrics['event_counts'].items():
            if count > 0:
                logger.info(f"  {stage:15}: {count:,}")

        # Reduction rates
        if 'reduction_rates' in self.metrics:
            logger.info("\nReduction Rates:")
            for rate_name, rate in self.metrics['reduction_rates'].items():
                logger.info(f"  {rate_name:25}: {rate:.1%}")

        # Quality metrics
        logger.info("\nData Quality:")
        quality = self.metrics['quality_metrics']
        if quality.get('duplicate_rate'):
            logger.info(f"  Duplicate rate: {quality['duplicate_rate']:.1%}")
        if quality.get('high_confidence_rate'):
            logger.info(f"  High confidence (100+ mentions): {quality['high_confidence_rate']:.1%}")
        if quality.get('average_tone') is not None:
            logger.info(f"  Average tone: {quality['average_tone']:.2f}")
        if quality.get('average_mentions') is not None:
            logger.info(f"  Average mentions: {quality['average_mentions']:.1f}")
        if quality.get('estimated_accuracy'):
            logger.info(f"  Estimated accuracy: {quality['estimated_accuracy']:.1%}")

        # QuadClass distribution
        if quality.get('quadclass_distribution'):
            logger.info("\nQuadClass Distribution:")
            class_names = {
                '1': 'Verbal Cooperation',
                '2': 'Material Cooperation',
                '3': 'Verbal Conflict',
                '4': 'Material Conflict'
            }
            for class_id, count in quality['quadclass_distribution'].items():
                name = class_names.get(str(class_id), f'Class {class_id}')
                logger.info(f"  {name:20}: {count}")

        # Processing times
        if self.metrics['processing_times']:
            logger.info("\nProcessing Times:")
            for stage, duration in self.metrics['processing_times'].items():
                logger.info(f"  {stage:20}: {duration:.2f}s")

            if 'total_processing_time' in self.metrics:
                logger.info(f"  {'Total':20}: {self.metrics['total_processing_time']:.2f}s")

        # Errors
        if self.metrics['errors']:
            logger.info(f"\nErrors: {len(self.metrics['errors'])}")
            for error in self.metrics['errors'][:5]:  # Show first 5 errors
                logger.info(f"  - {error['message'][:100]}")

        logger.info("=" * 60)

    def save_metrics(self, filename: Optional[str] = None):
        """
        Save metrics to JSON file.

        Args:
            filename: Optional filename (defaults to timestamp)
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"metrics_{timestamp}.json"

        filepath = self.metrics_dir / filename

        try:
            with open(filepath, 'w') as f:
                json.dump(self.metrics, f, indent=2, default=str)
            logger.info(f"Metrics saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save metrics: {e}")

    def load_metrics(self, filename: str) -> Dict[str, Any]:
        """
        Load metrics from JSON file.

        Args:
            filename: Filename to load

        Returns:
            Loaded metrics dictionary
        """
        filepath = self.metrics_dir / filename

        try:
            with open(filepath, 'r') as f:
                metrics = json.load(f)
            logger.info(f"Metrics loaded from {filepath}")
            return metrics
        except Exception as e:
            logger.error(f"Failed to load metrics: {e}")
            return {}

    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of key metrics.

        Returns:
            Dictionary with summary metrics
        """
        counts = self.metrics['event_counts']
        quality = self.metrics['quality_metrics']

        summary = {
            'events_processed': counts.get('raw', 0),
            'events_stored': counts.get('stored', 0),
            'overall_reduction': 0,
            'duplicate_rate': quality.get('duplicate_rate', 0),
            'high_confidence_rate': quality.get('high_confidence_rate', 0),
            'estimated_accuracy': quality.get('estimated_accuracy', 0.55),
            'total_time': self.metrics.get('total_processing_time', 0),
            'errors': len(self.metrics['errors'])
        }

        if counts.get('raw', 0) > 0 and counts.get('stored', 0) > 0:
            summary['overall_reduction'] = 1 - (counts['stored'] / counts['raw'])

        return summary

    def compare_runs(self, other_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare current metrics with another run.

        Args:
            other_metrics: Metrics from another run

        Returns:
            Comparison dictionary
        """
        comparison = {}

        # Compare event counts
        for stage in self.metrics['event_counts']:
            current = self.metrics['event_counts'][stage]
            other = other_metrics.get('event_counts', {}).get(stage, 0)
            if current > 0 or other > 0:
                comparison[f'{stage}_events'] = {
                    'current': current,
                    'other': other,
                    'difference': current - other,
                    'pct_change': ((current - other) / other * 100) if other > 0 else 0
                }

        # Compare processing times
        current_total = self.metrics.get('total_processing_time', 0)
        other_total = other_metrics.get('total_processing_time', 0)
        if current_total > 0 or other_total > 0:
            comparison['processing_time'] = {
                'current': current_total,
                'other': other_total,
                'difference': current_total - other_total,
                'pct_change': ((current_total - other_total) / other_total * 100) if other_total > 0 else 0
            }

        # Compare quality metrics
        for metric in ['duplicate_rate', 'high_confidence_rate', 'estimated_accuracy']:
            current_val = self.metrics['quality_metrics'].get(metric, 0)
            other_val = other_metrics.get('quality_metrics', {}).get(metric, 0)
            comparison[metric] = {
                'current': current_val,
                'other': other_val,
                'difference': current_val - other_val
            }

        return comparison