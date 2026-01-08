"""
Complete GDELT data processing pipeline orchestration.
"""

import logging
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
import pandas as pd
import sys

from .gdelt_client import GDELTClient
from .fetch_events import fetch_conflict_diplomatic_events
from .filtering import combine_filters
from .deduplication import process_events_with_deduplication
from .sampling import sample_events_stratified
from .database import EventStorage, Event
from .monitoring import DataQualityMonitor
from .constants import DEFAULT_SAMPLE_SIZE, GDELT100_THRESHOLD

logger = logging.getLogger(__name__)


class GDELTPipeline:
    """Orchestrates the complete GDELT data processing pipeline."""

    def __init__(
        self,
        client: Optional[GDELTClient] = None,
        storage: Optional[EventStorage] = None,
        monitor: Optional[DataQualityMonitor] = None,
    ):
        """
        Initialize pipeline components.

        Args:
            client: GDELT API client (creates new if None)
            storage: Event storage (creates new if None)
            monitor: Data quality monitor (creates new if None)
        """
        self.client = client or GDELTClient()
        self.storage = storage or EventStorage()
        self.monitor = monitor or DataQualityMonitor()

        # Pipeline configuration
        self.config = {
            'use_filtering': True,
            'use_deduplication': True,
            'use_sampling': True,
            'max_events': DEFAULT_SAMPLE_SIZE,
            'min_mentions': GDELT100_THRESHOLD,
            'quadclasses': [1, 4],  # Verbal cooperation + Material conflict
            'save_checkpoints': True,
        }

        # Pipeline state
        self.current_data = None
        self.checkpoint_data = {}

    def configure(self, **kwargs):
        """
        Update pipeline configuration.

        Args:
            **kwargs: Configuration options to update
        """
        self.config.update(kwargs)
        logger.info(f"Pipeline configured with: {self.config}")

    def run(
        self,
        date_str: Optional[str] = None,
        timespan: str = "24h",
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """
        Run complete pipeline for specified time period.

        Args:
            date_str: Optional specific date (YYYY-MM-DD)
            timespan: Time window to process (e.g., "24h", "7d")
            dry_run: If True, don't store events in database

        Returns:
            Dictionary with pipeline results
        """
        logger.info("=" * 60)
        logger.info("Starting GDELT Processing Pipeline")
        logger.info("=" * 60)

        self.monitor.start_pipeline()
        results = {
            'success': False,
            'events_processed': 0,
            'events_stored': 0,
            'errors': []
        }

        try:
            # Stage 1: Fetch Events
            events_df = self._fetch_stage(date_str, timespan)
            if events_df is None or events_df.empty:
                raise ValueError("No events fetched from GDELT")

            # Stage 2: Filter Events
            if self.config['use_filtering']:
                events_df = self._filter_stage(events_df)
                if events_df.empty:
                    logger.warning("All events filtered out")
                    results['events_processed'] = self.monitor.metrics['event_counts']['raw']
                    return results

            # Stage 3: Deduplicate Events
            if self.config['use_deduplication']:
                events_df = self._deduplication_stage(events_df)
                if events_df.empty:
                    logger.info("All events were duplicates")
                    results['events_processed'] = self.monitor.metrics['event_counts']['raw']
                    return results

            # Stage 4: Sample Events
            if self.config['use_sampling'] and len(events_df) > self.config['max_events']:
                events_df = self._sampling_stage(events_df)

            # Stage 5: Store Events
            if not dry_run:
                stored_count = self._storage_stage(events_df)
                results['events_stored'] = stored_count
            else:
                logger.info("Dry run - skipping storage stage")
                results['events_stored'] = len(events_df)

            # Calculate final metrics
            self.monitor.calculate_reduction_rates()
            self.monitor.estimate_accuracy()
            self.monitor.end_pipeline()

            # Log and save metrics
            self.monitor.log_metrics()
            self.monitor.save_metrics()

            # Update results
            results['success'] = True
            results['events_processed'] = self.monitor.metrics['event_counts']['raw']
            results.update(self.monitor.get_summary())

        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            self.monitor.add_error(str(e), "pipeline")
            results['errors'].append(str(e))
            self.monitor.end_pipeline()
            self.monitor.save_metrics()

        return results

    def _fetch_stage(self, date_str: Optional[str], timespan: str) -> Optional[pd.DataFrame]:
        """
        Stage 1: Fetch events from GDELT.

        Args:
            date_str: Optional specific date
            timespan: Time window

        Returns:
            DataFrame with fetched events
        """
        logger.info("\nStage 1: Fetching Events")
        logger.info("-" * 40)

        self.monitor.start_stage("fetch")

        try:
            events_df = fetch_conflict_diplomatic_events(
                date_str=date_str,
                timespan=timespan,
                client=self.client
            )

            event_count = len(events_df) if events_df is not None else 0
            self.monitor.record_event_count('raw', event_count)
            self.monitor.end_stage('fetch', events_fetched=event_count)

            logger.info(f"✓ Fetched {event_count} events")

            if self.config['save_checkpoints']:
                self.checkpoint_data['after_fetch'] = events_df

            return events_df

        except Exception as e:
            logger.error(f"Fetch stage failed: {e}")
            self.monitor.add_error(str(e), "fetch")
            self.monitor.end_stage('fetch', error=str(e))
            raise

    def _filter_stage(self, events_df: pd.DataFrame) -> pd.DataFrame:
        """
        Stage 2: Apply filtering.

        Args:
            events_df: Events to filter

        Returns:
            Filtered DataFrame
        """
        logger.info("\nStage 2: Filtering Events")
        logger.info("-" * 40)

        self.monitor.start_stage("filter")

        try:
            filtered_df = combine_filters(
                events_df,
                use_quadclass=True,
                quadclasses=self.config['quadclasses'],
                use_confidence=True,
                min_mentions=self.config['min_mentions'],
                use_tone=True,
                use_sources=False,
                use_actors=False,
            )

            self.monitor.record_event_count('filtered', len(filtered_df))
            self.monitor.end_stage(
                'filter',
                events_filtered=len(events_df) - len(filtered_df),
                events_remaining=len(filtered_df)
            )

            logger.info(f"✓ Filtered: {len(events_df)} → {len(filtered_df)} events")

            if self.config['save_checkpoints']:
                self.checkpoint_data['after_filter'] = filtered_df

            return filtered_df

        except Exception as e:
            logger.error(f"Filter stage failed: {e}")
            self.monitor.add_error(str(e), "filter")
            self.monitor.end_stage('filter', error=str(e))
            raise

    def _deduplication_stage(self, events_df: pd.DataFrame) -> pd.DataFrame:
        """
        Stage 3: Apply deduplication.

        Args:
            events_df: Events to deduplicate

        Returns:
            Deduplicated DataFrame
        """
        logger.info("\nStage 3: Deduplicating Events")
        logger.info("-" * 40)

        self.monitor.start_stage("deduplicate")

        try:
            deduplicated_df, batch_dup, db_dup = process_events_with_deduplication(
                events_df,
                self.storage
            )

            self.monitor.record_event_count('deduplicated', len(deduplicated_df))
            self.monitor.end_stage(
                'deduplicate',
                batch_duplicates=batch_dup,
                database_duplicates=db_dup,
                events_remaining=len(deduplicated_df)
            )

            # Calculate duplicate rate for quality metrics
            if len(events_df) > 0:
                duplicate_rate = (batch_dup + db_dup) / len(events_df)
                self.monitor.metrics['quality_metrics']['duplicate_rate'] = duplicate_rate

            logger.info(
                f"✓ Deduplicated: {len(events_df)} → {len(deduplicated_df)} events "
                f"({batch_dup} batch, {db_dup} database duplicates)"
            )

            if self.config['save_checkpoints']:
                self.checkpoint_data['after_deduplicate'] = deduplicated_df

            return deduplicated_df

        except Exception as e:
            logger.error(f"Deduplication stage failed: {e}")
            self.monitor.add_error(str(e), "deduplicate")
            self.monitor.end_stage('deduplicate', error=str(e))
            raise

    def _sampling_stage(self, events_df: pd.DataFrame) -> pd.DataFrame:
        """
        Stage 4: Apply sampling if needed.

        Args:
            events_df: Events to sample

        Returns:
            Sampled DataFrame
        """
        logger.info("\nStage 4: Sampling Events")
        logger.info("-" * 40)

        self.monitor.start_stage("sample")

        try:
            sampled_df = sample_events_stratified(
                events_df,
                max_events=self.config['max_events'],
                stratify_by='QuadClass' if 'QuadClass' in events_df.columns else 'quad_class',
                prioritize_by='NumMentions' if 'NumMentions' in events_df.columns else 'num_mentions'
            )

            self.monitor.record_event_count('sampled', len(sampled_df))
            self.monitor.end_stage(
                'sample',
                events_sampled=len(events_df) - len(sampled_df),
                events_remaining=len(sampled_df)
            )

            logger.info(f"✓ Sampled: {len(events_df)} → {len(sampled_df)} events")

            if self.config['save_checkpoints']:
                self.checkpoint_data['after_sample'] = sampled_df

            return sampled_df

        except Exception as e:
            logger.error(f"Sampling stage failed: {e}")
            self.monitor.add_error(str(e), "sample")
            self.monitor.end_stage('sample', error=str(e))
            raise

    def _storage_stage(self, events_df: pd.DataFrame) -> int:
        """
        Stage 5: Store events in database.

        Args:
            events_df: Events to store

        Returns:
            Number of events stored
        """
        logger.info("\nStage 5: Storing Events")
        logger.info("-" * 40)

        self.monitor.start_stage("store")

        try:
            # Convert DataFrame to Event objects
            events_to_store = []
            for _, row in events_df.iterrows():
                event = Event.from_gdelt_row(row.to_dict())

                # Add content hash and time window if available
                if 'content_hash' in row:
                    event.content_hash = row['content_hash']
                if 'time_window' in row:
                    event.time_window = row['time_window']

                events_to_store.append(event)

            # Store events
            stored_count = self.storage.insert_events(events_to_store)

            self.monitor.record_event_count('stored', stored_count)
            self.monitor.end_stage('store', events_stored=stored_count)

            # Record ingestion statistics in database
            if hasattr(self.storage, 'record_ingestion_stats'):
                self.storage.record_ingestion_stats(
                    events_fetched=self.monitor.metrics['event_counts']['raw'],
                    events_deduplicated=self.monitor.metrics['event_counts'].get('deduplicated', 0),
                    events_inserted=stored_count,
                    processing_time=sum(self.monitor.metrics['processing_times'].values())
                )

            logger.info(f"✓ Stored {stored_count} events in database")

            # Calculate final data quality metrics
            self.monitor.calculate_data_quality(events_df)

            return stored_count

        except Exception as e:
            logger.error(f"Storage stage failed: {e}")
            self.monitor.add_error(str(e), "store")
            self.monitor.end_stage('store', error=str(e))
            raise

    def run_with_recovery(
        self,
        date_str: Optional[str] = None,
        timespan: str = "24h",
        max_retries: int = 3,
    ) -> Dict[str, Any]:
        """
        Run pipeline with error recovery and retries.

        Args:
            date_str: Optional specific date
            timespan: Time window
            max_retries: Maximum retry attempts

        Returns:
            Pipeline results
        """
        for attempt in range(max_retries):
            try:
                logger.info(f"Pipeline attempt {attempt + 1}/{max_retries}")

                results = self.run(date_str, timespan)

                if results['success']:
                    return results

            except Exception as e:
                logger.error(f"Attempt {attempt + 1} failed: {e}")

                if attempt == max_retries - 1:
                    logger.error("All retry attempts exhausted")
                    return {
                        'success': False,
                        'events_processed': 0,
                        'events_stored': 0,
                        'errors': [f"All {max_retries} attempts failed"]
                    }

                # Wait before retry
                wait_time = 2 ** attempt  # Exponential backoff
                logger.info(f"Waiting {wait_time} seconds before retry...")
                import time
                time.sleep(wait_time)

        return {
            'success': False,
            'events_processed': 0,
            'events_stored': 0,
            'errors': ['Unexpected error in retry logic']
        }

    def get_checkpoint_data(self, checkpoint_name: str) -> Optional[pd.DataFrame]:
        """
        Get data from a specific checkpoint.

        Args:
            checkpoint_name: Name of checkpoint

        Returns:
            DataFrame from checkpoint or None
        """
        return self.checkpoint_data.get(checkpoint_name)