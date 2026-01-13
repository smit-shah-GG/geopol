"""
Process raw GDELT events into TKG-ready format.

Converts raw CSV files to (entity1, relation, entity2, timestamp) quadruples
suitable for temporal knowledge graph training.
"""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Default directories
DEFAULT_RAW_DIR = Path(__file__).parent.parent.parent / "data" / "gdelt" / "raw"
DEFAULT_PROCESSED_DIR = Path(__file__).parent.parent.parent / "data" / "gdelt" / "processed"


class GDELTDataProcessor:
    """
    Processes raw GDELT event data into TKG-ready format.

    Transforms GDELT events into quadruples:
    (entity1, relation, entity2, timestamp)

    Where:
    - entity1, entity2: Actor names (normalized)
    - relation: CAMEO EventCode + QuadClass
    - timestamp: Event date as datetime
    """

    def __init__(
        self,
        raw_dir: Optional[Path] = None,
        processed_dir: Optional[Path] = None,
    ):
        """
        Initialize processor.

        Args:
            raw_dir: Directory containing raw CSV files
            processed_dir: Directory for processed output
        """
        self.raw_dir = raw_dir or DEFAULT_RAW_DIR
        self.processed_dir = processed_dir or DEFAULT_PROCESSED_DIR
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def load_raw_events(self) -> pd.DataFrame:
        """
        Load all raw CSV files from the raw directory.

        Returns:
            Combined DataFrame of all events
        """
        csv_files = sorted(self.raw_dir.glob("gdelt_*.csv"))

        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {self.raw_dir}")

        logger.info(f"Loading {len(csv_files)} CSV files from {self.raw_dir}")

        dfs = []
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file, low_memory=False)
                dfs.append(df)
                logger.debug(f"Loaded {len(df):,} events from {csv_file.name}")
            except Exception as e:
                logger.warning(f"Failed to load {csv_file.name}: {e}")

        if not dfs:
            raise ValueError("No CSV files could be loaded")

        combined = pd.concat(dfs, ignore_index=True)
        logger.info(f"Loaded {len(combined):,} total raw events")

        return combined

    def extract_entities(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract and normalize entity names from Actor columns.

        Args:
            df: Raw events DataFrame

        Returns:
            DataFrame with normalized entity columns
        """
        # Extract actor names
        df = df.copy()

        # Normalize: uppercase, strip whitespace
        df["entity1"] = df["Actor1Name"].fillna("").astype(str).str.upper().str.strip()
        df["entity2"] = df["Actor2Name"].fillna("").astype(str).str.upper().str.strip()

        # Filter out events with missing actors (empty strings)
        original_count = len(df)
        df = df[(df["entity1"] != "") & (df["entity2"] != "")]

        filtered_count = original_count - len(df)
        if filtered_count > 0:
            logger.info(
                f"Filtered {filtered_count:,} events with missing actors "
                f"({filtered_count / original_count * 100:.1f}%)"
            )

        return df

    def extract_relations(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract relation types from EventCode and QuadClass.

        Args:
            df: Events DataFrame

        Returns:
            DataFrame with relation column
        """
        df = df.copy()

        # Create composite relation: EventCode_QuadClass
        # This captures both the specific action (EventCode) and the broader category (QuadClass)
        df["relation"] = (
            df["EventCode"].astype(str) + "_Q" + df["QuadClass"].astype(str)
        )

        return df

    def build_temporal_events(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build temporal event quadruples from processed DataFrame.

        Args:
            df: Processed DataFrame with entity1, entity2, relation columns

        Returns:
            DataFrame with (entity1, relation, entity2, timestamp) format
        """
        df = df.copy()

        # Convert SQLDATE (YYYYMMDD) to datetime
        df["timestamp"] = pd.to_datetime(df["SQLDATE"], format="%Y%m%d", errors="coerce")

        # Filter invalid dates
        invalid_dates = df["timestamp"].isna().sum()
        if invalid_dates > 0:
            logger.warning(f"Dropping {invalid_dates:,} events with invalid dates")
            df = df.dropna(subset=["timestamp"])

        # Select TKG columns plus useful metadata
        tkg_df = df[
            [
                "entity1",
                "relation",
                "entity2",
                "timestamp",
                "EventCode",
                "QuadClass",
                "GoldsteinScale",
                "NumMentions",
                "AvgTone",
                "GLOBALEVENTID",
            ]
        ].copy()

        # Sort by timestamp
        tkg_df = tkg_df.sort_values("timestamp").reset_index(drop=True)

        logger.info(f"Built {len(tkg_df):,} temporal event quadruples")

        return tkg_df

    def save_processed(self, df: pd.DataFrame, filename: str = "events.parquet") -> Path:
        """
        Save processed events to Parquet format.

        Args:
            df: Processed events DataFrame
            filename: Output filename

        Returns:
            Path to saved file
        """
        output_path = self.processed_dir / filename
        df.to_parquet(output_path, index=False, engine="pyarrow")
        logger.info(f"Saved {len(df):,} events to {output_path}")
        return output_path

    def process_all(self) -> pd.DataFrame:
        """
        Full processing pipeline: load → extract → transform → save.

        Returns:
            Processed TKG-ready DataFrame
        """
        logger.info("Starting GDELT data processing pipeline")

        # Load raw data
        raw_df = self.load_raw_events()
        logger.info(f"Raw columns: {raw_df.columns.tolist()[:10]}...")

        # Extract entities
        entity_df = self.extract_entities(raw_df)

        # Extract relations
        relation_df = self.extract_relations(entity_df)

        # Build temporal events
        tkg_df = self.build_temporal_events(relation_df)

        # Save processed data
        self.save_processed(tkg_df)

        # Log statistics
        logger.info("=" * 50)
        logger.info("Processing Complete")
        logger.info("=" * 50)
        logger.info(f"Total events: {len(tkg_df):,}")
        logger.info(f"Unique entities: {tkg_df['entity1'].nunique() + tkg_df['entity2'].nunique():,}")
        logger.info(f"Unique relations: {tkg_df['relation'].nunique():,}")
        logger.info(f"Date range: {tkg_df['timestamp'].min()} to {tkg_df['timestamp'].max()}")

        return tkg_df


def main():
    """Run processing pipeline."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    processor = GDELTDataProcessor()
    df = processor.process_all()

    # Show sample
    print("\nSample TKG events:")
    print(df[["entity1", "relation", "entity2", "timestamp"]].head(10).to_string())


if __name__ == "__main__":
    main()
