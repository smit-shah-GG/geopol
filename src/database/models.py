"""
Data models for GDELT event storage.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any
from datetime import datetime
import json


@dataclass
class Event:
    """Model for GDELT event data matching database schema."""

    # Required fields
    content_hash: str
    time_window: str
    event_date: str

    # GDELT identifiers
    gdelt_id: Optional[str] = None

    # Actor fields
    actor1_code: Optional[str] = None
    actor2_code: Optional[str] = None

    # Event classification
    event_code: Optional[str] = None
    quad_class: Optional[int] = None  # 1-4 for QuadClass categories

    # Event metrics
    goldstein_scale: Optional[float] = None
    num_mentions: Optional[int] = None
    num_sources: Optional[int] = None
    tone: Optional[float] = None

    # Source information
    url: Optional[str] = None
    title: Optional[str] = None
    domain: Optional[str] = None

    # Metadata
    raw_json: Optional[str] = None
    created_at: Optional[str] = field(default_factory=lambda: datetime.utcnow().isoformat())

    # Database ID (set after insertion)
    id: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for database operations."""
        return {k: v for k, v in asdict(self).items() if v is not None}

    @classmethod
    def from_gdelt_row(cls, row: Dict[str, Any]) -> 'Event':
        """
        Create Event instance from GDELT Doc API row.

        Args:
            row: Dictionary containing GDELT article data

        Returns:
            Event instance
        """
        # Extract date from seendate or use current date
        event_date = row.get('seendate', datetime.now().isoformat())
        if isinstance(event_date, str) and len(event_date) > 10:
            event_date = event_date[:10]  # Extract YYYY-MM-DD

        # Create event with available fields
        event = cls(
            gdelt_id=row.get('gdelt_id') or row.get('url'),  # Use URL as fallback ID
            event_date=event_date,
            url=row.get('url'),
            title=row.get('title'),
            domain=row.get('domain'),
            tone=row.get('tone'),
            content_hash='',  # Will be set by deduplication
            time_window='',   # Will be set by deduplication
            raw_json=json.dumps(row) if row else None
        )

        # Extract additional GDELT-specific fields if available
        if 'Actor1Code' in row:
            event.actor1_code = row['Actor1Code']
        if 'Actor2Code' in row:
            event.actor2_code = row['Actor2Code']
        if 'EventCode' in row:
            event.event_code = row['EventCode']
        if 'QuadClass' in row:
            event.quad_class = int(row['QuadClass'])
        if 'GoldsteinScale' in row:
            event.goldstein_scale = float(row['GoldsteinScale'])
        if 'NumMentions' in row:
            event.num_mentions = int(row['NumMentions'])
        if 'NumSources' in row:
            event.num_sources = int(row['NumSources'])

        return event

    def is_high_confidence(self, min_mentions: int = 100) -> bool:
        """
        Check if event meets high confidence criteria (GDELT100).

        Args:
            min_mentions: Minimum number of mentions required

        Returns:
            True if event has sufficient mentions
        """
        return self.num_mentions is not None and self.num_mentions >= min_mentions

    def is_conflict(self) -> bool:
        """Check if event is classified as material conflict (QuadClass 4)."""
        return self.quad_class == 4

    def is_diplomatic(self) -> bool:
        """Check if event is classified as verbal cooperation (QuadClass 1)."""
        return self.quad_class == 1

    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"Event(id={self.id}, date={self.event_date}, "
            f"actors=[{self.actor1_code}, {self.actor2_code}], "
            f"quad_class={self.quad_class}, mentions={self.num_mentions})"
        )