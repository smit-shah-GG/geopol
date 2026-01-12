"""
SQLite-backed prediction tracking system.

This module provides persistent storage for geopolitical forecasts, enabling:
- Historical prediction tracking with metadata
- Outcome resolution for calibration training
- Per-category prediction retrieval
- Performance analysis over time

The schema is optimized for calibration workloads with indices on:
- timestamp (for time-series analysis)
- category (for per-category calibration)
- resolution_date (for identifying resolved predictions)
"""

import json
import logging
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import declarative_base, sessionmaker

logger = logging.getLogger(__name__)

Base = declarative_base()


class Prediction(Base):
    """
    ORM model for geopolitical predictions.

    Attributes:
        id: Primary key
        query: Natural language forecasting question
        timestamp: When prediction was made
        raw_probability: Model's raw probability output (0-1)
        calibrated_probability: Calibrated probability (nullable until calibration applied)
        category: Event category (conflict/diplomatic/economic)
        entities: JSON array of entity names involved
        outcome: True outcome (0=no, 1=yes, nullable until resolved)
        resolution_date: When outcome was determined (nullable until resolved)
        prediction_metadata: JSON object for scenarios, reasoning, model details, etc.
    """

    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    query = Column(Text, nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, default=datetime.now, index=True)
    raw_probability = Column(Float, nullable=False)
    calibrated_probability = Column(Float, nullable=True)
    category = Column(String(32), nullable=False, index=True)
    entities = Column(JSON, nullable=False, default=list)
    outcome = Column(Float, nullable=True)  # 0.0 or 1.0 when resolved
    resolution_date = Column(DateTime, nullable=True, index=True)
    prediction_metadata = Column(JSON, nullable=False, default=dict)

    # Indices for common queries
    __table_args__ = (
        Index("ix_category_timestamp", "category", "timestamp"),
        Index("ix_resolved", "resolution_date"),
    )

    def __repr__(self) -> str:
        return (
            f"<Prediction(id={self.id}, query='{self.query[:50]}...', "
            f"category={self.category}, raw_p={self.raw_probability:.3f})>"
        )


class PredictionStore:
    """
    SQLite-backed prediction storage system.

    Provides CRUD operations for predictions with optimized queries for:
    - Calibration training (resolved predictions by category)
    - Performance monitoring (recent predictions)
    - Outcome resolution (updating predictions with ground truth)

    Thread-safe via session management.
    """

    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize prediction store.

        Args:
            db_path: Path to SQLite database file. Defaults to ./data/predictions.db
        """
        if db_path is None:
            db_path = "./data/predictions.db"

        # Ensure parent directory exists
        db_file = Path(db_path)
        db_file.parent.mkdir(parents=True, exist_ok=True)

        # Create engine with SQLite-specific optimizations
        self.engine = create_engine(
            f"sqlite:///{db_path}",
            echo=False,  # Set to True for SQL debugging
            connect_args={"check_same_thread": False},  # Allow multi-threading
        )

        # Create tables if they don't exist
        Base.metadata.create_all(self.engine)

        # Session factory
        self.SessionLocal = sessionmaker(bind=self.engine)

        logger.info(f"Initialized PredictionStore at {db_path}")

    @contextmanager
    def get_session(self):
        """
        Context manager for database sessions.

        Ensures proper session cleanup and transaction management.

        Yields:
            SQLAlchemy session
        """
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            session.close()

    def store_prediction(
        self,
        query: str,
        raw_probability: float,
        category: str,
        entities: List[str],
        metadata: Optional[Dict] = None,
        calibrated_probability: Optional[float] = None,
    ) -> int:
        """
        Store a new prediction in the database.

        Args:
            query: Natural language forecasting question
            raw_probability: Model's raw probability output (0-1)
            category: Event category (conflict/diplomatic/economic)
            entities: List of entity names involved
            metadata: Optional dict for scenarios, reasoning, model details (stored as prediction_metadata)
            calibrated_probability: Optional calibrated probability if already computed

        Returns:
            ID of inserted prediction

        Raises:
            ValueError: If probability out of range or category invalid
        """
        # Validate inputs
        if not 0.0 <= raw_probability <= 1.0:
            raise ValueError(f"Raw probability must be in [0, 1], got {raw_probability}")

        if calibrated_probability is not None and not 0.0 <= calibrated_probability <= 1.0:
            raise ValueError(
                f"Calibrated probability must be in [0, 1], got {calibrated_probability}"
            )

        valid_categories = {"conflict", "diplomatic", "economic"}
        if category.lower() not in valid_categories:
            raise ValueError(f"Category must be one of {valid_categories}, got {category}")

        # Create prediction
        prediction = Prediction(
            query=query,
            timestamp=datetime.now(),
            raw_probability=raw_probability,
            calibrated_probability=calibrated_probability,
            category=category.lower(),
            entities=entities,
            prediction_metadata=metadata or {},
        )

        with self.get_session() as session:
            session.add(prediction)
            session.flush()  # Get ID before commit
            pred_id = prediction.id
            logger.info(f"Stored prediction {pred_id}: {query[:50]}...")

        return pred_id

    def update_outcome(self, prediction_id: int, outcome: float) -> None:
        """
        Update a prediction with its ground truth outcome.

        Args:
            prediction_id: ID of prediction to update
            outcome: True outcome (0.0=no, 1.0=yes)

        Raises:
            ValueError: If outcome not 0.0 or 1.0
            KeyError: If prediction_id not found
        """
        if outcome not in [0.0, 1.0]:
            raise ValueError(f"Outcome must be 0.0 or 1.0, got {outcome}")

        with self.get_session() as session:
            prediction = session.query(Prediction).filter_by(id=prediction_id).first()

            if prediction is None:
                raise KeyError(f"Prediction {prediction_id} not found")

            prediction.outcome = outcome
            prediction.resolution_date = datetime.now()

            logger.info(
                f"Updated prediction {prediction_id} with outcome {outcome} "
                f"(query: {prediction.query[:50]}...)"
            )

    def get_predictions_for_calibration(
        self,
        category: Optional[str] = None,
        resolved_only: bool = False,
        min_date: Optional[datetime] = None,
    ) -> List[Dict]:
        """
        Retrieve predictions for calibration training.

        Args:
            category: Filter by category (conflict/diplomatic/economic). None = all.
            resolved_only: If True, only return predictions with known outcomes
            min_date: Minimum timestamp (for time-based filtering)

        Returns:
            List of dicts with keys: id, query, raw_probability, calibrated_probability,
            category, entities, outcome, timestamp, resolution_date, prediction_metadata
        """
        with self.get_session() as session:
            query = session.query(Prediction)

            # Apply filters
            if category:
                query = query.filter(Prediction.category == category.lower())

            if resolved_only:
                query = query.filter(Prediction.outcome.isnot(None))

            if min_date:
                query = query.filter(Prediction.timestamp >= min_date)

            # Order by timestamp descending (most recent first)
            query = query.order_by(Prediction.timestamp.desc())

            predictions = query.all()

            # Convert to dicts
            results = []
            for pred in predictions:
                results.append(
                    {
                        "id": pred.id,
                        "query": pred.query,
                        "raw_probability": pred.raw_probability,
                        "calibrated_probability": pred.calibrated_probability,
                        "category": pred.category,
                        "entities": pred.entities,
                        "outcome": pred.outcome,
                        "timestamp": pred.timestamp,
                        "resolution_date": pred.resolution_date,
                        "prediction_metadata": pred.prediction_metadata,
                    }
                )

            logger.info(
                f"Retrieved {len(results)} predictions "
                f"(category={category}, resolved_only={resolved_only})"
            )

            return results

    def get_recent_predictions(self, days: int = 30) -> List[Dict]:
        """
        Retrieve predictions from the last N days.

        Args:
            days: Number of days to look back

        Returns:
            List of prediction dicts
        """
        min_date = datetime.now() - timedelta(days=days)
        return self.get_predictions_for_calibration(min_date=min_date)

    def get_prediction_by_id(self, prediction_id: int) -> Optional[Dict]:
        """
        Retrieve a single prediction by ID.

        Args:
            prediction_id: ID of prediction to retrieve

        Returns:
            Prediction dict or None if not found
        """
        with self.get_session() as session:
            pred = session.query(Prediction).filter_by(id=prediction_id).first()

            if pred is None:
                return None

            return {
                "id": pred.id,
                "query": pred.query,
                "raw_probability": pred.raw_probability,
                "calibrated_probability": pred.calibrated_probability,
                "category": pred.category,
                "entities": pred.entities,
                "outcome": pred.outcome,
                "timestamp": pred.timestamp,
                "resolution_date": pred.resolution_date,
                "prediction_metadata": pred.prediction_metadata,
            }

    def update_calibrated_probability(self, prediction_id: int, calibrated_prob: float) -> None:
        """
        Update a prediction with its calibrated probability.

        Args:
            prediction_id: ID of prediction to update
            calibrated_prob: Calibrated probability (0-1)

        Raises:
            ValueError: If calibrated_prob out of range
            KeyError: If prediction_id not found
        """
        if not 0.0 <= calibrated_prob <= 1.0:
            raise ValueError(f"Calibrated probability must be in [0, 1], got {calibrated_prob}")

        with self.get_session() as session:
            prediction = session.query(Prediction).filter_by(id=prediction_id).first()

            if prediction is None:
                raise KeyError(f"Prediction {prediction_id} not found")

            prediction.calibrated_probability = calibrated_prob

            logger.debug(
                f"Updated prediction {prediction_id} with calibrated probability {calibrated_prob:.3f}"
            )

    def get_statistics(self) -> Dict:
        """
        Get database statistics.

        Returns:
            Dict with total predictions, resolved count, per-category counts
        """
        with self.get_session() as session:
            total = session.query(Prediction).count()
            resolved = session.query(Prediction).filter(Prediction.outcome.isnot(None)).count()

            # Per-category counts
            categories = {}
            for category in ["conflict", "diplomatic", "economic"]:
                count = session.query(Prediction).filter(Prediction.category == category).count()
                categories[category] = count

            return {
                "total_predictions": total,
                "resolved_predictions": resolved,
                "unresolved_predictions": total - resolved,
                "by_category": categories,
            }
