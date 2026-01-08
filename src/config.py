"""
Configuration management for the geopolitical forecasting engine.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
METRICS_DIR = DATA_DIR / "metrics"
EVENTS_DB_PATH = DATA_DIR / "events.db"

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
METRICS_DIR.mkdir(exist_ok=True)

# GDELT API settings
GDELT_REQUEST_TIMEOUT = int(os.getenv("GDELT_REQUEST_TIMEOUT", "30"))  # seconds
GDELT_MAX_RETRIES = int(os.getenv("GDELT_MAX_RETRIES", "5"))
GDELT_BASE_DELAY = float(os.getenv("GDELT_BASE_DELAY", "1.0"))  # seconds

# Filtering settings
GDELT100_THRESHOLD = int(os.getenv("GDELT100_THRESHOLD", "100"))  # minimum mentions for high confidence
MIN_GOLDSTEIN_CONFLICT = float(os.getenv("MIN_GOLDSTEIN_CONFLICT", "-5.0"))
TONE_RANGE_DIPLOMATIC = (-2, 2)  # Tone range for diplomatic events

# QuadClass definitions (CAMEO event categories)
QUADCLASS_VERBAL_COOPERATION = 1
QUADCLASS_MATERIAL_COOPERATION = 2
QUADCLASS_VERBAL_CONFLICT = 3
QUADCLASS_MATERIAL_CONFLICT = 4

# Sampling settings
MAX_EVENTS_PER_BATCH = int(os.getenv("MAX_EVENTS_PER_BATCH", "10000"))
STRATIFIED_SAMPLE_SIZE = int(os.getenv("STRATIFIED_SAMPLE_SIZE", "5000"))

# Database settings
DB_BATCH_SIZE = int(os.getenv("DB_BATCH_SIZE", "1000"))

# Logging settings
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Date format for GDELT queries (gdeltdoc expects "YYYY-MM-DD HH:MM")
GDELT_DATE_FORMAT = "%Y-%m-%d %H:%M"