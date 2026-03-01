"""
Dual-database persistence layer.

PostgreSQL (async): Forecast predictions, outcomes, calibration, API keys.
SQLite (sync): GDELT event store and partition index (retained from v1.x).
"""
