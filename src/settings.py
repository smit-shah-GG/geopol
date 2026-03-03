"""
Centralized application settings via pydantic-settings.

All configuration is loaded from environment variables with sensible
development defaults. Production deployments override via .env file
or container environment.
"""

from __future__ import annotations

from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration with env-var binding."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # .env contains legacy vars (GEMINI_API_KEY, etc.)
    )

    # -- Database --
    database_url: str = (
        "postgresql+asyncpg://geopol:geopol_dev@localhost:5432/geopol"
    )
    gdelt_db_path: str = "data/events.db"

    # -- Redis --
    redis_url: str = "redis://localhost:6379/0"

    # -- Runtime --
    environment: Literal["development", "production", "testing"] = "development"
    use_fixtures: bool = False  # Enable mock fixture fallback (dev only, USE_FIXTURES=1)

    # -- API --
    api_key_header: str = "X-API-Key"
    cors_origins: list[str] = [
        "http://localhost:5173",
        "http://localhost:3000",
    ]

    # -- GDELT Ingest --
    gdelt_poll_interval: int = 900  # 15 minutes in seconds
    gdelt_backfill_on_start: bool = True

    # -- RSS Ingest --
    rss_poll_interval_tier1: int = 900  # 15 minutes
    rss_poll_interval_tier2: int = 3600  # 1 hour
    rss_article_retention_days: int = 90

    # -- Gemini LLM --
    gemini_api_key: str = ""
    gemini_model: str = "models/gemini-3-pro-preview"
    gemini_fallback_model: str = "models/gemini-2.5-pro"
    gemini_max_rpm: int = 25  # Requests per minute (match API key tier)

    # -- Daily Pipeline --
    gemini_daily_budget: int = 25  # Max questions per day

    # -- TKG Model --
    tkg_backend: Literal["tirgn", "regcn"] = "tirgn"

    # -- Logging --
    log_level: str = "INFO"
    log_json: bool = False

    # -- Calibration --
    calibration_min_samples: int = 10
    calibration_max_deviation: float = 0.20
    calibration_recompute_day: int = 0  # 0=Monday (weekday index)

    # -- Monitoring / Alerting --
    smtp_host: str = ""
    smtp_port: int = 587
    smtp_username: str = ""
    smtp_password: str = ""
    smtp_sender: str = ""
    alert_recipient: str = ""
    alert_cooldown_minutes: int = 60
    feed_staleness_hours: float = 1.0
    drift_threshold_pct: float = 10.0
    disk_warning_pct: float = 80.0
    disk_critical_pct: float = 90.0

    # -- Polymarket --
    polymarket_enabled: bool = True
    polymarket_poll_interval: int = 3600  # 1 hour in seconds
    polymarket_match_threshold: float = 0.6

    # -- ACLED --
    acled_email: str = ""
    acled_password: str = ""
    acled_poll_interval: int = 86400  # Daily in seconds
    acled_event_types: list[str] = [
        "Battles",
        "Explosions/Remote violence",
        "Violence against civilians",
    ]

    # -- Government Advisories --
    advisory_poll_interval: int = 86400  # Daily in seconds

    # -- Logging (file rotation) --
    log_dir: str = "data/logs"
    log_retention_days: int = 30

    @property
    def is_production(self) -> bool:
        return self.environment == "production"

    @property
    def is_testing(self) -> bool:
        return self.environment == "testing"


_settings: Settings | None = None


def get_settings() -> Settings:
    """Return a cached singleton Settings instance."""
    global _settings  # noqa: PLW0603
    if _settings is None:
        _settings = Settings()
    return _settings
