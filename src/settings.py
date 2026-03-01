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

    # -- Daily Pipeline --
    gemini_daily_budget: int = 25  # Max questions per day

    # -- TKG Model --
    tkg_backend: Literal["tirgn", "regcn"] = "regcn"

    # -- Logging --
    log_level: str = "INFO"
    log_json: bool = False

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
