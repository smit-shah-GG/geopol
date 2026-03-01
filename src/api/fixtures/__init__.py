"""
Mock forecast fixtures and factory functions.

Provides hand-crafted JSON fixtures for key conflict scenarios
(Syria, Ukraine, Myanmar) and a factory for generating arbitrary
mock forecasts. All output is validated against the Pydantic DTOs.

Usage:
    from src.api.fixtures import load_fixture, create_mock_forecast

    syria = load_fixture("SY")
    random_forecast = create_mock_forecast(country_iso="IR", horizon_days=60)
"""

from src.api.fixtures.factory import (
    create_mock_country_risk,
    create_mock_forecast,
    get_empty_country_response,
    load_all_fixtures,
    load_fixture,
)

__all__ = [
    "load_fixture",
    "load_all_fixtures",
    "create_mock_forecast",
    "create_mock_country_risk",
    "get_empty_country_response",
]
