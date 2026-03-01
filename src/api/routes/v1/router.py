"""
V1 API router — aggregates all v1 sub-routers.

Included in the app at ``/api/v1`` prefix by ``create_app()``.
"""

from __future__ import annotations

from fastapi import APIRouter

from src.api.routes.v1.countries import router as countries_router
from src.api.routes.v1.forecasts import router as forecasts_router
from src.api.routes.v1.health import router as health_router

v1_router = APIRouter()

# Health is public — no auth prefix tag
v1_router.include_router(health_router, tags=["health"])

# Forecast endpoints
v1_router.include_router(forecasts_router, prefix="/forecasts", tags=["forecasts"])

# Country risk endpoints
v1_router.include_router(countries_router, prefix="/countries", tags=["countries"])
