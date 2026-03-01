"""
CORS middleware configuration.

Permissive in development (all origins), strict in production (configured
allowlist from ``Settings.cors_origins``).
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.settings import get_settings


def configure_cors(app: FastAPI) -> None:
    """Add CORS middleware to the FastAPI application.

    In development mode, allows all origins for frictionless local
    frontend development. In production, restricts to the explicit
    origin allowlist from settings.
    """
    settings = get_settings()

    if settings.environment == "development":
        origins = ["*"]
    else:
        origins = settings.cors_origins

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["X-API-Key", "Content-Type", "Accept"],
        expose_headers=["X-Request-ID"],
    )
