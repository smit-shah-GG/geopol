"""
RFC 9457 Problem Details error handlers.

Ensures ALL error responses from the API use the ``application/problem+json``
content type with a structured body matching ``ProblemDetail``. Handles:
- FastAPI validation errors (422)
- Explicit HTTPException raises (4xx/5xx)
- Unhandled exceptions (500)
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from src.api.schemas.common import ProblemDetail

logger = logging.getLogger(__name__)

_PROBLEM_JSON = "application/problem+json"


def _problem_response(
    status: int,
    title: str,
    detail: str | None = None,
    instance: str | None = None,
    extra: dict[str, Any] | None = None,
) -> JSONResponse:
    """Build a RFC 9457 Problem Details JSON response."""
    body = ProblemDetail(
        type="about:blank",
        title=title,
        status=status,
        detail=detail,
        instance=instance,
    )
    content = body.model_dump(exclude_none=True)
    if extra:
        content.update(extra)
    return JSONResponse(status_code=status, content=content, media_type=_PROBLEM_JSON)


async def _validation_error_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Handle Pydantic/FastAPI request validation errors (422)."""
    errors = exc.errors()
    detail_parts: list[str] = []
    for err in errors:
        loc = " -> ".join(str(x) for x in err.get("loc", []))
        msg = err.get("msg", "validation error")
        detail_parts.append(f"{loc}: {msg}")

    return _problem_response(
        status=422,
        title="Validation Error",
        detail="; ".join(detail_parts),
        instance=str(request.url),
        extra={"errors": errors},
    )


async def _http_exception_handler(
    request: Request, exc: StarletteHTTPException
) -> JSONResponse:
    """Handle explicit HTTPException raises (any status code)."""
    return _problem_response(
        status=exc.status_code,
        title=exc.detail if isinstance(exc.detail, str) else str(exc.detail),
        instance=str(request.url),
    )


async def _generic_exception_handler(
    request: Request, exc: Exception
) -> JSONResponse:
    """Handle unhandled exceptions as 500 Internal Server Error.

    Logs the full traceback but returns a generic message to the client.
    Never leak stack traces to external consumers.
    """
    logger.exception("Unhandled exception on %s %s", request.method, request.url)
    return _problem_response(
        status=500,
        title="Internal Server Error",
        detail="An unexpected error occurred. Check server logs for details.",
        instance=str(request.url),
    )


def register_error_handlers(app: FastAPI) -> None:
    """Register all exception handlers on the FastAPI application.

    Called once during app factory setup. Order doesn't matter — FastAPI
    dispatches by exception type, not registration order.
    """
    app.add_exception_handler(RequestValidationError, _validation_error_handler)  # type: ignore[arg-type]
    app.add_exception_handler(StarletteHTTPException, _http_exception_handler)  # type: ignore[arg-type]
    app.add_exception_handler(Exception, _generic_exception_handler)  # type: ignore[arg-type]
