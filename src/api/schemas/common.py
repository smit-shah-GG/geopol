"""
Shared API DTOs: error responses, pagination, and utility functions.

ProblemDetail follows RFC 9457 (supersedes RFC 7807) for machine-parseable
error responses. PaginatedResponse provides cursor-based pagination for
list endpoints.
"""

import base64
import json
from typing import Generic, Optional, TypeVar

from pydantic import BaseModel, ConfigDict, Field

T = TypeVar("T")


class ProblemDetail(BaseModel):
    """RFC 9457 Problem Details error response.

    All API error responses use this format with Content-Type
    application/problem+json. Machine-parseable, well-documented,
    and compatible with standard HTTP error handling libraries.
    """

    model_config = ConfigDict(from_attributes=True)

    type: str = Field(
        "about:blank",
        description="URI reference identifying the problem type",
    )
    title: str = Field(..., description="Short human-readable summary")
    status: int = Field(..., description="HTTP status code")
    detail: Optional[str] = Field(
        None, description="Human-readable explanation specific to this occurrence"
    )
    instance: Optional[str] = Field(
        None,
        description="URI reference identifying the specific occurrence of the problem",
    )


class PaginatedResponse(BaseModel, Generic[T]):
    """Cursor-based paginated response wrapper.

    List endpoints return items with an opaque next_cursor token.
    The client passes the cursor back as a query parameter to get the
    next page. Cursors encode the last item's (forecast_id, created_at)
    for keyset pagination — no offset/limit, no skip-scan.
    """

    model_config = ConfigDict(from_attributes=True)

    items: list[T] = Field(..., description="Page of result items")
    next_cursor: Optional[str] = Field(
        None,
        description="Opaque cursor token for the next page (null if no more pages)",
    )
    has_more: bool = Field(
        False, description="Whether more pages exist after this one"
    )


def encode_cursor(forecast_id: str, created_at: str) -> str:
    """Encode pagination cursor as base64url JSON.

    The cursor is opaque to the client — they pass it back verbatim.
    Internally it contains the keyset pagination coordinates.

    Args:
        forecast_id: ID of the last item on the current page.
        created_at: ISO timestamp of the last item on the current page.

    Returns:
        Base64url-encoded JSON string.
    """
    payload = json.dumps({"id": forecast_id, "ts": created_at}, separators=(",", ":"))
    return base64.urlsafe_b64encode(payload.encode("utf-8")).decode("ascii")


def decode_cursor(cursor: str) -> dict:
    """Decode a pagination cursor back to its coordinates.

    Args:
        cursor: Base64url-encoded cursor string from the client.

    Returns:
        Dict with "id" and "ts" keys.

    Raises:
        ValueError: If the cursor is malformed or tampered with.
    """
    try:
        raw = base64.urlsafe_b64decode(cursor.encode("ascii"))
        data = json.loads(raw.decode("utf-8"))
    except Exception as exc:
        raise ValueError(f"Invalid cursor: {exc}") from exc

    if "id" not in data or "ts" not in data:
        raise ValueError("Cursor missing required fields 'id' and 'ts'")

    return data
