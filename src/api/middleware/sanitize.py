"""
Input sanitization for forecast question submissions.

Three layers of defense:
1. **Blocklist**: Known prompt injection phrases are rejected outright.
2. **Structural validation**: Input must contain at least one geopolitical
   keyword to ensure the question is on-topic.
3. **Length enforcement**: 10-500 chars (defense-in-depth over Pydantic).

Also provides ``sanitize_error_response()`` to strip system internals
(file paths, API keys, model names) from error messages before they
reach the client.

This module exports utility functions, NOT ASGI middleware. Wire them
as FastAPI ``Depends()`` callables or call directly from route handlers.
"""

from __future__ import annotations

import logging
import re

from fastapi import HTTPException

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------
# Prompt injection blocklist
# -----------------------------------------------------------------------

INJECTION_PATTERNS: list[str] = [
    "ignore previous",
    "ignore all",
    "system prompt",
    "api key",
    "internal",
    "secret",
    "you are now",
    "forget everything",
    "disregard",
    "override",
    "reveal your",
    "show me your",
    "what are your instructions",
    "repeat after me",
]

# -----------------------------------------------------------------------
# Geopolitical keyword set -- at least one must appear in the question.
# Kept deliberately broad to avoid false-positive rejections on valid
# geopolitical questions.
# -----------------------------------------------------------------------

_GEOPOLITICAL_KEYWORDS: set[str] = {
    # Actors
    "country",
    "government",
    "president",
    "minister",
    "military",
    "army",
    "navy",
    "parliament",
    "congress",
    "senate",
    "coalition",
    "opposition",
    "rebel",
    "insurgent",
    "militia",
    "nato",
    "united nations",
    "un",
    "eu",
    "african union",
    "asean",
    # Actions / events
    "will",
    "conflict",
    "war",
    "peace",
    "ceasefire",
    "election",
    "vote",
    "referendum",
    "sanctions",
    "embargo",
    "agreement",
    "treaty",
    "diplomatic",
    "diplomacy",
    "trade",
    "tariff",
    "crisis",
    "coup",
    "protest",
    "invasion",
    "occupation",
    "withdrawal",
    "deploy",
    "nuclear",
    "missile",
    "attack",
    "border",
    "territory",
    "annex",
    "negotiate",
    "summit",
    "alliance",
    "aid",
    "humanitarian",
    "refugee",
    "migration",
    "stability",
    "instability",
    "escalation",
    "deescalation",
}

# Minimum and maximum question length (chars).
_MIN_LENGTH: int = 10
_MAX_LENGTH: int = 500

# -----------------------------------------------------------------------
# Regex for stripping system internals from error messages.
# -----------------------------------------------------------------------

# Matches Unix/Windows file paths like /home/user/app/src/foo.py or C:\Users\...
_PATH_RE = re.compile(r"(?:[A-Z]:\\|/)[\w/\\.\\-]+", re.IGNORECASE)

# Matches things that look like API keys (long alphanumeric + dashes)
_API_KEY_RE = re.compile(r"[A-Za-z0-9_\-]{20,}")

# Matches known model identifiers (gemini-*, gpt-*, claude-*, etc.)
_MODEL_RE = re.compile(
    r"\b(?:gemini|gpt|claude|llama|mixtral|command)[\w\-./]*",
    re.IGNORECASE,
)


def validate_forecast_question(question: str) -> str:
    """Validate and sanitize a forecast question submission.

    Applies three layers of defense in order:
    1. Length check (10-500 chars).
    2. Prompt injection blocklist.
    3. Geopolitical keyword presence.

    Args:
        question: Raw user-submitted question.

    Returns:
        Stripped, length-capped question on success.

    Raises:
        HTTPException: 400 if any validation fails. Error messages are
            intentionally generic to avoid leaking blocklist contents.
    """
    cleaned = question.strip()

    # Layer 1: length enforcement (defense-in-depth over Pydantic)
    if len(cleaned) < _MIN_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Question too short (minimum {_MIN_LENGTH} characters)."
            ),
        )
    if len(cleaned) > _MAX_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Question too long (maximum {_MAX_LENGTH} characters)."
            ),
        )

    # Layer 2: prompt injection blocklist
    lower = cleaned.lower()
    for pattern in INJECTION_PATTERNS:
        if pattern in lower:
            logger.warning(
                "Blocked injection attempt: %.80s...",
                cleaned,
            )
            raise HTTPException(
                status_code=400,
                detail="Invalid question format.",
            )

    # Layer 3: geopolitical relevance
    if not any(kw in lower for kw in _GEOPOLITICAL_KEYWORDS):
        raise HTTPException(
            status_code=400,
            detail="Question must be about a specific geopolitical topic.",
        )

    return cleaned


def sanitize_error_response(error: Exception) -> dict[str, str]:
    """Strip system internals from an error message before returning to client.

    Removes file paths, API key-like strings, and model identifiers.
    For truly unexpected errors, returns a generic message.

    Args:
        error: The caught exception.

    Returns:
        A dict with a ``"detail"`` key safe for client consumption.
    """
    message = str(error)

    # If the message contains any internal-looking content, nuke it entirely
    if _PATH_RE.search(message) or _API_KEY_RE.search(message):
        return {"detail": "An internal error occurred."}

    # Strip model names even if no paths found
    sanitized = _MODEL_RE.sub("[model]", message)

    # If sanitization changed the message substantially, use generic
    if sanitized != message:
        return {"detail": "An internal error occurred."}

    # For simple, safe error messages (e.g. "Question not found")
    # Return as-is, but cap length to prevent data exfiltration
    if len(sanitized) > 200:
        return {"detail": "An internal error occurred."}

    return {"detail": sanitized}
