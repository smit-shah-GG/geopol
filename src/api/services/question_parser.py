"""
LLM-powered natural language question parser for the submission flow.

Accepts a free-form geopolitical forecast question and uses Gemini to extract:
    - country_iso_list: ISO 3166-1 alpha-2 codes mentioned or implied
    - horizon_days: temporal horizon inferred from language (7-365)
    - category: CAMEO-aligned forecast category

Uses the existing GeminiClient from src/forecasting/gemini_client.py for
API interaction with rate limiting and retry logic already built in.

On parse failure, returns safe defaults rather than crashing -- a failed
parse must never prevent a user from submitting a question.
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

# Known valid ISO 3166-1 alpha-2 codes for validation
_VALID_ISO_CODES: set[str] = {
    "AF", "AL", "DZ", "AD", "AO", "AG", "AR", "AM", "AU", "AT", "AZ", "BS", "BH", "BD", "BB",
    "BY", "BE", "BZ", "BJ", "BT", "BO", "BA", "BW", "BR", "BN", "BG", "BF", "BI", "KH", "CM",
    "CA", "CV", "CF", "TD", "CL", "CN", "CO", "KM", "CG", "CD", "CR", "CI", "HR", "CU", "CY",
    "CZ", "DK", "DJ", "DM", "DO", "EC", "EG", "SV", "GQ", "ER", "EE", "SZ", "ET", "FJ", "FI",
    "FR", "GA", "GM", "GE", "DE", "GH", "GR", "GD", "GT", "GN", "GW", "GY", "HT", "HN", "HU",
    "IS", "IN", "ID", "IR", "IQ", "IE", "IL", "IT", "JM", "JP", "JO", "KZ", "KE", "KI", "KP",
    "KR", "KW", "KG", "LA", "LV", "LB", "LS", "LR", "LY", "LI", "LT", "LU", "MG", "MW", "MY",
    "MV", "ML", "MT", "MH", "MR", "MU", "MX", "FM", "MD", "MC", "MN", "ME", "MA", "MZ", "MM",
    "NA", "NR", "NP", "NL", "NZ", "NI", "NE", "NG", "NO", "OM", "PK", "PW", "PS", "PA", "PG",
    "PY", "PE", "PH", "PL", "PT", "QA", "RO", "RU", "RW", "KN", "LC", "VC", "WS", "SM", "ST",
    "SA", "SN", "RS", "SC", "SL", "SG", "SK", "SI", "SB", "SO", "ZA", "SS", "ES", "LK", "SD",
    "SR", "SE", "CH", "SY", "TW", "TJ", "TZ", "TH", "TL", "TG", "TO", "TT", "TN", "TR", "TM",
    "TV", "UG", "UA", "AE", "GB", "US", "UY", "UZ", "VU", "VE", "VN", "YE", "ZM", "ZW",
}

_VALID_CATEGORIES: set[str] = {
    "conflict", "diplomatic", "economic", "security", "political", "GENERAL",
}

_PARSE_PROMPT_TEMPLATE = """Parse this geopolitical forecast question into structured form.

QUESTION: {question}

Return ONLY a JSON object with:
- "country_iso_list": array of ISO 3166-1 alpha-2 codes (uppercase)
- "horizon_days": integer (7-365), inferred from temporal language
- "category": one of "conflict", "diplomatic", "economic", "security", "political", "GENERAL"

If the question mentions a region (e.g., "Middle East"), expand to relevant country codes.
If no time horizon is specified, default to 30 days.
If category is ambiguous, use "GENERAL".

Return ONLY the JSON object. No markdown, no explanation."""

# Default result when LLM parsing fails entirely
_FALLBACK_RESULT: dict[str, Any] = {
    "country_iso_list": ["XX"],
    "horizon_days": 30,
    "category": "GENERAL",
}


async def parse_question(question: str) -> dict[str, Any]:
    """Parse a natural language forecast question into structured form via Gemini.

    Uses the existing GeminiClient with its built-in rate limiting and
    retry logic. On any failure (API error, malformed JSON, validation
    failure), returns safe defaults -- never raises.

    Args:
        question: The user's natural language forecast question.

    Returns:
        Dict with keys: country_iso_list, horizon_days, category.
    """
    try:
        return await _call_gemini_parser(question)
    except Exception as exc:
        logger.warning(
            "Question parsing failed, using fallback defaults: %s", exc,
            exc_info=True,
        )
        return _FALLBACK_RESULT.copy()


async def _call_gemini_parser(question: str) -> dict[str, Any]:
    """Internal: call Gemini and validate the response.

    Separated from parse_question so the outer function can catch
    any exception class uniformly.
    """
    import asyncio

    from src.forecasting.gemini_client import GeminiClient

    prompt = _PARSE_PROMPT_TEMPLATE.format(question=question)

    # GeminiClient.generate_content is synchronous -- wrap in thread
    client = GeminiClient()
    response = await asyncio.to_thread(
        client.generate_content,
        prompt=prompt,
        system_instruction=(
            "You are a geopolitical analysis assistant. Parse forecast "
            "questions into structured JSON. Be precise with country codes."
        ),
    )

    raw_text = response.text.strip()

    # Strip markdown code fences if Gemini wraps the JSON
    if raw_text.startswith("```"):
        lines = raw_text.split("\n")
        # Remove first line (```json) and last line (```)
        lines = [ln for ln in lines if not ln.strip().startswith("```")]
        raw_text = "\n".join(lines).strip()

    parsed = json.loads(raw_text)

    return _validate_parsed_result(parsed)


def _validate_parsed_result(parsed: Any) -> dict[str, Any]:
    """Validate and sanitize the LLM's parsed output.

    Ensures country codes are valid ISO-2, horizon is in range,
    and category is from the known set. Invalid values are corrected
    to safe defaults rather than rejected.
    """
    if not isinstance(parsed, dict):
        logger.warning("LLM returned non-dict: %s", type(parsed))
        return _FALLBACK_RESULT.copy()

    # Validate country_iso_list
    raw_codes = parsed.get("country_iso_list", [])
    if not isinstance(raw_codes, list):
        raw_codes = []

    valid_codes = [
        code.upper()
        for code in raw_codes
        if isinstance(code, str) and code.upper() in _VALID_ISO_CODES
    ]

    if not valid_codes:
        # All codes invalid or empty -- mark as unknown
        valid_codes = ["XX"]

    # Validate horizon_days
    horizon = parsed.get("horizon_days", 30)
    if not isinstance(horizon, (int, float)):
        horizon = 30
    horizon = max(7, min(365, int(horizon)))

    # Validate category
    category = parsed.get("category", "GENERAL")
    if not isinstance(category, str) or category not in _VALID_CATEGORIES:
        category = "GENERAL"

    return {
        "country_iso_list": valid_codes,
        "horizon_days": horizon,
        "category": category,
    }
