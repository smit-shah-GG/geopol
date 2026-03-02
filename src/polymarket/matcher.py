"""
Polymarket-to-Geopol prediction matcher using keyword pre-filter + LLM ranking.

Two-phase matching pipeline:
  1. Keyword pre-filter: country ISO match or >15% word overlap
  2. LLM ranking: Gemini evaluates top candidates for semantic match

Designed to be conservative -- only matches above the configured threshold
(default 0.6) are accepted. LLM failures degrade gracefully to no-match.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.db.models import Prediction
    from src.forecasting.gemini_client import GeminiClient

logger = logging.getLogger(__name__)


def _extract_words(text: str) -> set[str]:
    """Extract lowercase alphanumeric tokens from text."""
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _compute_word_overlap(words_a: set[str], words_b: set[str]) -> float:
    """Jaccard-like overlap: |intersection| / min(|a|, |b|).

    Uses min denominator rather than union to avoid penalizing
    short prediction questions matched against verbose PM descriptions.
    Returns 0.0 if either set is empty.
    """
    if not words_a or not words_b:
        return 0.0
    overlap = len(words_a & words_b)
    return overlap / min(len(words_a), len(words_b))


class PolymarketMatcher:
    """Matches Polymarket events to Geopol predictions via keyword + LLM pipeline.

    Args:
        gemini_client: Existing GeminiClient instance for LLM ranking.
        match_threshold: Minimum confidence (0.0-1.0) for accepting a match.
            Sourced from Settings.polymarket_match_threshold (default 0.6).
    """

    _MAX_CANDIDATES = 10

    _MATCH_PROMPT_TEMPLATE = """You are matching prediction market questions to geopolitical forecasts.

Given the Polymarket question below, determine which (if any) of the candidate forecasts is asking about the same underlying event or outcome.

**Polymarket question:**
Title: {pm_title}
Description: {pm_description}

**Candidate forecasts:**
{candidates_block}

Return ONLY valid JSON (no markdown, no explanation):
{{"match_id": "<id of best match or null>", "confidence": <float 0.0-1.0>}}

Rules:
- confidence > 0.8: clearly the same question/event
- confidence 0.5-0.8: related but may differ in scope or timeframe
- confidence < 0.5: not a reliable match
- If no candidate matches, return {{"match_id": null, "confidence": 0.0}}
"""

    def __init__(
        self, gemini_client: GeminiClient, match_threshold: float = 0.6
    ) -> None:
        self._gemini = gemini_client
        self._match_threshold = match_threshold

    async def match_event_to_predictions(
        self,
        polymarket_event: dict[str, Any],
        active_predictions: list[Prediction],
    ) -> tuple[str | None, float]:
        """Match a single Polymarket event to the best Geopol prediction.

        Phase 1 applies keyword/country pre-filter to prune candidates.
        Phase 2 (only if candidates exist) uses Gemini for semantic ranking.

        Args:
            polymarket_event: Event dict from PolymarketClient with at
                minimum 'title' key.
            active_predictions: List of active Prediction ORM objects.

        Returns:
            (prediction_id, confidence) if match found above threshold,
            (None, 0.0) otherwise.
        """
        pm_title = polymarket_event.get("title", "")
        pm_description = polymarket_event.get("description", "")
        pm_text = f"{pm_title} {pm_description}".strip()

        if not pm_text:
            return None, 0.0

        # Phase 1: Keyword pre-filter
        pm_words = _extract_words(pm_text)
        candidates: list[tuple[Prediction, float]] = []

        for pred in active_predictions:
            pred_words = _extract_words(pred.question)

            # Check 1: Country ISO match (fast path)
            country_match = False
            if pred.country_iso:
                iso_lower = pred.country_iso.lower()
                if iso_lower in pm_text.lower():
                    country_match = True

            # Check 2: Word overlap > 15%
            overlap = _compute_word_overlap(pred_words, pm_words)

            if country_match or overlap > 0.15:
                candidates.append((pred, overlap))

        if not candidates:
            return None, 0.0

        # Sort by overlap descending, take top N
        candidates.sort(key=lambda x: x[1], reverse=True)
        candidates = candidates[: self._MAX_CANDIDATES]

        logger.debug(
            "Pre-filter passed %d candidates for PM event '%s'",
            len(candidates),
            pm_title[:60],
        )

        # Phase 2: LLM ranking
        return await self._llm_rank(pm_title, pm_description, candidates)

    async def _llm_rank(
        self,
        pm_title: str,
        pm_description: str,
        candidates: list[tuple[Prediction, float]],
    ) -> tuple[str | None, float]:
        """Ask Gemini to pick the best match from pre-filtered candidates."""
        candidates_block_lines: list[str] = []
        valid_ids: set[str] = set()

        for i, (pred, _overlap) in enumerate(candidates, 1):
            candidates_block_lines.append(
                f"{i}. ID: {pred.id}\n"
                f"   Question: {pred.question}\n"
                f"   Country: {pred.country_iso or 'N/A'}\n"
                f"   Probability: {pred.probability:.2f}"
            )
            valid_ids.add(pred.id)

        prompt = self._MATCH_PROMPT_TEMPLATE.format(
            pm_title=pm_title,
            pm_description=pm_description or "(no description)",
            candidates_block="\n".join(candidates_block_lines),
        )

        try:
            # GeminiClient.generate_content is synchronous -- wrap in thread
            response = await asyncio.to_thread(
                self._gemini.generate_content, prompt
            )

            raw_text = response.text.strip()
            # Strip markdown fences if present
            if raw_text.startswith("```"):
                raw_text = re.sub(r"^```(?:json)?\s*", "", raw_text)
                raw_text = re.sub(r"\s*```$", "", raw_text)

            result = json.loads(raw_text)
            match_id = result.get("match_id")
            confidence = float(result.get("confidence", 0.0))

            # Validate
            if match_id is not None and match_id not in valid_ids:
                logger.warning(
                    "LLM returned invalid match_id %r (not in candidates)", match_id
                )
                return None, 0.0

            if match_id is None or confidence < self._match_threshold:
                logger.debug(
                    "LLM match below threshold: id=%s, confidence=%.2f (threshold=%.2f)",
                    match_id,
                    confidence,
                    self._match_threshold,
                )
                return None, 0.0

            logger.info(
                "LLM matched PM '%s' -> prediction %s (confidence=%.2f)",
                pm_title[:50],
                match_id,
                confidence,
            )
            return match_id, confidence

        except json.JSONDecodeError as exc:
            logger.warning("Failed to parse LLM match response as JSON: %s", exc)
            return None, 0.0
        except Exception as exc:
            logger.warning("LLM matching failed for '%s': %s", pm_title[:50], exc)
            return None, 0.0
