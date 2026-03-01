"""
Outcome resolution comparing expired predictions against GDELT ground truth.

For each expired prediction that lacks an OutcomeRecord, the resolver:
1. Queries GDELT events in the prediction's time window.
2. Uses Gemini for LLM-based resolution (returns 0.0-1.0 outcome score).
3. Falls back to heuristic actor+event_code matching if Gemini is
   unavailable or budget is exhausted.
4. Persists an OutcomeRecord row with resolution method and evidence.

LLM-based resolution is more accurate than keyword matching for nuanced
geopolitical outcomes, but the heuristic fallback ensures resolution
continues even without Gemini access.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from src.database.storage import EventStorage
from src.db.models import OutcomeRecord, Prediction
from src.forecasting.gemini_client import GeminiClient

logger = logging.getLogger(__name__)

_RESOLUTION_PROMPT = """\
You are a geopolitical outcome analyst. Given a prediction and the events \
that occurred during its forecast window, assess whether the predicted \
outcome occurred.

PREDICTION:
Question: {question}
Probability given: {probability:.3f}
Created: {created_at}
Expired: {expires_at}

EVENTS DURING WINDOW:
{events_text}

Rate the extent to which the predicted outcome occurred as a float \
between 0.0 (definitely did not occur) and 1.0 (definitely occurred).

Consider partial occurrences: if elements of the prediction materialized \
but not the full outcome, assign a proportional score.

Return ONLY a JSON object:
{{"outcome": <float 0.0-1.0>, "reasoning": "<brief explanation>"}}
"""


class OutcomeResolver:
    """Resolve expired predictions against GDELT ground truth.

    Uses LLM-based resolution (Gemini) as primary method, with heuristic
    actor+event_code matching as fallback.

    Attributes:
        async_session_factory: SQLAlchemy async session factory.
        event_storage: EventStorage for GDELT event queries.
        gemini_client: Optional GeminiClient for LLM-based resolution.
    """

    def __init__(
        self,
        async_session_factory: async_sessionmaker[AsyncSession],
        event_storage: EventStorage,
        gemini_client: GeminiClient | None = None,
    ) -> None:
        self._session_factory = async_session_factory
        self._event_storage = event_storage
        self._gemini_client = gemini_client

    async def resolve_expired_predictions(
        self,
        batch_size: int = 20,
    ) -> list[OutcomeRecord]:
        """Find and resolve expired predictions that lack OutcomeRecords.

        Queries PostgreSQL for predictions past their expires_at with no
        corresponding OutcomeRecord. For each, queries GDELT events in the
        prediction's time window and resolves via Gemini or heuristic.

        Args:
            batch_size: Maximum predictions to resolve per run.

        Returns:
            List of newly created OutcomeRecord instances.
        """
        async with self._session_factory() as session:
            # Find expired predictions without outcomes
            now = datetime.now(timezone.utc)

            # Subquery: prediction_ids that already have outcomes
            resolved_ids = select(OutcomeRecord.prediction_id).scalar_subquery()

            stmt = (
                select(Prediction)
                .where(
                    Prediction.expires_at < now,
                    Prediction.id.not_in(resolved_ids),
                )
                .order_by(Prediction.expires_at.asc())
                .limit(batch_size)
            )

            result = await session.execute(stmt)
            expired = list(result.scalars().all())

            if not expired:
                logger.info("No expired predictions requiring resolution")
                return []

            logger.info("Resolving %d expired predictions", len(expired))

            outcomes: list[OutcomeRecord] = []
            for prediction in expired:
                outcome = await self._resolve_single(prediction, session)
                if outcome is not None:
                    outcomes.append(outcome)

            await session.commit()
            logger.info(
                "Resolved %d/%d expired predictions", len(outcomes), len(expired)
            )
            return outcomes

    async def _resolve_single(
        self,
        prediction: Prediction,
        session: AsyncSession,
    ) -> OutcomeRecord | None:
        """Resolve a single expired prediction.

        Tries LLM-based resolution first, falls back to heuristic.
        """
        # Query GDELT events in the prediction's time window
        start_str = prediction.created_at.strftime("%Y-%m-%d")
        end_str = prediction.expires_at.strftime("%Y-%m-%d")

        events = await asyncio.to_thread(
            self._event_storage.get_events,
            start_date=start_str,
            end_date=end_str,
            min_mentions=3,
            limit=30,
        )

        # Try LLM-based resolution
        outcome_score: float
        evidence_ids: list[str]
        method: str
        notes: str

        if self._gemini_client is not None and events:
            try:
                outcome_score, evidence_ids, notes = await self._llm_resolve(
                    prediction, events
                )
                method = "gemini"
            except Exception as exc:
                logger.warning(
                    "LLM resolution failed for %s, falling back to heuristic: %s",
                    prediction.id,
                    exc,
                )
                outcome_score, evidence_ids = self._heuristic_resolve(
                    prediction, events
                )
                method = "heuristic"
                notes = f"LLM fallback due to: {exc}"
        else:
            outcome_score, evidence_ids = self._heuristic_resolve(
                prediction, events
            )
            method = "heuristic"
            notes = "No Gemini client available" if not self._gemini_client else "No events found"

        outcome = OutcomeRecord(
            prediction_id=prediction.id,
            outcome=outcome_score,
            resolution_date=datetime.now(timezone.utc),
            resolution_method=method,
            evidence_gdelt_ids=evidence_ids,
            notes=notes,
        )
        session.add(outcome)

        logger.info(
            "Resolved prediction %s: outcome=%.2f method=%s evidence_count=%d",
            prediction.id,
            outcome_score,
            method,
            len(evidence_ids),
        )
        return outcome

    async def _llm_resolve(
        self,
        prediction: Prediction,
        events: list,
    ) -> tuple[float, list[str], str]:
        """Use Gemini for LLM-based outcome resolution.

        Returns:
            Tuple of (outcome_score, evidence_gdelt_ids, reasoning_notes).
        """
        events_text = self._format_events_for_resolution(events)

        prompt = _RESOLUTION_PROMPT.format(
            question=prediction.question,
            probability=prediction.probability,
            created_at=prediction.created_at.isoformat(),
            expires_at=prediction.expires_at.isoformat(),
            events_text=events_text,
        )

        response = await asyncio.to_thread(
            self._gemini_client.generate,
            prompt=prompt,
        )

        # Parse response
        text = response.strip()
        if text.startswith("```"):
            first_newline = text.index("\n")
            text = text[first_newline + 1 :]
        if text.endswith("```"):
            text = text[:-3].rstrip()

        try:
            data = json.loads(text)
            outcome = float(data.get("outcome", 0.5))
            outcome = max(0.0, min(1.0, outcome))
            reasoning = str(data.get("reasoning", ""))
        except (json.JSONDecodeError, ValueError, TypeError) as exc:
            logger.warning("Failed to parse LLM resolution response: %s", exc)
            outcome = 0.5
            reasoning = f"Parse error: {exc}"

        # Collect evidence IDs from events used
        evidence_ids = [
            e.gdelt_id for e in events[:10]
            if e.gdelt_id is not None
        ]

        return outcome, evidence_ids, reasoning

    def _heuristic_resolve(
        self,
        prediction: Prediction,
        events: list,
    ) -> tuple[float, list[str]]:
        """Heuristic fallback: actor+event_code matching from GDELT events.

        Extracts entity names from the prediction's entities JSON field.
        Searches events for actor code matches. Returns a score proportional
        to the number of matching events.

        Args:
            prediction: The expired Prediction ORM instance.
            events: GDELT events in the prediction's time window.

        Returns:
            Tuple of (outcome_score, matching_event_ids).
        """
        if not events:
            return 0.0, []

        # Extract entities from prediction
        pred_entities: set[str] = set()
        entities_data = prediction.entities
        if isinstance(entities_data, list):
            for entity in entities_data:
                if isinstance(entity, str):
                    pred_entities.add(entity.upper())

        # Also extract keywords from the question
        question_words = set(prediction.question.upper().split())

        matching_ids: list[str] = []
        match_score = 0.0

        for event in events:
            matched = False

            # Check actor code matches
            if event.actor1_code and event.actor1_code.upper() in pred_entities:
                matched = True
            if event.actor2_code and event.actor2_code.upper() in pred_entities:
                matched = True

            # Check if event actors appear in question keywords (broader match)
            if not matched and event.actor1_code:
                if event.actor1_code.upper() in question_words:
                    matched = True
            if not matched and event.actor2_code:
                if event.actor2_code.upper() in question_words:
                    matched = True

            if matched and event.gdelt_id:
                matching_ids.append(event.gdelt_id)
                # Weight by significance
                mentions = event.num_mentions or 1
                match_score += min(mentions / 100.0, 1.0)

        if not matching_ids:
            return 0.0, []

        # Normalize: more matching events = higher confidence in outcome
        # Cap at 1.0, with diminishing returns
        normalized = min(1.0, match_score / 3.0)
        return normalized, matching_ids

    def _format_events_for_resolution(self, events: list) -> str:
        """Format events for the LLM resolution prompt."""
        lines: list[str] = []
        for i, event in enumerate(events[:15], 1):
            actors = []
            if event.actor1_code:
                actors.append(event.actor1_code)
            if event.actor2_code:
                actors.append(event.actor2_code)
            actor_str = " vs ".join(actors) if actors else "unknown"

            line = (
                f"{i}. [{event.event_date}] {actor_str} | "
                f"EventCode={event.event_code or 'N/A'} | "
                f"GoldsteinScale={event.goldstein_scale or 0:.1f} | "
                f"Mentions={event.num_mentions or 0}"
            )
            if event.title:
                line += f" | {event.title[:100]}"
            lines.append(line)
        return "\n".join(lines)
