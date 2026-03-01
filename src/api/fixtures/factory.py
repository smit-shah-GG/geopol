"""
Mock forecast factory and fixture loader.

Provides functions for loading hand-crafted JSON fixtures (golden samples)
and generating arbitrary mock forecasts for testing. All output is validated
against the Pydantic DTOs — if the fixture data is structurally invalid,
it fails here rather than in the API layer.
"""

import json
import random
import string
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from src.api.schemas.country import CountryRiskSummary
from src.api.schemas.forecast import (
    CalibrationDTO,
    EnsembleInfoDTO,
    EvidenceDTO,
    ForecastResponse,
    ScenarioDTO,
)

_SCENARIOS_DIR = Path(__file__).parent / "scenarios"

# Country code -> fixture filename mapping
_FIXTURE_MAP: dict[str, str] = {
    "SY": "syria.json",
    "UA": "ukraine.json",
    "MM": "myanmar.json",
}

# Plausible template data for random forecast generation
_SAMPLE_QUESTIONS: list[dict[str, str]] = [
    {
        "iso": "SY",
        "q": "Will the Syrian government regain control of Idlib province within {h} days?",
        "category": "conflict",
    },
    {
        "iso": "UA",
        "q": "Will Russia launch a major offensive on the Zaporizhzhia front within {h} days?",
        "category": "conflict",
    },
    {
        "iso": "MM",
        "q": "Will ASEAN impose sanctions on Myanmar's military junta within {h} days?",
        "category": "diplomatic",
    },
    {
        "iso": "IR",
        "q": "Will Iran exceed 90% uranium enrichment within {h} days?",
        "category": "security",
    },
    {
        "iso": "TW",
        "q": "Will China conduct military exercises in the Taiwan Strait within {h} days?",
        "category": "conflict",
    },
    {
        "iso": "SD",
        "q": "Will the RSF and SAF agree to a ceasefire in Sudan within {h} days?",
        "category": "diplomatic",
    },
    {
        "iso": "KP",
        "q": "Will North Korea conduct a nuclear weapons test within {h} days?",
        "category": "security",
    },
    {
        "iso": "VE",
        "q": "Will Venezuela hold internationally recognized elections within {h} days?",
        "category": "political",
    },
]


def load_fixture(country_code: str) -> ForecastResponse:
    """Load a hand-crafted fixture file and validate it as a ForecastResponse.

    Args:
        country_code: ISO alpha-2 country code (e.g., "SY", "UA", "MM").

    Returns:
        Validated ForecastResponse instance.

    Raises:
        FileNotFoundError: If no fixture exists for the given country code.
        ValidationError: If the fixture JSON is structurally invalid.
    """
    filename = _FIXTURE_MAP.get(country_code.upper())
    if filename is None:
        raise FileNotFoundError(
            f"No fixture for country code '{country_code}'. "
            f"Available: {', '.join(sorted(_FIXTURE_MAP.keys()))}"
        )

    fixture_path = _SCENARIOS_DIR / filename
    with open(fixture_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return ForecastResponse.model_validate(data)


def load_all_fixtures() -> dict[str, ForecastResponse]:
    """Load all available fixture files, validated.

    Returns:
        Dict mapping country codes to validated ForecastResponse instances.
    """
    result: dict[str, ForecastResponse] = {}
    for code in _FIXTURE_MAP:
        result[code] = load_fixture(code)
    return result


def create_mock_forecast(
    country_iso: str = "XX",
    question: Optional[str] = None,
    horizon_days: int = 30,
    num_scenarios: int = 2,
    **overrides: object,
) -> ForecastResponse:
    """Generate a random but structurally valid mock forecast.

    The output is validated against ForecastResponse — if the factory
    produces invalid data, it fails immediately rather than silently
    passing garbage to consumers.

    Args:
        country_iso: ISO country code for the forecast.
        question: Forecast question text. Auto-generated if None.
        horizon_days: Forecast horizon in days.
        num_scenarios: Number of top-level scenarios to generate.
        **overrides: Any ForecastResponse field to override.

    Returns:
        Validated ForecastResponse instance with random but plausible data.
    """
    now = datetime.now(timezone.utc)
    forecast_id = f"fc-{country_iso.lower()}-{uuid.uuid4().hex[:8]}"

    if question is None:
        template = random.choice(_SAMPLE_QUESTIONS)
        question = template["q"].format(h=horizon_days)
        category = template["category"]
    else:
        category = "conflict"

    # Generate probability that isn't too extreme (0.15 - 0.85 range for realism)
    probability = round(random.uniform(0.15, 0.85), 2)
    confidence = round(random.uniform(0.45, 0.90), 2)

    # Generate scenario probabilities that are consistent
    # (affirmative scenarios sum to ~probability, negative to ~1-probability)
    scenarios = _generate_scenarios(
        country_iso=country_iso,
        num_scenarios=num_scenarios,
        total_probability=probability,
    )

    # Generate evidence across all scenarios
    evidence_count = sum(
        len(s.evidence_sources)
        + sum(len(c.evidence_sources) for c in s.child_scenarios)
        for s in scenarios
    )

    # LLM and TKG probabilities should average to roughly the final probability
    llm_weight = 0.6
    tkg_weight = 0.4
    llm_prob = round(
        min(1.0, max(0.0, probability + random.uniform(-0.1, 0.1))), 2
    )
    tkg_prob = round(
        min(1.0, max(0.0, probability + random.uniform(-0.15, 0.15))), 2
    )

    temperature = round(random.uniform(1.0, 1.3), 2)

    data: dict = {
        "forecast_id": forecast_id,
        "question": question,
        "prediction": f"Mock prediction for {country_iso}: {question[:80]}...",
        "probability": probability,
        "confidence": confidence,
        "horizon_days": horizon_days,
        "scenarios": [s.model_dump() for s in scenarios],
        "reasoning_summary": (
            f"This is a mock forecast for {country_iso}. "
            f"The ensemble combined LLM (P={llm_prob}) and TKG (P={tkg_prob}) "
            f"estimates with temperature scaling T={temperature}."
        ),
        "evidence_count": evidence_count,
        "ensemble_info": {
            "llm_probability": llm_prob,
            "tkg_probability": tkg_prob,
            "weights": {"llm": llm_weight, "tkg": tkg_weight},
            "temperature_applied": temperature,
        },
        "calibration": {
            "category": category,
            "temperature": temperature,
            "historical_accuracy": round(random.uniform(0.55, 0.85), 2),
            "brier_score": round(random.uniform(0.10, 0.35), 2),
            "sample_size": random.randint(20, 200),
        },
        "created_at": now.isoformat(),
        "expires_at": (now + timedelta(days=horizon_days)).isoformat(),
    }

    # Apply any caller overrides
    data.update(overrides)

    return ForecastResponse.model_validate(data)


def create_mock_country_risk(
    iso_code: str,
    risk_score: Optional[float] = None,
    forecast_count: Optional[int] = None,
) -> CountryRiskSummary:
    """Generate a random but valid CountryRiskSummary.

    Args:
        iso_code: ISO country code.
        risk_score: Override risk score (random if None).
        forecast_count: Override forecast count (random if None).

    Returns:
        Validated CountryRiskSummary instance.
    """
    now = datetime.now(timezone.utc)

    if risk_score is None:
        risk_score = round(random.uniform(0.1, 0.95), 2)
    if forecast_count is None:
        forecast_count = random.randint(1, 10)

    trends = ["rising", "stable", "falling"]
    # Weight trend by risk_score: high risk more likely rising
    if risk_score > 0.7:
        trend_weights = [0.6, 0.3, 0.1]
    elif risk_score < 0.3:
        trend_weights = [0.1, 0.3, 0.6]
    else:
        trend_weights = [0.33, 0.34, 0.33]

    trend = random.choices(trends, weights=trend_weights, k=1)[0]

    return CountryRiskSummary(
        iso_code=iso_code,
        risk_score=risk_score,
        forecast_count=forecast_count,
        top_question=f"Will a significant event occur in {iso_code} within 30 days?",
        top_probability=round(random.uniform(0.2, 0.8), 2),
        trend=trend,  # type: ignore[arg-type]
        last_updated=now,
    )


def get_empty_country_response(iso_code: str) -> list:
    """Return an empty forecast list for a country with no active forecasts.

    This is used for countries that exist but have no predictions,
    ensuring the frontend can handle the empty state gracefully.

    Args:
        iso_code: ISO country code.

    Returns:
        Empty list (the items field of PaginatedResponse).
    """
    return []


def _generate_scenarios(
    country_iso: str,
    num_scenarios: int,
    total_probability: float,
) -> list[ScenarioDTO]:
    """Generate a list of plausible mock scenarios.

    Ensures probabilities are distributed in a way that's coherent:
    the first scenario gets the largest share, subsequent scenarios
    get decreasing shares.
    """
    scenarios: list[ScenarioDTO] = []
    remaining = 1.0  # Total probability budget across all scenarios

    for i in range(num_scenarios):
        # Distribute probability: give each scenario a share of remaining
        if i == num_scenarios - 1:
            prob = round(remaining, 2)
        else:
            prob = round(remaining * random.uniform(0.3, 0.7), 2)
            remaining -= prob

        prob = max(0.05, min(0.95, prob))

        # First scenario tends to be affirmative
        answers_affirmative = i == 0 or random.random() < 0.4

        scenario_id = f"{country_iso.lower()}-mock-s{i + 1}"

        evidence = [
            EvidenceDTO(
                source=random.choice(["GDELT", "TKG pattern", "RAG match"]),
                description=f"Mock evidence {j + 1} for scenario {i + 1}",
                confidence=round(random.uniform(0.4, 0.9), 2),
                timestamp=datetime.now(timezone.utc) - timedelta(days=random.randint(1, 14)),
                gdelt_event_id=(
                    f"EVT{random.randint(100000, 999999)}"
                    if random.random() < 0.5
                    else None
                ),
            )
            for j in range(random.randint(1, 3))
        ]

        # Optionally add child scenarios to the first scenario
        children: list[ScenarioDTO] = []
        if i == 0 and num_scenarios > 1:
            child = ScenarioDTO(
                scenario_id=f"{scenario_id}-1",
                description=f"Sub-scenario: conditional outcome if {scenario_id} occurs",
                probability=round(random.uniform(0.2, 0.6), 2),
                answers_affirmative=answers_affirmative,
                entities=[country_iso, "External Actor"],
                timeline=["T+14d: Follow-on event occurs"],
                evidence_sources=[
                    EvidenceDTO(
                        source="TKG pattern",
                        description="Historical pattern match for conditional outcome",
                        confidence=round(random.uniform(0.4, 0.7), 2),
                    )
                ],
                child_scenarios=[],
            )
            children = [child]

        scenarios.append(
            ScenarioDTO(
                scenario_id=scenario_id,
                description=f"Mock scenario {i + 1} for {country_iso}: "
                + _random_scenario_description(),
                probability=prob,
                answers_affirmative=answers_affirmative,
                entities=[country_iso, "Regional Power", "International Organization"],
                timeline=[
                    f"T+{random.randint(1, 7)}d: Initial trigger event",
                    f"T+{random.randint(8, 20)}d: Escalation or de-escalation",
                    f"T+{random.randint(21, 30)}d: Resolution or continuation",
                ],
                evidence_sources=evidence,
                child_scenarios=children,
            )
        )

    return scenarios


def _random_scenario_description() -> str:
    """Generate a plausible-sounding scenario description fragment."""
    templates = [
        "Military escalation driven by border tensions and domestic political pressure",
        "Diplomatic breakthrough through multilateral negotiations and external mediation",
        "Status quo continuation with gradual deterioration of humanitarian conditions",
        "Economic sanctions trigger regime behavioral change and policy concessions",
        "Internal political crisis leads to power transition and policy realignment",
        "External intervention shifts the balance of power and forces negotiations",
        "Ceasefire agreement holds under international monitoring framework",
        "Proxy conflict escalation drawing in regional powers and complicating resolution",
    ]
    return random.choice(templates)
