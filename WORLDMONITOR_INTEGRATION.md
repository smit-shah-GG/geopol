# World Monitor x Geopol: Integration Analysis

> Cross-project synergy analysis between [World Monitor](https://worldmonitor.app) (public, TypeScript real-time OSINT dashboard) and Geopol (private, Python geopolitical forecasting engine).

---

## Table of Contents

- [Project Comparison](#project-comparison)
- [Philosophical Alignment](#philosophical-alignment)
- [Technical Integration Points](#technical-integration-points)
- [Integration Options Overview](#integration-options-overview)
- [Deep Dive: Forecast-Enriched World Monitor](#deep-dive-forecast-enriched-world-monitor)
  - [UX Vision](#ux-vision)
  - [What to Build: World Monitor Side](#what-to-build-world-monitor-side)
  - [What to Build: Geopol Side](#what-to-build-geopol-side)
  - [Data Flow Architecture](#data-flow-architecture)
  - [Forecast Output Mapping](#forecast-output-mapping)
  - [Panel Definitions](#panel-definitions)
  - [Map Layer Integration](#map-layer-integration)
  - [Country Brief Enhancement](#country-brief-enhancement)
  - [CII ↔ Geopol Feedback Loop](#cii--geopol-feedback-loop)
  - [Deployment Architecture](#deployment-architecture)
  - [Cost and Complexity Estimates](#cost-and-complexity-estimates)
  - [Critical Success Factors](#critical-success-factors)
  - [Phased Implementation Roadmap](#phased-implementation-roadmap)
- [Three-Project Synergy: World Monitor + Geopol + OSINT Double](#three-project-synergy-world-monitor--geopol--osint-double)

---

## Project Comparison

|                       | **World Monitor**                                    | **Geopol**                                            |
| --------------------- | ---------------------------------------------------- | ----------------------------------------------------- |
| **Core function**     | Real-time global intelligence *dashboard*            | Geopolitical *forecasting engine*                     |
| **Stack**             | TypeScript, Vite, deck.gl, Vercel Edge, Tauri        | Python, JAX, NetworkX, Gemini, LlamaIndex, ChromaDB   |
| **Time orientation**  | **Present** — what is happening now                  | **Future** — what will happen next                    |
| **Input**             | 50+ APIs, 298 RSS domains, WebSocket streams         | GDELT (500K-1M events/day), temporal knowledge graphs  |
| **Processing**        | Keyword classification, clustering, CII scoring      | Dual-path reasoning: LLM scenarios + TKG pattern matching |
| **Output**            | Interactive 3D globe + 45 panels                     | Probabilistic forecasts with reasoning chains          |
| **AI approach**       | LLM summarization (unverified)                       | Ensemble prediction (60% LLM + 40% TKG, calibrated)   |
| **Explainability**    | Minimal — "AI analysis" text blob                    | Full — scenario trees, evidence chains, reasoning traces |
| **Calibration**       | None — confidence is implicit                        | Temperature scaling + isotonic regression per category  |
| **Data model**        | Flat events + RSS items                              | Temporal knowledge graph (entities → relations → entities with timestamps) |
| **User model**        | Human analyst observing a dashboard                  | Analyst/decision-maker seeking probabilistic forecasts  |
| **Deployment**        | Vercel (web) + Tauri (desktop) + Railway (relay)     | CLI (local execution)                                  |
| **Version**           | v2.5.11                                              | v1.1.1                                                 |
| **License**           | AGPL-3.0 (public)                                    | Private                                                |

---

## Philosophical Alignment

### Where They Align

1. **OSINT for everyone** — World Monitor is "100% free & open source." Geopol targets the same cost asymmetry: "Government analysts can't process 500K daily events." Both reject the premise that intelligence requires six-figure tools or clearances.

2. **Multi-source as first principle** — World Monitor aggregates 50+ APIs because no single source tells the full story. Geopol's ensemble predictor exists for the same reason — neither LLMs alone nor graph patterns alone produce good forecasts. Both treat source diversity as architecturally fundamental, not an afterthought.

3. **Structured confidence over binary truth** — World Monitor's CII scores countries on a continuous scale with weighted multi-signal blending. Geopol produces calibrated probabilities with per-category temperature scaling. Neither says "this will happen" — both say "this is X% likely, here's why."

4. **Geographic grounding** — World Monitor renders everything on a 3D globe because location is meaning. Geopol's knowledge graph is built from GDELT events that carry actor country codes, event locations, and geographic context. Both treat geography as a first-class dimension of intelligence.

5. **Graceful degradation** — World Monitor's 4-tier LLM fallback (Ollama → Groq → OpenRouter → T5). Geopol's ensemble can run with TKG disabled, RAG disabled, or both — falling back to pure LLM scenarios. Both systems produce useful output even when subsystems fail.

### Where They Diverge

1. **Temporal axis** — This is the fundamental complementary axis. World Monitor answers "what is happening?" Geopol answers "what will happen?" One is a sensor, the other is an oracle. Together they form the complete question: "given what is happening, what will happen next?"

2. **Depth of reasoning** — World Monitor's AI is single-pass: feed news to LLM, get summary. Geopol uses 4-step multi-pass reasoning: scenario generation → graph validation → refinement → extraction, with RAG-retrieved historical precedents at each step. The reasoning depth difference is 1x vs. 4x.

3. **Data model sophistication** — World Monitor treats events as flat items (title, source, timestamp, location). Geopol models events as triples in a temporal knowledge graph (Actor1 → Relation → Actor2, with temporal metadata). The graph structure enables pattern matching that flat events cannot support — e.g., "every time Actor A sanctioned Actor B, Actor C increased military cooperation with B within 30 days."

4. **Calibration** — World Monitor's AI outputs are uncalibrated (the model says what it says). Geopol explicitly calibrates predictions using temperature scaling and isotonic regression against historical outcomes, decomposing accuracy into reliability, resolution, and uncertainty via Brier scores. This is the difference between "AI thinks X" and "this system is historically Y% accurate when it says X."

5. **Explainability architecture** — World Monitor's Country Brief shows an AI-generated text summary. Geopol returns structured reasoning: scenario trees (mutually exclusive branches), evidence sources (specific GDELT events), TKG pattern matches, ensemble component weights, and calibration diagnostics. Every number can be traced back to evidence.

### The Time Axis Complement

Together, the two projects create a temporal intelligence continuum:

| Time Horizon     | World Monitor                       | Geopol                                    |
| ---------------- | ----------------------------------- | ----------------------------------------- |
| **Past (7 days)**| 7-day event timeline, trend baselines | Historical pattern retrieval (RAG)       |
| **Present**      | Real-time dashboard, live feeds      | GDELT ingestion (15-min cycles)          |
| **Near future**  | Keyword spike detection (hours)      | Scenario generation (days-weeks)          |
| **Medium future** | —                                   | Calibrated forecasts (weeks-months)       |
| **Validation**   | CII score (reactive)                 | Brier score (predictive accuracy)         |

World Monitor shows the world as it is. Geopol shows the world as it might become. Neither alone tells the full story.

---

## Technical Integration Points

### 1. CII Scores as Forecast Triggers

World Monitor's Country Instability Index (CII) identifies countries experiencing rising instability. When CII spikes, Geopol could automatically generate forecasts for likely next developments.

**Data flow**: World Monitor computes CII from ACLED protests, conflict events, outages, displacement, and climate anomalies. When a country's CII crosses a threshold:

```
WM CII spike detected: Syria → 8.2 (was 5.1)
    → Auto-generate Geopol question:
      "Will the Syrian conflict escalate to involve direct
       foreign military intervention within 60 days?"
    → Geopol returns: P(yes) = 0.43, confidence = 0.68
    → Scenarios: [intervention via Turkey, via Israel, status quo]
    → Evidence: [GDELT events showing military buildup patterns]
```

### 2. Geopol Forecasts Enriching Country Briefs

World Monitor's Country Brief Pages currently show:
- CII score ring
- AI-generated text analysis (single LLM pass, unverified)
- Top news with citation anchoring
- 7-day event timeline
- Active signal chips

Geopol could add:
- **Probabilistic forecasts** — "67% chance of further escalation within 30 days"
- **Scenario trees** — Interactive branching scenarios the analyst can explore
- **Historical precedent matches** — "Similar patterns in 2014 led to X in 73% of cases"
- **Calibrated confidence** — "This model is historically 72% accurate at this confidence level"
- **Evidence chains** — Specific GDELT events and graph patterns supporting each scenario

### 3. GDELT as Shared Data Source

Both projects consume GDELT data, but differently:
- **World Monitor**: Uses GDELT for geo-events (protests, conflicts) via its GDELT API proxy
- **Geopol**: Ingests 500K-1M GDELT events daily into temporal knowledge graphs

Geopol's processed knowledge graph is a superset of World Monitor's GDELT usage. The graph could serve as a shared analytical backbone — World Monitor queries it for current events, Geopol queries it for historical patterns.

### 4. Entity Resolution Bridge

World Monitor has static entity catalogs (220+ military bases, 83 ports, 92 stock exchanges) with coordinates. Geopol normalizes GDELT actors to country codes and actor roles (GOV, MIL, REB, OPP).

A shared entity ontology would allow:
- Geopol forecasts to reference World Monitor's infrastructure entities ("conflict near Incirlik Air Base")
- World Monitor to display Geopol's actor-relation analysis on the globe
- Both systems to use consistent country/actor identification

### 5. Prediction Markets Calibration

World Monitor already displays Polymarket prediction markets data. Geopol produces calibrated probability forecasts. These are natural comparison points:

- Display Geopol's forecast alongside Polymarket's crowd prediction for the same question
- Track calibration: is Geopol or the prediction market more accurate over time?
- Use prediction market data as additional calibration signal for Geopol's temperature scaling

### 6. Temporal Knowledge Graph as Map Layer

Geopol's knowledge graph contains actor-to-actor relations with geographic grounding. This could be rendered as a new World Monitor map layer:

- **Actor relation arcs** — Lines between countries colored by relation type (cooperation: blue, conflict: red)
- **Temporal animation** — Playback of relation changes over time
- **Density heatmap** — Where the most geopolitical activity is concentrated
- **Prediction overlays** — Forecast probability mapped to geographic regions

---

## Integration Options Overview

### Option A: Forecast API (Loose Coupling)

Geopol exposes a REST API. World Monitor calls it when rendering Country Brief Pages or when CII spikes trigger.

```
World Monitor (dashboard)
    | HTTP GET /forecast?country=SY&horizon=30d
    v
Geopol API (FastAPI)
    | cached forecast or on-demand generation
    v
World Monitor Country Brief (enriched with forecasts)
```

**Effort**: Low-Medium. Geopol needs a FastAPI wrapper. World Monitor adds a service client.
**Benefit**: Country Briefs gain probabilistic forecasts with full explainability.

### Option B: Shared Knowledge Graph

Geopol's temporal knowledge graph becomes a shared analytical layer. World Monitor queries it for both current event context and historical patterns.

**Effort**: Medium-High. Requires graph database (Neo4j or similar) accessible to both systems.
**Benefit**: Deep analytical capability. World Monitor can answer "has this pattern happened before?"

### Option C: Forecast-Enriched Dashboard (Deep Integration)

World Monitor gains a forecasting dimension: every country, every hotspot, every CII spike comes with Geopol-generated probabilistic forecasts, scenario trees, and evidence chains. New panels, map layers, and interactive scenario exploration.

**Effort**: High. New panels, services, map layers, and a production Geopol API deployment.
**Benefit**: Transforms World Monitor from "what is happening" to "what is happening and what will happen."

---

## Deep Dive: Forecast-Enriched World Monitor

### UX Vision

The integration adds a **forward-looking dimension** to every surface of World Monitor:

1. **Country Brief Pages** gain a "Forecast" tab showing:
   - Top 3 probabilistic forecasts for the country (auto-generated from CII context)
   - Interactive scenario tree (click to expand branches, see evidence for each)
   - Historical precedent panel ("this pattern matched 4 previous events, outcomes were...")
   - Calibration badge ("this model is 72% accurate at this confidence level")

2. **CII Panel** gains forecast annotations:
   - Countries with rising CII show a small forecast chip: "67% escalation risk → 30d"
   - Click opens the full scenario tree

3. **New Forecast Panel** in the dashboard:
   - Top 10 most consequential active forecasts (across all countries)
   - Each shows: question, probability, confidence, trend arrow, last updated
   - Sortable by probability, confidence, or recency

4. **Globe Overlay** — new "Forecast Risk" layer:
   - Countries colored by aggregate forecast risk (green → yellow → red)
   - Distinct from CII (CII = present instability, forecast risk = future probability)
   - Click any country to see its forecasts

5. **Scenario Explorer** modal:
   - Full-screen interactive scenario tree
   - Each node shows: description, probability, evidence count, timeline
   - Branching visualization (like a decision tree)
   - Evidence panel showing specific GDELT events and graph patterns

### What to Build: World Monitor Side

#### New Components (~4 components)

| Component | Purpose | Complexity |
| --- | --- | --- |
| `ForecastPanel.ts` | Dashboard panel showing top active forecasts across countries | Medium |
| `ScenarioExplorer.ts` | Full-screen modal with interactive scenario tree visualization | High |
| `ForecastCard.ts` | Reusable card showing a single forecast (question, probability, confidence, scenarios) | Low |
| `CalibrationBadge.ts` | Small badge showing model calibration quality ("72% historically accurate") | Low |

#### New Services (~2 service files)

| Service | Purpose |
| --- | --- |
| `services/forecast/index.ts` | HTTP client for Geopol REST API (forecast CRUD, country queries) |
| `services/forecast/scenario-mapper.ts` | Maps Geopol's ForecastOutput to WM display types + map markers |

#### New Map Layer (~1 deck.gl layer)

| Layer | Rendering |
| --- | --- |
| `ForecastRiskLayer` | Choropleth layer coloring countries by aggregate forecast risk, distinct from CII |

#### Modified Components

| Component | Change |
| --- | --- |
| `CountryBriefPage.ts` | Add "Forecast" tab with scenario tree, evidence, calibration |
| `CIIPanel.ts` | Add forecast chips next to countries with rising CII |
| `DeckGLMap.ts` | Register ForecastRiskLayer |

#### Modified Services

| File | Change |
| --- | --- |
| `src/config/variants/full.ts` | Add `forecastPanel` to panels, `forecastRisk` to map layers |
| `src/app/data-loader.ts` | Add forecast data loading (lazy: only when panel enabled or country brief opened) |
| `src/app/refresh-scheduler.ts` | Add forecast refresh (hourly — forecasts don't change as fast as news) |

### What to Build: Geopol Side

#### 1. FastAPI REST Server (`src/api/`)

```
src/api/
  __init__.py
  app.py              # FastAPI application factory
  routes/
    forecasts.py      # Forecast generation + retrieval
    countries.py      # Country-level forecast summaries
    health.py         # Health check
  middleware/
    auth.py           # API key validation
    cors.py           # CORS for World Monitor origins
    cache.py          # Redis response caching
  dto/
    forecast.py       # ForecastResponse DTO (simplified from ForecastOutput)
    scenario.py       # ScenarioDTO (serializable scenario tree)
    country.py        # CountryForecastSummary DTO
```

#### 2. Key Endpoints

```
GET  /api/v1/forecasts/country/{iso_code}
     → Returns active forecasts for a country
     → Query params: ?horizon=30d&limit=5

POST /api/v1/forecasts
     → Generate a new forecast on demand
     → Body: { question, context[], horizon_days }

GET  /api/v1/forecasts/{forecast_id}
     → Full forecast detail (scenarios, evidence, calibration)

GET  /api/v1/forecasts/{forecast_id}/scenarios
     → Scenario tree with branching structure

GET  /api/v1/forecasts/top
     → Top N most consequential active forecasts globally
     → Query params: ?limit=10&sort=probability|confidence|recency

GET  /api/v1/countries/{iso_code}/risk-summary
     → Aggregate forecast risk for a country
     → Returns: { risk_score, top_scenarios, trend, last_updated }

GET  /api/v1/countries/risk-map
     → All countries with forecast risk scores (for choropleth layer)
     → Returns: [{ iso_code, risk_score, forecast_count, top_question }]

GET  /api/v1/health
     → System health, graph freshness, model status
```

#### 3. Data Transfer Objects

```python
class ForecastResponse(BaseModel):
    forecast_id: str
    question: str
    prediction: str                    # Natural language summary
    probability: float                 # Calibrated P(yes)
    confidence: float                  # Model confidence in this prediction
    horizon_days: int
    scenarios: list[ScenarioDTO]
    reasoning_summary: str
    evidence_count: int
    ensemble_info: EnsembleInfoDTO
    calibration: CalibrationDTO
    created_at: datetime
    expires_at: datetime               # When this forecast becomes stale

class ScenarioDTO(BaseModel):
    scenario_id: str
    description: str
    probability: float
    answers_affirmative: bool          # Does this scenario answer "yes" to the question?
    entities: list[str]                # Actor names involved
    timeline: list[str]               # Sequence of expected events
    evidence_sources: list[EvidenceDTO]
    child_scenarios: list[ScenarioDTO]  # Branching (recursive)

class EvidenceDTO(BaseModel):
    source: str                        # "GDELT", "TKG pattern", "RAG match"
    description: str
    confidence: float
    timestamp: Optional[datetime]
    gdelt_event_id: Optional[str]      # Cross-reference to GDELT event

class CalibrationDTO(BaseModel):
    category: str                      # "conflict", "diplomatic", "economic"
    temperature: float                 # Applied temperature scaling
    historical_accuracy: float         # How accurate the model is at this confidence
    brier_score: Optional[float]       # If validation data available
    sample_size: int                   # How many past predictions calibrated against

class CountryRiskSummary(BaseModel):
    iso_code: str
    risk_score: float                  # Aggregate forecast risk (0.0-1.0)
    forecast_count: int
    top_question: str                  # Most consequential active forecast
    top_probability: float
    trend: Literal["rising", "stable", "falling"]
    last_updated: datetime

class EnsembleInfoDTO(BaseModel):
    llm_probability: float
    tkg_probability: Optional[float]   # None if TKG disabled
    weights: dict[str, float]          # {"llm": 0.6, "tkg": 0.4}
    temperature_applied: float
```

#### 4. Forecast Cache Layer

Forecasts are expensive to generate (Gemini API calls, graph traversal). Need a caching strategy:

```python
# Cache hierarchy:
# 1. In-memory LRU (hot forecasts, 100 entries, 10-minute TTL)
# 2. Redis (warm cache, 1-hour TTL for country summaries, 6-hour for full forecasts)
# 3. SQLite/PostgreSQL (cold storage, all historical forecasts)

# Auto-generation rules:
# - CII spike → auto-generate top 3 forecasts for that country
# - Scheduled daily batch → refresh forecasts for top 20 CII countries
# - On-demand → user requests specific question
```

#### 5. Auto-Generation Engine

A background service that automatically generates forecasts based on World Monitor signals:

```python
class AutoForecastEngine:
    """Generates forecasts automatically based on CII signals."""

    async def on_cii_spike(self, country: str, cii_score: float, delta: float):
        """Called when World Monitor detects a CII spike."""
        questions = self.generate_questions(country, cii_score)
        # e.g., "Will {country} experience armed conflict escalation within 30 days?"
        #        "Will international sanctions be imposed on {country} within 60 days?"
        #        "Will {country} experience a leadership change within 90 days?"
        for q in questions:
            forecast = await self.engine.forecast(question=q)
            await self.store.save(forecast)

    async def daily_refresh(self):
        """Refresh forecasts for top CII countries."""
        top_countries = await self.get_top_cii_countries(limit=20)
        for country in top_countries:
            await self.on_cii_spike(country.iso, country.cii_score, delta=0)
```

### Data Flow Architecture

```
           WORLD MONITOR (TypeScript / Vercel)
┌────────────────────────────────────────────────────────────┐
│                                                            │
│  ┌──────────┐  ┌─────────────┐  ┌────────────────────┐   │
│  │ 3D Globe │  │ CII Panel   │  │ Country Brief Page │   │
│  │ + Forecast│  │ + forecast  │  │ + Forecast tab     │   │
│  │ Risk Layer│  │   chips     │  │ + Scenario tree    │   │
│  └────┬─────┘  └──────┬──────┘  └─────────┬──────────┘   │
│       │               │                    │              │
│       └───────────┬───┴────────────────────┘              │
│                   │                                       │
│        ┌──────────▼──────────┐                            │
│        │ services/forecast/  │                            │
│        │  index.ts (REST)    │                            │
│        │  scenario-mapper.ts │                            │
│        └──────────┬──────────┘                            │
│                   │                                       │
│     ┌─────────────┼──────────────┐                        │
│     │ CII spike   │  User opens  │                        │
│     │ detected    │  country     │                        │
│     │  (auto)     │  brief (lazy)│                        │
│     └──────┬──────┴──────┬───────┘                        │
│            │             │                                │
└────────────┼─────────────┼────────────────────────────────┘
             │  HTTPS      │
             │             │
═════════════╪═════════════╪════════════  Network Boundary
             │             │
┌────────────┼─────────────┼────────────────────────────────┐
│            │  GEOPOL     │  (Python / Railway or Fly.io)  │
│            │             │                                │
│  ┌─────────▼─────────────▼────────┐                       │
│  │       FastAPI Server            │                       │
│  │  /forecasts/country/{iso}       │                       │
│  │  /forecasts/top                 │                       │
│  │  /countries/risk-map            │                       │
│  └──────────────┬─────────────────┘                       │
│                 │                                         │
│    ┌────────────┼────────────┐                            │
│    │            │            │                            │
│  ┌─▼──────┐ ┌──▼─────┐ ┌───▼──────┐                     │
│  │Forecast│ │  RAG   │ │  TKG     │                     │
│  │Engine  │ │Pipeline│ │Predictor │                     │
│  │(Gemini)│ │(Chroma)│ │(JAX)    │                     │
│  └───┬────┘ └───┬────┘ └───┬──────┘                     │
│      │          │          │                             │
│      └──────────┼──────────┘                             │
│                 │                                        │
│       ┌─────────▼─────────┐                              │
│       │ Temporal Knowledge │                              │
│       │ Graph (NetworkX +  │                              │
│       │ 30-day partitions) │                              │
│       └─────────┬─────────┘                              │
│                 │                                        │
│       ┌─────────▼─────────┐                              │
│       │ GDELT Ingestion   │                              │
│       │ (500K events/day) │                              │
│       └───────────────────┘                              │
│                                                          │
│  ┌──────────────────────────┐                            │
│  │ Auto-Forecast Engine     │                            │
│  │ • CII webhook listener  │                            │
│  │ • Daily batch refresh   │                            │
│  │ • Question generation   │                            │
│  └──────────────────────────┘                            │
│                                                          │
│  ┌──────────────────────────┐                            │
│  │ PostgreSQL / SQLite      │                            │
│  │ • forecast_results       │                            │
│  │ • calibration_history    │                            │
│  │ • prediction_outcomes    │                            │
│  └──────────────────────────┘                            │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

### Forecast Output Mapping

How Geopol's `ForecastOutput` maps to World Monitor display elements:

| Geopol Field | WM Display | Location |
| --- | --- | --- |
| `question` | Forecast card title | ForecastPanel, CountryBrief |
| `probability` | Large percentage with color coding | ForecastCard (green <0.3, yellow 0.3-0.7, red >0.7) |
| `confidence` | Calibration badge ("72% historically accurate") | CalibrationBadge |
| `prediction` | Natural language summary text | ForecastCard body |
| `scenarios[]` | Interactive tree nodes | ScenarioExplorer |
| `scenario.probability` | Node size + opacity | ScenarioExplorer |
| `scenario.answers_affirmative` | Color coding (green=yes, red=no) | ScenarioExplorer |
| `scenario.entities[]` | Clickable chips linking to WM entity data | ScenarioExplorer |
| `scenario.evidence_sources[]` | Evidence sidebar with links | ScenarioExplorer |
| `ensemble_info.llm_probability` | "LLM: 70%" in component breakdown | ForecastCard detail view |
| `ensemble_info.tkg_probability` | "TKG: 60%" in component breakdown | ForecastCard detail view |
| `ensemble_info.weights` | Weight distribution donut chart | ForecastCard detail view |
| `calibration.historical_accuracy` | "Model is X% accurate at this confidence" | CalibrationBadge tooltip |
| `calibration.brier_score` | Brier decomposition (advanced view) | ForecastCard detail view |

### Panel Definitions

#### Forecast Panel (New Dashboard Panel)

```
┌─────────────────────────────────────────────────────────┐
│  Geopolitical Forecasts                    [Refresh ↻]  │
│─────────────────────────────────────────────────────────│
│                                                         │
│  🔴 67%  Will Syria conflict escalate to foreign        │
│          intervention within 60 days?                   │
│          Confidence: 72% · 3 scenarios · Updated 2h ago │
│                                                         │
│  🟡 43%  Will EU impose new sanctions on Russia         │
│          within 30 days?                                │
│          Confidence: 68% · 4 scenarios · Updated 4h ago │
│                                                         │
│  🟢 18%  Will Taiwan Strait see military                │
│          confrontation within 90 days?                  │
│          Confidence: 81% · 3 scenarios · Updated 1h ago │
│                                                         │
│  🟡 51%  Will Iran nuclear deal see diplomatic          │
│          breakthrough within 60 days?                   │
│          Confidence: 55% · 5 scenarios · Updated 6h ago │
│                                                         │
│  [Show more forecasts...]                               │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

#### CII Panel Enhancement

```
┌─────────────────────────────────────────────────────────┐
│  Country Instability Index                              │
│─────────────────────────────────────────────────────────│
│                                                         │
│  Syria       ████████░░ 8.2  ↑  [67% escalation →30d]  │
│  Ukraine     ███████░░░ 7.1  →  [43% new offensive →60d]│
│  Myanmar     ██████░░░░ 6.3  ↑  [38% regime change →90d]│
│  Sudan       █████░░░░░ 5.8  ↑  [51% famine crisis →30d]│
│  Venezuela   ████░░░░░░ 4.2  →  —                       │
│                                                         │
│  [forecast chips] = Geopol top forecast for that country │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

#### Country Brief Forecast Tab

```
┌─────────────────────────────────────────────────────────┐
│  Syria · Country Brief                                  │
│  [Overview] [News] [Forecast] [Timeline] [Signals]      │
│─────────────────────────────────────────────────────────│
│                                                         │
│  ┌─ Active Forecasts ──────────────────────────────┐    │
│  │                                                  │    │
│  │  Q: Will conflict escalate to foreign            │    │
│  │     intervention within 60 days?                 │    │
│  │                                                  │    │
│  │  Probability: ██████████████████░░░░░ 67%        │    │
│  │  Calibration: 72% accurate · Brier: 0.21        │    │
│  │  Ensemble: LLM 70% (α=0.6) + TKG 62% (α=0.4)   │    │
│  │                                                  │    │
│  │  Scenarios:                                      │    │
│  │  ├─ [45%] Turkish ground operation via north     │    │
│  │  │   └─ Evidence: 3 GDELT events, 2 TKG matches │    │
│  │  ├─ [22%] Israeli strikes expand to Damascus     │    │
│  │  │   └─ Evidence: 5 GDELT events, 1 TKG match   │    │
│  │  └─ [33%] Status quo — no direct intervention    │    │
│  │      └─ Evidence: 4 GDELT events, 3 TKG matches │    │
│  │                                                  │    │
│  │  Historical Precedent:                           │    │
│  │  "Similar CII/event patterns in 2018 led to     │    │
│  │   Turkish Operation Olive Branch (73% match)"    │    │
│  │                                                  │    │
│  │  [Explore Full Scenario Tree →]                  │    │
│  └──────────────────────────────────────────────────┘    │
│                                                         │
│  ┌─ Additional Forecasts ──────────────────────────┐    │
│  │  43%  Diplomatic settlement within 90 days       │    │
│  │  28%  Refugee crisis escalation within 30 days   │    │
│  └──────────────────────────────────────────────────┘    │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### Map Layer Integration

#### Forecast Risk Choropleth

A new deck.gl `GeoJsonLayer` coloring countries by aggregate forecast risk:

```typescript
// Forecast risk distinct from CII:
// CII = present instability (reactive)
// Forecast risk = future probability of destabilizing events (predictive)

const forecastRiskLayer = new GeoJsonLayer({
  id: 'forecast-risk',
  data: countryGeoJson,
  getFillColor: (d) => {
    const risk = forecastRiskMap.get(d.properties.iso_a2);
    if (!risk) return [128, 128, 128, 20];  // gray: no forecast
    // Green (low) → Yellow (medium) → Red (high)
    return interpolateRisk(risk.risk_score);
  },
  getLineColor: [80, 80, 80, 100],
  getLineWidth: 1,
  pickable: true,
  onClick: (info) => openCountryBrief(info.object.properties.iso_a2),
  opacity: 0.4,  // Semi-transparent to overlay on base map
});
```

**Toggle behavior**: Available in `full` and `analyst` variants. Off by default (analyst: on by default). Toggled independently from CII — they show different things.

### Country Brief Enhancement

The Country Brief Page (`CountryBriefPage.ts`) gains a new tab. Loading strategy:

```typescript
// Lazy loading: only fetch forecasts when user opens the Forecast tab
async function loadCountryForecasts(isoCode: string): Promise<ForecastResponse[]> {
  const cached = forecastCache.get(isoCode);
  if (cached && Date.now() - cached.fetchedAt < 3600_000) return cached.data;

  const response = await fetch(
    `${GEOPOL_API_URL}/api/v1/forecasts/country/${isoCode}?horizon=90d&limit=5`
  );
  const forecasts = await response.json();
  forecastCache.set(isoCode, { data: forecasts, fetchedAt: Date.now() });
  return forecasts;
}
```

### CII ↔ Geopol Feedback Loop

The most powerful integration pattern is a bidirectional feedback loop:

```
World Monitor CII computation
    │
    │ CII spike detected (reactive signal)
    │
    ▼
Geopol auto-generates forecasts
    │
    │ P(escalation) = 67% (predictive signal)
    │
    ▼
World Monitor displays forecast alongside CII
    │
    │ Analyst sees: "CII 8.2 + 67% escalation risk"
    │
    ▼
Time passes → event resolves
    │
    │ Outcome recorded
    │
    ▼
Geopol Brier score updated
    │
    │ Model accuracy improves over time
    │
    ▼
Better forecasts next cycle
```

**Implementation**: World Monitor's CII computation endpoint (`/api/intelligence/v1/get-risk-scores`) could include a webhook/callback mechanism. When CII delta exceeds a threshold, it notifies Geopol's auto-forecast engine.

Alternatively, Geopol's auto-forecast engine polls World Monitor's CII endpoint on a schedule (every 30 minutes) and triggers forecasts for countries with significant CII changes.

### Deployment Architecture

```
                    ┌──────────────────────────────┐
                    │         DNS / CDN            │
                    │   worldmonitor.app            │
                    │   api.geopol.worldmonitor.app │
                    └───────┬──────────┬───────────┘
                            │          │
              ┌─────────────┘          └─────────────┐
              │                                      │
    ┌─────────▼──────────┐              ┌────────────▼──────────┐
    │    Vercel           │              │  Railway / Fly.io     │
    │                     │              │                       │
    │  World Monitor      │              │  Geopol API Server    │
    │  (Vite + Edge Fns)  │   HTTPS →    │  (FastAPI + Uvicorn)  │
    │                     │              │                       │
    │  Serves:            │              │  Serves:              │
    │  • Dashboard        │              │  • /api/v1/forecasts  │
    │  • Existing APIs    │              │  • /api/v1/countries  │
    │  • CII computation  │              │  • Auto-forecast eng. │
    └─────────┬──────────┘              └────────────┬──────────┘
              │                                      │
              │    ┌──────────────────┐               │
              └───▶│  Upstash Redis   │◀──────────────┘
                   │  (shared cache)  │
                   └──────────────────┘
                            │
              ┌─────────────┘
              │
    ┌─────────▼──────────────────┐
    │  Geopol Data Layer          │
    │                             │
    │  PostgreSQL (forecasts,     │
    │    calibration history,     │
    │    prediction outcomes)     │
    │                             │
    │  SQLite (GDELT events,     │
    │    partition index)         │
    │                             │
    │  ChromaDB (RAG vectors)    │
    │                             │
    │  GraphML files (TKG        │
    │    temporal partitions)    │
    └─────────────────────────────┘
```

**Key decisions**:

- **Railway/Fly.io for Geopol**: Long-running Python process with JAX, ChromaDB, large in-memory graphs. Not suitable for Vercel serverless.
- **Shared Redis**: World Monitor already uses Upstash Redis. Geopol uses it for forecast response caching. Same instance, different key prefixes (`geopol:forecast:*`).
- **GDELT ingestion**: Runs as a scheduled task (cron) on Railway, not triggered by user requests. Graph freshness is background concern.
- **Auto-forecast**: Runs as a background worker alongside the API server. Polls WM CII or receives webhooks.

**Environment variables** (Geopol server):

```bash
GEMINI_API_KEY=...                     # For LLM reasoning
WORLDMONITOR_API_URL=https://worldmonitor.app  # For CII polling
WORLDMONITOR_API_KEY=wm_...            # Auth for WM API
UPSTASH_REDIS_REST_URL=...             # Shared cache
UPSTASH_REDIS_REST_TOKEN=...
DATABASE_URL=postgresql://...          # Forecast persistence
GEOPOL_API_KEYS=wm_geopol_...         # Validates incoming WM requests
FORECAST_DAILY_BUDGET=50               # Max auto-forecasts per day
```

**Environment variables** (World Monitor):

```bash
VITE_GEOPOL_API_URL=https://api.geopol.worldmonitor.app
GEOPOL_API_KEY=wm_geopol_...          # For server-side forecast fetching
```

### Cost and Complexity Estimates

#### Development Effort

| Component | Estimated Effort | Lines of Code |
| --- | --- | --- |
| **Geopol: FastAPI server + routes** | 3-4 days | ~700 LOC |
| **Geopol: DTOs + serialization** | 1-2 days | ~400 LOC |
| **Geopol: Forecast caching (Redis)** | 1 day | ~200 LOC |
| **Geopol: Auto-forecast engine** | 2-3 days | ~400 LOC |
| **Geopol: Database persistence** | 2 days | ~300 LOC |
| **Geopol: Auth middleware** | 0.5 days | ~100 LOC |
| **Geopol: Docker + deployment** | 1 day | ~100 LOC |
| **WM: ForecastPanel component** | 2 days | ~400 LOC |
| **WM: ScenarioExplorer modal** | 3-4 days | ~700 LOC |
| **WM: ForecastCard + CalibrationBadge** | 1 day | ~250 LOC |
| **WM: Forecast service client** | 1 day | ~200 LOC |
| **WM: ForecastRiskLayer (deck.gl)** | 1-2 days | ~250 LOC |
| **WM: CountryBrief Forecast tab** | 2 days | ~400 LOC |
| **WM: CII Panel forecast chips** | 0.5 days | ~100 LOC |
| **WM: Config (variant, panels, layers)** | 0.5 days | ~150 LOC |
| **Integration testing** | 3-4 days | — |
| **TOTAL** | **~4-5 weeks** | **~4,650 LOC** |

#### Ongoing Costs (Production)

| Service | Cost | Notes |
| --- | --- | --- |
| Railway/Fly (Python + JAX) | $10-30/mo | Geopol needs more memory than a typical web app (graph data) |
| Gemini API | $20-100/mo | Each forecast = 4-8 API calls. 50 forecasts/day × $0.02-0.05 |
| PostgreSQL (Railway) | $5/mo | Forecast storage |
| Redis (shared, Upstash) | $0-10/mo | Already used by WM |
| ChromaDB storage | Included | Runs in-process on Railway |
| **Total** | **$35-145/mo** | Scales with forecast frequency |

### Critical Success Factors

1. **Forecast freshness vs. cost** — Each forecast costs 4-8 Gemini API calls ($0.02-0.05). Can't regenerate all forecasts every hour. Need smart cache invalidation: regenerate only when CII significantly changes or after a major event. Daily budget cap (50/day) prevents runaway costs.

2. **Latency management** — Generating a forecast takes 30-120 seconds (multiple Gemini calls + graph traversal + RAG retrieval). The Country Brief must not block on forecast generation. Strategy: show cached forecasts instantly, show "generating fresh forecast..." spinner if cache is stale, update when ready.

3. **Calibration trust** — Geopol's calibration quality depends on prediction outcome data. Early on, the system won't have enough resolved predictions to show meaningful Brier scores. Don't show calibration badges until at least 50 resolved predictions per category. Show "calibrating..." instead.

4. **Question quality** — Auto-generated questions from CII spikes must be specific, falsifiable, and time-bounded. Bad: "Will things get worse in Syria?" Good: "Will Syria experience direct foreign military intervention within 60 days?" Invest in the question generation templates.

5. **Scenario tree UX** — The ScenarioExplorer is the most complex new component. It must be intuitive enough for a non-technical analyst. Tree visualization with click-to-expand, evidence on hover, probability as node size. Test with real analysts.

6. **CII ↔ Forecast confusion** — Users must understand the difference: CII = "how unstable is this country right now" (reactive), Forecast = "what will happen next" (predictive). Use distinct visual language (CII: ring chart, Forecast: probability bar). Different colors, different sections.

### Phased Implementation Roadmap

#### Phase 1: Geopol API (Week 1-2)

**Goal**: Geopol serves forecasts over HTTP.

- Build FastAPI server with forecast CRUD endpoints
- Implement forecast caching (Redis)
- Create DTOs for JSON serialization
- API key authentication
- Deploy to Railway
- Integration test: can request forecast via HTTP and get structured response

#### Phase 2: Country Forecasts (Week 2-3)

**Goal**: World Monitor Country Briefs show Geopol forecasts.

- Build `services/forecast/index.ts` REST client
- Build ForecastCard and CalibrationBadge components
- Add Forecast tab to CountryBriefPage
- Lazy loading: fetch forecasts only when tab opened
- Cached forecasts render instantly, stale ones show "refreshing" indicator

#### Phase 3: Forecast Panel (Week 3-4)

**Goal**: Dashboard shows top global forecasts.

- Build ForecastPanel component
- Add to full variant config
- Wire up data loading + refresh scheduling (hourly)
- Build ScenarioExplorer modal (basic version: list view, not tree)

#### Phase 4: Map Layer (Week 4-5)

**Goal**: Globe shows forecast risk choropleth.

- Build ForecastRiskLayer (GeoJsonLayer with risk coloring)
- Implement `/api/v1/countries/risk-map` endpoint in Geopol
- Add layer toggle to full variant
- Country click → opens Country Brief with Forecast tab

#### Phase 5: Auto-Triggers (Week 5-6)

**Goal**: CII spikes auto-generate forecasts.

- Build auto-forecast engine in Geopol
- CII polling mechanism (poll WM endpoint every 30 min)
- Question generation templates per country/event type
- Daily budget cap (50 forecasts/day)
- CII Panel shows forecast chips alongside scores

#### Phase 6: ScenarioExplorer + Polish (Week 6-7)

**Goal**: Full interactive scenario tree.

- ScenarioExplorer upgraded to tree visualization (D3 or custom SVG)
- Evidence sidebar with GDELT event links
- Historical precedent panel (RAG matches)
- Ensemble breakdown visualization
- Mobile-responsive panels

---

## Three-Project Synergy: World Monitor + Geopol + OSINT Double

When all three projects are considered together, they form a complete automated intelligence capability:

```
┌──────────────────────────────────────────────────────────────┐
│                                                              │
│    WORLD MONITOR          GEOPOL            OSINT DOUBLE     │
│    ══════════════         ══════            ═══════════════  │
│                                                              │
│    "What is               "What will        "Is this         │
│     happening?"            happen next?"     verified?"      │
│                                                              │
│    ┌──────────┐          ┌──────────┐      ┌──────────┐     │
│    │ 50+ APIs │          │ GDELT TKG│      │ Multi-   │     │
│    │ 298 RSS  │──CII───▶│ Gemini   │      │ Agent    │     │
│    │ 36 layers│  spike   │ Ensemble │      │ Crawlers │     │
│    └────┬─────┘          └────┬─────┘      └────┬─────┘     │
│         │                     │                  │           │
│    ┌────▼─────┐          ┌────▼─────┐      ┌────▼─────┐     │
│    │Dashboard │          │Forecasts │      │Verified  │     │
│    │ + Globe  │◀─────────│Scenarios │      │Facts     │     │
│    │ + Panels │ enriches │Evidence  │      │Provenance│     │
│    └────┬─────┘          └──────────┘      └────┬─────┘     │
│         │                                       │           │
│         │              ┌────────────┐            │           │
│         └─────────────▶│  Analyst   │◀───────────┘           │
│           hotspot      │  triggers  │  verified              │
│           detected     │investigation│  facts                │
│                        └────────────┘                        │
│                                                              │
│    OBSERVE              PREDICT             VERIFY           │
│    (present)            (future)            (past+present)   │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

**The complete intelligence cycle**:

1. **World Monitor** detects a CII spike in Syria (present observation)
2. **Geopol** auto-generates forecasts: "67% chance of escalation within 60 days" (future prediction)
3. **OSINT Double** investigates the underlying claims, verifying facts with source attribution (verification)
4. **World Monitor** displays all three layers: live events + probabilistic forecasts + verified facts
5. **Time passes** → outcome resolves → Geopol's Brier score improves → better forecasts next cycle

Each project handles what it does best. No project tries to do what the others do. The integration is through data flow, not code merger.

**Shared infrastructure**:
- **Upstash Redis**: All three systems cache through the same Redis instance
- **GDELT**: World Monitor and Geopol both consume GDELT (World Monitor for events, Geopol for knowledge graphs)
- **Entity ontology**: Shared entity catalog across all three
- **API keys**: Mutual authentication between all services

**The analyst experience**: One dashboard (World Monitor) that shows what's happening now (events), what will happen next (forecasts), and whether the underlying claims are verified (facts). Three engines, one interface.
