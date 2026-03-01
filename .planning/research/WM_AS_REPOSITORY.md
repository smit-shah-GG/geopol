# World Monitor as Repository: Analysis & Restructured v2.0 Plan

> Verbatim record of the architectural analysis produced 2026-02-27 during v2.0 planning. This document captures the "WM as repository" strategy — using World Monitor as a reference architecture and code quarry for Geopol's own frontend, not as an integration target.

---

## Context

The user clarified their intent: "we salvage as much of WM as possible, and it looks and feels like a fork on the surface, but isn't. I want to use WM as a repository in the true sense of the word: as direction and an expansion plan, and as a source for inspiration and potential features."

This is fundamentally different from the integration document's (`WORLDMONITOR_INTEGRATION.md`) premise of two separate running systems talking over HTTP. The user wants one system that absorbs WM's best ideas into Geopol's own frontend.

---

## What WM Actually Is (Architecturally)

The exploration reveals WM is **not** a React/Vue app. It's pure vanilla TypeScript with custom DOM manipulation via an `h()` hyperscript helper. No framework lock-in. The key salvageable scaffolding:

| Layer | What WM Has | Salvageable? |
|---|---|---|
| **Build system** | Vite 6 + TypeScript 5.7 + PWA + Brotli precompression | Yes — wholesale |
| **Variant system** | `VITE_VARIANT` build-time feature flags → 4 builds from 1 codebase | Yes — repurpose for forecast perspectives |
| **Panel architecture** | `Panel` base class with `refresh()` lifecycle, resizable/reorderable grid | Yes — the abstraction is domain-agnostic |
| **Globe rendering** | deck.gl 9.2 + maplibre-gl 5.16, 36+ toggleable layers, clustering | Yes — the rendering stack is independent of data |
| **State management** | `AppContext` singleton, `DataLoaderManager`, `RefreshScheduler` | Yes — patterns carry, types don't |
| **Design system** | CSS variables, dark/light themes, semantic severity colors, monospace typography | Yes — the palette and structure work for forecasting |
| **Tauri desktop** | Rust sidecar, keychain secrets, persistent cache, auto-update | Yes — wraps any web app |
| **Country Brief pages** | Full-screen modal with tabs, AI analysis, signals, export | **Partially** — structure reusable, content all replaced |
| **45 panels** | CII, LiveNews, Markets, Conflicts, Cascade, etc. | **No** — every panel is WM-specific content |
| **50+ API integrations** | ACLED, GDELT proxy, FRED, AIS, flights, etc. | **No** — data sources completely different |
| **CII computation** | Weighted multi-signal scoring with trend detection | **Pattern yes, code no** — same concept, different inputs |

The honest salvage ratio: **~35-40% by code volume, ~70% by architectural value.** You're keeping the skeleton and the skin. You're replacing every organ.

## What "WM as Repository" Means for v2.0

The v2.0 roadmap restructures around this thesis: **Geopol is a headless Python forecast engine. Its frontend is a WM-derived TypeScript dashboard.**

This makes the project polyglot. Python backend (FastAPI), TypeScript frontend (Vite + deck.gl). The FastAPI server is no longer optional or "nice for WM integration" — it's the **mandatory bridge** between the engine and the UI.

### Restructured Phase 9: API Foundation & Infrastructure

Everything from current Phase 9, plus FastAPI becomes load-bearing:

**Kept intact:**
- Structured logging (kill `print()`)
- TKGModelProtocol + jraph elimination
- Process isolation (API server + ingest daemon + training job)

**Changed:**
- SQLite stays for GDELT events and partition index, but **PostgreSQL** for forecast persistence (the API server needs a real database, not file-locked SQLite)
- Database schema designed for API consumption from day one: `forecasts`, `scenarios`, `calibration_history`, `prediction_outcomes`

**Added:**
- FastAPI server skeleton (`src/api/`) with health endpoint
- DTOs: `ForecastResponse`, `ScenarioDTO`, `CalibrationDTO`, `CountryRiskSummary`
- API key auth middleware
- Redis cache layer (forecast response caching)
- CORS configuration for the TypeScript frontend

**Success criteria**: `GET /api/v1/health` returns subsystem status. `POST /api/v1/forecasts` generates a forecast and returns a valid `ForecastResponse` JSON. Three concurrent Python processes (API, ingest, training) don't corrupt the database.

### Phase 10: Ingest & Forecast Pipeline

Essentially unchanged from current Phase 10, but the output is **API-queryable**, not just database rows:

- 15-min micro-batch GDELT ingest
- Daily automated forecast generation with question templates
- Prediction-outcome resolution
- Endpoints: `GET /forecasts/country/{iso}`, `GET /forecasts/top`, `GET /countries/risk-map`
- Cache warming: daily batch pre-populates Redis for top-20 countries

After Phase 10, the API is live and serving real forecast data. The frontend can develop against it.

### Phase 11: TKG Predictor Replacement

**Unchanged.** TiRGN/HisMatch JAX port, swappable via TKGModelProtocol. Parallelizable after Phase 9. The only difference is motivation: forecasts visible to dashboard users need to be accurate, not just "better than RE-GCN on a benchmark."

### Phase 12: Frontend (WM-Derived Dashboard)

This is the phase that changes completely. Streamlit is gone. In its place:

**Step 1 — Scaffold (the fork that isn't a fork):**
- Copy WM's build system: `vite.config.ts`, `tsconfig.json`, PWA config
- Copy WM's `Panel` base class, `AppContext` pattern, `DataLoaderManager` skeleton, `RefreshScheduler`
- Copy WM's CSS variable system and theme manager (dark/light)
- Copy WM's `DeckGLMap` component and maplibre-gl setup
- Copy WM's `h()` DOM helper and utility functions
- Strip **all** WM-specific panels, services, data sources, types
- Result: empty dashboard with a globe, a panel grid, and a theme toggle. No content.

**Step 2 — Forecast panels (replace WM's 45 panels with ~8):**

| New Panel | Derived From | Purpose |
|---|---|---|
| `ForecastPanel` | — (new) | Top N active forecasts globally, sorted by probability/confidence/recency |
| `ScenarioExplorer` | — (new) | Interactive tree visualization of scenario branches with evidence |
| `CalibrationPanel` | — (new) | Reliability diagram, Brier decomposition, per-CAMEO accuracy |
| `RiskIndexPanel` | `CIIPanel` | Aggregate forecast risk per country (CII's scoring pattern, Geopol's data) |
| `EventTimelinePanel` | `LiveNewsPanel` | Recent GDELT events feeding the knowledge graph |
| `EnsembleBreakdownPanel` | — (new) | LLM vs TKG component weights, per-category alpha visualization |
| `PredictionTrackPanel` | `MarketPanel` pattern | Historical prediction accuracy over time (track record) |
| `SystemHealthPanel` | — (new) | Ingest freshness, graph size, API budget, model status |

**Step 3 — Map layers (replace WM's 36 with ~5):**

| New Layer | Derived From | Rendering |
|---|---|---|
| `ForecastRiskChoropleth` | WM's `GeoJsonLayer` patterns | Countries colored by aggregate forecast risk |
| `ActiveForecastMarkers` | WM's `ScatterplotLayer` patterns | Points where forecasts are active, sized by probability |
| `KnowledgeGraphArcs` | WM's `ArcLayer` patterns | Actor-to-actor relations colored by type |
| `GDELTEventHeatmap` | WM's `HeatmapLayer` | Density of recent GDELT events |
| `ScenarioZones` | — (new) | Highlighted regions relevant to active scenario branches |

**Step 4 — Country Brief reimagined:**

WM's `CountryBriefPage` structure (full-screen modal, tabs, export) is directly reusable. The tabs change:

| WM Tab | Geopol Tab |
|---|---|
| AI Analysis (text blob) | **Active Forecasts** (probability bars, scenario trees, evidence chains) |
| Top News | **GDELT Events** (recent events feeding the knowledge graph) |
| 7-day Timeline | **Forecast History** (how predictions for this country evolved over time) |
| Signal Chips | **Risk Signals** (CAMEO event categories active for this country) |
| Infrastructure Exposure | **Entity Relations** (knowledge graph subgraph for this country's actors) |
| Prediction Markets | **Calibration** (per-category accuracy for this country's forecasts) |

**Step 5 — Forecast service layer:**

WM's `DataLoaderManager` pattern (circuit breaker, freshness tracking, `inFlight` dedup) maps directly. The service calls change from 50+ external APIs to one: Geopol's FastAPI server.

```
src/services/forecast/
  client.ts          # HTTP client for Geopol API (fetch + cache + retry)
  types.ts           # ForecastResponse, ScenarioDTO, etc. (mirrors Python DTOs)
  mapper.ts          # API response → panel display types
```

### Phase 13: Monitoring, Hardening & Desktop

Current Phase 13 monitoring work, plus:
- `SystemHealthPanel` displays `/api/v1/health` data
- Tauri desktop build (copy WM's `src-tauri/` scaffold, point at Geopol frontend)
- API key management via Tauri keychain (WM's pattern, reused exactly)
- PWA offline support (cached forecasts viewable without network)

---

## What WM's Feature Set Becomes (The "Expansion Plan")

This is where "WM as repository" pays off long-term. WM has features that Geopol doesn't need today but could adopt later:

| WM Feature | Geopol Future Use | When |
|---|---|---|
| **Variant system** (4 builds from 1 codebase) | Regional forecast variants: MENA, Europe, Asia-Pacific, Africa | v3.0+ |
| **RSS feed aggregation** (298 domains) | Open-source intelligence feed alongside forecasts | v3.0+ |
| **Prediction markets** (Polymarket data) | Side-by-side: Geopol forecast vs. market crowd prediction | v2.1 |
| **Virtual scrolling** (large news lists) | Large forecast history lists | As needed |
| **i18n / RTL** | Multi-language forecast display | v3.0+ |
| **Command palette** (⌘K search) | Search forecasts, countries, entities | Phase 12 or later |
| **URL state binding** (shareable links) | `?country=SY&forecast=abc123` deep links | Phase 12 |
| **Export** (JSON, CSV, PNG) | Export forecast reports | Phase 12 |
| **Playback mode** (time travel) | Replay how forecasts evolved over time | v3.0+ |

WM becomes a **feature backlog you can cherry-pick from**, with working reference implementations for each.

---

## Key Insight: Why This Works

The "fork that isn't a fork" strategy is architecturally sound because WM's **zero-framework approach** makes it possible. If WM were React, you'd inherit React's entire dependency tree, upgrade cadence, and opinions. Because it's vanilla TypeScript + CSS variables + deck.gl, you can take the patterns without the baggage. The `Panel` base class, the `AppContext` singleton, the `h()` helper — these are ~500 lines of framework-free scaffolding that carry no opinions about your domain. That's the real value of WM as a quarry: the rocks are clean.

The risk is the polyglot tax. You're now maintaining Python (engine) + TypeScript (frontend) + Rust (Tauri). For a single-developer project, that's three ecosystems to keep current. The mitigation: the TypeScript frontend is architecturally simple (no React state management hell, no build tool churn), and the Tauri shell is thin.

## Parallel Execution Model

The DTOs from the integration doc (ForecastResponse, ScenarioDTO, CalibrationDTO, CountryRiskSummary, EnsembleInfoDTO, EvidenceDTO) are already spec'd out in detail — lines 328-382 of WORLDMONITOR_INTEGRATION.md. These become the contract that both sides develop against.

```
Phase 9 (API skeleton + DTOs + DB)
    │
    ├──► Phase 10 (real data: ingest + forecast generation)
    │
    ├──► Phase 11 (TKG replacement) ──── parallelizable
    │
    └──► Phase 12 (frontend: mock API → real API when Phase 10 lands)
                │
                └──────────────────────────────► Phase 13 (monitoring + hardening)
```

Three branches running concurrently after Phase 9. Phase 13 waits for all three because it monitors the complete system.

The mock API strategy: Phase 9's FastAPI routes return **realistic static fixtures** — hardcoded `ForecastResponse` objects with plausible Syria/Ukraine/Myanmar scenarios, fake but structurally valid scenario trees, synthetic calibration data. The frontend builds and tests against those. When Phase 10 replaces stubs with live Gemini+TKG output, the frontend just works — the JSON shape hasn't changed.

This is contract-first API design. The DTO definitions become the load-bearing specification — Python `BaseModel` classes on the backend, TypeScript `interface` declarations on the frontend, both generated from the same source of truth. If Phase 9 gets the DTOs right, Phases 10 and 12 can't diverge.

### Timeline Impact

Instead of sequential (9 → 10 → 11 → 12 → 13), you get:

- **Phase 9**: ~2 weeks (the critical path — everything gates on this)
- **Phases 10, 11, 12**: parallel, ~3-4 weeks each (longest leg determines duration)
- **Phase 13**: ~1-2 weeks after the others converge

Worst case ~8 weeks total instead of ~14+ weeks sequential. The constraint shifts from "how fast can I build phases" to "how well did Phase 9 define the contract."
