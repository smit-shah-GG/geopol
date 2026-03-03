# Phase 17: Live Data Feeds & Country Depth - Context

**Gathered:** 2026-03-04
**Status:** Ready for planning

<domain>
## Phase Boundary

Wire real, live data into every existing panel and country screen. Backend exposes event and article API endpoints with full filter surfaces. EventTimelinePanel displays real GDELT events. Country screens are populated with meaningful content across all existing tabs. ACLED conflict data and government travel advisories are ingested as new data sources. SourcesPanel auto-discovers active sources from the backend.

This phase does NOT add new UI screens or new panel types. It fills existing scaffolding with real data and adds two new ingestion pipelines (ACLED, government advisories).

</domain>

<decisions>
## Implementation Decisions

### Event API Design
- Full filter surface: country, date range, CAMEO code, actor, Goldstein range, text search
- Separate endpoints: `GET /events` (GDELT + ACLED structured data) and `GET /articles` (RSS text content)
- Cursor-based pagination for both endpoints — handles real-time feed growth without offset drift
- Default time range: 30 days when no date filter provided
- Events query SQLite directly (no PostgreSQL mirror) — existing architecture preserved

### Article API Design
- Dual query mode: keyword+country filtering for listing, optional `?semantic=true` triggers ChromaDB vector similarity search
- Separate from events endpoint — different schema, different storage backend (ChromaDB vs SQLite)

### Event/Article Frontend Presentation
- Compact timeline rows: timestamp + headline + country flag + severity badge
- Click-to-expand shows curated summary: headline, actors, CAMEO description, Goldstein score, source URL, related article count
- NOT full record dump — human-readable curated view

### Country Screen Content
- Primary content beyond forecasts: event timeline + key actors (not risk dashboard)
- Entity depth: names + activity counts only (no interactive relationship graph)
- RSS articles shown in separate section from GDELT events on country pages (not blended)
- Keep all 6 existing CountryBriefPage tabs, populate them with real data from new endpoints

### Additional Data Sources — ACLED
- Armed conflict events only: battles, explosions, violence against civilians
- Skip protests, riots, strategic developments (GDELT covers those adequately)
- Unified event model: map ACLED fields to GDELT-compatible schema, single `/events` endpoint, `source` field distinguishes origin
- Daily polling frequency
- UI display only — does NOT enter knowledge graph or affect TKG training

### Additional Data Sources — Government Advisories
- Two feeds: US State Department and UK FCDO travel advisories
- **EU EEAS dropped:** Research found no structured API or machine-readable feed for EU EEAS travel advisories. EEAS publishes advice as unstructured web pages only. Two reliable sources (US State Dept JSON API, UK FCDO GOV.UK Content API) provide sufficient coverage. EU EEAS can be reconsidered if a structured feed becomes available in the future.
- Daily polling frequency
- Advisory risk levels displayed on country pages but NOT factored into computed risk score — risk score stays purely forecast-derived
- UI display only — no knowledge graph integration

### Panel Data Wiring
- All panels wired simultaneously — no partial/phased rollout
- Empty states: explanation text (e.g., "No events for Ukraine in the last 30 days") — panel stays visible, never hidden
- Per-panel refresh intervals matching source update rates: events ~30s, articles ~5min, advisories daily
- SourcesPanel: auto-discover via backend `/sources` endpoint — adding a new source auto-appears in health display without frontend changes

### Claude's Discretion
- Exact ACLED field-to-GDELT schema mapping
- ChromaDB semantic search parameters (similarity threshold, max results)
- Advisory API client implementation details (parsing, caching)
- Cursor encoding format for pagination
- Exact per-panel refresh intervals (directional: events fast, articles medium, advisories slow)
- Error/retry behavior for external API calls (ACLED, State Dept, FCDO)

</decisions>

<specifics>
## Specific Ideas

- Unified event model is critical: ACLED and GDELT events must be queryable through the same `/events` endpoint with a `source` filter. The frontend should not need to know which backend produced an event.
- Government advisories are a display-only signal — showing official risk assessments alongside AI-generated forecasts provides useful context without polluting the computed metrics.
- Compact timeline rows are the default density everywhere. Country pages get the same presentation as dashboard EventTimelinePanel — consistency over per-context variation.

</specifics>

<deferred>
## Deferred Ideas

- Full pipeline integration of ACLED/advisory data into knowledge graph and TKG training — evaluate after UI-only integration proves the data quality
- Entity relationship mini-graph on country pages — deferred; names + counts sufficient for now
- Risk dashboard with trend charts on country pages — analytics-centric view could be a future enhancement
- ACLED protest/riot/strategic development event types — add if armed conflict subset proves valuable
- Sanctions lists (OFAC SDN, EU restrictive measures) — entity-level data, different use case from country-level advisories
- EU EEAS travel advisories — no structured API currently available; revisit if machine-readable feed emerges

</deferred>

---

*Phase: 17-live-data-feeds-country-depth*
*Context gathered: 2026-03-04*
