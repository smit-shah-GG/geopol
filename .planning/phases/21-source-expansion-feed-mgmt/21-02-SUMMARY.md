---
phase: 21
plan: 02
subsystem: knowledge-graph, api
tags: [dedup, cross-source, fingerprint, cameo, chromadb, articles, recent-sort]
depends_on:
  requires: []
  provides:
    - "Cross-source dedup filter for knowledge graph insertion"
    - "Articles API recent-sort mode for NewsFeedPanel"
  affects:
    - "21-03+ (NewsFeedPanel consumes ?sort=recent endpoint)"
    - "Future ACLED/UCDP integration (dedup filter ready)"
tech-stack:
  added: []
  patterns:
    - "Session-scoped dedup filter with source priority"
    - "Conservative fingerprint: (date, country, coarse_event_type)"
    - "ChromaDB over-fetch + Python-side sort for time ordering"
key-files:
  created:
    - src/knowledge_graph/cross_source_dedup.py
    - src/knowledge_graph/test_cross_source_dedup.py
  modified:
    - src/knowledge_graph/graph_builder.py
    - src/api/routes/v1/articles.py
decisions:
  - id: "21-02-01"
    decision: "Actor1 code used as country_iso proxy for fingerprinting (first 3 chars)"
    rationale: "GDELT events don't have a dedicated country_iso field; actor1_code is the closest approximation at the graph insertion layer"
  - id: "21-02-02"
    decision: "First-insert wins structurally; cross-source collision logged but prior triple not retroactively removed"
    rationale: "Two-pass insertion would require graph mutation; single-pass with audit logging is simpler and the audit trail enables future tuning"
  - id: "21-02-03"
    decision: "Recent search uses has_more flag when over-fetched results exceed limit"
    rationale: "Signals to frontend that more articles exist beyond the requested page"
metrics:
  duration: "7 minutes"
  completed: "2026-03-06"
  tasks: "2/2"
  tests_added: 45
  tests_passing: "58 (13 existing + 45 new)"
---

# Phase 21 Plan 02: Cross-Source Dedup & Articles Recent Mode Summary

**One-liner:** Conservative (date, country, coarse_event_type) fingerprint dedup at graph insertion with ACLED priority, plus ChromaDB time-sorted recent endpoint for NewsFeedPanel.

## What Was Done

### Task 1: Cross-source dedup module + graph builder integration

Created `src/knowledge_graph/cross_source_dedup.py` with three exports:

1. **`cameo_to_coarse_type(cameo_code)`** -- Maps CAMEO 2-digit prefix to four coarse categories: cooperation (01-05), diplomacy (06-09), conflict (10-14), force (15-20). Invalid/empty codes return "unknown".

2. **`cross_source_fingerprint(event_date, country_iso, event_type)`** -- SHA-256 hash of `"{date}|{country}|{coarse_type}"`, truncated to 32 hex chars. Normalizes date to YYYY-MM-DD, country to uppercase (or "UNK"), type to uppercase.

3. **`CrossSourceDedupFilter`** -- Session-scoped filter maintaining `{fingerprint: (source, event_id)}` dict. Source priority: ACLED (3) > UCDP (2) > GDELT (1). Intra-source duplicates pass through (handled by existing `deduplication.py`). Cross-source collisions logged with both event IDs. Exposes `.stats` property with checked/suppressed/by_source_pair counts.

Integrated into `TemporalKnowledgeGraph`:
- `__init__()` creates a `CrossSourceDedupFilter` instance
- `add_event_from_db_row()` calls `dedup_filter.should_insert()` before entity resolution
- `add_events_batch()` logs dedup stats after batch completion
- Backward-compatible: `source` parameter defaults to "gdelt"

### Task 2: Articles API recent-sort mode

Extended `GET /api/v1/articles` with `?sort=recent` parameter:
- Fetches ChromaDB articles with `published_at >= (now - 24h)` filter
- Over-fetches (`limit * 3`, max 300) since ChromaDB `get()` returns unordered
- Sorts Python-side by `published_at` descending (ISO string comparison = chronological)
- Truncates to requested `limit`
- Sets `has_more=True` when more results exist beyond page
- Ignored in semantic mode (relevance-sorted by definition)
- Updated endpoint docstring and OpenAPI description

## Deviations from Plan

None -- plan executed exactly as written.

## Decisions Made

| # | Decision | Rationale |
|---|----------|-----------|
| 21-02-01 | Actor1 code used as country_iso proxy | GDELT events lack dedicated country_iso; actor1_code[:3] is the closest available signal |
| 21-02-02 | First-insert wins, collision logged | Single-pass with audit logging is simpler than two-pass graph mutation |
| 21-02-03 | has_more flag on recent search | Frontend can detect truncation and request larger pages |

## Verification

- `python -c "from src.knowledge_graph.cross_source_dedup import CrossSourceDedupFilter; print('OK')"` -- OK
- `graph_builder.py` imports and instantiates `CrossSourceDedupFilter` in `__init__`
- `add_event_from_db_row()` calls `should_insert()` before entity resolution
- Articles endpoint accepts `sort=recent` parameter (verified via `inspect.signature`)
- 58 tests pass (13 existing graph_builder + 45 new dedup)
- Full suite: 284 passed, 41 pre-existing failures (CUDA OOM, Phase 11 regcn default)

## Next Phase Readiness

Plan 21-02 provides two foundations:
1. **Dedup filter** is ready for ACLED and UCDP event ingestion -- callers just pass `source="acled"` or `source="ucdp"` to `add_event_from_db_row()`
2. **`?sort=recent`** endpoint is the data source for NewsFeedPanel (`/api/v1/articles?sort=recent&limit=100`)
