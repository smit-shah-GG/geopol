---
phase: 24-global-seeding-globe-layers
plan: 04
subsystem: seeding
tags: [h3, heatmap, arcs, risk-delta, compute-all, apscheduler, heavy-job, skip-if-locked]

dependency_graph:
  requires:
    - "24-01 (seeding package: fips, population, baseline_risk)"
    - "24-02 (PostgreSQL ORM models: BaselineCountryRisk, HeatmapHexbin, CountryArc, RiskDelta, TravelAdvisory)"
  provides:
    - "Globe data computation engine: baseline risk + heatmap + arcs + deltas in one pass"
    - "APScheduler hourly baseline_risk heavy job with skip-if-locked semantics"
    - "H3 hex binning, bilateral arc extraction, risk delta computation modules"
  affects:
    - "24-05 (API endpoints read from these pre-computed PostgreSQL tables)"
    - "24-06 (Globe layer wiring consumes API data from these tables)"

tech_stack:
  added: []
  patterns:
    - "Skip-if-locked heavy job semantics (baseline_risk skips when lock held, unlike queue pattern)"
    - "Full table replace in single transaction (DELETE all + INSERT new)"
    - "Master orchestrator pattern: compute_all_layers() coordinates 4 sub-computations"

key_files:
  created:
    - src/seeding/heatmap_binner.py
    - src/seeding/arc_builder.py
    - src/seeding/risk_delta.py
    - src/seeding/compute_all.py
  modified:
    - src/scheduler/job_wrappers.py
    - src/scheduler/heavy_runner.py
    - src/scheduler/registry.py

decisions:
  - id: "skip-if-locked-not-queue"
    description: "heavy_baseline_risk checks _heavy_job_lock.locked() BEFORE acquiring -- skips silently if held"
    rationale: "Globe data being 1-2 hours stale is acceptable; blocking the ProcessPoolExecutor for hours behind daily_pipeline/backtest is not"
  - id: "full-table-replace"
    description: "Each compute cycle DELETEs all rows then INSERTs new ones (not UPSERT)"
    rationale: "Simpler than UPSERT, avoids orphan rows from countries that drop off, single transaction guarantees consistency"
  - id: "heatmap-30d-window"
    description: "Heatmap uses 30-day window vs 90-day for baseline risk"
    rationale: "Heatmap shows recent event hotspots -- 90 days of decay-weighted events would over-spread the signal"

metrics:
  duration: "~4 min"
  completed: "2026-03-08"
---

# Phase 24 Plan 04: Seeding Computation Engine + APScheduler Wiring Summary

**One-liner:** H3 hex binning + bilateral arcs + risk deltas + compute_all orchestrator, wired as 10th APScheduler heavy job with skip-if-locked semantics.

## Performance

- **Duration:** 4 min
- **Started:** 2026-03-08T12:35:11Z
- **Completed:** 2026-03-08T12:39:11Z
- **Tasks:** 2
- **Files created:** 4
- **Files modified:** 3

## Accomplishments

### Task 1: Seeding Computation Modules

**heatmap_binner.py:**
- `bin_events_to_h3()`: aggregates geocoded events into H3 hex cells at resolution 3 (~9,229 km^2/cell)
- Weight formula: `severity * mentions_norm * decay_weight` where severity = abs(goldstein)/10, mentions_norm = min(mentions, 100)/100
- Events without lat/lon silently skipped
- Returns list of `{h3_index, weight, event_count}`

**arc_builder.py:**
- `extract_bilateral_arcs()`: extracts top-N bilateral country relationships from event pairs
- Actor1 country extraction: parses first 2-3 chars of actor1_code as FIPS/ISO prefix
- Canonical pair ordering (min_iso, max_iso) merges bidirectional relationships
- Domestic events (same source and target) excluded
- Returns list of `{source_iso, target_iso, event_count, avg_goldstein}`

**risk_delta.py:**
- `compute_risk_deltas()`: compares current vs previous baseline risk, filters to |delta| >= threshold (default 10.0)
- Countries absent from previous use 0.0 as baseline (new entries)
- Results sorted by absolute delta descending

### Task 2: Orchestrator + APScheduler Wiring

**compute_all.py:**
- `compute_all_layers()`: master async orchestrator that coordinates all 4 computations
- Loads events from SQLite (90-day window), advisories from PostgreSQL
- Iterates `get_sovereign_isos()` (~250 countries) -- no static dicts
- Computes per-country decay-weighted stats (GDELT/ACLED split)
- Writes all results to PostgreSQL in a single session (full table replace: DELETE + INSERT)
- Returns counts dict: `{countries, hexbins, arcs, deltas}`

**heavy_runner.py:**
- `run_baseline_risk()`: module-level pickleable function, creates own asyncio event loop

**job_wrappers.py:**
- `heavy_baseline_risk()`: checks `_heavy_job_lock.locked()` BEFORE acquiring (skip-if-locked)
- Unlike other heavy jobs which queue via `async with _heavy_job_lock`, this one returns immediately when lock is held
- Updated docstring: 10 total jobs (was 9), 4 heavy jobs (was 3)

**registry.py:**
- Registered `baseline_risk` job: `IntervalTrigger(seconds=3600)`, `coalesce=True`, `max_instances=1`, `misfire_grace_time=1800`
- Updated docstring: 10 total jobs
- Import of `heavy_baseline_risk` added

## Deviations from Plan

None -- plan executed exactly as written.

## Verification Results

| Check | Result |
|-------|--------|
| heatmap_binner imports cleanly | PASS |
| arc_builder imports cleanly | PASS |
| risk_delta functional test (SY delta=20, US delta=5 filtered) | PASS |
| compute_all imports cleanly | PASS |
| run_baseline_risk exists in heavy_runner.py | PASS |
| heavy_baseline_risk exists in job_wrappers.py | PASS |
| registry.py registers baseline_risk job | PASS |
| job_wrappers.py uses _heavy_job_lock.locked() check | PASS |
| registry.py docstring says "10 background jobs" | PASS |
| No static dicts > 30 entries in any module | PASS |
| Country iteration via get_sovereign_isos() | PASS |

## Commits

| # | Hash | Message |
|---|------|---------|
| 1 | 492a932 | feat(24-04): create heatmap binner, arc builder, and risk delta modules |
| 2 | 01f91c7 | feat(24-04): create compute_all orchestrator and wire APScheduler baseline_risk job |

## Next Phase Readiness

- API endpoints (Plan 05) can query `baseline_country_risk`, `heatmap_hexbins`, `country_arcs`, `risk_deltas` tables
- Globe layer wiring (Plan 06) can consume pre-computed data from API endpoints
- All computation is pre-computed hourly -- API just reads latest rows
- Skip-if-locked ensures baseline_risk never blocks other heavy jobs

---
*Phase: 24-global-seeding-globe-layers*
*Completed: 2026-03-08*
