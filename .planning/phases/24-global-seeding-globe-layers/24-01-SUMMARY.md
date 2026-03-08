---
phase: 24-global-seeding-globe-layers
plan: 01
subsystem: seeding
tags: [fips, iso, pycountry, pypopulation, h3, baseline-risk, csv]

dependency_graph:
  requires: []
  provides:
    - "src/seeding/ package with FIPS-to-ISO conversion, population lookup, baseline risk computation"
    - "FIPS-to-ISO CSV data file (251 mappings)"
    - "h3, pycountry, pypopulation dependencies"
  affects:
    - "24-02 (DB models reference seeding package)"
    - "24-03 (heavy job uses baseline_risk.compute_baseline_risk)"
    - "24-04 (GDELT poller FIPS conversion at ingestion)"
    - "24-05 (API routes query baseline risk data)"

tech_stack:
  added:
    - "h3 4.4.2 (H3 hex binning for heatmap layer)"
    - "pycountry 26.2.16 (canonical ISO country list + alpha-3 conversion)"
    - "pypopulation 2020.3 (World Bank population lookup)"
  patterns:
    - "CSV data file for large code mappings (content filter mitigation)"
    - "Module-level singleton dict loaded at import time"
    - "Library-backed dynamic country set (no static dicts > 30 entries)"

key_files:
  created:
    - "src/seeding/__init__.py"
    - "src/seeding/fips.py"
    - "src/seeding/population.py"
    - "src/seeding/baseline_risk.py"
    - "src/seeding/data/fips_to_iso.csv"
  modified:
    - "pyproject.toml"
    - ".gitignore"

decisions:
  - id: "fips-csv-under-src"
    description: "FIPS CSV placed under src/seeding/data/ (not top-level data/) because data/ is gitignored"
    rationale: "data/ gitignore pattern catches any nested data/ directory; added negation rule !src/seeding/data/"

metrics:
  duration: "~4 min"
  completed: "2026-03-08"
---

# Phase 24 Plan 01: Seeding Package Foundation Summary

**One-liner:** FIPS-to-ISO CSV (251 mappings) + seeding package with population lookup and 4-input baseline risk formula with advisory hard floors.

## What Was Done

### Task 1: FIPS CSV data file + Python dependencies

Created `src/seeding/` package with `data/fips_to_iso.csv` containing 251 FIPS 10-4 to ISO 3166-1 alpha-2 mappings. Generated programmatically from FIPS 10-4 standard with manual overrides for 25+ known collision mappings. All critical collisions verified: UK->GB, IS->IL, AS->AU, CH->CN, NI->NG, RS->RU, GM->DE, JA->JP, KS->KR, SF->ZA, etc.

Added h3 4.4.2, pycountry 26.2.16, pypopulation 2020.3 as core dependencies. Added `.gitignore` negation for `src/seeding/data/` since the blanket `data/` pattern blocked it.

### Task 2: Seeding modules (fips.py, population.py, baseline_risk.py)

**fips.py:**
- `_load_fips_mapping()` reads CSV via `csv.DictReader` at import time
- Module-level `FIPS_TO_ISO` dict (251 entries)
- `fips_to_iso(code)` handles 2-letter FIPS codes (CSV lookup) and 3-letter ISO alpha-3 (pycountry)
- `get_sovereign_isos()` builds ~250-code set dynamically from pycountry + XK

**population.py:**
- 3-entry `_POPULATION_OVERRIDES` dict (TW, EH, VA)
- `get_population(iso)` wraps pypopulation with overrides, returns 1 for unknown

**baseline_risk.py:**
- `WEIGHTS`: advisory 35%, ACLED 25%, GDELT 25%, Goldstein 15%
- `ADVISORY_FLOORS`: Level 4 >= 70, Level 3 >= 45
- `compute_baseline_risk()`: per-capita GDELT density, ACLED intensity, advisory score, inverted Goldstein
- `decay_weight(age_days, half_life)`: exponential decay utility for time-weighted aggregation

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] .gitignore data/ pattern blocked src/seeding/data/ directory**
- **Found during:** Task 1
- **Issue:** The blanket `data/` gitignore pattern catches any nested `data/` directory, including `src/seeding/data/`
- **Fix:** Added `!src/seeding/data/` negation rule to `.gitignore` (same pattern as existing `!frontend/public/data/`)
- **Files modified:** `.gitignore`
- **Commit:** 62b6241

## Verification Results

| Check | Result |
|-------|--------|
| FIPS CSV exists with 251 data rows | PASS |
| UK->GB mapping correct | PASS |
| IS->IL mapping correct | PASS |
| fips_to_iso('USA') returns 'US' (alpha-3 handling) | PASS |
| h3 4.4.2 importable | PASS |
| pycountry 249 countries | PASS |
| pypopulation returns US population | PASS |
| get_population('TW') returns 23900000 | PASS |
| get_population('XX') returns 1 (safe fallback) | PASS |
| Advisory level 4 score >= 70.0 (floor enforced) | PASS |
| decay_weight(0) == 1.0 | PASS |
| decay_weight(30) ~= 0.5 | PASS |
| No static dicts > 30 entries in any module | PASS |
| All modules import cleanly | PASS |

## Commits

| # | Hash | Message |
|---|------|---------|
| 1 | 62b6241 | feat(24-01): ship FIPS-to-ISO CSV data file and add seeding dependencies |
| 2 | ee80854 | feat(24-01): create seeding modules (fips, population, baseline_risk) |
