# UAT Issues: Phase 3 Plan 4

**Tested:** 2026-01-10
**Source:** .planning/phases/03-hybrid-forecasting/03-04-SUMMARY.md
**Tester:** User via /gsd:verify-work

## Open Issues

### UAT-001: Gemini API rejects additionalProperties in JSON schema

**Discovered:** 2026-01-10
**Phase/Plan:** 03-04
**Severity:** Blocker
**Feature:** Scenario generation with structured output
**Description:** When running forecast.py, Gemini API returns error: "additionalProperties is not supported in the Gemini API"
**Expected:** Structured generation should work with Pydantic models
**Actual:** API rejects the schema and falls back to unstructured generation which also fails
**Repro:**
1. Set GEMINI_API_KEY environment variable
2. Run: `python forecast.py "Will there be a cyberattack?" --output-format summary`
3. Observe error messages about additionalProperties

### UAT-002: Entity extraction tests failing in ensemble

**Discovered:** 2026-01-10
**Phase/Plan:** 03-04
**Severity:** Major
**Feature:** Entity extraction for TKG queries
**Description:** 3 tests in test_ensemble.py fail related to entity extraction from LLM scenarios
**Expected:** Entities should be extracted and TKG prediction should be available
**Actual:** TKG prediction marked as unavailable with "Insufficient entities for TKG query"
**Repro:**
1. Run: `pytest tests/test_ensemble.py -v`
2. Tests fail: TestEnsemblePerformance (2 tests) and TestEntityExtraction (1 test)

### UAT-003: Environment variable not loading from .env automatically

**Discovered:** 2026-01-10
**Phase/Plan:** 03-04
**Severity:** Minor
**Feature:** CLI environment configuration
**Description:** forecast.py doesn't automatically load .env file, requires manual export
**Expected:** Should load .env file if present
**Actual:** Must manually export GEMINI_API_KEY before running
**Repro:**
1. Have .env file with GEMINI_API_KEY
2. Run: `python forecast.py "test question"`
3. Get error about missing API key

## Resolved Issues

### UAT-001: Gemini API rejects additionalProperties in JSON schema ✅
**Resolved:** 2026-01-10
**Fix:** Removed additionalProperties, converted to array format, added meaningful properties to preserve functionality

### UAT-002: Entity extraction tests failing in ensemble ✅
**Resolved:** 2026-01-10
**Fix:** Fixed mock fixture to include scenarios in dict, enabling proper entity extraction

### UAT-003: Environment variable not loading from .env automatically ✅
**Resolved:** 2026-01-10
**Fix:** Added python-dotenv and load_dotenv() call in forecast.py

### Additional fixes applied:
- **Temperature scaling formula:** Fixed inverted formula from c^(1/T) to c^T
- **RAG validation test:** Fixed test to access nested validation_methods structure
- **Type mismatch:** Changed Dict[str, str] to Dict[str, Any] for flexible types
- **Model upgrade:** Updated to Gemini 3.0 Pro (gemini-3-pro-preview)

---

*Phase: 03-hybrid-forecasting*
*Plan: 04*
*Tested: 2026-01-10*