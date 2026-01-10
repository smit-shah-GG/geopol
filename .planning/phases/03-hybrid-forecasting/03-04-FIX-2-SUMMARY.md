# Phase 3 Plan 4 Fix Round 2: Schema Properties Summary

**Fixed Gemini API empty object properties while preserving metadata functionality**

## Issues Fixed

1. **Empty object properties for Entity.attributes**
   - Added `category` (string) and `confidence` (number) properties
   - Allows tracking entity categories (e.g., "military", "economic") and confidence scores

2. **Empty object properties for ScenarioTree.metadata**
   - Added `generated_at` (string): ISO timestamp of generation
   - Added `model` (string): Gemini model used for generation
   - Added `context_count` (integer): Number of context items provided

## Files Modified

### src/forecasting/models.py
- Modified `get_scenario_schema()`: Added properties to Entity.attributes object type
- Modified `get_scenario_tree_schema()`: Added properties to metadata object type
- Both changes satisfy Gemini API requirement that object types must have at least one property

### src/forecasting/scenario_generator.py
- Added `datetime` import for timestamp generation
- Modified `_parse_scenario_tree()`: Ensures metadata has required fields with defaults
- Modified `generate_scenarios()`: Populates `context_count` from input context
- Maintains backward compatibility by providing defaults for missing fields

## Technical Solution

**Root Cause**: Gemini API's structured output validation rejects object types with no properties defined (`{"type": "object"}` without `"properties"`).

**Fix**: Added meaningful properties to both empty object types:
- Entity attributes now support category classification and confidence tracking
- Metadata now captures generation context (timestamp, model, context count)

This preserves the flexible dictionary nature of these fields while satisfying API validation requirements.

## Verification

```bash
# Verify schema has properties
uv run python -c "from src.forecasting.models import get_scenario_tree_schema; import json; print(json.dumps(get_scenario_tree_schema()['properties']['metadata'], indent=2))"

# Verify generator imports
uv run python -c "from src.forecasting.scenario_generator import ScenarioGenerator; print('Generator loaded successfully')"
```

## Commits

1. `cd0fbd4` - fix(03-04-FIX-2): add properties to empty object types in Gemini schema
2. `56940e2` - fix(03-04-FIX-2): populate metadata fields in scenario parser

## Ready for Testing

The Gemini API schema validation should now pass without "should be non-empty for OBJECT type" errors. Run:

```bash
python forecast.py "Will there be a conflict?" --output-format summary
```

Expected outcome: No schema validation errors, successful forecast generation with populated metadata.

## Related Files

- Plan: `.planning/phases/03-hybrid-forecasting/03-04-FIX-2.md`
- Previous fix: `.planning/phases/03-hybrid-forecasting/03-04-FIX-SUMMARY.md`
- State: `.planning/STATE.md`
- Roadmap: `.planning/ROADMAP.md`
