---
phase: quick-001
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - README.md
  - scripts/forecast.py
  - scripts/preflight.py
autonomous: true

must_haves:
  truths:
    - "README install instructions match actual tooling (uv sync, not pip)"
    - "User can run a forecast from CLI with a single command"
    - "User can validate system readiness before forecasting"
  artifacts:
    - path: "README.md"
      provides: "Corrected install instructions"
      contains: "uv sync"
    - path: "scripts/forecast.py"
      provides: "CLI forecast entry point"
      exports: ["main"]
    - path: "scripts/preflight.py"
      provides: "System readiness validator"
      exports: ["main"]
  key_links:
    - from: "scripts/forecast.py"
      to: "src/forecasting/forecast_engine.py"
      via: "ForecastEngine import and .forecast() call"
      pattern: "ForecastEngine"
    - from: "scripts/preflight.py"
      to: "data/events.db, chroma_db/, models/tkg/"
      via: "Path.exists() and content validation"
      pattern: "Path.*exists"
---

<objective>
Fix broken README install instructions, add a forecast CLI script, and add a preflight check script.

Purpose: The README tells users to run `pip install -r requirements.txt` but no requirements.txt exists (project uses uv). The forecast CLI gives users a clean interface to the ForecastEngine without writing Python. The preflight script lets users verify all components are ready before attempting a forecast.

Output: Updated README.md, scripts/forecast.py, scripts/preflight.py
</objective>

<execution_context>
@~/.claude/get-shit-done/workflows/execute-plan.md
@~/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@README.md
@scripts/e2e_forecast_test.py
@scripts/bootstrap.py
@src/forecasting/forecast_engine.py
</context>

<tasks>

<task type="auto">
  <name>Task 1: Fix README install instructions</name>
  <files>README.md</files>
  <action>
Replace the Setup section item 1. The current text:

```
1. Install dependencies:
```bash
pip install -r requirements.txt
```
```

Must become:

```
1. Install dependencies (requires [uv](https://docs.astral.sh/uv/)):
```bash
uv sync
```
```

Also add a new section "Quick Start" after Setup that shows the forecast workflow:

```markdown
## Quick Start

1. Verify system readiness:
```bash
uv run python scripts/preflight.py
```

2. Run a forecast:
```bash
uv run python scripts/forecast.py --question "Will NATO expand to include new members in the next 6 months?"
```

3. For detailed output:
```bash
uv run python scripts/forecast.py --question "..." --verbose
```
```

Do NOT alter any other README content. Preserve all existing sections and text exactly.
  </action>
  <verify>Grep README.md for "pip install" — must return zero matches. Grep for "uv sync" — must return at least one match. Grep for "Quick Start" — must return one match.</verify>
  <done>README Setup section references `uv sync`, not pip. Quick Start section exists with preflight and forecast CLI examples.</done>
</task>

<task type="auto">
  <name>Task 2: Create preflight check script</name>
  <files>scripts/preflight.py</files>
  <action>
Create `scripts/preflight.py` — a system readiness validator that checks all components needed for forecasting.

Structure: Follow the same script patterns as bootstrap.py (shebang, PROJECT_ROOT setup, sys.path.insert, argparse).

The script runs these checks in order, printing a pass/fail checklist:

1. **Python imports** — Attempt importing: `dotenv`, `chromadb`, `google.generativeai`, `networkx`, `pandas`, `llama_index`. Each import is a separate check line. Failed imports print the package name to install.

2. **Environment** — Check `.env` exists at PROJECT_ROOT. If it exists, load it with `dotenv` and check `os.environ.get("GEMINI_API_KEY")` is set and is not empty or a placeholder like "your-key-here" or "PLACEHOLDER". Print the first 8 chars masked if found (e.g., `AIza****`).

3. **Event database** — Check `data/events.db` exists. If exists, open with sqlite3, run `SELECT COUNT(*) FROM events`, print row count. Catch any sqlite3 error gracefully.

4. **Knowledge graphs** — Check `data/graphs/` exists and contains at least one `.graphml` file (use `glob`).

5. **RAG vector store** — Check `chroma_db/` directory exists and is non-empty (has subdirectories/files).

6. **TKG model** (mark as OPTIONAL) — Check `models/tkg/regcn_trained.pt` exists. If missing, print "[OPTIONAL] TKG model not found — forecasting works without it (LLM-only mode)".

Output format — each check prints one line:
```
[PASS] Description
[FAIL] Description
       Fix: actionable guidance
[SKIP] Description (optional component)
```

At the end, print a summary: "X/Y checks passed". If any required check failed, print "Run `uv run python scripts/bootstrap.py` to initialize the system." and exit with code 1. If all required checks pass, exit 0.

Accept `--quiet` flag that suppresses PASS lines (only shows FAIL/SKIP).

Use no external dependencies beyond stdlib + dotenv (which is already a project dependency). Do NOT import any src/ modules — this script must work even if src/ is broken.
  </action>
  <verify>`uv run python scripts/preflight.py --help` exits 0 and shows usage. `uv run python scripts/preflight.py` runs without traceback (checks may pass or fail depending on local state, but the script itself must not crash).</verify>
  <done>preflight.py runs cleanly, prints a checklist of system component status, exits 0 if all required checks pass or 1 if any fail, and provides actionable fix guidance for failures.</done>
</task>

<task type="auto">
  <name>Task 3: Create forecast CLI script</name>
  <files>scripts/forecast.py</files>
  <action>
Create `scripts/forecast.py` — a CLI wrapper around ForecastEngine.

Structure: Follow e2e_forecast_test.py patterns exactly (shebang, PROJECT_ROOT, sys.path.insert, dotenv loading, logging setup with noise suppression for httpx/chromadb/sentence_transformers).

**Arguments (argparse):**
- `--question` / `-q`: Required. The forecast question string.
- `--verbose` / `-v`: Optional flag. Passed through to engine.forecast(verbose=True).
- `--no-rag`: Optional flag. Disables RAG pipeline.
- `--no-tkg`: Optional flag. Disables TKG predictor.
- `--alpha`: Optional float, default 0.6. LLM ensemble weight.
- `--temperature`: Optional float, default 1.0. Confidence calibration temperature.
- `--json`: Optional flag. Output raw JSON instead of formatted text.

**Component loading** — replicate the e2e_forecast_test.py pattern exactly:

1. Load RAGPipeline from `./chroma_db` via `RAGPipeline(persist_dir="./chroma_db")` then `rag.load_existing_index()`. If load fails or `--no-rag`, set rag to None. Wrap in try/except — if RAGPipeline import or init fails, print a warning to stderr and continue without RAG.

2. Load TKGPredictor. Check `models/tkg/regcn_trained.pt` exists, if so load via `tkg.load_pretrained(checkpoint_path)`. If missing or `--no-tkg`, set tkg to None. Wrap in try/except same as e2e script.

3. Initialize ForecastEngine with loaded components, alpha, temperature, enable_rag/enable_tkg flags matching what was loaded.

4. Call `engine.forecast(question, verbose=args.verbose)`.

**Output formatting (non-JSON mode):**

```
QUESTION: {question}

PREDICTION: {prediction}

PROBABILITY: {probability:.1%}
CONFIDENCE:  {confidence:.1%}

REASONING:
{reasoning_summary}

TOP SCENARIOS:
  1. {description} (P={probability:.1%})
  2. ...
  3. ...

ENSEMBLE:
  LLM: {llm_probability or "N/A"}  weight={llm_weight:.2f}
  TKG: {tkg_probability or "N/A"}  weight={tkg_weight:.2f}
```

**Output formatting (JSON mode):** `json.dumps(result, indent=2, default=str)` to stdout.

**Error handling:**
- If component loading fails entirely (no RAG, no TKG, no API key), print to stderr: "System not ready. Run `uv run python scripts/preflight.py` to diagnose." and exit 1.
- If forecast() raises, catch the exception, print the error to stderr, exit 1.
- On success, exit 0.

Do NOT use `any` types. Use proper type annotations on all functions. Keep the script under 200 lines.
  </action>
  <verify>`uv run python scripts/forecast.py --help` exits 0 and shows all arguments. `uv run python scripts/forecast.py -q "test" 2>&1` does not crash with import errors (it may fail if GEMINI_API_KEY is not set, but the error message should be clean and suggest running preflight).</verify>
  <done>forecast.py accepts --question, --verbose, --no-rag, --no-tkg, --alpha, --temperature, --json flags. Loads components following e2e_forecast_test.py patterns. Prints formatted or JSON output. Exits 0 on success, 1 on failure with actionable error messages.</done>
</task>

</tasks>

<verification>
1. `grep -c "pip install" README.md` returns 0
2. `grep -c "uv sync" README.md` returns >= 1
3. `uv run python scripts/preflight.py --help` exits 0
4. `uv run python scripts/preflight.py` runs without traceback
5. `uv run python scripts/forecast.py --help` exits 0
6. `uv run python scripts/forecast.py -q "test" 2>&1 || true` produces clean output (no traceback)
</verification>

<success_criteria>
- README install instructions use `uv sync`, not pip
- README has Quick Start section referencing both new scripts
- scripts/preflight.py validates all 6 component categories with pass/fail output
- scripts/forecast.py wraps ForecastEngine with full CLI argument support
- Both scripts follow existing codebase patterns (PROJECT_ROOT, sys.path, dotenv, logging)
- Both scripts handle missing components gracefully with actionable error messages
</success_criteria>

<output>
After completion, create `.planning/quick/001-fix-readme-add-forecast-cli-preflight/001-SUMMARY.md`
</output>
