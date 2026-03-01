# Phase 13: Calibration, Monitoring & Hardening - Research

**Researched:** 2026-03-02
**Domain:** Dynamic calibration optimization, operational monitoring, prediction market comparison, unattended operation
**Confidence:** HIGH (stdlib/scipy for calibration, well-understood patterns) / MEDIUM (Polymarket API)

## Summary

Phase 13 encompasses four distinct engineering domains: (1) dynamic per-CAMEO ensemble weight calibration via L-BFGS-B optimization on accumulated prediction-outcome data, (2) operational monitoring with SMTP alerting and structured logging, (3) Polymarket prediction market comparison for external calibration validation, and (4) hardening for 7-day unattended operation via systemd supervision, disk monitoring, and PostgreSQL graceful degradation.

The calibration subsystem is the most technically involved -- it requires a hierarchical weight resolution chain (CAMEO root code -> 4 super-categories -> global), weekly recomputation with guardrails, cold-start priors from literature, and version history for rollback. The monitoring subsystem leverages Python stdlib (`logging.handlers.TimedRotatingFileHandler`, `smtplib`/`email.mime`) with `psutil` for disk monitoring -- no heavy frameworks needed. Polymarket integration uses the Gamma Markets API (REST, tag-filtered) with an LLM-assisted matching step to pair Polymarket contracts with Geopol forecasts.

**Primary recommendation:** Build calibration as an isolated `src/calibration/weight_optimizer.py` module that reads from `outcome_records` + `predictions` tables, computes optimal alpha per CAMEO code via `scipy.optimize.minimize(method='L-BFGS-B')`, and writes to `calibration_weights` + `calibration_weight_history` tables. Monitoring is a separate `src/monitoring/` package with alert manager, health enrichment, and log configuration. Polymarket is a self-contained `src/polymarket/` client + matcher + comparison service.

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `scipy` | >=1.11 (already installed) | L-BFGS-B optimization for Brier score minimization | Industry standard bounded optimization; already used by `temperature_scaler.py` |
| `numpy` | >=1.24 (already installed) | Array operations for Brier score computation | Already a project dependency |
| `sklearn.metrics.brier_score_loss` | (already installed) | Brier score computation | Already used by `brier_scorer.py` |
| `psutil` | >=6.0 | Disk usage monitoring, process memory/uptime | De facto standard for system monitoring in Python; 6.0 has full Python 3.14 support |
| `smtplib` + `email.mime` | stdlib | SMTP email alerts | Part of Python stdlib; no external dependency needed |
| `logging.handlers` | stdlib | `TimedRotatingFileHandler` for 30-day log rotation | Stdlib; no external dependency needed |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `polymarket-apis` | >=0.4.6 | Unified Polymarket API client (Gamma, CLOB, Data) | For fetching prediction market data; requires Python >=3.12 (system uses 3.14) |
| `aiohttp` | >=3.9 (already installed) | Async HTTP for Polymarket API calls | Alternative to `polymarket-apis` if that package proves problematic |
| `tenacity` | >=8.0 (already installed) | Retry logic for email sending and API calls | Already in project; use for transient failure recovery |
| `netcal` | >=1.3.5 (already installed) | ECE/MCE/ACE calibration metrics | Already used by `calibration_metrics.py` |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `polymarket-apis` | Raw `aiohttp` + Gamma REST API | More control, no Python >=3.12 constraint, but more boilerplate. Recommendation: start with raw `aiohttp` against `gamma-api.polymarket.com`, add SDK later if needed. |
| `smtplib` | `sendgrid` / `mailgun` SDK | External SaaS dependency; SMTP is simpler for a single-user system |
| `python-json-logger` | Custom `_JSONFormatter` | Already have a working JSON formatter in `logging_config.py`; extend it rather than adding dependency |
| `psutil` | `/proc` filesystem parsing | Cross-platform matters less here (Linux target), but psutil is cleaner and handles edge cases |

**Installation:**
```bash
uv add psutil
# Polymarket: prefer raw HTTP first, add SDK if needed:
# uv add polymarket-apis
```

## Architecture Patterns

### Recommended Project Structure

```
src/
├── calibration/
│   ├── weight_optimizer.py     # L-BFGS-B Brier score optimization (NEW)
│   ├── weight_loader.py        # Per-CAMEO weight resolution + caching (NEW)
│   ├── temperature_scaler.py   # Existing -- no changes
│   ├── isotonic_calibrator.py  # Existing -- no changes
│   └── priors.py               # Literature-derived cold-start priors (NEW)
├── monitoring/
│   ├── __init__.py
│   ├── alert_manager.py        # SMTP alerting + rate limiting (NEW)
│   ├── health_enricher.py      # Extend health endpoint with process stats (NEW)
│   ├── feed_monitor.py         # GDELT staleness detection (NEW)
│   ├── drift_monitor.py        # Rolling Brier score computation (NEW)
│   ├── budget_monitor.py       # Gemini API usage tracking (NEW)
│   ├── disk_monitor.py         # Disk usage + emergency cleanup (NEW)
│   └── digest.py               # Daily digest email assembly (NEW)
├── polymarket/
│   ├── __init__.py
│   ├── client.py               # Gamma API HTTP client (NEW)
│   ├── matcher.py              # Keyword + LLM hybrid matching (NEW)
│   └── comparison.py           # Brier score comparison service (NEW)
├── evaluation/
│   ├── drift_detector.py       # Existing -- refactor to use rolling Brier (MODIFY)
│   └── brier_scorer.py         # Existing -- no changes
├── pipeline/
│   └── daily_forecast.py       # Existing -- add calibration + monitoring hooks (MODIFY)
└── db/
    └── models.py               # Add CalibrationWeightHistory, PolymarketComparison (MODIFY)
```

### Pattern 1: Hierarchical Weight Resolution

**What:** Three-level fallback for ensemble alpha weights: CAMEO root code -> super-category -> global.
**When to use:** At prediction time in `EnsemblePredictor._combine_predictions()`.

```python
# Source: Phase 13 design based on CONTEXT.md decisions
class WeightLoader:
    """Loads per-CAMEO alpha weights with hierarchical fallback."""

    # CAMEO root codes to QuadClass super-categories
    CAMEO_TO_SUPER = {
        "01": "verbal_coop", "02": "verbal_coop", "03": "verbal_coop",
        "04": "verbal_coop", "05": "verbal_coop",
        "06": "material_coop", "07": "material_coop", "08": "material_coop",
        "09": "material_coop",
        "10": "verbal_conflict", "11": "verbal_conflict", "12": "verbal_conflict",
        "13": "verbal_conflict", "14": "verbal_conflict",
        "15": "material_conflict", "16": "material_conflict",
        "17": "material_conflict", "18": "material_conflict",
        "19": "material_conflict", "20": "material_conflict",
    }

    def resolve_alpha(self, cameo_root_code: str) -> float:
        """Resolve alpha with hierarchical fallback.

        1. Try per-CAMEO-root-code weight (e.g., "14" -> 0.55)
        2. Fall back to super-category weight (e.g., "verbal_conflict" -> 0.58)
        3. Fall back to global weight
        """
        # Level 1: specific CAMEO code
        weight = self._weights.get(cameo_root_code)
        if weight and weight.sample_size >= self._min_samples:
            return weight.alpha

        # Level 2: super-category
        super_cat = self.CAMEO_TO_SUPER.get(cameo_root_code)
        if super_cat:
            weight = self._weights.get(f"super:{super_cat}")
            if weight and weight.sample_size >= self._min_samples:
                return weight.alpha

        # Level 3: global
        return self._global_alpha
```

### Pattern 2: Brier Score Optimization via L-BFGS-B

**What:** Minimize Brier score as objective function with bounded alpha in [0.0, 1.0].
**When to use:** Weekly calibration recomputation.

```python
# Source: scipy.optimize.minimize docs + existing temperature_scaler.py pattern
from scipy.optimize import minimize

def optimize_alpha_for_category(
    predictions: list[float],  # Predicted probabilities from ensemble
    outcomes: list[float],     # Actual outcomes (0.0 or 1.0)
    tkg_probs: list[float],    # TKG component probabilities
    llm_probs: list[float],    # LLM component probabilities
) -> tuple[float, float]:
    """Find alpha minimizing Brier score for this category.

    Returns (optimal_alpha, achieved_brier_score).
    """
    predictions_arr = np.array(predictions)
    outcomes_arr = np.array(outcomes)
    tkg_arr = np.array(tkg_probs)
    llm_arr = np.array(llm_probs)

    def brier_for_alpha(alpha: np.ndarray) -> float:
        a = alpha[0]
        ensemble = a * llm_arr + (1 - a) * tkg_arr
        return float(np.mean((ensemble - outcomes_arr) ** 2))

    result = minimize(
        brier_for_alpha,
        x0=[0.6],  # Start from current default
        method="L-BFGS-B",
        bounds=[(0.0, 1.0)],
        options={"maxiter": 200, "ftol": 1e-10},
    )

    return float(result.x[0]), float(result.fun)
```

### Pattern 3: Alert Manager with Rate Limiting

**What:** SMTP-based alerting with deduplication and rate limiting to prevent alert storms.
**When to use:** All alert-firing points (feed staleness, drift, budget, disk).

```python
# Source: Python stdlib smtplib + email.mime
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

class AlertManager:
    """SMTP alerting with per-alert-type cooldown."""

    def __init__(self, smtp_host: str, smtp_port: int, sender: str,
                 recipient: str, cooldown_minutes: int = 60):
        self._last_sent: dict[str, datetime] = {}
        self._cooldown = timedelta(minutes=cooldown_minutes)
        # ... SMTP config

    def send_alert(self, alert_type: str, subject: str, body: str) -> bool:
        """Send alert if not in cooldown for this alert_type."""
        now = datetime.now(timezone.utc)
        last = self._last_sent.get(alert_type)
        if last and (now - last) < self._cooldown:
            logger.debug("Alert %s suppressed (cooldown)", alert_type)
            return False

        msg = MIMEMultipart("alternative")
        msg["Subject"] = f"[Geopol] {subject}"
        msg["From"] = self._sender
        msg["To"] = self._recipient
        msg.attach(MIMEText(body, "plain"))

        with smtplib.SMTP(self._smtp_host, self._smtp_port) as server:
            server.starttls()
            server.login(self._username, self._password)
            server.send_message(msg)

        self._last_sent[alert_type] = now
        return True
```

### Pattern 4: Polymarket Gamma API Client

**What:** Fetch geopolitical markets from Polymarket's Gamma API, filter by tags, snapshot prices hourly.
**When to use:** Hourly background task for price snapshots + matching.

```python
# Source: Polymarket Gamma API docs (https://docs.polymarket.com)
import aiohttp

GAMMA_API_BASE = "https://gamma-api.polymarket.com"

async def fetch_geopolitical_markets(
    session: aiohttp.ClientSession,
    limit: int = 100,
) -> list[dict]:
    """Fetch active geopolitical markets from Polymarket Gamma API.

    Uses tag discovery to find geopolitics-related tags, then filters.
    Returns list of event dicts with markets, outcomePrices, etc.
    """
    # Step 1: Discover available tags
    async with session.get(f"{GAMMA_API_BASE}/tags") as resp:
        tags = await resp.json()

    # Filter for geopolitical tags (politics, world, geopolitics, etc.)
    geo_tag_ids = [
        t["id"] for t in tags
        if any(kw in t["label"].lower()
               for kw in ["politic", "geopolitic", "world", "war",
                           "election", "international", "government"])
    ]

    # Step 2: Fetch active events for each tag
    all_events = []
    for tag_id in geo_tag_ids:
        params = {
            "tag_id": tag_id,
            "active": "true",
            "closed": "false",
            "limit": limit,
        }
        async with session.get(f"{GAMMA_API_BASE}/events", params=params) as resp:
            events = await resp.json()
            all_events.extend(events)

    # Deduplicate by event ID
    seen = set()
    unique = []
    for event in all_events:
        if event["id"] not in seen:
            seen.add(event["id"])
            unique.append(event)

    return unique
```

### Anti-Patterns to Avoid

- **Coupling calibration to prediction hot path:** The weight optimization should run as a scheduled background task (weekly), NOT inline during prediction. Only the weight *lookup* happens at prediction time.
- **Blocking SMTP in async context:** Use `asyncio.to_thread()` to wrap synchronous `smtplib` calls; never block the event loop.
- **Alert storms:** Without per-type cooldowns, a sustained failure (e.g., GDELT down for hours) would generate an alert every 15 minutes. Use minimum 60-minute cooldown per alert type.
- **Unbounded Polymarket polling:** Hourly snapshots should have a circuit breaker; if Polymarket is down, don't let retries accumulate. Use the existing `tenacity` retry pattern.
- **Global alpha fallback of 0.6:** The cold-start prior should come from published literature, not the arbitrary 0.6 default. Literature suggests TKG models (RE-GCN, TiRGN) achieve MRR 0.3-0.5 vs LLM Brier ~0.25-0.35 on comparable tasks, supporting a range of 0.55-0.65 for LLM alpha depending on event type.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Brier score computation | Custom squared-error loop | `sklearn.metrics.brier_score_loss` | Already in use; handles edge cases, validated |
| Bounded optimization | Grid search over alpha values | `scipy.optimize.minimize(method='L-BFGS-B')` | Converges faster, handles bounds natively, already used in `temperature_scaler.py` |
| Log rotation | Custom file management | `logging.handlers.TimedRotatingFileHandler` | Stdlib, atomic rotation, configurable retention |
| Disk usage monitoring | `/proc` parsing | `psutil.disk_usage()` | Cross-platform, handles mount points, memory-safe |
| Email MIME construction | String concatenation | `email.mime.multipart.MIMEMultipart` | RFC 2822 compliant, handles encoding, attachments |
| CAMEO taxonomy mapping | Database lookup table | Hardcoded dict constant | 20 root codes is static; CAMEO hasn't changed since 2012; a DB table adds complexity for zero benefit |
| JSON structured logging | Custom formatter from scratch | Extend existing `_JSONFormatter` in `logging_config.py` | Already built, just needs file handler addition |

**Key insight:** This phase's primary complexity is in the *orchestration* (when to run what, how subsystems interact, failure cascading) not in individual algorithms. Every algorithm component has a well-tested stdlib or scipy solution. The engineering challenge is wiring them together correctly with proper error isolation.

## Common Pitfalls

### Pitfall 1: Brier Score Optimization on Insufficient Data

**What goes wrong:** L-BFGS-B converges to extreme alpha values (0.0 or 1.0) when sample size is small, producing wildly miscalibrated weights.
**Why it happens:** With <10 samples, the Brier score surface is noisy; the optimizer fits to noise rather than signal.
**How to avoid:** Enforce the minimum sample threshold (10 outcomes per category) before computing category-specific weights. Fall back to super-category or global. The 20% relative deviation guardrail catches post-optimization runaway.
**Warning signs:** Any optimized alpha at exact bounds (0.0 or 1.0); massive alpha swings between weekly runs.

### Pitfall 2: CAMEO Code Mismatch Between Subsystems

**What goes wrong:** The ensemble predictor uses categories like "conflict"/"diplomatic"/"economic" (from `_infer_category()`), but CAMEO root codes are "01"-"20". Weight resolution fails because the key space doesn't match.
**Why it happens:** The v1.0 `EnsemblePredictor` infers broad categories from keywords, not CAMEO codes. The `CalibrationWeight.cameo_code` field stores CAMEO codes.
**How to avoid:** The `WeightLoader` must accept BOTH the CAMEO root code (from GDELT `EventRootCode`) AND the keyword-inferred category. When CAMEO code is available, use it; when only keyword category exists, map it to the appropriate super-category. The `Prediction` model already has a `category` field -- extend it to also store `cameo_root_code` if the prediction originated from a GDELT event.
**Warning signs:** Weight resolution always falling back to global because no CAMEO code is found.

### Pitfall 3: Email Alert Blocking the Event Loop

**What goes wrong:** `smtplib.SMTP()` is synchronous and can block for 30+ seconds on connection timeout, freezing the async pipeline.
**Why it happens:** SMTP connection establishment involves DNS resolution + TCP handshake + TLS negotiation, all synchronous.
**How to avoid:** Always use `asyncio.to_thread(self._send_sync, ...)` for SMTP operations. Set `timeout=10` on SMTP connection. Fire-and-forget alert sending (don't let SMTP failures cascade into pipeline failures).
**Warning signs:** Pipeline latency spikes correlating with alert firing; SMTP timeout errors in logs.

### Pitfall 4: Polymarket Rate Limiting and API Changes

**What goes wrong:** Polymarket API rate limits or changes endpoint structure, breaking hourly snapshots silently (no data, no error).
**Why it happens:** Polymarket is a third-party service; their API is not versioned and has changed structure before.
**How to avoid:** Implement defensive parsing with fallback to empty results on unexpected response shapes. Add a circuit breaker that disables polling after 5 consecutive failures and alerts. Store the last successful fetch timestamp and alert if no successful fetch in >6 hours.
**Warning signs:** `polymarket_comparisons` table has no new rows for >24h; empty `outcomePrices` arrays in responses.

### Pitfall 5: CalibrationWeight Table Schema Mismatch

**What goes wrong:** The existing `calibration_weights` table has `cameo_code` as UNIQUE, meaning super-category weights and global weights can't coexist with root-code weights.
**Why it happens:** The v1.0 schema assumed one weight per CAMEO code, not a hierarchy.
**How to avoid:** The `cameo_code` column already allows string values. Use naming conventions: `"01"`-`"20"` for root codes, `"super:verbal_coop"`, `"super:material_coop"`, `"super:verbal_conflict"`, `"super:material_conflict"` for super-categories, and `"global"` for the global weight. This keeps the UNIQUE constraint valid while supporting hierarchy.
**Warning signs:** Alembic migration fails due to unique constraint violations.

### Pitfall 6: Rolling Brier Score with Sparse Outcome Data

**What goes wrong:** The 30-day rolling Brier score window has too few resolved predictions to be statistically meaningful, triggering false drift alerts.
**Why it happens:** If the system produces 25 predictions/day but outcomes resolve after 21 days (horizon_days), the 30-day window only contains predictions from the first 9 days that have resolved.
**How to avoid:** Require a minimum of 20 resolved predictions in the rolling window before computing drift metrics. Below that threshold, report "insufficient data" instead of a noisy Brier score.
**Warning signs:** Drift alerts firing immediately after deployment before enough outcomes have accumulated.

## Code Examples

### Weekly Calibration Pipeline (Pseudocode)

```python
# Source: Designed from CONTEXT.md decisions + existing codebase patterns

async def run_weekly_calibration(
    session: AsyncSession,
    min_samples: int = 10,
    max_relative_deviation: float = 0.20,
) -> CalibrationResult:
    """Weekly calibration recomputation pipeline.

    1. Query all resolved prediction-outcome pairs
    2. Group by CAMEO root code
    3. For each group with >= min_samples: optimize alpha via L-BFGS-B
    4. For groups below threshold: aggregate to super-category
    5. Apply guardrails (20% max deviation from current)
    6. Persist new weights + version history
    """
    # Fetch resolved predictions with their component probabilities
    pairs = await fetch_resolved_pairs(session)

    # Group by CAMEO root code
    by_cameo = group_by_cameo(pairs)

    new_weights: dict[str, tuple[float, float, int]] = {}  # code -> (alpha, brier, n)

    # Level 1: per-CAMEO root code
    for code, group in by_cameo.items():
        if len(group) >= min_samples:
            alpha, brier = optimize_alpha_for_category(
                outcomes=[p.outcome for p in group],
                tkg_probs=[p.tkg_probability for p in group],
                llm_probs=[p.llm_probability for p in group],
            )
            new_weights[code] = (alpha, brier, len(group))

    # Level 2: super-category aggregation for under-sampled codes
    super_groups = aggregate_to_super_categories(by_cameo)
    for super_code, group in super_groups.items():
        key = f"super:{super_code}"
        if key not in new_weights and len(group) >= min_samples:
            alpha, brier = optimize_alpha_for_category(...)
            new_weights[key] = (alpha, brier, len(group))

    # Level 3: global
    all_pairs = [p for pairs in by_cameo.values() for p in pairs]
    if len(all_pairs) >= min_samples:
        alpha, brier = optimize_alpha_for_category(...)
        new_weights["global"] = (alpha, brier, len(all_pairs))

    # Guardrail check: flag weights deviating >20% from current
    current_weights = await load_current_weights(session)
    flagged = check_guardrails(new_weights, current_weights, max_relative_deviation)

    if flagged:
        # Hold flagged weights, alert, apply only unflagged
        await alert_manager.send_alert(
            "calibration_guardrail",
            "Calibration guardrail triggered",
            f"Weights flagged: {flagged}",
        )

    # Persist (unflagged weights auto-applied, flagged held)
    await persist_weights(session, new_weights, flagged)
    await persist_weight_history(session, new_weights)

    return CalibrationResult(...)
```

### Health Endpoint Enhancement

```python
# Source: Extend existing src/api/routes/v1/health.py

import psutil

def _check_disk_usage() -> SubsystemStatus:
    """Check disk usage with alert thresholds."""
    now = datetime.now(timezone.utc)
    try:
        usage = psutil.disk_usage("/opt/geopol/data")
        pct = usage.percent
        free_gb = usage.free / (1024 ** 3)

        if pct >= 90:
            return SubsystemStatus(
                name="disk",
                healthy=False,
                detail=f"CRITICAL: {pct:.1f}% used, {free_gb:.1f} GB free",
                checked_at=now,
            )
        elif pct >= 80:
            return SubsystemStatus(
                name="disk",
                healthy=True,  # Degraded but not down
                detail=f"WARNING: {pct:.1f}% used, {free_gb:.1f} GB free",
                checked_at=now,
            )
        return SubsystemStatus(
            name="disk",
            healthy=True,
            detail=f"{pct:.1f}% used, {free_gb:.1f} GB free",
            checked_at=now,
        )
    except Exception as exc:
        return SubsystemStatus(
            name="disk", healthy=False, detail=str(exc)[:200], checked_at=now,
        )


def _check_process_health(process_name: str) -> dict:
    """Get process stats for systemd-managed daemons."""
    for proc in psutil.process_iter(["pid", "name", "cmdline", "create_time",
                                      "memory_info", "status"]):
        try:
            if process_name in " ".join(proc.info["cmdline"] or []):
                return {
                    "pid": proc.info["pid"],
                    "uptime_seconds": time.time() - proc.info["create_time"],
                    "memory_mb": proc.info["memory_info"].rss / (1024 * 1024),
                    "status": proc.info["status"],
                }
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return {"status": "not_found"}
```

### Structured Logging with File Rotation

```python
# Source: Python stdlib logging.handlers docs + existing logging_config.py

import logging
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

def setup_logging_with_rotation(
    level: str = "INFO",
    json_format: bool = True,
    log_dir: str = "/opt/geopol/data/logs",
    retention_days: int = 30,
) -> None:
    """Configure logging with daily file rotation and 30-day retention.

    Adds a TimedRotatingFileHandler alongside the existing stderr handler.
    """
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()

    # File handler with daily rotation
    file_handler = TimedRotatingFileHandler(
        filename=str(log_path / "geopol.log"),
        when="midnight",
        interval=1,
        backupCount=retention_days,
        encoding="utf-8",
        utc=True,
    )
    file_handler.setLevel(logging.DEBUG)  # Capture everything to file

    if json_format:
        file_handler.setFormatter(_JSONFormatter())
    else:
        file_handler.setFormatter(logging.Formatter(_HUMAN_FMT))

    root.addHandler(file_handler)
```

### Polymarket Matching via LLM

```python
# Source: CONTEXT.md decision on keyword + LLM hybrid matching

async def match_polymarket_to_geopol(
    polymarket_event: dict,
    active_predictions: list[Prediction],
    gemini_client: GeminiClient,
    threshold: float = 0.6,
) -> tuple[str | None, float]:
    """Match a Polymarket event to a Geopol prediction.

    Phase 1: Keyword filter (country + category)
    Phase 2: LLM ranks candidates and assigns confidence score

    Returns (prediction_id, confidence) or (None, 0.0) if no match.
    """
    pm_title = polymarket_event["title"]
    pm_description = polymarket_event.get("description", "")

    # Phase 1: Keyword pre-filter
    candidates = []
    for pred in active_predictions:
        # Country overlap check
        country_match = (
            pred.country_iso
            and pred.country_iso.upper() in (pm_title + pm_description).upper()
        )
        # Keyword overlap check
        pred_words = set(pred.question.lower().split())
        pm_words = set((pm_title + " " + pm_description).lower().split())
        keyword_overlap = len(pred_words & pm_words) / max(len(pred_words), 1)

        if country_match or keyword_overlap > 0.15:
            candidates.append(pred)

    if not candidates:
        return None, 0.0

    # Phase 2: LLM ranking
    candidates_text = "\n".join(
        f"{i+1}. [{c.id}] {c.question} (p={c.probability:.2f})"
        for i, c in enumerate(candidates[:10])
    )

    prompt = f"""Given this prediction market question:
"{pm_title}"
Description: {pm_description}

Which of these Geopol forecasts is the closest match?
{candidates_text}

Return JSON: {{"match_id": "<id or null>", "confidence": <0.0-1.0>}}
Only match if the questions are about substantially the same event/outcome."""

    response = await asyncio.to_thread(
        gemini_client.generate_content, prompt
    )
    # Parse response, extract match_id and confidence
    # Return (match_id, confidence) if confidence >= threshold
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Fixed alpha=0.6 | Per-CAMEO dynamic weights via L-BFGS-B | Phase 13 | Category-specific ensemble tuning; conflict events may favor TKG while diplomatic may favor LLM |
| v1.0 `DriftDetector` on JSON file | PostgreSQL-backed rolling Brier score | Phase 13 | Survives restarts; queryable; proper 30-day window |
| `api_budget` stub (always healthy) | Real Gemini usage tracking via `BudgetTracker` | Phase 13 | Actual budget remaining reported; daily cap enforced |
| No log files (stderr only) | `TimedRotatingFileHandler` with 30-day retention | Phase 13 | Persistent, rotated, searchable logs |
| No alerting | SMTP email alerts + daily digest | Phase 13 | Proactive incident notification |

**Deprecated/outdated:**
- `DriftDetector` JSON file storage (`data/calibration_metrics_history.json`): Replace with PostgreSQL queries against `outcome_records` + `predictions` tables.
- `DataQualityMonitor` (`src/monitoring.py`): Legacy v1.0 monitoring; functionality subsumed by new `src/monitoring/` package.
- `temperature_scaler.py` pickle persistence: Not removed, but calibration weights now live in PostgreSQL.

## Database Schema Extensions

### New Tables

```sql
-- Weight version history for rollback capability
CREATE TABLE calibration_weight_history (
    id SERIAL PRIMARY KEY,
    cameo_code VARCHAR(30) NOT NULL,     -- "01"-"20", "super:*", "global"
    alpha FLOAT NOT NULL,
    sample_size INTEGER NOT NULL,
    brier_score FLOAT,
    computed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    auto_applied BOOLEAN NOT NULL DEFAULT TRUE,
    flagged BOOLEAN NOT NULL DEFAULT FALSE,
    flag_reason TEXT
);
CREATE INDEX ix_cwh_computed ON calibration_weight_history(computed_at);
CREATE INDEX ix_cwh_cameo ON calibration_weight_history(cameo_code);

-- Polymarket comparison tracking
CREATE TABLE polymarket_comparisons (
    id SERIAL PRIMARY KEY,
    polymarket_event_id VARCHAR(100) NOT NULL,
    polymarket_slug VARCHAR(200) NOT NULL,
    polymarket_title TEXT NOT NULL,
    geopol_prediction_id VARCHAR(36) NOT NULL,
    match_confidence FLOAT NOT NULL,
    -- Snapshot fields
    polymarket_price FLOAT,              -- Latest Yes price (0.0-1.0)
    geopol_probability FLOAT,            -- Geopol ensemble probability
    last_snapshot_at TIMESTAMPTZ,
    -- Resolution fields
    status VARCHAR(20) NOT NULL DEFAULT 'active',  -- active | resolved
    polymarket_outcome FLOAT,            -- 0 or 1 when resolved
    geopol_brier FLOAT,                  -- Brier score for Geopol
    polymarket_brier FLOAT,              -- Brier score for Polymarket
    resolved_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX ix_pc_status ON polymarket_comparisons(status);
CREATE INDEX ix_pc_prediction ON polymarket_comparisons(geopol_prediction_id);

-- Polymarket price snapshots (hourly)
CREATE TABLE polymarket_snapshots (
    id SERIAL PRIMARY KEY,
    comparison_id INTEGER NOT NULL REFERENCES polymarket_comparisons(id),
    polymarket_price FLOAT NOT NULL,
    geopol_probability FLOAT NOT NULL,
    captured_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX ix_ps_comparison ON polymarket_snapshots(comparison_id);
```

### Modified Tables

The existing `calibration_weights` table schema is sufficient for the hierarchical weight storage (the `cameo_code` VARCHAR(10) column can store "01"-"20", but needs widening to VARCHAR(30) for "super:verbal_coop" etc.). Alternatively, keep it at 10 chars and use abbreviated codes ("sv_coop", "mc_conf", etc.).

The `Prediction` table should gain an optional `cameo_root_code` column to enable per-CAMEO weight lookup at prediction time.

## CAMEO Root Code to Super-Category Mapping

This is the authoritative mapping used for hierarchical calibration fallback.

| Root Code | Name | QuadClass | Super-Category |
|-----------|------|-----------|----------------|
| 01 | MAKE PUBLIC STATEMENT | 1 | verbal_coop |
| 02 | APPEAL | 1 | verbal_coop |
| 03 | EXPRESS INTENT TO COOPERATE | 1 | verbal_coop |
| 04 | CONSULT | 1 | verbal_coop |
| 05 | ENGAGE IN DIPLOMATIC COOPERATION | 1 | verbal_coop |
| 06 | ENGAGE IN MATERIAL COOPERATION | 2 | material_coop |
| 07 | PROVIDE AID | 2 | material_coop |
| 08 | YIELD | 2 | material_coop |
| 09 | INVESTIGATE | 2 | material_coop |
| 10 | DEMAND | 3 | verbal_conflict |
| 11 | DISAPPROVE | 3 | verbal_conflict |
| 12 | REJECT | 3 | verbal_conflict |
| 13 | THREATEN | 3 | verbal_conflict |
| 14 | PROTEST | 3 | verbal_conflict |
| 15 | EXHIBIT MILITARY POSTURE | 4 | material_conflict |
| 16 | REDUCE RELATIONS | 4 | material_conflict |
| 17 | COERCE | 4 | material_conflict |
| 18 | ASSAULT | 4 | material_conflict |
| 19 | FIGHT | 4 | material_conflict |
| 20 | ENGAGE IN UNCONVENTIONAL MASS VIOLENCE | 4 | material_conflict |

**Source:** CAMEO Codebook v1.1b3 (data.gdeltproject.org/documentation/CAMEO.Manual.1.1b3.pdf) + GDELT Data Format Codebook v2.0. This taxonomy has been stable since 2012.
**Confidence:** HIGH -- cross-verified against existing `src/constants.py` QuadClass definitions and `geopol.md` documentation.

## Cold-Start Prior Literature Values

For the cold-start alpha priors (before sufficient outcome data accumulates), the following literature-derived values are recommended:

| Category | Suggested Alpha (LLM weight) | Rationale |
|----------|------------------------------|-----------|
| verbal_coop | 0.65 | Diplomatic events require nuanced reasoning; LLM excels |
| material_coop | 0.60 | Mixed; economic cooperation has structural patterns TKG captures |
| verbal_conflict | 0.55 | Threat escalation has strong temporal patterns TKG captures |
| material_conflict | 0.50 | Armed conflict is most pattern-driven; TKG competitive with LLM |
| global | 0.58 | Weighted average reflecting overall LLM advantage but TKG utility |

**Source:** geopol.md reference to LLM forecasting evaluation (arXiv 2507.04562) showing crowd-aggregated LLM predictions achieving Brier ~0.25-0.35; TKG models (RE-GCN, TiRGN) achieving MRR 0.3-0.5 on temporal link prediction. The relative weighting is an informed estimate. **Confidence: MEDIUM** -- these are defensible starting points, not ground truth; the weekly calibration will rapidly adjust them.

## Settings Extensions

The following settings should be added to `src/settings.py`:

```python
# -- Calibration --
calibration_min_samples: int = 10          # Minimum outcomes for per-CAMEO weights
calibration_max_deviation: float = 0.20    # Relative deviation guardrail
calibration_recompute_day: int = 0         # Day of week (0=Monday)

# -- Monitoring --
smtp_host: str = ""
smtp_port: int = 587
smtp_username: str = ""
smtp_password: str = ""    # SecretStr in production
smtp_sender: str = ""
alert_recipient: str = ""
alert_cooldown_minutes: int = 60
feed_staleness_hours: float = 1.0
drift_threshold_pct: float = 10.0          # 10% worse than baseline
disk_warning_pct: float = 80.0
disk_critical_pct: float = 90.0

# -- Polymarket --
polymarket_enabled: bool = True
polymarket_poll_interval: int = 3600       # 1 hour
polymarket_match_threshold: float = 0.6

# -- Logging --
log_dir: str = "data/logs"
log_retention_days: int = 30
```

## Open Questions

1. **Component probability storage for calibration optimization**
   - What we know: The L-BFGS-B optimizer needs separate LLM and TKG probabilities per resolved prediction to recompute the optimal alpha. Currently, `Prediction` stores only the final ensemble probability.
   - What's unclear: Whether `ensemble_info_json` (a JSON blob on the Prediction model) reliably contains the individual component probabilities.
   - Recommendation: Verify that `ensemble_info_json` stores `llm_probability` and `tkg_probability` fields. If not, extend `EnsemblePredictor` to include them in the JSON blob during Phase 13 implementation. This is essential for calibration.

2. **Polymarket tag stability**
   - What we know: Polymarket's Gamma API supports tag filtering via `GET /tags`, but tag IDs are not documented as stable identifiers.
   - What's unclear: Whether tag IDs change, whether geopolitics has a dedicated tag, or if we need to discover tags dynamically each run.
   - Recommendation: Discover tags on first run, cache the mapping, re-discover weekly. If no "geopolitics" tag exists, use keyword filtering on event titles as primary strategy.

3. **systemd timer vs. internal scheduler for weekly calibration**
   - What we know: Daily forecast uses a systemd timer (`geopol-daily-forecast.timer`). Weekly calibration needs its own schedule.
   - What's unclear: Whether to add another systemd timer or schedule it within the daily pipeline as a "run calibration on Mondays" conditional.
   - Recommendation: Add it as a conditional in `daily_forecast.py` (if weekday == Monday, run calibration after outcomes). Avoids systemd unit proliferation and keeps related logic together.

## Sources

### Primary (HIGH confidence)
- `src/forecasting/ensemble_predictor.py` -- Current alpha=0.6 implementation, `_combine_predictions()` architecture
- `src/calibration/temperature_scaler.py` -- Existing L-BFGS-B pattern for temperature optimization
- `src/evaluation/brier_scorer.py` -- Brier score computation patterns
- `src/evaluation/drift_detector.py` -- Existing drift detection (to be upgraded)
- `src/db/models.py` -- Current `CalibrationWeight`, `OutcomeRecord`, `Prediction` schemas
- `src/api/routes/v1/health.py` -- Health endpoint architecture (8 subsystems)
- `src/pipeline/daily_forecast.py` -- Pipeline orchestration pattern
- `src/pipeline/budget_tracker.py` -- Gemini budget management
- `src/settings.py` -- Configuration pattern
- `src/logging_config.py` -- Existing JSON formatter
- `src/constants.py` -- QuadClass definitions, CAMEO event code mappings
- `geopol.md` -- CAMEO taxonomy, TKG performance benchmarks, LLM evaluation results
- [scipy.optimize.minimize L-BFGS-B docs](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html)
- [Python logging.handlers docs](https://docs.python.org/3/library/logging.handlers.html)
- [Python smtplib docs](https://docs.python.org/3/library/smtplib.html)

### Secondary (MEDIUM confidence)
- [Polymarket Gamma API Get Markets](https://docs.polymarket.com/developers/gamma-markets-api/get-markets) -- Verified endpoint structure and filtering params
- [Polymarket Fetch Markets Guide](https://docs.polymarket.com/developers/gamma-markets-api/fetch-markets-guide) -- Tag discovery and pagination
- [polymarket-apis PyPI](https://pypi.org/project/polymarket-apis/) -- v0.4.6, Python >=3.12, Pydantic validation
- Gamma API live response structure -- Verified via `gamma-api.polymarket.com/events` (fields: id, title, slug, markets[], outcomePrices[], tags[])
- [psutil documentation](https://psutil.readthedocs.io/) -- disk_usage, process_iter APIs

### Tertiary (LOW confidence)
- Cold-start alpha priors: derived from geopol.md literature citations + author judgment; not empirically validated
- Polymarket tag taxonomy for geopolitics: unverified; tag IDs may not include dedicated geopolitics category

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- scipy, numpy, stdlib logging/email are well-understood; psutil is mature
- Architecture: HIGH -- calibration optimization pattern mirrors existing `temperature_scaler.py`; monitoring uses stdlib
- Calibration L-BFGS-B: HIGH -- identical pattern to existing `_optimize_temperature()` in `temperature_scaler.py`
- Polymarket integration: MEDIUM -- API structure verified, but tag filtering and matching quality are unproven
- Cold-start priors: MEDIUM -- literature-informed but not empirically validated
- CAMEO mapping: HIGH -- cross-verified against codebase and official GDELT documentation
- Pitfalls: HIGH -- derived from direct codebase analysis of existing code patterns

**Research date:** 2026-03-02
**Valid until:** 2026-04-02 (30 days; Polymarket API may change faster)
