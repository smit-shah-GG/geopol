/**
 * TypeScript interfaces mirroring the Pydantic V2 DTOs from src/api/schemas/.
 *
 * Field names use snake_case to match the JSON wire format exactly.
 * This is the frontend's side of the API contract -- changes to the
 * Pydantic schemas MUST be reflected here.
 */

// --- forecast.py ---

/** A single piece of evidence supporting a scenario. */
export interface EvidenceDTO {
  source: string;
  description: string;
  confidence: number;
  timestamp: string | null;
  gdelt_event_id: string | null;
}

/** Ensemble weight breakdown between LLM and TKG. */
export interface EnsembleInfoDTO {
  llm_probability: number;
  tkg_probability: number | null;
  weights: Record<string, number>;
  temperature_applied: number;
}

/** Per-category calibration metadata. */
export interface CalibrationDTO {
  category: string;
  temperature: number;
  historical_accuracy: number;
  brier_score: number | null;
  sample_size: number;
}

/** A single scenario branch in the forecast scenario tree (recursive). */
export interface ScenarioDTO {
  scenario_id: string;
  description: string;
  probability: number;
  answers_affirmative: boolean;
  entities: string[];
  timeline: string[];
  evidence_sources: EvidenceDTO[];
  child_scenarios: ScenarioDTO[];
}

/** Complete forecast response -- the primary API contract. */
export interface ForecastResponse {
  forecast_id: string;
  question: string;
  prediction: string;
  probability: number;
  confidence: number;
  horizon_days: number;
  scenarios: ScenarioDTO[];
  reasoning_summary: string;
  evidence_count: number;
  ensemble_info: EnsembleInfoDTO;
  calibration: CalibrationDTO;
  created_at: string;
  expires_at: string;
}

// --- country.py ---

/** Aggregate risk summary for a single country. */
export interface CountryRiskSummary {
  iso_code: string;
  risk_score: number;
  forecast_count: number;
  top_forecast: string;
  top_probability: number;
  trend: 'rising' | 'stable' | 'falling';
  last_updated: string;
}

// --- health.py ---

/** Status of a single infrastructure subsystem. */
export interface SubsystemStatus {
  name: string;
  healthy: boolean;
  detail: string | null;
  checked_at: string;
}

/** Aggregate health status of the forecasting system. */
export interface HealthResponse {
  status: 'healthy' | 'degraded' | 'unhealthy';
  subsystems: SubsystemStatus[];
  timestamp: string;
  version: string;
}

// --- common.py ---

/** RFC 9457 Problem Details error response. */
export interface ProblemDetail {
  type: string;
  title: string;
  status: number;
  detail: string | null;
  instance: string | null;
}

/** Cursor-based paginated response wrapper. */
export interface PaginatedResponse<T> {
  items: T[];
  next_cursor: string | null;
  has_more: boolean;
}

// --- calibration.py (Polymarket comparison) ---

/** Single active or resolved Polymarket-vs-Geopol comparison. */
export interface PolymarketComparison {
  polymarket_title: string;
  polymarket_price: number;
  geopol_probability: number;
  match_confidence: number;
  last_snapshot_at: string | null;
  status: 'active' | 'resolved';
  geopol_brier: number | null;
  polymarket_brier: number | null;
  resolved_at: string | null;
}

// --- search.py ---

/** GET /forecasts/search result item. */
export interface SearchResult {
  forecast: ForecastResponse;
  relevance: number;
}

/** GET /forecasts/search response. */
export interface SearchResponse {
  results: SearchResult[];
  total: number;
  query: string;
  suggestions: string[] | null;
}

/** Full response from GET /api/v1/calibration/polymarket. */
export interface PolymarketComparisonResponse {
  active: PolymarketComparison[];
  resolved: PolymarketComparison[];
  summary: {
    active_count: number;
    resolved_count: number;
    geopol_avg_brier: number | null;
    polymarket_avg_brier: number | null;
    geopol_wins: number;
  };
  seeking_more_matches: boolean;
}
