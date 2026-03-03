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

// --- submission.py ---

/** POST /forecasts/submit response -- LLM-parsed question. */
export interface ParsedQuestionResponse {
  request_id: string;
  question: string;
  country_iso_list: string[];
  horizon_days: number;
  category: string;
  status: string;
  parsed_at: string;
}

/** GET /forecasts/requests item -- submission lifecycle status. */
export interface ForecastRequestStatus {
  request_id: string;
  question: string;
  country_iso_list: string[];
  horizon_days: number;
  category: string;
  status: 'pending' | 'confirmed' | 'processing' | 'complete' | 'failed';
  submitted_at: string;
  completed_at: string | null;
  prediction_ids: string[];
  error_message: string | null;
}

/** POST /forecasts/submit/{id}/confirm response. */
export interface ConfirmSubmissionResponse {
  request_id: string;
  status: string;
  message: string;
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

/** Single event from GET /calibration/polymarket/top. */
export interface PolymarketTopEvent {
  event_id: string;
  title: string;
  slug: string;
  volume: number;
  liquidity: number;
  // Optional Geopol match data
  geopol_prediction_id: string | null;
  geopol_probability: number | null;
  geopol_question: string | null;
  match_confidence: number | null;
}

/** GET /calibration/polymarket/top response. */
export interface PolymarketTopResponse {
  events: PolymarketTopEvent[];
  total_geo_markets: number;
}

// --- event.py ---

/** A single GDELT or ACLED event from GET /events. */
export interface EventDTO {
  id: number;
  gdelt_id: string | null;
  event_date: string;
  actor1_code: string | null;
  actor2_code: string | null;
  event_code: string | null;
  quad_class: number | null;
  goldstein_scale: number | null;
  num_mentions: number | null;
  num_sources: number | null;
  tone: number | null;
  url: string | null;
  title: string | null;
  country_iso: string | null;
  source: string;
}

// --- article.py ---

/** An RSS/ChromaDB article chunk from GET /articles. */
export interface ArticleDTO {
  chunk_id: string;
  title: string;
  url: string;
  source_feed: string;
  country_iso: string | null;
  published_at: string | null;
  snippet: string;
  relevance_score: number | null;
}

// --- advisory.py ---

/** A government travel advisory from GET /advisories. */
export interface AdvisoryDTO {
  source: string;
  country_iso: string | null;
  level: number;
  level_description: string;
  title: string;
  summary: string;
  published_at: string | null;
  updated_at: string | null;
  url: string | null;
}

// --- source.py ---

/** Ingestion source health status from GET /sources. */
export interface SourceStatusDTO {
  name: string;
  healthy: boolean;
  last_update: string | null;
  events_last_run: number;
  detail: string;
}
