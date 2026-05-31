/** Backend API response types for DeskCanvas Console. */

// ── Page Model Types (E Phase 2) ──

export interface PageModelResponse {
  page_model_id: string;
  app_id: string;
  page_class_prefix: string;
  display_name: string | null;
  surface_type: string | null;
  observe_count: number;
  state_count: number;
  last_seen_at: string;
}

export interface StateTemplateResponse {
  state_template_id: string;
  page_model_id: string;
  page_class: string;
  state_label: string | null;
  fixed_element_count: number;
  total_element_count: number;
  verify_count: number;
  snapshot_count: number;
  last_seen_at: string;
}

export interface CanvasSnapshotResponse {
  snapshot_id: string;
  canvas_id: string;
  page_model_id: string;
  state_template_id: string;
  captured_at: string;
  element_count: number;
  has_screenshot: boolean;
  available: boolean; // canvas 仍在内存缓存中
}

export interface PageModelTreeResponse {
  page_models: PageModelResponse[];
  state_templates: StateTemplateResponse[];
  canvas_snapshots: CanvasSnapshotResponse[];
}

// ── Virtual Model Types (persistent, SQLite-backed) ──

export interface VirtualModelCandidate {
  key_id: string;
  element_id?: string | null; // Present when candidate is derived from a live CanvasDetail.
  is_canvas_candidate?: boolean; // True for transient workbench candidates from the current screenshot.
  canonical_text: string;
  canonical_role: string;
  canonical_region: string;
  role_label: string | null;
  semantic_tags: string[];
  visual_type: string;
  relative_bounds: number[]; // [x1/W, y1/H, x2/W, y2/H] — 0-1 floats
  absolute_bounds?: number[] | null; // Live canvas bounds when available.
  confidence: number;
  provider_sources: string[];
  source?: string; // "vlm" | "manual" | "uia" | "ocr" | etc.
  is_fixed_control: boolean;
  verify_count: number;
  seen_count: number;
  permanence_state: string; // "new"|"provisional"|"stable"|"fixed_anchor"|"stale"|"degraded"|"retired"
  confidence_profile?: ConfidenceProfile | null;
}

export interface ConfidenceProfile {
  spatial_confidence: number;
  semantic_confidence: number;
  text_confidence: number;
  structure_confidence: number;
  visual_anchor_confidence: number;
  memory_confidence: number;
  action_confidence: number;
  conflict_penalty: number;
  fused_confidence: number;
  source_count: number;
  source_diversity: number;
}

export interface VirtualModelDetail {
  state_template_id: string;
  page_model_id: string;
  app_id: string;
  display_name: string | null;
  page_class: string;
  state_label: string | null;
  surface_type: string | null;
  layout_fingerprint: Record<string, unknown>;
  fixed_element_count: number;
  total_element_count: number;
  verify_count: number;
  snapshot_count: number;
  last_seen_at: string;
  has_available_canvas: boolean;
  candidates: VirtualModelCandidate[];
  regions: Record<string, unknown>[];
  vlm_semantic_model: Record<string, unknown> | null;
  model_layers: Record<string, unknown>;
  app_shell: Record<string, unknown>;
  region_templates: Record<string, unknown>[];
  element_templates: Record<string, unknown>[];
  visible_items: Record<string, unknown>[];
  missing_suggestions: Record<string, unknown>[];
}

export interface TransitionEdgeResponse {
  from_page_class: string;
  to_page_class: string;
  trigger_action: string | null;
  trigger_candidate_key: string | null;
  observe_count: number;
  success_count: number;
  success_rate: number;
  last_observed_at: string | null;
  drifted: boolean;
}

export interface TransitionGraphResponse {
  page_class: string | null;
  candidate_key: string | null;
  transitions: TransitionEdgeResponse[];
  total: number;
}

export interface ControlTransitionResponse {
  candidate_key: string;
  stable_key_id: string | null;
  from_page_class: string;
  to_page_class: string;
  action_type: string | null;
  observe_count: number;
  success_count: number;
  failure_count: number;
  success_rate: number;
  last_observed_at: string | null;
  canvas_id_before: string | null;
  canvas_id_after: string | null;
}

export interface ControlTransitionGraphResponse {
  page_class: string | null;
  candidate_key: string | null;
  transitions: ControlTransitionResponse[];
  total: number;
}

export interface ModelTemplatesResponse {
  state_template_id: string;
  app_shell: Record<string, unknown>;
  region_templates: Record<string, unknown>[];
  element_templates: Record<string, unknown>[];
}

export interface CandidateOverrideResponse {
  override_id: string;
  scope_key: string;
  app_id: string | null;
  page_model_id: string | null;
  state_template_id: string | null;
  canvas_id: string | null;
  element_id: string | null;
  stable_key_id: string | null;
  label: string | null;
  semantic_role: string | null;
  visual_type: string | null;
  region_id: string | null;
  kind: 'fixed' | 'dynamic' | 'container' | 'ignored' | null;
  relative_bounds: number[] | null;
  absolute_bounds: number[] | null;
  source: string;
  status: string;
  created_at: string;
  updated_at: string;
}

export interface CandidateOverrideUpsertRequest {
  scope_key: string;
  app_id?: string | null;
  page_model_id?: string | null;
  state_template_id?: string | null;
  canvas_id?: string | null;
  element_id?: string | null;
  stable_key_id?: string | null;
  label?: string | null;
  semantic_role?: string | null;
  visual_type?: string | null;
  region_id?: string | null;
  kind?: 'fixed' | 'dynamic' | 'container' | 'ignored' | null;
  relative_bounds?: number[] | null;
  absolute_bounds?: number[] | null;
  source?: string;
  status?: string;
}

// ── Canvas Types ──

export interface CanvasSummary {
  canvas_id: string;
  app_id: string | null;
  display_name: string | null; // 软件中文名
  window_title: string;
  process_name: string | null;
  page_class: string;
  surface_type: string;
  element_count: number;
  region_count: number;
  captured_at: string;
  providers_used: string[];
  stable: boolean;
  loading: boolean;
  partial: boolean;
  has_screenshot: boolean;
  page_model_id: string | null;
  state_template_id: string | null;
  processing_state: string;
  processing_error: string;
}

export interface CandidateResponse {
  element_id: string;
  semantic_role: string;
  text: string;
  name: string | null;
  control_type: string;
  bounds: number[] | null;
  click_point: number[] | null;
  confidence: number;
  confidence_level: string;
  risk_tags: string[];
  risk_level: string;
  provider_sources: string[];
  interactable: boolean;
  from_memory: boolean;
  suggest_confirm: boolean;
  region_id: string | null;
  locator_ids: string[];
  attributes: Record<string, unknown>;
  // Refine fields (three-layer design)
  visual_type: string;
  semantic_tags: string[];
  role_label: string | null;
  role_confidence: number;
  role_source: string;
  role_evidence: string[];
  refine_status: string; // unreviewed / refined / uncertain
  stable_key_id: string | null; // E Phase 1: persistent candidate identity
}

export interface RegionResponse {
  region_id: string;
  role: string;
  subtype: string;
  bounds: number[] | null;
  element_count: number;
}

export interface ProviderTraceResponse {
  uia_used: boolean;
  ocr_used: boolean;
  vision_used: boolean;
  vlm_used: boolean;
  dom_used: boolean;
  provider_details: Record<string, unknown>;
}

export interface GeometricRegionResponse {
  region_id: string;
  bounds: number[];
  boundary_evidence: string[];
  density_profile: number[];
  geometry_confidence: number;
  parent_region_id: string | null;
}

export interface CanvasDetail {
  canvas_id: string;
  app_id: string | null;
  window_title: string;
  page_class: string;
  surface_type: string;
  page_model_id: string | null; // E Phase 2
  state_template_id: string | null; // E Phase 2
  page_model_name: string | null; // E Phase 2: display_name from page_models
  state_label: string | null; // E Phase 2: state_label from state_templates
  stable: boolean;
  loading: boolean;
  partial: boolean;
  captured_at: string;
  window_width: number;
  window_height: number;
  screenshot_width: number;
  screenshot_height: number;
  elements: CandidateResponse[];
  regions: RegionResponse[];
  geometric_regions: GeometricRegionResponse[];
  fusion_diagnostics: Record<string, unknown>;
  roi_selection_plan?: Record<string, unknown>;
  ocr_blocks: Record<string, unknown>[];
  vision_candidates: Record<string, unknown>[];
  roi_vlm_semantic_supplements: RoiVlmSupplement[];
  roi_vlm_timeouts: Record<string, unknown>[];
  roi_vlm_late_failures: Record<string, unknown>[];
  roi_vlm_rejected_responses: Record<string, unknown>[];
  vlm_semantic_model?: Record<string, unknown> | null;
  provider_trace: ProviderTraceResponse | null;
  providers_used: string[];
  providers_failed: string[];
  has_screenshot: boolean;
  processing_state: string;
  processing_error: string;
  processing_updated_at: string;
}

export interface RoiVlmCandidateAnnotation {
  candidate_id: string;
  role?: string;
  label?: string;
}

export interface RoiVlmSupplement {
  roi_id: string;
  status: string;
  late_result_accepted?: boolean;
  accepted_at?: string;
  warnings?: string[];
  region_semantics?: { role?: string; summary?: string };
  candidate_annotations?: RoiVlmCandidateAnnotation[];
  review_only_hints?: string[];
}

export interface WindowListItem {
  hwnd: number;
  title: string;
  class_name: string | null;
  process_name: string | null;
  process_id: number | null;
  is_foreground: boolean;
  is_minimized: boolean;
}

export interface ObserveResponse {
  canvas_id: string;
  elapsed_ms: number;
  app_id: string | null;
  page_class: string;
  surface_type: string;
  page_model_id: string | null; // E Phase 2
  state_template_id: string | null; // E Phase 2
  model_match_status: string; // E Phase 2: "reused" | "new_page" | "new_state"
  stable: boolean;
  loading: boolean;
  partial: boolean;
  confidence: number;
  element_count: number;
  region_count: number;
  providers_used: string[];
  providers_failed: string[];
  canvas_schema_version: string;
  vlm_semantic_used: boolean;
  vlm_semantic_status: string; // "success" | "cached" | "skipped" | "failed"
  vlm_semantic_provider: string;
  processing_state: string;
  enhancement_job_id: string | null;
  geometric_region_count: number;
  fusion_diagnostics: Record<string, unknown>;
}

export interface QueryTargetModel {
  text?: string;
  semantic_role?: string;
  region?: string;
  natural_language?: string;
  composite?: Record<string, unknown>;
}

export interface QueryResponse {
  candidates: CandidateResponse[];
  suggestions: string[];
  total_matched: number;
}

export interface RefineResultItem {
  element_id: string;
  visual_type: string;
  semantic_tags: string[];
  role_label: string | null;
  role_confidence: number;
  role_source: string;
  role_evidence: string[];
  refine_status: string;
}

export interface RefineResponse {
  canvas_id: string;
  mode: string;
  results: RefineResultItem[];
  total_refined: number;
  total_uncertain: number;
  total_unreviewed: number;
}

// ── VLM Semantic Types ──

export interface VLMSemanticRequest {
  canvas_id: string;
  force?: boolean;
  sync?: boolean;
  task_mode?: string;
  candidate_ids?: string[];
}

export interface VLMSemanticResponse {
  canvas_id: string;
  status: string; // "success" | "cached" | "failed" | "skipped"
  provider: string;
  model: string;
  error: string | null;
  error_code: string | null;
  message: string;
  latency_ms: number;
  token_input: number;
  token_output: number;
  from_cache: boolean;
  region_count: number;
  control_count: number;
  correction_count: number;
  dynamic_zone_count: number;
  warning_count: number;
  job_id: string | null;
  processing_state: string;
}

export interface RoiVlmSupplementResponse {
  canvas_id: string;
  status: string;
  deadline_ms: number;
  elapsed_ms: number;
  accept_late: boolean;
  jobs: Record<string, unknown>[];
  results: Record<string, unknown>[];
  result_status_counts: Record<string, number>;
  message: string;
}

export interface CanvasProcessingResponse {
  canvas_id: string;
  processing_state: string;
  jobs: string[];
  last_error: string;
  updated_at: string;
  current_phase: string;
  is_processing: boolean;
  latest_job: JobStatusResponse | null;
  active_jobs: JobStatusResponse[];
}

export interface JobStatusResponse {
  job_id: string;
  canvas_id: string;
  job_type: string;
  state: string;
  error: string;
  started_at: string;
  updated_at: string;
}

// ── VLM Semantic Settings ──

export interface SemanticModelerSettings {
  enabled: boolean;
  provider: string;
  endpoint: string;
  model: string;
  fallback_provider: string;
  fallback_model: string;
  provider_variant: string;
  free_model_only: boolean;
  allow_model_fallback: boolean;
  timeout_seconds: number;
  max_retries: number;
  proxy_url: string;
  proxy_port: number;
  prompt_version: string;
  daily_call_limit: number;
  monthly_budget_usd: number;
  priority: number;
  allow_free_models: boolean;
  cost_per_1k_input: number;
  cost_per_1k_output: number;
  save_raw_response: boolean;
  redact_dynamic_content: boolean;
  thinking_mode: string;
  image_max_width: number;
  api_key_set: boolean;
}

export interface SemanticModelerSettingsUpdate {
  enabled?: boolean;
  provider?: string;
  api_key?: string;
  endpoint?: string;
  model?: string;
  fallback_provider?: string;
  fallback_model?: string;
  provider_variant?: string;
  free_model_only?: boolean;
  allow_model_fallback?: boolean;
  timeout_seconds?: number;
  max_retries?: number;
  proxy_url?: string;
  proxy_port?: number;
  prompt_version?: string;
  daily_call_limit?: number;
  monthly_budget_usd?: number;
  priority?: number;
  allow_free_models?: boolean;
  cost_per_1k_input?: number;
  cost_per_1k_output?: number;
  save_raw_response?: boolean;
  redact_dynamic_content?: boolean;
  thinking_mode?: string;
  image_max_width?: number;
}

export interface SemanticModelerTestResponse {
  success: boolean;
  available: boolean;
  provider: string;
  model: string;
  endpoint: string;
  message: string;
  error: string | null;
  latency_ms: number;
  token_input: number;
  token_output: number;
  finish_reason: string;
  raw_preview: string;
}

export interface SemanticModelerModelInfo {
  id: string;
  name: string;
  provider: string;
  description: string;
  vision_capable: boolean;
  model_type: string;
  capability_label: string;
  is_free: boolean;
  input_modalities: string[];
  output_modalities: string[];
  pricing_prompt: string;
  pricing_completion: string;
  context_length: number | null;
}

export interface SemanticModelerModelListResponse {
  success: boolean;
  provider: string;
  endpoint: string;
  models: SemanticModelerModelInfo[];
  total_count: number;
  filtered_count: number;
  error: string | null;
}
