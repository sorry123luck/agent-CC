"""API request/response models for the InteractionCanvas protocol."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


# ===== Unified response envelope =====

class ApiResponse(BaseModel):
    """Unified API response envelope."""
    success: bool
    data: Any = None
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


# ===== observe =====

class ObserveRequest(BaseModel):
    """Request to observe a window and generate an InteractionCanvas."""
    hwnd: int | None = None          # Window handle (default: foreground)
    allow_vlm: bool = False          # Allow internal VLM calls
    force_vlm: bool = False          # Force VLM even if local is sufficient
    detail_level: str = "summary"    # summary / full
    async_enhance: bool = True       # Return local canvas first, enhance in background
    vlm_modes: list[str] = Field(default_factory=list)
    capture_mode: str = "window"     # window / screen_region


class GeometricRegionResponse(BaseModel):
    """Neutral geometric region from Layer 1 partitioner."""
    region_id: str
    bounds: list[int] = Field(default_factory=list)
    boundary_evidence: list[str] = Field(default_factory=list)
    density_profile: list[float] = Field(default_factory=list)
    geometry_confidence: float = 0.0
    parent_region_id: str | None = None


class ObserveResponse(BaseModel):
    """Response from observe."""
    canvas_id: str
    elapsed_ms: int = 0
    app_id: str | None = None
    page_class: str = ""
    surface_type: str = ""
    stable: bool = True
    loading: bool = False
    partial: bool = False
    confidence: float = 0.0
    element_count: int = 0
    region_count: int = 0
    providers_used: list[str] = Field(default_factory=list)
    providers_failed: list[str] = Field(default_factory=list)
    canvas_schema_version: str = "1.0"
    page_model_id: str | None = None
    state_template_id: str | None = None
    model_match_status: str = ""  # "reused" | "new_page" | "new_state"
    vlm_semantic_used: bool = False
    vlm_semantic_status: str = ""  # "success" | "cached" | "skipped" | "failed"
    vlm_semantic_provider: str = ""
    processing_state: str = "local_ready"
    enhancement_job_id: str | None = None
    geometric_region_count: int = 0
    fusion_diagnostics: dict[str, Any] = Field(default_factory=dict)
    perception_quality: dict[str, Any] = Field(default_factory=dict)
    visual_pattern: dict[str, Any] = Field(default_factory=dict)
    roi_selection_plan: dict[str, Any] = Field(default_factory=dict)


# ===== query =====

class QueryTargetModel(BaseModel):
    """Query target specification."""
    text: str | None = None
    semantic_role: str | None = None
    region: str | None = None
    natural_language: str | None = None
    composite: dict[str, Any] | None = None


class QueryRequest(BaseModel):
    """Request to query candidates from a canvas."""
    canvas_id: str
    target: QueryTargetModel
    max_results: int = 5
    min_confidence: float = 0.2


class CandidateResponse(BaseModel):
    """A single candidate in a query response."""
    element_id: str
    semantic_role: str = ""
    text: str = ""
    name: str | None = None
    control_type: str = ""
    bounds: list[int] | None = None
    click_point: list[int] | None = None
    confidence: float = 0.0
    confidence_level: str = "medium"
    risk_tags: list[str] = Field(default_factory=list)
    risk_level: str = "L0"
    provider_sources: list[str] = Field(default_factory=list)
    interactable: bool = True
    from_memory: bool = False
    suggest_confirm: bool = False
    region_id: str | None = None
    locator_ids: list[str] = Field(default_factory=list)
    attributes: dict[str, Any] = Field(default_factory=dict)
    # Refine fields (三层可扩展设计)
    visual_type: str = ""
    semantic_tags: list[str] = Field(default_factory=list)
    role_label: str | None = None
    role_confidence: float = 0.0
    role_source: str = ""
    role_evidence: list[str] = Field(default_factory=list)
    refine_status: str = "unreviewed"
    stable_key_id: str | None = None


class QueryResponse(BaseModel):
    """Response from query."""
    candidates: list[CandidateResponse] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)
    total_matched: int = 0


class ReadRegionRequest(BaseModel):
    """Request to read text evidence from a canvas region."""
    region_id: str | None = None
    region_role: str | None = None
    method: str | None = None
    include_ocr: bool = True
    include_elements: bool = True
    allow_crop_ocr: bool = False


class ReadRegionTextBlock(BaseModel):
    """One text evidence block harvested from a region."""
    text: str
    bounds: list[int] = Field(default_factory=list)
    confidence: float = 0.0
    source: str = ""
    element_id: str | None = None


class ReadRegionResponse(BaseModel):
    """Response from read-region."""
    canvas_id: str
    region_id: str
    region_role: str = ""
    method: str = "region_text_harvest"
    read_scope: str = "current_viewport"
    scroll_context: dict[str, Any] | None = None
    long_content_strategy: dict[str, Any] = Field(default_factory=dict)
    status: str = "pass"
    text: str = ""
    text_blocks: list[ReadRegionTextBlock] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    crop_ocr_used: bool = False
    suggested_next_actions: list[dict[str, Any]] = Field(default_factory=list)


class ScrollRegionRequest(BaseModel):
    """Request a read-only scroll plan for a canvas region."""
    region_id: str | None = None
    region_role: str | None = None
    direction: str = "down"
    amount: str = "page"
    dry_run: bool = True
    execute_confirmed: bool = False
    observe_after: bool = True
    read_after: bool = True
    allow_crop_ocr_after: bool = False


class ScrollRegionResponse(BaseModel):
    """Response from scroll-region."""
    canvas_id: str
    region_id: str
    region_role: str = ""
    dry_run: bool = True
    status: str = "planned"
    can_execute: bool = False
    scroll_context: dict[str, Any] | None = None
    action_plan: dict[str, Any] = Field(default_factory=dict)
    execution_result: dict[str, Any] = Field(default_factory=dict)
    after_canvas_id: str | None = None
    before_read: dict[str, Any] | None = None
    after_read: dict[str, Any] | None = None
    stitched_text: str = ""
    stitch_report: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


# ===== diff =====

class DiffRequest(BaseModel):
    """Request to compare two canvas snapshots."""
    hwnd: int
    previous_canvas_id: str
    detail_level: str = "changes"    # summary / changes / full


class DiffResponse(BaseModel):
    """Response from diff."""
    new_canvas_id: str
    added: list[CandidateResponse] = Field(default_factory=list)
    removed: list[CandidateResponse] = Field(default_factory=list)
    preserved: list[CandidateResponse] = Field(default_factory=list)
    page_changed: bool = False
    page_class_changed: bool = False
    summary: str = ""


# ===== remember =====

class RememberRequest(BaseModel):
    """Request to confirm memory of a canvas."""
    canvas_id: str
    template_id: str | None = None   # Update existing template (None = create new)
    confirm: bool = True


class RememberResponse(BaseModel):
    """Response from remember."""
    template_id: str
    status: str = "created"   # created / updated / rejected


# ===== feedback =====

class FeedbackRequest(BaseModel):
    """Agent feedback on a candidate."""
    canvas_id: str
    candidate_key: str
    feedback_type: str      # success / failure / unclear
    detail: dict[str, Any] | None = None


class FeedbackResponse(BaseModel):
    """Response from feedback."""
    recorded: bool = True
    confidence_update: float | None = None


# ===== act =====

class ActRequest(BaseModel):
    """Request to execute an action on a candidate."""
    canvas_id: str | None = None
    candidate_id: str
    action: str             # click / double_click / type_text / key_press / etc.
    params: dict[str, Any] = Field(default_factory=dict)
    dry_run: bool = True
    execute_confirmed: bool = False


class ActResponse(BaseModel):
    """Response from act."""
    execution_result: str = "success"  # success / failed / timeout / blocked
    error_message: str | None = None
    triggered_diff: DiffResponse | None = None
    verification_result: dict[str, Any] = Field(default_factory=dict)
    can_execute: bool = False
    action_plan: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


# ===== capabilities =====

class CapabilitiesResponse(BaseModel):
    """Software capabilities."""
    version: str = "1.0"
    supported_actions: list[str] = Field(default_factory=list)
    action_policy: dict[str, Any] = Field(default_factory=dict)
    supported_query_types: list[str] = Field(default_factory=list)
    supported_canvas_region_apis: list[str] = Field(default_factory=list)
    supported_memory_apis: list[str] = Field(default_factory=list)
    providers: list[str] = Field(default_factory=list)
    surface_types: list[str] = Field(default_factory=list)


# ===== windows/resolve =====

class WindowResolveRequest(BaseModel):
    """Request to resolve windows for an app."""
    app_name: str
    prefer_session: bool = True


class WindowCandidateResponse(BaseModel):
    """A candidate window."""
    hwnd: int
    title: str
    process_name: str | None = None
    process_id: int | None = None
    is_foreground: bool = False
    source: str = "existing"  # current_session / existing
    confidence: float = 0.0


class LaunchTargetResponse(BaseModel):
    """A possible launch path for an app."""
    app_name: str
    executable: str
    args: list[str] = []
    description: str = ""


class WindowResolveResponse(BaseModel):
    """Response from windows/resolve."""
    app_name: str
    found_window: bool
    candidates: list[WindowCandidateResponse] = Field(default_factory=list)
    launch_targets: list[LaunchTargetResponse] = Field(default_factory=list)
    suggested_next_step: str = ""


# ===== Console: canvas browsing =====

class CanvasSummary(BaseModel):
    """Summary of a cached canvas for the canvas browser."""
    canvas_id: str
    app_id: str | None = None
    display_name: str | None = None  # 软件中文名
    window_title: str = ""
    process_name: str | None = None
    page_class: str = ""
    surface_type: str = ""
    element_count: int = 0
    region_count: int = 0
    captured_at: str = ""
    providers_used: list[str] = Field(default_factory=list)
    stable: bool = True
    loading: bool = False
    partial: bool = False
    has_screenshot: bool = False
    page_model_id: str | None = None
    state_template_id: str | None = None
    processing_state: str = "local_ready"
    processing_error: str = ""


class RegionResponse(BaseModel):
    """A region in the canvas detail response."""
    region_id: str
    role: str = ""
    subtype: str = ""
    bounds: list[int] | None = None
    element_count: int = 0


class ProviderTraceResponse(BaseModel):
    """Provider trace detail."""
    uia_used: bool = False
    ocr_used: bool = False
    vision_used: bool = False
    vlm_used: bool = False
    dom_used: bool = False
    provider_details: dict[str, Any] = Field(default_factory=dict)


class CanvasDetail(BaseModel):
    """Full canvas detail for the console."""
    canvas_id: str
    app_id: str | None = None
    window_title: str = ""
    page_class: str = ""
    surface_type: str = ""
    stable: bool = True
    loading: bool = False
    partial: bool = False
    captured_at: str = ""
    window_width: int = 0
    window_height: int = 0
    screenshot_width: int = 0
    screenshot_height: int = 0
    elements: list[CandidateResponse] = Field(default_factory=list)
    regions: list[RegionResponse] = Field(default_factory=list)
    geometric_regions: list[GeometricRegionResponse] = Field(default_factory=list)
    fusion_diagnostics: dict[str, Any] = Field(default_factory=dict)
    perception_quality: dict[str, Any] = Field(default_factory=dict)
    visual_pattern: dict[str, Any] = Field(default_factory=dict)
    roi_selection_plan: dict[str, Any] = Field(default_factory=dict)
    ocr_blocks: list[dict[str, Any]] = Field(default_factory=list)
    vision_candidates: list[dict[str, Any]] = Field(default_factory=list)
    roi_vlm_semantic_supplements: list[dict[str, Any]] = Field(default_factory=list)
    roi_vlm_timeouts: list[dict[str, Any]] = Field(default_factory=list)
    roi_vlm_late_failures: list[dict[str, Any]] = Field(default_factory=list)
    roi_vlm_rejected_responses: list[dict[str, Any]] = Field(default_factory=list)
    observe_timing: dict[str, Any] = Field(default_factory=dict)
    provider_trace: ProviderTraceResponse | None = None
    providers_used: list[str] = Field(default_factory=list)
    providers_failed: list[str] = Field(default_factory=list)
    has_screenshot: bool = False
    page_model_id: str | None = None
    state_template_id: str | None = None
    page_model_name: str | None = None
    state_label: str | None = None
    processing_state: str = "local_ready"
    processing_error: str = ""
    processing_updated_at: str = ""


class WindowListItem(BaseModel):
    """A visible window for the window browser."""
    hwnd: int
    title: str
    class_name: str | None = None
    process_name: str | None = None
    process_id: int | None = None
    is_foreground: bool = False
    is_minimized: bool = False


# ===== refine =====

class RefineResultItem(BaseModel):
    """Refine result for a single candidate."""
    element_id: str
    visual_type: str = "unknown"
    semantic_tags: list[str] = Field(default_factory=list)
    role_label: str | None = None
    role_confidence: float = 0.0
    role_source: str = ""
    role_evidence: list[str] = Field(default_factory=list)
    refine_status: str = "unreviewed"


class RefineRequest(BaseModel):
    """Request to refine candidates in a canvas."""
    candidate_ids: list[str] | None = None  # None = refine all
    mode: str = "heuristic"  # heuristic / vlm / agent
    include_crops: bool = True
    results: list[RefineResultItem] | None = None  # Agent 传入的分类结果（mode=agent 时使用）


class RefineResponse(BaseModel):
    """Response from refine."""
    canvas_id: str
    mode: str = "heuristic"
    results: list[RefineResultItem] = Field(default_factory=list)
    total_refined: int = 0
    total_uncertain: int = 0
    total_unreviewed: int = 0


# ===== E Phase 2: Page Model =====

class PageModelResponse(BaseModel):
    """页面模型响应。"""
    page_model_id: str
    app_id: str
    page_class_prefix: str
    display_name: str | None = None
    surface_type: str | None = None
    observe_count: int = 0
    state_count: int = 0
    last_seen_at: str = ""


class StateTemplateResponse(BaseModel):
    """页面状态模板响应。"""
    state_template_id: str
    page_model_id: str
    page_class: str
    state_label: str | None = None
    fixed_element_count: int = 0
    total_element_count: int = 0
    verify_count: int = 0
    snapshot_count: int = 0
    last_seen_at: str = ""


class CanvasSnapshotResponse(BaseModel):
    """画布快照归属响应。"""
    snapshot_id: str
    canvas_id: str
    page_model_id: str
    state_template_id: str
    captured_at: str
    element_count: int = 0
    has_screenshot: bool = False
    available: bool = False  # canvas 仍在内存缓存中可查看详情


class PageModelTreeResponse(BaseModel):
    """完整的软件模型树。"""
    page_models: list[PageModelResponse] = Field(default_factory=list)
    state_templates: list[StateTemplateResponse] = Field(default_factory=list)
    canvas_snapshots: list[CanvasSnapshotResponse] = Field(default_factory=list)


# ===== E Phase 2: Virtual Model (persistent, no CanvasCache dependency) =====

class VirtualModelCandidate(BaseModel):
    """A single candidate in the persistent virtual model."""
    key_id: str
    canonical_text: str = ""
    canonical_role: str = ""
    canonical_region: str = ""
    role_label: str | None = None
    semantic_tags: list[str] = Field(default_factory=list)
    visual_type: str = ""
    relative_bounds: list[float] = Field(default_factory=list)  # [x1/W, y1/H, x2/W, y2/H]
    confidence: float = 0.0
    provider_sources: list[str] = Field(default_factory=list)
    is_fixed_control: bool = False
    verify_count: int = 1
    seen_count: int = 1
    permanence_state: str = "new"  # "new"|"provisional"|"stable"|"fixed_anchor"|"stale"|"degraded"|"retired"
    confidence_profile: dict | None = None  # Phase 9: 7-dimension confidence breakdown
    source: str = ""  # "persistent" | "vlm" | "override" | "canvas_transient"


class VirtualModelDetail(BaseModel):
    """Persistent virtual model for a StateTemplate — reads from SQLite only."""
    state_template_id: str
    page_model_id: str
    app_id: str
    display_name: str | None = None
    page_class: str = ""
    state_label: str | None = None
    surface_type: str | None = None
    layout_fingerprint: dict[str, Any] = Field(default_factory=dict)
    fixed_element_count: int = 0
    total_element_count: int = 0
    verify_count: int = 0
    snapshot_count: int = 0
    last_seen_at: str = ""
    has_available_canvas: bool = False
    candidates: list[VirtualModelCandidate] = Field(default_factory=list)
    regions: list[dict[str, Any]] = Field(default_factory=list)
    vlm_semantic_model: dict[str, Any] | None = None
    model_layers: dict[str, Any] = Field(default_factory=dict)
    app_shell: dict[str, Any] = Field(default_factory=dict)
    region_templates: list[dict[str, Any]] = Field(default_factory=list)
    element_templates: list[dict[str, Any]] = Field(default_factory=list)
    visible_items: list[dict[str, Any]] = Field(default_factory=list)
    missing_suggestions: list[dict[str, Any]] = Field(default_factory=list)


class TransitionEdgeResponse(BaseModel):
    from_page_class: str
    to_page_class: str
    trigger_action: str | None = None
    trigger_candidate_key: str | None = None
    observe_count: int = 0
    success_count: int = 0
    success_rate: float = 0.0
    last_observed_at: str | None = None
    drifted: bool = False


class ControlTransitionResponse(BaseModel):
    candidate_key: str
    stable_key_id: str | None = None
    from_page_class: str
    to_page_class: str
    action_type: str | None = None
    observe_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    success_rate: float = 0.0
    last_observed_at: str | None = None
    canvas_id_before: str | None = None
    canvas_id_after: str | None = None


class ControlTransitionGraphResponse(BaseModel):
    page_class: str | None = None
    candidate_key: str | None = None
    transitions: list[ControlTransitionResponse] = Field(default_factory=list)
    total: int = 0


class ControlTransitionRecordRequest(BaseModel):
    candidate_key: str
    from_page_class: str
    to_page_class: str
    action_type: str | None = "click"
    stable_key_id: str | None = None
    canvas_id_before: str | None = None
    canvas_id_after: str | None = None
    success: bool = True
    metadata: dict[str, Any] | None = None


class TransitionGraphResponse(BaseModel):
    page_class: str | None = None
    candidate_key: str | None = None
    transitions: list[TransitionEdgeResponse] = Field(default_factory=list)
    total: int = 0


# ===== Candidate correction overlay =====

class CandidateOverrideUpsertRequest(BaseModel):
    """Create or update a candidate correction overlay."""
    scope_key: str
    app_id: str | None = None
    page_model_id: str | None = None
    state_template_id: str | None = None
    canvas_id: str | None = None
    element_id: str | None = None
    stable_key_id: str | None = None
    label: str | None = None
    semantic_role: str | None = None
    visual_type: str | None = None
    region_id: str | None = None
    kind: str | None = None
    relative_bounds: list[float] | None = None
    absolute_bounds: list[float] | None = None
    source: str = "manual"
    status: str = "active"


class CandidateOverrideResponse(BaseModel):
    """Persistent candidate correction overlay."""
    override_id: str
    scope_key: str
    app_id: str | None = None
    page_model_id: str | None = None
    state_template_id: str | None = None
    canvas_id: str | None = None
    element_id: str | None = None
    stable_key_id: str | None = None
    label: str | None = None
    semantic_role: str | None = None
    visual_type: str | None = None
    region_id: str | None = None
    kind: str | None = None
    relative_bounds: list[float] | None = None
    absolute_bounds: list[float] | None = None
    source: str = "manual"
    status: str = "active"
    created_at: str
    updated_at: str


# ===== VLM Semantic =====

class VLMSemanticRequest(BaseModel):
    """Request to trigger VLM semantic analysis on a canvas."""
    canvas_id: str
    force: bool = False
    sync: bool = False
    task_mode: str = "candidate_annotation"
    candidate_ids: list[str] = Field(default_factory=list)


class VLMSemanticResponse(BaseModel):
    """Response from VLM semantic analysis."""
    canvas_id: str
    status: str = ""  # "success" | "cached" | "failed" | "skipped"
    provider: str = ""
    model: str = ""
    error: str | None = None
    error_code: str | None = None
    message: str = ""
    latency_ms: int = 0
    token_input: int = 0
    token_output: int = 0
    from_cache: bool = False
    region_count: int = 0
    control_count: int = 0
    correction_count: int = 0
    dynamic_zone_count: int = 0
    warning_count: int = 0
    job_id: str | None = None
    processing_state: str = ""


class RoiVlmSupplementRequest(BaseModel):
    """Request to prepare or run ROI-scoped VLM semantic supplement."""
    roi_ids: list[str] = Field(default_factory=list)
    deadline_ms: int = 2000
    accept_late: bool = True
    dry_run: bool = True


class RoiVlmSupplementResponse(BaseModel):
    """Response for ROI-scoped VLM semantic supplement scheduling."""
    canvas_id: str
    status: str
    deadline_ms: int = 2000
    elapsed_ms: int = 0
    accept_late: bool = True
    jobs: list[dict[str, Any]] = Field(default_factory=list)
    results: list[dict[str, Any]] = Field(default_factory=list)
    result_status_counts: dict[str, int] = Field(default_factory=dict)
    message: str = ""


class LayoutAuditVlmRequest(BaseModel):
    """Request to prepare or run full-screenshot VLM layout audit."""
    deadline_ms: int = 2000
    accept_late: bool = True
    dry_run: bool = True


class LayoutAuditVlmResponse(BaseModel):
    """Response for VLM layout audit fallback."""
    canvas_id: str
    status: str
    deadline_ms: int = 2000
    elapsed_ms: int = 0
    accept_late: bool = True
    job: dict[str, Any] | None = None
    result: dict[str, Any] = Field(default_factory=dict)
    message: str = ""


class SemanticCompletionRequest(BaseModel):
    """Request to orchestrate quality-driven semantic completion."""
    deadline_ms: int = 2000
    accept_late: bool = True
    dry_run: bool = True
    roi_ids: list[str] = Field(default_factory=list)


class SemanticCompletionResponse(BaseModel):
    """Response for automatic layout-audit plus ROI semantic completion."""
    canvas_id: str
    status: str
    deadline_ms: int = 2000
    elapsed_ms: int = 0
    accept_late: bool = True
    stages: list[dict[str, Any]] = Field(default_factory=list)
    layout_audit: dict[str, Any] | None = None
    roi_vlm: dict[str, Any] | None = None
    next_action: str = ""
    message: str = ""


class JobStatusResponse(BaseModel):
    job_id: str
    canvas_id: str
    job_type: str
    state: str
    error: str = ""
    started_at: str = ""
    updated_at: str = ""


class CanvasProcessingResponse(BaseModel):
    canvas_id: str
    processing_state: str = "local_ready"
    jobs: list[str] = Field(default_factory=list)
    last_error: str = ""
    updated_at: str = ""
    current_phase: str = ""
    is_processing: bool = False
    latest_job: JobStatusResponse | None = None
    active_jobs: list[JobStatusResponse] = Field(default_factory=list)


# ===== VLM Semantic Settings =====

class SemanticModelerSettingsResponse(BaseModel):
    """Public runtime settings for VLM Semantic Modeler."""
    enabled: bool = False
    provider: str = "disabled"
    endpoint: str = ""
    model: str = ""
    fallback_provider: str = ""
    fallback_model: str = ""
    provider_variant: str = ""
    free_model_only: bool = False
    allow_model_fallback: bool = True
    timeout_seconds: int = 60
    max_retries: int = 2
    proxy_url: str = ""
    proxy_port: int = 0
    prompt_version: str = "1.0"
    daily_call_limit: int = 100
    monthly_budget_usd: float = 10.0
    priority: int = 1
    allow_free_models: bool = True
    cost_per_1k_input: float = 0.0
    cost_per_1k_output: float = 0.0
    save_raw_response: bool = False
    redact_dynamic_content: bool = True
    thinking_mode: str = "auto"
    image_max_width: int = 1280
    api_key_set: bool = False


class SemanticModelerSettingsUpdateRequest(BaseModel):
    """Update runtime settings for VLM Semantic Modeler.

    Omit api_key to preserve the currently saved key. Send api_key="" to clear it.
    """
    enabled: bool | None = None
    provider: str | None = None
    api_key: str | None = None
    endpoint: str | None = None
    model: str | None = None
    fallback_provider: str | None = None
    fallback_model: str | None = None
    provider_variant: str | None = None
    free_model_only: bool | None = None
    allow_model_fallback: bool | None = None
    timeout_seconds: int | None = None
    max_retries: int | None = None
    proxy_url: str | None = None
    proxy_port: int | None = None
    prompt_version: str | None = None
    daily_call_limit: int | None = None
    monthly_budget_usd: float | None = None
    priority: int | None = None
    allow_free_models: bool | None = None
    cost_per_1k_input: float | None = None
    cost_per_1k_output: float | None = None
    save_raw_response: bool | None = None
    redact_dynamic_content: bool | None = None
    thinking_mode: str | None = None
    image_max_width: int | None = None


class SemanticModelerTestRequest(SemanticModelerSettingsUpdateRequest):
    """Test a semantic modeler provider with current or supplied settings."""
    run_call: bool = False


class SemanticModelerTestResponse(BaseModel):
    """Result of a semantic modeler provider test."""
    success: bool
    available: bool = False
    provider: str = ""
    model: str = ""
    endpoint: str = ""
    message: str = ""
    error: str | None = None
    latency_ms: int = 0
    token_input: int = 0
    token_output: int = 0
    finish_reason: str = ""
    raw_preview: str = ""


class SemanticModelerModelListRequest(SemanticModelerSettingsUpdateRequest):
    """List models for a semantic modeler provider using current or supplied settings."""
    vision_only: bool = False


class SemanticModelerModelInfo(BaseModel):
    """A provider model available for VLM semantic modeling."""
    id: str
    name: str = ""
    provider: str = ""
    description: str = ""
    vision_capable: bool = False
    model_type: str = "unknown"  # text | image | multimodal | unknown
    capability_label: str = "未知"
    is_free: bool = False
    input_modalities: list[str] = Field(default_factory=list)
    output_modalities: list[str] = Field(default_factory=list)
    pricing_prompt: str = ""
    pricing_completion: str = ""
    context_length: int | None = None


class SemanticModelerModelListResponse(BaseModel):
    """Provider model list response."""
    success: bool
    provider: str = ""
    endpoint: str = ""
    models: list[SemanticModelerModelInfo] = Field(default_factory=list)
    total_count: int = 0
    filtered_count: int = 0
    error: str | None = None
