/** API client for OpenClaw Console. */

import type {
  CanvasSummary,
  CanvasDetail,
  WindowListItem,
  ObserveResponse,
  QueryTargetModel,
  QueryResponse,
  RefineResponse,
  PageModelResponse,
  StateTemplateResponse,
  PageModelTreeResponse,
  VirtualModelDetail,
  CandidateOverrideResponse,
  CandidateOverrideUpsertRequest,
  SemanticModelerModelListResponse,
  SemanticModelerSettings,
  SemanticModelerSettingsUpdate,
  SemanticModelerTestResponse,
  VLMSemanticResponse,
  RoiVlmSupplementResponse,
  CanvasProcessingResponse,
  JobStatusResponse,
  ModelTemplatesResponse,
  ControlTransitionGraphResponse,
  TransitionGraphResponse,
} from './types';

const BASE = '/api/v1';

async function request<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, init);
  if (!res.ok) {
    const text = await res.text().catch(() => res.statusText);
    throw new Error(`API ${res.status}: ${text}`);
  }
  return res.json();
}

export async function fetchWindows(): Promise<WindowListItem[]> {
  return request<WindowListItem[]>(`${BASE}/windows`);
}

export async function fetchCanvases(): Promise<CanvasSummary[]> {
  return request<CanvasSummary[]>(`${BASE}/canvases`);
}

export async function fetchCanvasDetail(canvasId: string): Promise<CanvasDetail> {
  return request<CanvasDetail>(`${BASE}/canvases/${encodeURIComponent(canvasId)}`);
}

export async function deleteCanvas(canvasId: string): Promise<{ deleted: boolean; canvas_id: string }> {
  return request<{ deleted: boolean; canvas_id: string }>(
    `${BASE}/canvases/${encodeURIComponent(canvasId)}`,
    { method: 'DELETE' },
  );
}

export async function fetchCanvasProcessing(canvasId: string): Promise<CanvasProcessingResponse> {
  return request<CanvasProcessingResponse>(`${BASE}/canvases/${encodeURIComponent(canvasId)}/processing`);
}

export async function fetchJobStatus(jobId: string): Promise<JobStatusResponse> {
  return request<JobStatusResponse>(`${BASE}/jobs/${encodeURIComponent(jobId)}`);
}

export async function cancelJob(jobId: string): Promise<JobStatusResponse> {
  return request<JobStatusResponse>(`${BASE}/jobs/${encodeURIComponent(jobId)}/cancel`, {
    method: 'POST',
  });
}

export async function fetchJobs(params: { canvasId?: string | null; limit?: number } = {}): Promise<JobStatusResponse[]> {
  const search = new URLSearchParams();
  if (params.canvasId) search.set('canvas_id', params.canvasId);
  if (params.limit) search.set('limit', String(params.limit));
  const qs = search.toString();
  return request<JobStatusResponse[]>(`${BASE}/jobs${qs ? `?${qs}` : ''}`);
}

export function getCanvasScreenshotUrl(
  canvasId: string,
  overlay: boolean,
  layers?: string[],
): string {
  const params = new URLSearchParams();
  if (overlay) params.set('overlay', 'true');
  if (layers && layers.length > 0) params.set('layers', layers.join(','));
  const qs = params.toString();
  return `${BASE}/canvases/${encodeURIComponent(canvasId)}/screenshot${qs ? `?${qs}` : ''}`;
}

export function getCandidateCropUrl(
  canvasId: string,
  elementId: string,
  padding = 4,
): string {
  return `${BASE}/canvases/${encodeURIComponent(canvasId)}/candidates/${encodeURIComponent(elementId)}/crop?padding=${padding}`;
}

export async function observeWindow(
  hwnd: number,
  allowVlm = false,
  forceVlm = false,
  asyncEnhance = true,
): Promise<ObserveResponse> {
  return request<ObserveResponse>(`${BASE}/observe`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      hwnd,
      allow_vlm: allowVlm,
      force_vlm: forceVlm,
      async_enhance: asyncEnhance,
    }),
  });
}

export async function queryCanvas(
  canvasId: string,
  target: QueryTargetModel,
  maxResults = 20,
): Promise<QueryResponse> {
  return request<QueryResponse>(`${BASE}/query`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ canvas_id: canvasId, target, max_results: maxResults }),
  });
}

export async function refineCanvas(
  canvasId: string,
  candidateIds?: string[],
  mode = 'heuristic',
): Promise<RefineResponse> {
  return request<RefineResponse>(`${BASE}/canvases/${encodeURIComponent(canvasId)}/refine`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ candidate_ids: candidateIds ?? null, mode }),
  });
}

// ── Page Model API (E Phase 2) ──

export async function fetchPageModels(appId?: string): Promise<PageModelResponse[]> {
  const params = appId ? `?app_id=${encodeURIComponent(appId)}` : '';
  return request<PageModelResponse[]>(`${BASE}/page-models${params}`);
}

export async function deletePageModel(pageModelId: string): Promise<{ deleted: boolean; page_model_id: string }> {
  return request<{ deleted: boolean; page_model_id: string }>(
    `${BASE}/page-models/${encodeURIComponent(pageModelId)}`,
    { method: 'DELETE' },
  );
}

export async function fetchStateTemplates(pageModelId: string): Promise<StateTemplateResponse[]> {
  return request<StateTemplateResponse[]>(
    `${BASE}/page-models/${encodeURIComponent(pageModelId)}/state-templates`,
  );
}

export async function deleteStateTemplate(stateTemplateId: string): Promise<{ deleted: boolean; state_template_id: string }> {
  return request<{ deleted: boolean; state_template_id: string }>(
    `${BASE}/state-templates/${encodeURIComponent(stateTemplateId)}`,
    { method: 'DELETE' },
  );
}

export async function fetchPageModelTree(): Promise<PageModelTreeResponse> {
  return request<PageModelTreeResponse>(`${BASE}/page-models/tree`);
}

export async function fetchVirtualModel(stateTemplateId: string): Promise<VirtualModelDetail> {
  return request<VirtualModelDetail>(`${BASE}/state-templates/${encodeURIComponent(stateTemplateId)}/virtual-model`);
}

export async function fetchStateTemplateTransitions(stateTemplateId: string): Promise<TransitionGraphResponse> {
  return request<TransitionGraphResponse>(
    `${BASE}/state-templates/${encodeURIComponent(stateTemplateId)}/transitions`,
  );
}

export async function fetchControlTransitions(params: {
  candidateKey?: string | null;
  pageClass?: string | null;
}): Promise<ControlTransitionGraphResponse> {
  const search = new URLSearchParams();
  if (params.candidateKey) search.set('candidate_key', params.candidateKey);
  if (params.pageClass) search.set('page_class', params.pageClass);
  const qs = search.toString();
  return request<ControlTransitionGraphResponse>(`${BASE}/control-transitions${qs ? `?${qs}` : ''}`);
}

export async function fetchStateTemplateModelTemplates(stateTemplateId: string): Promise<ModelTemplatesResponse> {
  return request<ModelTemplatesResponse>(
    `${BASE}/state-templates/${encodeURIComponent(stateTemplateId)}/model-templates`,
  );
}

export async function fetchCandidateOverrides(params: {
  stateTemplateId?: string | null;
  canvasId?: string | null;
  pageModelId?: string | null;
}): Promise<CandidateOverrideResponse[]> {
  const search = new URLSearchParams();
  if (params.stateTemplateId) search.set('state_template_id', params.stateTemplateId);
  if (params.canvasId) search.set('canvas_id', params.canvasId);
  if (params.pageModelId) search.set('page_model_id', params.pageModelId);
  const qs = search.toString();
  return request<CandidateOverrideResponse[]>(`${BASE}/candidate-overrides${qs ? `?${qs}` : ''}`);
}

export async function upsertCandidateOverride(
  body: CandidateOverrideUpsertRequest,
): Promise<CandidateOverrideResponse> {
  return request<CandidateOverrideResponse>(`${BASE}/candidate-overrides`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
}

export async function deleteCandidateOverride(scopeKey: string): Promise<{ deleted: boolean; scope_key: string }> {
  return request<{ deleted: boolean; scope_key: string }>(
    `${BASE}/candidate-overrides/${encodeURIComponent(scopeKey)}`,
    { method: 'DELETE' },
  );
}

// ── VLM Semantic ──

export async function triggerVLMSemantic(
  canvasId: string,
  force = false,
  sync = false,
  taskMode = 'candidate_annotation',
  candidateIds: string[] = [],
): Promise<VLMSemanticResponse> {
  return request<VLMSemanticResponse>(`${BASE}/vlm-semantic`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ canvas_id: canvasId, force, sync, task_mode: taskMode, candidate_ids: candidateIds }),
  });
}

export async function triggerRoiVlmSupplement(
  canvasId: string,
  roiIds: string[] = [],
  deadlineMs = 2000,
): Promise<RoiVlmSupplementResponse> {
  return request<RoiVlmSupplementResponse>(`${BASE}/canvases/${encodeURIComponent(canvasId)}/roi-vlm`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      dry_run: false,
      roi_ids: roiIds,
      deadline_ms: deadlineMs,
      accept_late: true,
    }),
  });
}

// ── VLM Semantic Settings ──

export async function fetchSemanticModelerSettings(): Promise<SemanticModelerSettings> {
  return request<SemanticModelerSettings>(`${BASE}/settings/semantic-modeler`);
}

export async function saveSemanticModelerSettings(
  body: SemanticModelerSettingsUpdate,
): Promise<SemanticModelerSettings> {
  return request<SemanticModelerSettings>(`${BASE}/settings/semantic-modeler`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
}

export async function testSemanticModelerSettings(
  body: SemanticModelerSettingsUpdate & { run_call?: boolean },
): Promise<SemanticModelerTestResponse> {
  return request<SemanticModelerTestResponse>(`${BASE}/settings/semantic-modeler/test`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
}

export async function fetchSemanticModelerModels(
  body: SemanticModelerSettingsUpdate & { vision_only?: boolean },
): Promise<SemanticModelerModelListResponse> {
  return request<SemanticModelerModelListResponse>(`${BASE}/settings/semantic-modeler/models`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
}
