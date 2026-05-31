import type { CanvasDetail, CandidateResponse, VirtualModelCandidate } from '../api/types';
import { DYNAMIC_CONTENT_ZONES, FIXED_CONTROL_ROLES } from '../constants/canvasZones';
import { t } from '../i18n/zh';

export type CandidateKind = 'fixed' | 'dynamic' | 'container' | 'ignored';

export interface CandidateEdit {
  label?: string;
  semanticRole?: string;
  visualType?: string;
  regionId?: string;
  kind?: CandidateKind;
  relativeBounds?: number[];
}

function asNumber(value: unknown, fallback = 0): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback;
}

export function normalizeRelativeBounds(bounds: unknown): number[] {
  if (!Array.isArray(bounds) || bounds.length < 4) return [];
  const [x1, y1, x2, y2] = bounds.map((value) => asNumber(value));
  if (x2 <= x1 || y2 <= y1) return [];
  return [x1, y1, x2, y2];
}

export function normalizeModelCandidate(candidate: VirtualModelCandidate): VirtualModelCandidate {
  return {
    ...candidate,
    key_id: candidate.key_id || candidate.element_id || 'unknown-candidate',
    canonical_text: candidate.canonical_text ?? '',
    canonical_role: candidate.canonical_role ?? 'unknown',
    canonical_region: candidate.canonical_region ?? 'unassigned',
    semantic_tags: Array.isArray(candidate.semantic_tags) ? candidate.semantic_tags : [],
    visual_type: candidate.visual_type ?? '',
    relative_bounds: normalizeRelativeBounds(candidate.relative_bounds),
    absolute_bounds: Array.isArray(candidate.absolute_bounds) ? candidate.absolute_bounds : null,
    confidence: asNumber(candidate.confidence),
    provider_sources: Array.isArray(candidate.provider_sources) ? candidate.provider_sources : [],
    is_fixed_control: Boolean(candidate.is_fixed_control),
    verify_count: asNumber(candidate.verify_count),
    seen_count: asNumber(candidate.seen_count),
    permanence_state: candidate.permanence_state || 'new',
  };
}

function getCanvasSize(canvas: CanvasDetail): { width: number; height: number } {
  let width = canvas.screenshot_width || canvas.window_width || 0;
  let height = canvas.screenshot_height || canvas.window_height || 0;

  if (width > 0 && height > 0) return { width, height };

  for (const element of canvas.elements) {
    if (!element.bounds || element.bounds.length < 4) continue;
    width = Math.max(width, element.bounds[2]);
    height = Math.max(height, element.bounds[3]);
  }

  return { width: Math.max(width, 1), height: Math.max(height, 1) };
}

export function getCanvasReferenceSize(canvas: CanvasDetail): { width: number; height: number } {
  return getCanvasSize(canvas);
}

export function getCandidateKeyFromElement(
  element: CandidateResponse,
  canvasId?: string | null,
  pageModelId?: string | null,
): string {
  if (element.stable_key_id) {
    return pageModelId
      ? `stable:${pageModelId}:${element.stable_key_id}`
      : `stable:${element.stable_key_id}`;
  }
  const scope = canvasId || 'unknown-canvas';
  return `canvas:${scope}:${element.element_id}`;
}

export function scopeVirtualModelCandidate(
  candidate: VirtualModelCandidate,
  stateTemplateId?: string | null,
  pageModelId?: string | null,
): VirtualModelCandidate {
  if (!candidate.key_id.startsWith('transient_')) {
    if (
      candidate.key_id.startsWith('stable:') ||
      candidate.key_id.startsWith('canvas:') ||
      candidate.key_id.startsWith('state:')
    ) {
      return candidate;
    }
    return {
      ...candidate,
      key_id: pageModelId ? `stable:${pageModelId}:${candidate.key_id}` : `stable:${candidate.key_id}`,
    };
  }
  const scope = stateTemplateId || 'unknown-state';
  return {
    ...candidate,
    key_id: `state:${scope}:${candidate.key_id}`,
  };
}

function relativeToAbsolute(bounds: number[], canvas: CanvasDetail): number[] | null {
  if (bounds.length < 4) return null;
  const { width, height } = getCanvasSize(canvas);
  return [
    Math.round(bounds[0] * width),
    Math.round(bounds[1] * height),
    Math.round(bounds[2] * width),
    Math.round(bounds[3] * height),
  ];
}

function toRelativeBounds(element: CandidateResponse, canvas: CanvasDetail): number[] {
  if (!element.bounds || element.bounds.length < 4) return [0, 0, 0, 0];

  const { width, height } = getCanvasSize(canvas);
  const [x1, y1, x2, y2] = element.bounds;

  return [
    Math.max(0, Math.min(1, x1 / width)),
    Math.max(0, Math.min(1, y1 / height)),
    Math.max(0, Math.min(1, x2 / width)),
    Math.max(0, Math.min(1, y2 / height)),
  ];
}

function isFixedWorkbenchCandidate(element: CandidateResponse): boolean {
  const role = element.role_label || element.semantic_role || element.visual_type || element.control_type || '';
  const region = element.region_id || '';
  if (region && DYNAMIC_CONTENT_ZONES.has(region)) return false;
  if (element.semantic_role === 'unknown' || element.semantic_role === 'text') return false;
  return Boolean(element.stable_key_id) || FIXED_CONTROL_ROLES.has(role) || FIXED_CONTROL_ROLES.has(element.semantic_role);
}

export function canvasElementToModelCandidate(
  element: CandidateResponse,
  canvas: CanvasDetail,
): VirtualModelCandidate {
  const role = element.role_label || element.semantic_role || element.visual_type || element.control_type || 'unknown';
  const text = element.text || element.name || '';
  const key = getCandidateKeyFromElement(element, canvas.canvas_id, canvas.page_model_id);

  return {
    key_id: key,
    element_id: element.element_id,
    is_canvas_candidate: true,
    canonical_text: text,
    canonical_role: role,
    canonical_region: element.region_id || 'unassigned',
    role_label: element.role_label || element.semantic_role || null,
    semantic_tags: element.semantic_tags || [],
    visual_type: element.visual_type || element.control_type || '',
    relative_bounds: toRelativeBounds(element, canvas),
    absolute_bounds: element.bounds,
    confidence: element.confidence,
    provider_sources: element.provider_sources || [],
    is_fixed_control: isFixedWorkbenchCandidate(element),
    verify_count: 0,
    seen_count: 1,
    permanence_state: 'new',
  };
}

export function canvasToModelCandidates(canvas: CanvasDetail | null | undefined): VirtualModelCandidate[] {
  if (!canvas) return [];
  return canvas.elements.map((element) => canvasElementToModelCandidate(element, canvas));
}

export function isLargeLayoutCandidate(candidate: VirtualModelCandidate): boolean {
  const bounds = normalizeRelativeBounds(candidate.relative_bounds);
  if (bounds.length < 4) return false;
  const [x1, y1, x2, y2] = bounds;
  const area = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  return area > 0.65;
}

export function getCandidateKind(
  candidate: VirtualModelCandidate,
  edit?: CandidateEdit,
): CandidateKind {
  if (edit?.kind) return edit.kind;
  if (isLargeLayoutCandidate(candidate)) return 'container';
  if (DYNAMIC_CONTENT_ZONES.has(candidate.canonical_region)) return 'dynamic';
  return candidate.is_fixed_control ? 'fixed' : 'dynamic';
}

export function isModelCandidateDynamic(
  candidate: VirtualModelCandidate,
  edits?: Map<string, CandidateEdit> | Set<string>,
): boolean {
  if (edits instanceof Set) {
    if (edits.has(candidate.key_id)) return true;
    if (DYNAMIC_CONTENT_ZONES.has(candidate.canonical_region)) return true;
    return !candidate.is_fixed_control;
  }
  return getCandidateKind(candidate, edits?.get(candidate.key_id)) === 'dynamic';
}

export function getModelCandidateLabel(
  candidate: VirtualModelCandidate,
  edits?: Map<string, CandidateEdit> | Map<string, string>,
): string {
  const edit = edits?.get(candidate.key_id);
  if (typeof edit === 'string') return edit;
  if (edit?.label) return edit.label;
  return (
    candidate.role_label ||
    candidate.canonical_text ||
    t(candidate.canonical_role, candidate.canonical_role) ||
    candidate.canonical_role ||
    '候选'
  );
}

export function getModelAspectRatio(
  canvas: CanvasDetail | null | undefined,
  fallback = 0.75,
): number {
  if (!canvas) return fallback;
  const { width, height } = getCanvasSize(canvas);
  return width > 0 && height > 0 ? height / width : fallback;
}

export function applyEditToModelCandidate(
  candidate: VirtualModelCandidate,
  edit?: CandidateEdit,
): VirtualModelCandidate {
  const normalized = normalizeModelCandidate(candidate);
  if (!edit) return normalized;
  const kind = edit.kind;
  return {
    ...normalized,
    role_label: edit.label ?? normalized.role_label,
    canonical_role: edit.semanticRole ?? normalized.canonical_role,
    visual_type: edit.visualType ?? normalized.visual_type,
    canonical_region: edit.regionId ?? normalized.canonical_region,
    relative_bounds: normalizeRelativeBounds(edit.relativeBounds ?? normalized.relative_bounds),
    is_fixed_control: kind ? kind === 'fixed' : normalized.is_fixed_control,
  };
}

export function applyEditToCanvasElement(
  element: CandidateResponse,
  canvas: CanvasDetail,
  edit?: CandidateEdit,
): CandidateResponse {
  if (!edit) return element;
  const bounds = edit.relativeBounds ? relativeToAbsolute(edit.relativeBounds, canvas) : element.bounds;
  const roleChanged = Boolean(edit.semanticRole || edit.label || edit.visualType || edit.kind || edit.regionId || edit.relativeBounds);
  return {
    ...element,
    semantic_role: edit.semanticRole ?? element.semantic_role,
    role_label: edit.label ?? element.role_label,
    visual_type: edit.visualType ?? element.visual_type,
    region_id: edit.regionId ?? element.region_id,
    bounds,
    refine_status: roleChanged ? 'refined' : element.refine_status,
    role_source: roleChanged ? 'manual' : element.role_source,
    role_confidence: roleChanged ? 1 : element.role_confidence,
    attributes: {
      ...element.attributes,
      ...(edit.kind ? { manual_kind: edit.kind } : {}),
      ...(edit.relativeBounds ? { manual_relative_bounds: edit.relativeBounds } : {}),
    },
  };
}
