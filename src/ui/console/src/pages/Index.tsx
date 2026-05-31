import { useState, useCallback, useEffect, useMemo, useRef, type MouseEvent as ReactMouseEvent } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { ArrowLeft as ArrowLeftIcon, Sparkles } from 'lucide-react';
import TopBar from '../components/TopBar';
import WindowBrowser from '../components/WindowBrowser';
import CanvasBrowser from '../components/CanvasBrowser';
import ElementList from '../components/ElementList';
import ReviewCanvas from '../components/ReviewCanvas';
import ReviewPanel from '../components/ReviewPanel';
import BottomBar from '../components/BottomBar';
import QueryPanel from '../components/QueryPanel';
import VirtualModelRenderer from '../components/VirtualModelRenderer';
import ModelCandidatePanel from '../components/ModelCandidatePanel';
import ModelOverviewPanel from '../components/ModelOverviewPanel';
import VLMSettingsPanel from '../components/VLMSettingsPanel';
import { useCanvasDetail, useRoiVlmSupplement, useVLMSemantic } from '../hooks/useCanvases';
import {
  refineCanvas,
  fetchVirtualModel,
  fetchPageModelTree,
  fetchCandidateOverrides,
  fetchCanvasProcessing,
  cancelJob,
  fetchControlTransitions,
  fetchJobs,
  fetchStateTemplateTransitions,
  upsertCandidateOverride,
} from '../api/client';
import type { CandidateOverrideResponse, CandidateOverrideUpsertRequest, CanvasDetail, ObserveResponse, RoiVlmSupplementResponse, VLMSemanticResponse, VirtualModelCandidate } from '../api/types';
import {
  applyEditToCanvasElement,
  applyEditToModelCandidate,
  canvasToModelCandidates,
  type CandidateEdit,
  getCandidateKeyFromElement,
  getCandidateKind,
  getModelAspectRatio,
  getModelCandidateLabel,
  isModelCandidateDynamic,
  normalizeRelativeBounds,
  scopeVirtualModelCandidate,
} from '../utils/modelCandidates';

type View = 'windows' | 'canvas-detail';
type HomeBrowserMode = 'model' | 'flat';

const defaultLayers: Record<string, boolean> = {
  regions: true,
  geometric_regions: false,
  roi: false,
  elements: true,
  anchors: false,
  locators: false,
  ocr: false,
  vision: false,
  vlm_full: false,
  labels: false,
  info: false,
};

const terminalProcessingStates = new Set([
  'enhanced_ready',
  'semantic_ready',
  'semantic_timeout',
  'semantic_partial',
  'semantic_failed',
  'semantic_late_merged',
  'semantic_late_failed',
  'failed',
  'cancelled',
]);

function processingLabel(state?: string): string {
  switch (state) {
    case 'enhanced_ready':
      return '增强完成';
    case 'semantic_pending':
      return 'ROI 语义补充中';
    case 'semantic_ready':
      return 'ROI 语义完成';
    case 'semantic_timeout':
      return 'ROI 语义超时';
    case 'semantic_partial':
      return 'ROI 语义部分完成';
    case 'semantic_failed':
      return 'ROI 语义失败';
    case 'semantic_late_merged':
      return 'ROI 晚到结果已合并';
    case 'semantic_late_failed':
      return 'ROI 晚到结果失败';
    case 'fusion_running':
      return '融合中';
    case 'anchor_running':
      return '锚点处理中';
    case 'vlm_queued':
      return 'VLM 排队中';
    case 'vlm_running':
      return 'VLM 分析中';
    case 'failed':
      return '增强失败';
    case 'cancelled':
      return '已停止';
    case 'local_ready':
      return '本地就绪';
    default:
      return state || '未知';
  }
}

function formatStatusCounts(counts?: Record<string, number>): string {
  if (!counts) return '-';
  const entries = Object.entries(counts).filter(([, count]) => Number(count) > 0);
  if (!entries.length) return '-';
  return entries.map(([status, count]) => `${status}:${count}`).join(' ');
}

function overrideToEdit(override: CandidateOverrideResponse): CandidateEdit {
  return {
    ...(override.label ? { label: override.label } : {}),
    ...(override.semantic_role ? { semanticRole: override.semantic_role } : {}),
    ...(override.visual_type ? { visualType: override.visual_type } : {}),
    ...(override.region_id ? { regionId: override.region_id } : {}),
    ...(override.kind ? { kind: override.kind } : {}),
    ...(override.relative_bounds ? { relativeBounds: override.relative_bounds } : {}),
  };
}

function applyRoiVlmSupplementsToElements(canvasDetail: CanvasDetail) {
  const supplements = canvasDetail.roi_vlm_semantic_supplements ?? [];
  if (!supplements.length) return canvasDetail.elements;

  const annotations = new Map<string, { role?: string; label?: string; roiId: string; status: string }>();
  for (const supplement of supplements) {
    for (const annotation of supplement.candidate_annotations ?? []) {
      if (!annotation.candidate_id) continue;
      annotations.set(annotation.candidate_id, {
        role: annotation.role,
        label: annotation.label,
        roiId: supplement.roi_id,
        status: supplement.status,
      });
    }
  }
  if (!annotations.size) return canvasDetail.elements;

  return canvasDetail.elements.map((element) => {
    const annotation = annotations.get(element.element_id);
    if (!annotation) return element;
    const semanticTags = new Set(element.semantic_tags ?? []);
    semanticTags.add('roi_vlm');
    if (annotation.roiId) semanticTags.add(annotation.roiId);
    return {
      ...element,
      semantic_tags: Array.from(semanticTags),
      role_label: element.role_label,
      role_source: element.role_source,
      role_confidence: element.role_confidence,
      refine_status: element.refine_status,
      provider_sources: Array.from(new Set([...(element.provider_sources ?? []), 'vlm'])),
      attributes: {
        ...(element.attributes ?? {}),
        roi_vlm: annotation,
      },
    };
  });
}

export default function Index() {
  const [view, setView] = useState<View>('windows');
  const [selectedCanvasId, setSelectedCanvasId] = useState<string | null>(null);
  const [selectedElementId, setSelectedElementId] = useState<string | null>(null);
  const [overlayEnabled, setOverlayEnabled] = useState(false);
  const [activeLayers, setActiveLayers] = useState<Record<string, boolean>>(defaultLayers);
  const [selectedStateTemplateId, setSelectedStateTemplateId] = useState<string | null>(null);
  const [selectedModelCandidateKey, setSelectedModelCandidateKey] = useState<string | null>(null);
  const [hiddenModelCandidateKeys, setHiddenModelCandidateKeys] = useState<Set<string>>(new Set());
  const [candidateEdits, setCandidateEdits] = useState<Map<string, CandidateEdit>>(new Map());
  const [vlmSettingsOpen, setVlmSettingsOpen] = useState(false);
  const [vlmSemanticStatus, setVlmSemanticStatus] = useState('');
  const [lastVlmResult, setLastVlmResult] = useState<VLMSemanticResponse | null>(null);
  const [lastRoiVlmResult, setLastRoiVlmResult] = useState<RoiVlmSupplementResponse | null>(null);
  const [leftTopRatio, setLeftTopRatio] = useState(0.52);
  const [homeBrowserMode, setHomeBrowserMode] = useState<HomeBrowserMode>('model');
  const leftSplitRef = useRef<HTMLDivElement | null>(null);

  const { data: canvasDetail } = useCanvasDetail(selectedCanvasId);
  const vlmSemantic = useVLMSemantic();
  const roiVlmSupplement = useRoiVlmSupplement();

  useEffect(() => {
    const state = canvasDetail?.processing_state;
    if (!state) return;
    if (state === 'vlm_queued' || state === 'vlm_running' || state === 'semantic_pending') {
      setVlmSemanticStatus('running');
    } else if (state === 'enhanced_ready' || state === 'semantic_ready' || state === 'semantic_late_merged') {
      setVlmSemanticStatus((prev) => (prev === 'running' ? 'success' : prev));
    } else if (state === 'semantic_timeout') {
      setVlmSemanticStatus('timeout');
    } else if (state === 'failed' || state === 'semantic_failed' || state === 'semantic_late_failed') {
      setVlmSemanticStatus('failed');
    }
  }, [canvasDetail?.processing_state]);

  // Fetch virtual model for VirtualModelRenderer (persistent, SQLite-backed)
  const { data: virtualModel, isLoading: vmLoading, error: vmError } = useQuery({
    queryKey: ['virtual-model', selectedStateTemplateId],
    queryFn: () => fetchVirtualModel(selectedStateTemplateId!),
    enabled: !!selectedStateTemplateId,
    retry: false,
  });

  const { data: transitionGraph } = useQuery({
    queryKey: ['state-template-transitions', selectedStateTemplateId],
    queryFn: () => fetchStateTemplateTransitions(selectedStateTemplateId!),
    enabled: !!selectedStateTemplateId,
    retry: false,
  });

  const { data: canvasJobs } = useQuery({
    queryKey: ['canvas-jobs', selectedCanvasId],
    queryFn: () => fetchJobs({ canvasId: selectedCanvasId, limit: 20 }),
    enabled: !!selectedCanvasId,
    refetchInterval: () => {
      const state = canvasDetail?.processing_state;
      return state === 'vlm_queued' || state === 'vlm_running' || state === 'semantic_pending' || state === 'fusion_running' || state === 'anchor_running'
        ? 1500
        : false;
    },
  });
  const { data: canvasProcessing } = useQuery({
    queryKey: ['canvas-processing', selectedCanvasId],
    queryFn: () => fetchCanvasProcessing(selectedCanvasId!),
    enabled: !!selectedCanvasId,
    refetchInterval: (query) => (query.state.data?.is_processing ? 1500 : false),
  });
  const processingState = canvasProcessing?.processing_state || canvasDetail?.processing_state || '';
  const currentPhase = canvasProcessing?.current_phase || '';
  const activeEnhancementJobs = canvasProcessing?.active_jobs ?? [];
  const showEnhancementPanel = Boolean(processingState && processingState !== 'local_ready');

  // Phase 9: Count fixed_anchor candidates for TopBar badge
  const fixedAnchorCount = useMemo(() => {
    if (!virtualModel?.candidates) return 0;
    return virtualModel.candidates.filter((c) => c.permanence_state === 'fixed_anchor').length;
  }, [virtualModel?.candidates]);

  // Fetch tree to get state template metadata (label, etc.)
  const { data: treeData } = useQuery({
    queryKey: ['page-model-tree'],
    queryFn: fetchPageModelTree,
    enabled: !!selectedStateTemplateId,
  });
  const selectedStateTemplate = treeData?.state_templates?.find(
    (st) => st.state_template_id === selectedStateTemplateId,
  );
  const latestModelSnapshotId = useMemo(() => {
    if (!selectedStateTemplateId) return null;
    const snapshots = (treeData?.canvas_snapshots ?? [])
      .filter((snapshot) => (
        snapshot.state_template_id === selectedStateTemplateId &&
        snapshot.has_screenshot &&
        snapshot.available
      ))
      .sort((a, b) => b.captured_at.localeCompare(a.captured_at));
    return snapshots[0]?.canvas_id ?? null;
  }, [selectedStateTemplateId, treeData?.canvas_snapshots]);
  const modelSourceCanvasId = selectedCanvasId ?? latestModelSnapshotId;
  const { data: modelSourceCanvasDetail } = useCanvasDetail(
    !selectedCanvasId && modelSourceCanvasId ? modelSourceCanvasId : null,
  );
  const queryClient = useQueryClient();
  const [refining, setRefining] = useState(false);

  const overrideContext = useMemo(() => ({
    canvasId: selectedCanvasId,
    stateTemplateId: selectedStateTemplateId ?? canvasDetail?.state_template_id ?? null,
    pageModelId: canvasDetail?.page_model_id ?? virtualModel?.page_model_id ?? null,
    appId: canvasDetail?.app_id ?? virtualModel?.app_id ?? null,
  }), [
    selectedCanvasId,
    selectedStateTemplateId,
    canvasDetail?.state_template_id,
    canvasDetail?.page_model_id,
    canvasDetail?.app_id,
    virtualModel?.page_model_id,
    virtualModel?.app_id,
  ]);

  const { data: persistedOverrides } = useQuery({
    queryKey: ['candidate-overrides', overrideContext],
    queryFn: () => fetchCandidateOverrides({
      canvasId: overrideContext.canvasId,
      stateTemplateId: overrideContext.stateTemplateId,
      pageModelId: overrideContext.pageModelId,
    }),
    enabled: Boolean(overrideContext.canvasId || overrideContext.stateTemplateId || overrideContext.pageModelId),
  });

  useEffect(() => {
    if (!persistedOverrides) return;
    setCandidateEdits((prev) => {
      const next = new Map(prev);
      for (const override of persistedOverrides) {
        if (override.status !== 'active') continue;
        next.set(override.scope_key, overrideToEdit(override));
      }
      return next;
    });
  }, [persistedOverrides]);

  const handleRunRefine = useCallback(async () => {
    if (!selectedCanvasId) return;
    setRefining(true);
    try {
      await refineCanvas(selectedCanvasId, undefined, 'heuristic');
      await queryClient.invalidateQueries({ queryKey: ['canvas', selectedCanvasId] });
    } catch (err) {
      console.error('Refine failed:', err);
    } finally {
      setRefining(false);
    }
  }, [selectedCanvasId, queryClient]);

  const handleObserve = useCallback((canvasId: string, response?: ObserveResponse) => {
    setSelectedCanvasId(canvasId);
    setSelectedElementId(null);
    setSelectedStateTemplateId(null);
    setVlmSemanticStatus(response?.vlm_semantic_status || (response?.vlm_semantic_used ? 'success' : ''));
    setLastVlmResult(null);
    setView('canvas-detail');
  }, []);

  const handleSelectCanvas = useCallback((canvasId: string) => {
    setSelectedCanvasId(canvasId);
    setSelectedElementId(null);
    setSelectedStateTemplateId(null);
    setVlmSemanticStatus('');
    setLastVlmResult(null);
    setHomeBrowserMode('flat');
    setView('canvas-detail');
  }, []);

  const handleSelectStateTemplate = useCallback((stateTemplateId: string) => {
    setSelectedStateTemplateId(stateTemplateId);
    setSelectedCanvasId(null);
    setSelectedElementId(null);
    setHomeBrowserMode('model');
    setView('canvas-detail');
  }, []);

  const handleDeletedCanvas = useCallback((canvasId: string) => {
    if (selectedCanvasId !== canvasId && modelSourceCanvasId !== canvasId) return;
    setSelectedCanvasId(null);
    setSelectedElementId(null);
    setSelectedModelCandidateKey(null);
    setLastVlmResult(null);
    setVlmSemanticStatus('');
    if (!selectedStateTemplateId) {
      setView('windows');
    }
  }, [modelSourceCanvasId, selectedCanvasId, selectedStateTemplateId]);

  const handleDeletedStateTemplate = useCallback((stateTemplateId: string) => {
    if (selectedStateTemplateId !== stateTemplateId) return;
    setSelectedStateTemplateId(null);
    setSelectedCanvasId(null);
    setSelectedElementId(null);
    setSelectedModelCandidateKey(null);
    setLastVlmResult(null);
    setVlmSemanticStatus('');
    setView('windows');
  }, [selectedStateTemplateId]);

  const handleDeletedPageModel = useCallback((pageModelId: string) => {
    const selectedStateBelongsToModel = treeData?.state_templates?.some(
      (st) => st.state_template_id === selectedStateTemplateId && st.page_model_id === pageModelId,
    );
    const selectedCanvasBelongsToModel = canvasDetail?.page_model_id === pageModelId;
    if (!selectedStateBelongsToModel && !selectedCanvasBelongsToModel) return;
    setSelectedStateTemplateId(null);
    setSelectedCanvasId(null);
    setSelectedElementId(null);
    setSelectedModelCandidateKey(null);
    setLastVlmResult(null);
    setVlmSemanticStatus('');
    setView('windows');
  }, [canvasDetail?.page_model_id, selectedStateTemplateId, treeData?.state_templates]);

  const handleBack = useCallback(() => {
    if (selectedStateTemplateId && selectedCanvasId) {
      // Return to canvas detail from virtual model
      setSelectedStateTemplateId(null);
      setSelectedModelCandidateKey(null);
    } else {
      setView('windows');
      setSelectedCanvasId(null);
      setSelectedElementId(null);
      setSelectedStateTemplateId(null);
      setSelectedModelCandidateKey(null);
    }
  }, [selectedStateTemplateId, selectedCanvasId]);

  const handleBackHome = useCallback(() => {
    setView('windows');
    setSelectedCanvasId(null);
    setSelectedElementId(null);
    setSelectedStateTemplateId(null);
    setSelectedModelCandidateKey(null);
  }, []);

  const handleBackToScreenshot = useCallback(() => {
    if (!modelSourceCanvasId) {
      handleBackHome();
      return;
    }
    setSelectedCanvasId(modelSourceCanvasId);
    setSelectedElementId(null);
    setSelectedStateTemplateId(null);
    setSelectedModelCandidateKey(null);
    setView('canvas-detail');
  }, [modelSourceCanvasId, handleBackHome]);

  const handleSelectElement = useCallback((id: string) => {
    setSelectedElementId(id);
  }, []);

  const handleViewVirtualModel = useCallback((stateTemplateId: string) => {
    setSelectedStateTemplateId(stateTemplateId);
    setSelectedElementId(null);
    setSelectedModelCandidateKey(null);
    setHomeBrowserMode('model');
    // Preserve selectedCanvasId so we can return to the screenshot
  }, []);

  const handleLeftSplitMouseDown = useCallback((event: ReactMouseEvent<HTMLDivElement>) => {
    event.preventDefault();
    const container = leftSplitRef.current;
    if (!container) return;
    const rect = container.getBoundingClientRect();
    const onMove = (moveEvent: MouseEvent) => {
      const raw = (moveEvent.clientY - rect.top) / Math.max(rect.height, 1);
      setLeftTopRatio(Math.min(0.78, Math.max(0.22, raw)));
    };
    const onUp = () => {
      window.removeEventListener('mousemove', onMove);
      window.removeEventListener('mouseup', onUp);
    };
    window.addEventListener('mousemove', onMove);
    window.addEventListener('mouseup', onUp);
  }, []);

  const handleTriggerVLMMode = useCallback((taskMode: string, candidateIds: string[] = []) => {
    if (!selectedCanvasId) return;
    setVlmSemanticStatus('running');
    vlmSemantic.mutate(
      { canvasId: selectedCanvasId, force: true, taskMode, candidateIds },
      {
        onSuccess: (result) => {
          setVlmSemanticStatus(result.processing_state ? 'running' : (result.status || 'success'));
          setLastVlmResult(result);
          void queryClient.invalidateQueries({ queryKey: ['canvas', selectedCanvasId] });
          void queryClient.invalidateQueries({ queryKey: ['virtual-model'] });
          void queryClient.invalidateQueries({ queryKey: ['page-model-tree'] });
        },
        onError: (err) => {
          setVlmSemanticStatus('failed');
          setLastVlmResult({
            canvas_id: selectedCanvasId,
            status: 'failed',
            provider: '',
            model: '',
            error: String(err),
            error_code: 'request_error',
            message: 'VLM 请求失败',
            latency_ms: 0,
            token_input: 0,
            token_output: 0,
            from_cache: false,
            region_count: 0,
            control_count: 0,
            correction_count: 0,
            dynamic_zone_count: 0,
            warning_count: 0,
            job_id: null,
            processing_state: '',
          });
        },
      },
    );
  }, [selectedCanvasId, vlmSemantic, queryClient]);

  const handleTriggerVLM = useCallback(() => {
    handleTriggerVLMMode('candidate_annotation');
  }, [handleTriggerVLMMode]);

  const handleTriggerRoiVLM = useCallback(() => {
    if (!selectedCanvasId || roiVlmSupplement.isPending) return;
    setLastRoiVlmResult(null);
    roiVlmSupplement.mutate(
      { canvasId: selectedCanvasId, deadlineMs: 2000 },
      {
        onSuccess: (result) => setLastRoiVlmResult(result),
      },
    );
  }, [roiVlmSupplement, selectedCanvasId]);

  const handleCancelJob = useCallback((jobId: string) => {
    void cancelJob(jobId).then(() => {
      void queryClient.invalidateQueries({ queryKey: ['canvas-jobs', selectedCanvasId] });
      if (selectedCanvasId) void queryClient.invalidateQueries({ queryKey: ['canvas', selectedCanvasId] });
    }).catch((err) => console.error('Cancel job failed:', err));
  }, [queryClient, selectedCanvasId]);

  const handleSelectModelCandidate = useCallback((keyId: string) => {
    setSelectedModelCandidateKey((prev) => (prev === keyId ? null : keyId));
  }, []);

  const handleToggleHiddenModelCandidate = useCallback((keyId: string) => {
    setHiddenModelCandidateKeys((prev) => {
      const next = new Set(prev);
      if (next.has(keyId)) next.delete(keyId);
      else next.add(keyId);
      return next;
    });
    setSelectedModelCandidateKey(null);
  }, []);

  const handleUpdateCandidateEdit = useCallback((keyId: string, patch: CandidateEdit) => {
    const nextEdit = { ...(candidateEdits.get(keyId) ?? {}), ...patch };
    setCandidateEdits((prev) => {
      const next = new Map(prev);
      const current = next.get(keyId) ?? {};
      next.set(keyId, { ...current, ...patch });
      return next;
    });

    let canvasId = overrideContext.canvasId;
    let stateTemplateId = overrideContext.stateTemplateId;
    let elementId: string | null = null;
    let stableKeyId: string | null = null;

    if (keyId.startsWith('stable:')) {
      const stableScope = keyId.slice('stable:'.length);
      const splitAt = stableScope.lastIndexOf(':');
      stableKeyId = splitAt > 0 ? stableScope.slice(splitAt + 1) : stableScope;
    } else if (keyId.startsWith('canvas:')) {
      const rest = keyId.slice('canvas:'.length);
      const splitAt = rest.lastIndexOf(':');
      if (splitAt > 0) {
        canvasId = rest.slice(0, splitAt);
        elementId = rest.slice(splitAt + 1);
      }
    } else if (keyId.startsWith('state:')) {
      const rest = keyId.slice('state:'.length);
      const splitAt = rest.indexOf(':');
      if (splitAt > 0) {
        stateTemplateId = rest.slice(0, splitAt);
        elementId = rest.slice(splitAt + 1).replace(/^transient_/, '');
      }
    } else {
      stableKeyId = keyId;
    }

    const body: CandidateOverrideUpsertRequest = {
      scope_key: keyId,
      app_id: overrideContext.appId,
      page_model_id: overrideContext.pageModelId,
      state_template_id: stateTemplateId,
      canvas_id: canvasId,
      element_id: elementId,
      stable_key_id: stableKeyId,
      label: nextEdit.label ?? null,
      semantic_role: nextEdit.semanticRole ?? null,
      visual_type: nextEdit.visualType ?? null,
      region_id: nextEdit.regionId ?? null,
      kind: nextEdit.kind ?? null,
      relative_bounds: nextEdit.relativeBounds ?? null,
      source: 'manual',
      status: 'active',
    };

    void upsertCandidateOverride(body)
      .then(() => queryClient.invalidateQueries({ queryKey: ['candidate-overrides'] }))
      .catch((err) => console.error('Persist candidate override failed:', err));
  }, [candidateEdits, overrideContext, queryClient]);

  const handleRelabelModelCandidate = useCallback((keyId: string, newLabel: string) => {
    handleUpdateCandidateEdit(keyId, { label: newLabel });
  }, [handleUpdateCandidateEdit]);

  const handleMarkDynamic = useCallback((keyId: string) => {
    handleUpdateCandidateEdit(keyId, { kind: 'dynamic' });
  }, [handleUpdateCandidateEdit]);

  const handlePreviewCandidateBounds = useCallback((keyId: string, bounds: number[]) => {
    setCandidateEdits((prev) => {
      const next = new Map(prev);
      const current = next.get(keyId) ?? {};
      next.set(keyId, { ...current, relativeBounds: bounds });
      return next;
    });
  }, []);

  const handleCommitCandidateBounds = useCallback((keyId: string, bounds: number[]) => {
    handleUpdateCandidateEdit(keyId, { relativeBounds: bounds });
  }, [handleUpdateCandidateEdit]);

  const handleToggleOverlay = useCallback(() => {
    setOverlayEnabled((prev) => !prev);
  }, []);

  const handleToggleLayer = useCallback((layer: string) => {
    setActiveLayers((prev) => ({ ...prev, [layer]: !prev[layer] }));
  }, []);

  const effectiveCanvasDetail = useMemo(() => {
    if (!canvasDetail) return null;
    const roiVlmElements = applyRoiVlmSupplementsToElements(canvasDetail);
    const elements = roiVlmElements
      .map((element) => {
        const key = getCandidateKeyFromElement(element, canvasDetail.canvas_id, canvasDetail.page_model_id);
        return applyEditToCanvasElement(element, canvasDetail, candidateEdits.get(key));
      })
      .filter((element) => {
        const key = getCandidateKeyFromElement(element, canvasDetail.canvas_id, canvasDetail.page_model_id);
        const edit = candidateEdits.get(key);
        return !hiddenModelCandidateKeys.has(key) && edit?.kind !== 'ignored';
      });
    return { ...canvasDetail, elements };
  }, [canvasDetail, candidateEdits, hiddenModelCandidateKeys]);

  const selectedElement = effectiveCanvasDetail?.elements.find((e) => e.element_id === selectedElementId) ?? null;

  const sourceCanvasMatchesModel = Boolean(
    selectedCanvasId &&
    canvasDetail &&
    selectedStateTemplateId &&
    canvasDetail.state_template_id === selectedStateTemplateId,
  );

  const workbenchCandidates = useMemo(() => {
    const rawCandidates = (() => {
    if (sourceCanvasMatchesModel) {
      return canvasToModelCandidates(canvasDetail);
    }
    return (virtualModel?.candidates ?? []).map((candidate) => (
      scopeVirtualModelCandidate(candidate, selectedStateTemplateId, virtualModel?.page_model_id ?? null)
    ));
    })();
    return rawCandidates
      .map((candidate) => applyEditToModelCandidate(candidate, candidateEdits.get(candidate.key_id)))
      .filter((candidate) => candidateEdits.get(candidate.key_id)?.kind !== 'ignored');
  }, [sourceCanvasMatchesModel, canvasDetail, virtualModel, candidateEdits]);

  const selectedModelCandidate: VirtualModelCandidate | null = useMemo(() => {
    if (!selectedModelCandidateKey) return null;
    const candidate = workbenchCandidates.find((c) => c.key_id === selectedModelCandidateKey);
    return candidate ?? null;
  }, [workbenchCandidates, selectedModelCandidateKey]);

  const { data: selectedControlTransitions } = useQuery({
    queryKey: ['control-transitions', selectedModelCandidate?.key_id],
    queryFn: () => fetchControlTransitions({ candidateKey: selectedModelCandidate!.key_id }),
    enabled: !!selectedModelCandidate?.key_id,
    retry: false,
  });

  return (
    <div
      className="flex flex-col"
      style={{
        width: '100vw',
        height: '100vh',
        minHeight: '100vh',
        background: 'var(--prc-bg)',
        overflow: 'hidden',
      }}
    >
      {/* Top Bar */}
      <TopBar
        view={view}
        canvasDetail={canvasDetail ?? null}
        onBack={selectedStateTemplateId ? handleBackHome : handleBack}
        isVirtualModelView={!!selectedStateTemplateId}
        vlmSemanticStatus={vlmSemantic.isPending ? 'running' : vlmSemanticStatus}
        onTriggerVLM={selectedCanvasId ? handleTriggerVLM : undefined}
        onOpenSettings={() => setVlmSettingsOpen(true)}
        fixedAnchorCount={fixedAnchorCount}
      />

      <VLMSettingsPanel open={vlmSettingsOpen} onClose={() => setVlmSettingsOpen(false)} />

      {/* Main Content - Three Column */}
      <div className="flex flex-1 min-h-0 overflow-hidden">
        {/* Left Column */}
        <div className="flex-shrink-0 h-full min-h-0 flex flex-col overflow-hidden" style={{ width: '260px' }}>
          {view === 'windows' ? (
            <div ref={leftSplitRef} className="flex flex-col h-full overflow-hidden">
              {/* Window Browser (top half) */}
              <div className="overflow-hidden" style={{ minHeight: 80, height: `${leftTopRatio * 100}%` }}>
                <WindowBrowser onObserved={handleObserve} />
              </div>
              <div
                className="flex-shrink-0 cursor-row-resize"
                style={{
                  height: '8px',
                  borderTop: '1px solid var(--prc-border)',
                  borderBottom: '1px solid var(--prc-border)',
                  background: 'var(--prc-surface-1)',
                }}
                title="拖动调整窗口列表和观察截图/模型区域高度"
                onMouseDown={handleLeftSplitMouseDown}
              />
              {/* Canvas Browser (bottom half) */}
              <div className="flex-1 overflow-hidden" style={{ minHeight: 100 }}>
                <CanvasBrowser
                  onSelectCanvas={handleSelectCanvas}
                  onSelectStateTemplate={handleSelectStateTemplate}
                  onDeletedCanvas={handleDeletedCanvas}
                  onDeletedStateTemplate={handleDeletedStateTemplate}
                  onDeletedPageModel={handleDeletedPageModel}
                  viewMode={homeBrowserMode}
                  onViewModeChange={setHomeBrowserMode}
                />
              </div>
            </div>
          ) : selectedStateTemplateId && virtualModel ? (
            <ModelCandidateList
              candidates={workbenchCandidates}
              selectedKeyId={selectedModelCandidateKey}
              hiddenKeyIds={hiddenModelCandidateKeys}
              candidateEdits={candidateEdits}
              sourceCanvasId={modelSourceCanvasId}
              onSelect={handleSelectModelCandidate}
            />
          ) : (
            <div className="flex flex-col h-full overflow-hidden">
              {/* Element list */}
              <div className="flex-1 overflow-hidden" style={{ minHeight: 0 }}>
                <ElementList
                  elements={effectiveCanvasDetail?.elements ?? []}
                  regions={effectiveCanvasDetail?.regions ?? []}
                  selectedId={selectedElementId}
                  onSelect={handleSelectElement}
                  canvasId={selectedCanvasId}
                />
              </div>
              {/* Query panel */}
              {selectedCanvasId && (
                <QueryPanel
                  canvasId={selectedCanvasId}
                  onSelectElement={handleSelectElement}
                  selectedElementId={selectedElementId}
                />
              )}
            </div>
          )}
        </div>

        {/* Center: Review Canvas or Virtual Model */}
        <div className="flex-1 overflow-hidden">
          {selectedStateTemplateId ? (
            <div className="flex flex-col h-full" style={{ background: 'var(--prc-bg)' }}>
              {/* State template header */}
              <div className="flex items-center justify-between px-3 py-1.5 border-b flex-shrink-0" style={{ background: 'var(--prc-surface)', borderColor: 'var(--prc-border)' }}>
                <div className="flex items-center gap-2">
                  <button
                    className="prc-btn prc-btn-ghost h-5 px-1.5 flex items-center gap-1"
                    style={{ fontSize: '0.6rem', color: 'var(--prc-text-dim)' }}
                    onClick={handleBackToScreenshot}
                  >
                    <ArrowLeftIcon size={10} />
                    <span>返回截图</span>
                  </button>
                  <span className="prc-text text-xs font-medium">
                    {selectedStateTemplate?.state_label ?? '页面状态'}
                  </span>
                  {selectedCanvasId && canvasDetail && (
                    <span className="prc-text-dim" style={{ fontSize: '0.58rem' }}>
                      来源: {canvasDetail.window_title || canvasDetail.canvas_id.slice(0, 12)}
                    </span>
                  )}
                </div>
                <span className="prc-text-dim mono" style={{ fontSize: '0.6rem' }}>
                  {workbenchCandidates.length} 项 · {selectedStateTemplate?.total_element_count ?? 0} 原始元素
                </span>
              </div>
              {/* Virtual model body */}
              <div className="flex-1 overflow-auto flex items-start justify-center p-6">
                {vmLoading ? (
                  <div className="flex flex-col items-center justify-center py-20">
                    <div className="prc-text-dim text-xs">加载虚拟模型...</div>
                  </div>
                ) : vmError ? (
                  <div className="flex flex-col items-center justify-center py-20">
                    <span className="prc-text-dim text-xs">无法加载虚拟模型，请重新观察窗口</span>
                  </div>
                ) : virtualModel ? (
                  <VirtualModelRenderer
                    model={virtualModel}
                    candidates={workbenchCandidates}
                    selectedKeyId={selectedModelCandidateKey}
                    hiddenKeyIds={hiddenModelCandidateKeys}
                    candidateEdits={candidateEdits}
                    sourceCanvasId={modelSourceCanvasId}
                    aspectRatio={getModelAspectRatio(effectiveCanvasDetail ?? modelSourceCanvasDetail ?? undefined)}
                    onSelectCandidate={handleSelectModelCandidate}
                    onPreviewBounds={handlePreviewCandidateBounds}
                    onCommitBounds={handleCommitCandidateBounds}
                    showMaterialArea={false}
                  />
                ) : null}
              </div>
            </div>
          ) : (
            <div className="relative h-full">
              <ReviewCanvas
                canvasDetail={effectiveCanvasDetail ?? null}
                selectedId={selectedElementId}
                onSelect={handleSelectElement}
                overlayEnabled={overlayEnabled}
                onToggleOverlay={handleToggleOverlay}
                activeLayers={activeLayers}
                onToggleLayer={handleToggleLayer}
              />
            </div>
          )}
        </div>

        {/* Right: Candidate Detail Panel */}
        <div className="flex-shrink-0 overflow-hidden" style={{ width: '320px' }}>
          {selectedStateTemplateId ? (
            selectedModelCandidate ? (
              <ModelCandidatePanel
                candidate={selectedModelCandidate}
                onClose={() => setSelectedModelCandidateKey(null)}
                onToggleHidden={handleToggleHiddenModelCandidate}
                onRelabel={handleRelabelModelCandidate}
                onMarkDynamic={handleMarkDynamic}
                onUpdateEdit={handleUpdateCandidateEdit}
                edit={selectedModelCandidateKey ? candidateEdits.get(selectedModelCandidateKey) : undefined}
                availableRegions={effectiveCanvasDetail?.regions.map((r) => r.region_id) ?? []}
                controlTransitions={selectedControlTransitions?.transitions ?? []}
              />
            ) : (
              <ModelOverviewPanel
                model={virtualModel ?? null}
                transitions={transitionGraph ?? null}
                jobs={canvasJobs ?? []}
              />
            )
          ) : (
            <div className="flex flex-col h-full">
              {selectedCanvasId && (
                <div className="flex-shrink-0 p-2 border-b" style={{ borderColor: 'var(--prc-border)', background: 'var(--prc-surface-1)' }}>
                  <button
                    className="prc-btn w-full h-7 flex items-center justify-center gap-1.5 mb-2"
                    style={{
                      background: 'rgba(56,189,248,0.12)',
                      color: '#38bdf8',
                      fontSize: '0.68rem',
                      borderColor: 'rgba(56,189,248,0.32)',
                    }}
                    onClick={handleTriggerVLM}
                    disabled={vlmSemantic.isPending}
                  >
                    <Sparkles size={12} />
                    <span>{vlmSemantic.isPending ? 'VLM 分析中...' : 'VLM 分析当前截图'}</span>
                  </button>
                  <div className="grid grid-cols-2 gap-1.5 mb-2">
                    <button
                      className="prc-btn prc-btn-ghost h-6"
                      style={{ fontSize: '0.6rem', color: '#a78bfa' }}
                      onClick={() => handleTriggerVLMMode('region_understanding')}
                      disabled={vlmSemantic.isPending}
                    >
                      区域理解
                    </button>
                    <button
                      className="prc-btn prc-btn-ghost h-6"
                      style={{ fontSize: '0.6rem', color: '#fbbf24' }}
                      onClick={() => handleTriggerVLMMode('missing_audit')}
                      disabled={vlmSemantic.isPending}
                    >
                      遗漏核对
                    </button>
                    <button
                      className="prc-btn prc-btn-ghost h-6 col-span-2"
                      style={{ fontSize: '0.6rem', color: selectedElementId ? '#34d399' : 'var(--prc-text-dim)' }}
                      onClick={() => selectedElementId && handleTriggerVLMMode('icon_crop_understanding', [selectedElementId])}
                      disabled={vlmSemantic.isPending || !selectedElementId}
                    >
                      识别选中图标
                    </button>
                    <button
                      className="prc-btn prc-btn-ghost h-6 col-span-2"
                      style={{ fontSize: '0.6rem', color: '#22d3ee' }}
                      onClick={handleTriggerRoiVLM}
                      disabled={roiVlmSupplement.isPending}
                    >
                      {roiVlmSupplement.isPending ? 'ROI 补充中...' : 'ROI 语义补充'}
                    </button>
                  </div>
                  {(lastRoiVlmResult || (effectiveCanvasDetail?.roi_vlm_semantic_supplements?.length ?? 0) > 0 || (effectiveCanvasDetail?.roi_vlm_late_failures?.length ?? 0) > 0) && (
                    <div className="mb-2 rounded border p-2" style={{ borderColor: 'var(--prc-border)', background: 'var(--prc-surface)' }}>
                      <div className="flex items-center justify-between gap-2 prc-text" style={{ fontSize: '0.62rem' }}>
                        <span>ROI VLM：{lastRoiVlmResult?.status ?? '已补充'}</span>
                        <span className="mono">{effectiveCanvasDetail?.roi_vlm_semantic_supplements?.length ?? 0} 条</span>
                      </div>
                      <div className="mt-1 grid grid-cols-2 gap-x-2 gap-y-1 prc-text-dim" style={{ fontSize: '0.56rem' }}>
                        {lastRoiVlmResult ? <span>耗时：{lastRoiVlmResult.elapsed_ms}ms</span> : null}
                        {lastRoiVlmResult ? <span>返回状态：{formatStatusCounts(lastRoiVlmResult.result_status_counts)}</span> : null}
                        <span>超时：{effectiveCanvasDetail?.roi_vlm_timeouts?.length ?? 0}</span>
                        <span>失败：{effectiveCanvasDetail?.roi_vlm_late_failures?.length ?? 0}</span>
                        <span>候选补充：{effectiveCanvasDetail?.roi_vlm_semantic_supplements?.reduce((sum, item) => sum + (item.candidate_annotations?.length ?? 0), 0) ?? 0}</span>
                        <span>晚到接收：{effectiveCanvasDetail?.roi_vlm_semantic_supplements?.some((item) => item.late_result_accepted) ? '是' : '否'}</span>
                      </div>
                    </div>
                  )}
                  {vlmSemanticStatus && (
                    <div className="mb-2 rounded border p-2" style={{ borderColor: 'var(--prc-border)', background: 'var(--prc-surface)' }}>
                      <div className="flex items-center justify-between gap-2 prc-text" style={{ fontSize: '0.62rem' }}>
                        <span>VLM 状态：{vlmSemanticStatus}</span>
                        {lastVlmResult?.latency_ms ? <span className="mono">{lastVlmResult.latency_ms}ms</span> : null}
                      </div>
                      {lastVlmResult && (
                        <div className="mt-1 grid grid-cols-2 gap-x-2 gap-y-1 prc-text-dim" style={{ fontSize: '0.56rem' }}>
                          <span>模型：{lastVlmResult.model || lastVlmResult.provider || '-'}</span>
                          <span>缓存：{lastVlmResult.from_cache ? '是' : '否'}</span>
                          <span>区域：{lastVlmResult.region_count}</span>
                          <span>控件：{lastVlmResult.control_count}</span>
                          <span>修正：{lastVlmResult.correction_count}</span>
                          <span>动态区：{lastVlmResult.dynamic_zone_count}</span>
                          <span>输入：{lastVlmResult.token_input}</span>
                          <span>输出：{lastVlmResult.token_output}</span>
                        </div>
                      )}
                      {lastVlmResult?.error && (
                        <div className="mt-1 prc-text mono" style={{ color: 'var(--prc-danger)', fontSize: '0.55rem', wordBreak: 'break-word' }}>
                          {lastVlmResult.error}
                        </div>
                      )}
                    </div>
                  )}
                  {showEnhancementPanel && (
                    <div className="mb-2 rounded border p-2" style={{ borderColor: 'var(--prc-border)', background: 'var(--prc-surface)' }}>
                      <div className="flex items-center justify-between prc-text" style={{ fontSize: '0.62rem' }}>
                        <span>后台增强</span>
                        <span className="mono">{processingLabel(processingState)}</span>
                      </div>
                      {currentPhase && !terminalProcessingStates.has(processingState) && (
                        <div className="mt-1 flex items-center justify-between prc-text-dim" style={{ fontSize: '0.55rem' }}>
                          <span>当前阶段</span>
                          <span className="mono">{processingLabel(currentPhase)}</span>
                        </div>
                      )}
                      {canvasDetail?.processing_error && (
                        <div className="mt-1 prc-text mono" style={{ color: 'var(--prc-danger)', fontSize: '0.55rem', wordBreak: 'break-word' }}>
                          {canvasDetail.processing_error}
                        </div>
                      )}
                      {activeEnhancementJobs.slice(0, 3).map((job) => (
                        <div key={job.job_id} className="mt-1 flex items-center justify-between gap-2 prc-text-dim" style={{ fontSize: '0.55rem' }}>
                          <span className="truncate">{job.job_type}</span>
                          <div className="flex items-center gap-1 flex-shrink-0">
                            <span className="mono">{processingLabel(currentPhase || job.state)}</span>
                            {!terminalProcessingStates.has(job.state) && (
                              <button
                                className="prc-btn prc-btn-ghost h-5 px-1"
                                style={{ fontSize: '0.52rem', color: 'var(--prc-warning)' }}
                                onClick={() => handleCancelJob(job.job_id)}
                              >
                                停止
                              </button>
                            )}
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                  {canvasDetail?.state_template_id && (
                  <button
                    className="prc-btn w-full h-7 flex items-center justify-center gap-1.5"
                    style={{
                      background: 'rgba(168,85,247,0.12)',
                      color: '#a855f7',
                      fontSize: '0.68rem',
                      borderColor: 'rgba(168,85,247,0.28)',
                    }}
                    onClick={() => handleViewVirtualModel(canvasDetail.state_template_id!)}
                  >
                    <Layers size={12} />
                    <span>查看当前截图模型</span>
                  </button>
                  )}
                </div>
              )}
              <div className="flex-1 overflow-hidden">
                <ReviewPanel
                  element={selectedElement}
                  canvasDetail={effectiveCanvasDetail ?? null}
                />
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Bottom Bar */}
      <BottomBar
        canvasDetail={effectiveCanvasDetail ?? canvasDetail ?? null}
        onRunRefine={selectedCanvasId ? handleRunRefine : undefined}
        refining={refining}
      />
    </div>
  );
}

// ── ModelCandidateList (left panel when viewing virtual model) ──

import { Eye, EyeOff, Tag, Layers } from 'lucide-react';
import { getCanvasScreenshotUrl } from '../api/client';
import { t, formatStableKeyId } from '../i18n/zh';

interface ModelCandidateListProps {
  candidates: import('../api/types').VirtualModelCandidate[];
  selectedKeyId: string | null;
  hiddenKeyIds: Set<string>;
  candidateEdits: Map<string, CandidateEdit>;
  sourceCanvasId?: string | null;
  onSelect: (keyId: string) => void;
}

function ModelCandidateList({ candidates, selectedKeyId, hiddenKeyIds, candidateEdits, sourceCanvasId, onSelect }: ModelCandidateListProps) {
  const visible = candidates.filter((c) => !hiddenKeyIds.has(c.key_id));
  const fixed = visible.filter((c) => getCandidateKind(c, candidateEdits.get(c.key_id)) === 'fixed');
  const containers = visible.filter((c) => getCandidateKind(c, candidateEdits.get(c.key_id)) === 'container');
  const dynamic = visible.filter((c) => getCandidateKind(c, candidateEdits.get(c.key_id)) === 'dynamic');

  return (
    <div className="flex flex-col h-full" style={{ borderRight: '1px solid var(--prc-border)' }}>
      {/* Header */}
      <div className="px-3 pt-3 pb-2 border-b" style={{ borderColor: 'var(--prc-border)' }}>
        <div className="flex items-center gap-1.5">
          <Layers size={12} style={{ color: 'var(--prc-accent)' }} />
          <span className="prc-text text-xs font-semibold">模型控件</span>
          <span className="prc-badge mono" style={{ background: 'var(--prc-surface-2)', color: 'var(--prc-text-dim)' }}>
            {fixed.length}
          </span>
        </div>
      </div>

      {/* List */}
      <div className="flex-1 overflow-y-auto">
        {fixed.map((c) => (
          <ModelCandidateListCard
            key={c.key_id}
            candidate={c}
            kind="fixed"
            selected={selectedKeyId === c.key_id}
            hidden={hiddenKeyIds.has(c.key_id)}
            sourceCanvasId={sourceCanvasId}
            label={getModelCandidateLabel(c, candidateEdits)}
            onSelect={onSelect}
          />
        ))}

        {containers.length > 0 && (
          <ModelCandidateSection title="结构区域" count={containers.length}>
            {containers.map((c) => (
              <ModelCandidateListCard
                key={c.key_id}
                candidate={c}
                kind="container"
                selected={selectedKeyId === c.key_id}
                hidden={hiddenKeyIds.has(c.key_id)}
                sourceCanvasId={sourceCanvasId}
                label={getModelCandidateLabel(c, candidateEdits)}
                onSelect={onSelect}
              />
            ))}
          </ModelCandidateSection>
        )}

        {/* Dynamic content section */}
        {dynamic.length > 0 && (
          <ModelCandidateSection title="动态内容" count={dynamic.length}>
            {dynamic.map((c) => {
              return (
                <ModelCandidateListCard
                  key={c.key_id}
                  candidate={c}
                  kind="dynamic"
                  selected={selectedKeyId === c.key_id}
                  hidden={hiddenKeyIds.has(c.key_id)}
                  sourceCanvasId={sourceCanvasId}
                  label={getModelCandidateLabel(c, candidateEdits)}
                  onSelect={onSelect}
                />
              );
            })}
          </ModelCandidateSection>
        )}
      </div>
    </div>
  );
}

function ModelCandidateSection({ title, count, children }: { title: string; count: number; children: React.ReactNode }) {
  return (
    <>
      <div
        className="px-3 py-1 flex items-center gap-1"
        style={{ background: 'var(--prc-surface-1)', borderBottom: '1px solid var(--prc-border)' }}
      >
        <span className="prc-text-dim" style={{ fontSize: '0.58rem' }}>{title}</span>
        <span className="prc-text-dim mono" style={{ fontSize: '0.52rem' }}>({count})</span>
      </div>
      {children}
    </>
  );
}

function ModelCandidateListCard({
  candidate,
  kind,
  selected,
  hidden,
  sourceCanvasId,
  label,
  onSelect,
}: {
  candidate: import('../api/types').VirtualModelCandidate;
  kind: 'fixed' | 'dynamic' | 'container';
  selected: boolean;
  hidden: boolean;
  sourceCanvasId?: string | null;
  label: string;
  onSelect: (keyId: string) => void;
}) {
  const screenshotUrl = sourceCanvasId ? getCanvasScreenshotUrl(sourceCanvasId, false) : null;
  const relativeBounds = normalizeRelativeBounds(candidate.relative_bounds);
  const accent = kind === 'fixed' ? 'var(--prc-accent)' : kind === 'container' ? '#94a3b8' : '#38bdf8';
  const kindLabel = kind === 'fixed' ? '固定控件' : kind === 'container' ? '结构区域' : '动态内容';

  return (
    <div
      className="px-3 py-2 cursor-pointer flex items-start gap-2 transition-colors"
      style={{
        background: selected ? 'var(--prc-surface-2)' : 'transparent',
        borderLeft: selected ? `2px solid ${accent}` : '2px solid transparent',
        opacity: hidden ? 0.4 : kind === 'dynamic' ? 0.82 : 1,
        borderBottom: '1px solid var(--prc-border)',
      }}
      onClick={() => onSelect(candidate.key_id)}
    >
      <div
        className="flex-shrink-0 rounded overflow-hidden"
        style={{
          width: '36px',
          height: '30px',
          background: 'var(--prc-surface-2)',
          border: '1px solid var(--prc-border)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
        }}
      >
        {screenshotUrl && relativeBounds.length >= 4 ? (
          <RelativeCropPreview
            screenshotUrl={screenshotUrl}
            bounds={relativeBounds}
            alt={label}
          />
        ) : kind === 'dynamic' ? (
          <Eye size={11} className="prc-text-dim" />
        ) : (
          <Tag size={11} style={{ color: accent }} />
        )}
      </div>
      <div className="min-w-0 flex-1">
        <div className="flex items-center gap-1 min-w-0">
          <span className={kind === 'dynamic' ? 'prc-text-dim text-xs truncate' : 'prc-text text-xs truncate'}>
            {label}
          </span>
          {hidden && <EyeOff size={9} className="prc-text-dim flex-shrink-0" />}
        </div>
        <div className="flex items-center gap-1 mt-0.5 flex-wrap">
          <span className="prc-badge" style={{ background: `${accent}18`, color: accent, fontSize: '0.52rem' }}>
            {kindLabel}
          </span>
          <span className="prc-text-dim mono truncate" style={{ fontSize: '0.52rem' }}>
            {candidate.element_id || formatStableKeyId(candidate.key_id)}
          </span>
        </div>
        <div className="flex items-center gap-1 mt-0.5">
          <span className="prc-text-dim truncate" style={{ fontSize: '0.52rem' }}>
            {t(candidate.canonical_role, candidate.canonical_role)}
          </span>
          <span className="prc-text-dim" style={{ fontSize: '0.52rem' }}>
            {t(candidate.canonical_region, candidate.canonical_region)}
          </span>
          <span className="mono" style={{
            fontSize: '0.52rem',
            color: candidate.confidence >= 0.8 ? 'var(--prc-confirmed)' : candidate.confidence >= 0.6 ? 'var(--prc-warning)' : 'var(--prc-danger)',
          }}>
            {Math.round(candidate.confidence * 100)}%
          </span>
        </div>
      </div>
    </div>
  );
}

function RelativeCropPreview({
  screenshotUrl,
  bounds,
  alt,
}: {
  screenshotUrl: string;
  bounds: number[];
  alt: string;
}) {
  const [x1, y1, x2, y2] = bounds;
  const width = Math.max(0.001, x2 - x1);
  const height = Math.max(0.001, y2 - y1);
  return (
    <div className="relative w-full h-full overflow-hidden">
      <img
        src={screenshotUrl}
        alt={alt}
        draggable={false}
        loading="lazy"
        style={{
          position: 'absolute',
          width: `${100 / width}%`,
          height: `${100 / height}%`,
          left: `-${(x1 / width) * 100}%`,
          top: `-${(y1 / height) * 100}%`,
          maxWidth: 'none',
          maxHeight: 'none',
        }}
      />
    </div>
  );
}
