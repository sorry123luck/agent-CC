import { useState, useMemo, type MouseEvent } from 'react';
import {
  ChevronRight,
  ChevronDown,
  Layers,
  Monitor,
  Clock,
  Loader2,
  AlertCircle,
  RefreshCw,
  AppWindow,
  FileStack,
  Camera,
  Trash2,
} from 'lucide-react';
import { useDeletePageModel, useDeleteStateTemplate, usePageModelTree } from '../hooks/usePageModels';
import { useDeleteCanvas } from '../hooks/useCanvases';
import type {
  PageModelResponse,
  StateTemplateResponse,
  CanvasSnapshotResponse,
} from '../api/types';
import { t } from '../i18n/zh';
import { formatRelativeTime } from '../utils/formatTime';

interface PageModelBrowserProps {
  onSelectCanvas: (canvasId: string) => void;
  onSelectStateTemplate?: (stateTemplateId: string) => void;
  onDeletedCanvas?: (canvasId: string) => void;
  onDeletedStateTemplate?: (stateTemplateId: string) => void;
  onDeletedPageModel?: (pageModelId: string) => void;
}

function cleanModelDisplayName(value: string | null | undefined, fallback: string): string {
  const raw = (value || fallback || '').trim();
  if (!raw) return fallback;
  return raw.replace(/(应用界面|聊天页面|联系人页面|文档编辑页|主界面|设置页面|搜索页面|信息流页面|详情页面|网页|文档页面|表单页面|仪表盘)$/u, '') || raw;
}

function sameDisplayName(a: string | null | undefined, b: string | null | undefined): boolean {
  return (a || '').trim().toLowerCase() === (b || '').trim().toLowerCase();
}

function isUnknownAppGroup(appId: string): boolean {
  return appId.trim().toLowerCase() === 'unknown';
}

function formatAppGroupLabel(appId: string, displayName: string | null | undefined): string {
  if (isUnknownAppGroup(appId)) return '未识别应用';
  return cleanModelDisplayName(displayName, appId);
}

function formatAppGroupTitle(appId: string, label: string): string {
  if (!isUnknownAppGroup(appId)) return label || appId;
  return 'app_id=unknown，通常来自窗口句柄无效、截图为空或应用识别失败；建议作为诊断样本保留或手动清理。';
}

export default function PageModelBrowser({
  onSelectCanvas,
  onSelectStateTemplate,
  onDeletedCanvas,
  onDeletedStateTemplate,
  onDeletedPageModel,
}: PageModelBrowserProps) {
  const { data, isLoading, error, refetch } = usePageModelTree();
  const deletePageModelMutation = useDeletePageModel();
  const deleteStateTemplateMutation = useDeleteStateTemplate();
  const deleteCanvasMutation = useDeleteCanvas();
  const [expandedApps, setExpandedApps] = useState<Set<string>>(new Set());
  const [expandedModels, setExpandedModels] = useState<Set<string>>(new Set());
  const [expandedStates, setExpandedStates] = useState<Set<string>>(new Set());

  const toggleApp = (id: string) => {
    setExpandedApps((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  const toggleModel = (id: string) => {
    setExpandedModels((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  const toggleState = (id: string) => {
    setExpandedStates((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  const handleDeletePageModel = (event: MouseEvent, pm: PageModelResponse) => {
    event.stopPropagation();
    const ok = window.confirm(`删除这个页面模型及其所有状态和截图？\n${pm.display_name ?? pm.page_class_prefix}`);
    if (ok) {
      deletePageModelMutation.mutate(pm.page_model_id, {
        onSuccess: () => onDeletedPageModel?.(pm.page_model_id),
      });
    }
  };

  const handleDeleteStateTemplate = (event: MouseEvent, st: StateTemplateResponse) => {
    event.stopPropagation();
    const ok = window.confirm(`删除这个页面状态及其截图？\n${st.state_label ?? st.page_class}`);
    if (ok) {
      deleteStateTemplateMutation.mutate(st.state_template_id, {
        onSuccess: () => onDeletedStateTemplate?.(st.state_template_id),
      });
    }
  };

  const handleDeleteCanvas = (event: MouseEvent, cs: CanvasSnapshotResponse) => {
    event.stopPropagation();
    const ok = window.confirm(`删除这张观察截图？\n${cs.canvas_id}`);
    if (ok) {
      deleteCanvasMutation.mutate(cs.canvas_id, {
        onSuccess: () => onDeletedCanvas?.(cs.canvas_id),
      });
    }
  };

  const validModelIds = useMemo(() => {
    const ids = new Set<string>();
    for (const st of data?.state_templates ?? []) {
      ids.add(st.page_model_id);
    }
    for (const cs of data?.canvas_snapshots ?? []) {
      ids.add(cs.page_model_id);
    }
    return ids;
  }, [data]);

  const appGroups = useMemo(() => {
    const map = new Map<string, PageModelResponse[]>();
    for (const pm of data?.page_models ?? []) {
      if (!validModelIds.has(pm.page_model_id)) continue;
      const group = map.get(pm.app_id) ?? [];
      group.push(pm);
      map.set(pm.app_id, group);
    }
    return map;
  }, [data, validModelIds]);

  const appGroupLabels = useMemo(() => {
    const labels = new Map<string, string>();
    for (const [appId, models] of appGroups.entries()) {
      const display = models.find((pm) => pm.display_name)?.display_name;
      labels.set(appId, formatAppGroupLabel(appId, display));
    }
    return labels;
  }, [appGroups]);

  const statesByModel = useMemo(() => {
    const map = new Map<string, StateTemplateResponse[]>();
    for (const st of data?.state_templates ?? []) {
      const group = map.get(st.page_model_id) ?? [];
      group.push(st);
      map.set(st.page_model_id, group);
    }
    return map;
  }, [data]);

  const snapshotsByState = useMemo(() => {
    const map = new Map<string, CanvasSnapshotResponse[]>();
    for (const cs of data?.canvas_snapshots ?? []) {
      const group = map.get(cs.state_template_id) ?? [];
      group.push(cs);
      map.set(cs.state_template_id, group);
    }
    return map;
  }, [data]);

  const renderStateRows = (
    states: StateTemplateResponse[],
    options: {
      statePadding: string;
      snapshotPadding: string;
    },
  ) =>
    states.map((st) => {
      const isStateExpanded = expandedStates.has(st.state_template_id);
      const snapshots = snapshotsByState.get(st.state_template_id) ?? [];

      return (
        <div key={st.state_template_id}>
          <div
            className="px-3 py-2 cursor-pointer flex items-center gap-1.5 transition-colors"
            style={{
              paddingLeft: options.statePadding,
              borderBottom: '1px solid var(--prc-border)',
            }}
            onClick={() => { toggleState(st.state_template_id); onSelectStateTemplate?.(st.state_template_id); }}
          >
            {isStateExpanded ? (
              <ChevronDown size={12} className="prc-text-dim flex-shrink-0" />
            ) : (
              <ChevronRight size={12} className="prc-text-dim flex-shrink-0" />
            )}
            <Layers size={12} style={{ color: 'var(--prc-confirmed)' }} className="flex-shrink-0" />
            <span className="prc-text text-xs font-medium truncate flex-1 min-w-0" title={st.state_label ?? st.page_class}>
              {st.state_label ?? st.page_class}
            </span>
            <span className="prc-text-dim mono flex-shrink-0 whitespace-nowrap" style={{ fontSize: '0.55rem' }}>
              {st.fixed_element_count} 控件
            </span>
            <span className="prc-text-dim mono flex-shrink-0 whitespace-nowrap" style={{ fontSize: '0.55rem' }}>
              {st.snapshot_count} 截图
            </span>
            <button
              className="prc-btn prc-btn-ghost h-6 w-6 p-0 flex-shrink-0 inline-flex items-center justify-center"
              title="删除页面状态"
              onClick={(event) => handleDeleteStateTemplate(event, st)}
              disabled={deleteStateTemplateMutation.isPending && deleteStateTemplateMutation.variables === st.state_template_id}
              style={{
                color: '#fca5a5',
                background: 'rgba(239,68,68,0.12)',
                borderColor: 'rgba(239,68,68,0.45)',
              }}
            >
              {deleteStateTemplateMutation.isPending && deleteStateTemplateMutation.variables === st.state_template_id ? (
                <Loader2 size={13} className="animate-spin" />
              ) : (
                <Trash2 size={13} strokeWidth={2.4} className="flex-shrink-0" style={{ width: 13, minWidth: 13, height: 13 }} />
              )}
            </button>
          </div>

          {isStateExpanded &&
            snapshots.map((cs) => (
              <div
                key={cs.snapshot_id}
                className={`px-3 py-2 flex items-center gap-1.5 transition-colors ${cs.available ? 'cursor-pointer' : ''}`}
                style={{
                  paddingLeft: options.snapshotPadding,
                  borderBottom: '1px solid var(--prc-border)',
                  opacity: cs.available ? 1 : 0.5,
                }}
                onClick={() => cs.available && onSelectCanvas(cs.canvas_id)}
              >
                <Camera size={11} className="prc-text-dim flex-shrink-0" />
                <span className="prc-text-dim mono" style={{ fontSize: '0.6rem' }}>
                  {cs.canvas_id.slice(0, 12)}...
                </span>
                <span className="prc-text-dim mono" style={{ fontSize: '0.55rem' }}>
                  {cs.element_count} 元素
                </span>
                {!cs.available && (
                  <span className="prc-badge" style={{ background: 'rgba(107,114,128,0.15)', color: 'var(--prc-muted)', fontSize: '0.5rem' }}>
                    已过期
                  </span>
                )}
                <div className="flex items-center gap-1">
                  <Clock size={8} className="prc-text-dim" />
                  <span className="prc-text-dim mono" style={{ fontSize: '0.55rem' }}>
                    {formatRelativeTime(cs.captured_at)}
                  </span>
                </div>
                <button
                  className="prc-btn prc-btn-ghost h-6 w-6 p-0 flex-shrink-0 inline-flex items-center justify-center"
                  title="删除观察截图"
                  onClick={(event) => handleDeleteCanvas(event, cs)}
                  disabled={deleteCanvasMutation.isPending && deleteCanvasMutation.variables === cs.canvas_id}
                  style={{
                    color: '#fca5a5',
                    background: 'rgba(239,68,68,0.12)',
                    borderColor: 'rgba(239,68,68,0.45)',
                    marginLeft: 'auto',
                  }}
                >
                  {deleteCanvasMutation.isPending && deleteCanvasMutation.variables === cs.canvas_id ? (
                    <Loader2 size={13} className="animate-spin" />
                  ) : (
                    <Trash2 size={13} strokeWidth={2.4} className="flex-shrink-0" style={{ width: 13, minWidth: 13, height: 13 }} />
                  )}
                </button>
              </div>
            ))}
        </div>
      );
    });

  return (
    <div className="flex flex-col h-full" style={{ borderRight: '1px solid var(--prc-border)', minHeight: 0 }}>
      <div className="px-3 pt-3 pb-2 border-b" style={{ borderColor: 'var(--prc-border)' }}>
        <div className="flex items-center justify-between mb-2">
          <div className="flex items-center gap-1.5">
            <AppWindow size={12} style={{ color: 'var(--prc-accent)' }} />
            <span className="prc-text text-xs font-semibold">{t('page_model', '软件模型')}</span>
            <span
              className="prc-badge mono"
              style={{
                background: 'var(--prc-surface-2)',
                color: 'var(--prc-text-dim)',
              }}
            >
              {validModelIds.size}
            </span>
          </div>
          <button
            className="prc-btn prc-btn-ghost h-6 px-2"
            onClick={() => refetch()}
            disabled={isLoading}
          >
            <RefreshCw size={11} className={isLoading ? 'animate-spin' : ''} />
          </button>
        </div>
      </div>

      <div className="flex-1 min-h-0 overflow-y-auto">
        {isLoading && (
          <div className="flex flex-col items-center justify-center py-10">
            <Loader2 size={20} className="prc-text-dim mb-2 animate-spin" />
            <span className="prc-text-dim text-xs">加载模型树...</span>
          </div>
        )}

        {error && (
          <div className="flex flex-col items-center justify-center py-10 px-4">
            <AlertCircle size={20} style={{ color: 'var(--prc-danger)' }} className="mb-2" />
            <span className="prc-text-dim text-xs text-center">加载失败: {error.message}</span>
          </div>
        )}

        {!isLoading && !error && appGroups.size === 0 && (
          <div className="flex flex-col items-center justify-center py-10">
            <Layers size={20} className="prc-text-dim mb-2" />
            <span className="prc-text-dim text-xs">暂无页面模型</span>
            <span className="prc-text-dim text-xs mt-1" style={{ fontSize: '0.65rem' }}>
              请先 Observe 一个窗口
            </span>
          </div>
        )}

        {Array.from(appGroups.entries()).map(([appId, models]) => {
          return (
            <div key={appId}>
              <div
                className="px-3 py-2 flex items-center gap-1.5 cursor-pointer"
                style={{
                  background: 'var(--prc-surface-1)',
                  borderBottom: '1px solid var(--prc-border)',
                }}
                onClick={() => toggleApp(appId)}
              >
                {expandedApps.has(appId) ? (
                  <ChevronDown size={12} className="prc-text-dim flex-shrink-0" />
                ) : (
                  <ChevronRight size={12} className="prc-text-dim flex-shrink-0" />
                )}
                <Monitor size={12} style={{ color: 'var(--prc-accent)' }} className="flex-shrink-0" />
                <span
                  className="prc-text text-xs font-semibold truncate min-w-0 flex-1"
                  title={formatAppGroupTitle(appId, appGroupLabels.get(appId) || appId)}
                >
                  {appGroupLabels.get(appId) || appId}
                </span>
                <span className="prc-text-dim flex-shrink-0" style={{ fontSize: '0.6rem' }}>
                  ({models.length} 模型)
                </span>
              </div>

              {expandedApps.has(appId) && models.map((pm) => {
                const isModelExpanded = expandedModels.has(pm.page_model_id);
                const states = statesByModel.get(pm.page_model_id) ?? [];
                const appLabel = appGroupLabels.get(appId) || appId;
                const modelLabel = cleanModelDisplayName(pm.display_name, pm.page_class_prefix);
                const flattenModel = models.length === 1 || sameDisplayName(modelLabel, appLabel);

                if (flattenModel) {
                  return (
                    <div key={pm.page_model_id}>
                      {renderStateRows(states, {
                        statePadding: '16px',
                        snapshotPadding: '32px',
                      })}
                    </div>
                  );
                }

                return (
                  <div key={pm.page_model_id}>
                    <div
                      className="px-3 py-2 flex items-center gap-1.5 transition-colors group"
                      style={{
                        paddingLeft: '16px',
                        borderBottom: '1px solid var(--prc-border)',
                      }}
                    >
                      <div
                        className="flex items-center gap-1.5 flex-1 min-w-0 cursor-pointer overflow-hidden"
                        onClick={() => toggleModel(pm.page_model_id)}
                        title={modelLabel}
                      >
                        {isModelExpanded ? (
                          <ChevronDown size={12} className="prc-text-dim flex-shrink-0" />
                        ) : (
                          <ChevronRight size={12} className="prc-text-dim flex-shrink-0" />
                        )}
                        <FileStack size={12} className="flex-shrink-0" style={{ color: 'var(--prc-warning)' }} />
                        <span className="prc-text text-xs font-medium truncate min-w-0">
                          {modelLabel}
                        </span>
                      </div>
                      <span className="prc-text-dim mono flex-shrink-0 whitespace-nowrap" style={{ fontSize: '0.55rem' }}>
                        {t('state_template', '状态')} {pm.state_count}
                      </span>
                      <span className="prc-text-dim mono flex-shrink-0 whitespace-nowrap" style={{ fontSize: '0.55rem' }}>
                        {pm.observe_count}x
                      </span>
                      <button
                        className="prc-btn prc-btn-ghost h-6 w-6 p-0 flex-shrink-0 inline-flex items-center justify-center"
                        title="删除页面模型"
                        onClick={(event) => handleDeletePageModel(event, pm)}
                        disabled={deletePageModelMutation.isPending && deletePageModelMutation.variables === pm.page_model_id}
                        style={{
                          color: '#fca5a5',
                          background: 'rgba(239,68,68,0.12)',
                          borderColor: 'rgba(239,68,68,0.45)',
                        }}
                      >
                        {deletePageModelMutation.isPending && deletePageModelMutation.variables === pm.page_model_id ? (
                          <Loader2 size={14} className="animate-spin" />
                        ) : (
                          <Trash2 size={14} strokeWidth={2.4} className="flex-shrink-0" style={{ width: 14, minWidth: 14, height: 14 }} />
                        )}
                      </button>
                    </div>

                    {isModelExpanded &&
                      renderStateRows(states, {
                        statePadding: '32px',
                        snapshotPadding: '48px',
                      })}
                  </div>
                );
              })}
            </div>
          );
        })}
      </div>
    </div>
  );
}
