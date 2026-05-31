import { useState, useMemo, type MouseEvent } from 'react';
import { Layers, Search, RefreshCw, Loader2, AlertCircle, Monitor, Clock, Filter, LayoutList, TreePine, ChevronDown, ChevronRight, Trash2 } from 'lucide-react';
import { useCanvases, useDeleteCanvas } from '../hooks/useCanvases';
import PageModelBrowser from './PageModelBrowser';
import type { CanvasSummary } from '../api/types';
import { t, surfaceTypeLabels, providerLabels } from '../i18n/zh';
import { formatRelativeTime } from '../utils/formatTime';

type ViewMode = 'model' | 'flat';

interface CanvasBrowserProps {
  onSelectCanvas: (canvasId: string) => void;
  onSelectStateTemplate?: (stateTemplateId: string) => void;
  onDeletedCanvas?: (canvasId: string) => void;
  onDeletedStateTemplate?: (stateTemplateId: string) => void;
  onDeletedPageModel?: (pageModelId: string) => void;
  viewMode?: ViewMode;
  onViewModeChange?: (mode: ViewMode) => void;
}

const surfaceTypeLabel = surfaceTypeLabels;

function processingStateLabel(state?: string): string {
  switch (state) {
    case 'enhanced_ready':
      return '已增强';
    case 'semantic_pending':
      return 'ROI 语义中';
    case 'semantic_ready':
      return 'ROI 完成';
    case 'semantic_timeout':
      return 'ROI 超时';
    case 'semantic_partial':
      return 'ROI 部分';
    case 'semantic_failed':
      return 'ROI 失败';
    case 'semantic_late_merged':
      return 'ROI 晚到';
    case 'semantic_late_failed':
      return 'ROI 晚到失败';
    default:
      return state || '';
  }
}

function processingStateTone(state?: string): 'danger' | 'warning' | 'info' | 'success' {
  if (state === 'failed' || state === 'semantic_failed' || state === 'semantic_late_failed') return 'danger';
  if (state === 'semantic_timeout' || state === 'semantic_partial') return 'warning';
  if (state === 'enhanced_ready' || state === 'semantic_ready' || state === 'semantic_late_merged') return 'success';
  return 'info';
}

export default function CanvasBrowser({
  onSelectCanvas,
  onSelectStateTemplate,
  onDeletedCanvas,
  onDeletedStateTemplate,
  onDeletedPageModel,
  viewMode: controlledViewMode,
  onViewModeChange,
}: CanvasBrowserProps) {
  const [localViewMode, setLocalViewMode] = useState<ViewMode>('model');
  const viewMode = controlledViewMode ?? localViewMode;
  const setViewMode = (mode: ViewMode) => {
    setLocalViewMode(mode);
    onViewModeChange?.(mode);
  };

  return (
    <div className="flex flex-col h-full overflow-hidden">
      {/* View mode toggle */}
      <div
        className="px-2 py-1.5 flex items-center gap-1 border-b"
        style={{ borderColor: 'var(--prc-border)', background: 'var(--prc-surface-1)' }}
      >
        <button
          className="prc-btn prc-btn-ghost h-6 px-2 flex items-center gap-1"
          style={{
            background: viewMode === 'model' ? 'var(--prc-accent)' : 'transparent',
            color: viewMode === 'model' ? 'white' : 'var(--prc-text-dim)',
          }}
          onClick={() => setViewMode('model')}
        >
          <TreePine size={10} />
          <span style={{ fontSize: '0.65rem' }}>模型</span>
        </button>
        <button
          className="prc-btn prc-btn-ghost h-6 px-2 flex items-center gap-1"
          style={{
            background: viewMode === 'flat' ? 'var(--prc-accent)' : 'transparent',
            color: viewMode === 'flat' ? 'white' : 'var(--prc-text-dim)',
          }}
          onClick={() => setViewMode('flat')}
        >
          <LayoutList size={10} />
          <span style={{ fontSize: '0.65rem' }}>观察截图</span>
        </button>
      </div>

      {viewMode === 'model' ? (
        <PageModelBrowser
          onSelectCanvas={onSelectCanvas}
          onSelectStateTemplate={onSelectStateTemplate}
          onDeletedCanvas={onDeletedCanvas}
          onDeletedStateTemplate={onDeletedStateTemplate}
          onDeletedPageModel={onDeletedPageModel}
        />
      ) : (
        <FlatCanvasBrowser onSelectCanvas={onSelectCanvas} onDeletedCanvas={onDeletedCanvas} />
      )}
    </div>
  );
}

/** Group key: `${app_id}::${window_title}` */
function groupKey(c: CanvasSummary): string {
  return `${c.app_id ?? 'unknown'}::${c.window_title || '(无标题)'}`;
}

function processStem(processName: string | null | undefined): string {
  return (processName || '').replace(/\.exe$/i, '');
}

function compactName(value: string | null | undefined): string {
  return (value || '').replace(/(应用界面|聊天页面|联系人页面|文档编辑页|主界面|设置页面|搜索页面|信息流页面|详情页面|网页|文档页面|表单页面|仪表盘)$/u, '').trim();
}

function canvasDisplayTitle(c: CanvasSummary): string {
  const app = compactName(c.display_name) || c.app_id || processStem(c.process_name) || 'unknown';
  const title = (c.window_title || '').trim();
  if (!title || title === app || title.toLowerCase() === app.toLowerCase()) return app;
  return `${app} - ${title}`;
}

function isTestFixtureCanvas(c: CanvasSummary): boolean {
  return (
    c.app_id === 'test_app' &&
    c.process_name === 'test_app.exe' &&
    c.window_title === 'test_app Window'
  );
}

interface CanvasGroup {
  app_id: string | null;
  display_name: string | null;
  window_title: string;
  process_name: string | null;
  canvases: CanvasSummary[];
  latest_at: string;
}

function buildGroups(canvases: CanvasSummary[]): CanvasGroup[] {
  const map = new Map<string, CanvasGroup>();
  for (const c of canvases) {
    const key = groupKey(c);
    let g = map.get(key);
    if (!g) {
      g = { app_id: c.app_id, display_name: c.display_name, window_title: c.window_title || '(无标题)', process_name: c.process_name, canvases: [], latest_at: c.captured_at };
      map.set(key, g);
    }
    g.canvases.push(c);
    if (c.captured_at > g.latest_at) g.latest_at = c.captured_at;
  }
  for (const g of map.values()) {
    g.canvases.sort((a, b) => b.captured_at.localeCompare(a.captured_at));
  }
  return Array.from(map.values()).sort((a, b) => b.latest_at.localeCompare(a.latest_at));
}

function FlatCanvasBrowser({ onSelectCanvas, onDeletedCanvas }: CanvasBrowserProps) {
  const { data: canvases, isLoading, error, refetch } = useCanvases();
  const deleteCanvasMutation = useDeleteCanvas();
  const [search, setSearch] = useState('');
  const [surfaceFilter, setSurfaceFilter] = useState<string>('all');
  const [expandedGroups, setExpandedGroups] = useState<Set<string>>(new Set());
  const screenshotCanvases = (canvases ?? [])
    .filter((c) => !isTestFixtureCanvas(c))
    .filter((c) => c.has_screenshot);

  const filtered = screenshotCanvases.filter((c) => {
    if (surfaceFilter !== 'all' && c.surface_type !== surfaceFilter) return false;
    if (!search) return true;
    const q = search.toLowerCase();
    return (
      c.canvas_id.toLowerCase().includes(q) ||
      c.window_title.toLowerCase().includes(q) ||
      (c.app_id ?? '').toLowerCase().includes(q) ||
      (c.process_name ?? '').toLowerCase().includes(q)
    );
  });

  const groups = useMemo(() => buildGroups(filtered), [filtered]);
  const surfaceTypes = Array.from(new Set(screenshotCanvases.map((c) => c.surface_type))).filter(Boolean);

  const toggleGroup = (key: string) => {
    setExpandedGroups((prev) => {
      const next = new Set(prev);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });
  };

  return (
    <div className="flex flex-col h-full overflow-hidden" style={{ borderRight: '1px solid var(--prc-border)' }}>
      {/* Header */}
      <div className="px-3 pt-3 pb-2 border-b" style={{ borderColor: 'var(--prc-border)' }}>
        <div className="flex items-center justify-between mb-2">
          <div className="flex items-center gap-1.5">
            <Layers size={12} style={{ color: 'var(--prc-accent)' }} />
            <span className="prc-text text-xs font-semibold">观察截图</span>
            <span className="prc-badge mono" style={{ background: 'var(--prc-surface-2)', color: 'var(--prc-text-dim)' }}>
              {filtered.length}
            </span>
            <span className="prc-text-dim" style={{ fontSize: '0.58rem' }}>
              {groups.length} 组
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

        {/* Search */}
        <div className="flex items-center gap-2 rounded px-2.5 h-7 mb-2" style={{ background: 'var(--prc-surface-2)', border: '1px solid var(--prc-border)' }}>
          <Search size={11} className="prc-text-dim flex-shrink-0" />
          <input
            className="flex-1 bg-transparent prc-text text-xs outline-none"
            style={{ color: 'var(--prc-text)', fontSize: '0.75rem' }}
            placeholder="搜索 canvas_id / 窗口标题 / app / 进程..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
          />
        </div>

        {/* Surface type filter */}
        {surfaceTypes.length > 1 && (
          <div className="flex items-center gap-1 flex-wrap">
            <Filter size={10} className="prc-text-dim flex-shrink-0" />
            <button
              className="prc-badge cursor-pointer transition-all"
              style={{
                background: surfaceFilter === 'all' ? 'var(--prc-accent)' : 'var(--prc-surface-2)',
                color: surfaceFilter === 'all' ? 'white' : 'var(--prc-text-dim)',
                border: `1px solid ${surfaceFilter === 'all' ? 'var(--prc-accent)' : 'var(--prc-border)'}`,
                padding: '1px 6px',
                fontSize: '0.6rem',
              }}
              onClick={() => setSurfaceFilter('all')}
            >
              全部
            </button>
            {surfaceTypes.map((st) => (
              <button
                key={st}
                className="prc-badge cursor-pointer transition-all"
                style={{
                  background: surfaceFilter === st ? 'var(--prc-accent)' : 'var(--prc-surface-2)',
                  color: surfaceFilter === st ? 'white' : 'var(--prc-text-dim)',
                  border: `1px solid ${surfaceFilter === st ? 'var(--prc-accent)' : 'var(--prc-border)'}`,
                  padding: '1px 6px',
                  fontSize: '0.6rem',
                }}
                onClick={() => setSurfaceFilter(st)}
              >
                {surfaceTypeLabel[st] ?? st}
              </button>
            ))}
          </div>
        )}
      </div>

      {/* Grouped Canvas List */}
      <div className="flex-1 min-h-0 overflow-y-auto">
        {isLoading && (
          <div className="flex flex-col items-center justify-center py-10">
            <Loader2 size={20} className="prc-text-dim mb-2 animate-spin" />
            <span className="prc-text-dim text-xs">加载画布列表...</span>
          </div>
        )}

        {error && (
          <div className="flex flex-col items-center justify-center py-10 px-4">
            <AlertCircle size={20} style={{ color: 'var(--prc-danger)' }} className="mb-2" />
            <span className="prc-text-dim text-xs text-center">加载失败: {error.message}</span>
          </div>
        )}

        {!isLoading && !error && groups.length === 0 && (
          <div className="flex flex-col items-center justify-center py-10">
            <Layers size={20} className="prc-text-dim mb-2" />
            <span className="prc-text-dim text-xs">暂无观察截图</span>
            <span className="prc-text-dim text-xs mt-1" style={{ fontSize: '0.65rem' }}>请先 {t('observe', '观察')} 一个窗口</span>
          </div>
        )}

        {groups.map((g) => {
          const gk = groupKey(g.canvases[0]);
          const collapsed = !expandedGroups.has(gk);
          const latest = g.canvases[0];
          const title = canvasDisplayTitle(latest);
          const surfaceLabel = surfaceTypeLabel[latest.surface_type] ?? latest.surface_type;
          return (
            <div key={gk}>
              {/* Group header */}
              <div
                className="px-3 py-2 cursor-pointer flex items-center gap-1.5 sticky top-0"
                style={{
                  background: 'var(--prc-surface-1)',
                  borderBottom: '1px solid var(--prc-border)',
                  zIndex: 1,
                }}
                onClick={() => toggleGroup(gk)}
              >
                {collapsed ? <ChevronRight size={12} className="prc-text-dim flex-shrink-0" /> : <ChevronDown size={12} className="prc-text-dim flex-shrink-0" />}
                <Monitor size={12} style={{ color: 'var(--prc-accent)' }} className="flex-shrink-0" />
                <span className="prc-text text-xs font-semibold truncate min-w-0 flex-1" title={title}>
                  {title}
                </span>
                <span className="prc-badge flex-shrink-0" style={{ background: 'var(--prc-surface-2)', color: 'var(--prc-accent)', fontSize: '0.62rem' }}>
                  {surfaceLabel}
                </span>
                <span className="prc-badge mono flex-shrink-0" style={{ background: 'var(--prc-surface-2)', color: 'var(--prc-text-dim)', fontSize: '0.6rem' }}>
                  {g.canvases.length}
                </span>
              </div>
              {/* Group items */}
              {!collapsed && g.canvases.map((c) => (
                <CanvasCard
                  key={c.canvas_id}
                  canvas={c}
                  onClick={() => onSelectCanvas(c.canvas_id)}
                  onDelete={() => deleteCanvasMutation.mutate(c.canvas_id, {
                    onSuccess: () => onDeletedCanvas?.(c.canvas_id),
                  })}
                  deleting={deleteCanvasMutation.isPending && deleteCanvasMutation.variables === c.canvas_id}
                />
              ))}
            </div>
          );
        })}
      </div>
    </div>
  );
}

function CanvasCard({
  canvas: c,
  onClick,
  onDelete,
  deleting,
}: {
  canvas: CanvasSummary;
  onClick: () => void;
  onDelete: () => void;
  deleting: boolean;
}) {
  const title = canvasDisplayTitle(c);
  const handleDelete = (event: MouseEvent) => {
    event.stopPropagation();
    const ok = window.confirm(`删除这张观察截图？\n${c.window_title || c.canvas_id}`);
    if (ok) onDelete();
  };

  return (
    <div
      className="px-3 py-3 cursor-pointer transition-colors"
      style={{ borderBottom: '1px solid var(--prc-border)' }}
      onClick={onClick}
    >
      <div className="flex items-start justify-between gap-2">
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-1.5">
            <Monitor size={12} style={{ color: 'var(--prc-accent)' }} />
            <span className="prc-text font-medium truncate" style={{ fontSize: '0.78rem' }} title={title}>{title}</span>
          </div>
          <div className="flex items-center gap-2 mt-0.5 flex-wrap">
            {c.app_id && <span className="prc-text-dim mono" style={{ fontSize: '0.66rem' }}>{c.app_id}</span>}
            {c.process_name && <span className="prc-text-dim mono" style={{ fontSize: '0.66rem' }}>{c.process_name}</span>}
          </div>
          <div className="flex items-center gap-3 mt-1.5">
            <span className="mono prc-text-dim" style={{ fontSize: '0.64rem' }}>
              {c.element_count} 元素
            </span>
            <span className="mono prc-text-dim" style={{ fontSize: '0.64rem' }}>
              {c.region_count} 区域
            </span>
            <div className="flex items-center gap-1">
              <Clock size={9} className="prc-text-dim" />
              <span className="mono prc-text-dim" style={{ fontSize: '0.64rem' }}>
                {formatRelativeTime(c.captured_at)}
              </span>
            </div>
          </div>
          {/* Provider tags */}
          <div className="flex items-center gap-1 mt-1.5 flex-wrap">
            {c.providers_used.map((p) => (
              <span
                key={p}
                className="prc-badge"
                style={{
                  background: 'rgba(34,197,94,0.1)',
                  color: 'var(--prc-confirmed)',
                  fontSize: '0.62rem',
                  padding: '1px 5px',
                }}
              >
                {providerLabels[p] ?? p}
              </span>
            ))}
            {c.has_screenshot && (
              <span
                className="prc-badge"
                style={{
                  background: 'rgba(59,130,246,0.1)',
                  color: 'var(--prc-accent)',
                  fontSize: '0.62rem',
                  padding: '1px 5px',
                }}
              >
                截图
              </span>
            )}
          </div>
        </div>
        {/* Status indicators */}
        <div className="flex flex-col items-end gap-1 flex-shrink-0">
          <button
            className="prc-btn prc-btn-ghost h-6 w-6 p-0 inline-flex items-center justify-center flex-shrink-0"
            title="删除观察截图"
            onClick={handleDelete}
            disabled={deleting}
            style={{
              color: '#fca5a5',
              background: 'rgba(239,68,68,0.12)',
              borderColor: 'rgba(239,68,68,0.45)',
            }}
          >
            {deleting ? (
              <Loader2 size={14} className="animate-spin flex-shrink-0" />
            ) : (
              <Trash2 size={14} strokeWidth={2.4} className="flex-shrink-0" style={{ width: 14, minWidth: 14, height: 14 }} />
            )}
          </button>
          {c.stable && (
            <span className="prc-badge" style={{ background: 'rgba(34,197,94,0.12)', color: 'var(--prc-confirmed)', fontSize: '0.62rem' }}>
              稳定
            </span>
          )}
          {c.loading && (
            <span className="prc-badge" style={{ background: 'rgba(245,158,11,0.12)', color: 'var(--prc-warning)', fontSize: '0.62rem' }}>
              加载中
            </span>
          )}
          {c.partial && (
            <span className="prc-badge" style={{ background: 'rgba(245,158,11,0.12)', color: 'var(--prc-warning)', fontSize: '0.62rem' }}>
              部分
            </span>
          )}
          {c.processing_state && c.processing_state !== 'local_ready' && (
            <span
              className="prc-badge"
              title={c.processing_error || c.processing_state}
              style={{
                background: processingStateTone(c.processing_state) === 'danger'
                  ? 'rgba(239,68,68,0.12)'
                  : processingStateTone(c.processing_state) === 'warning'
                    ? 'rgba(245,158,11,0.12)'
                    : processingStateTone(c.processing_state) === 'success'
                      ? 'rgba(34,197,94,0.12)'
                      : 'rgba(59,130,246,0.12)',
                color: processingStateTone(c.processing_state) === 'danger'
                  ? 'var(--prc-danger)'
                  : processingStateTone(c.processing_state) === 'warning'
                    ? 'var(--prc-warning)'
                    : processingStateTone(c.processing_state) === 'success'
                      ? 'var(--prc-confirmed)'
                      : 'var(--prc-accent)',
                fontSize: '0.62rem',
              }}
            >
              {processingStateLabel(c.processing_state)}
            </span>
          )}
        </div>
      </div>
    </div>
  );
}
