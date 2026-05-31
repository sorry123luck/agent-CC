import { Monitor, ArrowLeft, Eye, Cpu, Clock, FileStack, Layers, Settings, Sparkles, Anchor } from 'lucide-react';
import type { CanvasDetail } from '../api/types';
import { t, providerLabels, surfaceTypeLabels } from '../i18n/zh';
import { providerColor } from '../utils/providerColors';

interface TopBarProps {
  view: 'windows' | 'canvas-detail';
  canvasDetail: CanvasDetail | null;
  onBack: () => void;
  isVirtualModelView?: boolean;
  vlmSemanticStatus?: string; // "success" | "cached" | "skipped" | "failed"
  onTriggerVLM?: () => void;
  onOpenSettings?: () => void;
  fixedAnchorCount?: number; // Phase 9: count of fixed_anchor candidates
}

export default function TopBar({ view, canvasDetail, onBack, isVirtualModelView, vlmSemanticStatus, onTriggerVLM, onOpenSettings, fixedAnchorCount }: TopBarProps) {
  return (
    <div
      data-cmp="TopBar"
      className="prc-surface border-b prc-border flex items-center h-10 px-3 gap-3 flex-shrink-0"
      style={{ borderBottomColor: 'var(--prc-border)' }}
    >
      {/* Logo / Back */}
      {view === 'windows' ? (
        <div className="flex items-center gap-2">
          <div className="flex items-center justify-center w-5 h-5 rounded" style={{ background: 'var(--prc-accent)' }}>
            <Eye size={11} color="white" strokeWidth={2.5} />
          </div>
          <div>
            <span className="prc-text font-semibold text-xs leading-tight">DeskCanvas Console</span>
            <span className="prc-text-dim ml-2" style={{ fontSize: '0.6rem' }}>本地调试视图</span>
          </div>
        </div>
      ) : (
        <button
          className="prc-btn prc-btn-ghost h-7 px-2 flex items-center gap-1.5"
          onClick={onBack}
        >
          <ArrowLeft size={12} />
          <span style={{ fontSize: '0.73rem' }}>
            {isVirtualModelView ? '返回主页' : '返回'}
          </span>
        </button>
      )}

      {view === 'windows' && onOpenSettings && (
        <button
          className="prc-btn prc-btn-ghost h-7 px-2 flex items-center gap-1.5 ml-auto"
          onClick={onOpenSettings}
          title="VLM 语义模型配置"
        >
          <Settings size={12} />
          <span style={{ fontSize: '0.68rem' }}>VLM 设置</span>
        </button>
      )}

      {/* Canvas metadata (only in canvas-detail view) */}
      {view === 'canvas-detail' && canvasDetail && (
        <>
          <div className="w-px h-5" style={{ background: 'var(--prc-border)' }} />

          <div className="flex items-center gap-2 min-w-0 flex-1">
            <Monitor size={11} className="prc-text-dim flex-shrink-0" />
            <span className="prc-text text-xs font-medium truncate">
              {canvasDetail.window_title || '(无标题)'}
            </span>
            {canvasDetail.app_id && (
              <span className="prc-text-dim mono" style={{ fontSize: '0.62rem' }}>{canvasDetail.app_id}</span>
            )}
            {canvasDetail.page_class && (
              <span className="prc-badge" style={{ background: 'var(--prc-surface-2)', color: 'var(--prc-text-dim)', fontSize: '0.58rem' }}>
                {canvasDetail.page_class}
              </span>
            )}
            <span className="prc-badge" style={{ background: 'var(--prc-surface-2)', color: 'var(--prc-accent)', fontSize: '0.58rem' }}>
              {surfaceTypeLabels[canvasDetail.surface_type] ?? canvasDetail.surface_type}
            </span>
            {/* E Phase 2: PageModel/StateTemplate display */}
            {canvasDetail.page_model_id && (
              <span className="prc-badge" style={{ background: 'rgba(168,85,247,0.12)', color: '#a855f7', fontSize: '0.55rem' }}>
                <FileStack size={8} className="inline mr-0.5" />
                {canvasDetail.page_model_name || canvasDetail.page_model_id.slice(0, 8)}
              </span>
            )}
            {canvasDetail.state_template_id && (
              <span className="prc-badge" style={{ background: 'rgba(34,197,94,0.12)', color: 'var(--prc-confirmed)', fontSize: '0.55rem' }}>
                <Layers size={8} className="inline mr-0.5" />
                {canvasDetail.state_label || canvasDetail.state_template_id.slice(0, 8)}
              </span>
            )}
          </div>

          {/* Provider tags */}
          <div className="flex items-center gap-1 flex-shrink-0">
            {canvasDetail.providers_used.map((p) => (
              <span
                key={p}
                className="prc-badge"
                style={{
                  background: `${providerColor[p] ?? 'var(--prc-muted)'}15`,
                  color: providerColor[p] ?? 'var(--prc-muted)',
                  fontSize: '0.55rem',
                  padding: '0 5px',
                  border: `1px solid ${providerColor[p] ?? 'var(--prc-muted)'}30`,
                }}
              >
                {providerLabels[p] ?? p}
              </span>
            ))}
            {canvasDetail.providers_failed.map((p) => (
              <span
                key={p}
                className="prc-badge"
                style={{
                  background: 'rgba(239,68,68,0.1)',
                  color: 'var(--prc-danger)',
                  fontSize: '0.55rem',
                  padding: '0 5px',
                  border: '1px solid rgba(239,68,68,0.3)',
                  textDecoration: 'line-through',
                }}
              >
                {providerLabels[p] ?? p}
              </span>
            ))}
          </div>

          {/* Status indicators */}
          <div className="flex items-center gap-2 flex-shrink-0">
            {canvasDetail.stable && (
              <span className="prc-badge" style={{ background: 'rgba(34,197,94,0.12)', color: 'var(--prc-confirmed)', fontSize: '0.55rem' }}>
                稳定
              </span>
            )}
            {canvasDetail.loading && (
              <span className="prc-badge" style={{ background: 'rgba(245,158,11,0.12)', color: 'var(--prc-warning)', fontSize: '0.55rem' }}>
                加载中
              </span>
            )}
            {canvasDetail.partial && (
              <span className="prc-badge" style={{ background: 'rgba(245,158,11,0.12)', color: 'var(--prc-warning)', fontSize: '0.55rem' }}>
                部分
              </span>
            )}
            {/* VLM Semantic status */}
            {vlmSemanticStatus === 'success' && (
              <span className="prc-badge" style={{ background: 'rgba(56,189,248,0.12)', color: '#38bdf8', fontSize: '0.55rem' }}>
                <Sparkles size={8} className="inline mr-0.5" />
                VLM
              </span>
            )}
            {vlmSemanticStatus === 'cached' && (
              <span className="prc-badge" style={{ background: 'rgba(56,189,248,0.08)', color: '#7dd3fc', fontSize: '0.55rem' }}>
                <Sparkles size={8} className="inline mr-0.5" />
                VLM 缓存
              </span>
            )}
            {vlmSemanticStatus === 'failed' && (
              <span className="prc-badge" style={{ background: 'rgba(239,68,68,0.1)', color: 'var(--prc-danger)', fontSize: '0.55rem', textDecoration: 'line-through' }}>
                VLM 失败
              </span>
            )}
            {vlmSemanticStatus === 'running' && (
              <span className="prc-badge" style={{ background: 'rgba(56,189,248,0.12)', color: '#38bdf8', fontSize: '0.55rem' }}>
                <Sparkles size={8} className="inline mr-0.5" />
                VLM 分析中
              </span>
            )}
            {fixedAnchorCount !== undefined && fixedAnchorCount > 0 && (
              <span className="prc-badge" style={{ background: 'rgba(251,191,36,0.12)', color: '#fbbf24', fontSize: '0.55rem' }}>
                <Anchor size={8} className="inline mr-0.5" />
                {fixedAnchorCount} 锚点
              </span>
            )}
            {onTriggerVLM && (
              <button
                className="prc-btn prc-btn-ghost h-5 px-1.5 flex items-center gap-0.5"
                style={{ fontSize: '0.55rem', color: '#38bdf8' }}
                onClick={onTriggerVLM}
                title="触发 VLM 语义分析"
              >
                <Sparkles size={9} />
                VLM分析
              </button>
            )}
          </div>

          {/* Window size */}
          <div className="flex items-center gap-1.5 flex-shrink-0">
            <span className="prc-text-dim mono" style={{ fontSize: '0.6rem' }}>
              {canvasDetail.window_width}×{canvasDetail.window_height}
            </span>
          </div>

          {/* Canvas ID */}
          <div className="flex items-center gap-1.5 flex-shrink-0">
            <span className="prc-text-dim mono truncate" style={{ fontSize: '0.58rem', maxWidth: '120px' }}>
              {canvasDetail.canvas_id}
            </span>
          </div>
        </>
      )}
    </div>
  );
}
