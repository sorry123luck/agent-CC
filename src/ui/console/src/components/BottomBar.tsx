import { Activity as ActivityIcon, CheckCircle, XCircle, MinusCircle, Cpu, Clock, Wand2 } from 'lucide-react';
import type { CanvasDetail, ProviderTraceResponse } from '../api/types';
import { t, providerLabels as zhProviderLabels } from '../i18n/zh';

interface BottomBarProps {
  canvasDetail: CanvasDetail | null;
  onRunRefine?: () => void;
  refining?: boolean;
}

const providerTraceDefs: { key: keyof ProviderTraceResponse; label: string; zhKey: string }[] = [
  { key: 'uia_used', label: 'UIA', zhKey: 'uia' },
  { key: 'ocr_used', label: 'OCR', zhKey: 'ocr' },
  { key: 'vision_used', label: 'Vision', zhKey: 'vision' },
  { key: 'vlm_used', label: 'VLM', zhKey: 'vlm' },
  { key: 'dom_used', label: 'DOM', zhKey: 'dom' },
];

export default function BottomBar({ canvasDetail, onRunRefine, refining }: BottomBarProps) {
  const trace = canvasDetail?.provider_trace;
  const providersUsed = canvasDetail?.providers_used ?? [];
  const providersFailed = canvasDetail?.providers_failed ?? [];

  // Refine stats
  const refineCounts = canvasDetail ? {
    refined: canvasDetail.elements.filter((e) => e.refine_status === 'refined').length,
    uncertain: canvasDetail.elements.filter((e) => e.refine_status === 'uncertain').length,
    total: canvasDetail.elements.length,
  } : null;

  return (
    <div
      data-cmp="BottomBar"
      className="flex-shrink-0 flex items-center h-8 px-3 gap-4"
      style={{ background: 'var(--prc-surface)', borderTop: '1px solid var(--prc-border)' }}
    >
      {/* Provider status lights */}
      <div className="flex items-center gap-3">
        <div className="flex items-center gap-1">
          <ActivityIcon size={10} className="prc-text-dim" />
          <span className="prc-text-dim" style={{ fontSize: '0.6rem' }}>{t('providers', '感知源')}:</span>
        </div>
        {providerTraceDefs.map(({ key, label, zhKey }) => {
          const used = trace ? trace[key] : providersUsed.includes(label.toLowerCase());
          const failed = providersFailed.includes(label.toLowerCase());
          const color = failed ? 'var(--prc-danger)' : used ? 'var(--prc-confirmed)' : 'var(--prc-muted)';
          const Icon = failed ? XCircle : used ? CheckCircle : MinusCircle;

          return (
            <div key={key} className="flex items-center gap-1">
              <Icon size={9} style={{ color }} />
              <span
                className="mono"
                style={{
                  fontSize: '0.58rem',
                  color,
                  textDecoration: failed ? 'line-through' : 'none',
                }}
              >
                {zhProviderLabels[zhKey] ?? label}
              </span>
            </div>
          );
        })}
      </div>

      {/* Separator */}
      <div className="w-px h-4" style={{ background: 'var(--prc-border)' }} />

      {/* Canvas status */}
      <div className="flex items-center gap-3">
        {canvasDetail?.stable && (
          <div className="flex items-center gap-1">
            <div className="w-1.5 h-1.5 rounded-full" style={{ background: 'var(--prc-confirmed)' }} />
            <span className="prc-text-dim" style={{ fontSize: '0.58rem' }}>稳定</span>
          </div>
        )}
        {canvasDetail?.loading && (
          <div className="flex items-center gap-1">
            <div className="w-1.5 h-1.5 rounded-full animate-pulse" style={{ background: 'var(--prc-warning)' }} />
            <span className="prc-text-dim" style={{ fontSize: '0.58rem' }}>加载中</span>
          </div>
        )}
        {canvasDetail?.partial && (
          <div className="flex items-center gap-1">
            <div className="w-1.5 h-1.5 rounded-full" style={{ background: 'var(--prc-warning)' }} />
            <span className="prc-text-dim" style={{ fontSize: '0.58rem' }}>部分</span>
          </div>
        )}
      </div>

      {/* Element count */}
      {canvasDetail && (
        <>
          <div className="w-px h-4" style={{ background: 'var(--prc-border)' }} />
          <div className="flex items-center gap-1.5">
            <span className="mono prc-text-dim" style={{ fontSize: '0.58rem' }}>
              {canvasDetail.elements.length} 元素
            </span>
            <span className="mono prc-text-dim" style={{ fontSize: '0.58rem' }}>
              {canvasDetail.regions.length} 区域
            </span>
          </div>
        </>
      )}

      {/* Refine stats + button */}
      {canvasDetail && (
        <>
          <div className="w-px h-4" style={{ background: 'var(--prc-border)' }} />
          <div className="flex items-center gap-2">
            {refineCounts && (refineCounts.refined > 0 || refineCounts.uncertain > 0) && (
              <div className="flex items-center gap-1.5">
                {refineCounts.refined > 0 && (
                  <span className="mono" style={{ fontSize: '0.55rem', color: 'var(--prc-confirmed)' }}>
                    {refineCounts.refined} 已复核
                  </span>
                )}
                {refineCounts.uncertain > 0 && (
                  <span className="mono" style={{ fontSize: '0.55rem', color: 'var(--prc-warning)' }}>
                    {refineCounts.uncertain} 待确认
                  </span>
                )}
              </div>
            )}
            {onRunRefine && (
              <button
                className="flex items-center gap-1 px-2 py-0.5 rounded transition-colors"
                style={{
                  background: 'var(--prc-surface-2)',
                  border: '1px solid var(--prc-border)',
                  color: refining ? 'var(--prc-muted)' : 'var(--prc-accent)',
                  fontSize: '0.58rem',
                  opacity: refining ? 0.6 : 1,
                  cursor: refining ? 'not-allowed' : 'pointer',
                }}
                onClick={refining ? undefined : onRunRefine}
                disabled={refining}
                title={t('heuristic_refine', '规则复核')}
              >
                <Wand2 size={9} />
                <span>{refining ? t('refining...', '复核中...') : t('heuristic_refine', '规则复核')}</span>
              </button>
            )}
          </div>
        </>
      )}

      {/* Spacer */}
      <div className="flex-1" />

      {/* Captured time */}
      {canvasDetail?.captured_at && (
        <div className="flex items-center gap-1">
          <Clock size={9} className="prc-text-dim" />
          <span className="mono prc-text-dim" style={{ fontSize: '0.55rem' }}>
            {new Date(canvasDetail.captured_at).toLocaleTimeString()}
          </span>
        </div>
      )}
    </div>
  );
}
