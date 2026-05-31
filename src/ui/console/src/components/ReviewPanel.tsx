import { useState } from 'react';
import { Target, MapPin, Shield, Tag, ChevronDown, ChevronUp, Info, Layers, Zap, Eye } from 'lucide-react';
import { getCandidateCropUrl } from '../api/client';
import type { CandidateResponse, CanvasDetail } from '../api/types';

interface CandidateDetailPanelProps {
  element: CandidateResponse | null;
  canvasDetail: CanvasDetail | null;
}

const providerColor: Record<string, string> = {
  uia: 'var(--prc-accent)',
  ocr: 'var(--prc-warning)',
  vision: '#a78bfa',
  vlm: '#22d3ee',
  dom: 'var(--prc-confirmed)',
  memory: 'var(--prc-muted)',
};

const riskLevelColor: Record<string, string> = {
  none: 'var(--prc-confirmed)',
  low: 'var(--prc-warning)',
  medium: '#f59e0b',
  high: 'var(--prc-danger)',
  critical: '#dc2626',
};

function isVisionPrimary(el: CandidateResponse): boolean {
  const strongSources = ['uia', 'ocr', 'dom'];
  const hasStrong = el.provider_sources.some((s) => strongSources.includes(s));
  return !hasStrong && el.provider_sources.some((s) => ['vision', 'vlm', 'omniparser'].includes(s));
}

function isIconLike(el: CandidateResponse): boolean {
  return !el.text && !el.name && !!el.bounds && el.bounds.length >= 4;
}

export default function ReviewPanel({ element = null, canvasDetail = null }: CandidateDetailPanelProps) {
  const [showAttributes, setShowAttributes] = useState(false);

  if (!element) {
    return (
      <div data-cmp="ReviewPanel" className="flex flex-col h-full prc-panel items-center justify-center" style={{ borderLeft: '1px solid var(--prc-border)' }}>
        <div className="flex flex-col items-center gap-3 px-6 text-center">
          <div className="w-10 h-10 rounded-full flex items-center justify-center" style={{ background: 'var(--prc-surface-2)', border: '1px solid var(--prc-border)' }}>
            <Target size={18} className="prc-text-dim" />
          </div>
          <div>
            <div className="prc-text text-sm font-medium mb-1">候选详情</div>
            <div className="prc-text-dim text-xs">点击画布或列表中的元素查看详情</div>
          </div>
        </div>
      </div>
    );
  }

  const confPct = Math.round(element.confidence * 100);
  const confColor = element.confidence >= 0.85 ? 'var(--prc-confirmed)' : element.confidence >= 0.6 ? 'var(--prc-warning)' : 'var(--prc-danger)';
  const riskColor = riskLevelColor[element.risk_level] ?? 'var(--prc-muted)';
  const iconic = isIconLike(element);
  const visionPrimary = isVisionPrimary(element);
  const semanticUncertain = iconic && visionPrimary;

  return (
    <div data-cmp="ReviewPanel" className="flex flex-col h-full prc-panel overflow-hidden" style={{ borderLeft: '1px solid var(--prc-border)' }}>
      {/* Header */}
      <div className="px-3 pt-3 pb-2 border-b flex-shrink-0" style={{ borderColor: 'var(--prc-border)' }}>
        <div className="flex items-start justify-between gap-2 mb-2">
          <div className="min-w-0 flex-1">
            <div className="flex items-center gap-1.5 mb-1 flex-wrap">
              {iconic && (
                <span className="prc-badge" style={{ background: 'rgba(167,139,250,0.12)', color: '#a78bfa', fontSize: '0.6rem' }}>
                  图标类
                </span>
              )}
              <span
                className="prc-badge"
                style={{
                  background: semanticUncertain ? 'rgba(245,158,11,0.1)' : 'var(--prc-surface-2)',
                  color: semanticUncertain ? 'var(--prc-warning)' : 'var(--prc-accent)',
                  fontSize: '0.6rem',
                }}
              >
                {semanticUncertain ? `${element.semantic_role || '未知'} (推测)` : element.semantic_role}
              </span>
              {element.interactable && (
                <span className="prc-badge" style={{ background: 'rgba(34,197,94,0.12)', color: 'var(--prc-confirmed)', fontSize: '0.6rem' }}>可交互</span>
              )}
              {element.from_memory && (
                <span className="prc-badge" style={{ background: 'rgba(75,85,99,0.15)', color: 'var(--prc-muted)', fontSize: '0.6rem' }}>记忆</span>
              )}
              {element.suggest_confirm && (
                <span className="prc-badge" style={{ background: 'rgba(245,158,11,0.12)', color: 'var(--prc-warning)', fontSize: '0.6rem' }}>建议确认</span>
              )}
            </div>
            <div className="prc-text text-sm font-semibold truncate">
              {element.text || element.name || (iconic ? (element.control_type || '图标') : element.element_id)}
            </div>
            {semanticUncertain && (
              <div className="mt-0.5" style={{ fontSize: '0.6rem', color: 'var(--prc-warning)', lineHeight: 1.3 }}>
                此元素主要由视觉检测发现，语义角色仅为推测，请以截图判断为准
              </div>
            )}
            <div className="prc-text-dim mono" style={{ fontSize: '0.6rem' }}>{element.element_id}</div>
          </div>
        </div>

        {/* Confidence Bar */}
        <div className="flex items-center gap-2 rounded px-2.5 py-1.5" style={{ background: 'var(--prc-surface-2)', border: '1px solid var(--prc-border)' }}>
          <span className="prc-text-dim" style={{ fontSize: '0.68rem', whiteSpace: 'nowrap' }}>置信度</span>
          <div className="flex-1 rounded-full h-1.5 overflow-hidden" style={{ background: 'var(--prc-bg)' }}>
            <div className="h-full rounded-full transition-all" style={{ width: `${confPct}%`, background: confColor }} />
          </div>
          <span className="mono font-bold" style={{ fontSize: '0.72rem', color: confColor }}>{confPct}%</span>
          <span className="prc-badge" style={{ background: `${confColor}20`, color: confColor, fontSize: '0.58rem' }}>
            {element.confidence_level}
          </span>
        </div>
      </div>

      {/* Crop Preview */}
      {canvasDetail?.canvas_id && element.bounds && element.bounds.length >= 4 && (
        <div className="px-3 py-2 border-b flex-shrink-0" style={{ borderColor: 'var(--prc-border)' }}>
          <div
            className="rounded overflow-hidden mx-auto"
            style={{
              background: 'var(--prc-surface-2)',
              border: '1px solid var(--prc-border)',
              maxHeight: '120px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
            }}
          >
            <img
              src={getCandidateCropUrl(canvasDetail.canvas_id, element.element_id, 8)}
              alt={`Crop of ${element.element_id}`}
              className="max-w-full max-h-[120px] object-contain"
              draggable={false}
              onError={(e) => { (e.target as HTMLImageElement).style.display = 'none'; }}
            />
          </div>
        </div>
      )}

      {/* Scrollable Content */}
      <div className="flex-1 overflow-y-auto">
        {/* Basic Info */}
        <div className="px-3 py-2.5 border-b" style={{ borderColor: 'var(--prc-border)' }}>
          <SectionTitle title="基本信息" />
          <div className="flex flex-col gap-1.5 mt-2">
            <InfoRow label="名称" value={<span className="prc-text text-xs">{element.name || '—'}</span>} />
            <InfoRow label="文本" value={<span className="prc-text text-xs">{element.text || '—'}</span>} />
            <InfoRow label="角色" value={<span className="prc-badge" style={{ background: 'var(--prc-surface-2)', color: 'var(--prc-accent)', fontSize: '0.6rem' }}>{element.semantic_role}</span>} />
            <InfoRow label="控件类型" value={<span className="prc-text text-xs">{element.control_type || '—'}</span>} />
            {element.region_id && (
              <InfoRow label="区域" value={<span className="prc-text text-xs">{element.region_id}</span>} />
            )}
          </div>
        </div>

        {/* Agent Refine */}
        <div className="px-3 py-2.5 border-b" style={{ borderColor: 'var(--prc-border)' }}>
          <SectionTitle title="Agent 复核" />
          {element.refine_status === 'unreviewed' && !element.role_label ? (
            <div className="mt-2 flex items-center gap-2 rounded px-2.5 py-2" style={{ background: 'var(--prc-surface-2)', border: '1px solid var(--prc-border)' }}>
              <Eye size={12} className="prc-text-dim" />
              <span className="prc-text-dim text-xs">未复核</span>
            </div>
          ) : (
            <div className="flex flex-col gap-1.5 mt-2">
              <InfoRow label="视觉类型" value={<span className="prc-text text-xs">{element.visual_type || '—'}</span>} />
              <InfoRow label="语义角色" value={
                element.role_label ? (
                  <span className="prc-badge" style={{
                    background: element.refine_status === 'refined' ? 'rgba(34,197,94,0.12)' : element.refine_status === 'uncertain' ? 'rgba(245,158,11,0.1)' : 'var(--prc-surface-2)',
                    color: element.refine_status === 'refined' ? 'var(--prc-confirmed)' : element.refine_status === 'uncertain' ? 'var(--prc-warning)' : 'var(--prc-text-dim)',
                    fontSize: '0.6rem',
                  }}>
                    {element.role_label}
                  </span>
                ) : <span className="prc-text-dim text-xs">—</span>
              } />
              {element.semantic_tags.length > 0 && (
                <InfoRow label="语义标签" value={
                  <div className="flex flex-wrap gap-1">
                    {element.semantic_tags.map((tag) => (
                      <span key={tag} className="prc-badge" style={{ background: 'rgba(99,102,241,0.1)', color: '#818cf8', fontSize: '0.58rem' }}>
                        {tag}
                      </span>
                    ))}
                  </div>
                } />
              )}
              <InfoRow label="语义置信度" value={
                <div className="flex items-center gap-2">
                  <div className="flex-1 rounded-full h-1.5 overflow-hidden" style={{ background: 'var(--prc-bg)', maxWidth: '80px' }}>
                    <div className="h-full rounded-full transition-all" style={{ width: `${Math.round(element.role_confidence * 100)}%`, background: element.role_confidence >= 0.7 ? 'var(--prc-confirmed)' : element.role_confidence >= 0.3 ? 'var(--prc-warning)' : 'var(--prc-danger)' }} />
                  </div>
                  <span className="mono" style={{ fontSize: '0.63rem', color: 'var(--prc-text-dim)' }}>{Math.round(element.role_confidence * 100)}%</span>
                </div>
              } />
              <InfoRow label="来源" value={<span className="prc-text text-xs">{element.role_source || '—'}</span>} />
              {element.role_evidence.length > 0 && (
                <InfoRow label="证据" value={
                  <div className="flex flex-col gap-0.5">
                    {element.role_evidence.map((ev, i) => (
                      <span key={i} className="prc-text-dim" style={{ fontSize: '0.6rem' }}>· {ev}</span>
                    ))}
                  </div>
                } />
              )}
              <InfoRow label="复核状态" value={
                <span className="prc-badge" style={{
                  background: element.refine_status === 'refined' ? 'rgba(34,197,94,0.12)' : element.refine_status === 'uncertain' ? 'rgba(245,158,11,0.1)' : 'var(--prc-surface-2)',
                  color: element.refine_status === 'refined' ? 'var(--prc-confirmed)' : element.refine_status === 'uncertain' ? 'var(--prc-warning)' : 'var(--prc-text-dim)',
                  fontSize: '0.6rem',
                }}>
                  {element.refine_status === 'refined' ? '已确认' : element.refine_status === 'uncertain' ? '语义待确认' : element.refine_status}
                </span>
              } />
            </div>
          )}
        </div>

        {/* Position */}
        <div className="px-3 py-2.5 border-b" style={{ borderColor: 'var(--prc-border)' }}>
          <SectionTitle title="位置" />
          <div className="flex flex-col gap-1.5 mt-2">
            {element.bounds && (
              <InfoRow label="边界框" value={
                <span className="mono prc-text-dim" style={{ fontSize: '0.63rem' }}>
                  [{element.bounds.join(', ')}]
                </span>
              } />
            )}
            {element.click_point && (
              <InfoRow label="点击坐标" value={
                <span className="mono prc-text-dim" style={{ fontSize: '0.63rem' }}>
                  ({element.click_point[0]}, {element.click_point[1]})
                </span>
              } />
            )}
          </div>
        </div>

        {/* Risk */}
        {(element.risk_tags.length > 0 || element.risk_level !== 'none') && (
          <div className="px-3 py-2.5 border-b" style={{ borderColor: 'var(--prc-border)' }}>
            <SectionTitle title="风险" />
            <div className="flex flex-col gap-1.5 mt-2">
              <InfoRow label="风险等级" value={
                <span className="prc-badge" style={{ background: `${riskColor}15`, color: riskColor, fontSize: '0.6rem' }}>
                  {element.risk_level}
                </span>
              } />
              {element.risk_tags.length > 0 && (
                <InfoRow label="风险标签" value={
                  <div className="flex flex-wrap gap-1">
                    {element.risk_tags.map((tag) => (
                      <span key={tag} className="prc-badge" style={{ background: 'rgba(239,68,68,0.1)', color: 'var(--prc-danger)', fontSize: '0.58rem' }}>
                        {tag}
                      </span>
                    ))}
                  </div>
                } />
              )}
            </div>
          </div>
        )}

        {/* Provider Sources */}
        <div className="px-3 py-2.5 border-b" style={{ borderColor: 'var(--prc-border)' }}>
          <SectionTitle title="识别来源" />
          <div className="flex flex-col gap-1.5 mt-2">
            <div className="flex items-center gap-2 rounded px-2.5 py-2" style={{ background: 'var(--prc-surface-2)', border: '1px solid var(--prc-border)' }}>
              <div className="flex items-center gap-2 flex-wrap">
                {element.provider_sources.map((p) => (
                  <div key={p} className="flex items-center gap-1.5">
                    <div className="w-2 h-2 rounded-full" style={{ background: providerColor[p] ?? 'var(--prc-muted)' }} />
                    <span className="mono" style={{ fontSize: '0.63rem', color: providerColor[p] ?? 'var(--prc-muted)', fontWeight: 600 }}>
                      {p.toUpperCase()}
                    </span>
                  </div>
                ))}
              </div>
            </div>
            {element.locator_ids.length > 0 && (
              <InfoRow label="定位器" value={
                <div className="flex flex-wrap gap-1">
                  {element.locator_ids.map((lid) => (
                    <span key={lid} className="prc-badge mono" style={{ background: 'var(--prc-surface-2)', color: 'var(--prc-text-dim)', fontSize: '0.58rem' }}>
                      {lid}
                    </span>
                  ))}
                </div>
              } />
            )}
          </div>
        </div>

        {/* Attributes (collapsible) */}
        {Object.keys(element.attributes).length > 0 && (
          <div className="px-3 py-2.5 border-b" style={{ borderColor: 'var(--prc-border)' }}>
            <button className="flex items-center justify-between w-full" onClick={() => setShowAttributes(!showAttributes)}>
              <SectionTitle title="属性" />
              {showAttributes ? <ChevronUp size={12} className="prc-text-dim" /> : <ChevronDown size={12} className="prc-text-dim" />}
            </button>
            {showAttributes && (
              <div className="mt-2 rounded px-2.5 py-2 overflow-x-auto" style={{ background: 'var(--prc-surface-2)', border: '1px solid var(--prc-border)' }}>
                <pre className="mono prc-text-dim" style={{ fontSize: '0.6rem', whiteSpace: 'pre-wrap', wordBreak: 'break-all' }}>
                  {JSON.stringify(element.attributes, null, 2)}
                </pre>
              </div>
            )}
          </div>
        )}

        {/* Status flags */}
        <div className="px-3 py-2.5">
          <SectionTitle title="状态" />
          <div className="flex flex-col gap-1.5 mt-2">
            <div className="flex items-center gap-2">
              <StatusFlag label="可交互" value={element.interactable} />
              <StatusFlag label="来自记忆" value={element.from_memory} />
              <StatusFlag label="建议确认" value={element.suggest_confirm} />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function SectionTitle({ title }: { title: string }) {
  return (
    <div className="flex items-center gap-1.5">
      <div className="w-1 h-3 rounded-full flex-shrink-0" style={{ background: 'var(--prc-accent)' }} />
      <span className="prc-text-dim text-xs font-semibold uppercase tracking-wider" style={{ fontSize: '0.65rem' }}>{title}</span>
    </div>
  );
}

function InfoRow({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div className="flex items-start gap-2">
      <span className="prc-text-dim flex-shrink-0" style={{ fontSize: '0.68rem', width: '72px', paddingTop: '1px' }}>{label}</span>
      <div className="flex-1 min-w-0">{value}</div>
    </div>
  );
}

function StatusFlag({ label, value }: { label: string; value: boolean }) {
  return (
    <div
      className="flex items-center gap-1.5 px-2 py-1 rounded"
      style={{
        background: value ? 'rgba(34,197,94,0.1)' : 'rgba(75,85,99,0.1)',
        border: `1px solid ${value ? 'rgba(34,197,94,0.3)' : 'var(--prc-border)'}`,
      }}
    >
      <div className="w-1.5 h-1.5 rounded-full" style={{ background: value ? 'var(--prc-confirmed)' : 'var(--prc-muted)' }} />
      <span className="prc-text-dim" style={{ fontSize: '0.62rem' }}>{label}</span>
    </div>
  );
}
