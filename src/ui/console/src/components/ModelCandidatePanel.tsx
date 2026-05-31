/**
 * ModelCandidatePanel — detail panel for a selected candidate in the virtual model.
 *
 * Separate from ReviewPanel because VirtualModelCandidate and Canvas CandidateResponse
 * have different structures. This panel reads from the persistent virtual model (SQLite),
 * not from CanvasCache.
 */

import { useEffect, useState } from 'react';
import { X, Tag, Eye, Layers, MapPin, BarChart3, Shield, Edit3, Zap, Anchor, Pin } from 'lucide-react';
import type { ControlTransitionResponse, VirtualModelCandidate, ConfidenceProfile } from '../api/types';
import { t, formatStableKeyId } from '../i18n/zh';
import {
  type CandidateEdit,
  getCandidateKind,
  normalizeModelCandidate,
  normalizeRelativeBounds,
} from '../utils/modelCandidates';

interface ModelCandidatePanelProps {
  candidate: VirtualModelCandidate | null;
  onClose?: () => void;
  onToggleHidden?: (keyId: string) => void;
  onRelabel?: (keyId: string, newLabel: string) => void;
  onMarkDynamic?: (keyId: string) => void;
  onUpdateEdit?: (keyId: string, patch: CandidateEdit) => void;
  edit?: CandidateEdit;
  availableRegions?: string[];
  controlTransitions?: ControlTransitionResponse[];
}

export default function ModelCandidatePanel({
  candidate,
  onClose,
  onToggleHidden,
  onRelabel,
  onMarkDynamic,
  onUpdateEdit,
  edit,
  availableRegions = [],
  controlTransitions = [],
}: ModelCandidatePanelProps) {
  if (!candidate) {
    return (
      <div
        data-cmp="ModelCandidatePanel"
        className="flex flex-col h-full items-center justify-center px-4"
        style={{ background: 'var(--prc-surface-1)' }}
      >
        <Eye size={20} className="prc-text-dim mb-2" />
        <span className="prc-text-dim text-xs text-center">
          点击模型中的控件查看详情
        </span>
      </div>
    );
  }

  const safeCandidate = normalizeModelCandidate(candidate);
  const roleLabel = safeCandidate.role_label || t(safeCandidate.canonical_role, safeCandidate.canonical_role);
  const kind = getCandidateKind(safeCandidate, edit);
  const isDynamic = kind === 'dynamic';

  return (
    <div
      data-cmp="ModelCandidatePanel"
      className="flex flex-col h-full overflow-y-auto"
      style={{ background: 'var(--prc-surface-1)' }}
    >
      {/* Header */}
      <div
        className="flex items-center justify-between px-3 py-2 border-b flex-shrink-0"
        style={{ borderColor: 'var(--prc-border)' }}
      >
        <div className="flex items-center gap-1.5 min-w-0">
          <Tag size={11} style={{ color: 'var(--prc-accent)' }} />
          <span className="prc-text text-xs font-semibold truncate">{roleLabel}</span>
          {isDynamic && (
            <span className="prc-badge" style={{ background: 'rgba(148,163,184,0.15)', color: '#94a3b8', fontSize: '0.55rem' }}>
              动态
            </span>
          )}
          {safeCandidate.provider_sources.includes('vlm') && (
            <span className="prc-badge" style={{ background: 'rgba(56,189,248,0.12)', color: '#38bdf8', fontSize: '0.55rem' }}>
              VLM
            </span>
          )}
          <PermanenceBadge state={safeCandidate.permanence_state} />
        </div>
        {onClose && (
          <button
            className="prc-btn prc-btn-ghost h-5 w-5 flex items-center justify-center"
            onClick={onClose}
          >
            <X size={11} />
          </button>
        )}
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto px-3 py-2 space-y-3">
        {/* Identity */}
        <Section title="身份" icon={<Shield size={10} />}>
          <Field label="稳定标识" value={formatStableKeyId(safeCandidate.key_id)} />
          {safeCandidate.element_id && <Field label="元素ID" value={safeCandidate.element_id} />}
          <Field label="语义角色" value={safeCandidate.canonical_role} />
          {safeCandidate.role_label && (
            <Field label="角色标签" value={safeCandidate.role_label} />
          )}
          <Field label="视觉类型" value={safeCandidate.visual_type || '-'} />
          {safeCandidate.semantic_tags.length > 0 && (
            <Field label="语义标签" value={safeCandidate.semantic_tags.join(', ')} />
          )}
        </Section>

        {/* Location */}
        <Section title="位置" icon={<MapPin size={10} />}>
          <Field label="区域" value={safeCandidate.canonical_region} />
          <Field
            label="相对坐标"
            value={
              safeCandidate.relative_bounds.length >= 4
                ? `[${safeCandidate.relative_bounds.map((v) => v.toFixed(3)).join(', ')}]`
                : '-'
            }
            mono
          />
          {safeCandidate.absolute_bounds && (
            <Field
              label="截图坐标"
              value={`[${safeCandidate.absolute_bounds.join(', ')}]`}
              mono
            />
          )}
        </Section>

        {/* Statistics */}
        <Section title="统计" icon={<BarChart3 size={10} />}>
          <Field label="可信度" value={`${(safeCandidate.confidence * 100).toFixed(0)}%`} />
          <Field label="出现次数" value={String(safeCandidate.seen_count)} />
          <Field label="验证次数" value={String(safeCandidate.verify_count)} />
          {safeCandidate.provider_sources.length > 0 && (
            <Field label="感知来源" value={safeCandidate.provider_sources.join(', ')} />
          )}
          {safeCandidate.confidence_profile && (
            <ConfidenceProfileBars profile={safeCandidate.confidence_profile} />
          )}
          {safeCandidate.confidence_profile && (safeCandidate.confidence_profile.source_count > 0) && (
            <Field
              label="证据来源"
              value={`${safeCandidate.confidence_profile.source_count} 条证据 / ${safeCandidate.confidence_profile.source_diversity} 个来源`}
            />
          )}
        </Section>

        {/* Classification */}
        <Section title="分类" icon={<Layers size={10} />}>
          <Field
            label="控件性质"
            value={kind === 'fixed' ? '固定控件' : kind === 'container' ? '结构区域' : kind === 'ignored' ? '已忽略' : '动态内容'}
          />
          {safeCandidate.canonical_text && (
            <Field label="标准文本" value={safeCandidate.canonical_text} />
          )}
        </Section>

        {controlTransitions.length > 0 && (
          <Section title="页面跳转" icon={<Zap size={10} />}>
            {controlTransitions.slice(0, 5).map((edge, index) => (
              <div key={`${edge.from_page_class}-${edge.to_page_class}-${index}`} className="space-y-0.5">
                <div className="prc-text" style={{ fontSize: '0.62rem', lineHeight: 1.35 }}>
                  {edge.from_page_class} → {edge.to_page_class}
                </div>
                <div className="prc-text-dim mono" style={{ fontSize: '0.52rem' }}>
                  {edge.action_type || 'click'} · {Math.round(edge.success_rate * 100)}% · {edge.observe_count} 次
                </div>
              </div>
            ))}
          </Section>
        )}

        {onUpdateEdit && (
          <Section title="人工修正" icon={<Edit3 size={10} />}>
            <CorrectionEditor
              candidate={safeCandidate}
              edit={edit}
              availableRegions={availableRegions}
              onUpdate={(patch) => onUpdateEdit(safeCandidate.key_id, patch)}
            />
          </Section>
        )}
      </div>

      {/* Actions */}
      <div
        className="flex-shrink-0 px-3 py-2 border-t space-y-1.5"
        style={{ borderColor: 'var(--prc-border)' }}
      >
        {/* Relabel */}
        {onRelabel && (
          <RelabelButton
            currentLabel={roleLabel}
            onConfirm={(newLabel) => onRelabel(safeCandidate.key_id, newLabel)}
          />
        )}

        {/* Mark as dynamic */}
        {onMarkDynamic && safeCandidate.is_fixed_control && (
          <button
            className="prc-btn prc-btn-ghost h-6 w-full text-xs flex items-center justify-center gap-1"
            style={{ color: '#94a3b8', fontSize: '0.65rem' }}
            onClick={() => onMarkDynamic(safeCandidate.key_id)}
          >
            <Zap size={10} />
            标为动态内容
          </button>
        )}

        {onUpdateEdit && (
          <button
            className="prc-btn prc-btn-ghost h-6 w-full text-xs flex items-center justify-center gap-1"
            style={{ color: 'var(--prc-confirmed)', fontSize: '0.65rem' }}
            onClick={() => onUpdateEdit(safeCandidate.key_id, {
              kind: 'fixed',
              visualType: safeCandidate.visual_type || 'button',
              semanticRole: safeCandidate.canonical_role || 'button',
            })}
          >
            <Pin size={10} />
            固定控件并显示截图映射
          </button>
        )}

        {onUpdateEdit && (
          <button
            className="prc-btn prc-btn-ghost h-6 w-full text-xs flex items-center justify-center gap-1"
            style={{ color: '#94a3b8', fontSize: '0.65rem' }}
            onClick={() => onUpdateEdit(safeCandidate.key_id, { kind: 'ignored' })}
          >
            临时忽略此候选
          </button>
        )}

        {/* Hide */}
        {onToggleHidden && (
          <button
            className="prc-btn prc-btn-ghost h-6 w-full text-xs"
            style={{ color: 'var(--prc-warning)', fontSize: '0.65rem' }}
            onClick={() => onToggleHidden(safeCandidate.key_id)}
          >
            临时隐藏此控件
          </button>
        )}
      </div>
    </div>
  );
}

// ── Sub-components ──

const roleOptions = [
  ['button', '按钮'],
  ['icon_button', '图标按钮'],
  ['menu_item', '菜单项'],
  ['nav_item', '导航项'],
  ['text_input', '文本输入框'],
  ['message_input', '消息输入框'],
  ['text', '文本'],
  ['chat_item', '聊天列表项'],
  ['message_content', '聊天正文'],
  ['document_text', '文档正文'],
  ['sidebar', '侧边栏'],
  ['toolbar', '工具栏'],
  ['title_bar', '标题栏'],
  ['container', '容器'],
  ['unknown', '未知'],
] as const;

const visualTypeOptions = [
  ['button', '按钮'],
  ['icon', '图标'],
  ['text', '文本'],
  ['input', '输入框'],
  ['panel', '面板'],
  ['list_item', '列表项'],
  ['container', '容器'],
  ['unknown', '未知'],
] as const;

function CorrectionEditor({
  candidate,
  edit,
  availableRegions,
  onUpdate,
}: {
  candidate: VirtualModelCandidate;
  edit?: CandidateEdit;
  availableRegions: string[];
  onUpdate: (patch: CandidateEdit) => void;
}) {
  const regions = Array.from(new Set([candidate.canonical_region, ...availableRegions].filter(Boolean)));
  return (
    <div className="space-y-2">
      <SelectField
        label="控件性质"
        value={edit?.kind ?? getCandidateKind(candidate)}
        options={[
          ['fixed', '固定控件'],
          ['dynamic', '动态内容'],
          ['container', '结构区域'],
          ['ignored', '忽略'],
        ]}
        onChange={(value) => onUpdate({ kind: value as CandidateEdit['kind'] })}
      />
      <SelectField
        label="语义角色"
        value={edit?.semanticRole ?? candidate.canonical_role}
        options={roleOptions}
        onChange={(value) => onUpdate({ semanticRole: value })}
      />
      <SelectField
        label="视觉类型"
        value={edit?.visualType ?? (candidate.visual_type || 'unknown')}
        options={visualTypeOptions}
        onChange={(value) => onUpdate({ visualType: value })}
      />
      <SelectField
        label="所属区域"
        value={edit?.regionId ?? candidate.canonical_region}
        options={regions.map((region) => [region, t(region, region)] as const)}
        onChange={(value) => onUpdate({ regionId: value })}
      />
      <BoundsEditor
        bounds={edit?.relativeBounds ?? candidate.relative_bounds}
        onUpdate={(relativeBounds) => onUpdate({ relativeBounds })}
      />
    </div>
  );
}

function SelectField({
  label,
  value,
  options,
  onChange,
}: {
  label: string;
  value: string;
  options: readonly (readonly [string, string])[];
  onChange: (value: string) => void;
}) {
  return (
    <label className="flex items-center gap-2">
      <span className="prc-text-dim flex-shrink-0" style={{ fontSize: '0.62rem', width: '60px' }}>{label}</span>
      <select
        className="flex-1 rounded px-2 h-6 prc-text outline-none"
        style={{ background: 'var(--prc-surface-2)', border: '1px solid var(--prc-border)', fontSize: '0.65rem' }}
        value={value}
        onChange={(e) => onChange(e.target.value)}
      >
        {options.map(([optionValue, optionLabel]) => (
          <option key={optionValue} value={optionValue}>{optionLabel}</option>
        ))}
      </select>
    </label>
  );
}

function BoundsEditor({
  bounds,
  onUpdate,
}: {
  bounds: number[];
  onUpdate: (bounds: number[]) => void;
}) {
  const [draft, setDraft] = useState(() => normalizeBounds(bounds));

  useEffect(() => {
    setDraft(normalizeBounds(bounds));
  }, [bounds]);

  const setPart = (index: number, value: string) => {
    const next = [...draft];
    next[index] = value;
    setDraft(next);
  };

  const apply = () => {
    const parsed = draft.map((v) => Number(v));
    if (parsed.length === 4 && parsed.every((v) => Number.isFinite(v))) {
      onUpdate(parsed.map((v) => Math.max(0, Math.min(1, v))));
    }
  };

  return (
    <div>
      <div className="flex items-center gap-2 mb-1">
        <span className="prc-text-dim flex-shrink-0" style={{ fontSize: '0.62rem', width: '60px' }}>相对坐标</span>
        <span className="prc-text-dim" style={{ fontSize: '0.56rem' }}>x1 / y1 / x2 / y2</span>
      </div>
      <div className="grid grid-cols-4 gap-1">
        {draft.map((value, index) => (
          <input
            key={index}
            className="rounded px-1 h-6 prc-text mono outline-none"
            style={{ background: 'var(--prc-surface-2)', border: '1px solid var(--prc-border)', fontSize: '0.58rem' }}
            value={value}
            onChange={(e) => setPart(index, e.target.value)}
            onBlur={apply}
            onKeyDown={(e) => {
              if (e.key === 'Enter') apply();
            }}
          />
        ))}
      </div>
    </div>
  );
}

function normalizeBounds(bounds: number[]): string[] {
  const source = normalizeRelativeBounds(bounds).length >= 4 ? normalizeRelativeBounds(bounds) : [0, 0, 0, 0];
  return source.slice(0, 4).map((v) => Number(v).toFixed(3));
}

function Section({
  title,
  icon,
  children,
}: {
  title: string;
  icon: React.ReactNode;
  children: React.ReactNode;
}) {
  return (
    <div>
      <div className="flex items-center gap-1 mb-1.5">
        <span className="prc-text-dim">{icon}</span>
        <span className="prc-text-dim font-medium" style={{ fontSize: '0.65rem' }}>
          {title}
        </span>
      </div>
      <div className="space-y-1 pl-3">{children}</div>
    </div>
  );
}

function Field({
  label,
  value,
  mono = false,
}: {
  label: string;
  value: string;
  mono?: boolean;
}) {
  return (
    <div className="flex items-start gap-2">
      <span
        className="prc-text-dim flex-shrink-0"
        style={{ fontSize: '0.62rem', minWidth: '60px' }}
      >
        {label}
      </span>
      <span
        className={`prc-text ${mono ? 'mono' : ''}`}
        style={{
          fontSize: '0.62rem',
          wordBreak: 'break-all',
          lineHeight: 1.4,
        }}
      >
        {value}
      </span>
    </div>
  );
}

// ── Permanence State Badge ──

const PERMANENCE_STYLES: Record<string, { bg: string; color: string; border: string; label: string }> = {
  new:          { bg: 'rgba(148,163,184,0.12)', color: '#94a3b8', border: 'none', label: '新增' },
  provisional:  { bg: 'rgba(56,189,248,0.12)',  color: '#38bdf8', border: 'none', label: '暂定' },
  stable:       { bg: 'rgba(52,211,153,0.12)',  color: '#34d399', border: 'none', label: '稳定' },
  fixed_anchor: { bg: 'rgba(251,191,36,0.15)',  color: '#fbbf24', border: '1px solid #fbbf24', label: '锚点' },
  stale:        { bg: 'rgba(251,146,60,0.12)',  color: '#fb923c', border: '1px dashed #fb923c', label: '过期' },
  degraded:     { bg: 'rgba(248,113,113,0.12)', color: '#f87171', border: '1px dashed #f87171', label: '退化' },
  retired:      { bg: 'rgba(148,163,184,0.08)', color: '#64748b', border: 'none', label: '退役' },
};

function PermanenceBadge({ state }: { state: string }) {
  const style = PERMANENCE_STYLES[state] || PERMANENCE_STYLES.new;
  return (
    <span
      className="prc-badge flex items-center gap-0.5"
      style={{
        background: style.bg,
        color: style.color,
        border: style.border,
        fontSize: '0.55rem',
      }}
    >
      {state === 'fixed_anchor' && <Anchor size={8} />}
      {style.label}
    </span>
  );
}

// ── Confidence Profile Bars ──

const DIMENSION_CONFIG: { key: keyof ConfidenceProfile; label: string; color: string }[] = [
  { key: 'spatial_confidence',        label: '位置',   color: '#60a5fa' },
  { key: 'semantic_confidence',       label: '语义',   color: '#a78bfa' },
  { key: 'visual_anchor_confidence',  label: '视觉',   color: '#34d399' },
  { key: 'memory_confidence',         label: '记忆',   color: '#fbbf24' },
  { key: 'text_confidence',           label: '文字',   color: '#f472b6' },
  { key: 'structure_confidence',      label: '结构',   color: '#38bdf8' },
  { key: 'action_confidence',         label: '执行',   color: '#fb923c' },
];

function ConfidenceProfileBars({ profile }: { profile: ConfidenceProfile }) {
  return (
    <div className="mt-1 space-y-0.5">
      {DIMENSION_CONFIG.map(({ key, label, color }) => {
        const value = profile[key] as number;
        return (
          <div key={key} className="flex items-center gap-1.5">
            <span className="prc-text-dim flex-shrink-0" style={{ fontSize: '0.56rem', width: '28px' }}>
              {label}
            </span>
            <div
              className="flex-1 rounded-full overflow-hidden"
              style={{ height: '4px', background: 'var(--prc-surface-2)' }}
            >
              <div
                className="h-full rounded-full"
                style={{ width: `${Math.round(value * 100)}%`, background: color, transition: 'width 0.3s' }}
              />
            </div>
            <span className="prc-text-dim flex-shrink-0" style={{ fontSize: '0.52rem', width: '28px', textAlign: 'right' }}>
              {(value * 100).toFixed(0)}%
            </span>
          </div>
        );
      })}
    </div>
  );
}

function RelabelButton({ currentLabel, onConfirm }: { currentLabel: string; onConfirm: (label: string) => void }) {
  const [editing, setEditing] = useState(false);
  const [value, setValue] = useState('');

  if (editing) {
    return (
      <div className="flex items-center gap-1">
        <input
          className="flex-1 bg-transparent prc-text text-xs outline-none rounded px-2 h-6"
          style={{ background: 'var(--prc-surface-2)', border: '1px solid var(--prc-border)', fontSize: '0.65rem' }}
          placeholder={currentLabel}
          value={value}
          onChange={(e) => setValue(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && value.trim()) {
              onConfirm(value.trim());
              setEditing(false);
              setValue('');
            }
            if (e.key === 'Escape') {
              setEditing(false);
              setValue('');
            }
          }}
          autoFocus
        />
        <button
          className="prc-btn prc-btn-ghost h-6 px-2"
          style={{ fontSize: '0.6rem', color: 'var(--prc-accent)' }}
          onClick={() => {
            if (value.trim()) {
              onConfirm(value.trim());
              setEditing(false);
              setValue('');
            }
          }}
        >
          确定
        </button>
      </div>
    );
  }

  return (
    <button
      className="prc-btn prc-btn-ghost h-6 w-full text-xs flex items-center justify-center gap-1"
      style={{ color: 'var(--prc-accent)', fontSize: '0.65rem' }}
      onClick={() => setEditing(true)}
    >
      <Edit3 size={10} />
      临时改标签
    </button>
  );
}
