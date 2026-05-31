import { useState, useMemo, useEffect, useRef } from 'react';
import { Search, Filter, ChevronDown, ChevronRight, AlertCircle, AlertTriangle, Layers, Fingerprint } from 'lucide-react';
import { getCandidateCropUrl } from '../api/client';
import type { CandidateResponse, RegionResponse } from '../api/types';
import { t, formatStableKeyId, providerLabels, uiLabels } from '../i18n/zh';
import { providerColor } from '../utils/providerColors';
import { DYNAMIC_CONTENT_ZONES, FIXED_CONTROL_ROLES } from '../constants/canvasZones';

interface ElementListProps {
  elements: CandidateResponse[];
  regions: RegionResponse[];
  selectedId: string | null;
  onSelect: (id: string) => void;
  canvasId?: string | null;
}

const filterTabs = [
  { id: 'all', labelKey: 'all' },
  { id: 'low-conf', labelKey: 'low_confidence' },
  { id: 'high-risk', labelKey: 'high_risk' },
  { id: 'interactable', labelKey: 'interactable' },
  { id: 'memory', labelKey: 'from_memory' },
  { id: 'refined', labelKey: 'refined' },
  { id: 'uncertain', labelKey: 'uncertain' },
];

export default function ElementList({ elements = [], regions = [], selectedId = null, onSelect = () => {}, canvasId = null }: ElementListProps) {
  const [activeFilter, setActiveFilter] = useState('all');
  const [searchText, setSearchText] = useState('');
  const [groupBy, setGroupBy] = useState<'region' | 'role' | 'provider'>('region');
  const [showGroupMenu, setShowGroupMenu] = useState(false);
  const groupMenuRef = useRef<HTMLDivElement>(null);

  // Click-outside dismiss for group menu
  useEffect(() => {
    if (!showGroupMenu) return;
    const handleClick = (e: MouseEvent) => {
      if (groupMenuRef.current && !groupMenuRef.current.contains(e.target as Node)) {
        setShowGroupMenu(false);
      }
    };
    document.addEventListener('mousedown', handleClick);
    return () => document.removeEventListener('mousedown', handleClick);
  }, [showGroupMenu]);

  const filteredElements = useMemo(() => elements.filter((el) => {
    const matchSearch = searchText === '' ||
      (el.text ?? '').toLowerCase().includes(searchText.toLowerCase()) ||
      el.element_id.toLowerCase().includes(searchText.toLowerCase()) ||
      el.semantic_role.toLowerCase().includes(searchText.toLowerCase()) ||
      (el.name ?? '').toLowerCase().includes(searchText.toLowerCase());
    if (!matchSearch) return false;
    if (activeFilter === 'all') return true;
    if (activeFilter === 'low-conf') return el.confidence < 0.7;
    if (activeFilter === 'high-risk') return el.risk_level === 'high' || el.risk_level === 'critical' || el.risk_tags.length > 0;
    if (activeFilter === 'interactable') return el.interactable;
    if (activeFilter === 'memory') return el.from_memory;
    if (activeFilter === 'refined') return el.refine_status === 'refined';
    if (activeFilter === 'uncertain') return el.refine_status === 'uncertain';
    return true;
  }), [elements, searchText, activeFilter]);

  const sorted = useMemo(() => [...filteredElements].sort((a, b) => b.confidence - a.confidence), [filteredElements]);

  const grouped = useMemo(() => {
    const map: Record<string, CandidateResponse[]> = {};
    sorted.forEach((el) => {
      let key: string;
      if (groupBy === 'region') {
        key = el.region_id || uiLabels['unassigned_region'];
      } else if (groupBy === 'role') {
        key = el.semantic_role || 'unknown';
      } else {
        key = el.provider_sources.join('+') || 'unknown';
      }
      if (!map[key]) map[key] = [];
      map[key].push(el);
    });
    return map;
  }, [sorted, groupBy]);

  // Resolve region names for group headers
  const regionNameMap = new Map(regions.map((r) => [r.region_id, r.role || r.region_id]));

  return (
    <div data-cmp="ElementList" className="flex flex-col h-full prc-panel" style={{ borderRight: '1px solid var(--prc-border)' }}>
      {/* Header */}
      <div className="px-3 pt-3 pb-2 border-b" style={{ borderColor: 'var(--prc-border)' }}>
        <div className="flex items-center justify-between mb-2">
          <div className="flex items-center gap-1.5">
            <Filter size={12} style={{ color: 'var(--prc-accent)' }} />
            <span className="prc-text text-xs font-semibold">元素列表</span>
            <span className="prc-badge mono" style={{ background: 'var(--prc-surface-2)', color: 'var(--prc-text-dim)' }}>{elements.length}</span>
          </div>
          <div className="relative" ref={groupMenuRef}>
            <button
              className="flex items-center gap-1 prc-text-dim text-xs px-2 py-0.5 rounded transition-colors"
              style={{ border: '1px solid var(--prc-border)' }}
              onClick={() => setShowGroupMenu(!showGroupMenu)}
            >
              {groupBy === 'region' ? uiLabels['group_by_region'] : groupBy === 'role' ? uiLabels['group_by_role'] : uiLabels['group_by_provider']}
              <ChevronDown size={10} />
            </button>
            {showGroupMenu && (
              <div className="absolute right-0 top-7 z-50 rounded shadow-custom py-1" style={{ background: 'var(--prc-surface-2)', border: '1px solid var(--prc-border)', width: '80px' }}>
                {([['region', 'group_by_region'], ['role', 'group_by_role'], ['provider', 'group_by_provider']] as const).map(([k, lk]) => (
                  <button key={k} onClick={() => { setGroupBy(k); setShowGroupMenu(false); }}
                    className="w-full text-left px-3 py-1.5 text-xs transition-colors"
                    style={{ color: groupBy === k ? 'var(--prc-accent)' : 'var(--prc-text-dim)' }}
                  >{uiLabels[lk] ?? lk}</button>
                ))}
              </div>
            )}
          </div>
        </div>

        {/* Search */}
        <div className="flex items-center gap-2 rounded px-2.5 h-7 mb-2" style={{ background: 'var(--prc-surface-2)', border: '1px solid var(--prc-border)' }}>
          <Search size={11} className="prc-text-dim flex-shrink-0" />
          <input
            className="flex-1 bg-transparent prc-text text-xs outline-none"
            style={{ color: 'var(--prc-text)', fontSize: '0.75rem' }}
            placeholder={uiLabels['search_hint']}
            value={searchText}
            onChange={(e) => setSearchText(e.target.value)}
          />
        </div>

        {/* Filter pills */}
        <div className="flex flex-wrap gap-1">
          {filterTabs.map((ft) => (
            <button
              key={ft.id}
              onClick={() => setActiveFilter(ft.id)}
              className="prc-badge transition-all cursor-pointer"
              style={{
                background: activeFilter === ft.id ? 'var(--prc-accent)' : 'var(--prc-surface-2)',
                color: activeFilter === ft.id ? 'white' : 'var(--prc-text-dim)',
                border: `1px solid ${activeFilter === ft.id ? 'var(--prc-accent)' : 'var(--prc-border)'}`,
                padding: '2px 8px',
                fontSize: '0.62rem',
              }}
            >
              {uiLabels[ft.labelKey] ?? ft.labelKey}
            </button>
          ))}
        </div>
      </div>

      {/* Element Cards */}
      <div className="flex-1 overflow-y-auto">
        {Object.entries(grouped).map(([groupName, groupEls]) => (
          <ElementGroup
            key={groupName}
            groupName={groupBy === 'region' ? (regionNameMap.get(groupName) ?? groupName) : groupName}
            elements={groupEls}
            selectedId={selectedId}
            onSelect={onSelect}
            canvasId={canvasId}
          />
        ))}
        {sorted.length === 0 && (
          <div className="flex flex-col items-center justify-center py-10">
            <AlertCircle size={20} className="prc-text-dim mb-2" />
            <span className="prc-text-dim text-xs">{uiLabels['no_elements']}</span>
          </div>
        )}
      </div>
    </div>
  );
}

function ElementGroup({ groupName, elements, selectedId, onSelect, canvasId }: {
  groupName: string;
  elements: CandidateResponse[];
  selectedId: string | null;
  onSelect: (id: string) => void;
  canvasId?: string | null;
}) {
  const [collapsed, setCollapsed] = useState(false);

  return (
    <div>
      <button
        className="w-full flex items-center gap-1.5 px-3 py-1.5 text-left"
        onClick={() => setCollapsed(!collapsed)}
        style={{ borderBottom: '1px solid var(--prc-border)' }}
      >
        {collapsed ? <ChevronRight size={10} className="prc-text-dim" /> : <ChevronDown size={10} className="prc-text-dim" />}
        <span className="prc-text-dim mono" style={{ fontSize: '0.65rem' }}>{groupName}</span>
        <span className="prc-badge mono" style={{ background: 'var(--prc-surface-2)', color: 'var(--prc-text-dim)', fontSize: '0.6rem' }}>{elements.length}</span>
      </button>
      {!collapsed && elements.map((el) => (
        <ElementCard key={el.element_id} element={el} selected={el.element_id === selectedId} onSelect={onSelect} canvasId={canvasId} />
      ))}
    </div>
  );
}

/** Check if an element is primarily from vision sources (no UIA/OCR strong evidence). */
function isVisionPrimary(el: CandidateResponse): boolean {
  const strongSources = ['uia', 'ocr', 'dom'];
  const hasStrong = el.provider_sources.some((s) => strongSources.includes(s));
  return !hasStrong && el.provider_sources.some((s) => ['vision', 'vlm', 'omniparser'].includes(s));
}

/** Check if an element is icon-like: no readable text, primarily visual. */
function isIconLike(el: CandidateResponse): boolean {
  return !el.text && !el.name && !!el.bounds && el.bounds.length >= 4;
}

/** Get a display label for an element, avoiding misleading text for icons. */
function getDisplayLabel(el: CandidateResponse): string {
  // If refined with a role_label, prefer it for display
  if (el.refine_status !== 'unreviewed' && el.role_label) {
    if (el.visual_type === 'icon' && el.refine_status === 'uncertain') {
      return '未知图标';
    }
    return el.role_label;
  }
  if (el.text) return el.text;
  if (el.name) return el.name;
  if (el.control_type) return el.control_type;
  if (isIconLike(el)) return '图标';
  return el.element_id;
}

/** Get semantic role display — mark as uncertain for vision-primary icons. */
function getSemanticDisplay(el: CandidateResponse): { label: string; uncertain: boolean } {
  // If refined, use refine data
  if (el.refine_status !== 'unreviewed' && el.role_label) {
    if (el.refine_status === 'uncertain') {
      return { label: el.role_label, uncertain: true };
    }
    return { label: el.role_label, uncertain: false };
  }
  if (isIconLike(el) && isVisionPrimary(el)) {
    return { label: el.semantic_role || '未知', uncertain: true };
  }
  if (isIconLike(el) && el.confidence < 0.7) {
    return { label: el.semantic_role || '未知', uncertain: true };
  }
  return { label: el.semantic_role, uncertain: false };
}

function ElementCard({ element: el, selected, onSelect, canvasId }: { element: CandidateResponse; selected: boolean; onSelect: (id: string) => void; canvasId?: string | null }) {
  const confColor = el.confidence >= 0.85 ? 'var(--prc-confirmed)' : el.confidence >= 0.6 ? 'var(--prc-warning)' : 'var(--prc-danger)';
  const confPct = Math.round(el.confidence * 100);
  const cropUrl = canvasId ? getCandidateCropUrl(canvasId, el.element_id, 4) : null;
  const hasBounds = el.bounds && el.bounds.length >= 4;
  const iconic = isIconLike(el);
  const displayLabel = getDisplayLabel(el);
  const semantic = getSemanticDisplay(el);

  return (
    <div
      className="px-3 py-2 cursor-pointer element-card-hover transition-all"
      style={{
        borderBottom: '1px solid var(--prc-border)',
        borderLeft: selected ? `2px solid var(--prc-accent)` : '2px solid transparent',
        background: selected ? 'var(--prc-surface-2)' : 'transparent',
      }}
      onClick={() => onSelect(el.element_id)}
    >
      <div className="flex items-start gap-2">
        {/* Crop thumbnail — primary for icons */}
        {cropUrl && hasBounds && (
          <div
            className="flex-shrink-0 rounded overflow-hidden"
            style={{
              width: iconic ? '36px' : '32px',
              height: iconic ? '36px' : '24px',
              background: 'var(--prc-surface-2)',
              border: '1px solid var(--prc-border)',
            }}
          >
            <img
              src={cropUrl}
              alt=""
              className="w-full h-full object-contain"
              loading="lazy"
              draggable={false}
              onError={(e) => { (e.target as HTMLImageElement).style.display = 'none'; }}
            />
          </div>
        )}

        <div className="flex-1 min-w-0">
          <div className="flex items-start justify-between gap-2">
            <div className="flex items-center gap-1.5 min-w-0">
              <span className="prc-text text-xs font-medium truncate">{displayLabel}</span>
              {el.suggest_confirm && (
                <AlertTriangle size={10} style={{ color: 'var(--prc-warning)', flexShrink: 0 }} />
              )}
              {el.from_memory && (
                <span className="prc-badge flex-shrink-0" style={{ background: 'rgba(75,85,99,0.15)', color: 'var(--prc-muted)', fontSize: '0.55rem' }}>记忆</span>
              )}
              {el.refine_status === 'refined' && (el.role_source === 'agent' || el.role_source === 'memory') && (
                <span className="prc-badge flex-shrink-0" style={{ background: 'rgba(34,197,94,0.12)', color: 'var(--prc-confirmed)', fontSize: '0.55rem' }}>已复核</span>
              )}
              {el.refine_status === 'refined' && el.role_source === 'heuristic' && (
                <span className="prc-badge flex-shrink-0" style={{ background: 'rgba(245,158,11,0.1)', color: 'var(--prc-warning)', fontSize: '0.55rem' }}>规则推断</span>
              )}
              {el.refine_status === 'uncertain' && (
                <span className="prc-badge flex-shrink-0" style={{ background: 'rgba(245,158,11,0.1)', color: 'var(--prc-warning)', fontSize: '0.55rem' }}>待确认</span>
              )}
              {el.region_id && DYNAMIC_CONTENT_ZONES.has(el.region_id) && (
                <span className="prc-badge flex-shrink-0" style={{ background: 'rgba(107,114,128,0.12)', color: 'var(--prc-muted)', fontSize: '0.55rem' }}>动态文本</span>
              )}
            </div>
            <span
              className="prc-badge flex-shrink-0"
              style={{
                background: semantic.uncertain ? 'rgba(245,158,11,0.1)' : 'var(--prc-surface-2)',
                color: semantic.uncertain ? 'var(--prc-warning)' : 'var(--prc-text-dim)',
                fontSize: '0.58rem',
              }}
            >
              {semantic.uncertain ? `${semantic.label}?` : semantic.label}
            </span>
          </div>

          {/* Control type + region + stable_key_id */}
          <div className="flex items-center gap-2 mt-0.5 flex-wrap">
            {el.control_type && (
              <span className="prc-text-dim mono" style={{ fontSize: '0.6rem' }}>{el.control_type}</span>
            )}
            {iconic && isVisionPrimary(el) && (
              <span className="prc-badge" style={{ background: 'rgba(167,139,250,0.1)', color: '#a78bfa', fontSize: '0.55rem' }}>视觉检测</span>
            )}
            {el.risk_tags.length > 0 && (
              <span className="prc-badge" style={{ background: 'rgba(239,68,68,0.1)', color: 'var(--prc-danger)', fontSize: '0.55rem' }}>
                {el.risk_tags[0]}
              </span>
            )}
            {/* E Phase 1: stable_key_id display */}
            <span
              className="prc-badge"
              style={{
                background: el.stable_key_id ? 'rgba(59,130,246,0.1)' : 'rgba(107,114,128,0.1)',
                color: el.stable_key_id ? 'var(--prc-accent)' : 'var(--prc-muted)',
                fontSize: '0.55rem',
              }}
            >
              <Fingerprint size={8} className="inline mr-0.5" />
              {formatStableKeyId(el.stable_key_id)}
            </span>
          </div>

          {/* Element nature tags */}
          <div className="flex items-center gap-1.5 mt-0.5 flex-wrap">
            {FIXED_CONTROL_ROLES.has(el.semantic_role) && (
              <span className="prc-badge" style={{ background: 'rgba(59,130,246,0.08)', color: 'var(--prc-accent)', fontSize: '0.52rem' }}>固定控件</span>
            )}
            {el.region_id && DYNAMIC_CONTENT_ZONES.has(el.region_id) && (
              <span className="prc-badge" style={{ background: 'rgba(107,114,128,0.1)', color: 'var(--prc-muted)', fontSize: '0.52rem' }}>动态内容</span>
            )}
            {el.from_memory && (
              <span className="prc-badge" style={{ background: 'rgba(167,139,250,0.1)', color: '#a78bfa', fontSize: '0.52rem' }}>可入记忆</span>
            )}
            {el.role_source === 'agent' && (
              <span className="prc-badge" style={{ background: 'rgba(34,197,94,0.1)', color: 'var(--prc-confirmed)', fontSize: '0.52rem' }}>Agent 复核</span>
            )}
            {el.role_source === 'heuristic' && (
              <span className="prc-badge" style={{ background: 'rgba(245,158,11,0.08)', color: 'var(--prc-warning)', fontSize: '0.52rem' }}>规则推断</span>
            )}
          </div>

          {/* Confidence + providers */}
          <div className="flex items-center gap-2 mt-1">
            <div className="flex items-center gap-1">
              <div
                className="h-1 rounded-full"
                style={{ width: `${confPct * 0.4}px`, background: confColor }}
              />
              <span className="mono" style={{ fontSize: '0.6rem', color: confColor }}>
                {confPct}%
              </span>
            </div>
            <div className="flex items-center gap-1">
              {el.provider_sources.map((p) => (
                <span
                  key={p}
                  className="prc-badge"
                  style={{
                    background: `${providerColor[p] ?? 'var(--prc-muted)'}15`,
                    color: providerColor[p] ?? 'var(--prc-muted)',
                    fontSize: '0.52rem',
                    padding: '0 3px',
                  }}
                >
                  {providerLabels[p] ?? p}
                </span>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
