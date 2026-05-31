import { useState } from 'react';
import { Search, Loader2, AlertCircle, ChevronDown, ChevronUp, Zap, Eye, Crosshair } from 'lucide-react';
import { useCanvasQuery } from '../hooks/useQuery';
import { getCandidateCropUrl } from '../api/client';
import type { CandidateResponse, QueryTargetModel, QueryResponse } from '../api/types';
import { providerLabels } from '../i18n/zh';

interface QueryPanelProps {
  canvasId: string;
  onSelectElement: (elementId: string) => void;
  selectedElementId: string | null;
}

type QueryType = 'text' | 'semantic_role' | 'region' | 'natural_language';

const queryTypeOptions: { value: QueryType; label: string; placeholder: string }[] = [
  { value: 'text', label: '文本', placeholder: '搜索元素文本...' },
  { value: 'semantic_role', label: '语义角色', placeholder: '如: button, textbox, link...' },
  { value: 'region', label: '区域', placeholder: '搜索区域 ID...' },
  { value: 'natural_language', label: '自然语言', placeholder: '如: 发送按钮在哪？' },
];

const providerColor: Record<string, string> = {
  uia: 'var(--prc-accent)',
  ocr: 'var(--prc-warning)',
  vision: '#a78bfa',
  vlm: '#22d3ee',
  dom: 'var(--prc-confirmed)',
  memory: 'var(--prc-muted)',
};

export default function QueryPanel({ canvasId, onSelectElement, selectedElementId }: QueryPanelProps) {
  const [queryType, setQueryType] = useState<QueryType>('text');
  const [queryText, setQueryText] = useState('');
  const [collapsed, setCollapsed] = useState(true);
  const query = useCanvasQuery();

  const handleQuery = () => {
    if (!queryText.trim() || !canvasId) return;
    const target: QueryTargetModel = { [queryType]: queryText.trim() };
    query.mutate({ canvasId, target });
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') handleQuery();
  };

  const currentOption = queryTypeOptions.find((o) => o.value === queryType)!;
  const results = query.data?.candidates ?? [];

  return (
    <div className="border-t" style={{ borderColor: 'var(--prc-border)' }}>
      {/* Header */}
      <button
        className="w-full flex items-center justify-between px-3 py-2 transition-colors"
        style={{ color: 'var(--prc-text-dim)' }}
        onClick={() => setCollapsed(!collapsed)}
      >
        <span className="flex items-center gap-1.5">
          <Search size={12} style={{ color: 'var(--prc-accent)' }} />
          <span className="prc-text text-xs font-semibold">查询面板</span>
          {query.data && (
            <span className="prc-badge mono" style={{ background: 'var(--prc-surface-2)', color: 'var(--prc-text-dim)' }}>
              {query.data.total_matched}
            </span>
          )}
        </span>
        {collapsed ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
      </button>

      {!collapsed && (
        <div className="px-3 pb-3">
          {/* Query type tabs */}
          <div className="flex items-center gap-1 mb-2">
            {queryTypeOptions.map((opt) => (
              <button
                key={opt.value}
                className="prc-badge cursor-pointer transition-all"
                style={{
                  background: queryType === opt.value ? 'var(--prc-accent)' : 'var(--prc-surface-2)',
                  color: queryType === opt.value ? 'white' : 'var(--prc-text-dim)',
                  border: `1px solid ${queryType === opt.value ? 'var(--prc-accent)' : 'var(--prc-border)'}`,
                  padding: '1px 6px',
                  fontSize: '0.6rem',
                }}
                onClick={() => setQueryType(opt.value)}
              >
                {opt.label}
              </button>
            ))}
          </div>

          {/* Input */}
          <div className="flex items-center gap-1.5">
            <div
              className="flex-1 flex items-center gap-2 rounded px-2.5 h-7"
              style={{ background: 'var(--prc-surface-2)', border: '1px solid var(--prc-border)' }}
            >
              <Search size={11} className="prc-text-dim flex-shrink-0" />
              <input
                className="flex-1 bg-transparent prc-text text-xs outline-none"
                style={{ color: 'var(--prc-text)', fontSize: '0.75rem' }}
                placeholder={currentOption.placeholder}
                value={queryText}
                onChange={(e) => setQueryText(e.target.value)}
                onKeyDown={handleKeyDown}
              />
            </div>
            <button
              className="prc-btn prc-btn-primary h-7 px-3"
              style={{ fontSize: '0.7rem' }}
              onClick={handleQuery}
              disabled={query.isPending || !queryText.trim()}
            >
              {query.isPending ? <Loader2 size={11} className="animate-spin" /> : '查询'}
            </button>
          </div>

          {/* Error */}
          {query.isError && (
            <div className="flex items-center gap-2 mt-2 px-2 py-1.5 rounded" style={{ background: 'rgba(239,68,68,0.1)', border: '1px solid rgba(239,68,68,0.2)' }}>
              <AlertCircle size={12} style={{ color: 'var(--prc-danger)' }} />
              <span className="prc-text-dim text-xs">{query.error?.message ?? '查询失败'}</span>
            </div>
          )}

          {/* Results */}
          {query.data && results.length > 0 && (
            <div className="mt-2 flex flex-col gap-1 max-h-48 overflow-y-auto">
              {results.map((c) => (
                <QueryResultCard
                  key={c.element_id}
                  candidate={c}
                  canvasId={canvasId}
                  selected={c.element_id === selectedElementId}
                  onClick={() => onSelectElement(c.element_id)}
                />
              ))}
            </div>
          )}

          {/* Empty result */}
          {query.data && results.length === 0 && (
            <div className="flex flex-col items-center py-4">
              <Search size={16} className="prc-text-dim mb-1" />
              <span className="prc-text-dim text-xs">无匹配结果</span>
            </div>
          )}

          {/* Suggestions */}
          {query.data && query.data.suggestions.length > 0 && (
            <div className="mt-2 flex items-center gap-1.5 flex-wrap">
              <Zap size={10} style={{ color: 'var(--prc-warning)' }} />
              <span className="prc-text-dim" style={{ fontSize: '0.6rem' }}>建议:</span>
              {query.data.suggestions.map((s, i) => (
                <button
                  key={i}
                  className="prc-badge cursor-pointer"
                  style={{ background: 'var(--prc-surface-2)', color: 'var(--prc-text-dim)', fontSize: '0.58rem', padding: '0 4px' }}
                  onClick={() => {
                    setQueryType('natural_language');
                    setQueryText(s);
                  }}
                >
                  {s}
                </button>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function QueryResultCard({ candidate: c, canvasId, selected, onClick }: { candidate: CandidateResponse; canvasId: string; selected: boolean; onClick: () => void }) {
  const confColor = c.confidence >= 0.85 ? 'var(--prc-confirmed)' : c.confidence >= 0.6 ? 'var(--prc-warning)' : 'var(--prc-danger)';
  const cropUrl = getCandidateCropUrl(canvasId, c.element_id);

  return (
    <div
      className="px-2 py-1.5 rounded transition-all group"
      style={{
        background: selected ? 'var(--prc-surface-2)' : 'transparent',
        border: selected ? '1px solid var(--prc-accent)' : '1px solid transparent',
      }}
    >
      <div className="flex items-center justify-between gap-2" onClick={onClick} style={{ cursor: 'pointer' }}>
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-1.5">
            <span className="prc-text text-xs font-medium truncate">{c.text || c.name || c.element_id}</span>
            <span className="prc-badge" style={{ background: 'var(--prc-surface-2)', color: 'var(--prc-text-dim)', fontSize: '0.55rem' }}>
              {c.semantic_role}
            </span>
          </div>
          <div className="flex items-center gap-1 mt-0.5">
            {c.provider_sources.map((p) => (
              <span
                key={p}
                className="prc-badge"
                style={{
                  background: `${providerColor[p] ?? 'var(--prc-muted)'}20`,
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
        <div className="flex items-center gap-1 flex-shrink-0">
          <div
            className="h-1 rounded-full"
            style={{ width: `${c.confidence * 30}px`, background: confColor }}
          />
          <span className="mono" style={{ fontSize: '0.6rem', color: confColor }}>
            {Math.round(c.confidence * 100)}%
          </span>
        </div>
      </div>
      {/* Action buttons (visible on hover) */}
      <div
        className="flex items-center gap-1 mt-1 opacity-0 group-hover:opacity-100 transition-opacity"
        style={{ paddingLeft: '1px' }}
      >
        <button
          className="prc-btn prc-btn-ghost h-5 px-1.5 flex items-center gap-0.5"
          style={{ fontSize: '0.55rem', color: 'var(--prc-accent)' }}
          onClick={(e) => { e.stopPropagation(); onClick(); }}
          title="选中此元素"
        >
          <Crosshair size={9} />
          <span>选中</span>
        </button>
        <button
          className="prc-btn prc-btn-ghost h-5 px-1.5 flex items-center gap-0.5"
          style={{ fontSize: '0.55rem', color: 'var(--prc-text-dim)' }}
          onClick={(e) => {
            e.stopPropagation();
            window.open(cropUrl, '_blank');
          }}
          title="查看裁剪图"
        >
          <Eye size={9} />
          <span>Crop</span>
        </button>
      </div>
    </div>
  );
}
