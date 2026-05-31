import { useState } from 'react';
import { Search, Monitor, RefreshCw, Loader2, AlertCircle, Radio, X } from 'lucide-react';
import { useWindows } from '../hooks/useWindows';
import { useObserve } from '../hooks/useCanvases';
import type { ObserveResponse, WindowListItem } from '../api/types';
import { t } from '../i18n/zh';

interface WindowBrowserProps {
  onObserved: (canvasId: string, response: ObserveResponse) => void;
}

function parseObserveError(err: unknown): string {
  if (!err) return '';
  const msg = err instanceof Error ? err.message : String(err);
  // Extract detail from "API 422: ..." or "API 500: ..." format
  const match = msg.match(/API \d+: (.+)/);
  const detail = match ? match[1] : msg;
  if (detail.includes('window_minimized')) return '窗口已最小化，请先恢复窗口';
  if (detail.includes('capture_invalid')) return '截图无效：' + detail;
  if (detail.includes('Screenshot')) return '截图失败：' + detail;
  return detail;
}

export default function WindowBrowser({ onObserved }: WindowBrowserProps) {
  const { data: windows, isLoading, error, refetch } = useWindows();
  const observe = useObserve();
  const [search, setSearch] = useState('');
  const [observeError, setObserveError] = useState<string | null>(null);
  const [observingHwnd, setObservingHwnd] = useState<number | null>(null);

  const filtered = (windows ?? []).filter((w) => {
    if (!search) return true;
    const q = search.toLowerCase();
    return (
      w.title.toLowerCase().includes(q) ||
      (w.process_name ?? '').toLowerCase().includes(q)
    );
  });

  const handleObserve = (w: WindowListItem) => {
    setObserveError(null);
    setObservingHwnd(w.hwnd);
    observe.mutate(
      { hwnd: w.hwnd },
      {
        onSuccess: (data) => {
          setObservingHwnd(null);
          onObserved(data.canvas_id, data);
        },
        onError: (err) => {
          setObservingHwnd(null);
          setObserveError(parseObserveError(err));
        },
      },
    );
  };

  return (
    <div className="flex flex-col h-full" style={{ borderRight: '1px solid var(--prc-border)' }}>
      {/* Header */}
      <div className="px-3 pt-3 pb-2 border-b" style={{ borderColor: 'var(--prc-border)' }}>
        <div className="flex items-center justify-between mb-2">
          <div className="flex items-center gap-1.5">
            <Monitor size={12} style={{ color: 'var(--prc-accent)' }} />
            <span className="prc-text text-xs font-semibold">窗口列表</span>
            <span className="prc-badge mono" style={{ background: 'var(--prc-surface-2)', color: 'var(--prc-text-dim)' }}>
              {filtered.length}
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
        <div className="flex items-center gap-2 rounded px-2.5 h-7" style={{ background: 'var(--prc-surface-2)', border: '1px solid var(--prc-border)' }}>
          <Search size={11} className="prc-text-dim flex-shrink-0" />
          <input
            className="flex-1 bg-transparent prc-text text-xs outline-none"
            style={{ color: 'var(--prc-text)', fontSize: '0.75rem' }}
            placeholder="搜索窗口标题或进程名..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
          />
        </div>
      </div>

      {/* Window List */}
      <div className="flex-1 overflow-y-auto">
        {isLoading && (
          <div className="flex flex-col items-center justify-center py-10">
            <Loader2 size={20} className="prc-text-dim mb-2 animate-spin" />
            <span className="prc-text-dim text-xs">正在枚举窗口...</span>
          </div>
        )}

        {error && (
          <div className="flex flex-col items-center justify-center py-10 px-4">
            <AlertCircle size={20} style={{ color: 'var(--prc-danger)' }} className="mb-2" />
            <span className="prc-text-dim text-xs text-center">枚举窗口失败: {error.message}</span>
          </div>
        )}

        {!isLoading && !error && filtered.length === 0 && (
          <div className="flex flex-col items-center justify-center py-10">
            <Monitor size={20} className="prc-text-dim mb-2" />
            <span className="prc-text-dim text-xs">未发现可见窗口</span>
          </div>
        )}

        {/* Observe error banner */}
        {observeError && (
          <div
            className="px-3 py-2 flex items-start gap-2"
            style={{ background: 'rgba(239,68,68,0.08)', borderBottom: '1px solid rgba(239,68,68,0.2)' }}
          >
            <AlertCircle size={14} style={{ color: 'var(--prc-danger)', flexShrink: 0, marginTop: '1px' }} />
            <span className="text-xs flex-1" style={{ color: 'var(--prc-danger)' }}>{observeError}</span>
            <button onClick={() => setObserveError(null)} className="flex-shrink-0" style={{ color: 'var(--prc-text-dim)' }}>
              <X size={12} />
            </button>
          </div>
        )}

        {filtered.map((w) => {
          const isThisObserving = observingHwnd === w.hwnd;
          return (
          <div
            key={w.hwnd}
            className="px-3 py-2 cursor-pointer transition-colors"
            style={{ borderBottom: '1px solid var(--prc-border)' }}
          >
            <div className="flex items-start justify-between gap-2">
              <div className="min-w-0 flex-1">
                <div className="flex items-center gap-1.5">
                  {w.is_foreground && (
                    <Radio size={10} style={{ color: 'var(--prc-confirmed)' }} />
                  )}
                  {w.is_minimized && (
                    <span className="prc-badge" style={{ background: 'rgba(245,158,11,0.15)', color: 'var(--prc-warning)', fontSize: '0.55rem' }}>最小化</span>
                  )}
                  <span className="prc-text text-xs font-medium truncate">{w.title || '(无标题)'}</span>
                </div>
                <div className="flex items-center gap-2 mt-0.5">
                  <span className="prc-text-dim mono" style={{ fontSize: '0.62rem' }}>{w.process_name ?? '—'}</span>
                  <span className="prc-text-dim mono" style={{ fontSize: '0.62rem' }}>hwnd:{w.hwnd}</span>
                </div>
              </div>
              <button
                className="prc-btn prc-btn-primary h-6 px-2 flex-shrink-0"
                style={{ fontSize: '0.68rem' }}
                onClick={() => handleObserve(w)}
                disabled={isThisObserving}
              >
                {isThisObserving ? <Loader2 size={10} className="animate-spin" /> : t('observe', '观察')}
              </button>
            </div>
          </div>
          );
        })}
      </div>
    </div>
  );
}
