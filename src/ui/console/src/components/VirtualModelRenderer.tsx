/**
 * VirtualModelRenderer — desensitized structural rendering from persistent virtual model.
 *
 * Reads from SQLite-backed VirtualModelDetail (not CanvasCache).
 * Renders fixed candidates as labeled rectangles. Dynamic content areas
 * are replaced with opaque placeholder labels so original text is never exposed.
 *
 * relative_bounds are 0-1 floats [x1/W, y1/H, x2/W, y2/H] — direct percentage mapping.
 */

import { useState, useEffect, useRef } from 'react';
import type { CSSProperties, MouseEvent as ReactMouseEvent } from 'react';
import type { VirtualModelCandidate, VirtualModelDetail, CanvasSnapshotResponse, CandidateResponse } from '../api/types';
import { fetchPageModelTree, fetchCanvasDetail, getCanvasScreenshotUrl, getCandidateCropUrl } from '../api/client';
import { t, providerLabels } from '../i18n/zh';
import {
  type CandidateEdit,
  getCandidateKind,
  getModelCandidateLabel,
  normalizeModelCandidate,
} from '../utils/modelCandidates';

// ── Permanence state short labels ──
const PERMANENCE_LABELS: Record<string, string> = {
  new: '新',
  provisional: '暂',
  stable: '稳',
  fixed_anchor: '锚',
  stale: '旧',
  degraded: '退',
  retired: '废',
};

// ── Color maps ──

/** Fixed-control border colours keyed by semantic role category. */
function getControlColor(role: string): string {
  if (role.includes('button') || role === 'send_button' || role === 'submit_button' || role === 'cancel_button' || role === 'icon_button' || role === 'toggle_button') {
    return '#22c55e';
  }
  if (role.includes('input') || role.includes('Input')) {
    return '#f97316';
  }
  if (role === 'title_bar' || role === 'status_bar' || role === 'toolbar' || role === 'menu_bar') {
    return '#6b7280';
  }
  if (role.includes('icon') || role === 'tab' || role === 'nav_item' || role === 'menu_item') {
    return '#a855f7';
  }
  return '#9ca3af';
}

// ── Props ──

interface VirtualModelRendererProps {
  model: VirtualModelDetail;
  candidates?: VirtualModelCandidate[];
  className?: string;
  selectedKeyId?: string | null;
  hiddenKeyIds?: Set<string>;
  candidateEdits?: Map<string, CandidateEdit>;
  sourceCanvasId?: string | null;
  aspectRatio?: number;
  onSelectCandidate?: (keyId: string) => void;
  onPreviewBounds?: (keyId: string, bounds: number[]) => void;
  onCommitBounds?: (keyId: string, bounds: number[]) => void;
  showMaterialArea?: boolean;
}

// ── Helpers ──

/** Resolve a human-readable label for a semantic role. */
function roleLabel(role: string): string {
  return t(role, role);
}

/** Convert relative_bounds [x1, y1, x2, y2] (0-1 floats) to percentage style. */
function relBoundsToPercent(bounds: number[]): { left: string; top: string; width: string; height: string } {
  const [x1, y1, x2, y2] = bounds;
  return {
    left: `${x1 * 100}%`,
    top: `${y1 * 100}%`,
    width: `${(x2 - x1) * 100}%`,
    height: `${(y2 - y1) * 100}%`,
  };
}

function ScreenshotBackedRegion({
  screenshotUrl,
  bounds,
}: {
  screenshotUrl: string;
  bounds: number[];
}) {
  const [x1, y1, x2, y2] = bounds;
  const width = Math.max(0.001, x2 - x1);
  const height = Math.max(0.001, y2 - y1);
  return (
    <div
      aria-hidden="true"
      style={{
        position: 'absolute',
        inset: 0,
        overflow: 'hidden',
        borderRadius: '1px',
        opacity: 0.92,
        pointerEvents: 'none',
      }}
    >
      <img
        src={screenshotUrl}
        alt=""
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

// ── Component ──

export default function VirtualModelRenderer({
  model,
  candidates,
  className,
  selectedKeyId,
  hiddenKeyIds,
  candidateEdits,
  sourceCanvasId,
  aspectRatio = 0.75,
  onSelectCandidate,
  onPreviewBounds,
  onCommitBounds,
  showMaterialArea = true,
}: VirtualModelRendererProps) {
  const surfaceRef = useRef<HTMLDivElement | null>(null);
  const allCandidates = (candidates ?? model.candidates ?? []).map(normalizeModelCandidate);
  const visibleCandidates = allCandidates.filter((c) => !hiddenKeyIds?.has(c.key_id));
  const screenshotUrl = sourceCanvasId ? getCanvasScreenshotUrl(sourceCanvasId, false) : null;

  // Material area: fetch associated canvas snapshots
  const [snapshots, setSnapshots] = useState<CanvasSnapshotResponse[]>([]);
  const [loadingSnapshots, setLoadingSnapshots] = useState(false);

  useEffect(() => {
    if (!showMaterialArea || !model.page_model_id) return;
    let cancelled = false;
    setLoadingSnapshots(true);
    fetchPageModelTree()
      .then((tree) => {
        if (cancelled) return;
        const related = tree.canvas_snapshots.filter(
          (s) => s.state_template_id === model.state_template_id && s.available,
        );
        setSnapshots(related);
      })
      .catch(() => {})
      .finally(() => {
        if (!cancelled) setLoadingSnapshots(false);
      });
    return () => { cancelled = true; };
  }, [model.state_template_id, model.page_model_id, showMaterialArea]);
  // Separate candidates: fixed controls vs. dynamic content
  // A candidate goes to fixed ONLY if it's truly fixed AND not in a dynamic zone
  const fixedCandidates = visibleCandidates.filter(
    (c) => getCandidateKind(c, candidateEdits?.get(c.key_id)) === 'fixed',
  );

  // Dynamic/temporary candidates are still rendered and selectable in the workbench.
  const dynamicCandidates = visibleCandidates.filter(
    (c) => getCandidateKind(c, candidateEdits?.get(c.key_id)) !== 'fixed',
  );

  const beginResize = (
    event: ReactMouseEvent,
    candidate: VirtualModelCandidate,
    corner: 'nw' | 'ne' | 'sw' | 'se',
  ) => {
    if (!onPreviewBounds && !onCommitBounds) return;
    if (candidate.relative_bounds.length < 4) return;

    event.preventDefault();
    event.stopPropagation();

    const start = [...candidate.relative_bounds];
    let latest = start;

    const clamp = (value: number) => Math.max(0, Math.min(1, value));
    const normalize = (bounds: number[]) => {
      let [x1, y1, x2, y2] = bounds.map(clamp);
      const minSize = 0.008;
      if (x2 - x1 < minSize) {
        if (corner.includes('w')) x1 = Math.max(0, x2 - minSize);
        else x2 = Math.min(1, x1 + minSize);
      }
      if (y2 - y1 < minSize) {
        if (corner.includes('n')) y1 = Math.max(0, y2 - minSize);
        else y2 = Math.min(1, y1 + minSize);
      }
      return [x1, y1, x2, y2];
    };

    const updateFromPointer = (clientX: number, clientY: number) => {
      const rect = surfaceRef.current?.getBoundingClientRect();
      if (!rect || rect.width <= 0 || rect.height <= 0) return;
      const x = clamp((clientX - rect.left) / rect.width);
      const y = clamp((clientY - rect.top) / rect.height);
      const next = [...latest];
      if (corner.includes('w')) next[0] = x;
      if (corner.includes('e')) next[2] = x;
      if (corner.includes('n')) next[1] = y;
      if (corner.includes('s')) next[3] = y;
      latest = normalize(next);
      onPreviewBounds?.(candidate.key_id, latest);
    };

    const onMove = (moveEvent: MouseEvent) => {
      updateFromPointer(moveEvent.clientX, moveEvent.clientY);
    };
    const onUp = (upEvent: MouseEvent) => {
      updateFromPointer(upEvent.clientX, upEvent.clientY);
      onCommitBounds?.(candidate.key_id, latest);
      window.removeEventListener('mousemove', onMove);
      window.removeEventListener('mouseup', onUp);
    };

    window.addEventListener('mousemove', onMove);
    window.addEventListener('mouseup', onUp);
  };

  const renderResizeHandles = (candidate: VirtualModelCandidate, disabled: boolean) => {
    if (disabled || selectedKeyId !== candidate.key_id || !onPreviewBounds) return null;
    const handles: Array<['nw' | 'ne' | 'sw' | 'se', CSSProperties]> = [
      ['nw', { left: -4, top: -4, cursor: 'nwse-resize' }],
      ['ne', { right: -4, top: -4, cursor: 'nesw-resize' }],
      ['sw', { left: -4, bottom: -4, cursor: 'nesw-resize' }],
      ['se', { right: -4, bottom: -4, cursor: 'nwse-resize' }],
    ];
    return (
      <>
        {handles.map(([corner, style]) => (
          <span
            key={corner}
            style={{
              position: 'absolute',
              width: 8,
              height: 8,
              borderRadius: 2,
              background: '#38bdf8',
              border: '1px solid var(--prc-bg, #0e1117)',
              boxSizing: 'border-box',
              zIndex: 10,
              ...style,
            }}
            onMouseDown={(event) => beginResize(event, candidate, corner)}
          />
        ))}
      </>
    );
  };

  return (
    <div
      data-cmp="VirtualModelRenderer"
      className={className}
      style={{
        width: '100%',
        maxWidth: '900px',
        margin: '0 auto',
      }}
    >
      {/* Aspect-ratio container */}
      <div
        style={{
          position: 'relative',
          width: '100%',
          paddingBottom: `${aspectRatio * 100}%`,
          background: 'var(--prc-bg, #0e1117)',
          borderRadius: '6px',
          border: '1px solid var(--prc-border, #2d3748)',
          overflow: 'hidden',
        }}
      >
        {/* Absolute-positioned inner surface */}
        <div
          ref={surfaceRef}
          style={{
            position: 'absolute',
            inset: 0,
          }}
        >
          {/* ── Dynamic content zone placeholders ── */}
          {dynamicCandidates.map((c) => {
            if (c.relative_bounds.length < 4) return null;
            const pos = relBoundsToPercent(c.relative_bounds);
            const label = getModelCandidateLabel(c, candidateEdits);
            const isSelected = selectedKeyId === c.key_id;
            const isLargeContainer = (
              c.relative_bounds.length >= 4 &&
              (c.relative_bounds[2] - c.relative_bounds[0]) * (c.relative_bounds[3] - c.relative_bounds[1]) > 0.7
            );

            return (
              <div
                key={`dyn-${c.key_id}`}
                style={{
                  position: 'absolute',
                  ...pos,
                  border: `${isSelected ? 2 : 1.5}px dashed ${isSelected ? '#38bdf8' : '#6b7280'}`,
                  borderRadius: '2px',
                  background: isSelected ? 'rgba(56,189,248,0.10)' : 'rgba(148,163,184,0.06)',
                  boxSizing: 'border-box',
                  cursor: onSelectCandidate && !isLargeContainer ? 'pointer' : 'default',
                  zIndex: isLargeContainer ? 0 : 1,
                  pointerEvents: isLargeContainer ? 'none' : 'auto',
                }}
                onClick={(e) => {
                  e.stopPropagation();
                  onSelectCandidate?.(c.key_id);
                }}
              >
                <span
                  style={{
                    position: 'absolute',
                    top: '2px',
                    left: '4px',
                    fontSize: '0.58rem',
                    fontFamily: 'monospace',
                    color: isSelected ? '#38bdf8' : '#6b7280',
                    opacity: 0.85,
                    whiteSpace: 'nowrap',
                    lineHeight: 1.2,
                    userSelect: 'none',
                  }}
                >
                  {label}
                </span>
                <div
                  style={{
                    position: 'absolute',
                    inset: '18px 4px 4px 4px',
                    display: isLargeContainer ? 'flex' : 'none',
                    alignItems: 'center',
                    justifyContent: 'center',
                    background: 'rgba(148,163,184,0.06)',
                    borderRadius: '2px',
                    border: '1px dashed rgba(148,163,184,0.2)',
                  }}
                >
                  <span
                    style={{
                      fontSize: '0.72rem',
                      color: 'var(--prc-text-dim, #718096)',
                      opacity: 0.6,
                      userSelect: 'none',
                      letterSpacing: '0.05em',
                    }}
                  >
                    {t(c.canonical_region, c.canonical_region)}
                  </span>
                </div>
                {renderResizeHandles(c, isLargeContainer)}
              </div>
            );
          })}

          {/* ── Fixed-control candidates ── */}
          {fixedCandidates.map((c) => {
            if (c.relative_bounds.length < 4) return null;

            const color = getControlColor(c.canonical_role);
            const pos = relBoundsToPercent(c.relative_bounds);
            const label = getModelCandidateLabel(c, candidateEdits) || roleLabel(c.canonical_role);
            const isSelected = selectedKeyId === c.key_id;
            const isInput = c.canonical_role.includes('input') || c.canonical_role.includes('Input');
            const isLowConfidence = c.confidence < 0.5;
            const isLargeContainer = (
              c.relative_bounds.length >= 4 &&
              (c.relative_bounds[2] - c.relative_bounds[0]) * (c.relative_bounds[3] - c.relative_bounds[1]) > 0.7
            );

            // Border style: selected > fixed_anchor > input > low confidence > normal
            const isFixedAnchor = c.permanence_state === 'fixed_anchor';
            let borderStyle = `1.5px solid ${color}`;
            let bgStyle = 'transparent';
            let boxShadow = 'none';
            if (isSelected) {
              borderStyle = `2px solid ${color}`;
              bgStyle = `${color}15`;
            } else if (isFixedAnchor) {
              borderStyle = '2px solid #fbbf24';
              bgStyle = 'rgba(251,191,36,0.06)';
              boxShadow = '0 0 6px rgba(251,191,36,0.25)';
            } else if (isLowConfidence) {
              borderStyle = `1.5px dashed ${color}`;
              bgStyle = 'rgba(245,158,11,0.06)';
            }

            return (
              <div
                key={c.key_id}
                style={{
                  position: 'absolute',
                  ...pos,
                  border: borderStyle,
                  borderRadius: '2px',
                  boxSizing: 'border-box',
                  background: bgStyle,
                  boxShadow,
                  cursor: onSelectCandidate && !isLargeContainer ? 'pointer' : 'default',
                  transition: 'border-width 0.1s, background 0.1s',
                  zIndex: isLargeContainer ? 0 : 2,
                  pointerEvents: isLargeContainer ? 'none' : 'auto',
                }}
                onClick={(e) => {
                  e.stopPropagation();
                  onSelectCandidate?.(c.key_id);
                }}
              >
                {/* Control label (top-left, on the border) */}
                {screenshotUrl && !isLargeContainer && (
                  <ScreenshotBackedRegion screenshotUrl={screenshotUrl} bounds={c.relative_bounds} />
                )}
                <span
                  style={{
                    position: 'absolute',
                    top: '-1px',
                    left: '3px',
                    fontSize: '0.55rem',
                    fontFamily: 'monospace',
                    fontWeight: 600,
                    color: color,
                    background: 'var(--prc-bg, #0e1117)',
                    padding: '0 2px',
                    lineHeight: 1.3,
                    whiteSpace: 'nowrap',
                    userSelect: 'none',
                  }}
                >
                  {label}
                  {c.provider_sources.includes('vlm') && (
                    <span
                      style={{
                        marginLeft: '2px',
                        fontSize: '0.45rem',
                        color: '#38bdf8',
                        opacity: 0.8,
                      }}
                    >
                      [VLM]
                    </span>
                  )}
                  {c.permanence_state && c.permanence_state !== 'new' && (
                    <span
                      style={{
                        marginLeft: '2px',
                        fontSize: '0.42rem',
                        color: isFixedAnchor ? '#fbbf24' : '#94a3b8',
                        opacity: 0.9,
                      }}
                    >
                      [{PERMANENCE_LABELS[c.permanence_state] ?? c.permanence_state}]
                    </span>
                  )}
                </span>
                {/* Low confidence warning indicator */}
                {isLowConfidence && (
                  <span
                    style={{
                      position: 'absolute',
                      top: '-1px',
                      right: '3px',
                      fontSize: '0.5rem',
                      color: '#f59e0b',
                      background: 'var(--prc-bg, #0e1117)',
                      padding: '0 2px',
                      lineHeight: 1.3,
                      userSelect: 'none',
                    }}
                  >
                    ?
                  </span>
                )}
                {renderResizeHandles(c, isLargeContainer)}
              </div>
            );
          })}
        </div>
      </div>

      {/* Summary footer */}
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          padding: '6px 0',
          fontSize: '0.6rem',
          color: 'var(--prc-text-dim, #718096)',
          fontFamily: 'monospace',
        }}
      >
        <span>{fixedCandidates.length} 固定</span>
        <span>{dynamicCandidates.length} 临时</span>
        <span>{visibleCandidates.length} 元素</span>
        <span>验证 {model.verify_count} 次</span>
      </div>

      {/* Candidate Material Area */}
      {showMaterialArea && (
        <CandidateMaterialArea
          model={model}
          snapshots={snapshots}
          loading={loadingSnapshots}
          candidates={visibleCandidates}
          sourceCanvasId={sourceCanvasId}
          selectedKeyId={selectedKeyId}
          candidateEdits={candidateEdits}
          onSelectCandidate={onSelectCandidate}
        />
      )}
    </div>
  );
}

// ── Candidate Material Area ──

interface CandidateMaterialAreaProps {
  model: VirtualModelDetail;
  snapshots: CanvasSnapshotResponse[];
  loading: boolean;
  candidates: VirtualModelCandidate[];
  sourceCanvasId?: string | null;
  selectedKeyId?: string | null;
  candidateEdits?: Map<string, CandidateEdit>;
  onSelectCandidate?: (keyId: string) => void;
}

function CandidateMaterialArea({
  model,
  snapshots,
  loading,
  candidates,
  sourceCanvasId,
  selectedKeyId,
  candidateEdits,
  onSelectCandidate,
}: CandidateMaterialAreaProps) {
  const displayCandidates = candidates;

  // Fetch CanvasDetail for available snapshots to build stable_key_id → element_id mapping
  const [elementMap, setElementMap] = useState<Map<string, string>>(new Map());
  const [activeCanvasId, setActiveCanvasId] = useState<string | null>(null);
  const [loadingDetails, setLoadingDetails] = useState(false);

  useEffect(() => {
    if (sourceCanvasId) {
      setActiveCanvasId(sourceCanvasId);
    }
    if (snapshots.length === 0) return;
    let cancelled = false;
    setLoadingDetails(true);

    // Try snapshots in order, use the first one that returns valid detail
    const loadDetails = async () => {
      for (const snap of snapshots) {
        if (cancelled) break;
        try {
          const detail = await fetchCanvasDetail(snap.canvas_id);
          if (cancelled) break;
          // Build stable_key_id → element_id map
          const map = new Map<string, string>();
          for (const el of detail.elements) {
            if (el.stable_key_id) {
              map.set(el.stable_key_id, el.element_id);
            }
          }
          setElementMap(map);
          setActiveCanvasId(snap.canvas_id);
          break; // Use first available
        } catch {
          // Canvas expired, try next
          continue;
        }
      }
      if (!cancelled) setLoadingDetails(false);
    };
    loadDetails();
    return () => { cancelled = true; };
  }, [snapshots, sourceCanvasId]);

  if (loading || loadingDetails) {
    return (
      <div style={{ padding: '12px 0', textAlign: 'center' }}>
        <span style={{ fontSize: '0.65rem', color: 'var(--prc-text-dim, #718096)' }}>
          加载候选素材...
        </span>
      </div>
    );
  }

  if (!activeCanvasId) {
    return (
      <div style={{ padding: '12px 0', textAlign: 'center' }}>
        <span style={{ fontSize: '0.65rem', color: 'var(--prc-text-dim, #718096)' }}>
          需要重新观察以查看候选素材
        </span>
      </div>
    );
  }

  return (
    <div style={{ marginTop: '8px' }}>
      <div
        style={{
          fontSize: '0.62rem',
          fontWeight: 600,
          color: 'var(--prc-text-dim, #718096)',
          marginBottom: '6px',
          display: 'flex',
          alignItems: 'center',
          gap: '4px',
        }}
      >
        候选素材
        <span
          style={{
            fontSize: '0.55rem',
            fontWeight: 400,
            color: 'var(--prc-text-dim, #718096)',
            opacity: 0.7,
          }}
        >
          ({displayCandidates.length} 项)
        </span>
      </div>
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fill, minmax(80px, 1fr))',
          gap: '6px',
        }}
      >
        {displayCandidates.map((c) => {
          const isSelected = selectedKeyId === c.key_id;
          const label = getModelCandidateLabel(c, candidateEdits);
          // Map stable_key_id → element_id for crop URL
          const elementId = c.element_id || elementMap.get(c.key_id);
          const cropUrl = elementId ? getCandidateCropUrl(activeCanvasId, elementId) : null;

          return (
            <div
              key={c.key_id}
              style={{
                border: `1px solid ${isSelected ? 'var(--prc-accent, #3b82f6)' : 'var(--prc-border, #2d3748)'}`,
                borderRadius: '4px',
                overflow: 'hidden',
                cursor: onSelectCandidate ? 'pointer' : 'default',
                background: isSelected ? 'rgba(59,130,246,0.08)' : 'var(--prc-surface-1, #1a1f2e)',
                transition: 'border-color 0.15s',
              }}
              onClick={() => onSelectCandidate?.(c.key_id)}
            >
              {/* Crop image */}
              <div
                style={{
                  width: '100%',
                  aspectRatio: '1',
                  background: 'var(--prc-surface-2, #252b3b)',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  overflow: 'hidden',
                }}
              >
                {cropUrl ? (
                  <img
                    src={cropUrl}
                    alt={label}
                    style={{
                      maxWidth: '100%',
                      maxHeight: '100%',
                      objectFit: 'contain',
                    }}
                    onError={(e) => {
                      (e.target as HTMLImageElement).style.display = 'none';
                      const parent = (e.target as HTMLImageElement).parentElement;
                      if (parent && !parent.querySelector('.crop-fallback')) {
                        const fb = document.createElement('span');
                        fb.className = 'crop-fallback';
                        fb.textContent = label.charAt(0).toUpperCase();
                        fb.style.cssText = 'font-size:1.2rem;font-weight:700;color:var(--prc-text-dim,#718096);';
                        parent.appendChild(fb);
                      }
                    }}
                  />
                ) : (
                  <span style={{ fontSize: '1rem', color: 'var(--prc-text-dim, #718096)', opacity: 0.5 }}>
                    {label.charAt(0).toUpperCase()}
                  </span>
                )}
              </div>
              {/* Label */}
              <div style={{ padding: '3px 4px' }}>
                <div
                  style={{
                    fontSize: '0.55rem',
                    fontWeight: 500,
                    color: 'var(--prc-text, #e2e8f0)',
                    whiteSpace: 'nowrap',
                    overflow: 'hidden',
                    textOverflow: 'ellipsis',
                  }}
                >
                  {label}
                </div>
                {c.canonical_text && (
                  <div
                    style={{
                      fontSize: '0.5rem',
                      color: 'var(--prc-text-dim, #718096)',
                      whiteSpace: 'nowrap',
                      overflow: 'hidden',
                      textOverflow: 'ellipsis',
                    }}
                  >
                    {c.canonical_text}
                  </div>
                )}
                <div
                  style={{
                    fontSize: '0.5rem',
                    color: 'var(--prc-text-dim, #718096)',
                    display: 'flex',
                    justifyContent: 'space-between',
                  }}
                >
                  <span>{(c.confidence * 100).toFixed(0)}%</span>
                  <span>{providerLabels[c.provider_sources[0]] ?? c.provider_sources[0] ?? ''}</span>
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
