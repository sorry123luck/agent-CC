import { useState, useMemo, useEffect, useRef } from 'react';
import { ZoomIn, ZoomOut, Maximize2, Eye, EyeOff, Layers, Monitor, ToggleLeft, ToggleRight, Image as ImageIcon } from 'lucide-react';
import { getCanvasScreenshotUrl } from '../api/client';
import type { CandidateResponse, CanvasDetail } from '../api/types';
import { t, providerLabels } from '../i18n/zh';
import { providerColor as baseProviderColor } from '../utils/providerColors';
import { DYNAMIC_CONTENT_ZONES } from '../constants/canvasZones';

interface ReviewCanvasProps {
  canvasDetail: CanvasDetail | null;
  selectedId: string | null;
  onSelect: (id: string) => void;
  overlayEnabled: boolean;
  onToggleOverlay: () => void;
  activeLayers: Record<string, boolean>;
  onToggleLayer: (layer: string) => void;
}

const layerDefs = [
  { key: 'regions', label: '区域', desc: '功能区域划分（顶部栏、输入区等）' },
  { key: 'geometric_regions', label: '几何分区', desc: '纯视觉边界检测的中性分区（灰色虚线），无语义标签' },
  { key: 'roi', label: 'ROI裁剪', desc: '发送给 ROI VLM 的局部截图范围，来自 roi_selection_plan' },
  { key: 'elements', label: '最终候选', desc: '融合后的可操作候选元素' },
  { key: 'anchors', label: '锚点', desc: '用于相对定位的参考点' },
  { key: 'locators', label: '定位器', desc: '各候选的多种定位方式' },
  { key: 'ocr', label: 'OCR', desc: 'PaddleOCR 原始文本检测结果' },
  { key: 'vision', label: 'Omni候选', desc: 'OmniParser 本地视觉检测框；为空不代表 ROI VLM 没有运行' },
  { key: 'vlm_full', label: '全图 VLM', desc: '旧的全截图/候选裁剪语义模型结果，区别于 ROI VLM 裁剪链路' },
  { key: 'labels', label: '文字标签', desc: '在框上显示元素名称/角色文字' },
  { key: 'info', label: '信息浮层', desc: '左上角/右上角的调试信息' },
];

// 画布叠加用 hex 色值（CSS 变量在 canvas overlay 中不可用）
const providerColor: Record<string, string> = {
  ...baseProviderColor,
  uia: '#3b82f6',
  ocr: '#eab308',
  dom: '#22c55e',
  memory: '#6b7280',
  merged: '#22c55e',
};

function getColorForElement(el: CandidateResponse): string {
  // If multiple sources, use merged color
  if (el.provider_sources.length > 1) return providerColor.merged;
  return providerColor[el.provider_sources[0]] ?? '#3b82f6';
}

interface RoiOverlay {
  id: string;
  purpose: string;
  bounds: number[];
  status: string;
}

function asRecord(value: unknown): Record<string, unknown> | null {
  return value && typeof value === 'object' && !Array.isArray(value) ? value as Record<string, unknown> : null;
}

function asArray(value: unknown): unknown[] {
  return Array.isArray(value) ? value : [];
}

function parseBounds(value: unknown): number[] | null {
  if (!Array.isArray(value) || value.length < 4) return null;
  const bounds = value.slice(0, 4).map((item) => Number(item));
  return bounds.every((item) => Number.isFinite(item)) ? bounds : null;
}

function getNestedNumber(root: Record<string, unknown> | null | undefined, path: string[]): number {
  let current: unknown = root;
  for (const key of path) {
    const record = asRecord(current);
    if (!record) return 0;
    current = record[key];
  }
  return typeof current === 'number' && Number.isFinite(current) ? current : 0;
}

function buildRoiOverlays(canvasDetail: CanvasDetail | null): RoiOverlay[] {
  const plan = asRecord(canvasDetail?.roi_selection_plan);
  const supplements = canvasDetail?.roi_vlm_semantic_supplements ?? [];
  return asArray(plan?.rois).flatMap((item, index) => {
    const roi = asRecord(item);
    const bounds = parseBounds(roi?.bounds);
    if (!roi || !bounds) return [];
    const id = String(roi.roi_id ?? `roi_${index}`);
    const supplement = supplements.find((entry) => entry.roi_id === id);
    return [{
      id,
      purpose: String(roi.purpose ?? roi.role ?? 'roi'),
      bounds,
      status: String(supplement?.status ?? 'pending'),
    }];
  });
}

export default function ReviewCanvas({
  canvasDetail,
  selectedId,
  onSelect,
  overlayEnabled,
  onToggleOverlay,
  activeLayers,
  onToggleLayer,
}: ReviewCanvasProps) {
  const [zoom, setZoom] = useState(100);
  const [showLayerPanel, setShowLayerPanel] = useState(false);
  const [hoveredId, setHoveredId] = useState<string | null>(null);
  const layerPanelRef = useRef<HTMLDivElement>(null);
  const viewportRef = useRef<HTMLDivElement>(null);
  const [viewportSize, setViewportSize] = useState({ width: 0, height: 0 });

  // Click-outside dismiss for layer panel
  useEffect(() => {
    if (!showLayerPanel) return;
    const handleClick = (e: MouseEvent) => {
      if (layerPanelRef.current && !layerPanelRef.current.contains(e.target as Node)) {
        setShowLayerPanel(false);
      }
    };
    document.addEventListener('mousedown', handleClick);
    return () => document.removeEventListener('mousedown', handleClick);
  }, [showLayerPanel]);

  useEffect(() => {
    const node = viewportRef.current;
    if (!node) return;
    const update = () => setViewportSize({ width: node.clientWidth, height: node.clientHeight });
    update();
    const observer = new ResizeObserver(update);
    observer.observe(node);
    return () => observer.disconnect();
  }, []);

  const handleZoomIn = () => setZoom((z) => Math.min(200, z + 20));
  const handleZoomOut = () => setZoom((z) => Math.max(40, z - 20));
  const handleFit = () => setZoom(100);

  const canvasId = canvasDetail?.canvas_id;
  const hasScreenshot = canvasDetail?.has_screenshot ?? false;
  const elements = canvasDetail?.elements ?? [];
  const screenshotW = canvasDetail?.screenshot_width ?? 0;
  const screenshotH = canvasDetail?.screenshot_height ?? 0;
  // Canvas bounds are canonical original-screenshot coordinates. Use the
  // screenshot dimensions as the overlay denominator, not window/client rects.
  const winW = screenshotW || 1;
  const winH = screenshotH || 1;
  const showLabels = activeLayers['labels'] ?? false;
  const showRoiLayer = Boolean(activeLayers['roi']);
  const roiOverlays = useMemo(() => buildRoiOverlays(canvasDetail), [canvasDetail]);
  const providerDetails = canvasDetail?.provider_trace?.provider_details;
  const omniCandidateCount = getNestedNumber(asRecord(providerDetails), ['vision_provider', 'candidate_count'])
    || getNestedNumber(asRecord(providerDetails), ['vision', 'candidate_count']);
  const fullVlmRegionCount = Array.isArray(canvasDetail?.vlm_semantic_model?.regions)
    ? canvasDetail.vlm_semantic_model.regions.length
    : 0;

  // Build screenshot URL
  const screenshotUrl = useMemo(() => {
    if (!canvasId || !hasScreenshot) return null;
    if (overlayEnabled) {
      const activeLayerKeys = Object.entries(activeLayers)
        .filter(([, v]) => v)
        .map(([k]) => k);
      return getCanvasScreenshotUrl(canvasId, true, activeLayerKeys.length > 0 ? activeLayerKeys : undefined);
    }
    return getCanvasScreenshotUrl(canvasId, false);
  }, [canvasId, hasScreenshot, overlayEnabled, activeLayers]);

  const selectedElement = elements.find((e) => e.element_id === selectedId) ?? null;

  return (
    <div data-cmp="ReviewCanvas" className="flex flex-col h-full" style={{ background: 'var(--prc-bg)' }}>
      {/* Toolbar */}
      <div className="flex items-center justify-between px-3 py-1.5 border-b flex-shrink-0" style={{ background: 'var(--prc-surface)', borderColor: 'var(--prc-border)' }}>
        {/* Left: overlay toggle */}
        <div className="flex items-center gap-2">
          <button
            className="px-2.5 h-6 rounded text-xs font-medium transition-all"
            style={{
              background: !overlayEnabled ? 'var(--prc-accent)' : 'var(--prc-surface-2)',
              color: !overlayEnabled ? 'white' : 'var(--prc-text-dim)',
              border: `1px solid ${!overlayEnabled ? 'var(--prc-accent)' : 'var(--prc-border)'}`,
            }}
            onClick={() => { if (overlayEnabled) onToggleOverlay(); }}
          >
            原图
          </button>
          <button
            className="px-2.5 h-6 rounded text-xs font-medium transition-all"
            style={{
              background: overlayEnabled ? 'var(--prc-accent)' : 'var(--prc-surface-2)',
              color: overlayEnabled ? 'white' : 'var(--prc-text-dim)',
              border: `1px solid ${overlayEnabled ? 'var(--prc-accent)' : 'var(--prc-border)'}`,
            }}
            onClick={() => { if (!overlayEnabled) onToggleOverlay(); }}
          >
            叠加
          </button>
        </div>

        {/* Right: layers + zoom */}
        <div className="flex items-center gap-2">
          {/* Layer panel (only when overlay is enabled) */}
          {overlayEnabled && (
            <div className="relative" ref={layerPanelRef}>
              <button
                className="prc-btn prc-btn-ghost flex items-center gap-1.5 h-6 px-2.5"
                onClick={() => setShowLayerPanel(!showLayerPanel)}
                style={{ fontSize: '0.73rem' }}
              >
                <Layers size={11} />
                图层
                {showLayerPanel ? <EyeOff size={10} /> : <Eye size={10} />}
              </button>
              {showLayerPanel && (
                <div
                  className="absolute right-0 top-8 z-50 rounded shadow-custom py-2 px-2"
                  style={{ background: 'var(--prc-surface-2)', border: '1px solid var(--prc-border)', minWidth: '140px' }}
                >
                  <div className="prc-text-dim mono mb-1.5 px-1" style={{ fontSize: '0.62rem' }}>叠加图层</div>
                  {layerDefs.map((ld) => (
                    <div key={ld.key} className="mb-0.5">
                      <button
                        className="w-full flex items-center justify-between px-1.5 py-1 rounded text-xs transition-colors"
                        style={{ color: activeLayers[ld.key] ? 'var(--prc-text)' : 'var(--prc-text-dim)' }}
                        onClick={() => onToggleLayer(ld.key)}
                      >
                        <span>{ld.label}</span>
                        {activeLayers[ld.key] ? <ToggleRight size={14} style={{ color: 'var(--prc-accent)' }} /> : <ToggleLeft size={14} className="prc-text-dim" />}
                      </button>
                      <div className="px-1.5 pb-0.5" style={{ fontSize: '0.55rem', color: 'var(--prc-text-dim)', opacity: 0.7, lineHeight: 1.3 }}>
                        {ld.desc}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

          {/* Zoom */}
          <div className="flex items-center gap-1 rounded px-1.5 py-1" style={{ background: 'var(--prc-surface-2)', border: '1px solid var(--prc-border)' }}>
            <button onClick={handleZoomOut} className="prc-text-dim hover:text-white transition-colors p-0.5">
              <ZoomOut size={12} />
            </button>
            <span className="mono px-1" style={{ fontSize: '0.68rem', color: 'var(--prc-text)' }}>{zoom}%</span>
            <button onClick={handleZoomIn} className="prc-text-dim hover:text-white transition-colors p-0.5">
              <ZoomIn size={12} />
            </button>
            <div className="w-px h-3 mx-1" style={{ background: 'var(--prc-border)' }} />
            <button onClick={handleFit} className="prc-text-dim hover:text-white transition-colors p-0.5">
              <Maximize2 size={11} />
            </button>
          </div>

          {/* Element count */}
          <div className="flex items-center gap-1.5 mono" style={{ fontSize: '0.65rem', color: 'var(--prc-text-dim)' }}>
            <Monitor size={10} />
            <span>{elements.length} 元素</span>
          </div>
        </div>
      </div>

      {/* Main Canvas Area */}
      <div ref={viewportRef} className="flex-1 overflow-auto flex items-start justify-center p-6" style={{ background: 'var(--prc-bg)' }}>
        {!canvasDetail ? (
          <div className="flex flex-col items-center justify-center py-20">
            <ImageIcon size={32} className="prc-text-dim mb-3" />
            <span className="prc-text-dim text-xs">选择窗口并 {t('observe', '观察')} 以查看截图</span>
          </div>
        ) : (() => {
          // Use real screenshot dimensions for aspect ratio; fall back to window dims
          const refW = screenshotW > 0 ? screenshotW : (canvasDetail.window_width || 800);
          const refH = screenshotH > 0 ? screenshotH : (canvasDetail.window_height || 600);
          const scale = zoom / 100;
          const availableW = Math.max(320, (viewportSize.width || refW) - 48);
          const availableH = Math.max(240, (viewportSize.height || refH) - 48);
          const fitScale = Math.max(0.1, Math.min(availableW / refW, availableH / refH));
          const displayW = refW * fitScale * scale;
          const displayH = refH * fitScale * scale;

          return (
          <div
            className="relative select-none"
            style={{
              width: `${displayW}px`,
              height: `${displayH}px`,
              transition: 'all 0.2s',
            }}
          >
            {/* Screenshot image */}
            {screenshotUrl ? (
              <img
                src={screenshotUrl}
                alt="Window screenshot"
                className="absolute inset-0 w-full h-full"
                style={{ borderRadius: '4px' }}
                draggable={false}
              />
            ) : (
              <div
                className="absolute inset-0 flex items-center justify-center rounded"
                style={{ background: 'var(--prc-surface-2)', border: '1px solid var(--prc-border)' }}
              >
                <div className="flex flex-col items-center gap-2">
                  <ImageIcon size={24} className="prc-text-dim" />
                  <span className="prc-text-dim text-xs">无截图数据</span>
                </div>
              </div>
            )}

            {/* ROI crop plan overlay. This is frontend-only so it remains visible
                even when the backend screenshot overlay has no ROI layer yet. */}
            {showRoiLayer && roiOverlays.map((roi) => {
              const [l, t, r, b] = roi.bounds;
              const leftPct = (l / winW) * 100;
              const topPct = (t / winH) * 100;
              const widthPct = ((r - l) / winW) * 100;
              const heightPct = ((b - t) / winH) * 100;
              return (
                <div
                  key={roi.id}
                  className="absolute pointer-events-none"
                  style={{
                    left: `${leftPct}%`,
                    top: `${topPct}%`,
                    width: `${widthPct}%`,
                    height: `${heightPct}%`,
                    border: '1.5px dashed #f59e0b',
                    background: 'rgba(245,158,11,0.055)',
                    zIndex: 18,
                    borderRadius: '2px',
                  }}
                >
                  <div
                    className="absolute left-1 top-1 flex items-center gap-1 rounded px-1.5 py-0.5"
                    style={{
                      maxWidth: 'calc(100% - 8px)',
                      background: 'rgba(15,23,42,0.82)',
                      color: '#fbbf24',
                      fontSize: '0.56rem',
                      lineHeight: 1.2,
                    }}
                  >
                    <span className="mono">{roi.id}</span>
                    <span className="truncate">{roi.purpose}</span>
                    <span className="mono prc-text-dim">{roi.status}</span>
                  </div>
                </div>
              );
            })}

            {/* Frontend bbox overlays (only in non-overlay mode) */}
            {!overlayEnabled && elements.map((el) => {
              if (!el.bounds || el.bounds.length < 4) return null;
              const [l, t, r, b] = el.bounds;
              const isSelected = el.element_id === selectedId;
              const isHovered = el.element_id === hoveredId;
              const color = getColorForElement(el);
              const isDynamic = el.region_id && DYNAMIC_CONTENT_ZONES.has(el.region_id);

              // Convert to percentage of canvas
              const leftPct = (l / winW) * 100;
              const topPct = (t / winH) * 100;
              const widthPct = ((r - l) / winW) * 100;
              const heightPct = ((b - t) / winH) * 100;

              const borderStyle = isDynamic ? 'dashed' : 'solid';
              const borderValue = isSelected
                ? `2px solid white`
                : isHovered
                  ? `1.5px ${borderStyle} ${color}`
                  : `1px ${borderStyle} ${color}90`;

              return (
                <div
                  key={el.element_id}
                  className="absolute cursor-pointer transition-all"
                  style={{
                    left: `${leftPct}%`,
                    top: `${topPct}%`,
                    width: `${widthPct}%`,
                    height: `${heightPct}%`,
                    border: borderValue,
                    background: isSelected ? `rgba(255,255,255,0.08)` : isHovered ? `${color}15` : `${color}08`,
                    boxShadow: isSelected ? `0 0 0 1px var(--prc-accent), 0 0 16px rgba(59,130,246,0.35)` : 'none',
                    zIndex: isSelected ? 20 : isHovered ? 15 : 10,
                    borderRadius: '2px',
                  }}
                  onClick={() => onSelect(el.element_id)}
                  onMouseEnter={() => setHoveredId(el.element_id)}
                  onMouseLeave={() => setHoveredId(null)}
                >
                  {/* Dynamic content indicator */}
                  {isDynamic && (
                    <span
                      className="absolute pointer-events-none"
                      style={{
                        top: '1px',
                        right: '2px',
                        fontSize: '0.5rem',
                        color: 'var(--prc-muted)',
                        opacity: 0.7,
                        lineHeight: 1,
                      }}
                    >
                      动态
                    </span>
                  )}
                  {/* Label tag — only when labels layer is on */}
                  {showLabels && (isSelected || isHovered) && (
                    <div
                      className="absolute flex items-center gap-1 px-1 py-0.5 rounded-sm pointer-events-none"
                      style={{
                        bottom: '100%',
                        left: 0,
                        marginBottom: '2px',
                        background: isSelected ? 'var(--prc-accent)' : color,
                        opacity: 0.9,
                        zIndex: 30,
                        maxWidth: '200px',
                        whiteSpace: 'nowrap',
                      }}
                    >
                      <span className="mono text-white" style={{ fontSize: '0.58rem', fontWeight: 600 }}>
                        {el.semantic_role}
                      </span>
                      <span className="text-white truncate" style={{ fontSize: '0.6rem' }}>
                        {el.text || el.name || el.element_id}
                      </span>
                    </div>
                  )}
                </div>
              );
            })}

            {/* Geometric region overlays (neutral gray dashed borders, no semantic label) */}
            {(activeLayers['geometric_regions'] ?? false) && (canvasDetail.geometric_regions ?? []).map((gr) => {
              if (!gr.bounds || gr.bounds.length < 4) return null;
              const [l, t, r, b] = gr.bounds;
              const leftPct = (l / winW) * 100;
              const topPct = (t / winH) * 100;
              const widthPct = ((r - l) / winW) * 100;
              const heightPct = ((b - t) / winH) * 100;

              return (
                <div
                  key={gr.region_id}
                  className="absolute pointer-events-none"
                  style={{
                    left: `${leftPct}%`,
                    top: `${topPct}%`,
                    width: `${widthPct}%`,
                    height: `${heightPct}%`,
                    border: '1px dashed #6b7280',
                    background: 'rgba(107,114,128,0.04)',
                    zIndex: 5,
                    borderRadius: '1px',
                  }}
                >
                  <span
                    className="absolute mono pointer-events-none"
                    style={{
                      top: '1px',
                      left: '3px',
                      fontSize: '0.5rem',
                      color: '#6b7280',
                      opacity: 0.65,
                      lineHeight: 1,
                    }}
                  >
                    {gr.region_id}
                  </span>
                </div>
              );
            })}

          </div>
          );
        })()}
      </div>

      {/* Status Bar */}
      <div className="flex items-center justify-between px-3 py-1 border-t flex-shrink-0" style={{ background: 'var(--prc-surface)', borderColor: 'var(--prc-border)' }}>
        <div className="flex items-center gap-4">
          {Object.entries(providerColor).filter(([k]) => k !== 'merged').map(([key, color]) => (
            <div key={key} className="flex items-center gap-1">
              <div className="w-2 h-2 rounded-sm" style={{ background: color }} />
              <span className="prc-text-dim" style={{ fontSize: '0.6rem' }}>{providerLabels[key] ?? key.toUpperCase()}</span>
            </div>
          ))}
        </div>
        <div className="flex items-center gap-3">
          {canvasDetail && (
            <>
              <span className="mono prc-text-dim" style={{ fontSize: '0.6rem' }}>
                ROI {roiOverlays.length}
              </span>
              <span className="mono prc-text-dim" style={{ fontSize: '0.6rem' }}>
                Omni {omniCandidateCount}
              </span>
              <span className="mono prc-text-dim" style={{ fontSize: '0.6rem' }}>
                全图VLM {fullVlmRegionCount}
              </span>
              <span className="mono prc-text-dim" style={{ fontSize: '0.6rem' }}>
                {screenshotW > 0 ? `${screenshotW}×${screenshotH}` : `${canvasDetail.window_width}×${canvasDetail.window_height}`}
              </span>
              <span className="mono prc-text-dim" style={{ fontSize: '0.6rem' }}>
                {canvasDetail.canvas_id}
              </span>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
