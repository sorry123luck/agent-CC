"""Visual Trace Layer (VTL) data models.

Shadow-only data models for VisualTraceObject.
Does NOT participate in the main observe/query/ROI pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ── Object level constants ──────────────────────────────────────

OBJECT_LEVEL_PAGE_CONTAINER = "page_container"
OBJECT_LEVEL_LOCAL_OBJECT = "local_object"
OBJECT_LEVEL_SUBSTRUCTURE = "substructure"


# ── Object class constants ──────────────────────────────────────

OBJECT_CLASS_PAGE_CONTAINER = "page_container"
OBJECT_CLASS_ACTIONABLE = "actionable_object"
OBJECT_CLASS_CONTENT = "content_object"
OBJECT_CLASS_TEXT_SUB = "text_substructure"
OBJECT_CLASS_VISUAL_CANDIDATE = "local_visual_candidate"  # pixel-only, no Omni/UIA


# ── Detection method constants ──────────────────────────────────

DETECT_BG_COLOR_BLOCK = "bg_color_block"
DETECT_LOW_TEXTURE = "low_texture"
DETECT_SHELL_TOP_BAND = "shell_top_band"
DETECT_SHELL_SIDE_RAIL = "shell_side_rail"
DETECT_BOTTOM_BAR = "bottom_bar"
DETECT_CONNECTED_COMPONENT = "connected_component"
DETECT_BORDERED_RECT = "bordered_rect"
DETECT_COLOR_BLOB = "color_blob"


# ── Data model ──────────────────────────────────────────────────

@dataclass
class VisualTraceObject:
    """Visual trace layer object."""
    trace_id: str
    bounds: tuple[int, int, int, int]
    shape_type: str  # "rectangle" / "rounded_rect" / "blob" / "container" / "unknown"
    object_level: str  # "page_container" / "local_object" / "substructure"
    object_class: str  # "page_container" / "actionable_object" / "content_object" / "text_substructure"
    detection_method: str
    evidence_sources: list[str] = field(default_factory=list)
    contained_omni_ids: list[str] = field(default_factory=list)
    contained_ocr_ids: list[str] = field(default_factory=list)
    parent_trace_id: str | None = None
    child_trace_ids: list[str] = field(default_factory=list)
    confidence: float = 0.0
    semantic_hint: str = "unknown"
    area_ratio: float = 0.0
    aspect_ratio: float = 0.0
    pixel_uniformity: float = 0.0
    edge_density: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "bounds": [int(x) for x in self.bounds],
            "shape_type": self.shape_type,
            "object_level": self.object_level,
            "object_class": self.object_class,
            "detection_method": self.detection_method,
            "evidence_sources": list(self.evidence_sources),
            "contained_omni_ids": list(self.contained_omni_ids),
            "contained_ocr_ids": list(self.contained_ocr_ids),
            "parent_trace_id": self.parent_trace_id,
            "child_trace_ids": list(self.child_trace_ids),
            "confidence": round(self.confidence, 3),
            "semantic_hint": self.semantic_hint,
            "area_ratio": round(self.area_ratio, 4),
            "aspect_ratio": round(self.aspect_ratio, 3),
            "pixel_uniformity": round(self.pixel_uniformity, 1),
            "edge_density": round(self.edge_density, 4),
        }


# ── Helper ──────────────────────────────────────────────────────

def make_trace_object(
    trace_id: str,
    bounds: tuple[int, int, int, int],
    shape_type: str,
    object_level: str,
    object_class: str,
    detection_method: str,
    confidence: float,
    window_area: int,
    evidence_sources: list[str] | None = None,
) -> VisualTraceObject:
    """Create a VisualTraceObject with computed metrics."""
    l, t, r, b = (int(x) for x in bounds)
    w = max(1, r - l)
    h = max(1, b - t)
    area = w * h
    return VisualTraceObject(
        trace_id=trace_id,
        bounds=bounds,
        shape_type=shape_type,
        object_level=object_level,
        object_class=object_class,
        detection_method=detection_method,
        evidence_sources=evidence_sources or ["pixel"],
        confidence=confidence,
        area_ratio=area / max(1, window_area),
        aspect_ratio=round(w / h, 3) if h > 0 else 0.0,
    )
