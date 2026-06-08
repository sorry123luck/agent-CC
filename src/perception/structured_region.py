"""StructuredRegion — Universal Region Engine data model (Phase R2).

Read-only overlay that unifies geometric regions, semantic fusion results,
and multi-source evidence into a single structured region representation.

This module does NOT replace ZonePartitioner or page_compiler regions.
It writes to artifacts["structured_region_overlay"] for diagnostics and
future Region Engine phases.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# =============================================================================
# Structure Type
# =============================================================================


class StructureType(Enum):
    """Structural type of a region, based on geometry + evidence."""

    TOP_BAR = "top_bar"
    TOOLBAR = "toolbar"
    SIDE_RAIL = "side_rail"
    LIST_REGION = "list_region"
    GRID_REGION = "grid_region"
    CANVAS_REGION = "canvas_region"
    DOCUMENT_REGION = "document_region"
    MEDIA_CONTROL_BAR = "media_control_bar"
    INPUT_REGION = "input_region"
    STATUS_REGION = "status_region"
    FLOATING_PANEL = "floating_panel"
    DIALOG = "dialog"
    CONTROL_STRIP = "control_strip"      # 通用控制条（非媒体特定）
    CONTENT_STREAM = "content_stream"    # 动态消息流/聊天流/列表流
    UNKNOWN_STRUCTURED = "unknown_structured"


# Map fusion semantic_label → StructureType
_FUSION_LABEL_TO_STRUCTURE: dict[str, StructureType] = {
    "navigation": StructureType.SIDE_RAIL,
    "side_panel": StructureType.SIDE_RAIL,
    "sidebar": StructureType.SIDE_RAIL,
    "action_bar": StructureType.TOOLBAR,
    "toolbar": StructureType.TOOLBAR,
    "top_bar": StructureType.TOP_BAR,
    "header": StructureType.TOP_BAR,
    "title_bar": StructureType.TOP_BAR,
    "status_bar": StructureType.STATUS_REGION,
    "content": StructureType.CANVAS_REGION,
    "main_content": StructureType.CANVAS_REGION,
    "editor": StructureType.DOCUMENT_REGION,
    "document": StructureType.DOCUMENT_REGION,
    "list": StructureType.LIST_REGION,
    "list_panel": StructureType.LIST_REGION,
    "grid": StructureType.GRID_REGION,
    "table": StructureType.GRID_REGION,
    "input": StructureType.INPUT_REGION,
    "composer": StructureType.INPUT_REGION,
    "search": StructureType.INPUT_REGION,
    "media": StructureType.MEDIA_CONTROL_BAR,
    "media_control": StructureType.MEDIA_CONTROL_BAR,
    "message_stream": StructureType.CONTENT_STREAM,
    "chat": StructureType.CONTENT_STREAM,
    "timeline": StructureType.CONTENT_STREAM,
    "dialog": StructureType.DIALOG,
    "modal": StructureType.DIALOG,
    "floating": StructureType.FLOATING_PANEL,
    "panel": StructureType.FLOATING_PANEL,
    "bottom_panel": StructureType.FLOATING_PANEL,
    "terminal": StructureType.FLOATING_PANEL,
}


# =============================================================================
# Region Evidence
# =============================================================================


@dataclass
class RegionEvidence:
    """A single piece of evidence contributing to region classification."""

    source: str  # "geometric", "fusion", "uia", "ocr", "omni", "positional"
    detail: str = ""
    confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "detail": self.detail,
            "confidence": round(self.confidence, 3),
        }


# =============================================================================
# Structured Region
# =============================================================================


@dataclass
class StructuredRegion:
    """A unified structural region with multi-source evidence."""

    region_id: str
    structure_type: StructureType = StructureType.UNKNOWN_STRUCTURED
    semantic_label: str | None = None
    bounds: tuple[int, int, int, int] = (0, 0, 0, 0)
    confidence: float = 0.0
    evidence: list[RegionEvidence] = field(default_factory=list)
    is_stable: bool = False
    roi_eligible: bool = False
    icon_memory_eligible: bool = False
    element_ids: list[str] = field(default_factory=list)
    parent_region_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "region_id": self.region_id,
            "structure_type": self.structure_type.value,
            "semantic_label": self.semantic_label,
            "bounds": list(self.bounds),
            "confidence": round(self.confidence, 3),
            "evidence": [e.to_dict() for e in self.evidence],
            "is_stable": self.is_stable,
            "roi_eligible": self.roi_eligible,
            "icon_memory_eligible": self.icon_memory_eligible,
            "element_ids": list(self.element_ids),
            "parent_region_id": self.parent_region_id,
        }


# =============================================================================
# Overlay Diagnostics
# =============================================================================


@dataclass
class StructuredRegionOverlay:
    """Aggregate overlay output for one observe run.

    R3.2: Contains both original (per-geometric-region) and grouped regions.
    R3.3: Contains band proposals (thin top/bottom bands missed by GeometricPartitioner).
    U1-shadow: Contains shadow comparison (TP band proposals injected as shadow regions).
    """

    regions: list[StructuredRegion] = field(default_factory=list)
    groups: list[StructuredRegion] = field(default_factory=list)
    band_proposals: list[dict[str, Any]] = field(default_factory=list)
    total_regions: int = 0
    unknown_count: int = 0
    with_semantic_label: int = 0
    roi_eligible_count: int = 0
    total_groups: int = 0
    group_unknown_count: int = 0
    group_roi_eligible_count: int = 0
    total_band_proposals: int = 0
    band_proposal_audit: dict[str, Any] = field(default_factory=dict)
    shadow_comparison: dict[str, Any] = field(default_factory=dict)
    uia_aware_comparison: dict[str, Any] = field(default_factory=dict)
    ocr_visual_comparison: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_regions": self.total_regions,
            "unknown_count": self.unknown_count,
            "with_semantic_label": self.with_semantic_label,
            "roi_eligible_count": self.roi_eligible_count,
            "regions": [r.to_dict() for r in self.regions],
            "total_groups": self.total_groups,
            "group_unknown_count": self.group_unknown_count,
            "group_roi_eligible_count": self.group_roi_eligible_count,
            "groups": [g.to_dict() for g in self.groups],
            "total_band_proposals": self.total_band_proposals,
            "band_proposals": list(self.band_proposals),
            "band_proposal_audit": dict(self.band_proposal_audit),
            "shadow_comparison": dict(self.shadow_comparison),
            "uia_aware_comparison": dict(self.uia_aware_comparison),
            "ocr_visual_comparison": dict(self.ocr_visual_comparison),
        }
