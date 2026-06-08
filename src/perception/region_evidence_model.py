"""UniversalRegionEngine P0: Region Evidence Model.

Data models for PageRegion, RegionSubstructure, BoundaryEvidence,
and RegionSemanticHint. P0 outputs shadow artifacts only,
does NOT participate in the main observe/query/ROI pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ── Skeleton source constants ──────────────────────────────────

SKELETON_SOURCE_GEOMETRIC = "geometric_region"
SKELETON_SOURCE_STRUCTURED_GROUP = "structured_group"
SKELETON_SOURCE_BAND_PROPOSAL = "band_proposal"
SKELETON_SOURCE_U4_CANDIDATE = "u4_candidate"
SKELETON_SOURCE_FULL_WINDOW = "full_window_fallback"


# ── Data models ────────────────────────────────────────────────

@dataclass
class RegionSubstructure:
    """Region internal substructure. Not a page-level region."""
    sub_id: str
    sub_type: str  # text_row / list_item / code_line / icon / control / ...
    bounds: tuple[int, int, int, int]
    evidence_sources: list[str]  # ["ocr"] / ["vision"] / ["uia"]
    confidence: float  # 0-1
    text: str = ""
    control_type: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "sub_id": self.sub_id,
            "sub_type": self.sub_type,
            "bounds": list(self.bounds),
            "evidence_sources": list(self.evidence_sources),
            "confidence": round(self.confidence, 3),
            "text": self.text,
            "control_type": self.control_type,
        }


@dataclass
class BoundaryEvidence:
    """Evidence for why a region boundary exists."""
    boundary_id: str
    direction: str  # "horizontal" / "vertical"
    position: int  # y or x coordinate
    evidence_type: str  # "whitespace_gap" / "density_transition" / "uia_boundary" / "visual_change"
    strength: float  # 0-1
    source: str  # "pixel" / "uia" / "ocr"

    def to_dict(self) -> dict[str, Any]:
        return {
            "boundary_id": self.boundary_id,
            "direction": self.direction,
            "position": self.position,
            "evidence_type": self.evidence_type,
            "strength": round(self.strength, 3),
            "source": self.source,
        }


@dataclass
class RegionSemanticHint:
    """Semantic hint for a region. P0: always empty."""
    hint_id: str
    source: str  # "roi_vlm" / "memory" / "feedback" / "local_rule"
    label: str
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "hint_id": self.hint_id,
            "source": self.source,
            "label": self.label,
            "confidence": round(self.confidence, 3),
        }


@dataclass
class PageRegion:
    """Page-level functional region. P0: all region_type=UNKNOWN."""
    region_id: str
    bounds: tuple[int, int, int, int]
    region_type: str  # P0: always "unknown"
    confidence: float  # P0: always 0.0
    skeleton_source: str
    evidence_sources: list[str]
    substructures: list[RegionSubstructure] = field(default_factory=list)
    boundary_evidence: list[BoundaryEvidence] = field(default_factory=list)
    semantic_hints: list[RegionSemanticHint] = field(default_factory=list)
    area_ratio: float = 0.0
    evidence_summary: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "region_id": self.region_id,
            "bounds": list(self.bounds),
            "region_type": self.region_type,
            "confidence": round(self.confidence, 3),
            "skeleton_source": self.skeleton_source,
            "evidence_sources": list(self.evidence_sources),
            "substructures": [s.to_dict() for s in self.substructures],
            "boundary_evidence": [b.to_dict() for b in self.boundary_evidence],
            "semantic_hints": [h.to_dict() for h in self.semantic_hints],
            "area_ratio": round(self.area_ratio, 4),
            "evidence_summary": dict(self.evidence_summary),
        }


# ── Helper functions ───────────────────────────────────────────

def compute_evidence_sources(
    ocr_count: int,
    vision_count: int,
    uia_count: int,
    has_pixel_evidence: bool = False,
) -> list[str]:
    """Compute evidence source list from counts."""
    sources: list[str] = []
    if ocr_count > 0:
        sources.append("ocr")
    if vision_count > 0:
        sources.append("vision")
    if uia_count > 0:
        sources.append("uia")
    if has_pixel_evidence:
        sources.append("pixel")
    return sources


def compute_evidence_summary(
    ocr_count: int,
    vision_count: int,
    uia_count: int,
) -> dict[str, int]:
    """Compute evidence summary dict."""
    return {
        "ocr_count": ocr_count,
        "vision_count": vision_count,
        "uia_count": uia_count,
    }
