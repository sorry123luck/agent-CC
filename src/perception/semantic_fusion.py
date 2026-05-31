"""Semantic fusion engine — Layer 2 of the two-layer architecture.

Collects evidence from UIA, OCR, OmniParser, and VLM for each geometric region,
then applies fusion rules to assign semantic labels with confidence scores.

When evidence is insufficient the region stays unknown (semantic_label=None).
No hardcoded app-specific rules. No Chinese keyword matching.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .geometric_partitioner import GeometricRegion


# =============================================================================
# Data model
# =============================================================================


@dataclass
class RegionEvidence:
    """Collected evidence for a single geometric region."""

    region_id: str
    uia_element_count: int = 0
    uia_control_types: list[str] = field(default_factory=list)
    ocr_text_summary: str = ""
    omni_detection_count: int = 0
    position_label: str = ""  # "top", "bottom", "left", "right", "center"
    area_ratio: float = 0.0
    aspect_ratio: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "region_id": self.region_id,
            "uia_element_count": self.uia_element_count,
            "uia_control_types": self.uia_control_types,
            "ocr_text_summary": self.ocr_text_summary[:200],
            "omni_detection_count": self.omni_detection_count,
            "position_label": self.position_label,
            "area_ratio": round(self.area_ratio, 4),
            "aspect_ratio": round(self.aspect_ratio, 3),
        }


@dataclass
class FusionResult:
    """Fusion output for one region."""

    region_id: str
    semantic_label: str | None = None
    confidence: float = 0.0
    evidence_summary: dict[str, Any] = field(default_factory=dict)
    vlm_annotation: dict[str, Any] | None = None
    fusion_source: str = "none"  # "uia_geometry", "ocr_pattern", "vlm", "none"
    reason: str = ""  # why this label was chosen, or why unknown

    def to_dict(self) -> dict[str, Any]:
        return {
            "region_id": self.region_id,
            "semantic_label": self.semantic_label,
            "confidence": self.confidence,
            "evidence": self.evidence_summary,
            "vlm_annotation": self.vlm_annotation,
            "fusion_source": self.fusion_source,
            "reason": self.reason,
        }


@dataclass
class FusionDiagnostics:
    """Aggregate diagnostics for one observe run."""

    geometric_region_count: int = 0
    labeled_count: int = 0
    unknown_count: int = 0
    regions: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "geometric_region_count": self.geometric_region_count,
            "labeled_count": self.labeled_count,
            "unknown_count": self.unknown_count,
            "regions": self.regions,
        }


# =============================================================================
# Configuration
# =============================================================================


@dataclass(frozen=True)
class FusionConfig:
    """Centralized thresholds for SemanticFusion.

    All values are initial defaults — tune based on diagnostic output.
    """

    uia_input_confidence: float = 0.85
    action_bar_min_buttons: int = 3
    action_bar_band_ratio: float = 0.22
    navigation_min_items: int = 3
    navigation_side_band_ratio: float = 0.30
    content_min_area_ratio: float = 0.40
    unknown_threshold: float = 0.35


# =============================================================================
# SemanticFusion engine
# =============================================================================


class SemanticFusion:
    """Layer 2 engine: collects multi-source evidence and assigns semantic labels.

    Usage::

        fusion = SemanticFusion(config)
        evidence = fusion.collect_evidence(geometric_regions, raw_elements,
                                           ocr_blocks, vision_candidates)
        results = fusion.fuse(evidence)
        diagnostics = fusion.build_diagnostics(results)
    """

    def __init__(self, config: FusionConfig | None = None):
        self.config = config or FusionConfig()

    # -------------------------------------------------------------------------
    # Evidence collection
    # -------------------------------------------------------------------------

    def collect_evidence(
        self,
        geometric_regions: list[GeometricRegion],
        raw_elements: list[dict[str, Any]] | None = None,
        ocr_blocks: list[dict[str, Any]] | None = None,
        vision_candidates: list[dict[str, Any]] | None = None,
        image_width: int = 1920,
        image_height: int = 1080,
    ) -> dict[str, RegionEvidence]:
        """Assign UIA/OCR/Omni data to each geometric region by spatial overlap."""

        raw_elements = raw_elements or []
        ocr_blocks = ocr_blocks or []
        vision_candidates = vision_candidates or []

        total_area = image_width * image_height

        evidence_map: dict[str, RegionEvidence] = {}
        for gr in geometric_regions:
            ev = RegionEvidence(region_id=gr.region_id)
            l, t, r, b = gr.bounds
            region_area = max(1, (r - l) * (b - t))
            ev.area_ratio = region_area / max(1, total_area)
            ev.aspect_ratio = (r - l) / max(1, (b - t))
            ev.position_label = self._classify_position(l, t, r, b, image_width, image_height, self.config)
            evidence_map[gr.region_id] = ev

        # Assign UIA elements to regions by center-point overlap
        for el in raw_elements:
            el_bounds = el.get("bounds") or el.get("bounding_rectangle")
            if not el_bounds or len(el_bounds) < 4:
                continue
            cx = (el_bounds[0] + el_bounds[2]) // 2
            cy = (el_bounds[1] + el_bounds[3]) // 2
            for gr in geometric_regions:
                l, t, r, b = gr.bounds
                if l <= cx < r and t <= cy < b:
                    ev = evidence_map[gr.region_id]
                    ev.uia_element_count += 1
                    ctrl_type = el.get("control_type") or el.get("ControlType", "")
                    if ctrl_type:
                        ev.uia_control_types.append(str(ctrl_type))
                    break

        # Count OCR blocks per region
        for block in ocr_blocks:
            bbox = block.get("bbox") or block.get("bounds", [])
            if not bbox or len(bbox) < 4:
                continue
            bcx = (bbox[0] + bbox[2]) // 2
            bcy = (bbox[1] + bbox[3]) // 2
            for gr in geometric_regions:
                l, t, r, b = gr.bounds
                if l <= bcx < r and t <= bcy < b:
                    ev = evidence_map[gr.region_id]
                    text = str(block.get("text", ""))
                    if len(ev.ocr_text_summary) < 200:
                        remaining = 200 - len(ev.ocr_text_summary)
                        ev.ocr_text_summary += text[:remaining] + " "
                    break

        # Count Omni detections per region
        for det in vision_candidates:
            bbox = det.get("bbox", [])
            if not bbox or len(bbox) < 4:
                continue
            dcx = (bbox[0] + bbox[2]) // 2
            dcy = (bbox[1] + bbox[3]) // 2
            for gr in geometric_regions:
                l, t, r, b = gr.bounds
                if l <= dcx < r and t <= dcy < b:
                    evidence_map[gr.region_id].omni_detection_count += 1
                    break

        return evidence_map

    # -------------------------------------------------------------------------
    # Fusion rules
    # -------------------------------------------------------------------------

    def fuse(self, evidence_map: dict[str, RegionEvidence]) -> list[FusionResult]:
        """Apply deterministic fusion rules to assign semantic labels."""

        results: list[FusionResult] = []
        cfg = self.config

        for ev in evidence_map.values():
            result = self._fuse_one(ev, cfg)
            results.append(result)

        return results

    def _fuse_one(self, ev: RegionEvidence, cfg: FusionConfig) -> FusionResult:
        """Apply fusion rules to a single region's evidence."""

        control_types = [t.lower() for t in ev.uia_control_types]

        # Rule: UIA EditControl/TextBox → input_area
        edit_count = sum(1 for t in control_types if t in ("edit", "editcontrol", "textbox", "document"))
        if edit_count >= 1:
            return FusionResult(
                region_id=ev.region_id,
                semantic_label="input_area",
                confidence=cfg.uia_input_confidence,
                evidence_summary=ev.to_dict(),
                fusion_source="uia_geometry",
                reason="UIA EditControl/TextBox present",
            )

        # Rule: ≥3 buttons in bottom band → action_bar
        btn_count = sum(1 for t in control_types if t in ("button", "buttonelement", "splitbutton"))
        if btn_count >= cfg.action_bar_min_buttons and ev.position_label == "bottom":
            return FusionResult(
                region_id=ev.region_id,
                semantic_label="action_bar",
                confidence=0.65,
                evidence_summary=ev.to_dict(),
                fusion_source="uia_geometry",
                reason=f"{btn_count} buttons in bottom region",
            )

        # Rule: ≥3 list/tree items in left band → navigation
        nav_count = sum(
            1 for t in control_types
            if t in ("list", "listitem", "tree", "treeitem", "menuitem", "tabitem", "pivotitem")
        )
        if nav_count >= cfg.navigation_min_items and ev.position_label in ("left", "right"):
            return FusionResult(
                region_id=ev.region_id,
                semantic_label="navigation",
                confidence=0.55,
                evidence_summary=ev.to_dict(),
                fusion_source="uia_geometry",
                reason=f"{nav_count} list/tree items in side band",
            )

        # Rule: large center pane → content
        if ev.position_label == "center" and ev.area_ratio >= cfg.content_min_area_ratio:
            return FusionResult(
                region_id=ev.region_id,
                semantic_label="content",
                confidence=0.50,
                evidence_summary=ev.to_dict(),
                fusion_source="uia_geometry",
                reason=f"large center pane area={ev.area_ratio:.3f}",
            )

        # Rule: mixed buttons + edit in bottom → input_area (weaker)
        if ev.position_label == "bottom" and btn_count >= 1 and edit_count >= 1:
            return FusionResult(
                region_id=ev.region_id,
                semantic_label="input_area",
                confidence=0.50,
                evidence_summary=ev.to_dict(),
                fusion_source="uia_geometry",
                reason="mixed buttons+edit in bottom region",
            )

        # Insufficient evidence → unknown
        return FusionResult(
            region_id=ev.region_id,
            semantic_label=None,
            confidence=0.0,
            evidence_summary=ev.to_dict(),
            fusion_source="none",
            reason="insufficient_evidence",
        )

    # -------------------------------------------------------------------------
    # VLM annotation (stub — non-blocking, async in later phase)
    # -------------------------------------------------------------------------

    def apply_vlm_annotations(
        self,
        fusion_results: list[FusionResult],
        vlm_response: dict[str, Any] | None,
    ) -> list[FusionResult]:
        """Merge VLM annotations into fusion results.

        VLM annotations override low-confidence (< unknown_threshold) labels.
        This is a stub in the current phase — VLM integration happens later.
        """
        if not vlm_response:
            return fusion_results

        annotations = vlm_response.get("regions", {})
        updated: list[FusionResult] = []
        for result in fusion_results:
            anno = annotations.get(result.region_id)
            if anno is None:
                updated.append(result)
                continue
            label = anno.get("semantic_label")
            conf = anno.get("confidence", 0.0)
            if label and conf > result.confidence:
                updated.append(FusionResult(
                    region_id=result.region_id,
                    semantic_label=label,
                    confidence=conf,
                    evidence_summary=result.evidence_summary,
                    vlm_annotation=anno,
                    fusion_source="vlm",
                    reason="VLM annotation overrode low-confidence result",
                ))
            else:
                updated.append(result)

        return updated

    # -------------------------------------------------------------------------
    # Diagnostics
    # -------------------------------------------------------------------------

    def build_diagnostics(self, fusion_results: list[FusionResult]) -> FusionDiagnostics:
        """Build aggregate diagnostic summary."""

        diag = FusionDiagnostics()
        diag.geometric_region_count = len(fusion_results)
        diag.labeled_count = sum(1 for r in fusion_results if r.semantic_label is not None)
        diag.unknown_count = sum(1 for r in fusion_results if r.semantic_label is None)
        diag.regions = [r.to_dict() for r in fusion_results]
        return diag

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _classify_position(
        left: int, top: int, right: int, bottom: int,
        img_w: int, img_h: int,
        cfg: FusionConfig | None = None,
    ) -> str:
        """Classify region position in the window."""
        _ = (left + right) / 2, (top + bottom) / 2  # available for future use

        side_ratio = cfg.navigation_side_band_ratio if cfg else 0.30
        bottom_ratio = cfg.action_bar_band_ratio if cfg else 0.22

        # Top band
        if top < img_h * 0.10 and bottom < img_h * side_ratio:
            return "top"
        # Bottom band
        if top > img_h * (1.0 - bottom_ratio):
            return "bottom"
        # Left band
        if left < img_w * side_ratio and right < img_w * 0.50:
            return "left"
        # Right band
        if left > img_w * 0.50:
            return "right"
        # Center
        return "center"
