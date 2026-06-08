"""StructuredRegion overlay builder (Phase R2 + R3.1).

Read-only adapter that converts GeometricRegion + FusionDiagnostics +
raw elements into StructuredRegionOverlay.

R3.1: Uses StructuralClassifier with combination evidence metrics
to determine structure_type instead of single-keyword rules.

Writes to artifacts["structured_region_overlay"]. Does NOT modify
regions, element.region_id, ROI selector, or query_engine.
"""

from __future__ import annotations

import os
from typing import Any

from src.perception.geometric_partitioner import GeometricRegion
from src.perception.region_grouper import RegionGrouper
from src.perception.structural_classifier import (
    RegionMetrics,
    StructuralClassifier,
    compute_region_metrics,
)
from src.perception.structured_region import (
    RegionEvidence,
    StructuredRegion,
    StructuredRegionOverlay,
    StructureType,
    _FUSION_LABEL_TO_STRUCTURE,
)


# ROI-eligible structure types (toolbar, input, navigation — where icons live)
_ROI_ELIGIBLE_TYPES = {
    StructureType.TOP_BAR,
    StructureType.TOOLBAR,
    StructureType.SIDE_RAIL,
    StructureType.INPUT_REGION,
    StructureType.MEDIA_CONTROL_BAR,
    StructureType.CONTROL_STRIP,
    StructureType.STATUS_REGION,
    StructureType.FLOATING_PANEL,
    StructureType.DIALOG,
}

# Icon-memory-eligible structure types (stable regions with fixed controls)
_ICON_MEMORY_ELIGIBLE_TYPES = {
    StructureType.TOP_BAR,
    StructureType.TOOLBAR,
    StructureType.SIDE_RAIL,
    StructureType.STATUS_REGION,
}


class StructuredRegionOverlayBuilder:
    """Build StructuredRegionOverlay from geometric + fusion + element evidence.

    R3.1: Uses StructuralClassifier with combination evidence metrics.
    """

    def __init__(self) -> None:
        self._classifier = StructuralClassifier()
        self._grouper = RegionGrouper()

    def build(
        self,
        geometric_regions: list[GeometricRegion] | None = None,
        fusion_diagnostics: dict[str, Any] | None = None,
        raw_elements: list[dict[str, Any]] | None = None,
        ocr_blocks: list[dict[str, Any]] | None = None,
        vision_candidates: list[dict[str, Any]] | None = None,
        window_width: int = 0,
        window_height: int = 0,
        screenshot: Any = None,
    ) -> StructuredRegionOverlay:
        """Build overlay. Returns empty overlay if no geometric_regions."""
        if not geometric_regions:
            # Even with no geometric regions, run shadow passes if enabled
            overlay = StructuredRegionOverlay()
            if os.environ.get("OPENCLAW_URE_P0_SHADOW") == "1":
                ure_result = self._run_ure_p0_shadow(
                    [], [], [],
                    ocr_blocks, vision_candidates, raw_elements,
                    window_width, window_height, None,
                )
                overlay.ocr_visual_comparison["universal_region_shadow"] = ure_result
            if os.environ.get("OPENCLAW_VTL_0_SHADOW") == "1":
                vtl_result = self._run_vtl_0_shadow(
                    screenshot, ocr_blocks, vision_candidates, raw_elements,
                    window_width, window_height,
                )
                overlay.ocr_visual_comparison["visual_trace_shadow"] = vtl_result
            return overlay

        # Parse fusion results
        fusion_results = self._parse_fusion_results(fusion_diagnostics)

        # Build original structured regions (per geometric region)
        regions: list[StructuredRegion] = []
        for gr in geometric_regions:
            sr = self._build_region(
                gr,
                fusion_results.get(gr.region_id),
                raw_elements,
                ocr_blocks,
                vision_candidates,
                window_width,
                window_height,
            )
            regions.append(sr)

        # R3.2: Evidence-aware region grouping
        groups_raw = self._grouper.group(
            geometric_regions, window_width, window_height, ocr_blocks, vision_candidates,
        )

        # Build structured regions for groups
        grouped_regions: list[StructuredRegion] = []
        for group in groups_raw:
            # Create a synthetic GeometricRegion for the group
            group_gr = GeometricRegion(
                region_id=group.group_id,
                bounds=group.bounds,
                boundary_evidence=[f"group: {group.reason}"],
                geometry_confidence=0.5,
            )
            # Merge fusion results from child regions
            group_fusion = self._merge_fusion_for_group(group.child_region_ids, fusion_results)

            sr = self._build_region(
                group_gr,
                group_fusion,
                raw_elements,
                ocr_blocks,
                vision_candidates,
                window_width,
                window_height,
            )
            # Add group metadata
            sr.evidence.append(RegionEvidence(
                source="grouping",
                detail=f"children={group.child_region_ids}, alignment={group.alignment}",
                confidence=0.0,
            ))
            grouped_regions.append(sr)

        # Compute aggregates for original regions
        total = len(regions)
        unknown = sum(1 for r in regions if r.structure_type == StructureType.UNKNOWN_STRUCTURED)
        with_label = sum(1 for r in regions if r.semantic_label is not None)
        roi_eligible = sum(1 for r in regions if r.roi_eligible)

        # Compute aggregates for grouped regions
        total_groups = len(grouped_regions)
        group_unknown = sum(1 for r in grouped_regions if r.structure_type == StructureType.UNKNOWN_STRUCTURED)
        group_roi = sum(1 for r in grouped_regions if r.roi_eligible)

        # R3.3: Band proposal pass (read-only diagnostic)
        from src.perception.band_proposal_pass import BandProposalPass
        band_pass = BandProposalPass()
        band_diag = band_pass.detect(
            screenshot=screenshot,
            ocr_blocks=ocr_blocks,
            raw_elements=raw_elements,
            window_width=window_width,
            window_height=window_height,
        )
        band_proposals = [p.to_dict() for p in band_diag.proposals]

        # False proposal audit
        band_audit = self._audit_band_proposals(band_diag.proposals, grouped_regions, window_width, window_height)

        # U1-shadow: inject TP band proposals and compare with original
        shadow_comparison = self._build_shadow_overlay(
            geometric_regions, band_audit,
            fusion_results, raw_elements, ocr_blocks, vision_candidates,
            window_width, window_height,
        )

        # U2-shadow: UIA boundary shadow pass (only when env var set)
        uia_aware_comparison: dict[str, Any] = {}
        if os.environ.get("OPENCLAW_U2_UIA_SHADOW") == "1":
            uia_aware_comparison = self._run_u2_shadow(
                geometric_regions, screenshot, raw_elements,
                fusion_results, ocr_blocks, vision_candidates,
                window_width, window_height,
            )

        # U3-shadow: OCR/visual density region proposal (only when env var set)
        ocr_visual_comparison: dict[str, Any] = {}
        if os.environ.get("OPENCLAW_U3_OCR_SHADOW") == "1":
            ocr_visual_comparison = self._run_u3_shadow(
                geometric_regions, ocr_blocks, raw_elements,
                window_width, window_height,
            )

        # U4-shadow: Layout boundary detection (only when env var set)
        u4_layout = None
        if os.environ.get("OPENCLAW_U4_LAYOUT_SHADOW") == "1":
            u4_result = self._run_u4_shadow(
                screenshot, ocr_blocks, vision_candidates, raw_elements,
                window_width, window_height,
            )
            ocr_visual_comparison["u4_layout"] = u4_result
            u4_layout = u4_result

        # URE P0 shadow: Region Evidence Model (only when env var set)
        if os.environ.get("OPENCLAW_URE_P0_SHADOW") == "1":
            ure_result = self._run_ure_p0_shadow(
                geometric_regions, grouped_regions, band_proposals,
                ocr_blocks, vision_candidates, raw_elements,
                window_width, window_height, u4_layout,
            )
            ocr_visual_comparison["universal_region_shadow"] = ure_result

        # VTL-0 shadow: Visual Trace Layer (only when env var set)
        if os.environ.get("OPENCLAW_VTL_0_SHADOW") == "1":
            vtl_result = self._run_vtl_0_shadow(
                screenshot, ocr_blocks, vision_candidates, raw_elements,
                window_width, window_height,
            )
            ocr_visual_comparison["visual_trace_shadow"] = vtl_result

        return StructuredRegionOverlay(
            regions=regions,
            groups=grouped_regions,
            band_proposals=band_proposals,
            total_regions=total,
            unknown_count=unknown,
            with_semantic_label=with_label,
            roi_eligible_count=roi_eligible,
            total_groups=total_groups,
            group_unknown_count=group_unknown,
            group_roi_eligible_count=group_roi,
            total_band_proposals=len(band_proposals),
            band_proposal_audit=band_audit,
            shadow_comparison=shadow_comparison,
            uia_aware_comparison=uia_aware_comparison,
            ocr_visual_comparison=ocr_visual_comparison,
        )

    def _parse_fusion_results(
        self,
        fusion_diagnostics: dict[str, Any] | None,
    ) -> dict[str, dict[str, Any]]:
        """Parse fusion_diagnostics into region_id → fusion result dict."""
        if not fusion_diagnostics:
            return {}
        results: dict[str, dict[str, Any]] = {}
        for region_data in fusion_diagnostics.get("regions", []):
            rid = region_data.get("region_id", "")
            if rid:
                results[rid] = region_data
        return results

    def _merge_fusion_for_group(
        self,
        child_ids: list[str],
        fusion_results: dict[str, dict[str, Any]],
    ) -> dict[str, Any] | None:
        """Merge fusion results from child regions into a single result for the group.

        Picks the child with highest confidence fusion label.
        """
        best: dict[str, Any] | None = None
        best_conf = -1.0
        for cid in child_ids:
            fr = fusion_results.get(cid)
            if fr:
                conf = float(fr.get("confidence", 0.0))
                if conf > best_conf:
                    best_conf = conf
                    best = fr
        return best

    def _audit_band_proposals(
        self,
        band_proposals: list[Any],
        grouped_regions: list[StructuredRegion],
        window_width: int,
        window_height: int,
    ) -> dict[str, Any]:
        """Audit band proposals for false positives.

        Returns TP/FP/AMB counts with per-band details.
        """
        tp = 0
        fp = 0
        amb = 0
        details: list[dict[str, Any]] = []

        for band in band_proposals:
            bounds = band.bounds if hasattr(band, "bounds") else band.get("bounds", [])
            if len(bounds) < 4:
                continue
            l, t, r, b = bounds
            height_px = b - t
            height_ratio = height_px / max(window_height, 1)
            band_type = band.band_type if hasattr(band, "band_type") else band.get("band_type", "?")
            detection_method = band.detection_method if hasattr(band, "detection_method") else band.get("detection_method", "?")

            # Parse evidence
            evidence = band.evidence if hasattr(band, "evidence") else band.get("evidence", [])
            ocr_count = 0
            uia_count = 0
            boundary_y = None
            for ev in evidence:
                ev_str = str(ev)
                if "ocr_count=" in ev_str:
                    try:
                        ocr_count = int(ev_str.split("ocr_count=")[1].split(",")[0])
                    except ValueError:
                        pass
                if "uia_count=" in ev_str:
                    try:
                        uia_count = int(ev_str.split("uia_count=")[1].split(",")[0])
                    except ValueError:
                        pass
                if "boundary_y=" in ev_str:
                    try:
                        boundary_y = int(ev_str.split("boundary_y=")[1].split(",")[0])
                    except ValueError:
                        pass

            # Check if band overlaps with any existing grouped region
            overlaps_existing = False
            for gr in grouped_regions:
                gl, gt, gr_, gb = gr.bounds
                if t < gb and b > gt:
                    overlap_width = min(r, gr_) - max(l, gl)
                    if overlap_width > (r - l) * 0.5:
                        overlaps_existing = True
                        break

            # Audit rules
            verdict = "TP"
            reason = ""

            # FP: band is too tall
            if band_type == "top_band" and height_ratio > 0.15:
                verdict = "FP"
                reason = f"too_tall ({height_ratio:.1%})"
            elif band_type == "bottom_band" and height_ratio > 0.10:
                verdict = "FP"
                reason = f"too_tall ({height_ratio:.1%})"

            # FP: no evidence at all (no OCR, no UIA, no boundary)
            elif ocr_count == 0 and uia_count == 0 and boundary_y is None:
                verdict = "FP"
                reason = "no_evidence"

            # AMB: overlaps with existing region
            elif overlaps_existing:
                verdict = "AMB"
                reason = "overlaps_existing_region"

            # TP: thin band with OCR content
            elif height_px <= 80 and ocr_count >= 1:
                verdict = "TP"
                reason = f"thin_band_with_content (ocr={ocr_count}, uia={uia_count})"

            # TP/AMB: UIA-only band (menu/toolbar detected by UIA boundaries)
            elif uia_count >= 1 and height_px <= 80:
                verdict = "TP"
                reason = f"uia_boundary_band (uia={uia_count})"

            # AMB: everything else
            else:
                verdict = "AMB"
                reason = f"uncertain (height={height_px}, ocr={ocr_count}, uia={uia_count})"

            if verdict == "TP":
                tp += 1
            elif verdict == "FP":
                fp += 1
            else:
                amb += 1

            details.append({
                "band_id": band.band_id if hasattr(band, "band_id") else band.get("band_id", "?"),
                "band_type": band_type,
                "verdict": verdict,
                "reason": reason,
                "bounds": list(bounds),
                "height_px": height_px,
                "ocr_count": ocr_count,
                "uia_count": uia_count,
                "detection_method": detection_method,
            })

        # Post-pass: detect over-fragmentation in top band stacks.
        # Conservative: only flag 4+ consecutive very thin bands (<20px).
        # Do NOT filter by evidence here — useful/noisy classification
        # happens in shadow_comparison after injection.
        over_fragmented = 0
        top_bands = [d for d in details if d["band_type"] == "top_band" and d["verdict"] == "TP"]
        if len(top_bands) >= 4:
            top_bands.sort(key=lambda d: d["bounds"][1])
            consecutive_thin = 0
            for i, d in enumerate(top_bands):
                if d["height_px"] <= 20:
                    consecutive_thin += 1
                    if consecutive_thin >= 4:
                        for j in range(i - consecutive_thin + 1, i + 1):
                            if top_bands[j]["verdict"] == "TP":
                                top_bands[j]["verdict"] = "AMB"
                                top_bands[j]["reason"] = "top_stack_over_fragmented"
                                tp -= 1
                                amb += 1
                                over_fragmented += 1
                        consecutive_thin = 0
                else:
                    consecutive_thin = 0

        return {
            "tp": tp,
            "fp": fp,
            "amb": amb,
            "total": len(band_proposals),
            "over_fragmented": over_fragmented,
            "details": details,
        }

    def _build_shadow_overlay(
        self,
        original_geometric: list[GeometricRegion] | None,
        band_audit: dict[str, Any],
        fusion_results: dict[str, dict[str, Any]],
        raw_elements: list[dict[str, Any]] | None,
        ocr_blocks: list[dict[str, Any]] | None,
        vision_candidates: list[dict[str, Any]] | None,
        window_width: int,
        window_height: int,
    ) -> dict[str, Any]:
        """U1-shadow: combine original + TP band proposals and compare.

        Pipeline: original_geo + TP bands → combined_geo → build regions → group → classify.
        Returns comparison dict. Does NOT modify original regions/groups.
        """
        original_geo = list(original_geometric or [])
        tp_details = [d for d in band_audit.get("details", []) if d.get("verdict") == "TP"]
        amb_count = sum(1 for d in band_audit.get("details", []) if d.get("verdict") == "AMB")
        fp_count = sum(1 for d in band_audit.get("details", []) if d.get("verdict") == "FP")

        # Convert TP bands to GeometricRegions and track details
        injected_geo: list[GeometricRegion] = []
        band_detail_map: dict[str, dict[str, Any]] = {}  # band_id → detail dict
        for d in tp_details:
            bounds = d.get("bounds", [0, 0, 0, 0])
            band_id = d["band_id"]
            geo = GeometricRegion(
                region_id=f"band_{band_id}",
                bounds=tuple(bounds),
                boundary_evidence=[f"band_proposal_{d.get('detection_method', 'unknown')}"],
                geometry_confidence=0.5,
            )
            injected_geo.append(geo)
            band_detail_map[band_id] = {
                "band_id": band_id,
                "band_type": d.get("band_type", "?"),
                "bounds": list(bounds),
                "height_px": d.get("height_px", 0),
                "detection_method": d.get("detection_method", "?"),
                "audit_reason": d.get("reason", ""),
            }

        # Combined = original + injected
        combined_geo = original_geo + injected_geo

        # Compute original aggregates
        original_count = len(original_geo)
        original_regions = [self._build_region(gr, fusion_results.get(gr.region_id), raw_elements, ocr_blocks, vision_candidates, window_width, window_height) for gr in original_geo]
        original_unknown = sum(1 for r in original_regions if r.structure_type == StructureType.UNKNOWN_STRUCTURED)
        original_roi = sum(1 for r in original_regions if r.roi_eligible)

        # Build combined structured regions
        combined_regions: list[StructuredRegion] = []
        for gr in combined_geo:
            sr = self._build_region(
                gr,
                fusion_results.get(gr.region_id),
                raw_elements,
                ocr_blocks,
                vision_candidates,
                window_width,
                window_height,
            )
            combined_regions.append(sr)

        # Populate injected band details with classification results
        injected_band_details: list[dict[str, Any]] = []
        for sr in combined_regions:
            rid = sr.region_id
            if rid.startswith("band_"):
                band_id = rid[5:]  # strip "band_" prefix
                detail = band_detail_map.get(band_id, {})
                detail["resulting_structure_type"] = sr.structure_type.value
                detail["roi_eligible"] = sr.roi_eligible
                detail["classification_reason"] = next(
                    (e.detail for e in sr.evidence if e.source == "metrics"), ""
                )
                detail["confidence"] = sr.confidence
                injected_band_details.append(detail)

        # Combined grouping
        combined_groups_raw = self._grouper.group(
            combined_geo, window_width, window_height, ocr_blocks, vision_candidates,
        )
        combined_grouped: list[StructuredRegion] = []
        for group in combined_groups_raw:
            group_gr = GeometricRegion(
                region_id=group.group_id,
                bounds=group.bounds,
                boundary_evidence=[f"group: {group.reason}"],
                geometry_confidence=0.5,
            )
            sr = self._build_region(
                group_gr, None, raw_elements, ocr_blocks, vision_candidates,
                window_width, window_height,
            )
            sr.evidence.append(RegionEvidence(
                source="shadow_grouping",
                detail=f"children={group.child_region_ids}",
                confidence=0.0,
            ))
            combined_grouped.append(sr)

        # Combined aggregates
        combined_count = len(combined_regions)
        combined_unknown = sum(1 for r in combined_regions if r.structure_type == StructureType.UNKNOWN_STRUCTURED)
        combined_roi = sum(1 for r in combined_regions if r.roi_eligible)
        combined_group_count = len(combined_grouped)
        combined_group_unknown = sum(1 for r in combined_grouped if r.structure_type == StructureType.UNKNOWN_STRUCTURED)
        combined_group_roi = sum(1 for r in combined_grouped if r.roi_eligible)

        # Classify injected bands as useful vs noisy
        _USEFUL_TYPES = {
            "top_bar", "toolbar", "status_region", "control_strip",
            "media_control_bar", "input_region", "side_rail",
        }
        useful_details = []
        noisy_details = []
        for d in injected_band_details:
            st = d.get("resulting_structure_type", "unknown_structured")
            if st in _USEFUL_TYPES:
                useful_details.append(d)
            else:
                noisy_details.append(d)

        return {
            "original_region_count": original_count,
            "original_unknown_count": original_unknown,
            "original_roi_eligible_count": original_roi,
            "injected_band_count": len(injected_geo),
            "skipped_amb_band_count": amb_count,
            "skipped_fp_band_count": fp_count,
            "combined_region_count": combined_count,
            "combined_unknown_count": combined_unknown,
            "combined_roi_eligible_count": combined_roi,
            "combined_group_count": combined_group_count,
            "combined_group_unknown_count": combined_group_unknown,
            "combined_group_roi_eligible_count": combined_group_roi,
            "injected_band_details": injected_band_details,
            "useful_injected_band_count": len(useful_details),
            "noisy_injected_band_count": len(noisy_details),
            "useful_injected_band_details": useful_details,
            "noisy_injected_band_details": noisy_details,
        }

    def _run_u2_shadow(
        self,
        original_geometric: list[GeometricRegion] | None,
        screenshot: Any,
        raw_elements: list[dict[str, Any]] | None,
        fusion_results: dict[str, dict[str, Any]],
        ocr_blocks: list[dict[str, Any]] | None,
        vision_candidates: list[dict[str, Any]] | None,
        window_width: int,
        window_height: int,
    ) -> dict[str, Any]:
        """U2-shadow: run UIA boundary shadow pass and compare with original + U1."""
        from src.perception.uia_boundary_shadow import UiaBoundaryShadowPass

        shadow_pass = UiaBoundaryShadowPass()
        u2_result = shadow_pass.run(
            screenshot=screenshot,
            raw_elements=raw_elements,
            existing_regions=list(original_geometric or []),
            window_width=window_width,
            window_height=window_height,
        )

        # Build structured regions from uia_aware_regions
        u2_structured: list[StructuredRegion] = []
        for r_dict in u2_result.uia_aware_regions:
            gr = GeometricRegion(
                region_id=r_dict.get("region_id", "?"),
                bounds=tuple(r_dict.get("bounds", [0, 0, 0, 0])),
                boundary_evidence=r_dict.get("boundary_evidence", []),
                geometry_confidence=r_dict.get("geometry_confidence", 0.6),
            )
            sr = self._build_region(
                gr, None, raw_elements, ocr_blocks, vision_candidates,
                window_width, window_height,
            )
            u2_structured.append(sr)

        # Compute U2 aggregates
        u2_count = len(u2_structured)
        u2_unknown = sum(1 for r in u2_structured if r.structure_type == StructureType.UNKNOWN_STRUCTURED)
        u2_roi = sum(1 for r in u2_structured if r.roi_eligible)

        _USEFUL_TYPES = {
            "top_bar", "toolbar", "status_region", "control_strip",
            "media_control_bar", "input_region", "side_rail",
        }
        u2_useful = sum(
            1 for r in u2_structured
            if r.structure_type.value in _USEFUL_TYPES
        )
        u2_noisy = u2_count - u2_useful

        # Original aggregates
        original_geo = list(original_geometric or [])
        orig_regions = [
            self._build_region(gr, fusion_results.get(gr.region_id), raw_elements, ocr_blocks, vision_candidates, window_width, window_height)
            for gr in original_geo
        ]
        orig_count = len(orig_regions)
        orig_unknown = sum(1 for r in orig_regions if r.structure_type == StructureType.UNKNOWN_STRUCTURED)
        orig_roi = sum(1 for r in orig_regions if r.roi_eligible)

        return {
            "u2_diagnostics": u2_result.to_dict(),
            "original": {
                "region_count": orig_count,
                "unknown_count": orig_unknown,
                "roi_eligible": orig_roi,
            },
            "u2_shadow": {
                "region_count": u2_count,
                "unknown_count": u2_unknown,
                "roi_eligible": u2_roi,
                "useful": u2_useful,
                "noisy": u2_noisy,
            },
            "u2_regions": [r.to_dict() for r in u2_structured],
        }

    def _run_u3_shadow(
        self,
        original_geometric: list[GeometricRegion] | None,
        ocr_blocks: list[dict[str, Any]] | None,
        raw_elements: list[dict[str, Any]] | None,
        window_width: int,
        window_height: int,
    ) -> dict[str, Any]:
        """U3-shadow: OCR/visual density region proposal."""
        from src.perception.ocr_visual_shadow import OcrVisualShadowPass

        shadow_pass = OcrVisualShadowPass()
        u3_result = shadow_pass.run(
            ocr_blocks=ocr_blocks,
            window_width=window_width,
            window_height=window_height,
        )

        # Original aggregates
        original_geo = list(original_geometric or [])
        orig_regions = [
            self._build_region(gr, None, raw_elements, None, None, window_width, window_height)
            for gr in original_geo
        ]
        orig_count = len(orig_regions)
        orig_unknown = sum(1 for r in orig_regions if r.structure_type == StructureType.UNKNOWN_STRUCTURED)
        orig_roi = sum(1 for r in orig_regions if r.roi_eligible)

        return {
            "u3_diagnostics": u3_result.to_dict(),
            "original": {
                "region_count": orig_count,
                "unknown_count": orig_unknown,
                "roi_eligible": orig_roi,
            },
        }

    def _run_u4_shadow(
        self,
        screenshot: Any,
        ocr_blocks: list[dict[str, Any]] | None,
        vision_candidates: list[dict[str, Any]] | None,
        raw_elements: list[dict[str, Any]] | None,
        window_width: int,
        window_height: int,
    ) -> dict[str, Any]:
        """U4-shadow: Pixel projection + whitespace separator detection."""
        from src.perception.layout_boundary_shadow import LayoutBoundaryShadowPass

        shadow_pass = LayoutBoundaryShadowPass()
        u4_result = shadow_pass.run(
            screenshot=screenshot,
            ocr_blocks=ocr_blocks,
            vision_candidates=vision_candidates,
            raw_elements=raw_elements,
            window_width=window_width,
            window_height=window_height,
        )
        return u4_result.to_dict()

    def _run_ure_p0_shadow(
        self,
        geometric_regions: list[Any],
        grouped_regions: list[Any],
        band_proposals: list[dict[str, Any]],
        ocr_blocks: list[dict[str, Any]] | None,
        vision_candidates: list[dict[str, Any]] | None,
        raw_elements: list[dict[str, Any]] | None,
        window_width: int,
        window_height: int,
        u4_layout: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """URE P0 shadow: Region Evidence Model."""
        from src.perception.universal_region_engine import UniversalRegionEngine

        engine = UniversalRegionEngine()

        # Convert grouped_regions to dicts for URE
        groups_dicts = []
        for g in grouped_regions:
            if hasattr(g, "to_dict"):
                groups_dicts.append(g.to_dict())
            elif isinstance(g, dict):
                groups_dicts.append(g)

        return engine.build(
            geometric_regions=geometric_regions,
            structured_overlay={"groups": groups_dicts} if groups_dicts else None,
            band_proposals=band_proposals if band_proposals else None,
            ocr_blocks=ocr_blocks,
            vision_candidates=vision_candidates,
            raw_elements=raw_elements,
            window_width=window_width,
            window_height=window_height,
            u4_layout=u4_layout,
        )

    def _run_vtl_0_shadow(
        self,
        screenshot: Any,
        ocr_blocks: list[dict[str, Any]] | None,
        vision_candidates: list[dict[str, Any]] | None,
        raw_elements: list[dict[str, Any]] | None,
        window_width: int,
        window_height: int,
    ) -> dict[str, Any]:
        """VTL-0 shadow: Visual Trace Layer (page_container + local_object)."""
        from src.perception.visual_trace_engine import VisualTraceEngine

        engine = VisualTraceEngine()
        return engine.build(
            screenshot=screenshot,
            ocr_blocks=ocr_blocks,
            vision_candidates=vision_candidates,
            raw_elements=raw_elements,
            window_width=window_width,
            window_height=window_height,
        )

    def _build_region(
        self,
        gr: GeometricRegion,
        fusion_result: dict[str, Any] | None,
        raw_elements: list[dict[str, Any]] | None,
        ocr_blocks: list[dict[str, Any]] | None,
        vision_candidates: list[dict[str, Any]] | None,
        window_width: int,
        window_height: int,
    ) -> StructuredRegion:
        """Build a single StructuredRegion from geometric + fusion + elements.

        R3.1: Uses StructuralClassifier with combination evidence metrics.
        """
        evidence: list[RegionEvidence] = []
        semantic_label: str | None = None
        fusion_confidence: float = 0.0

        # 1. Geometric evidence
        for be in gr.boundary_evidence:
            evidence.append(RegionEvidence(
                source="geometric",
                detail=be,
                confidence=gr.geometry_confidence,
            ))

        # 2. Fusion evidence
        if fusion_result:
            semantic_label = fusion_result.get("semantic_label")
            fusion_confidence = float(fusion_result.get("confidence", 0.0))
            fusion_source = fusion_result.get("fusion_source", "none")
            reason = fusion_result.get("reason", "")
            evidence.append(RegionEvidence(
                source="fusion",
                detail=f"label={semantic_label}, source={fusion_source}, reason={reason}",
                confidence=fusion_confidence,
            ))

        # 3. Compute region metrics (R3.1)
        metrics = compute_region_metrics(
            bounds=gr.bounds,
            window_width=window_width,
            window_height=window_height,
            raw_elements=raw_elements,
            ocr_blocks=ocr_blocks,
            vision_candidates=vision_candidates,
            fusion_result=fusion_result,
        )

        # 4. Classify using combination evidence (R3.1)
        #    Fusion label takes priority if present
        structure_type = StructureType.UNKNOWN_STRUCTURED
        classifier_confidence = 0.0
        classifier_reason = ""

        if semantic_label:
            mapped = _FUSION_LABEL_TO_STRUCTURE.get(semantic_label)
            if mapped:
                structure_type = mapped
                classifier_confidence = fusion_confidence
                classifier_reason = f"fusion_label={semantic_label}"
            else:
                # Fusion label exists but not mapped → use classifier
                structure_type, classifier_confidence, classifier_reason = (
                    self._classifier.classify(metrics)
                )
        else:
            # No fusion label → use structural classifier
            structure_type, classifier_confidence, classifier_reason = (
                self._classifier.classify(metrics)
            )

        # 5. Add metrics evidence
        evidence.append(RegionEvidence(
            source="metrics",
            detail=(
                f"pos={metrics.position_band}, area={metrics.area_ratio:.3f}, "
                f"aspect={metrics.aspect_ratio:.1f}, ocr={metrics.ocr_count}, "
                f"short_text={metrics.short_text_count}, h_row={metrics.horizontal_text_row_score:.1f}, "
                f"v_list={metrics.vertical_list_score:.1f}, buttons={metrics.uia_button_count}, "
                f"edits={metrics.uia_edit_count}, lists={metrics.uia_list_count}, "
                f"vision={metrics.vision_candidate_count}, icons={metrics.icon_like_count}"
            ),
            confidence=classifier_confidence,
        ))

        # 6. Add keyword hints as weak evidence (NOT used for classification)
        if metrics.keyword_hints:
            evidence.append(RegionEvidence(
                source="keyword_hints",
                detail=", ".join(metrics.keyword_hints[:5]),
                confidence=0.0,
            ))

        # 7. Assign elements to this region (element_ids from raw_elements only)
        element_ids = self._assign_elements(gr.bounds, raw_elements)

        # 8. OCR evidence (count + text snippet)
        ocr_count, ocr_snippet = self._count_ocr_in_region(gr.bounds, ocr_blocks)
        if ocr_count > 0:
            evidence.append(RegionEvidence(
                source="ocr",
                detail=f"count={ocr_count}, text={ocr_snippet}",
                confidence=0.0,
            ))

        # 9. Vision evidence (count)
        vision_count = self._count_items_in_region(gr.bounds, vision_candidates)
        if vision_count > 0:
            evidence.append(RegionEvidence(
                source="vision",
                detail=f"count={vision_count}",
                confidence=0.0,
            ))

        # 10. Determine eligibility
        roi_eligible = structure_type in _ROI_ELIGIBLE_TYPES
        icon_memory_eligible = structure_type in _ICON_MEMORY_ELIGIBLE_TYPES

        # 11. Stability
        is_stable = structure_type not in {
            StructureType.FLOATING_PANEL,
            StructureType.DIALOG,
            StructureType.UNKNOWN_STRUCTURED,
        }

        # Overall confidence: max of fusion and classifier
        overall_confidence = max(fusion_confidence, classifier_confidence)

        return StructuredRegion(
            region_id=gr.region_id,
            structure_type=structure_type,
            semantic_label=semantic_label,
            bounds=gr.bounds,
            confidence=overall_confidence,
            evidence=evidence,
            is_stable=is_stable,
            roi_eligible=roi_eligible,
            icon_memory_eligible=icon_memory_eligible,
            element_ids=element_ids,
            parent_region_id=gr.parent_region_id,
        )

    def _assign_elements(
        self,
        region_bounds: tuple[int, int, int, int],
        raw_elements: list[dict[str, Any]] | None,
    ) -> list[str]:
        """Assign element IDs to this region by spatial containment (raw_elements only)."""
        element_ids: list[str] = []
        rl, rt, rr, rb = region_bounds

        if raw_elements:
            for elem in raw_elements:
                bounds = self._get_bounds(elem)
                if not bounds:
                    continue
                el, et, er, eb = bounds
                cx, cy = (el + er) // 2, (et + eb) // 2
                if rl <= cx <= rr and rt <= cy <= rb:
                    eid = elem.get("element_id", "")
                    if eid:
                        element_ids.append(eid)

        return element_ids

    def _count_ocr_in_region(
        self,
        region_bounds: tuple[int, int, int, int],
        ocr_blocks: list[dict[str, Any]] | None,
    ) -> tuple[int, str]:
        """Count OCR blocks in region, return (count, text_snippet)."""
        if not ocr_blocks:
            return 0, ""
        rl, rt, rr, rb = region_bounds
        count = 0
        texts: list[str] = []
        for block in ocr_blocks:
            bbox = block.get("bbox")
            if not bbox or len(bbox) < 4:
                continue
            bl, bt, br, bb = bbox
            cx, cy = (bl + br) // 2, (bt + bb) // 2
            if rl <= cx <= rr and rt <= cy <= rb:
                count += 1
                text = str(block.get("text", "")).strip()
                if text and len(texts) < 3:
                    texts.append(text[:20])
        snippet = ", ".join(texts)
        if count > len(texts):
            snippet += f" +{count - len(texts)}more"
        return count, snippet

    def _count_items_in_region(
        self,
        region_bounds: tuple[int, int, int, int],
        items: list[dict[str, Any]] | None,
    ) -> int:
        """Count items (vision_candidates or similar) in region by center containment."""
        if not items:
            return 0
        rl, rt, rr, rb = region_bounds
        count = 0
        for item in items:
            bounds = self._get_bounds(item)
            if not bounds:
                continue
            bl, bt, br, bb = bounds
            cx, cy = (bl + br) // 2, (bt + bb) // 2
            if rl <= cx <= rr and rt <= cy <= rb:
                count += 1
        return count

    def _get_bounds(
        self, element: dict[str, Any]
    ) -> tuple[int, int, int, int] | None:
        """Extract bounds from element dict."""
        bounds = element.get("bounding_rect") or element.get("bounds") or element.get("bbox")
        if bounds and len(bounds) >= 4:
            return (bounds[0], bounds[1], bounds[2], bounds[3])
        return None
