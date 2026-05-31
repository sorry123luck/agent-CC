"""Unit tests for SemanticFusion — Layer 2 evidence fusion engine.

All tests use mock data. No real screenshots or running apps required.
"""

import pytest

from src.perception.geometric_partitioner import GeometricRegion
from src.perception.semantic_fusion import (
    FusionConfig,
    FusionDiagnostics,
    FusionResult,
    RegionEvidence,
    SemanticFusion,
)


def _make_geo_region(
    region_id: str,
    left: int, top: int, right: int, bottom: int,
) -> GeometricRegion:
    return GeometricRegion(
        region_id=region_id,
        bounds=(left, top, right, bottom),
        boundary_evidence=["vertical_separator_L90%"],
        density_profile=(12.0, 0.04),
        geometry_confidence=0.8,
    )


def _make_uia_element(
    control_type: str,
    bounds: list[int],
    element_id: str = "elem_1",
) -> dict:
    return {
        "element_id": element_id,
        "control_type": control_type,
        "bounds": bounds,
        "text": "",
        "name": "",
    }


class TestSemanticFusion:
    """Tier 1: mock-data-driven unit tests for SemanticFusion."""

    # ── Evidence collection ──

    def test_collect_evidence_creates_one_per_region(self):
        regions = [_make_geo_region("R0", 0, 0, 960, 1080)]
        fusion = SemanticFusion()
        evidence = fusion.collect_evidence(regions, [], [], [], 1920, 1080)
        assert "R0" in evidence
        assert evidence["R0"].region_id == "R0"

    def test_collect_evidence_assigns_uia_by_center_overlap(self):
        regions = [
            _make_geo_region("R0", 0, 0, 960, 1080),
            _make_geo_region("R1", 960, 0, 1920, 1080),
        ]
        elements = [
            _make_uia_element("Edit", [100, 100, 300, 150], "e1"),
            _make_uia_element("Button", [1200, 900, 1400, 950], "e2"),
        ]
        fusion = SemanticFusion()
        evidence = fusion.collect_evidence(regions, elements, [], [], 1920, 1080)
        assert evidence["R0"].uia_element_count == 1
        assert "edit" in [t.lower() for t in evidence["R0"].uia_control_types]
        assert evidence["R1"].uia_element_count == 1
        assert "button" in [t.lower() for t in evidence["R1"].uia_control_types]

    # ── Fusion rules ──

    def test_uia_edit_control_gives_input_area(self):
        regions = [_make_geo_region("R0", 0, 900, 1920, 1080)]
        elements = [_make_uia_element("Edit", [100, 920, 500, 970])]
        fusion = SemanticFusion()
        evidence = fusion.collect_evidence(regions, elements, [], [], 1920, 1080)
        results = fusion.fuse(evidence)
        assert results[0].semantic_label == "input_area"
        assert results[0].confidence >= 0.50

    def test_bottom_buttons_give_action_bar(self):
        regions = [_make_geo_region("R0", 0, 900, 1920, 1080)]
        elements = [
            _make_uia_element("Button", [200, 920, 300, 970], "b1"),
            _make_uia_element("Button", [400, 920, 500, 970], "b2"),
            _make_uia_element("Button", [600, 920, 700, 970], "b3"),
        ]
        fusion = SemanticFusion()
        evidence = fusion.collect_evidence(regions, elements, [], [], 1920, 1080)
        results = fusion.fuse(evidence)
        assert results[0].semantic_label == "action_bar"
        assert results[0].confidence == 0.65

    def test_left_list_items_give_navigation(self):
        regions = [_make_geo_region("R0", 0, 100, 500, 1000)]
        elements = [
            _make_uia_element("ListItem", [20, 150, 480, 200], "li1"),
            _make_uia_element("ListItem", [20, 220, 480, 270], "li2"),
            _make_uia_element("ListItem", [20, 290, 480, 340], "li3"),
        ]
        fusion = SemanticFusion()
        evidence = fusion.collect_evidence(regions, elements, [], [], 1920, 1080)
        results = fusion.fuse(evidence)
        # Left band with list items → navigation
        labels = [r.semantic_label for r in results]
        assert "navigation" in labels

    def test_insufficient_evidence_stays_unknown(self):
        regions = [_make_geo_region("R0", 500, 400, 800, 600)]
        elements = [_make_uia_element("Group", [510, 410, 700, 500])]
        fusion = SemanticFusion()
        evidence = fusion.collect_evidence(regions, elements, [], [], 1920, 1080)
        results = fusion.fuse(evidence)
        assert results[0].semantic_label is None
        assert results[0].confidence == 0.0
        assert results[0].fusion_source == "none"
        assert results[0].reason == "insufficient_evidence"

    def test_large_center_pane_gives_content(self):
        regions = [_make_geo_region("R0", 200, 200, 1720, 880)]
        elements = [
            _make_uia_element("Pane", [210, 210, 1710, 870]),
            _make_uia_element("Text", [300, 300, 500, 350]),
        ]
        fusion = SemanticFusion()
        evidence = fusion.collect_evidence(regions, elements, [], [], 1920, 1080)
        results = fusion.fuse(evidence)
        assert results[0].semantic_label == "content"
        assert results[0].confidence == 0.50

    def test_vlm_annotation_overrides_low_confidence(self):
        regions = [_make_geo_region("R0", 200, 200, 800, 600)]
        elements = [_make_uia_element("Group", [210, 210, 700, 500])]
        fusion = SemanticFusion()
        evidence = fusion.collect_evidence(regions, elements, [], [], 1920, 1080)
        results = fusion.fuse(evidence)
        # Without VLM: unknown
        assert results[0].semantic_label is None

        # Apply VLM annotation
        vlm_resp = {"regions": {"R0": {"semantic_label": "card_grid", "confidence": 0.75}}}
        updated = fusion.apply_vlm_annotations(results, vlm_resp)
        assert updated[0].semantic_label == "card_grid"
        assert updated[0].confidence == 0.75
        assert updated[0].fusion_source == "vlm"

    def test_vlm_annotation_does_not_override_high_confidence(self):
        regions = [_make_geo_region("R0", 0, 900, 1920, 1080)]
        elements = [_make_uia_element("Edit", [100, 920, 500, 970])]
        fusion = SemanticFusion()
        evidence = fusion.collect_evidence(regions, elements, [], [], 1920, 1080)
        results = fusion.fuse(evidence)
        # UIA Edit → input_area with high confidence
        assert results[0].semantic_label == "input_area"
        assert results[0].confidence >= 0.50

        # VLM with lower confidence → should not override
        vlm_resp = {"regions": {"R0": {"semantic_label": "content", "confidence": 0.40}}}
        updated = fusion.apply_vlm_annotations(results, vlm_resp)
        assert updated[0].semantic_label == "input_area"

    def test_build_diagnostics(self):
        results = [
            FusionResult(region_id="R0", semantic_label="input_area", confidence=0.85,
                         fusion_source="uia_geometry", reason="UIA EditControl"),
            FusionResult(region_id="R1", semantic_label=None, confidence=0.0,
                         fusion_source="none", reason="insufficient_evidence"),
            FusionResult(region_id="R2", semantic_label="navigation", confidence=0.55,
                         fusion_source="uia_geometry", reason="list items in side band"),
        ]
        fusion = SemanticFusion()
        diag = fusion.build_diagnostics(results)
        assert diag.geometric_region_count == 3
        assert diag.labeled_count == 2
        assert diag.unknown_count == 1
        assert len(diag.regions) == 3

    def test_empty_regions_returns_empty(self):
        fusion = SemanticFusion()
        evidence = fusion.collect_evidence([], [], [], [], 1920, 1080)
        assert evidence == {}
        results = fusion.fuse(evidence)
        assert results == []

    def test_position_label_correct(self):
        regions = [
            _make_geo_region("R_top", 0, 0, 1920, 150),
            _make_geo_region("R_bottom", 0, 900, 1920, 1080),
            _make_geo_region("R_left", 0, 200, 400, 800),
            _make_geo_region("R_right", 1500, 200, 1920, 800),
            _make_geo_region("R_center", 400, 200, 1500, 800),
        ]
        fusion = SemanticFusion()
        evidence = fusion.collect_evidence(regions, [], [], [], 1920, 1080)
        assert evidence["R_top"].position_label == "top"
        assert evidence["R_bottom"].position_label == "bottom"
        assert evidence["R_left"].position_label == "left"
        assert evidence["R_right"].position_label == "right"
        assert evidence["R_center"].position_label == "center"
