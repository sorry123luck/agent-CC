"""Tests for StructuredRegion data model and overlay builder (Phase R2)."""

import json

from src.perception.geometric_partitioner import GeometricRegion
from src.perception.structured_region import (
    RegionEvidence,
    StructuredRegion,
    StructuredRegionOverlay,
    StructureType,
)
from src.perception.structured_region_builder import StructuredRegionOverlayBuilder


# =============================================================================
# Data model tests
# =============================================================================


class TestStructureType:
    def test_values_are_strings(self):
        for st in StructureType:
            assert isinstance(st.value, str)

    def test_unknown_structured_exists(self):
        assert StructureType.UNKNOWN_STRUCTURED.value == "unknown_structured"


class TestRegionEvidence:
    def test_to_dict(self):
        e = RegionEvidence(source="geometric", detail="vertical_separator", confidence=0.8)
        d = e.to_dict()
        assert d["source"] == "geometric"
        assert d["detail"] == "vertical_separator"
        assert d["confidence"] == 0.8

    def test_to_dict_rounds_confidence(self):
        e = RegionEvidence(source="test", confidence=0.123456)
        assert e.to_dict()["confidence"] == 0.123


class TestStructuredRegion:
    def test_defaults(self):
        sr = StructuredRegion(region_id="R0")
        assert sr.structure_type == StructureType.UNKNOWN_STRUCTURED
        assert sr.semantic_label is None
        assert sr.confidence == 0.0
        assert sr.is_stable is False  # UNKNOWN_STRUCTURED is not stable
        assert sr.roi_eligible is False
        assert sr.icon_memory_eligible is False
        assert sr.element_ids == []

    def test_to_dict(self):
        sr = StructuredRegion(
            region_id="R1",
            structure_type=StructureType.TOOLBAR,
            semantic_label="action_bar",
            bounds=(0, 0, 800, 60),
            confidence=0.75,
            evidence=[RegionEvidence(source="fusion", detail="test", confidence=0.75)],
            is_stable=True,
            roi_eligible=True,
            icon_memory_eligible=True,
            element_ids=["e1", "e2"],
        )
        d = sr.to_dict()
        assert d["region_id"] == "R1"
        assert d["structure_type"] == "toolbar"
        assert d["semantic_label"] == "action_bar"
        assert d["bounds"] == [0, 0, 800, 60]
        assert d["confidence"] == 0.75
        assert len(d["evidence"]) == 1
        assert d["is_stable"] is True
        assert d["roi_eligible"] is True
        assert d["icon_memory_eligible"] is True
        assert d["element_ids"] == ["e1", "e2"]

    def test_to_dict_json_serializable(self):
        sr = StructuredRegion(
            region_id="R0",
            structure_type=StructureType.CANVAS_REGION,
            bounds=(100, 100, 500, 400),
        )
        # Should not raise
        json.dumps(sr.to_dict())


class TestStructuredRegionOverlay:
    def test_empty_overlay(self):
        overlay = StructuredRegionOverlay()
        d = overlay.to_dict()
        assert d["total_regions"] == 0
        assert d["unknown_count"] == 0
        assert d["with_semantic_label"] == 0
        assert d["roi_eligible_count"] == 0
        assert d["regions"] == []

    def test_to_dict_json_serializable(self):
        overlay = StructuredRegionOverlay(
            regions=[StructuredRegion(region_id="R0")],
            total_regions=1,
            unknown_count=1,
        )
        json.dumps(overlay.to_dict())


# =============================================================================
# Builder tests
# =============================================================================


class TestStructuredRegionOverlayBuilder:
    def setup_method(self):
        self.builder = StructuredRegionOverlayBuilder()

    def test_empty_geometric_returns_empty_overlay(self):
        overlay = self.builder.build(geometric_regions=None)
        assert overlay.total_regions == 0
        assert overlay.regions == []

    def test_empty_list_returns_empty_overlay(self):
        overlay = self.builder.build(geometric_regions=[])
        assert overlay.total_regions == 0

    def test_single_geometric_region(self):
        gr = GeometricRegion(
            region_id="R0",
            bounds=(0, 0, 800, 600),
            boundary_evidence=["horizontal_separator_Y50%"],
            geometry_confidence=0.7,
        )
        overlay = self.builder.build(geometric_regions=[gr])
        assert overlay.total_regions == 1
        assert overlay.regions[0].region_id == "R0"
        assert overlay.regions[0].bounds == (0, 0, 800, 600)
        assert len(overlay.regions[0].evidence) >= 1

    def test_fusion_label_maps_to_structure_type(self):
        gr = GeometricRegion(
            region_id="R0",
            bounds=(0, 0, 800, 60),
            geometry_confidence=0.6,
        )
        fusion = {
            "regions": [
                {
                    "region_id": "R0",
                    "semantic_label": "action_bar",
                    "confidence": 0.8,
                    "fusion_source": "uia_geometry",
                    "reason": "3 buttons in bottom band",
                }
            ]
        }
        overlay = self.builder.build(
            geometric_regions=[gr],
            fusion_diagnostics=fusion,
        )
        sr = overlay.regions[0]
        assert sr.structure_type == StructureType.TOOLBAR
        assert sr.semantic_label == "action_bar"
        assert sr.confidence == 0.8
        assert sr.roi_eligible is True

    def test_unknown_fusion_label_stays_unknown(self):
        gr = GeometricRegion(
            region_id="R0",
            bounds=(0, 0, 800, 600),
            geometry_confidence=0.5,
        )
        fusion = {
            "regions": [
                {
                    "region_id": "R0",
                    "semantic_label": "some_unmapped_label",
                    "confidence": 0.3,
                }
            ]
        }
        overlay = self.builder.build(
            geometric_regions=[gr],
            fusion_diagnostics=fusion,
        )
        sr = overlay.regions[0]
        assert sr.structure_type == StructureType.UNKNOWN_STRUCTURED
        assert sr.semantic_label == "some_unmapped_label"

    def test_positional_heuristic_top_bar(self):
        """Top bar requires combination evidence: position + wide + horizontal text row."""
        gr = GeometricRegion(
            region_id="R0",
            bounds=(0, 0, 1920, 50),
            geometry_confidence=0.3,
        )
        ocr_blocks = [
            {"text": "文件", "bbox": (10, 10, 60, 35)},
            {"text": "编辑", "bbox": (70, 12, 120, 37)},
            {"text": "查看", "bbox": (130, 10, 180, 35)},
        ]
        overlay = self.builder.build(
            geometric_regions=[gr],
            ocr_blocks=ocr_blocks,
            window_width=1920,
            window_height=1080,
        )
        sr = overlay.regions[0]
        assert sr.structure_type == StructureType.TOP_BAR

    def test_positional_heuristic_status_bar(self):
        """Status bar requires combination evidence: position + wide + short texts."""
        gr = GeometricRegion(
            region_id="R0",
            bounds=(0, 1020, 1920, 1080),
            geometry_confidence=0.3,
        )
        ocr_blocks = [
            {"text": "100%", "bbox": (1700, 1030, 1750, 1050)},
            {"text": "UTF-8", "bbox": (1500, 1030, 1560, 1050)},
        ]
        overlay = self.builder.build(
            geometric_regions=[gr],
            ocr_blocks=ocr_blocks,
            window_width=1920,
            window_height=1080,
        )
        sr = overlay.regions[0]
        assert sr.structure_type == StructureType.STATUS_REGION

    def test_element_assignment(self):
        gr = GeometricRegion(
            region_id="R0",
            bounds=(0, 0, 400, 300),
            geometry_confidence=0.5,
        )
        raw_elements = [
            {"element_id": "e1", "bounding_rect": (10, 10, 50, 50)},
            {"element_id": "e2", "bounding_rect": (500, 500, 600, 600)},  # outside
            {"element_id": "e3", "bounding_rect": (200, 200, 250, 250)},  # inside
        ]
        overlay = self.builder.build(
            geometric_regions=[gr],
            raw_elements=raw_elements,
        )
        sr = overlay.regions[0]
        assert "e1" in sr.element_ids
        assert "e3" in sr.element_ids
        assert "e2" not in sr.element_ids

    def test_multiple_regions(self):
        regions = [
            GeometricRegion(region_id="R0", bounds=(0, 0, 1920, 50), geometry_confidence=0.6),
            GeometricRegion(region_id="R1", bounds=(0, 50, 1920, 1080), geometry_confidence=0.5),
        ]
        overlay = self.builder.build(
            geometric_regions=regions,
            window_width=1920,
            window_height=1080,
        )
        assert overlay.total_regions == 2
        assert overlay.unknown_count >= 0  # at least one might be unknown

    def test_no_fusion_diagnostics_still_works(self):
        gr = GeometricRegion(
            region_id="R0",
            bounds=(0, 0, 800, 600),
            geometry_confidence=0.5,
        )
        overlay = self.builder.build(
            geometric_regions=[gr],
            fusion_diagnostics=None,
        )
        assert overlay.total_regions == 1
        assert overlay.regions[0].semantic_label is None

    def test_roi_eligible_types(self):
        """Toolbar and side_rail should be roi_eligible."""
        gr_toolbar = GeometricRegion(region_id="R0", bounds=(0, 0, 800, 60), geometry_confidence=0.5)
        gr_canvas = GeometricRegion(region_id="R1", bounds=(0, 60, 800, 600), geometry_confidence=0.5)
        fusion = {
            "regions": [
                {"region_id": "R0", "semantic_label": "action_bar", "confidence": 0.7},
                {"region_id": "R1", "semantic_label": "content", "confidence": 0.6},
            ]
        }
        overlay = self.builder.build(
            geometric_regions=[gr_toolbar, gr_canvas],
            fusion_diagnostics=fusion,
        )
        by_id = {r.region_id: r for r in overlay.regions}
        assert by_id["R0"].roi_eligible is True  # toolbar
        assert by_id["R1"].roi_eligible is False  # canvas_region

    def test_icon_memory_eligible_types(self):
        """Only top_bar/toolbar/side_rail/status_region are icon_memory_eligible."""
        gr = GeometricRegion(region_id="R0", bounds=(0, 0, 800, 60), geometry_confidence=0.5)
        fusion = {"regions": [{"region_id": "R0", "semantic_label": "action_bar", "confidence": 0.7}]}
        overlay = self.builder.build(geometric_regions=[gr], fusion_diagnostics=fusion)
        assert overlay.regions[0].icon_memory_eligible is True

    def test_stability(self):
        """Floating panels and dialogs should not be stable."""
        gr = GeometricRegion(region_id="R0", bounds=(100, 100, 400, 300), geometry_confidence=0.5)
        fusion = {"regions": [{"region_id": "R0", "semantic_label": "dialog", "confidence": 0.6}]}
        overlay = self.builder.build(geometric_regions=[gr], fusion_diagnostics=fusion)
        assert overlay.regions[0].is_stable is False

    def test_ocr_evidence_added(self):
        """OCR blocks whose center falls in region should appear as evidence."""
        gr = GeometricRegion(region_id="R0", bounds=(0, 0, 400, 300), geometry_confidence=0.5)
        ocr_blocks = [
            {"text": "文件", "bbox": (10, 10, 50, 30)},
            {"text": "编辑", "bbox": (60, 10, 100, 30)},
            {"text": "远处", "bbox": (500, 500, 600, 520)},  # outside
        ]
        overlay = self.builder.build(geometric_regions=[gr], ocr_blocks=ocr_blocks)
        sr = overlay.regions[0]
        ocr_ev = [e for e in sr.evidence if e.source == "ocr"]
        assert len(ocr_ev) == 1
        assert "count=2" in ocr_ev[0].detail
        assert "文件" in ocr_ev[0].detail

    def test_ocr_evidence_text_snippet_limit(self):
        """OCR evidence should include at most 3 text snippets."""
        gr = GeometricRegion(region_id="R0", bounds=(0, 0, 400, 300), geometry_confidence=0.5)
        ocr_blocks = [
            {"text": "A", "bbox": (10, 10, 30, 30)},
            {"text": "B", "bbox": (40, 10, 60, 30)},
            {"text": "C", "bbox": (70, 10, 90, 30)},
            {"text": "D", "bbox": (100, 10, 120, 30)},
        ]
        overlay = self.builder.build(geometric_regions=[gr], ocr_blocks=ocr_blocks)
        ocr_ev = [e for e in overlay.regions[0].evidence if e.source == "ocr"]
        assert "+1more" in ocr_ev[0].detail

    def test_vision_evidence_added(self):
        """Vision candidates whose center falls in region should appear as evidence."""
        gr = GeometricRegion(region_id="R0", bounds=(0, 0, 400, 300), geometry_confidence=0.5)
        vision_candidates = [
            {"bounding_rect": (10, 10, 50, 50)},
            {"bounding_rect": (60, 60, 100, 100)},
            {"bounding_rect": (500, 500, 600, 600)},  # outside
        ]
        overlay = self.builder.build(geometric_regions=[gr], vision_candidates=vision_candidates)
        sr = overlay.regions[0]
        vis_ev = [e for e in sr.evidence if e.source == "vision"]
        assert len(vis_ev) == 1
        assert "count=2" in vis_ev[0].detail

    def test_no_ocr_vision_evidence_when_empty(self):
        """No OCR/vision evidence when those inputs are empty."""
        gr = GeometricRegion(region_id="R0", bounds=(0, 0, 400, 300), geometry_confidence=0.5)
        overlay = self.builder.build(geometric_regions=[gr])
        sr = overlay.regions[0]
        assert not any(e.source == "ocr" for e in sr.evidence)
        assert not any(e.source == "vision" for e in sr.evidence)

    def test_vision_uses_bbox_field(self):
        """Vision candidates using 'bbox' field should also be counted."""
        gr = GeometricRegion(region_id="R0", bounds=(0, 0, 400, 300), geometry_confidence=0.5)
        vision_candidates = [
            {"bbox": (10, 10, 50, 50)},
        ]
        overlay = self.builder.build(geometric_regions=[gr], vision_candidates=vision_candidates)
        vis_ev = [e for e in overlay.regions[0].evidence if e.source == "vision"]
        assert len(vis_ev) == 1
        assert "count=1" in vis_ev[0].detail
